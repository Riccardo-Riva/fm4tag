"""Constrained f_c / f_tau search, per model, over a set of fm4tag checkpoints.

For every model listed in the config, grid-searches the full (f_c, f_tau)
simplex (f_c, f_tau >= 0, f_c + f_tau < 1) at the configured step, evaluating
all three background rejections (c, u, tau) at every point — the discriminant
is already computed there, so the extra rejection calls are cheap.

The chosen point per model is the one maximising c-jet rejection subject to::

    u_rejection   >= floor_frac * u_rejection_GN2
    tau_rejection >= floor_frac * tau_rejection_GN2

where u/tau_rejection_GN2 are the reference tagger's own rejections at the
same working point (computed once, from its own fixed fractions). This
directly targets "beat GN2 on c-rejection without losing more than
(1 - floor_frac) of GN2's light/tau rejection" — unlike
fraction_two_stage_search.py's two-stage rule, which picks f_c to purely
maximise c-rejection first (ignoring u/tau entirely) and only rescans f_tau
afterwards. That greedy rule structurally starves u/tau of denominator
weight once f_c is pinned near its unconstrained maximum (observed to land
at f_c = 0.9-0.99 for every model), which is exactly why light/tau-rejection
collapsed in that search.

If no grid point satisfies both floors for a model, the model is flagged
infeasible and the point maximising min(u_rejection / floor_u,
tau_rejection / floor_tau) is reported instead (closest approach to
feasibility), together with a warning.

Model loading, cached inference and the discriminant itself are reused from
roc_comparison.py, exactly as fraction_two_stage_search.py does. The grid
itself is evaluated in parallel across CPU processes (each grid point is
independent), since this is pure numpy/CPU work once predictions are
cached — no GPU needed. Run it on a CPU partition with many cores; see
scripts/submit_fraction_constrained_search.sh.

Outputs, under --output:
  - ``<label_slug>_grid.csv``         — every grid point for that model
                                         (f_c, f_tau, c/u/tau rejection)
  - ``optimal_fractions.csv``         — one row per model: chosen (f_c, f_tau),
                                         rejections and whether it was feasible
  - ``roc_comparison_config.yaml``    — a ready-to-run scripts/roc_comparison.py
                                         config with each model's fractions set
                                         to its own constrained optimum

Usage::

    python scripts/fraction_constrained_search.py \\
        --config scripts/configs/fraction_two_stage_search_lambda_scan.yaml \\
        --floor-frac 0.95 \\
        --workers 64
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from ftag.utils import calculate_rejection
from ftag.utils.metrics import weighted_percentile

sys.path.insert(0, str(Path(__file__).parent))
import roc_comparison as rc  # noqa: E402  -- checkpoint/model/inference/discriminant reuse

_G: dict = {}


def _pool_init(
    probs: dict[str, np.ndarray],
    sig_mask: np.ndarray,
    bkg_masks: dict[str, np.ndarray],
    signal: str,
    wp: float,
) -> None:
    """Cache the per-model arrays every grid point needs.

    The discriminant denominator is ``f_c p_c + f_tau p_tau + (1-f_c-f_tau) p_u
    = p_u + f_c (p_c - p_u) + f_tau (p_tau - p_u)``, so ``p_c - p_u`` and
    ``p_tau - p_u`` are precomputed here and each point is two scalar-weighted
    adds.  The ``log`` in ``rc.discriminant`` is dropped: rejections and working
    points are invariant under it (it is monotone), and skipping it saves an
    N-element ``np.log`` per point.
    """
    global _G
    _G = {
        'pb': probs[signal],
        'pu': probs['u'],
        'dc': probs['c'] - probs['u'],
        'dtau': probs['tau'] - probs['u'],
        'sig': sig_mask,
        'bkg': bkg_masks,
        'pct': np.array([1.0 - wp]),
    }


def _eval_point(point: tuple[float, float]) -> tuple[float, float, float, float, float]:
    fc, ftau = point
    disc = _G['pb'] / (_G['pu'] + fc * _G['dc'] + ftau * _G['dtau'])
    # Signal cut is the (1-wp) percentile of the signal discriminant — identical
    # for all three backgrounds, so compute it once (was 3x inside
    # calculate_rejection) and count each background above it directly.
    cut = float(weighted_percentile(disc[_G['sig']], _G['pct'])[0])
    rej = []
    for b in ('c', 'u', 'tau'):
        bd = disc[_G['bkg'][b]]
        n_pass = int(np.count_nonzero(bd >= cut))
        rej.append(bd.size / n_pass if n_pass else np.inf)
    return (fc, ftau, rej[0], rej[1], rej[2])


def sweep_joint(
    probs: dict[str, np.ndarray],
    sig_mask: np.ndarray,
    bkg_masks: dict[str, np.ndarray],
    signal: str,
    points: list[tuple[float, float]],
    wp: float,
    workers: int,
) -> pd.DataFrame:
    """Evaluate c/u/tau rejection at every (f_c, f_tau) in ``points`` (at ``wp``)."""
    with mp.Pool(
        workers,
        initializer=_pool_init,
        initargs=(probs, sig_mask, bkg_masks, signal, wp),
    ) as pool:
        chunksize = max(1, len(points) // (workers * 8))
        results = pool.map(_eval_point, points, chunksize=chunksize)
    return pd.DataFrame(
        results, columns=['f_c', 'f_tau', 'c_rejection', 'u_rejection', 'tau_rejection']
    )


def gn2_floor(cfg: dict, wp: float) -> dict[str, float]:
    """u/tau rejection of the reference tagger at ``wp``, from its own file+fractions."""
    ref = cfg['roc_comparison']['reference_tagger']
    signal = cfg.get('signal', 'b')
    jets = rc.read_jets(Path(ref['file']), list(ref['probs'].values()))
    mask = rc.cut_mask(jets, cfg.get('cuts', []))
    truth = jets['HadronConeExclTruthLabelID'][mask]
    flavour_masks = {f: truth == rc.TRUTH_ID[f] for f in rc.FLAVOURS}
    probs = {f: jets[col][mask] for f, col in ref['probs'].items()}
    disc = rc.discriminant(probs, signal, ref['fractions'])
    sig_disc = disc[flavour_masks[signal]]
    return {
        bkg: float(
            calculate_rejection(sig_disc, disc[flavour_masks[bkg]], target_eff=wp)
        )
        for bkg in ('u', 'tau')
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Constrained f_c / f_tau search: maximise c-rejection subject to a floor on u/tau-rejection.'
    )
    parser.add_argument('--config', type=Path, required=True, help='YAML config file')
    parser.add_argument('--output', type=Path, default=None, help='Output directory')
    parser.add_argument(
        '--recompute', action='store_true', help='Ignore cached predictions'
    )
    parser.add_argument(
        '--max-jets', type=int, default=None, help='Truncate the test set'
    )
    parser.add_argument(
        '--device', default=None, help='Inference device for cache misses'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='CPU worker processes (default: all allocated)',
    )
    parser.add_argument(
        '--floor-frac',
        type=float,
        default=0.95,
        help='Required fraction of GN2 u- AND tau-rejection (default: 0.95). '
        'Overridden per background by --floor-u / --floor-tau.',
    )
    parser.add_argument(
        '--floor-u',
        type=float,
        default=None,
        help='Required fraction of GN2 u-rejection (default: --floor-frac).',
    )
    parser.add_argument(
        '--floor-tau',
        type=float,
        default=None,
        help='Required fraction of GN2 tau-rejection (default: --floor-frac).',
    )
    parser.add_argument(
        '--working-point',
        type=float,
        default=None,
        help='Signal-efficiency working point (default: config working_point, else 0.8).',
    )
    parser.add_argument(
        '--grid-dir',
        type=Path,
        default=None,
        help='Shared directory for per-model <slug>_grid_wp<wp>.csv. The joint '
        '(f_c, f_tau) rejection grid depends only on the checkpoint + wp, '
        'not on the floors, so runs that vary only --floor-u / --floor-tau '
        'reuse it here instead of recomputing the sweep. Default: --output.',
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = args.output or Path(
        cfg.get('plot', {}).get('out_dir', 'plots/fraction_constrained_search')
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {out_dir}')

    workers = args.workers or len(os.sched_getaffinity(0))
    print(f'CPU workers: {workers}')

    device = torch.device(
        args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    torch.set_float32_matmul_precision('high')

    signal = cfg.get('signal', 'b')
    wp = float(
        args.working_point
        if args.working_point is not None
        else cfg.get('working_point', 0.8)
    )
    step = float(cfg['step'])
    frac_u = args.floor_u if args.floor_u is not None else args.floor_frac
    frac_tau = args.floor_tau if args.floor_tau is not None else args.floor_frac

    # The full (f_c, f_tau) simplex is 0..1 x 0..1, but every constrained optimum
    # observed lands at f_c < 0.4, f_tau < 0.06.  Restrict the grid to a box that
    # comfortably contains that (config-overridable) — it is the dominant cost.
    fc_max = float(cfg.get('fc_max', 0.6))
    ftau_max = float(cfg.get('ftau_max', 0.15))
    fc_values = np.round(np.arange(0.0, fc_max, step), 10)
    ftau_values = np.round(np.arange(0.0, ftau_max, step), 10)

    floor = gn2_floor(cfg, wp)
    floor_u, floor_tau = frac_u * floor['u'], frac_tau * floor['tau']
    print(
        f'GN2 @ wp={wp}: u_rejection={floor["u"]:.3f}  tau_rejection={floor["tau"]:.3f}  '
        f'-> floors: u>={floor_u:.3f} ({frac_u}x)  tau>={floor_tau:.3f} ({frac_tau}x)'
    )

    inference_cfg = cfg.get('inference', {})
    model_entries = cfg['models']

    # ── Resolve runs, checkpoints and the shared test file ─────────────────
    run_cfgs, ckpts = [], []
    for entry in model_entries:
        run_dir = Path(entry['dir'])
        run_cfg = rc.find_run_config(run_dir)
        run_cfgs.append(run_cfg)
        ckpts.append(
            rc.resolve_checkpoint(run_dir, run_cfg, entry.get('checkpoint', 'best'))
        )

    if cfg.get('test_file'):
        test_file = Path(cfg['test_file'])
    else:
        test_files = {str(c.test_dataset_path) for c in run_cfgs}
        if len(test_files) > 1:
            raise ValueError(
                f'Runs used different test files {test_files}; set test_file explicitly'
            )
        test_file = Path(test_files.pop())
    print(f'Test file: {test_file}')

    with h5py.File(test_file, 'r') as f:
        n_total = f['jets'].shape[0]
    n_jets = n_total if args.max_jets is None else min(args.max_jets, n_total)

    jets = rc.read_jets(test_file, [], n_jets)
    mask = rc.cut_mask(jets, cfg.get('cuts', []))
    truth = jets['HadronConeExclTruthLabelID'][mask]
    flavour_masks = {f: truth == rc.TRUTH_ID[f] for f in rc.FLAVOURS}
    sig_mask = flavour_masks[signal]
    bkg_masks = {f: flavour_masks[f] for f in rc.FLAVOURS if f != signal}

    # ── Per-model constrained search ────────────────────────────────────────
    summary_rows = []
    model_results = []  # (entry, fc, ftau)
    points = [(fc, ftau) for ftau in ftau_values for fc in fc_values if fc + ftau < 1.0]
    print(
        f'grid: f_c in [0, {fc_max}) x f_tau in [0, {ftau_max}) at step {step} '
        f'-> {len(points)} points'
    )

    grid_dir = args.grid_dir or out_dir
    grid_dir.mkdir(parents=True, exist_ok=True)

    for entry, run_cfg, ckpt_path in zip(model_entries, run_cfgs, ckpts):
        label = entry['label']
        slug = rc.re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_').lower()
        print(f'\n=== {label} ({Path(entry["dir"]).name}) ===')

        # The joint rejection grid is a function of (checkpoint, wp) only.  Reuse
        # a shared copy when one exists so a floor sweep doesn't recompute it.
        shared_grid = grid_dir / f'{slug}_grid_wp{wp:g}.csv'
        if shared_grid.is_file() and not args.recompute:
            df = pd.read_csv(shared_grid)
            print(f'  using cached grid: {shared_grid}  ({len(df)} points)')
        else:
            probs = rc.cached_probs(
                out_dir / 'cache',
                label,
                ckpt_path,
                test_file,
                n_jets,
                args.recompute,
                compute=lambda: rc.run_inference(
                    rc.build_module(run_cfg, ckpt_path),
                    run_cfg,
                    test_file,
                    device,
                    inference_cfg.get('batch_size', 1024),
                    inference_cfg.get('num_workers', 4),
                    n_jets,
                    desc=label,
                ),
            )
            probs_masked = {f: probs[:, i][mask] for i, f in enumerate(rc.FLAVOURS)}
            df = sweep_joint(
                probs_masked, sig_mask, bkg_masks, signal, points, wp, workers
            )
            tmp = shared_grid.with_name(f'.{shared_grid.name}.{os.getpid()}')
            df.to_csv(tmp, index=False)
            os.replace(tmp, shared_grid)  # atomic — concurrent floor jobs may race

        df.to_csv(out_dir / f'{slug}_grid.csv', index=False)

        feasible = df[(df.u_rejection >= floor_u) & (df.tau_rejection >= floor_tau)]
        if len(feasible):
            best = feasible.loc[feasible.c_rejection.idxmax()]
            is_feasible = True
        else:
            score = np.minimum(df.u_rejection / floor_u, df.tau_rejection / floor_tau)
            best = df.loc[score.idxmax()]
            is_feasible = False
            print(
                f'  WARNING: no point meets both floors for {label} — reporting closest approach'
            )

        fc, ftau = float(best.f_c), float(best.f_tau)
        print(
            f'  chosen: f_c={fc:.4f}  f_tau={ftau:.4f}  feasible={is_feasible}  '
            f'c_rej={best.c_rejection:.3f}  u_rej={best.u_rejection:.3f}  tau_rej={best.tau_rejection:.3f}'
        )
        if fc >= fc_max - 2 * step or ftau >= ftau_max - 2 * step:
            print(
                f'  WARNING: optimum for {label} is at the grid edge '
                f'(f_c_max={fc_max}, f_tau_max={ftau_max}) — widen fc_max/ftau_max'
            )

        summary_rows.append(
            {
                'label': label,
                'f_c': fc,
                'f_tau': ftau,
                'c_rejection': float(best.c_rejection),
                'u_rejection': float(best.u_rejection),
                'tau_rejection': float(best.tau_rejection),
                'feasible': is_feasible,
            }
        )
        model_results.append((entry, fc, ftau))

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / 'optimal_fractions.csv'
    summary.to_csv(summary_path, index=False)
    print(f'\nSaved {summary_path}\n{summary.to_string(index=False)}')

    # ── Emit a ready-to-run roc_comparison.py config ────────────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from fraction_two_stage_search import build_roc_config  # noqa: E402

    cfg['working_point'] = wp  # so the emitted config's WPs reflect --working-point
    roc_cfg = build_roc_config(cfg, model_results)
    roc_cfg_path = out_dir / 'roc_comparison_config.yaml'
    with roc_cfg_path.open('w') as f:
        yaml.safe_dump(roc_cfg, f, sort_keys=False)
    print(f'Saved {roc_cfg_path}')


if __name__ == '__main__':
    main()
