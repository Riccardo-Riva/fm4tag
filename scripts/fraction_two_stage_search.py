"""Two-stage f_c / f_tau search, per model, over a set of fm4tag checkpoints.

For every model listed in the config, independently:

  Stage 1 — grid-search the full (f_c, f_tau) simplex (f_c, f_tau >= 0,
            f_c + f_tau < 1) at the configured step, to find the point
            (f_c0, f_tau0) that maximises ``stage1_target``'s rejection
            (default: c-jet) at the fixed signal-efficiency working point.
  Stage 2 — fix f_c = f_c0 and re-scan f_tau alone (same step) to find the
            f_tau1 that instead maximises ``stage2_target``'s rejection
            (default: u/light-jet), holding f_c fixed at its stage-1 value.

Each model's final fractions are (f_c0, f_tau1). This is deliberately a
two-objective search, not a single joint optimum: stage 1 picks f_c purely to
beat one background, stage 2 then re-tunes f_tau for a different background
without touching f_c again.

Model loading, cached inference and the discriminant itself are reused from
roc_comparison.py, exactly as fraction_grid_search.py does. The grid itself
is evaluated in parallel across CPU processes (each grid point is
independent), since this is pure numpy/CPU work once predictions are
cached — no GPU needed. Run it on a CPU partition with many cores; see
scripts/submit_fraction_two_stage_search.sh.

Outputs, under --output:
  - ``<label_slug>_stage1_grid.csv``  — every stage-1 grid point for that model
  - ``<label_slug>_stage2_scan.csv``  — every stage-2 f_tau point for that model
  - ``optimal_fractions.csv``         — one row per model: chosen (f_c0, f_tau1)
                                         and the rejections behind them
  - ``roc_comparison_config.yaml``    — a ready-to-run scripts/roc_comparison.py
                                         config with each model's fractions set
                                         to its own optimum

Usage::

    python scripts/fraction_two_stage_search.py \\
        --config scripts/configs/fraction_two_stage_search_lambda_scan.yaml \\
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

sys.path.insert(0, str(Path(__file__).parent))
import roc_comparison as rc  # noqa: E402  -- checkpoint/model/inference/discriminant reuse

_G: dict = {}


def _pool_init(
    probs: dict[str, np.ndarray],
    sig_mask: np.ndarray,
    tgt_mask: np.ndarray,
    signal: str,
) -> None:
    global _G
    _G = {'probs': probs, 'sig': sig_mask, 'tgt': tgt_mask, 'signal': signal}


def _eval_point(point: tuple[float, float, float]) -> tuple[float, float, float]:
    fc, ftau, wp = point
    disc = rc.discriminant(_G['probs'], _G['signal'], {'c': fc, 'tau': ftau})
    rej = calculate_rejection(disc[_G['sig']], disc[_G['tgt']], target_eff=wp)
    return fc, ftau, float(rej)


def sweep(
    probs_full: dict[str, np.ndarray],
    sig_mask: np.ndarray,
    target_mask: np.ndarray,
    signal: str,
    points: list[tuple[float, float, float]],
    workers: int,
    target_col: str,
) -> pd.DataFrame:
    """Evaluate ``target_col`` rejection at every (f_c, f_tau, wp) in ``points``.

    Restricts every array to signal + target-background jets before handing
    them to worker processes — the discriminant is never needed for the other
    backgrounds here, and this roughly halves the per-point cost.
    """
    relevant = sig_mask | target_mask
    probs_relevant = {f: p[relevant] for f, p in probs_full.items()}
    sig_local = sig_mask[relevant]
    tgt_local = target_mask[relevant]

    with mp.Pool(
        workers,
        initializer=_pool_init,
        initargs=(probs_relevant, sig_local, tgt_local, signal),
    ) as pool:
        chunksize = max(1, len(points) // (workers * 8))
        results = pool.map(_eval_point, points, chunksize=chunksize)

    return pd.DataFrame(results, columns=['f_c', 'f_tau', target_col])


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Two-stage f_c / f_tau search maximising two backgrounds in turn.'
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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = args.output or Path(
        cfg.get('plot', {}).get('out_dir', 'plots/fraction_two_stage_search')
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
    stage1_target = cfg.get('stage1_target', 'c')
    stage2_target = cfg.get('stage2_target', 'u')
    wp = float(cfg.get('working_point', 0.8))
    step = float(cfg['step'])

    values = np.round(np.arange(0.0, 1.0, step), 10)

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

    # ── Per-model two-stage search ──────────────────────────────────────────
    summary_rows = []
    model_results = []  # (entry, fc0, ftau1)
    for entry, run_cfg, ckpt_path in zip(model_entries, run_cfgs, ckpts):
        label = entry['label']
        slug = rc.re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_').lower()
        print(f'\n=== {label} ({Path(entry["dir"]).name}) ===')

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

        # Stage 1: full (f_c, f_tau) grid, maximise stage1_target rejection.
        points1 = [
            (fc, ftau, wp) for ftau in values for fc in values if fc + ftau < 1.0
        ]
        df1 = sweep(
            probs_masked,
            sig_mask,
            flavour_masks[stage1_target],
            signal,
            points1,
            workers,
            f'{stage1_target}_rejection',
        )
        df1.to_csv(out_dir / f'{slug}_stage1_grid.csv', index=False)
        best1 = df1.loc[df1[f'{stage1_target}_rejection'].idxmax()]
        fc0 = float(best1.f_c)
        print(
            f'  stage 1: f_c0={fc0:.4f}  f_tau0={best1.f_tau:.4f}  '
            f'{stage1_target}-rejection={best1[f"{stage1_target}_rejection"]:.3f}'
        )

        # Stage 2: fix f_c = f_c0, scan f_tau alone, maximise stage2_target rejection.
        points2 = [(fc0, ftau, wp) for ftau in values if fc0 + ftau < 1.0]
        df2 = sweep(
            probs_masked,
            sig_mask,
            flavour_masks[stage2_target],
            signal,
            points2,
            workers,
            f'{stage2_target}_rejection',
        )
        df2.to_csv(out_dir / f'{slug}_stage2_scan.csv', index=False)
        best2 = df2.loc[df2[f'{stage2_target}_rejection'].idxmax()]
        ftau1 = float(best2.f_tau)
        print(
            f'  stage 2: f_c={fc0:.4f} (fixed)  f_tau1={ftau1:.4f}  '
            f'{stage2_target}-rejection={best2[f"{stage2_target}_rejection"]:.3f}'
        )

        # Report every background's rejection at the final chosen point.
        final = {'label': label, 'f_c': fc0, 'f_tau': ftau1}
        disc_final = rc.discriminant(probs_masked, signal, {'c': fc0, 'tau': ftau1})
        for bkg in (f for f in rc.FLAVOURS if f != signal):
            final[f'{bkg}_rejection'] = float(
                calculate_rejection(
                    disc_final[sig_mask], disc_final[flavour_masks[bkg]], target_eff=wp
                )
            )
        summary_rows.append(final)
        model_results.append((entry, fc0, ftau1))

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / 'optimal_fractions.csv'
    summary.to_csv(summary_path, index=False)
    print(f'\nSaved {summary_path}\n{summary.to_string(index=False)}')

    # ── Emit a ready-to-run roc_comparison.py config ────────────────────────
    roc_cfg = build_roc_config(cfg, model_results)
    roc_cfg_path = out_dir / 'roc_comparison_config.yaml'
    with roc_cfg_path.open('w') as f:
        yaml.safe_dump(roc_cfg, f, sort_keys=False)
    print(f'Saved {roc_cfg_path}')


def build_roc_config(cfg: dict, model_results: list) -> dict:
    roc_cfg = cfg.get('roc_comparison', {})
    palette = roc_cfg.get('colours') or [None] * len(model_results)

    models = []
    for (entry, fc, ftau), colour in zip(model_results, palette):
        base_label = entry.get('base_label', entry['label'])
        models.append(
            {
                'dir': entry['dir'],
                'label': f'{base_label} (f_c = {fc:.3f}, f_tau = {ftau:.3f})',
                'fractions': {'c': round(fc, 6), 'tau': round(ftau, 6)},
                **({'colour': colour} if colour else {}),
            }
        )

    gn2 = roc_cfg['reference_tagger']
    gn2_label = f'{gn2["label"]} (f_c = {gn2["fractions"]["c"]:g}, f_tau = {gn2["fractions"]["tau"]:g})'

    return {
        'models': models,
        'reference_taggers': [
            {
                'label': gn2_label,
                'file': gn2['file'],
                'probs': gn2['probs'],
                'fractions': gn2['fractions'],
                'colour': gn2.get('colour', '#EE6677'),
            }
        ],
        'reference': gn2_label,
        'test_file': cfg.get('test_file'),
        'cuts': cfg.get('cuts', []),
        'signal': cfg.get('signal', 'b'),
        'backgrounds': ['u', 'c', 'tau'],
        'fractions': {'c': 0.2, 'tau': 0.01},
        'sig_eff': {'min': 0.6, 'max': 1.0, 'num': 100},
        'working_points': [0.70, 0.77, cfg.get('working_point', 0.8), 0.85],
        'eff_vs_pt': {
            'working_point': cfg.get('working_point', 0.8),
            'bins_gev': [20, 30, 40, 60, 85, 110, 140, 175, 250],
            'flat_per_bin': False,
        },
        'inference': cfg.get('inference', {'batch_size': 1024, 'num_workers': 8}),
        'plot': roc_cfg.get('plot', {}),
    }


if __name__ == '__main__':
    main()
