"""Grid search the D_b discriminant's f_c / f_tau background weights.

For a single fm4tag classifier checkpoint, this sweeps the fraction-weighted
log-likelihood-ratio discriminant

    D_b = log( p_b / (f_c * p_c + f_tau * p_tau + f_u * p_u) )      f_u = 1 - f_c - f_tau

over a 2D (f_c, f_tau) grid and, at each point, measures a target
background's rejection at a fixed b-jet efficiency working point. Unlike
roc_comparison.py (which compares several taggers at one fixed set of
fractions), this holds the tagger fixed and searches over fractions to find
the point that maximises rejection against one background.

Model loading, inference and the discriminant itself are reused from
roc_comparison.py (checkpoint resolution, ``FinetuneModule`` rebuild, cached
predictions, ``discriminant()``) rather than reimplemented — this script only
adds the grid sweep and its outputs:

  - ``grid_search.csv``       — every (f_c, f_tau) point with rejection
                                 against every background at the working point
  - ``fraction_grid_heatmap.png`` — target-background rejection over the grid,
                                 with the maximum marked
  - ``best_point.yaml``       — the winning (f_c, f_tau) and its rejections

Usage::

    python scripts/fraction_grid_search.py --config scripts/configs/fraction_grid_search_best_ce.yaml
    python scripts/fraction_grid_search.py --config ... --output plots/my_grid
    python scripts/fraction_grid_search.py --config ... --recompute       # ignore cached predictions
    python scripts/fraction_grid_search.py --config ... --max-jets 50000  # quick partial run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from ftag.utils import calculate_rejection

sys.path.insert(0, str(Path(__file__).parent))
import roc_comparison as rc  # noqa: E402  -- checkpoint/model/inference/discriminant reuse


def build_grid(axis_cfg: dict) -> np.ndarray:
    return np.linspace(axis_cfg['min'], axis_cfg['max'], int(axis_cfg['num']))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Grid search f_c / f_tau to maximise a background rejection.'
    )
    parser.add_argument('--config', type=Path, required=True, help='YAML config file')
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output directory (overrides plot.out_dir from the config)',
    )
    parser.add_argument(
        '--recompute', action='store_true', help='Ignore cached predictions'
    )
    parser.add_argument(
        '--max-jets', type=int, default=None, help='Truncate the test set (smoke test)'
    )
    parser.add_argument(
        '--device', default=None, help='Inference device (default: cuda if available)'
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    plot_cfg = cfg.get('plot', {})
    out_dir = args.output or Path(plot_cfg.get('out_dir', 'plots/fraction_grid_search'))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {out_dir}')

    device = torch.device(
        args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    if device.type == 'cpu':
        print(
            'WARNING: running inference on CPU — use a GPU node for the full test set'
        )
    torch.set_float32_matmul_precision('high')

    # ── Resolve the run and the test file ──────────────────────────────────
    model_cfg = cfg['model']
    run_dir = Path(model_cfg['dir'])
    label = model_cfg.get('label', run_dir.name)
    run_cfg = rc.find_run_config(run_dir)
    ckpt_path = rc.resolve_checkpoint(
        run_dir, run_cfg, model_cfg.get('checkpoint', 'best')
    )

    test_file = (
        Path(cfg['test_file'])
        if cfg.get('test_file')
        else Path(str(run_cfg.test_dataset_path))
    )
    print(f'Model: {label} ({run_dir.name})\nTest file: {test_file}')

    with h5py.File(test_file, 'r') as f:
        n_total = f['jets'].shape[0]
    n_jets = n_total if args.max_jets is None else min(args.max_jets, n_total)
    if n_jets < n_total:
        print(f'Truncating test set to the first {n_jets:,} of {n_total:,} jets')

    inference_cfg = cfg.get('inference', {})
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

    # ── Jets / cuts ─────────────────────────────────────────────────────────
    jets = rc.read_jets(test_file, [], n_jets)
    mask = rc.cut_mask(jets, cfg.get('cuts', []))
    truth = jets['HadronConeExclTruthLabelID'][mask]
    probs_masked = {f: probs[:, i][mask] for i, f in enumerate(rc.FLAVOURS)}

    signal = cfg.get('signal', 'b')
    target = cfg.get('target_background', 'c')
    backgrounds = [f for f in rc.FLAVOURS if f != signal]
    wp = float(cfg.get('working_point', 0.8))

    flavour_masks = {f: truth == rc.TRUTH_ID[f] for f in rc.FLAVOURS}
    sig_mask = flavour_masks[signal]
    counts = '  '.join(f'{f}: {flavour_masks[f].sum():,}' for f in rc.FLAVOURS)
    print(f'Jets after cuts — {counts}')

    # ── Grid sweep ──────────────────────────────────────────────────────────
    grid_cfg = cfg['grid']
    fc_vals = build_grid(grid_cfg['fc'])
    ftau_vals = build_grid(grid_cfg['ftau'])

    rej_grid = np.full((len(ftau_vals), len(fc_vals)), np.nan)
    rows = []
    for i, ftau in enumerate(ftau_vals):
        for j, fc in enumerate(fc_vals):
            if fc + ftau >= 1.0:
                continue  # f_u = 1 - fc - ftau would go negative
            disc = rc.discriminant(probs_masked, signal, {'c': fc, 'tau': ftau})
            sig_disc = disc[sig_mask]
            row = {'f_c': fc, 'f_tau': ftau}
            for bkg in backgrounds:
                rej = calculate_rejection(
                    sig_disc, disc[flavour_masks[bkg]], target_eff=wp
                )
                row[f'{bkg}_rejection'] = float(rej)
            rows.append(row)
            rej_grid[i, j] = row[f'{target}_rejection']

    df = pd.DataFrame(rows)
    csv_path = out_dir / 'grid_search.csv'
    df.to_csv(csv_path, index=False)
    print(f'Saved {csv_path}')

    best = df.loc[df[f'{target}_rejection'].idxmax()]
    other_rejs = '  '.join(
        f'{b}-rej={best[f"{b}_rejection"]:.2f}' for b in backgrounds if b != target
    )
    print(
        f'\nBest point — f_c={best.f_c:.4f}  f_tau={best.f_tau:.4f}  '
        f'{target}-rejection={best[f"{target}_rejection"]:.2f} @ {signal}-eff={wp:g} '
        f'({other_rejs})'
    )

    best_path = out_dir / 'best_point.yaml'
    with best_path.open('w') as f:
        yaml.safe_dump(
            {
                'model': label,
                'checkpoint': str(ckpt_path),
                'signal': signal,
                'target_background': target,
                'working_point': wp,
                'f_c': float(best.f_c),
                'f_tau': float(best.f_tau),
                **{
                    f'{b}_rejection': float(best[f'{b}_rejection']) for b in backgrounds
                },
            },
            f,
            sort_keys=False,
        )
    print(f'Saved {best_path}')

    # ── Heatmap ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=tuple(plot_cfg.get('figsize', [7.0, 6.0])))
    mesh = ax.pcolormesh(
        fc_vals,
        ftau_vals,
        rej_grid,
        shading='auto',
        cmap=plot_cfg.get('cmap', 'viridis'),
    )
    fig.colorbar(mesh, ax=ax, label=f'{target}-jet rejection @ {signal}-eff = {wp:g}')
    ax.scatter(
        [best.f_c],
        [best.f_tau],
        marker='*',
        s=250,
        color='red',
        edgecolor='white',
        linewidth=0.8,
        zorder=5,
        label=f'max: f_c={best.f_c:.3f}, f_tau={best.f_tau:.3f}',
    )
    ax.set_xlabel('$f_c$')
    ax.set_ylabel('$f_\\tau$')
    ax.set_title(
        plot_cfg.get('title')
        or f'{label} — {target}-jet rejection @ {signal}-eff = {wp:g}'
    )
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    heatmap_path = out_dir / 'fraction_grid_heatmap.png'
    fig.savefig(heatmap_path, dpi=plot_cfg.get('dpi', 200), bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {heatmap_path}')


if __name__ == '__main__':
    main()
