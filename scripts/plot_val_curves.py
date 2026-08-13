"""Overlay a per-epoch validation metric across several training runs.

Each entry of the YAML config points at a training-run directory.  For every run
the script

1. loads the run's own ``.hydra/config.yaml`` to recover the loss weights
   (``finetune.loss_weights.*``) and the augmentation strength ``lambda``,
2. reads the per-epoch metric from the Lightning ``metrics.csv``,
3. draws the curve, coloured on a blue -> red scale by ``lambda`` (runs whose
   jet-contrastive weight is 0 are drawn in the baseline colour instead), and
   marks the best (lowest) epoch with a star of the same colour.

Usage::

    python scripts/plot_val_curves.py --config scripts/configs/val_loss_ce_lambda_scan.yaml
    python scripts/plot_val_curves.py --config ... --output plots/my_scan

``--output`` overrides ``plot.out_dir`` from the config.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Run resolution: hydra config + metrics.csv
# ---------------------------------------------------------------------------


def find_run_config(run_dir: Path) -> DictConfig:
    """Load the newest ``.hydra/config.yaml`` under ``run_dir`` (one per rank)."""
    candidates = sorted(
        run_dir.glob('outputs/*/*/.hydra/config.yaml'),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f'No .hydra/config.yaml found under {run_dir}')
    return OmegaConf.load(candidates[-1])


def find_metrics(run_dir: Path) -> Path:
    """Locate the newest Lightning ``metrics.csv`` under ``run_dir``."""
    candidates = sorted(
        run_dir.glob('*/version_*/metrics.csv'), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(f'No */version_*/metrics.csv found under {run_dir}')
    return candidates[-1]


def read_metric(metrics_csv: Path, metric: str) -> pd.DataFrame:
    """Return the per-epoch ``metric`` series as columns ``epoch`` / ``value``.

    Rows without the metric (train-step rows) are dropped; epochs logged more
    than once are averaged.
    """
    df = pd.read_csv(metrics_csv)
    if metric not in df.columns:
        raise KeyError(
            f'{metrics_csv} has no column {metric!r} (has {list(df.columns)})'
        )

    series = df[df[metric].notna()][['epoch', metric]]
    series = series.groupby('epoch', as_index=False)[metric].mean()
    series = series.rename(columns={metric: 'value'}).sort_values('epoch')
    series['epoch'] = series['epoch'].astype(int)
    return series.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model entries
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """One training run: its curve plus the hyper-parameters that label it."""

    label: str | None
    lam: float
    jc: float
    ce: float
    curve: pd.DataFrame
    color: str | None = None

    @property
    def best_epoch(self) -> int:
        return int(self.curve.loc[self.curve['value'].idxmin(), 'epoch'])

    @property
    def best_value(self) -> float:
        return float(self.curve['value'].min())


def load_run(entry: DictConfig, metric: str) -> Run:
    """Build a :class:`Run` from a config entry, reading what it does not override."""
    run_dir = Path(entry['dir'])
    if not run_dir.is_dir():
        raise FileNotFoundError(f'Run directory {run_dir} not found')

    run_cfg = find_run_config(run_dir)
    weights = OmegaConf.select(run_cfg, 'finetune.loss_weights') or {}

    return Run(
        label=entry.get('label'),
        lam=float(entry.get('lam', OmegaConf.select(run_cfg, 'lambda'))),
        jc=float(entry.get('jc', weights.get('jet_contrastive', 0.0))),
        ce=float(entry.get('ce', weights.get('cross_entropy', 0.0))),
        curve=read_metric(find_metrics(run_dir), metric),
        color=entry.get('color'),
    )


def auto_label(run: Run, uniform_jc: bool) -> str:
    """Legend label: only the fields that actually vary across the comparison."""
    if run.label is not None:
        return run.label
    if run.jc == 0.0:
        return f'no JC  ($\\lambda$ = {run.lam:g})'
    if uniform_jc:
        return f'$\\lambda$ = {run.lam:g}'
    return f'JC {run.jc:g},  $\\lambda$ = {run.lam:g}'


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_curves(runs: list[Run], cfg: DictConfig, out_dir: Path) -> Path:
    """Draw all curves on one axes and save the figure."""
    style = cfg.get('color', {})
    cmap = LinearSegmentedColormap.from_list(
        'blue_red', list(style.get('cmap', ['#1f3ecc', '#7a1fa2', '#cc1f1f']))
    )
    norm = Normalize(
        vmin=float(style.get('vmin', 0.0)), vmax=float(style.get('vmax', 1.0))
    )
    baseline_color = style.get('baseline_color', 'black')

    # Baselines first, then the scan in ascending lambda, so the legend reads as
    # a gradient.
    runs = sorted(runs, key=lambda r: (r.jc != 0.0, r.lam))
    uniform_jc = len({r.jc for r in runs if r.jc != 0.0}) <= 1

    plot_cfg = cfg.get('plot', {})
    fig, ax = plt.subplots(figsize=tuple(plot_cfg.get('figsize', [8.0, 5.0])))

    for run in runs:
        if run.color is not None:
            color = run.color
        elif run.jc == 0.0:
            color = baseline_color
        else:
            color = cmap(norm(run.lam))

        ax.plot(
            run.curve['epoch'],
            run.curve['value'],
            color=color,
            linewidth=1.4,
            label=auto_label(run, uniform_jc),
        )
        ax.plot(
            run.best_epoch,
            run.best_value,
            marker='*',
            markersize=12,
            color=color,
            markeredgecolor='white',
            markeredgewidth=0.5,
            linestyle='none',
            zorder=5,
        )

    ax.set_xlabel(plot_cfg.get('xlabel', 'Epoch'))
    ax.set_ylabel(plot_cfg.get('ylabel', cfg['metric']))
    if plot_cfg.get('title'):
        ax.set_title(plot_cfg['title'])
    if plot_cfg.get('ylim'):
        ax.set_ylim(*plot_cfg['ylim'])
    if plot_cfg.get('xlim'):
        ax.set_xlim(*plot_cfg['xlim'])
    ax.grid(True, alpha=0.3)

    legend_kwargs = {'fontsize': 8, 'title': plot_cfg.get('legend_title')}
    if plot_cfg.get('legend_outside', True):
        legend_kwargs |= {'loc': 'upper left', 'bbox_to_anchor': (1.01, 1.0)}
    else:
        legend_kwargs |= {'loc': 'best', 'ncol': 2}
    ax.legend(**legend_kwargs)

    if style.get('colorbar', False):
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
        bar.set_label('$\\lambda$')

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / plot_cfg.get('filename', 'val_curves.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Overlay a per-epoch validation metric across training runs.'
    )
    parser.add_argument('--config', type=Path, required=True, help='YAML config path')
    parser.add_argument('--output', type=Path, help='Override plot.out_dir')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    metric = cfg.get('metric', 'val/head/loss_ce')

    runs = []
    for entry in cfg['models']:
        run = load_run(entry, metric)
        runs.append(run)
        print(
            f'{Path(entry["dir"]).name}\n'
            f'  ce={run.ce:g}  jc={run.jc:g}  lambda={run.lam:g}  '
            f'epochs={len(run.curve)}  best={run.best_value:.5f} @ epoch {run.best_epoch}'
        )

    out_dir = args.output or Path(
        cfg.get('plot', {}).get('out_dir', 'plots/val_curves')
    )
    print(f'\nSaved {plot_curves(runs, cfg, out_dir)}')


if __name__ == '__main__':
    main()
