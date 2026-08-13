"""Overlay a per-epoch validation metric across several training configurations.

Each entry of the YAML config is one CONFIGURATION, backed by one or more
training runs — the repetitions of the same configuration under different seeds
that slurm/classify_param_scan.sh produces.  For every entry the script

1. loads the runs' own ``.hydra/config.yaml`` to recover the loss weights
   (``finetune.loss_weights.*``) and the augmentation strength ``lambda``,
2. reads the per-epoch metric from each run's Lightning ``metrics.csv``,
3. draws the seed-mean curve with a ±spread band, and marks the best (lowest)
   value reached with a star carrying the spread across seeds as an error bar,
4. writes a ``summary.csv`` with the per-configuration numbers behind the stars.

A single-run entry degenerates to exactly the old behaviour: plain curve, plain
star, no band, no error bar.

Entries name their runs in one of three ways::

    - dir:  /path/to/run                 # one run
    - cell: /path/to/cell                # every <cell>/seed_*/ under it
    - dirs: [/path/a, /path/b]           # an explicit list

``label:``, ``lam:``, ``jc:``, ``ce:`` and ``color:`` remain per-entry overrides
of what is otherwise read from the run's own config.

Usage::

    python scripts/plot_val_curves.py --config scripts/configs/val_loss_ce_lambda_scan.yaml
    python scripts/plot_val_curves.py --config ... --output plots/my_scan

``--output`` overrides ``plot.out_dir`` from the config.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
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


def read_run_info(run_dir: Path) -> dict:
    """Read ``run_info.yaml`` (written per seed by classify_param_scan.sh).

    Returns ``{}`` for runs that predate it — those are never filtered out.
    Parsed with plain yaml, not OmegaConf: the file has dotted keys such as
    ``finetune.jet_contrastive_loss.temperature`` that OmegaConf would try to
    read as a nested path.
    """
    info_path = run_dir / 'run_info.yaml'
    if not info_path.is_file():
        return {}
    with info_path.open() as fh:
        return yaml.safe_load(fh) or {}


def resolve_run_dirs(entry: DictConfig, seed_status: list[str] | None) -> list[Path]:
    """Expand one config entry into the list of run directories it covers.

    ``cell:`` entries are expanded over ``<cell>/<seed_glob>`` and then filtered
    on the ``status`` recorded in each run's ``run_info.yaml``, so a truncated or
    failed seed does not silently drag the mean around.
    """
    if 'dir' in entry:
        run_dirs = [Path(entry['dir'])]
    elif 'dirs' in entry:
        run_dirs = [Path(d) for d in entry['dirs']]
    elif 'cell' in entry:
        cell = Path(entry['cell'])
        if not cell.is_dir():
            raise FileNotFoundError(f'Cell directory {cell} not found')
        run_dirs = sorted(
            (d for d in cell.glob(entry.get('seed_glob', 'seed_*')) if d.is_dir()),
            key=lambda d: seed_of(d),
        )
        if not run_dirs:
            raise FileNotFoundError(f'No seed directories under {cell}')
    else:
        raise KeyError(f"Entry {entry} has none of 'dir', 'dirs', 'cell'")

    kept = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f'Run directory {run_dir} not found')
        status = read_run_info(run_dir).get('status')
        if seed_status and status is not None and status not in seed_status:
            print(f'  skip {run_dir.name}: status {status!r}')
            continue
        kept.append(run_dir)

    if not kept:
        raise FileNotFoundError(
            f'Entry {entry} matched runs, but none with status in {seed_status}'
        )
    return kept


def seed_of(run_dir: Path) -> int:
    """Seed number from a ``seed_<n>`` directory name (-1 if not seed-shaped).

    Legacy run directories end in the sweep's run index, which is NOT a seed —
    hence the prefix check rather than "trailing digits".
    """
    name = run_dir.name
    if not name.startswith('seed_') or not name[5:].isdigit():
        return -1
    return int(name[5:])


# ---------------------------------------------------------------------------
# Configuration entries: one or more seeds of the same setup
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    """One configuration: the curves of its seeds plus the labels for it."""

    label: str | None
    lam: float
    jc: float
    ce: float
    curves: list[pd.DataFrame]
    seeds: list[int] = field(default_factory=list)
    color: str | None = None
    error: str = 'std'  # 'std' | 'sem' | 'minmax'

    @property
    def n_seeds(self) -> int:
        return len(self.curves)

    @property
    def best_values(self) -> np.ndarray:
        """Best (lowest) value each seed reached, over its own full curve."""
        return np.array([float(c['value'].min()) for c in self.curves])

    @property
    def best_epochs(self) -> np.ndarray:
        return np.array([int(c.loc[c['value'].idxmin(), 'epoch']) for c in self.curves])

    @property
    def best_value(self) -> float:
        """Mean over seeds of the best value reached — the star's height."""
        return float(self.best_values.mean())

    @property
    def best_epoch(self) -> float:
        return float(self.best_epochs.mean())

    @property
    def best_error(self) -> float:
        """Spread of the best value across seeds; 0 for a single run."""
        values = self.best_values
        if values.size < 2:
            return 0.0
        if self.error == 'minmax':
            return float((values.max() - values.min()) / 2)
        std = float(values.std(ddof=1))
        return std / np.sqrt(values.size) if self.error == 'sem' else std

    @property
    def curve(self) -> pd.DataFrame:
        """Seed-mean curve with its spread, over the epochs EVERY seed reached.

        Seeds early-stop at different epochs, and averaging over a shrinking set
        would put a kink in the mean exactly where a seed drops out.  The common
        prefix keeps every point of the mean curve built from the same runs;
        ``lo`` / ``hi`` bound the band and collapse onto the mean at n = 1.
        """
        wide = pd.concat(
            [
                c.set_index('epoch')['value'].rename(i)
                for i, c in enumerate(self.curves)
            ],
            axis=1,
        ).dropna()

        mean = wide.mean(axis=1)
        if self.n_seeds < 2:
            lo = hi = mean
        elif self.error == 'minmax':
            lo, hi = wide.min(axis=1), wide.max(axis=1)
        else:
            spread = wide.std(axis=1, ddof=1)
            if self.error == 'sem':
                spread = spread / np.sqrt(self.n_seeds)
            lo, hi = mean - spread, mean + spread

        return pd.DataFrame(
            {
                'epoch': mean.index.astype(int),
                'value': mean.to_numpy(),
                'lo': lo.to_numpy(),
                'hi': hi.to_numpy(),
            }
        ).reset_index(drop=True)


def load_entry(entry: DictConfig, metric: str, cfg: DictConfig) -> Entry:
    """Build an :class:`Entry` from a config entry, reading what it does not override."""
    seed_status = cfg.get('seed_status', ['completed'])
    if seed_status is not None:
        seed_status = list(seed_status)

    run_dirs = resolve_run_dirs(entry, seed_status)

    # Hyper-parameters come from the first run; the others must agree, or the
    # entry is not one configuration and its mean would be meaningless.
    run_cfgs = [find_run_config(d) for d in run_dirs]
    keys = {
        'lam': 'lambda',
        'jc': 'finetune.loss_weights.jet_contrastive',
        'ce': 'finetune.loss_weights.cross_entropy',
    }
    values = {}
    for name, path in keys.items():
        seen = [float(OmegaConf.select(rc, path) or 0.0) for rc in run_cfgs]
        if len(set(seen)) > 1:
            print(
                f'  WARNING: seeds disagree on {path}: {sorted(set(seen))} — '
                f'using {seen[0]:g} from {run_dirs[0].name}'
            )
        values[name] = float(entry.get(name, seen[0]))

    return Entry(
        label=entry.get('label'),
        lam=values['lam'],
        jc=values['jc'],
        ce=values['ce'],
        curves=[read_metric(find_metrics(d), metric) for d in run_dirs],
        seeds=[seed_of(d) for d in run_dirs],
        color=entry.get('color'),
        error=cfg.get('error', 'std'),
    )


def auto_label(entry: Entry, uniform_jc: bool, show_n: bool) -> str:
    """Legend label: only the fields that actually vary across the comparison."""
    if entry.label is not None:
        label = entry.label
    elif entry.jc == 0.0:
        label = f'no JC  ($\\lambda$ = {entry.lam:g})'
    elif uniform_jc:
        label = f'$\\lambda$ = {entry.lam:g}'
    else:
        label = f'JC {entry.jc:g},  $\\lambda$ = {entry.lam:g}'

    if show_n and entry.n_seeds > 1:
        label += f'  (n = {entry.n_seeds})'
    return label


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def assign_colors(
    entries: list[Entry], cmap, scale: str, norm: Normalize, baseline_color: str
) -> list:
    """One colour per entry, in the order given.

    ``scale='uniform'`` spaces the scan entries evenly over the colormap by their
    rank in lambda, so consecutive lambdas stay equally distinguishable however
    they are spaced; ``scale='value'`` maps the lambda value through ``norm``,
    which is quantitative but crowds together lambdas that sit close.  Entries
    with ``jc == 0`` get ``baseline_color``, and an explicit ``color:`` in the
    config always wins.
    """
    scan = [e for e in entries if e.jc != 0.0]
    rank = {id(e): k for k, e in enumerate(sorted(scan, key=lambda e: e.lam))}
    span = max(len(scan) - 1, 1)

    colors = []
    for entry in entries:
        if entry.color is not None:
            colors.append(entry.color)
        elif entry.jc == 0.0:
            colors.append(baseline_color)
        elif scale == 'uniform':
            colors.append(cmap(rank[id(entry)] / span))
        else:
            colors.append(cmap(norm(entry.lam)))
    return colors


def plot_curves(entries: list[Entry], cfg: DictConfig, out_dir: Path) -> Path:
    """Draw all curves on one axes and save the figure."""
    style = cfg.get('color', {})
    cmap = LinearSegmentedColormap.from_list(
        'blue_red', list(style.get('cmap', ['#1f3ecc', '#7a1fa2', '#cc1f1f']))
    )
    norm = Normalize(
        vmin=float(style.get('vmin', 0.0)), vmax=float(style.get('vmax', 1.0))
    )

    # Baselines first, then the scan in ascending lambda, so the legend reads as
    # a gradient.
    entries = sorted(entries, key=lambda e: (e.jc != 0.0, e.lam))
    colors = assign_colors(
        entries,
        cmap,
        style.get('scale', 'value'),
        norm,
        style.get('baseline_color', 'black'),
    )
    uniform_jc = len({e.jc for e in entries if e.jc != 0.0}) <= 1

    plot_cfg = cfg.get('plot', {})
    show_n = plot_cfg.get('show_n', True)
    band = plot_cfg.get('band', True)

    fig, ax = plt.subplots(figsize=tuple(plot_cfg.get('figsize', [8.0, 5.0])))

    for entry, color in zip(entries, colors):
        curve = entry.curve

        if band and entry.n_seeds > 1:
            ax.fill_between(
                curve['epoch'],
                curve['lo'],
                curve['hi'],
                color=color,
                alpha=plot_cfg.get('band_alpha', 0.15),
                linewidth=0,
            )

        ax.plot(
            curve['epoch'],
            curve['value'],
            color=color,
            linewidth=1.4,
            label=auto_label(entry, uniform_jc, show_n),
        )
        # The star sits at the mean over seeds of the best value each one
        # reached — NOT at the minimum of the mean curve, which would be biased
        # low by whichever seed happened to dip at that epoch.
        ax.errorbar(
            entry.best_epoch,
            entry.best_value,
            yerr=entry.best_error or None,
            marker='*',
            markersize=12,
            color=color,
            markeredgecolor='white',
            markeredgewidth=0.5,
            linestyle='none',
            elinewidth=1.2,
            capsize=3,
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


def summarise(entries: list[Entry], error: str) -> pd.DataFrame:
    """The numbers behind the stars, one row per configuration."""
    return pd.DataFrame(
        [
            {
                'label': e.label or '',
                'ce': e.ce,
                'jc': e.jc,
                'lambda': e.lam,
                'n_seeds': e.n_seeds,
                # Empty for legacy run directories, which carry a sweep index
                # rather than a seed in their name.
                'seeds': ' '.join(str(s) for s in e.seeds if s >= 0),
                'best_mean': e.best_value,
                f'best_{error}': e.best_error,
                'best_min': e.best_values.min(),
                'best_max': e.best_values.max(),
                'best_epoch_mean': e.best_epoch,
                'common_epochs': len(e.curve),
                'best_per_seed': ' '.join(f'{v:.5f}' for v in e.best_values),
            }
            for e in sorted(entries, key=lambda e: (e.jc != 0.0, e.lam))
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Overlay a per-epoch validation metric across training runs.'
    )
    parser.add_argument('--config', type=Path, required=True, help='YAML config path')
    parser.add_argument('--output', type=Path, help='Override plot.out_dir')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    metric = cfg.get('metric', 'val/head/loss_ce')
    error = cfg.get('error', 'std')

    entries = []
    for raw in cfg['models']:
        entry = load_entry(raw, metric, cfg)
        entries.append(entry)
        spread = f' ± {entry.best_error:.5f}' if entry.n_seeds > 1 else ''
        print(
            f'{raw.get("label") or raw.get("cell") or raw.get("dir") or raw["dirs"][0]}\n'
            f'  ce={entry.ce:g}  jc={entry.jc:g}  lambda={entry.lam:g}  '
            f'n={entry.n_seeds}  best={entry.best_value:.5f}{spread} '
            f'@ epoch {entry.best_epoch:g}'
        )

    out_dir = args.output or Path(
        cfg.get('plot', {}).get('out_dir', 'plots/val_curves')
    )
    out_path = plot_curves(entries, cfg, out_dir)

    summary = summarise(entries, error)
    summary_path = out_dir / f'{out_path.stem}_summary.csv'
    summary.to_csv(summary_path, index=False)

    print(f'\n{summary.to_string(index=False)}')
    print(f'\nSaved {out_path}\nSaved {summary_path}')


if __name__ == '__main__':
    main()
