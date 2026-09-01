"""Per-seed validation-metric plots — one line (or one marker) per seed.

Companion to ``scripts/plot_val_curves.py``.  That script collapses the seeds of
a configuration into a seed-mean curve with a band and a single star; this one
keeps every seed separate:

  ``--style curves``   every seed's own curve, thin, plus a small star at that
                       seed's best (lowest) value.
  ``--style markers``  no curves — only a borderless square at each seed's best
                       value (same axes: epoch vs metric).
  ``--style both``     write both (default); the markers file gets a
                       ``_markers`` suffix.

All seeds of one configuration share a colour; every ``jc == 0`` entry is drawn
in ``color.baseline_color`` (black by default).  The config format is exactly
``plot_val_curves.py``'s — ``metric``, ``seed_status``, ``paths``, ``color``,
``plot`` and a ``models`` list of ``cell:`` / ``dir:`` / ``dirs:`` entries — and
the run-resolution / metric-reading / colour machinery is imported from it
rather than duplicated.

Usage::

    python scripts/plot_seed_curves.py --config scripts/configs/val_loss_ce_seed_curves.yaml
    python scripts/plot_seed_curves.py --config ... --style markers --output plots/x
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
import plot_val_curves as pvc  # noqa: E402 -- run/metric/colour helpers reused


# Qualitative default: one clearly distinct hue per lambda (matplotlib tab10
# order, minus its grey which is too close to the jc=0 baseline black).  Used
# unless the config's `color.palette` overrides it.  A ramp colormap
# (`color.cmap` + `color.scale`) is still available by setting
# `color.categorical: false`.
_DISTINCT_PALETTE = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#17becf',  # cyan
    '#bcbd22',  # olive
]


def _entries_and_colors(cfg):
    """Load every model entry and assign one colour per entry.

    Default: a qualitative palette — each lambda gets a wholly distinct hue, by
    its rank in lambda; every ``jc == 0`` entry gets ``color.baseline_color``
    (black); an explicit per-entry ``color:`` always wins.  Set
    ``color.categorical: false`` to fall back to pvc's ramp-colormap shading.
    """
    metric = cfg.get('metric', 'val/head/loss_ce')
    entries = [pvc.load_entry(raw, metric, cfg) for raw in cfg['models']]
    # Baselines first, then ascending lambda — same order pvc plots in, so the
    # legend reads the same across the two scripts.
    entries.sort(key=lambda e: (e.jc != 0.0, e.lam))

    style = cfg.get('color', {})
    baseline_color = style.get('baseline_color', 'black')

    if style.get('categorical', True):
        palette = list(style.get('palette', _DISTINCT_PALETTE))
        scan = sorted((e for e in entries if e.jc != 0.0), key=lambda e: e.lam)
        rank = {id(e): k for k, e in enumerate(scan)}
        colors = [
            e.color
            if e.color is not None
            else baseline_color
            if e.jc == 0.0
            else palette[rank[id(e)] % len(palette)]
            for e in entries
        ]
    else:
        cmap = LinearSegmentedColormap.from_list(
            'blue_red', list(style.get('cmap', ['#1f3ecc', '#7a1fa2', '#cc1f1f']))
        )
        norm = Normalize(
            vmin=float(style.get('vmin', 0.0)), vmax=float(style.get('vmax', 1.0))
        )
        colors = pvc.assign_colors(
            entries, cmap, style.get('scale', 'uniform'), norm, baseline_color
        )
    return entries, colors, metric


def _apply_axes(ax, cfg, metric):
    p = cfg.get('plot', {})
    ax.set_xlabel(p.get('xlabel', 'Epoch'))
    ax.set_ylabel(p.get('ylabel', metric))
    if p.get('title'):
        ax.set_title(p['title'])
    if p.get('ylim'):
        ax.set_ylim(*p['ylim'])
    if p.get('xlim'):
        ax.set_xlim(*p['xlim'])
    ax.grid(True, alpha=0.3)


def _legend(ax, entries, colors, cfg):
    p = cfg.get('plot', {})
    uniform_jc = len({e.jc for e in entries if e.jc != 0.0}) <= 1
    show_n = p.get('show_n', True)
    handles = [
        Line2D([], [], color=c, lw=2, label=pvc.auto_label(e, uniform_jc, show_n))
        for e, c in zip(entries, colors)
    ]
    kw = {'fontsize': 8, 'title': p.get('legend_title')}
    if p.get('legend_outside', True):
        kw |= {'loc': 'upper left', 'bbox_to_anchor': (1.01, 1.0)}
    else:
        kw |= {'loc': 'best', 'ncol': 2}
    ax.legend(handles=handles, **kw)


def _figure(cfg):
    p = cfg.get('plot', {})
    return plt.subplots(figsize=tuple(p.get('figsize', [8.5, 5.0])))


def plot_seed_curves(entries, colors, cfg, metric, out_path: Path) -> Path:
    """Every seed's own curve + a small star at its best value."""
    p = cfg.get('plot', {})
    star_ms = float(p.get('star_size', 6))
    fig, ax = _figure(cfg)

    for entry, color in zip(entries, colors):
        for curve in entry.curves:
            ax.plot(
                curve['epoch'],
                curve['value'],
                color=color,
                linewidth=0.8,
                alpha=p.get('curve_alpha', 0.65),
            )
            imin = curve['value'].idxmin()
            ax.plot(
                curve.loc[imin, 'epoch'],
                curve.loc[imin, 'value'],
                marker='*',
                markersize=star_ms,
                color=color,
                markeredgecolor='white',
                markeredgewidth=0.3,
                linestyle='none',
                zorder=5,
            )

    _apply_axes(ax, cfg, metric)
    _legend(ax, entries, colors, cfg)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_seed_markers(entries, colors, cfg, metric, out_path: Path) -> Path:
    """Only a borderless square at each seed's best value — no curves."""
    p = cfg.get('plot', {})
    sq_ms = float(p.get('marker_size', 6))
    fig, ax = _figure(cfg)

    for entry, color in zip(entries, colors):
        for be, bv in zip(entry.best_epochs, entry.best_values):
            ax.plot(
                be,
                bv,
                marker='s',
                markersize=sq_ms,
                color=color,
                markeredgewidth=0,
                linestyle='none',
                zorder=5,
            )

    _apply_axes(ax, cfg, metric)
    _legend(ax, entries, colors, cfg)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Override plot.out_dir')
    parser.add_argument(
        '--style', choices=['curves', 'markers', 'both'], default='both'
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    entries, colors, metric = _entries_and_colors(cfg)

    for entry in entries:
        print(
            f'  ce={entry.ce:g}  jc={entry.jc:g}  lambda={entry.lam:g}  '
            f'n={entry.n_seeds}  best_per_seed='
            + ' '.join(f'{v:.5f}' for v in entry.best_values)
        )

    out_dir = args.output or Path(
        cfg.get('plot', {}).get('out_dir', 'plots/val_curves')
    )
    fname = cfg.get('plot', {}).get('filename', 'seed_curves.png')
    stem = Path(fname).stem
    suffix = Path(fname).suffix or '.png'

    if args.style in ('curves', 'both'):
        pth = plot_seed_curves(
            entries, colors, cfg, metric, out_dir / f'{stem}{suffix}'
        )
        print(f'Saved {pth}')
    if args.style in ('markers', 'both'):
        pth = plot_seed_markers(
            entries, colors, cfg, metric, out_dir / f'{stem}_markers{suffix}'
        )
        print(f'Saved {pth}')


if __name__ == '__main__':
    main()
