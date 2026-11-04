"""Best validation CE against one config value (lr, number of labelled jets, ...) per training method (arm).

Every completed run under ``method_scan.stages`` is put in an arm from its own .hydra config, as in
``plot_arm_epochs.py``; its x value is the config key ``x_key`` (``x_null`` when the key is null, e.g. no
``train_max_rows`` = the whole train split).  ``extra_runs`` adds runs from other stages (e.g. the full-data final
runs as the 100% point of a label-fraction scan).  Two figures:

  best_vs_x   best val/head/loss_ce of every run (small markers) and the arm's mean ± sample std per x value
              (large markers, joined by a line), one colour per arm
  curves      one panel per arm: validation CE (solid) and training CE (dashed) per epoch of every run, shaded
              light to dark by x value

plus ``run_selection.csv``.  Every word on the figures comes from ``method_scan.texts``; a missing key is an error.

Usage::

    python scripts/plot_method_scan.py --config scripts/configs/method_scan_lr.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import to_rgb
from omegaconf import OmegaConf

# PNG only, never a GUI window (a dead X11 forward would hang plt.subplots()).
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402 -- after use()
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import plot_arm_epochs as pae  # noqa: E402 -- run config + arm classification
import plot_val_curves as pvc  # noqa: E402 -- metric readers


def select(ac: dict) -> pd.DataFrame:
    """Completed runs of the listed arms, with their x value and best validation CE."""
    dirs = [d for stage in ac['stages'] for d in sorted(Path(stage).glob('*/seed_*'))]
    dirs += [Path(d) for d in ac.get('extra_runs') or []]
    rows = []
    for run_dir in dirs:
        info = yaml.safe_load((run_dir / 'run_info.yaml').read_text())
        if info.get('status') != 'completed':
            continue
        meta = pae.classify(run_dir)
        arm = next(
            (
                key
                for key, a in ac['arms'].items()
                if a['encoder'] == meta['encoder']
                and a['pretraining'] == meta['pretraining']
                and float(a['jet_contrastive']) == meta['jet_contrastive']
            ),
            None,
        )
        if arm is None:
            continue
        x = OmegaConf.select(pae.run_config(run_dir), ac['x_key'])
        val = pvc.read_metric(pvc.find_metrics(run_dir), ac['val_metric'])
        best = val.loc[val['value'].idxmin()]
        rows.append(
            {
                'arm': arm,
                'x': float(ac['x_null'] if x is None else x),
                'seed': meta['seed'],
                'run_dir': str(run_dir),
                'best_val_ce': float(best['value']),
                'best_epoch': int(best['epoch']) + int(ac['epoch_offset']),
                'epochs_run': len(val),
            }
        )
    runs = pd.DataFrame(rows)
    print(runs.sort_values(['arm', 'x', 'seed']).to_string(index=False))
    return runs


def shades(colour: str, n: int) -> list[tuple[float, float, float]]:
    """``n`` steps of one hue, light to dark (mixed with white)."""
    rgb = np.array(to_rgb(colour))
    mix = np.linspace(0.35, 1.0, n) if n > 1 else np.array([1.0])
    return [tuple(1 - t * (1 - rgb)) for t in mix]


def plot_best(runs: pd.DataFrame, ac: dict, out_path: Path) -> None:
    texts = ac['texts']
    fig, ax = plt.subplots(figsize=tuple(ac['best_figsize']))
    handles = []
    for key, arm in ac['arms'].items():
        sub = runs[runs['arm'] == key]
        if sub.empty:
            continue
        ax.plot(
            sub['x'], sub['best_val_ce'], 'o', ms=4, color=arm['colour'], alpha=0.45
        )
        stats = sub.groupby('x')['best_val_ce'].agg(['mean', 'std']).fillna(0.0)
        handles.append(
            ax.errorbar(
                stats.index,
                stats['mean'],
                yerr=stats['std'],
                marker='o',
                ms=7,
                lw=1.6,
                capsize=4,
                color=arm['colour'],
                label=texts['arm'][key],
            )
        )
    ax.set_xscale(ac['x_scale'])
    ax.set_xlabel(texts['xlabel'])
    ax.set_ylabel(texts['ylabel_best'])
    if ac.get('ylim_best'):
        ax.set_ylim(*ac['ylim_best'])
    if texts['title_best']:
        ax.set_title(texts['title_best'], fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    handles += [
        Line2D(
            [],
            [],
            ls='',
            marker='o',
            ms=4,
            color=ac['ink'],
            alpha=0.45,
            label=texts['run'],
        ),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc=ac['legend_loc'])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_curves(runs: pd.DataFrame, ac: dict, out_path: Path) -> None:
    texts, offset = ac['texts'], int(ac['epoch_offset'])
    arms = [k for k in ac['arms'] if k in set(runs['arm'])]
    xs = sorted(runs['x'].unique())
    fig, axes = plt.subplots(
        1, len(arms), figsize=tuple(ac['curves_figsize']), sharey=True, squeeze=False
    )
    for ax, key in zip(axes[0], arms):
        colours = dict(zip(xs, shades(ac['arms'][key]['colour'], len(xs))))
        for run in runs[runs['arm'] == key].itertuples():
            csv_path = pvc.find_metrics(Path(run.run_dir))
            for split, style in (('val', '-'), ('train', '--')):
                s = pvc.read_metric(csv_path, ac[f'{split}_metric'])
                ax.plot(
                    s['epoch'] + offset,
                    s['value'],
                    ls=style,
                    lw=1.2,
                    color=colours[run.x],
                )
        ax.set_title(texts['arm'][key], fontsize=10)
        ax.set_xlabel(texts['xlabel_curves'])
        ax.grid(True, alpha=0.3)
        handles = [
            Line2D([], [], color=colours[x], lw=2, label=texts['x_value'].format(x=x))
            for x in xs
            if ((runs['arm'] == key) & (runs['x'] == x)).any()
        ]
        handles += [
            Line2D([], [], color=ac['ink'], ls='-', lw=1.2, label=texts['val']),
            Line2D([], [], color=ac['ink'], ls='--', lw=1.2, label=texts['train']),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc=ac['legend_loc'])
    axes[0][0].set_ylabel(texts['ylabel_curves'])
    if ac.get('ylim_curves'):
        axes[0][0].set_ylim(*ac['ylim_curves'])
    if texts['title_curves']:
        fig.suptitle(texts['title_curves'], fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=Path, required=True, help='YAML config')
    args = parser.parse_args()

    ac = OmegaConf.to_container(
        OmegaConf.load(args.config)['method_scan'], resolve=True
    )
    out_dir = Path(ac['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = select(ac)
    runs.to_csv(out_dir / 'run_selection.csv', index=False)
    for name, plot in (('best_vs_x', plot_best), ('curves', plot_curves)):
        path = out_dir / ac['filenames'][name]
        plot(runs, ac, path)
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
