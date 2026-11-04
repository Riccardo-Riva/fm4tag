"""Best validation CE of every run, per training method, without and with the jet-embedding contrastive term (JC).

The runs are selected and put in arms from their .hydra configs by ``plot_arm_epochs.select_runs``.  Each arm
belongs to a method (x position) and a group (without / with JC: marker and offset), both set in the YAML.  Every
run is one small marker; the mean ± sample std of the arm's shown runs is the large one.  Runs listed under
``exclude`` (collapsed trainings) are left out explicitly and counted next to their arm.  Every word on the figure
comes from ``arm_summary.texts``; a missing key is an error.

Usage::

    python scripts/plot_arm_summary.py --config scripts/configs/arm_summary.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
from omegaconf import OmegaConf

# PNG only, never a GUI window (a dead X11 forward would hang plt.subplots()).
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402 -- after use()
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import plot_arm_epochs as pae  # noqa: E402 -- run selection from .hydra configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=Path, required=True, help='YAML config')
    args = parser.parse_args()

    ac = OmegaConf.to_container(
        OmegaConf.load(args.config)['arm_summary'], resolve=True
    )
    texts, groups = ac['texts'], ac['groups']
    runs = pae.select_runs(ac)
    excluded = runs['run_dir'].isin(ac['exclude'])
    missing = set(ac['exclude']) - set(runs['run_dir'])
    if missing:
        raise ValueError(
            f'exclude lists runs that were not selected: {sorted(missing)}'
        )
    out_dir = Path(ac['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs.assign(excluded=excluded).to_csv(out_dir / 'run_selection.csv', index=False)

    fig, ax = plt.subplots(figsize=tuple(ac['figsize']))
    for key, arm in ac['arms'].items():
        group, colour = groups[arm['group']], ac['colours'][arm['method']]
        x0 = ac['methods'].index(arm['method']) + float(group['offset'])
        shown = runs[(runs['arm'] == key) & ~excluded]['best_val_ce'].to_numpy()
        n_out = int(((runs['arm'] == key) & excluded).sum())
        face = colour if group['filled'] else 'white'
        jitter = float(ac['jitter'])
        for i, value in enumerate(shown):
            x = x0 + (i - (len(shown) - 1) / 2) * jitter
            ax.plot(
                x,
                value,
                marker=group['marker'],
                markersize=5,
                color=colour,
                mfc=face,
                alpha=0.6,
            )
        if len(shown):
            mean = shown.mean()
            err = shown.std(ddof=1) if len(shown) > 1 else 0.0
            ax.errorbar(
                x0 + float(ac['mean_offset']),
                mean,
                yerr=err,
                marker=group['marker'],
                markersize=9,
                color=colour,
                mfc=face,
                mew=1.6,
                capsize=4,
                elinewidth=1.4,
            )
        if n_out:
            ax.annotate(
                texts['excluded'].format(n=n_out),
                (x0, shown.max() if len(shown) else ax.get_ylim()[1]),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=8,
                color=ac['ink'],
            )

    ax.set_xticks(range(len(ac['methods'])))
    ax.set_xticklabels([texts['method'][m] for m in ac['methods']])
    ax.set_xlim(-0.6, len(ac['methods']) - 0.4)
    ax.set_ylabel(texts['ylabel'])
    if ac.get('ylim'):
        ax.set_ylim(*ac['ylim'])
    if texts['title']:
        ax.set_title(texts['title'], fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    handles = [
        Line2D(
            [],
            [],
            linestyle='',
            marker=g['marker'],
            markersize=8,
            color=ac['ink'],
            mfc=ac['ink'] if g['filled'] else 'white',
            label=texts['group'][name],
        )
        for name, g in groups.items()
    ]
    handles += [
        Line2D(
            [],
            [],
            linestyle='',
            marker='o',
            markersize=5,
            color=ac['ink'],
            alpha=0.6,
            label=texts['run'],
        ),
        Line2D(
            [],
            [],
            linestyle='-',
            marker='o',
            markersize=8,
            color=ac['ink'],
            label=texts['mean'],
        ),
    ]
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=9)
    if texts['footnote']:
        fig.text(
            0.01,
            -0.02,
            texts['footnote'],
            fontsize=8,
            color=ac['ink'],
            va='top',
            wrap=True,
        )
    fig.tight_layout()
    out_path = out_dir / ac['filename']
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
