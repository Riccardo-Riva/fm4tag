"""Test-set rejections of every final run per training method, with their spread over seeds.

Reads ``per_run.csv`` of ``arm_test_metrics.py`` (one row per run plus the reference tagger's row) and draws, per
figure, one panel per background flavour (c, light, tau): the rejection at one b-efficiency working point with
either the nominal or each run's own validation-optimised (f_c, f_tau).  Each arm sits at its method (x) in its
group (without / with JC: marker and offset), as in ``plot_arm_summary.py``; every run is a small marker and the
arm's mean ± sample std the large one.  The reference tagger is a dashed line when the figure asks for it.  A figure
with ``relative_to: <arm>`` divides every rejection of a panel by the mean rejection of that arm (the arm sits at
1.0, drawn as a line).  Every word on the figures comes from ``test_rejections.texts``; a missing key is an error.

Usage::

    python scripts/plot_test_rejections.py --config scripts/configs/test_rejections.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from omegaconf import OmegaConf

# PNG only, never a GUI window (a dead X11 forward would hang plt.subplots()).
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402 -- after use()
from matplotlib.lines import Line2D  # noqa: E402


def column(fractions: str, flavour: str, wp: float) -> str:
    """``per_run.csv`` column of one rejection (as written by arm_test_metrics.py)."""
    return f'{fractions}_{flavour}_rej@{wp:g}'


def draw_panel(
    ax,
    per_run: pd.DataFrame,
    ac: dict,
    col: str,
    reference: bool,
    relative_to: str | None = None,
) -> None:
    groups, texts = ac['groups'], ac['texts']
    scale = 1.0
    if relative_to:
        base = per_run.loc[per_run['arm'] == relative_to, col].dropna()
        if not len(base):
            raise ValueError(f'no runs of arm {relative_to} in {ac["per_run_csv"]}')
        scale = 1.0 / float(base.mean())
    for key, arm in ac['arms'].items():
        values = scale * per_run.loc[per_run['arm'] == key, col].to_numpy()
        if not len(values):
            raise ValueError(f'no runs of arm {key} in {ac["per_run_csv"]}')
        group, colour = groups[arm['group']], ac['colours'][arm['method']]
        x0 = ac['methods'].index(arm['method']) + float(group['offset'])
        face = colour if group['filled'] else 'white'
        jitter = float(ac['jitter'])
        for i, value in enumerate(values):
            ax.plot(
                x0 + (i - (len(values) - 1) / 2) * jitter,
                value,
                marker=group['marker'],
                markersize=5,
                color=colour,
                mfc=face,
                alpha=0.6,
            )
        ax.errorbar(
            x0,
            values.mean(),
            yerr=values.std(ddof=1) if len(values) > 1 else 0.0,
            marker=group['marker'],
            markersize=9,
            color=colour,
            mfc=face,
            mew=1.6,
            capsize=4,
            elinewidth=1.4,
        )
    if reference:
        ref = per_run.loc[per_run['arm'] == ac['reference_label'], col]
        ax.axhline(scale * float(ref.iloc[0]), color=ac['ink'], ls='--', lw=1.2)
    if relative_to:
        ax.axhline(1.0, color=ac['ink'], ls=':', lw=1.0)
    ax.set_xticks(range(len(ac['methods'])))
    ax.set_xticklabels([texts['method'][m] for m in ac['methods']], fontsize=9)
    ax.set_xlim(-0.6, len(ac['methods']) - 0.4)
    ax.grid(True, axis='y', alpha=0.3)


def legend_handles(ac: dict, reference: bool, relative: bool) -> list[Line2D]:
    texts, ink = ac['texts'], ac['ink']
    handles = [
        Line2D(
            [],
            [],
            ls='',
            marker=g['marker'],
            markersize=8,
            color=ink,
            mfc=ink if g['filled'] else 'white',
            label=texts['group'][name],
        )
        for name, g in ac['groups'].items()
    ]
    handles += [
        Line2D(
            [],
            [],
            ls='',
            marker='o',
            markersize=5,
            color=ink,
            alpha=0.6,
            label=texts['run'],
        ),
        Line2D(
            [], [], ls='-', marker='o', markersize=8, color=ink, label=texts['mean']
        ),
    ]
    if reference:
        handles.append(
            Line2D([], [], ls='--', lw=1.2, color=ink, label=texts['reference'])
        )
    if relative:
        handles.append(
            Line2D(
                [], [], ls=':', lw=1.0, color=ink, label=texts['relative']['baseline']
            )
        )
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=Path, required=True, help='YAML config')
    args = parser.parse_args()

    ac = OmegaConf.to_container(
        OmegaConf.load(args.config)['test_rejections'], resolve=True
    )
    texts = ac['texts']
    per_run = pd.read_csv(ac['per_run_csv'])
    out_dir = Path(ac['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    for fig_cfg in ac['figures']:
        wp, fractions = float(fig_cfg['working_point']), fig_cfg['fractions']
        reference = bool(fig_cfg['reference'])
        relative_to = fig_cfg.get('relative_to')
        rel = texts['relative'] if relative_to else None
        fig, axes = plt.subplots(
            1, len(ac['flavours']), figsize=tuple(ac['figsize']), squeeze=False
        )
        for ax, flavour in zip(axes[0], ac['flavours']):
            draw_panel(
                ax, per_run, ac, column(fractions, flavour, wp), reference, relative_to
            )
            ax.set_title(texts['flavour'][flavour], fontsize=10)
        axes[0][0].set_ylabel((rel or texts)['ylabel'].format(wp=100 * wp))
        title = (rel or texts)['title'].format(
            wp=100 * wp, fractions=texts['fractions'][fractions]
        )
        if title:
            fig.suptitle(title, fontsize=11)
        fig.legend(
            handles=legend_handles(ac, reference, bool(relative_to)),
            loc='lower center',
            ncol=len(ac['groups']) + 2 + int(reference) + int(bool(relative_to)),
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, -0.02),
        )
        footnote = '\n'.join(
            t
            for t in (
                texts['footnote'],
                texts['footnote_fractions'][fractions],
                rel['footnote'] if rel else '',
            )
            if t
        )
        if footnote:
            fig.text(
                0.5,
                -0.07,
                footnote,
                ha='center',
                va='top',
                fontsize=8,
                color=ac['ink'],
            )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        path = out_dir / fig_cfg['filename']
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
