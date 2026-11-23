"""Learning-rate search of the WSD redo (W_lr.cells): best validation CE per learning rate, and the validation-CE
curves at each cell's chosen learning rate.

Reads every ``<stage>/<cell>/seed_<seed>`` run of the stages in the config (cell names
``<arm>[_d02]_lr<tag>[_bb10|_bb30]``, scripts/make_wsd_cells.py) and writes:

- ``lr_scan.png``: one panel per arm, best val CE (over the 40 epochs) against the learning rate, no dropout filled /
  dropout open, backbone lr/10 solid / lr/30 dashed; dotted line = the best from-scratch run without dropout.
- ``curves_at_best_lr.png``: one panel per arm, val CE per epoch at the chosen learning rate, no dropout vs dropout;
  the shaded band is the WSD decay.
- ``selection.csv`` (cell,lr,bb,best,best_epoch): the chosen lr per (arm, dropout) = the lowest best val CE over lr and
  backbone ratio; the input of ``make_wsd_cells.py --kind final --choice``.  ``runs.csv``: every run.

All wording is in the config's ``texts``.

    MPLBACKEND=Agg .venv/bin/python scripts/plot_lr_search.py --config experiments/2026-11_wsd-lr-search/configs/wsd_lr_search.yaml
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

CELL = re.compile(r'(?P<arm>.+?)(?P<drop>_d02)?_lr(?P<lr>\d+)(?:_(?P<bb>bb\d+))?$')


def load_runs(cfg: dict) -> tuple[pd.DataFrame, dict]:
    rows, curves = [], {}
    for stage in cfg['stages']:
        for d in sorted(glob.glob(f'{stage}/*/seed_{cfg["seed"]}')):
            cell = Path(d).parent.name
            m = CELL.match(cell)
            if not m:
                continue
            files = glob.glob(f'{d}/**/metrics.csv', recursive=True)
            if not files:
                continue
            df = pd.concat([pd.read_csv(f) for f in files])
            v = (
                df[['epoch', cfg['metric']]]
                .dropna()
                .groupby('epoch')[cfg['metric']]
                .last()
            )
            key = (m['arm'], bool(m['drop']), m['lr'], m['bb'] or '-')
            curves[key] = v
            rows.append(
                dict(
                    cell=cell,
                    arm=m['arm'],
                    dropout=bool(m['drop']),
                    lr_tag=m['lr'],
                    lr=float(cfg['lr_values'][m['lr']]),
                    bb=m['bb'] or '-',
                    best=float(v.min()),
                    best_epoch=int(v.idxmin()),
                    last=float(v.iloc[-1]),
                    epochs=int(v.index.max()) + 1,
                )
            )
    return pd.DataFrame(rows), curves


def select(runs: pd.DataFrame) -> pd.DataFrame:
    best = runs.loc[runs.groupby(['arm', 'dropout'])['best'].idxmin()].copy()
    best['cell'] = best['arm'] + np.where(best['dropout'], '_d02', '')
    best['bb_ratio'] = best['bb'].str.replace('bb', '').replace('-', '')
    return best


def panel_grid(cfg: dict):
    arms = cfg['arms']
    ncol = int(cfg['ncol'])
    nrow = int(np.ceil(len(arms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=tuple(cfg['figsize']), squeeze=False)
    for ax in axes.flat[len(arms) :]:
        ax.set_visible(False)
    return fig, axes.flat


def legend(fig, cfg: dict, items: list[tuple[str, dict]]) -> None:
    handles = [Line2D([], [], label=label, **style) for label, style in items]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )


def footnote(fig, cfg: dict, text: str) -> None:
    if text:
        fig.text(0.5, -0.02, text, ha='center', va='top', fontsize=8, color=cfg['ink'])


def plot_scan(runs: pd.DataFrame, cfg: dict, out: Path) -> None:
    t = cfg['texts']
    fig, axes = panel_grid(cfg)
    ref = runs.loc[
        (runs['arm'] == cfg['reference_arm']) & ~runs['dropout'], 'best'
    ].min()
    for ax, (arm, spec) in zip(axes, cfg['arms'].items()):
        colour = cfg['colours'][spec['method']]
        sub = runs[runs['arm'] == arm]
        for (drop, bb), g in sub.groupby(['dropout', 'bb']):
            g = g.sort_values('lr')
            ax.plot(
                g['lr'],
                g['best'],
                color=colour,
                ls='--' if bb == 'bb30' else '-',
                lw=2,
                marker='o',
                markersize=8,
                mfc='white' if drop else colour,
                mew=2,
            )
        ax.axhline(ref, color=cfg['ink'], ls=':', lw=1)
        ax.set_xscale('log')
        ax.set_xticks(list(cfg['lr_values'].values()))
        ax.set_xticklabels(
            [t['lr_tick'].format(lr=x) for x in cfg['lr_values'].values()]
        )
        ax.minorticks_off()
        ax.set_title(t['arm'][arm], fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(labelsize=8)
    for ax in list(axes)[:: int(cfg['ncol'])]:
        ax.set_ylabel(t['scan_ylabel'], fontsize=9)
    for ax in list(axes)[-int(cfg['ncol']) :]:
        ax.set_xlabel(t['lr_label'], fontsize=9)
    fig.suptitle(t['scan_title'], fontsize=11)
    ink = cfg['ink']
    legend(
        fig,
        cfg,
        [
            (
                t['no_dropout'],
                dict(color=ink, marker='o', mfc=ink, ls='', markersize=8),
            ),
            (
                t['dropout'],
                dict(color=ink, marker='o', mfc='white', mew=2, ls='', markersize=8),
            ),
            (t['bb10'], dict(color=ink, ls='-', lw=2)),
            (t['bb30'], dict(color=ink, ls='--', lw=2)),
            (t['reference'].format(ref=ref), dict(color=ink, ls=':', lw=1)),
        ],
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    footnote(fig, cfg, t['scan_footnote'])
    fig.savefig(out / 'lr_scan.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_curves(sel: pd.DataFrame, curves: dict, cfg: dict, out: Path) -> None:
    t = cfg['texts']
    fig, axes = panel_grid(cfg)
    decay_start = float(cfg['epochs']) * (1 - float(cfg['decay_frac']))
    for ax, (arm, spec) in zip(axes, cfg['arms'].items()):
        colour = cfg['colours'][spec['method']]
        ax.axvspan(
            decay_start - 0.5,
            float(cfg['epochs']) - 0.5,
            color=cfg['ink'],
            alpha=0.07,
            lw=0,
        )
        for _, r in sel[sel['arm'] == arm].iterrows():
            v = curves[(arm, r['dropout'], r['lr_tag'], r['bb'])]
            ax.plot(
                v.index, v.values, color=colour, lw=2, ls='--' if r['dropout'] else '-'
            )
            ax.plot(
                r['best_epoch'],
                r['best'],
                marker='o',
                markersize=8,
                color=colour,
                mfc='white' if r['dropout'] else colour,
                mew=2,
            )
        lo = sel[sel['arm'] == arm]['best'].min()
        ax.set_ylim(lo - 0.01, lo + float(cfg['curves_ylim_span']))
        ax.set_title(t['arm'][arm], fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(labelsize=8)
    for ax in list(axes)[:: int(cfg['ncol'])]:
        ax.set_ylabel(t['curves_ylabel'], fontsize=9)
    for ax in list(axes)[-int(cfg['ncol']) :]:
        ax.set_xlabel(t['epoch_label'], fontsize=9)
    fig.suptitle(t['curves_title'], fontsize=11)
    ink = cfg['ink']
    legend(
        fig,
        cfg,
        [
            (t['no_dropout'], dict(color=ink, ls='-', lw=2)),
            (t['dropout'], dict(color=ink, ls='--', lw=2)),
            (t['best_marker'], dict(color=ink, marker='o', ls='', markersize=8)),
            (t['decay_band'], dict(color=ink, alpha=0.25, lw=8)),
        ],
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    footnote(fig, cfg, t['curves_footnote'])
    fig.savefig(out / 'curves_at_best_lr.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config)['lr_search'], resolve=True)
    out = Path(cfg['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    runs, curves = load_runs(cfg)
    runs.sort_values(['arm', 'dropout', 'bb', 'lr']).to_csv(
        out / 'runs.csv', index=False
    )
    sel = select(runs)
    sel[['cell', 'lr', 'bb_ratio', 'best', 'best_epoch']].rename(
        columns={'bb_ratio': 'bb'}
    ).to_csv(out / 'selection.csv', index=False)
    print(f'{len(runs)} runs; selection:')
    print(sel[['cell', 'lr', 'bb', 'best', 'best_epoch']].to_string(index=False))
    plot_scan(runs, cfg, out)
    plot_curves(sel, curves, cfg, out)
    print(
        f'Saved {out}/lr_scan.png, {out}/curves_at_best_lr.png, {out}/selection.csv, {out}/runs.csv'
    )


if __name__ == '__main__':
    main()
