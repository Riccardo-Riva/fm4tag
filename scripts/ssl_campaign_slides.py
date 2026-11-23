#!/usr/bin/env python
"""Build the SSL campaign slides: plots from the run directories, then a pptx.

    uv run --no-sync --with python-pptx python scripts/ssl_campaign_slides.py \
        --config experiments/2026-10_ssl-campaign/configs/slides/ssl_campaign.yaml

Every word on the slides and in the plots comes from the YAML (slide texts,
plot ``texts:``). Re-run after editing it or when runs finish: each plot
re-reads its runs, so new results appear by themselves, and a plot without
data yet leaves its ``placeholder`` text on the slide.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # a dead X11 forward hangs the Tk backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.5
TITLE_Y, TITLE_H = 0.35, 0.75
BODY_Y = 1.3
BODY_BOTTOM = 6.85


# ── Run directories ──────────────────────────────────────────────────────────
def find_cell(stages: list[str], cell: str, seed: int = 42) -> Path | None:
    """The seed directory of ``cell`` in the first stage that has it."""
    for pattern in stages:
        for stage in sorted(glob.glob(pattern)):
            run = Path(stage) / cell / f'seed_{seed}'
            if run.is_dir():
                return run
    return None


def metrics(run: Path | None) -> pd.DataFrame | None:
    if run is None:
        return None
    files = sorted(run.glob('*/version_*/metrics.csv'), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    return pd.read_csv(files[-1])


def per_epoch(df: pd.DataFrame) -> pd.DataFrame:
    # Lightning writes train-step, train-epoch and val rows separately.
    return df.dropna(subset=['epoch']).groupby('epoch').last()


def run_status(run: Path | None) -> str:
    if run is None:
        return 'queued'
    info = run / 'run_info.yaml'
    if info.exists():
        return yaml.safe_load(info.read_text()).get('status', 'running')
    return 'running'


def max_epochs(run: Path) -> int | None:
    for cfg in sorted(run.glob('outputs/**/.hydra/config.yaml')):
        m = re.search(r'max_epochs:\s*(\d+)', cfg.read_text())
        if m:
            return int(m.group(1))
    return None


# ── Plot style ───────────────────────────────────────────────────────────────
def setup_style(style: dict) -> None:
    plt.rcParams.update(
        {
            'font.family': style['font'],
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'text.color': style['ink'],
            'axes.labelcolor': style['ink'],
            'axes.edgecolor': style['muted'],
            'xtick.color': style['muted'],
            'ytick.color': style['muted'],
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.color': style['grid'],
            'grid.linewidth': 0.8,
            'axes.axisbelow': True,
            'lines.linewidth': 2.0,
            'legend.frameon': False,
            'figure.dpi': 100,
            'savefig.dpi': 220,
        }
    )


def save(fig, path: Path) -> Path:
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# ── Plots: each returns the PNG path, or None when it has no data yet ────────
def plot_model(cfg, style, data, out):
    t = cfg['texts']
    c = style['colors']
    fig, ax = plt.subplots(figsize=cfg['size'])
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0.9, 12)
    ax.axis('off')

    def box(
        x, y, w, h, text, fill='#f4f4f2', edge=style['muted'], size=10.5, bold=False
    ):
        ax.add_patch(
            FancyBboxPatch(
                (x, y - h / 2),
                w,
                h,
                boxstyle='round,pad=0.02,rounding_size=0.18',
                fc=fill,
                ec=edge,
                lw=1.0,
            )
        )
        ax.text(
            x + w / 2,
            y,
            text,
            ha='center',
            va='center',
            fontsize=size,
            fontweight='bold' if bold else 'normal',
            color=style['ink'],
        )

    def arrow(x0, y0, x1, y1):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle='-|>',
                mutation_scale=11,
                color=style['muted'],
                lw=1.2,
                shrinkA=0,
                shrinkB=0,
            )
        )

    lx, lw, rx, rw = 0.1, 4.3, 4.7, 3.9  # tracks column, jet column
    rows = {'in': 11.2, 'enc': 9.55, 'emb': 7.9, 'agg': 6.25, 'glob': 4.6, 'head': 2.95}
    box(lx, rows['in'], lw, 0.8, t['tracks_in'])
    box(rx, rows['in'], rw, 0.8, t['jets_in'])
    box(lx, rows['enc'], lw, 1.25, t['track_enc'])
    box(rx, rows['enc'], rw, 1.25, t['jet_enc'])
    feat = '#e6effa'
    box(lx, rows['emb'], lw, 0.8, t['track_emb'], fill=feat, edge=c['P0'])
    box(rx, rows['emb'], rw, 0.8, t['jet_emb'], fill=feat, edge=c['P0'])
    box(lx, rows['agg'], lw, 1.1, t['aggregator'])
    box(
        lx,
        rows['glob'],
        rx + rw - lx,
        0.8,
        t['global_emb'],
        fill='#e3f5ee',
        edge=c['P2'],
    )
    box(lx, rows['head'], rx + rw - lx, 0.8, t['head'])
    for x in (lx + lw / 2, rx + rw / 2):
        arrow(x, rows['in'] - 0.4, x, rows['enc'] + 0.62)
        arrow(x, rows['enc'] - 0.62, x, rows['emb'] + 0.4)
    arrow(lx + lw / 2, rows['emb'] - 0.4, lx + lw / 2, rows['agg'] + 0.55)
    arrow(lx + lw / 2, rows['agg'] - 0.55, lx + lw / 2, rows['glob'] + 0.4)
    arrow(rx + rw / 2, rows['emb'] - 0.4, rx + rw / 2, rows['glob'] + 0.4)
    arrow(
        lx + (rx + rw - lx) / 2,
        rows['glob'] - 0.4,
        lx + (rx + rw - lx) / 2,
        rows['head'] + 0.4,
    )
    # Loss badges, right of the embeddings they act on
    bx, bw = 9.0, 3.5
    box(bx, rows['emb'], bw, 1.45, t['loss_feat'], fill=c['P0'], edge=c['P0'], size=9.5)
    ax.texts[-1].set_color('white')
    box(bx, rows['glob'], bw, 1.45, t['loss_jc'], fill=c['P2'], edge=c['P2'], size=9.5)
    ax.texts[-1].set_color('white')
    ax.text(
        lx,
        1.6,
        t['chunk_note'],
        ha='left',
        va='center',
        fontsize=10,
        style='italic',
        color=style['muted'],
    )
    return save(fig, out)


def plot_efficiency(cfg, style, data, out):
    t = cfg['texts']
    tables = {
        k: pd.read_csv(data[k]).set_index('point')
        for k in {b['table'] for b in cfg['bars']}
    }
    rows = [tables[b['table']].loc[b['point']] for b in cfg['bars']]
    jets = np.array([r['jets_s'] for r in rows]) / 1e3
    gib = np.array([r['peak_gib'] / r['batch'] * 1e3 for r in rows])  # GiB per 1k jets
    labels = [b['label'] for b in cfg['bars']]

    text = Path(data['loader_bench']).read_text()
    ms = {}
    for key in ('cold', 'warm'):
        m = re.search(
            rf'^{key}.*?jets read\s+([\d.]+) ms\s+tracks read\s+([\d.]+) ms', text, re.M
        )
        ms[key] = float(m.group(1)) + float(m.group(2))

    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=cfg['size'], gridspec_kw={'height_ratios': [2.3, 1]}
    )
    y = np.arange(len(jets))[::-1]
    colors = [style['colors']['bar_muted']] * (len(jets) - 1) + [style['colors']['bar']]
    a1.barh(y, jets, color=colors, height=0.62)
    for yi, j, g in zip(y, jets, gib):
        a1.text(
            j + jets.max() * 0.02,
            yi,
            t['value_fmt'].format(k=j, gib=g),
            va='center',
            fontsize=10,
            color=style['ink'],
        )
    a1.set_yticks(y, labels)
    a1.set_xlim(0, jets.max() * 1.45)
    a1.set_xlabel(t['xlabel_speed'])
    a1.set_title(t['title_speed'], loc='left')
    a1.grid(axis='y', visible=False)

    y2 = np.array([1, 0])
    a2.barh(
        y2,
        [ms['cold'], ms['warm']],
        color=[style['colors']['bar_muted'], style['colors']['bar']],
        height=0.55,
    )
    for yi, v in zip(y2, (ms['cold'], ms['warm'])):
        a2.text(
            v + ms['cold'] * 0.02,
            yi,
            t['ms_fmt'].format(ms=v),
            va='center',
            fontsize=10,
        )
    a2.set_yticks(y2, [t['cold'], t['warm']])
    a2.set_xlim(0, ms['cold'] * 1.3)
    a2.set_xlabel(t['xlabel_loader'])
    a2.set_title(t['title_loader'], loc='left')
    a2.grid(axis='y', visible=False)
    fig.tight_layout(h_pad=1.5)
    return save(fig, out)


def plot_s0_speed(cfg, style, data, out):
    t = cfg['texts']
    df = pd.read_csv(data['s0_table'])
    df = df[df['point'].str.startswith('A_') & df['error'].isna()]
    fig, axes = plt.subplots(1, len(cfg['panels']), figsize=cfg['size'], sharey=True)
    for ax, panel in zip(np.atleast_1d(axes), cfg['panels']):
        for mode, key in (('pt', 'per_token'), ('cc', 'concat')):
            d = df[df['point'].str.startswith(f'A_{panel["arm"]}_{mode}_')].sort_values(
                'chunk'
            )
            ax.plot(
                d['chunk'],
                d['jets_s'] / 1e3,
                marker='o',
                ms=6,
                color=style['colors'][key],
            )
            ax.annotate(
                t[key],
                (d['chunk'].iloc[-1], d['jets_s'].iloc[-1] / 1e3),
                xytext=(6, 0),
                textcoords='offset points',
                va='center',
                fontsize=10,
                color=style['ink'],
            )
        for c in (128, 512):
            ax.axvline(c, color=style['grid'], lw=6, zorder=0)
        ax.set_xscale('log', base=2)
        ax.set_xticks([64, 128, 256, 512, 1024], ['64', '128', '256', '512', '1024'])
        ax.set_xlim(48, 2600)
        ax.set_xlabel(t['xlabel'])
        ax.set_title(panel['title'], loc='left')
    top = df['jets_s'].max() / 1e3 * 1.12
    for ax in np.atleast_1d(axes):
        ax.set_ylim(0, top)
    np.atleast_1d(axes)[0].set_ylabel(t['ylabel'])
    np.atleast_1d(axes)[0].text(
        128,
        0.3,
        t['candidates'],
        rotation=90,
        va='bottom',
        ha='right',
        fontsize=9,
        color=style['muted'],
    )
    fig.tight_layout()
    return save(fig, out)


def plot_s0_memory(cfg, style, data, out):
    t = cfg['texts']
    df = pd.read_csv(data['s0_table'])
    df = df[
        df['point'].str.startswith('B_')
        & df['error'].isna()
        & (df['chunk'] == cfg['chunk'])
    ]
    fig, ax = plt.subplots(figsize=cfg['size'])
    xs = np.linspace(0, cfg['x_max'], 50)
    for arm in cfg['arms']:
        d = df[df['point'].str.startswith(f'B_{arm["arm"]}_{cfg["row_mode"]}_')]
        slope, icpt = np.polyfit(d['batch'], d['peak_gib'], 1)
        color = (
            style['colors'][arm['arm']]
            if arm['arm'] in style['colors']
            else style['ink']
        )
        b = arm['batch']
        ax.plot(
            xs / 1e3,
            icpt + slope * xs,
            color=color,
            ls=arm['ls'],
            lw=1.6,
            label=t['arm_fmt'].format(label=arm['label'], batch=b),
        )
        ax.plot(d['batch'] / 1e3, d['peak_gib'], 'o', color=color, ms=6)
        ax.plot(
            b / 1e3,
            icpt + slope * b,
            marker='D',
            ms=9,
            color=color,
            mec='white',
            mew=1.5,
        )
    ax.axhline(cfg['target_gib'], color=style['muted'], ls=':', lw=1.2)
    ax.axhline(cfg['card_gib'], color=style['muted'], lw=1.2)
    ax.text(
        0.3, cfg['target_gib'] + 0.5, t['target'], fontsize=9.5, color=style['muted']
    )
    ax.text(0.3, cfg['card_gib'] + 0.5, t['card'], fontsize=9.5, color=style['muted'])
    ax.plot([], [], 'o', color=style['muted'], label=t['measured'])
    ax.plot([], [], 'D', color=style['muted'], ms=8, label=t['chosen'])
    ax.legend(loc='lower right', fontsize=9.5)
    ax.set_xlim(0, cfg['x_max'] / 1e3 * 1.6)
    ax.set_ylim(0, cfg['card_gib'] + 3)
    ax.set_xlabel(t['xlabel'])
    ax.set_ylabel(t['ylabel'])
    fig.tight_layout()
    return save(fig, out)


def plot_s1(cfg, style, data, out):
    t = cfg['texts']
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=cfg['size'], gridspec_kw={'width_ratios': [2.4, 1]}
    )
    best = []
    for i, cell in enumerate(cfg['cells']):
        color = style['colors'][cell['mode']]
        ls = cfg['linestyle'][cell['chunk']]
        vals = []
        for seed_dir in sorted((Path(data['s1_stage']) / cell['cell']).glob('seed_*')):
            df = metrics(seed_dir)
            if df is None:
                continue
            ep = per_epoch(df)['val/head/loss_ce'].dropna()
            a1.plot(ep.index + 1, ep.values, color=color, ls=ls, lw=1.6)
            a1.plot(
                ep.idxmin() + 1, ep.min(), 'o', color=color, ms=7, mec='white', mew=1.2
            )
            vals.append(ep.min())
        a1.plot([], [], color=color, ls=ls, label=cell['label'])
        best.append(vals)
        a2.scatter([i] * len(vals), vals, color=color, s=40, zorder=3)
        if vals:
            a2.hlines(np.mean(vals), i - 0.28, i + 0.28, color=color, lw=2.5)
    a1.plot([], [], 'o', color=style['muted'], mec='white', label=t['best_marker'])
    a1.set_ylim(*cfg['ylim'])
    a1.set_xlabel(t['xlabel'])
    a1.set_ylabel(t['ylabel'])
    a1.set_title(t['title_curves'], loc='left')
    a1.legend(ncol=1, loc='upper center')
    a2.set_xticks(
        range(len(cfg['cells'])), [c['label'].replace(', ', '\n') for c in cfg['cells']]
    )
    a2.set_xlim(-0.6, len(cfg['cells']) - 0.4)
    a2.set_title(t['title_best'], loc='left')
    a2.grid(axis='x', visible=False)
    fig.tight_layout(w_pad=2.5)
    return save(fig, out)


def plot_s2_loss(cfg, style, data, out):
    t = cfg['texts']
    rows, recipes = cfg['rows'], cfg['recipes']
    fig, axes = plt.subplots(
        len(rows), len(recipes), figsize=cfg['size'], squeeze=False
    )
    for r, row in enumerate(rows):
        for k, rec in enumerate(recipes):
            ax = axes[r][k]
            run = find_cell(data['s2_stages'], f'{row["prefix"]}_{rec["suffix"]}')
            df = metrics(run)
            col = 'train/loss_step'
            if df is None or col not in df or df[col].dropna().empty:
                ax.text(
                    0.5,
                    0.5,
                    t['queued'] if run is None else t['no_data'],
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                    color=style['muted'],
                    fontsize=10,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            else:
                d = df[['step', col]].dropna()
                spe = df.loc[df['epoch'] == 0, 'step'].max() + 1
                ax.plot(
                    d['step'] / spe, d[col], color=style['colors'][row['key']], lw=1.2
                )
                if 'lr-AdamW' in df:
                    lr = df[['step', 'lr-AdamW']].dropna()
                    ax.axvline(
                        lr.loc[lr['lr-AdamW'].idxmax(), 'step'] / spe,
                        color=style['muted'],
                        lw=0.8,
                        ls=':',
                    )
                ax.set_xlim(0, 10)
                ax.tick_params(labelsize=9)
            if r == 0:
                ax.set_title(rec['label'], fontsize=11)
            if k == 0:
                ax.set_ylabel(f'{row["label"]}\n{t["ylabel"]}', fontsize=10)
            if r == len(rows) - 1 and ax.get_xticks().size:
                ax.set_xlabel(t['xlabel'], fontsize=10)
    axes[0][0].text(
        0.12,
        0.97,
        t['lr_peak'],
        transform=axes[0][0].transAxes,
        fontsize=8.5,
        color=style['muted'],
        va='top',
    )
    fig.tight_layout(h_pad=0.6, w_pad=1.0)
    return save(fig, out)


def plot_s2_collapse(cfg, style, data, out, rows_cfg):
    t = cfg['texts']
    keyed = {r['key']: r for r in rows_cfg}
    fig, axes = plt.subplots(
        len(cfg['metrics']),
        len(cfg['recipes']),
        figsize=cfg['size'],
        sharex=True,
        sharey='row',
        squeeze=False,
    )
    labels = {r['suffix']: r['label'] for r in data['_recipes']}
    drawn = set()
    for k, rec in enumerate(cfg['recipes']):
        for key in cfg['rows']:
            df = metrics(find_cell(data['s2_stages'], f'{keyed[key]["prefix"]}_{rec}'))
            if df is None:
                continue
            ep = per_epoch(df)
            for m, met in enumerate(cfg['metrics']):
                if met['column'] in ep:
                    s = ep[met['column']].dropna()
                    axes[m][k].plot(
                        s.index + 1,
                        s.values,
                        color=style['colors'][key],
                        marker='o',
                        ms=4,
                        lw=1.6,
                        label=keyed[key]['label'] if key not in drawn else None,
                    )
                    drawn.add(key)
        axes[0][k].set_title(labels.get(rec, rec), fontsize=11)
        axes[-1][k].set_xlabel(t['xlabel'])
    for m, met in enumerate(cfg['metrics']):
        axes[m][0].set_ylabel(met['label'], fontsize=10)
    for ax in axes.flat:
        ax.tick_params(labelsize=9)
    fig.legend(loc='lower center', ncol=len(cfg['rows']), bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, out)


def plot_s2_probes(cfg, style, data, out, rows_cfg):
    t = cfg['texts']
    keyed = {r['key']: r for r in rows_cfg}
    labels = {r['suffix']: r['label'] for r in data['_recipes']}

    def best(cell):
        run = find_cell(data['s2_stages'], cell)
        df = metrics(run)
        if df is None or cfg['metric'] not in df or df[cfg['metric']].dropna().empty:
            return None
        ep = per_epoch(df)[cfg['metric']].dropna()
        return ep.min(), run_status(run), int(ep.index.max()) + 1, max_epochs(run)

    points, ticks, groups = [], [], []
    x = 0
    for key in cfg['rows']:
        x0 = x
        for rec in cfg['recipes']:
            res = best(f'{cfg["probe_prefix"]}{keyed[key]["prefix"]}_{rec}')
            ticks.append((x, labels.get(rec, rec).replace(', ', '\n')))
            if res is not None:
                points.append((x, key, *res))
            x += 1
        groups.append(
            ((x0 + x - 1) / 2, keyed[key].get('short', keyed[key]['label']), key)
        )
        x += 0.6
    floor = best(cfg['floor_cell'])
    if not points and floor is None:
        return None
    fig, ax = plt.subplots(figsize=cfg['size'])
    top = cfg['ylim'][1] if cfg.get('ylim') else None
    if top is not None:
        ax.set_ylim(*cfg['ylim'])
        # Points above the axis: a marker on the top edge with their value
        for i, (xi, key, val, *rest) in enumerate(points):
            if val > top:
                ax.plot(xi, top, '^', ms=10, color=style['colors'][key], clip_on=False)
                ax.annotate(
                    t['off_scale'].format(value=val),
                    (xi, top),
                    xytext=(0, -16),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    color=style['ink'],
                )
        points = [p for p in points if p[2] <= top]
    trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for gx, name, key in groups:
        ax.text(
            gx,
            -0.3,
            name,
            transform=trans,
            ha='center',
            va='top',
            fontsize=11,
            fontweight='bold',
            color=style['colors'].get(key, style['ink']),
        )
    if floor is not None:
        ax.axhline(floor[0], color=style['colors']['random'], lw=1.6)
        ax.text(
            x - 0.5,
            floor[0],
            f'  {t["floor"]}',
            va='bottom',
            ha='right',
            fontsize=10,
            color=style['muted'],
        )
    for xi, key, val, status, ep, n_ep in points:
        done = status == 'completed'
        color = style['colors'][key]
        ax.plot(xi, val, 'o', ms=10, color=color, mfc=color if done else 'white', mew=2)
        if not done:
            ax.annotate(
                t['running'].format(epoch=ep, epochs=n_ep or '?'),
                (xi, val),
                xytext=(0, 9),
                textcoords='offset points',
                ha='center',
                fontsize=8.5,
                color=style['muted'],
            )
    ax.set_xticks([p for p, _ in ticks], [s for _, s in ticks], fontsize=9.5)
    ax.set_xlim(-0.6, x - 0.4)
    ax.set_ylabel(t['ylabel'])
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    return save(fig, out)


def plot_full_runs(cfg, style, data, out):
    """Validation loss and the largest gradient norm of the long pretrainings."""
    t = cfg['texts']
    fig, axes = plt.subplots(1, 2, figsize=cfg['size'])
    drawn = 0
    for r in cfg['runs']:
        df = metrics(Path(r['stage']) / r['cell'] / f'seed_{r["seed"]}')
        if df is None or cfg['val_metric'] not in df:
            continue
        kw = {'color': style['colors'][r['key']], 'ls': r.get('ls', '-')}
        val = per_epoch(df)[cfg['val_metric']].dropna()
        axes[0].plot(
            val.index + 1,
            val.values,
            marker='o',
            ms=3.5,
            lw=1.6,
            label=r['label'],
            **kw,
        )
        g = df[['step', 'train/grad_norm_max']].dropna()
        spe = df.loc[df['epoch'] == 0, 'step'].max() + 1
        axes[1].plot(g['step'] / spe, g['train/grad_norm_max'], lw=1.0, **kw)
        drawn += 1
    if not drawn:
        return None
    axes[0].set_ylabel(t['ylabel_val'])
    axes[1].set_yscale('log')
    axes[1].set_ylabel(t['ylabel_grad'])
    axes[1].axhline(cfg['spike_level'], color=style['muted'], lw=0.8, ls=':')
    axes[1].text(
        0.99,
        cfg['spike_level'] * 0.85,
        t['spike_level'],
        transform=matplotlib.transforms.blended_transform_factory(
            axes[1].transAxes, axes[1].transData
        ),
        ha='right',
        va='top',
        fontsize=8.5,
        color=style['muted'],
    )
    for ax in axes:
        ax.set_xlabel(t['xlabel'])
        ax.tick_params(labelsize=9)
    axes[0].legend(fontsize=9)
    fig.tight_layout(w_pad=2.0)
    return save(fig, out)


def plot_attn_logits(cfg, style, data, out):
    """Largest logit and entropy of the first row-attention blocks, per checkpoint."""
    path = Path(data[cfg['table']])
    if not path.exists():
        return None
    df = pd.read_csv(path)
    t = cfg['texts']
    fig, axes = plt.subplots(2, 1, figsize=cfg['size'], sharex=True)
    x, ticks, groups = 0, [], []
    for g in cfg['groups']:
        x0 = x
        for e in g['entries']:
            rows = df[df['label'] == e['label']]
            color = style['colors'][e.get('key', g['key'])]
            for lay in cfg['layers']:
                row = rows[rows['layer'] == lay['layer']]
                if row.empty:
                    continue
                for ax, col in zip(axes, ('max_logit', 'entropy')):
                    ax.plot(
                        x,
                        row[col].iloc[0],
                        lay['marker'],
                        ms=7,
                        mew=1.6,
                        color=color,
                        mfc=color if lay.get('filled', True) else 'white',
                    )
            ticks.append((x, e['tick']))
            x += 1
        groups.append(((x0 + x - 1) / 2, g['label'], g['key']))
        x += 0.7
    axes[0].set_yscale('log')
    axes[0].set_ylabel(t['ylabel_logit'], fontsize=10)
    axes[1].set_ylabel(t['ylabel_entropy'], fontsize=10)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xticks([p for p, _ in ticks], [s for _, s in ticks], fontsize=8.5)
    axes[1].set_xlim(-0.6, x - 0.5)
    trans = matplotlib.transforms.blended_transform_factory(
        axes[1].transData, axes[1].transAxes
    )
    for gx, name, key in groups:
        axes[1].text(
            gx,
            -0.2,
            name,
            transform=trans,
            ha='center',
            va='top',
            fontsize=9,
            fontweight='bold',
            color=style['colors'].get(key, style['ink']),
        )
    handles = [
        matplotlib.lines.Line2D(
            [],
            [],
            ls='',
            marker=lay['marker'],
            ms=7,
            mew=1.6,
            color=style['muted'],
            mfc=style['muted'] if lay.get('filled', True) else 'white',
            label=lay['label'],
        )
        for lay in cfg['layers']
    ]
    fig.legend(
        handles=handles,
        fontsize=9,
        loc='upper center',
        ncol=len(handles),
        frameon=False,
    )
    for ax in axes:
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(axis='x', visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, out)


# ── pptx helpers ─────────────────────────────────────────────────────────────
def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color.lstrip('#').upper())


class Deck:
    def __init__(self, cfg: dict, date: str):
        self.cfg = cfg
        self.s = cfg['style']
        self.date = date
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(SLIDE_W), Inches(SLIDE_H)
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    # text
    def text(
        self,
        slide,
        box,
        paragraphs,
        *,
        size=16,
        color=None,
        bold=False,
        align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.TOP,
        bullets=False,
        space_after=8,
        name=None,
    ):
        x, y, w, h = box
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        if name:
            tb.name = name
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(0.02)
        tf.margin_top = tf.margin_bottom = Inches(0.02)
        for i, para in enumerate(paragraphs):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            p.space_after = Pt(space_after)
            runs = para if isinstance(para, list) else [(para, {})]
            for txt, opts in runs:
                r = p.add_run()
                r.text = txt
                f = r.font
                f.name = self.s['deck_font']
                f.size = Pt(opts.get('size', size))
                f.bold = opts.get('bold', bold)
                f.color.rgb = rgb(opts.get('color', color or self.s['ink']))
            if bullets:
                self._bullet(p)
        return tb

    def _bullet(self, p):
        pPr = p._p.get_or_add_pPr()
        pPr.set('marL', str(Emu(Inches(0.28))))
        pPr.set('indent', str(-Emu(Inches(0.28))))
        bu_clr = pPr.makeelement(qn('a:buClr'), {})
        clr = bu_clr.makeelement(
            qn('a:srgbClr'), {'val': self.s['accent'].lstrip('#').upper()}
        )
        bu_clr.append(clr)
        pPr.append(bu_clr)
        pPr.append(pPr.makeelement(qn('a:buFont'), {'typeface': 'Arial'}))
        pPr.append(pPr.makeelement(qn('a:buChar'), {'char': '•'}))

    def rect(self, slide, box, fill, *, rounded=True, line=None, name=None):
        x, y, w, h = box
        shp = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
            Inches(x),
            Inches(y),
            Inches(w),
            Inches(h),
        )
        if rounded:
            shp.adjustments[0] = min(0.5, 0.1 / min(w, h))  # ~0.1" corner radius
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(fill)
        if line:
            shp.line.color.rgb = rgb(line)
            shp.line.width = Pt(1)
        else:
            shp.line.fill.background()
        shp.shadow.inherit = False
        if name:
            shp.name = name
        return shp

    def image(self, slide, path, box):
        from PIL import Image

        x, y, w, h = box
        with Image.open(path) as im:
            iw, ih = im.size
        scale = min(w / iw, h / ih)
        pw, ph = iw * scale, ih * scale
        slide.shapes.add_picture(
            str(path),
            Inches(x + (w - pw) / 2),
            Inches(y + (h - ph) / 2),
            Inches(pw),
            Inches(ph),
        )

    # frame
    def new(self, dark=False):
        slide = self.prs.slides.add_slide(self.blank)
        bg = slide.background.fill
        bg.solid()
        bg.fore_color.rgb = rgb(self.s['dark_bg'] if dark else '#ffffff')
        self.n += 1
        return slide

    def title(self, slide, text):
        self.text(
            slide,
            (MARGIN, TITLE_Y, SLIDE_W - 2 * MARGIN, TITLE_H),
            [text],
            size=30,
            bold=True,
            anchor=MSO_ANCHOR.MIDDLE,
            space_after=0,
            name='Title',
        )

    def footer(self, slide):
        y = SLIDE_H - 0.42
        self.text(
            slide,
            (MARGIN, y, 9, 0.3),
            [f'{self.cfg["footer"]} · {self.cfg["updated_label"]} {self.date}'],
            size=10,
            color=self.s['muted'],
            space_after=0,
        )
        self.text(
            slide,
            (SLIDE_W - MARGIN - 1, y, 1, 0.3),
            [str(self.n)],
            size=10,
            color=self.s['muted'],
            align=PP_ALIGN.RIGHT,
            space_after=0,
        )

    def card(self, slide, box, header, body, size=14):
        self.rect(slide, box, self.s['card_tint'], name='Decision card')
        x, y, w, h = box
        self.text(
            slide,
            (x + 0.2, y + 0.12, w - 0.4, h - 0.24),
            [
                [
                    (header + '  ', {'bold': True, 'color': self.s['accent_ink']}),
                    (body, {}),
                ]
            ],
            size=size,
            anchor=MSO_ANCHOR.MIDDLE,
            space_after=0,
        )

    def status_cell(self, cell, value):
        key = value.split(':')[0].strip().lower()
        fill, ink = self.s['status'].get(key, ['#ffffff', self.s['ink']])
        return fill, ink

    def table(self, slide, box, spec, size=13):
        x, y, w, _ = box
        header, rows, widths = spec['header'], spec['rows'], spec['widths']
        scale = w / sum(widths)
        cols = [wd * scale for wd in widths]

        def height(row):  # rows grow with their text: ~144/size characters per inch
            lines = max(
                -(-len(str(v)) // max(1, int((cw - 0.16) * 144 / size)))
                for v, cw in zip(row, cols)
            )
            return max(0.42, 0.14 + lines * size * 1.25 / 72)

        heights = [height(r) for r in [header] + rows]
        gt = slide.shapes.add_table(
            len(rows) + 1,
            len(header),
            Inches(x),
            Inches(y),
            Inches(w),
            Inches(sum(heights)),
        ).table
        for j, cw in enumerate(cols):
            gt.columns[j].width = Inches(cw)
        for i, rh in enumerate(heights):
            gt.rows[i].height = Inches(rh)
        status_col = spec.get('status_col')
        for i, row in enumerate([header] + rows):
            for j, value in enumerate(row):
                cell = gt.cell(i, j)
                fill, ink, bold = '#ffffff', self.s['ink'], False
                if i == 0:
                    fill, ink, bold = self.s['header_fill'], '#ffffff', True
                elif j == status_col:
                    fill, ink = self.status_cell(cell, value)
                elif i % 2 == 0:
                    fill = '#f7f7f5'
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb(fill)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.margin_left = cell.margin_right = Inches(0.08)
                cell.margin_top = cell.margin_bottom = Inches(0.04)
                p = cell.text_frame.paragraphs[0]
                cell.text_frame.word_wrap = True
                r = p.add_run()
                r.text = str(value)
                r.font.name = self.s['deck_font']
                r.font.size = Pt(size)
                r.font.bold = bold or (j == 0 and i > 0)
                r.font.color.rgb = rgb(ink)
        return sum(heights)


# ── Slide layouts ────────────────────────────────────────────────────────────
def build_title(deck, s, _plots):
    slide = deck.new(dark=True)
    st = deck.s
    deck.text(
        slide,
        (0.9, 2.2, 11.5, 1.8),
        [s['title']],
        size=40,
        bold=True,
        color=st['dark_ink'],
        anchor=MSO_ANCHOR.BOTTOM,
        space_after=0,
    )
    deck.text(
        slide,
        (0.9, 4.15, 11.0, 1.0),
        [s['subtitle']],
        size=20,
        color=st['dark_muted'],
        space_after=0,
    )
    deck.text(
        slide,
        (0.9, 5.4, 11.0, 0.5),
        [f'{deck.cfg["updated_label"]} {deck.date}'],
        size=14,
        color=st['accent'],
        bold=True,
        space_after=0,
    )
    deck.text(
        slide,
        (0.9, 6.7, 11.5, 0.35),
        [deck.cfg['footer']],
        size=10,
        color=st['dark_muted'],
        space_after=0,
    )


def build_text_plot(deck, s, plots):
    slide = deck.new()
    deck.title(slide, s['title'])
    tw = s.get('text_width', 4.6)
    gap = 0.35
    pw = SLIDE_W - 2 * MARGIN - tw - gap
    left = s.get('plot_left', False)
    tx = MARGIN + pw + gap if left else MARGIN
    px = MARGIN if left else MARGIN + tw + gap
    label, body = card_text(deck, s)
    card_h = card_height(label, body, tw, 13) if body else 0
    body_h = BODY_BOTTOM - BODY_Y - card_h - (0.25 if card_h else 0)
    deck.text(
        slide,
        (tx, BODY_Y + 0.1, tw, body_h),
        s['bullets'],
        size=16,
        bullets=True,
        space_after=10,
    )
    if body:
        deck.card(slide, (tx, BODY_BOTTOM - card_h, tw, card_h), label, body, size=13)
    place_plot(deck, slide, plots, s, (px, BODY_Y, pw, BODY_BOTTOM - BODY_Y))
    deck.footer(slide)


def build_plot_full(deck, s, plots):
    slide = deck.new()
    deck.title(slide, s['title'])
    label, body = card_text(deck, s)
    w = SLIDE_W - 2 * MARGIN
    card_h = card_height(label, body, w, 14) if body else 0
    place_plot(
        deck, slide, plots, s, (MARGIN, BODY_Y, w, BODY_BOTTOM - BODY_Y - card_h - 0.2)
    )
    if body:
        deck.card(
            slide, (MARGIN, BODY_BOTTOM - card_h, w, card_h), label, body, size=14
        )
    deck.footer(slide)


def card_text(deck, s):
    if s.get('decision'):
        return deck.cfg['decision_label'], s['decision']
    if s.get('result'):
        return deck.cfg['result_label'], s['result']
    return None, None


def card_height(label, body, width, size):
    per_line = max(1, int((width - 0.45) * 144 / size))  # ~144/size characters per inch
    lines = -(-(len(label) + 2 + len(body)) // per_line)
    return 0.4 + lines * size * 1.3 / 72


def place_plot(deck, slide, plots, s, box):
    path = plots.get(s['plot'])
    if path is not None:
        deck.image(slide, path, box)
        return
    x, y, w, h = box
    deck.rect(slide, box, '#f4f4f2', name='Placeholder')
    deck.text(
        slide,
        (x + 0.6, y + 0.4, w - 1.2, h - 0.8),
        [deck.cfg['plots'][s['plot']].get('placeholder', '')],
        size=16,
        color=deck.s['muted'],
        anchor=MSO_ANCHOR.MIDDLE,
        align=PP_ALIGN.CENTER,
    )


def build_tables(deck, s, _plots):
    slide = deck.new()
    deck.title(slide, s['title'])
    y = BODY_Y + 0.05
    w = SLIDE_W - 2 * MARGIN
    for spec in s['tables']:
        if spec.get('caption'):
            deck.text(
                slide,
                (MARGIN, y, w, 0.35),
                [spec['caption']],
                size=14,
                bold=True,
                space_after=0,
            )
            y += 0.42
        y += deck.table(slide, (MARGIN, y, w, 0), spec) + 0.35
    if s.get('bullets'):
        if s.get('bullets_title'):
            deck.text(
                slide,
                (MARGIN, y, w, 0.35),
                [s['bullets_title']],
                size=14,
                bold=True,
                space_after=0,
            )
            y += 0.45
        deck.text(
            slide, (MARGIN, y, w, BODY_BOTTOM - y), s['bullets'], size=15, bullets=True
        )
        y += 0.5 * len(s['bullets']) + 0.2
    if s.get('note'):
        deck.text(
            slide,
            (MARGIN, y, w, BODY_BOTTOM - y),
            [s['note']],
            size=12,
            color=deck.s['muted'],
        )
    deck.footer(slide)


def build_stages(deck, s, _plots):
    slide = deck.new()
    deck.title(slide, s['title'])
    stages = s['stages']
    n = len(stages)
    gap = 0.35
    w = (SLIDE_W - 2 * MARGIN - gap * (n - 1)) / n
    y, h = BODY_Y + 0.3, 3.9
    for i, st in enumerate(stages):
        x = MARGIN + i * (w + gap)
        fill, ink = deck.s['status'][st['status']]
        deck.rect(slide, (x, y, w, h), '#f7f7f5', name=f'Stage {st["name"]}')
        deck.text(
            slide,
            (x + 0.18, y + 0.15, w - 0.36, 0.55),
            [st['name']],
            size=24,
            bold=True,
            space_after=0,
        )
        chip = deck.rect(slide, (x + 0.18, y + 0.78, 1.3, 0.34), fill, name='Status')
        chip.text_frame.text = st['status']
        para = chip.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.CENTER
        para.runs[0].font.size = Pt(11)
        para.runs[0].font.bold = True
        para.runs[0].font.name = deck.s['deck_font']
        para.runs[0].font.color.rgb = rgb(ink)
        deck.text(
            slide,
            (x + 0.18, y + 1.3, w - 0.36, 1.6),
            [st['what']],
            size=13,
            space_after=0,
        )
        deck.text(
            slide,
            (x + 0.18, y + 2.85, w - 0.36, 1.0),
            [
                [
                    (
                        deck.cfg['decides_label'] + ': ',
                        {'bold': True, 'color': deck.s['accent_ink']},
                    ),
                    (st['decides'], {}),
                ]
            ],
            size=13,
            space_after=0,
        )
        if i < n - 1:
            arr = slide.shapes.add_shape(
                MSO_SHAPE.CHEVRON,
                Inches(x + w + 0.08),
                Inches(y + h / 2 - 0.15),
                Inches(gap - 0.16),
                Inches(0.3),
            )
            arr.fill.solid()
            arr.fill.fore_color.rgb = rgb(deck.s['muted'])
            arr.line.fill.background()
    if s.get('note'):
        deck.text(
            slide,
            (MARGIN, y + h + 0.45, SLIDE_W - 2 * MARGIN, 1.0),
            [s['note']],
            size=13,
            color=deck.s['muted'],
        )
    deck.footer(slide)


def build_cards(deck, s, _plots):
    slide = deck.new()
    deck.title(slide, s['title'])
    cards = s['cards']
    cols = 2
    gap = 0.4
    w = (SLIDE_W - 2 * MARGIN - gap) / cols
    rows = (len(cards) + cols - 1) // cols
    h = (BODY_BOTTOM - BODY_Y - gap * (rows - 1)) / rows
    for i, c in enumerate(cards):
        x = MARGIN + (i % cols) * (w + gap)
        y = BODY_Y + (i // cols) * (h + gap)
        deck.rect(slide, (x, y, w, h), '#f7f7f5', name=c['header'])
        circle = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, Inches(x + 0.3), Inches(y + 0.3), Inches(0.5), Inches(0.5)
        )
        circle.fill.solid()
        circle.fill.fore_color.rgb = rgb(deck.s['accent'])
        circle.line.fill.background()
        circle.text_frame.text = str(i + 1)
        para = circle.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.CENTER
        para.runs[0].font.size = Pt(16)
        para.runs[0].font.bold = True
        para.runs[0].font.name = deck.s['deck_font']
        para.runs[0].font.color.rgb = rgb('#ffffff')
        deck.text(
            slide,
            (x + 1.0, y + 0.3, w - 1.3, 0.5),
            [c['header']],
            size=20,
            bold=True,
            anchor=MSO_ANCHOR.MIDDLE,
            space_after=0,
        )
        deck.text(
            slide,
            (x + 1.0, y + 0.95, w - 1.3, h - 1.15),
            [c['text']],
            size=15,
            space_after=0,
        )
    deck.footer(slide)


LAYOUTS = {
    'title': build_title,
    'text_plot': build_text_plot,
    'plot_full': build_plot_full,
    'tables': build_tables,
    'stages': build_stages,
    'cards': build_cards,
}


def make_plots(cfg: dict, plot_dir: Path) -> dict[str, Path | None]:
    style, data, pc = cfg['style'], dict(cfg['data']), cfg['plots']
    data['_recipes'] = pc['s2_loss']['recipes']
    rows = pc['s2_loss']['rows']
    makers = {
        'model': lambda c, o: plot_model(c, style, data, o),
        'efficiency': lambda c, o: plot_efficiency(c, style, data, o),
        's0_speed': lambda c, o: plot_s0_speed(c, style, data, o),
        's0_memory': lambda c, o: plot_s0_memory(c, style, data, o),
        's1': lambda c, o: plot_s1(c, style, data, o),
        's2_loss': lambda c, o: plot_s2_loss(c, style, data, o),
        's2_collapse': lambda c, o: plot_s2_collapse(c, style, data, o, rows),
        's2_probes': lambda c, o: plot_s2_probes(c, style, data, o, rows),
        'full_runs': lambda c, o: plot_full_runs(c, style, data, o),
        'attn_logits': lambda c, o: plot_attn_logits(c, style, data, o),
    }
    out = {}
    for name, plot_cfg in pc.items():
        if 'file' in plot_cfg:  # a figure made by another script, used as it is
            path = REPO / plot_cfg['file']
            out[name] = path if path.is_file() else None
        else:
            out[name] = makers[name](plot_cfg, plot_dir / f'{name}.png')
        print(f'  plot {name}: {out[name] or "no data yet (placeholder)"}')
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument(
        '--config', default=str(REPO / 'experiments/2026-10_ssl-campaign/configs/slides/ssl_campaign.yaml')
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = REPO / cfg['output']['dir']
    plot_dir = out_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    setup_style(cfg['style'])
    plots = make_plots(cfg, plot_dir)
    deck = Deck(cfg, dt.date.today().isoformat())
    for s in cfg['slides']:
        LAYOUTS[s['layout']](deck, s, plots)
    pptx = out_dir / cfg['output']['pptx']
    deck.prs.save(pptx)
    print(f'{deck.n} slides -> {pptx}')


if __name__ == '__main__':
    main()
