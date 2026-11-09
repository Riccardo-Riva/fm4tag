#!/usr/bin/env python
"""Every plot of the SSL report in one pptx: one or two plots per slide, with the settings of the models they show.

    .venv/bin/python scripts/report_deck.py --config scripts/configs/slides/report.yaml
    (on a CPU node: sbatch slurm/plot_job.sh scripts/report_deck.py:scripts/configs/slides/report.yaml)

The slides come from ``slides`` in the YAML: a title, section dividers, tables of the models' settings, and plot
slides of one or two plots.  A plot slide lists the models its plots show (``models``: keys of ``models``, each a
list of run directories); its info box is built from those runs alone:

  * their resolved .hydra config: architecture, loss weights, optimiser, batch, epochs, data paths;
  * the parameter counts Lightning printed at the start of training (``rank_0.err``);
  * the number of jets in the data files they read (and ``train_max_rows``).

A setting shared by every model on the slide is written once; a setting that differs between the runs of one model
lists one value per configuration (ordered by the model's ``order_by``).  The layout (the info box beside or under
the plots, two plots side by side or stacked) is the one that shows the plots largest; text is measured with the
font, so it fits.

Also writes ``<output dir>/<preview>/slide_NN.png``, a drawing of every slide for checking the layout without
PowerPoint, and prints every model's info.  Every word on the slides comes from the YAML.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import os
import re
import sys
from pathlib import Path

import h5py
import matplotlib
import yaml
from omegaconf import OmegaConf
from PIL import Image, ImageFont

matplotlib.use('Agg')  # PNG previews only: a dead X11 forward would hang matplotlib
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import plot_arm_epochs as pae  # noqa: E402 -- .hydra config reader, method of a run
import ssl_campaign_slides as scs  # noqa: E402 -- the deck class and its geometry

REPO = scs.REPO
SLIDE_W, SLIDE_H, MARGIN = scs.SLIDE_W, scs.SLIDE_H, scs.MARGIN
BODY_W = SLIDE_W - 2 * MARGIN
BODY_BOTTOM = 6.95
TITLE_Y, TITLE_H, SUB_Y = 0.3, 0.62, 0.95
GAP = 0.2  # between plots, and between the plots and the info box
PAD = 0.12  # inside the info box
TEXT_MARGIN = 0.02  # python-pptx text frame margin set by scs.Deck.text
LINE = 1.22  # line height / font size: PowerPoint's single spacing, with a margin
CELL_PAD_X, CELL_PAD_Y = 0.08, 0.04


# ── Text measurement ─────────────────────────────────────────────────────────
class Measure:
    """Width of text in inches with the first available font of ``families``.

    Liberation Sans has Arial's metrics; DejaVu Sans is wider, so a layout measured with it also fits in Arial.
    """

    def __init__(self, families: list[str]):
        self.paths = {}
        for bold in (False, True):
            for family in families:
                prop = font_manager.FontProperties(
                    family=family, weight='bold' if bold else 'normal'
                )
                try:
                    self.paths[bold] = font_manager.findfont(
                        prop, fallback_to_default=False
                    )
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError(f'none of the fonts {families} is installed')
        self.fonts = {b: ImageFont.truetype(p, 100) for b, p in self.paths.items()}
        print(f'measuring text with {self.paths}')

    def width(self, text: str, size: float, bold: bool = False) -> float:
        return self.fonts[bool(bold)].getlength(text) / 100 * size / 72


def runs_of(paragraph) -> list[tuple[str, dict]]:
    return paragraph if isinstance(paragraph, list) else [(paragraph, {})]


def wrap(paragraph, width: float, size: float, bold: bool, measure: Measure) -> list:
    """Greedy word wrap of a paragraph (a string or (text, opts) runs) into lines; a line is a list of words, a
    word a list of (text, opts) pieces (runs can meet inside a word, e.g. a bold label and its colon)."""
    tokens, space = [], False
    for text, opts in runs_of(paragraph):
        for i, part in enumerate(text.split(' ')):
            if i > 0:
                space = True
            if part:
                tokens.append((part, opts, space and bool(tokens)))
                space = False
    words = []
    for part, opts, spaced in tokens:
        if spaced or not words:
            words.append([(part, opts)])
        else:
            words[-1].append((part, opts))
    space_w = measure.width(' ', size)
    lines, line, x = [], [], 0.0
    for word in words:
        w = sum(
            measure.width(p, o.get('size', size), o.get('bold', bold)) for p, o in word
        )
        add = w + (space_w if line else 0.0)
        if line and x + add > width:
            lines.append(line)
            line, x, add = [], 0.0, w
        line.append(word)
        x += add
    lines.append(line)
    return lines


# ── Deck with measured text and a record of every shape (for the previews) ──
class ReportDeck(scs.Deck):
    def __init__(self, cfg: dict, date: str, measure: Measure):
        super().__init__(cfg, date)
        self.measure = measure
        self.pages: list[dict] = []

    def new(self, dark=False):
        slide = super().new(dark)
        self.pages.append({'dark': dark, 'ops': []})
        return slide

    def text(self, slide, box, paragraphs, **kw):
        self.pages[-1]['ops'].append(('text', box, paragraphs, kw))
        return super().text(slide, box, paragraphs, **kw)

    def rect(self, slide, box, fill, **kw):
        self.pages[-1]['ops'].append(('rect', box, fill))
        return super().rect(slide, box, fill, **kw)

    def image(self, slide, path, box):
        self.pages[-1]['ops'].append(('image', box, path))
        return super().image(slide, path, box)

    def text_height(
        self, paragraphs, width, size, bold=False, space_after=0.0
    ) -> float:
        """Height of a text box of ``width`` holding ``paragraphs`` (as scs.Deck.text draws them)."""
        h = 2 * TEXT_MARGIN
        for i, p in enumerate(paragraphs):
            n = len(wrap(p, width - 2 * TEXT_MARGIN, size, bold, self.measure))
            h += n * size * LINE / 72
            if i < len(paragraphs) - 1:
                h += space_after / 72
        return h

    def title(self, slide, text):
        """The slide title, at the style's size or smaller until it fits on one line."""
        size = float(self.s['title_size'])
        while self.measure.width(text, size, True) > BODY_W - 0.1 and size > 16:
            size -= 1
        self.text(
            slide,
            (MARGIN, TITLE_Y, BODY_W, TITLE_H),
            [text],
            size=size,
            bold=True,
            anchor=MSO_ANCHOR.MIDDLE,
            space_after=0,
            name='Title',
        )

    def table_layout(self, header, rows, w, size) -> tuple[list, list, list]:
        """Column widths, merged groups per row and row heights of a spec table of width ``w``."""
        label_w = min(float(self.s['table_label_width']), w / 3)
        n = len(header) - 1
        cols = [label_w] + [(w - label_w) / n] * n

        def cell_h(text, width, bold):
            lines = wrap(str(text), width - 2 * CELL_PAD_X, size, bold, self.measure)
            return len(lines) * size * LINE / 72 + 2 * CELL_PAD_Y + 0.04

        spans = []  # per row: [(first col, last col)] of the groups of equal cells after the label
        for row in rows:
            groups, j = [], 1
            while j < len(row):
                k = j
                while k + 1 < len(row) and row[k + 1] == row[j]:
                    k += 1
                groups.append((j, k))
                j = k + 1
            spans.append(groups)
        heights = [max(cell_h(t, cw, True) for t, cw in zip(header, cols))]
        for row, groups in zip(rows, spans):
            hs = [cell_h(row[0], cols[0], True)]
            hs += [cell_h(row[a], sum(cols[a : b + 1]), False) for a, b in groups]
            heights.append(max(hs))
        return cols, spans, heights

    def spec_table(self, slide, box, header, rows, size, header_fills=None) -> float:
        """A table whose rows grow with their text; equal neighbouring cells after the label column are merged."""
        x, y, w, _ = box
        cols, spans, heights = self.table_layout(header, rows, w, size)
        table = slide.shapes.add_table(
            len(rows) + 1,
            len(header),
            Inches(x),
            Inches(y),
            Inches(w),
            Inches(sum(heights)),
        ).table
        for j, cw in enumerate(cols):
            table.columns[j].width = Inches(cw)
        for i, rh in enumerate(heights):
            table.rows[i].height = Inches(rh)
        cells = []  # for the preview: (box, text, fill, ink, bold)
        grid = [[(t, 0, j, j) for j, t in enumerate(header)]]
        grid += [
            [(row[0], i + 1, 0, 0)] + [(row[a], i + 1, a, b) for a, b in groups]
            for i, (row, groups) in enumerate(zip(rows, spans))
        ]
        for line in grid:
            for text, i, a, b in line:
                if b > a:
                    table.cell(i, a).merge(table.cell(i, b))
                cell = table.cell(i, a)
                fill, ink, bold = '#ffffff', self.s['ink'], a == 0 and i > 0
                if i == 0:
                    fill, ink, bold = self.s['header_fill'], '#ffffff', True
                    if header_fills and a > 0:  # a column of a model: its colour
                        fill = header_fills[a - 1]
                elif i % 2 == 0:
                    fill = '#f7f7f5'
                cell.fill.solid()
                cell.fill.fore_color.rgb = scs.rgb(fill)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.margin_left = cell.margin_right = Inches(CELL_PAD_X)
                cell.margin_top = cell.margin_bottom = Inches(CELL_PAD_Y)
                cell.text_frame.word_wrap = True
                run = cell.text_frame.paragraphs[0].add_run()
                run.text = str(text)
                run.font.name = self.s['deck_font']
                run.font.size = Pt(size)
                run.font.bold = bold
                run.font.color.rgb = scs.rgb(ink)
                cx = x + sum(cols[:a])
                cy = y + sum(heights[:i])
                cells.append(
                    (
                        (cx, cy, sum(cols[a : b + 1]), heights[i]),
                        str(text),
                        fill,
                        ink,
                        bold,
                    )
                )
        self.pages[-1]['ops'].append(('table', cells, size))
        return sum(heights)


# ── Facts about runs ─────────────────────────────────────────────────────────
def h5_rows(path: str, obj: str, cache: dict) -> int:
    if path not in cache:
        with h5py.File(path, 'r') as f:
            cache[path] = int(f[obj].shape[0])
    return cache[path]


def parse_params(err: Path) -> dict:
    """Lightning's parameter summary at the start of training (its rounding): total and trainable."""
    out = {}
    for line in err.read_text(errors='replace').splitlines():
        m = re.match(r'^\s*([\d.]+)\s*([KMBT]?)\s+(Trainable|Total) params', line)
        if m:
            out[m.group(3).lower()] = f'{m.group(1)}{m.group(2)}'
    if {'trainable', 'total'} - set(out):
        raise ValueError(f'{err}: no parameter summary')
    return out


def sci(x: float) -> str:
    """1.1e-3, 8e-4, 0.05: short scientific notation below 1e-2."""
    if x == 0:
        return '0'
    e = math.floor(math.log10(abs(x)))
    if e < -2:
        return f'{x / 10**e:.3g}e{e}'
    return f'{x:.3g}'


def si(n: int) -> str:
    """18.6M, 558k."""
    if n >= 1_000_000:
        return f'{n / 1e6:.4g}M'
    if n >= 1_000:
        return f'{n / 1e3:.4g}k'
    return str(n)


def run_fields(run_dir: Path, texts: dict, cache: dict) -> dict:
    """Every value the info templates can use, for one run, from its config, log and data files."""
    cfg = pae.run_config(run_dir)

    def sel(key, default=None):
        return OmegaConf.select(cfg, key, default=default)

    trk = cfg.encoders.constituents[cfg.constituent_objects[0]]
    layer = trk.layers[0]
    clip = sel('trainer.gradient_clip_val')
    f = {
        'phase': cfg.phase,
        'seed': int(cfg.seed),
        'enc_dim': trk.dim,
        'enc_depth': layer.depth,
        'row_heads': layer.row_heads,
        'row_mode': layer.row_mode,
        'proj_out': trk.proj_out,
        'chunk': cfg.chunk_size,
        'qk_norm': texts['on_off'][bool(sel('row_qk_norm', False))],
        'agg_depth': cfg.aggregator.depth,
        'agg_heads': cfg.aggregator.heads,
        'agg_dim_head': cfg.aggregator.dim_head,
        'agg_dropout': float(cfg.aggregator.ff_dropout),
        'head_hidden': sel('head.mlp_hidden'),
        'head_dropout': float(sel('head.mlp_dropout', 0.0)),
        'batch': int(cfg.datamodule.batch_size),
        'lr': float(cfg.optimizer.lr),
        'lr_s': sci(float(cfg.optimizer.lr)),
        'wd': float(cfg.optimizer.weight_decay),
        'wd_s': sci(float(cfg.optimizer.weight_decay)),
        'max_epochs': int(cfg.trainer.max_epochs),
        'precision': cfg.trainer.precision,
        'clip_s': texts['none'] if clip is None else f'{float(clip):g}',
        'patience': int(cfg.callbacks.early_stopping.patience),
        'lam': float(cfg['lambda']),
        'views': texts['views_join'].join(
            texts['augmentations_join'].join(
                a._target_.rsplit('.', 1)[-1] for a in view.augmentations
            )
            for view in cfg.views
        ),
    }
    if cfg.phase == 'pretrain':
        w = cfg.pretrain.loss_weights
        f.update(
            w_contrastive=float(w.contrastive),
            w_jet_contrastive=float(w.jet_contrastive),
            w_denoising_cat=float(w.denoising_cat),
            w_denoising_con=float(w.denoising_con),
            t_contrastive=float(cfg.pretrain.contrastive_loss.temperature),
            t_jet_contrastive=float(cfg.pretrain.jet_contrastive_loss.temperature),
            warmup=float(cfg.pretrain.warmup_frac),
            agg_state=texts['aggregator'][
                'trained' if float(w.jet_contrastive) > 0 else 'untrained'
            ],
        )
        data_path = sel('pretrain_dataset_path')
    else:
        w = cfg.finetune.loss_weights
        meta = pae.classify(run_dir)
        backbone_lr = sel('optimizer.backbone_lr')
        f.update(
            w_cross_entropy=float(w.cross_entropy),
            w_jet_contrastive=float(w.jet_contrastive),
            t_jet_contrastive=float(cfg.finetune.jet_contrastive_loss.temperature),
            warmup=float(cfg.finetune.warmup_frac),
            unfreeze=int(cfg.finetune.unfreeze_at_epoch),
            backbone_lr_s=sci(float(backbone_lr)) if backbone_lr is not None else '',
        )
        pre_seed = (
            int(pae.run_config(Path(meta['pretraining_run'])).seed)
            if meta['pretraining_run']
            else None
        )
        f['start'] = texts['start'][meta['pretraining'] or 'none'].format(seed=pre_seed)
        f['trained'] = texts['trained'][meta['encoder']].format(**f)
        f['lr_text'] = (  # fine-tuning: the pretrained parts have their own (lower) lr
            texts['lr_pretrained'].format(lr=f['lr_s'], backbone=f['backbone_lr_s'])
            if meta['encoder'] == 'finetune'
            else f['lr_s']
        )
        data_path = sel('train_dataset_path')
    full = h5_rows(data_path, cfg.global_object, cache)
    max_rows = sel('datamodule.train_max_rows')
    f['n_train'] = min(full, int(max_rows)) if max_rows else full
    f['n_train_s'] = si(f['n_train'])
    f['n_val'] = h5_rows(sel('val_dataset_path'), cfg.global_object, cache)
    f['n_val_s'] = si(f['n_val'])
    params = parse_params(run_dir / 'rank_0.err')
    f['params_total'], f['params_trained'] = params['total'], params['trainable']
    return f


def merge(values: list[str], sep: str) -> str:
    """One text for several values of a setting: the shared start and end once, the differing middles joined.

    ``['lr 3e-4', 'lr 6e-4']`` -> ``'lr 3e-4 / 6e-4'``.  The shared start and end are cut at word boundaries, so
    a value (``1.2e-3``, ``28,672``) is never split; a punctuation mark every middle ends with (a comma) goes
    after the joined values.
    """
    if len(set(values)) == 1:
        return values[0]
    pre = os.path.commonprefix(values)
    pre = pre[: pre.rfind(' ') + 1]  # whole words only ('' without a space)
    rest = [v[len(pre) :] for v in values]
    suf = os.path.commonprefix([r[::-1] for r in rest])[::-1]
    suf = suf[suf.find(' ') :] if ' ' in suf else ''
    mids = [r[: len(r) - len(suf)] for r in rest]
    trail = ''
    while all(m and m[-1] in ',;:' for m in mids) and len({m[-1] for m in mids}) == 1:
        trail = mids[0][-1] + trail
        mids = [m[:-1] for m in mids]
    return pre + sep.join(mids) + trail + suf


class Model:
    """One model of the YAML: its runs, grouped into configurations, and the text of every info item.

    A configuration is a set of runs whose non-``per_run`` info texts are equal (one learning rate, one label
    fraction, ...); ``per_run`` items (seeds, the pretraining seed) are merged over the runs of a configuration.
    ``order_by`` (run fields) orders the configurations, ``config_labels`` names them in that order (the labels
    of the plot's x axis), and ``short`` is the model's name inside a table cell.
    """

    def __init__(self, key: str, spec: dict, info: dict, cache: dict):
        self.key, self.label, self.colour = key, spec['label'], spec['colour']
        self.short = spec.get('short', self.label)
        dirs = [Path(d) for p in spec['runs'] for d in sorted(glob.glob(p))]
        if not dirs:
            raise FileNotFoundError(f'model {key}: no run matches {spec["runs"]}')
        self.runs = [run_fields(d, info['texts'], cache) for d in dirs]
        phases = {r['phase'] for r in self.runs}
        if len(phases) != 1:
            raise ValueError(f'model {key}: runs of several phases {phases}')
        self.phase = phases.pop()
        self.items = info['items'][self.phase]
        self.configs = self._configs(spec.get('order_by') or [], info['values_sep'])
        self.config_labels = spec.get('config_labels')
        if self.config_labels and len(self.config_labels) != len(self.configs):
            raise ValueError(
                f'model {key}: {len(self.config_labels)} config_labels for {len(self.configs)} configurations'
            )
        self.values = self._values(info['values_sep'])

    def _configs(self, order_by: list[str], sep: str) -> list[dict]:
        def text(item, run):
            if item.get('only_if') and not run.get(item['only_if']):
                return None
            return item['value'].format(**run)

        texts = [{it['key']: text(it, run) for it in self.items} for run in self.runs]
        shared = [it['key'] for it in self.items if not it.get('per_run')]
        groups: dict[tuple, list[int]] = {}
        for i, t in enumerate(texts):
            groups.setdefault(tuple(t[k] for k in shared), []).append(i)
        order = list(groups.values())
        if order_by:
            order.sort(key=lambda g: tuple(self.runs[g[0]][f] for f in order_by))
        configs = []
        for group in order:
            config = {}
            for it in self.items:
                k = it['key']
                vals = list(dict.fromkeys(texts[i][k] for i in group))
                vals = [v for v in vals if v is not None]
                if not vals:
                    config[k] = None
                elif it.get('per_run'):
                    config[k] = merge(vals, sep)
                else:
                    config[k] = vals[0]
            configs.append(config)
        return configs

    def _values(self, sep: str) -> dict[str, str]:
        """One text per item over all configurations (the settings tables of single-configuration models)."""
        values = {}
        for it in self.items:
            vals = [c[it['key']] for c in self.configs if c[it['key']] is not None]
            if vals:
                values[it['key']] = merge(vals, sep)
        return values

    def label_of(self, key: str) -> str:
        return next(it['label'] for it in self.items if it['key'] == key)


class Info:
    """What a slide says about its models: shared settings as one paragraph, the rest as a table.

    ``common`` is a list of paragraphs (runs of (text, options)), ``header`` / ``fills`` / ``rows`` the table (or
    None): its columns are the models, or - when a model has several configurations - the configurations.
    """

    def __init__(self, common, header=None, fills=None, rows=None):
        self.common, self.header, self.fills, self.rows = common, header, fills, rows


def build_info(models: list[Model], keys: list[str] | None, deck: ReportDeck) -> Info:
    """The info of a slide's models, restricted to ``keys`` when given (in the order of the models' items)."""
    cfg, st = deck.cfg['info'], deck.s
    order = []
    for m in models:
        order += [it['key'] for it in m.items if it['key'] not in order]
    if keys:
        order = [k for k in order if k in keys]
    order = [k for k in order if any(k in m.values for m in models)]
    label = {k: next(m.label_of(k) for m in models if k in m.values) for k in order}

    multi = [m for m in models if len(m.configs) > 1]
    if multi:
        n = len(multi[0].configs)
        if any(len(m.configs) != n for m in multi):
            raise ValueError('models with different numbers of configurations')
        if not multi[0].config_labels:
            raise ValueError(
                f'model {multi[0].key}: several configurations, no config_labels'
            )
        header, fills = list(multi[0].config_labels), None
        columns = [
            [(m, m.configs[j if len(m.configs) > 1 else 0]) for m in models]
            for j in range(n)
        ]
    else:
        header, fills = [m.label for m in models], [m.colour for m in models]
        columns = [[(m, m.configs[0])] for m in models]

    def cell(k, column):
        parts = [(m, c.get(k)) for m, c in column if c.get(k) is not None]
        if not parts:
            return None
        if len({t for _, t in parts}) == 1:
            return parts[0][1]
        return ' · '.join(f'{m.short}: {t}' for m, t in parts)

    cells = {k: [cell(k, col) for col in columns] for k in order}
    cells = {k: v for k, v in cells.items() if any(t is not None for t in v)}
    one_column = len(columns) == 1
    common = [
        k
        for k, v in cells.items()
        if one_column or (None not in v and len(set(v)) == 1)
    ]

    runs = []
    if len(models) == 1:
        runs.append(
            (
                models[0].label + cfg['common_sep'],
                {'bold': True, 'color': models[0].colour},
            )
        )
    else:
        runs.append((cfg['common_label'] + cfg['common_sep'], {'bold': True}))
    for i, k in enumerate(common):
        if i:
            runs.append((cfg['join'], {'color': st['muted']}))
        runs.append(
            (label[k] + cfg['label_sep'], {'bold': True, 'color': st['accent_ink']})
        )
        runs.append((cells[k][0], {}))
    paragraphs = [runs] if common else []

    rows = [
        [label[k][:1].upper() + label[k][1:]]
        + [t if t is not None else cfg['missing'] for t in v]
        for k, v in cells.items()
        if k not in common
    ]
    if not rows:
        return Info(paragraphs)
    return Info(
        paragraphs, header=[cfg['table_corner']] + header, fills=fills, rows=rows
    )


def info_height(
    info: Info, width: float, fs: tuple[float, float], deck: ReportDeck
) -> float:
    """Height of the info box of ``width``: text of ``fs[0]`` pt, table of ``fs[1]`` pt."""
    inner = width - 2 * PAD
    h = 2 * PAD
    if info.common:
        h += deck.text_height(info.common, inner, fs[0])
    if info.rows:
        h += (0.08 if info.common else 0.0) + sum(
            deck.table_layout(info.header, info.rows, inner, fs[1])[2]
        )
    return h


def draw_info(
    deck: ReportDeck, slide, box, info: Info, fs: tuple[float, float]
) -> None:
    x, y, w, _ = box
    deck.rect(slide, box, deck.cfg['info']['box_fill'], name='Model info')
    cy = y + PAD
    if info.common:
        th = deck.text_height(info.common, w - 2 * PAD, fs[0])
        deck.text(
            slide,
            (x + PAD, cy, w - 2 * PAD, th),
            info.common,
            size=fs[0],
            space_after=0,
        )
        cy += th + 0.08
    if info.rows:
        deck.spec_table(
            slide,
            (x + PAD, cy, w - 2 * PAD, 0),
            info.header,
            info.rows,
            fs[1],
            info.fills,
        )


# ── Plot placement ───────────────────────────────────────────────────────────
def native_size(path: Path) -> tuple[float, float]:
    with Image.open(path) as im:
        dpi = im.info.get('dpi', (200, 200))[0] or 200
        return im.size[0] / dpi, im.size[1] / dpi


def place_images(sizes: list[tuple[float, float]], box) -> tuple[float, list]:
    """Boxes for one or two images in ``box`` (side by side or stacked, whichever is larger), with their scale."""
    x, y, w, h = box
    if not sizes:
        return 0.0, []
    options = []
    if len(sizes) == 1:
        options.append(('one', min(w / sizes[0][0], h / sizes[0][1])))
    else:
        row = min(min((w - GAP) / 2 / iw, h / ih) for iw, ih in sizes)
        col = min(min(w / iw, (h - GAP) / 2 / ih) for iw, ih in sizes)
        options += [('row', row), ('col', col)]
    mode, s = max(options, key=lambda o: o[1])
    dims = [(iw * s, ih * s) for iw, ih in sizes]
    if mode == 'col':
        total = sum(d[1] for d in dims) + GAP * (len(dims) - 1)
        cy, boxes = y + (h - total) / 2, []
        for dw, dh in dims:
            boxes.append((x + (w - dw) / 2, cy, dw, dh))
            cy += dh + GAP
    else:
        total = sum(d[0] for d in dims) + GAP * (len(dims) - 1)
        cx, boxes = x + (w - total) / 2, []
        for dw, dh in dims:
            boxes.append((cx, y + (h - dh) / 2, dw, dh))
            cx += dw + GAP
    return s, boxes


# ── Slide layouts ────────────────────────────────────────────────────────────
def subtitle(deck: ReportDeck, slide, text: str | None) -> float:
    """The line(s) under the title saying what is drawn; returns where the body starts."""
    if not text:
        return TITLE_Y + TITLE_H + 0.15
    size = float(deck.s['subtitle_size'])
    h = deck.text_height([text], BODY_W, size)
    deck.text(
        slide,
        (MARGIN, SUB_Y, BODY_W, h),
        [text],
        size=size,
        color=deck.s['muted'],
        space_after=0,
    )
    return SUB_Y + h + 0.15


def build_section(deck, s, _ctx):
    slide = deck.new(dark=True)
    st = deck.s
    deck.text(
        slide,
        (0.9, 2.5, 11.5, 1.3),
        [s['title']],
        size=36,
        bold=True,
        color=st['dark_ink'],
        anchor=MSO_ANCHOR.BOTTOM,
        space_after=0,
    )
    if s.get('subtitle'):
        deck.text(
            slide,
            (0.9, 4.0, 11.5, 1.6),
            [s['subtitle']],
            size=18,
            color=st['dark_muted'],
            space_after=0,
        )
    deck.footer(slide)


def build_models_table(deck, s, ctx):
    slide = deck.new()
    deck.title(slide, s['title'])
    top = subtitle(deck, slide, s.get('subtitle'))
    models = [ctx['models'][k] for k in s['models']]
    info = deck.cfg['info']
    keys = []
    for m in models:
        keys += [it['key'] for it in m.items if it['key'] not in keys]
    header = [info['table_corner']] + [m.label for m in models]
    rows = []
    for k in keys:
        if not any(k in m.values for m in models):
            continue
        label = next(m.label_of(k) for m in models if k in m.values)
        rows.append(
            [label[:1].upper() + label[1:]]
            + [m.values.get(k, info['missing']) for m in models]
        )
    size = float(deck.s['table_size'])
    while (
        size > 7
        and sum(deck.table_layout(header, rows, BODY_W, size)[2]) > BODY_BOTTOM - top
    ):
        size -= 0.5  # shrink the font until the table fits
    deck.spec_table(slide, (MARGIN, top, BODY_W, 0), header, rows, size)
    print(f'  {deck.n:2d} {s["title"]}: table at {size:g} pt')
    deck.footer(slide)


def build_plots(deck, s, ctx):
    slide = deck.new()
    deck.title(slide, s['title'])
    top = subtitle(deck, slide, s.get('subtitle'))
    height = BODY_BOTTOM - top
    paths = [ctx['plots_dir'] / p for p in s['plots']]
    missing = [p for p in paths if not p.is_file()]
    for p in missing:
        print(f'  MISSING plot {p}')
    present = [p for p in paths if p.is_file()]
    sizes = [native_size(p) for p in present]
    models = [ctx['models'][k] for k in s.get('models') or []]
    info = build_info(models, s.get('info_items'), deck) if models else None
    cfg = deck.cfg['info']

    # The layout that shows the plots largest: the info under them (full width) or beside them (one of the
    # `side_widths`), at the largest font of `font_sizes` that still leaves the plots `min_plot_scale` of their
    # size (else the layout with the largest plots over all fonts).
    best = None
    for fs in cfg['font_sizes']:
        fs = (float(fs[0]), float(fs[1]))
        cands = []
        if info is None:
            cands.append(
                ('none', *place_images(sizes, (MARGIN, top, BODY_W, height)), None)
            )
        else:
            h = info_height(info, BODY_W, fs, deck)
            if h <= height - 1.0:
                box = (MARGIN, top, BODY_W, height - h - GAP)
                cands.append(
                    (
                        'bottom',
                        *place_images(sizes, box),
                        (MARGIN, BODY_BOTTOM - h, BODY_W, h),
                    )
                )
            for w in cfg['side_widths']:
                h = info_height(info, float(w), fs, deck)
                if h <= height:
                    box = (MARGIN, top, BODY_W - float(w) - GAP, height)
                    cands.append(
                        (
                            'side',
                            *place_images(sizes, box),
                            (MARGIN + BODY_W - float(w), top, float(w), h),
                        )
                    )
        if not cands:
            continue
        pick = max(cands, key=lambda c: c[1])
        if best is None or pick[1] > best[1][1]:
            best = (fs, pick)
        if pick[1] >= float(cfg['min_plot_scale']):
            best = (fs, pick)
            break
    if best is None:
        raise ValueError(
            f'slide "{s["title"]}": the info fits nowhere; list fewer info_items'
        )
    fs, (name, scale, boxes, info_box) = best
    print(
        f'  {deck.n:2d} {s["title"]}: {name}, plots at {scale:.0%} of their size, info {fs[0]:g}/{fs[1]:g} pt'
    )

    for path, box in zip(present, boxes):
        deck.image(slide, path, box)
    if missing:  # say which plots are missing, where they would be
        deck.text(
            slide,
            (MARGIN, top, 6.0, 0.8),
            [deck.cfg['missing_plot'].format(files=', '.join(p.name for p in missing))],
            size=14,
            color=deck.s['muted'],
        )
    if info_box:
        draw_info(deck, slide, info_box, info, fs)
    deck.footer(slide)


LAYOUTS = {
    'title': lambda deck, s, ctx: scs.build_title(deck, s, None),
    'section': build_section,
    'models_table': build_models_table,
    'plots': build_plots,
}


# ── Previews ─────────────────────────────────────────────────────────────────
def draw_text(ax, deck: ReportDeck, box, paragraphs, kw):
    x, y, w, h = box
    size = float(kw.get('size', 16))
    bold = kw.get('bold', False)
    ink = kw.get('color') or deck.s['ink']
    space = float(kw.get('space_after', 8)) / 72
    family = font_manager.FontProperties(fname=deck.measure.paths[False]).get_name()
    blocks = [
        wrap(p, w - 2 * TEXT_MARGIN, size, bold, deck.measure) for p in paragraphs
    ]
    total = sum(len(b) for b in blocks) * size * LINE / 72 + space * (len(blocks) - 1)
    anchor = kw.get('anchor', MSO_ANCHOR.TOP)
    cy = {
        MSO_ANCHOR.MIDDLE: y + (h - total) / 2,
        MSO_ANCHOR.BOTTOM: y + h - total - TEXT_MARGIN,
    }.get(anchor, y + TEXT_MARGIN)
    align = kw.get('align', PP_ALIGN.LEFT)
    for lines in blocks:
        for line in lines:
            pieces = []
            for i, word in enumerate(line):
                for j, (text, opts) in enumerate(word):
                    pieces.append((('' if i == 0 or j > 0 else ' ') + text, opts))
            widths = [
                deck.measure.width(t, opts.get('size', size), opts.get('bold', bold))
                for t, opts in pieces
            ]
            lw = sum(widths)
            cx = {
                PP_ALIGN.CENTER: x + (w - lw) / 2,
                PP_ALIGN.RIGHT: x + w - TEXT_MARGIN - lw,
            }.get(align, x + TEXT_MARGIN)
            for (t, opts), pw in zip(pieces, widths):
                ax.text(
                    cx,
                    cy,
                    t,
                    fontsize=opts.get('size', size),
                    fontweight='bold' if opts.get('bold', bold) else 'normal',
                    color=opts.get('color', ink),
                    family=family,
                    va='top',
                    ha='left',
                )
                cx += pw
            cy += size * LINE / 72
        cy += space


def render_previews(deck: ReportDeck, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob('slide_*.png'):
        old.unlink()
    for n, page in enumerate(deck.pages, 1):
        fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=72)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, SLIDE_W)
        ax.set_ylim(SLIDE_H, 0)
        ax.axis('off')
        fig.patch.set_facecolor(deck.s['dark_bg'] if page['dark'] else '#ffffff')
        for op in page['ops']:
            if op[0] == 'rect':
                x, y, w, h = op[1]
                ax.add_patch(
                    FancyBboxPatch(
                        (x, y),
                        w,
                        h,
                        boxstyle='round,pad=0,rounding_size=0.08',
                        fc=op[2],
                        ec='none',
                    )
                )
            elif op[0] == 'image':
                x, y, w, h = op[1]
                img = plt.imread(op[2])
                ih, iw = img.shape[:2]
                s = min(w / iw, h / ih)
                pw, ph = iw * s, ih * s
                x0, y0 = x + (w - pw) / 2, y + (h - ph) / 2
                ax.imshow(img, extent=(x0, x0 + pw, y0 + ph, y0), aspect='auto')
            elif op[0] == 'text':
                draw_text(ax, deck, op[1], op[2], op[3])
            elif op[0] == 'table':
                for box, text, fill, ink, bold in op[1]:
                    x, y, w, h = box
                    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec='#d0d0cc', lw=0.5))
                    draw_text(
                        ax,
                        deck,
                        (
                            x + CELL_PAD_X - TEXT_MARGIN,
                            y + CELL_PAD_Y,
                            w - 2 * CELL_PAD_X + 2 * TEXT_MARGIN,
                            h - 2 * CELL_PAD_Y,
                        ),
                        [text],
                        {
                            'size': op[2],
                            'bold': bold,
                            'color': ink,
                            'anchor': MSO_ANCHOR.MIDDLE,
                            'space_after': 0,
                        },
                    )
        ax.set_xlim(0, SLIDE_W)
        ax.set_ylim(SLIDE_H, 0)
        fig.savefig(
            out_dir / f'slide_{n:02d}.png', dpi=72, facecolor=fig.get_facecolor()
        )
        plt.close(fig)
    print(f'{len(deck.pages)} previews -> {out_dir}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True, help='YAML config')
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = REPO / cfg['output']['dir']
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict = {}
    models = {
        k: Model(k, spec, cfg['info'], cache) for k, spec in cfg['models'].items()
    }
    for m in models.values():
        print(f'== {m.key} ({m.label}, {len(m.runs)} runs, {m.phase})')
        for k, v in m.values.items():
            print(f'   {m.label_of(k)}: {v}')

    measure = Measure(cfg['style']['measure_fonts'])
    deck = ReportDeck(cfg, dt.date.today().isoformat(), measure)
    ctx = {'models': models, 'plots_dir': REPO / cfg['plots_dir']}
    for s in cfg['slides']:
        LAYOUTS[s['layout']](deck, s, ctx)
    pptx = out_dir / cfg['output']['pptx']
    deck.prs.save(pptx)
    print(f'{deck.n} slides -> {pptx}')
    if cfg['output'].get('preview'):
        render_previews(deck, out_dir / cfg['output']['preview'])


if __name__ == '__main__':
    main()
