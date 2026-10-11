"""Mean best validation metric per lambda, with its seed-to-seed error.

Companion to ``scripts/plot_seed_curves.py``.  That script draws every seed
separately; this one collapses the seeds of each lambda into ONE number — the
mean over seeds of the best (lowest) value each training reached — so the
lambdas can be compared directly:

  * one point per lambda that has at least ``best_vs_lambda.min_seeds`` usable
    seeds, with an error bar (``std``, ``sem`` or ``minmax`` across those seeds)
    and, behind it, the individual seeds.  A lambda with a single usable seed has
    no spread to show and gets no error bar;
  * every ``jc == 0`` entry of the config (the no-contrastive baselines, which
    have no lambda) is drawn the same way, as a point with its error bar, in a
    slot of its own to the left of the scan behind a dotted divider.  Baselines
    named in ``best_vs_lambda.hide_references`` (by their ``label:``) are left off
    the figure, and its footnotes no longer mention them; they stay in the CSV.

Trainings that got stuck on a high plateau are left out of the mean and the
error, in the scan and in the references alike.  A training is STUCK when its
metric never gets down to ``best_vs_lambda.stuck_check.threshold`` within
``within_epochs`` epochs (every healthy run does, in a handful of epochs).  The
stuck trainings are NAMED in ``best_vs_lambda.exclude_broken`` (a list of seed
directories) rather than found by the rule, so what counts as stuck is visible in
the config and nobody is dropped silently; the rule only audits that list, and
the run reports any training it contradicts.

On top of that, ``best_vs_lambda.reference_outliers_above`` cuts the jc == 0 baselines
alone: a baseline seed whose best value is above it is left out as an outlier.  Those
trainings ran normally (they are not stuck); the lambda points are not cut.  They are
counted on the axis, named in a footnote and listed on stdout.

Every word on the figure comes from ``best_vs_lambda.texts`` (the placeholders
are documented next to each string in the config); ``DEFAULT_TEXTS`` is only the
fallback for a key the config leaves out.  A text set to ``null`` (or ``''``)
switches its element off: ``mean_entry`` the legend box, ``tick_outliers`` the
outlier line of a baseline's tick label, and ``tick_stuck`` /
``tick_reference_stuck`` the "stuck" variant of a tick label (the plain ``tick`` /
``tick_reference`` is used instead).

``best_vs_lambda.curves: true`` also writes the per-seed validation curves, a star at
each seed's best value, of EXACTLY the trainings behind the figure, to
``<filename stem>_curves.png``.  They are drawn from the entries of the points on the
figure, so whatever was left out (stuck runs, baseline outliers, hidden baselines,
lambdas under ``min_seeds``) is left out of the curves too and the two figures cannot
drift apart.  Axes, size and legend come from the config's ``plot:`` block, as for
``plot_seed_curves.py``.

It reads the SAME config as ``plot_seed_curves.py`` (``models``, ``color``,
``paths``, ``metric``, ``seed_status``), so the pool and the colours match the
curve plots; the extra settings live under ``best_vs_lambda:``.  A summary CSV
with the numbers behind every point is written next to the figure.

Colours come from the config's ``color:`` block, as for ``plot_seed_curves.py`` (a
colormap over lambda, or a categorical palette).  ``color.by_lambda`` ({'0.05': '#e6194b',
...}) pins one colour per lambda value instead, so a lambda keeps its colour whichever
other lambdas are on the figure; the ROC plots can read the same map.

A variant of the figure (another ``min_seeds``, a baseline hidden, other texts, its
own ``filename``) is a small OVERLAY file: ``--config`` takes several files and merges
them in the order given, so the overlay holds only the ``best_vs_lambda:`` keys that
differ and the pool stays defined in one place.

Usage::

    python scripts/plot_best_vs_lambda.py --config scripts/configs/val_loss_ce_seed_curves_cmap.yaml
    python scripts/plot_best_vs_lambda.py --config BASE.yaml OVERLAY.yaml
    python scripts/plot_best_vs_lambda.py --config ... --min-seeds 3 --error sem
"""

from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

# This script only writes PNG files, so never open a GUI window.  With DISPLAY set (an SSH X11 forward)
# matplotlib picks its Tk backend, and when the forward is dead plt.subplots() waits for it forever.
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402 -- after use()

sys.path.insert(0, str(Path(__file__).parent))
import plot_seed_curves as psc  # noqa: E402 -- pool / colour machinery
import plot_val_curves as pvc  # noqa: E402 -- run + metric readers

# Fallback wording for any key the config's ``best_vs_lambda.texts`` leaves out.  The
# config carries the same strings so they can be edited in one place; nothing else in
# this file puts words on the figure.
DEFAULT_TEXTS = {
    'title': '',
    'xlabel': r'augmentation strength $\lambda$',
    'ylabel': 'Best validation cross-entropy loss (per seed, then mean)',
    'legend_title': None,
    'mean_entry': (
        'mean best CE ± {error_name} over seeds\n'
        '(faint dots: individual seeds; no bar: single seed)'
    ),
    'error_name': {
        'std': 'standard deviation',
        'sem': 'standard error',
        'minmax': 'half range',
    },
    'tick': '{lam:g}\nn={n}',
    'tick_stuck': '{lam:g}\nn={n}\n({stuck} stuck)',
    'tick_reference': '{label}\nn={n}',
    'tick_reference_stuck': '{label}\nn={n}\n({stuck} stuck)',
    'tick_outliers': '({outliers} outlier)',
    'lambda_name': 'λ = {lam:g}',
    'note_stuck': (
        'Left out as stuck (validation CE never ≤ {threshold:.3f} within the first '
        '{within} epochs): {runs}.'
    ),
    'stuck_run': '{who} seed {seed} ({run})',
    'note_outliers': 'Left out of the baselines as outliers (best CE above {above:.3f}): {runs}.',
    'outlier_run': '{who} seed {seed} ({run}, best {best:.5f})',
    'note_hidden': 'Not shown (fewer than {min_seeds} usable seeds): {lambdas}.',
    'hidden_item': '{who} (n={n})',
}

# Tick labels of lambdas closer than this (in lambda units) would print on top of each
# other; the later one is moved to a second row.
_TICK_MIN_GAP = 0.03
# Spacing (lambda units) of the jc = 0 baseline slots, the last one a gap before the first
# lambda; the dotted divider sits halfway in that last gap.  Wide enough for the 4-line
# baseline tick labels not to touch.
_REF_GAP = 0.12
_NOTE_FONTSIZE = 7


class _Point(NamedTuple):
    """One point of the plot: a configuration, its colour and how many trainings it lost."""

    entry: object  # plot_val_curves.Entry
    colour: str
    n_stuck: int  # left out as stuck
    n_cut: int  # left out as baseline outliers (jc = 0 entries only)


class _Keep(dict):
    """``format_map`` namespace that leaves an unknown ``{name}`` as typed.

    Lets a template carry mathtext braces (``$\\alpha_{Agg}$``) without doubling them.
    """

    def __missing__(self, key: str) -> str:
        return '{' + key + '}'


def _fmt(template: str, **fields) -> str:
    return template.format_map(_Keep(fields))


def _plain(node, default):
    """Config node as plain Python; ``default`` if the key is absent."""
    return default if node is None else OmegaConf.to_container(node, resolve=True)


def _texts(bl) -> dict:
    """``DEFAULT_TEXTS`` overridden by ``best_vs_lambda.texts``; a mistyped key is an error."""
    given = _plain(bl.get('texts'), {})
    unknown = sorted(set(given) - set(DEFAULT_TEXTS))
    if unknown:
        raise KeyError(
            f'best_vs_lambda.texts has unknown key(s) {unknown}; valid keys: {sorted(DEFAULT_TEXTS)}'
        )
    return {**DEFAULT_TEXTS, **given}


def _error(values: np.ndarray, kind: str) -> float:
    """Spread across seeds: ``std`` (ddof=1), ``sem`` or ``minmax`` (half range)."""
    if values.size < 2:
        return 0.0
    if kind == 'minmax':
        return float((values.max() - values.min()) / 2)
    std = float(values.std(ddof=1))
    return std / np.sqrt(values.size) if kind == 'sem' else std


def _drop_broken(cfg, broken: set[Path]):
    """Copy of ``cfg`` whose ``dirs:`` no longer list the trainings in ``broken``.

    Returns the copy and the ``(directory, model label)`` pairs actually removed.  Only
    explicit ``dirs:`` entries can be filtered; a seed to drop hiding inside a ``cell:`` or
    ``dir:`` entry is an error rather than a silent no-op.  A model left with no seed at
    all is dropped from the copy (its trainings are still named in the footnote).
    """
    plain = OmegaConf.to_container(cfg, resolve=True)
    dropped: list[tuple[Path, str | None]] = []
    for model in plain['models']:
        if 'dirs' in model:
            hit = [d for d in model['dirs'] if Path(d).resolve() in broken]
            model['dirs'] = [d for d in model['dirs'] if d not in hit]
            dropped += [(Path(d), model.get('label')) for d in hit]
            continue
        root = Path(model.get('cell') or model.get('dir')).resolve()
        if any(b == root or root in b.parents for b in broken):
            raise ValueError(
                f'a training to drop lies inside {root}, which the config lists '
                "as 'cell:'/'dir:' — list its seeds with 'dirs:' so one can be dropped"
            )
    plain['models'] = [m for m in plain['models'] if m.get('dirs', True)]
    missing = broken - {p.resolve() for p, _ in dropped}
    for path in sorted(missing):
        print(f'  WARNING: training to drop not found in any model: {path}')
    return OmegaConf.create(plain), dropped


def _first_below(curve, threshold: float) -> int | None:
    """First epoch whose value is <= ``threshold``; None if the run never gets there."""
    hit = curve[curve['value'] <= threshold]
    return int(hit['epoch'].min()) if len(hit) else None


def _is_stuck(first: int | None, within: int) -> bool:
    return first is None or first > within


def _describe(run_dir: Path, metric: str, label: str | None, threshold: float) -> dict:
    """lambda / jc (from the run's own config), best value and plateau exit of one training."""
    rc = pvc.find_run_config(run_dir)
    curve = pvc.read_metric(pvc.find_metrics(run_dir), metric)
    imin = curve['value'].idxmin()
    return {
        'dir': run_dir,
        'label': label,
        'seed': pvc.seed_of(run_dir),
        'lam': float(OmegaConf.select(rc, 'lambda')),
        'jc': float(OmegaConf.select(rc, 'finetune.loss_weights.jet_contrastive') or 0),
        'best': float(curve.loc[imin, 'value']),
        'epoch': int(curve.loc[imin, 'epoch']),
        'first': _first_below(curve, threshold),
    }


def _audit_stuck(cfg_used, metric: str, threshold: float, within: int) -> list[dict]:
    """Trainings still in the pool that fail the stuck rule — there should be none."""
    seed_status = list(cfg_used.get('seed_status', ['completed']))
    suspects = []
    for model in cfg_used['models']:
        for run_dir in pvc.resolve_run_dirs(model, seed_status):
            info = _describe(run_dir, metric, model.get('label'), threshold)
            if _is_stuck(info['first'], within):
                suspects.append(info)
    return suspects


def _reference_outliers(cfg, metric: str, above: float, threshold: float) -> list[dict]:
    """Trainings of the jc = 0 models whose best value is above ``above`` (``reference_outliers_above``)."""
    seed_status = list(cfg.get('seed_status', ['completed']))
    found = []
    for model in cfg['models']:
        run_dirs = pvc.resolve_run_dirs(model, seed_status)
        # jc as pvc.load_entry reads it: the entry's own override, else the first run's config
        jc = model.get('jc')
        if jc is None:
            rc = pvc.find_run_config(run_dirs[0])
            jc = OmegaConf.select(rc, 'finetune.loss_weights.jet_contrastive') or 0
        if float(jc) != 0.0:
            continue
        infos = (_describe(d, metric, model.get('label'), threshold) for d in run_dirs)
        found += [i for i in infos if i['best'] > above]
    return found


def select_pool(cfg, metric: str, threshold: float):
    """The trainings the figures are built from: ``cfg.models`` minus the stuck runs and the baseline outliers.

    First the trainings named in ``best_vs_lambda.exclude_broken`` go, then the jc = 0 seeds whose best
    value is above ``best_vs_lambda.reference_outliers_above``.  Returns the reduced config and the
    :func:`_describe` records of what was left out (stuck, outliers).  Shared with
    ``plot_ce_and_embedding.py``, so every figure drawn from this pool agrees on what is in it.
    """
    bl = cfg.get('best_vs_lambda', {})
    broken = {Path(p).resolve() for p in _plain(bl.get('exclude_broken'), [])}
    cfg_used, dropped = _drop_broken(cfg, broken)
    dropped_info = [_describe(d, metric, label, threshold) for d, label in dropped]

    # Outliers of the jc = 0 baselines, cut from what the stuck rule left.
    above = bl.get('reference_outliers_above')
    outlier_info = []
    if above is not None:
        outlier_info = _reference_outliers(cfg_used, metric, float(above), threshold)
        cfg_used, _ = _drop_broken(cfg_used, {o['dir'].resolve() for o in outlier_info})
    return cfg_used, dropped_info, outlier_info


def hidden_references(bl, baseline_labels: set) -> set:
    """``best_vs_lambda.hide_references`` as a set of labels.

    A label that matches no baseline is an error: a mistyped one would otherwise leave the
    baseline on the figure without a word.
    """
    hide = set(_plain(bl.get('hide_references'), []))
    if hide - baseline_labels:
        raise KeyError(
            f'best_vs_lambda.hide_references names {sorted(hide - baseline_labels)}, which is not the label '
            f'of a jc = 0 model; the baselines are {sorted(label for label in baseline_labels if label)}'
        )
    return hide


def _lambda_colours(entries, colours: list, by_lambda: dict) -> list:
    """Apply ``color.by_lambda``: one FIXED colour per lambda value, whichever other lambdas are present.

    ``by_lambda`` maps a lambda (``'0.05'``) to a colour.  It wins over the palette / colormap for every
    jc != 0 entry without a ``color:`` of its own; a lambda it does not list keeps the colour
    ``plot_seed_curves`` gave it, with a warning.  The jc = 0 baselines are not touched.
    """
    if not by_lambda:
        return colours
    fixed = {f'{float(k):g}': v for k, v in by_lambda.items()}
    out = []
    for entry, colour in zip(entries, colours):
        if entry.jc == 0.0 or entry.color is not None:
            out.append(colour)
        elif f'{entry.lam:g}' in fixed:
            out.append(fixed[f'{entry.lam:g}'])
        else:
            print(
                f'  WARNING: color.by_lambda has no colour for lambda {entry.lam:g}; keeping {colour}'
            )
            out.append(colour)
    return out


def _who(texts: dict, ref_names: dict, label: str | None, jc: float, lam: float) -> str:
    """How a configuration is called in the notes: its (renamed) baseline label, else the lambda.

    One line, whatever line breaks the label carries on the axis.
    """
    if jc == 0.0 and label:
        return ' '.join(ref_names.get(label, label).split())
    return _fmt(texts['lambda_name'], lam=lam)


def _summary_row(
    entry, kind: str, shown: bool, n_stuck: int, n_cut: int, error: str
) -> dict:
    vals = entry.best_values
    return {
        'kind': kind,
        'label': entry.label or f'lambda = {entry.lam:g}',
        'lambda': entry.lam,
        'jc': entry.jc,
        'n_used': entry.n_seeds,
        'n_excluded_broken': n_stuck,
        'n_excluded_outliers': n_cut,
        'shown_in_plot': shown,
        'mean_best': float(vals.mean()),
        'std': _error(vals, 'std'),
        'sem': _error(vals, 'sem'),
        'min_best': float(vals.min()),
        'max_best': float(vals.max()),
        'error_kind': error,
        'error_value': _error(vals, error),
        'seeds_used': ' '.join(str(s) for s in entry.seeds),
        'best_per_seed': ' '.join(f'{v:.5f}' for v in vals),
    }


def _stagger(xs: list[float], labels: list[str], min_gap: float) -> list[str]:
    """Tick labels, with any closer than ``min_gap`` to the last first-tier label moved to a second tier.

    The second tier starts below the tallest label, so a moved label cannot land beside the
    extra lines (``n=``, ``(k stuck)``) of its neighbours.
    """
    tier = max(label.count('\n') + 1 for label in labels)
    out, last = [], -np.inf
    for x, label in zip(xs, labels):
        if x - last < min_gap:
            out.append('\n' * tier + label)
        else:
            out.append(label)
            last = x
    return out


def _footnote(fig, ax, notes: list[str]) -> None:
    """Small grey notes right under the x label, wrapped to the width of the axes."""
    if not notes:
        return
    fig.draw_without_rendering()
    label_box = ax.xaxis.label.get_window_extent().transformed(
        fig.transFigure.inverted()
    )
    axes_box = ax.get_position()
    chars = int(
        axes_box.width * fig.get_figwidth() * 72 / (_NOTE_FONTSIZE * 0.58)
    )  # ~0.58 em/char
    fig.text(
        axes_box.x0,
        label_box.y0 - 0.015,
        '\n'.join(textwrap.fill(n, chars) for n in notes),
        ha='left',
        va='top',
        fontsize=_NOTE_FONTSIZE,
        color='0.3',
    )


def _point(ax, x: float, vals: np.ndarray, colour, error: str, rng) -> None:
    """One configuration: its individual seeds faint behind, mean ± error in front.

    A single seed has no spread: its marker is the seed, with no bar and no extra dot.
    """
    if vals.size > 1:
        ax.scatter(
            x + rng.uniform(-0.006, 0.006, vals.size),
            vals,
            s=16,
            color=colour,
            alpha=0.45,
            linewidths=0,
            zorder=3,
        )
    ax.errorbar(
        x,
        float(vals.mean()),
        yerr=_error(vals, error) if vals.size > 1 else None,
        fmt='o',
        markersize=7,
        color=colour,
        markeredgecolor='black',
        markeredgewidth=0.7,
        ecolor='0.2',
        elinewidth=1.4,
        capsize=3,
        capthick=1.4,
        zorder=5,
    )


def plot_best_vs_lambda(scan, refs, cfg, texts, error, notes, out_path):
    """``scan`` / ``refs``: a :class:`_Point` per lambda / per jc = 0 baseline."""
    bl = cfg.get('best_vs_lambda', {})
    ref_names = _plain(bl.get('reference_labels'), {})
    above = bl.get('reference_outliers_above')
    fig, ax = plt.subplots(figsize=tuple(bl.get('figsize', [11.0, 5.6])))
    rng = np.random.default_rng(0)
    xs, ticks = [], []  # every point's x and tick label, baselines first

    # Baselines (jc == 0) have no lambda: one slot each to the left of the scan, in config
    # order, the last a gap before the first lambda, behind a dotted divider.
    first_lam = min(p.entry.lam for p in scan)
    for k, p in enumerate(refs):
        x = first_lam - _REF_GAP * (len(refs) - k)
        _point(ax, x, p.entry.best_values, p.colour, error, rng)
        xs.append(x)
        tick = _fmt(
            texts['tick_reference_stuck']
            if p.n_stuck and texts['tick_reference_stuck']
            else texts['tick_reference'],
            label=ref_names.get(p.entry.label, p.entry.label),
            n=p.entry.n_seeds,
            stuck=p.n_stuck,
        )
        if (
            p.n_cut and texts['tick_outliers']
        ):  # an empty line would still take up a row
            tick += '\n' + _fmt(texts['tick_outliers'], outliers=p.n_cut, above=above)
        ticks.append(tick)
    if refs:
        ax.axvline(
            first_lam - _REF_GAP / 2,
            color='0.55',
            linestyle=':',
            linewidth=1.0,
            zorder=1,
        )

    for p in scan:
        _point(ax, p.entry.lam, p.entry.best_values, p.colour, error, rng)
        xs.append(p.entry.lam)
        ticks.append(
            _fmt(
                texts['tick_stuck']
                if p.n_stuck and texts['tick_stuck']
                else texts['tick'],
                lam=p.entry.lam,
                n=p.entry.n_seeds,
                stuck=p.n_stuck,
            )
        )

    if texts['mean_entry']:  # no text, no legend box
        marker = Line2D(
            [],
            [],
            marker='o',
            markersize=7,
            color='0.2',
            markerfacecolor='0.75',
            markeredgecolor='black',
            linestyle='none',
            label=_fmt(
                texts['mean_entry'], error_name=texts['error_name'].get(error, error)
            ),
        )
        ax.legend(
            handles=[marker],
            title=texts['legend_title'],
            loc='upper right',
            fontsize=8,
            framealpha=0.95,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(_stagger(xs, ticks, _TICK_MIN_GAP))
    ax.set_xlim(min(xs) - 0.06, max(xs) + 0.04)
    if 'ylim' in bl:
        ax.set_ylim(*bl['ylim'])
    ax.set_xlabel(texts['xlabel'])
    ax.set_ylabel(texts['ylabel'])
    ax.set_title(texts['title'])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _footnote(fig, ax, notes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        nargs='+',
        help='YAML config; any further file is merged on top of it, in the order given '
        '(an overlay holding only the best_vs_lambda keys that differ)',
    )
    parser.add_argument('--output', type=Path, help='Override plot.out_dir')
    parser.add_argument(
        '--min-seeds', type=int, help='Override best_vs_lambda.min_seeds'
    )
    parser.add_argument(
        '--error',
        choices=['std', 'sem', 'minmax'],
        help='Override best_vs_lambda.error',
    )
    args = parser.parse_args()

    cfg = OmegaConf.merge(*(OmegaConf.load(p) for p in args.config))
    bl = cfg.get('best_vs_lambda', {})
    min_seeds = args.min_seeds or int(bl.get('min_seeds', 1))
    error = args.error or bl.get('error', 'std')
    metric = cfg.get('metric', 'val/head/loss_ce')
    texts = _texts(bl)
    ref_names = _plain(bl.get('reference_labels'), {})
    stuck_cfg = bl.get('stuck_check', {})
    threshold = float(stuck_cfg.get('threshold', 0.610))
    within = int(stuck_cfg.get('within_epochs', 20))

    cfg_used, dropped_info, outlier_info = select_pool(cfg, metric, threshold)
    above = bl.get('reference_outliers_above')
    if above is not None:
        above = float(above)

    # Keyed by label too: both baselines are (jc, lambda) = (0, 0.1).
    lost = Counter((d['label'], d['jc'], d['lam']) for d in dropped_info)
    cut = Counter((o['label'], o['jc'], o['lam']) for o in outlier_info)
    entries, colours, _ = psc._entries_and_colors(cfg_used)
    colours = _lambda_colours(
        entries, colours, _plain(cfg.get('color', {}).get('by_lambda'), {})
    )

    # Baselines left off the figure.
    hide = hidden_references(bl, {e.label for e in entries if e.jc == 0.0})

    def drawn(info: dict) -> bool:
        """False for a training of a hidden baseline: the figure never mentions that baseline."""
        return not (info['jc'] == 0.0 and info['label'] in hide)

    scan, refs, hidden, rows = [], [], [], []
    for entry, colour in zip(entries, colours):
        key = (entry.label, entry.jc, entry.lam)
        point = _Point(entry, colour, lost[key], cut[key])
        if entry.jc == 0.0:
            shown = entry.label not in hide
            if shown:
                refs.append(point)
            rows.append(
                _summary_row(
                    entry, 'reference', shown, point.n_stuck, point.n_cut, error
                )
            )
        elif entry.n_seeds >= min_seeds:
            scan.append(point)
            rows.append(
                _summary_row(entry, 'scan', True, point.n_stuck, point.n_cut, error)
            )
        else:
            hidden.append(entry)
            rows.append(
                _summary_row(entry, 'scan', False, point.n_stuck, point.n_cut, error)
            )

    print(
        f'{"":<40}{"n":>3}  {"mean best":>10}  {error:>8}  {"sem":>8}   best per seed'
    )
    for r in rows:
        if r['shown_in_plot']:
            tag = ''
        elif r['kind'] == 'reference':
            tag = '  (not drawn: hide_references)'
        else:
            tag = f'  (not shown: < {min_seeds} seeds)'
        name = r['label'] if r['kind'] == 'reference' else f'lambda = {r["lambda"]:g}'
        print(
            f'{name:<40}{r["n_used"]:>3}  {r["mean_best"]:>10.5f}  {r["error_value"]:>8.5f}'
            f'  {r["sem"]:>8.5f}   {r["best_per_seed"]}{tag}'
        )

    # Audit of exclude_broken against the stuck rule, in both directions.
    print(
        f'\nLeft out as stuck (val metric never <= {threshold:g} within {within} epochs):'
    )
    for d in dropped_info:
        first = 'never' if d['first'] is None else f'epoch {d["first"]}'
        flag = (
            ''
            if _is_stuck(d['first'], within)
            else '   <-- WARNING: passes the stuck rule'
        )
        flag += '' if drawn(d) else '   (baseline not drawn)'
        who = _who(texts, ref_names, d['label'], d['jc'], d['lam'])
        print(
            f'  {who:<34} seed {d["seed"]:<10} {d["dir"].parent.parent.name}  '
            f'first <= {threshold:g}: {first}; best {d["best"]:.5f} @ epoch {d["epoch"]}{flag}'
        )
    suspects = _audit_stuck(cfg_used, metric, threshold, within)
    for s in suspects:
        who = _who(texts, ref_names, s['label'], s['jc'], s['lam'])
        print(
            f'  WARNING: {who} seed {s["seed"]} ({s["dir"].parent.parent.name}) fails the stuck '
            f'rule (first <= {threshold:g}: {s["first"]}) but is still in the pool'
        )
    if not suspects:
        n_pool = sum(e.n_seeds for e in entries)
        print(
            f'Audit: all {n_pool} trainings in the pool reach <= {threshold:g} within {within} epochs.'
        )

    if above is not None:
        print(
            f'\nLeft out of the jc = 0 baselines as outliers (best value above {above:g}):'
        )
        for o in outlier_info:
            who = _who(texts, ref_names, o['label'], o['jc'], o['lam'])
            flag = (
                '   <-- WARNING: also stuck, list it in exclude_broken'
                if _is_stuck(o['first'], within)
                else ''
            )
            flag += '' if drawn(o) else '   (baseline not drawn)'
            print(
                f'  {who:<34} seed {o["seed"]:<10} {o["dir"].parent.parent.name}  '
                f'best {o["best"]:.5f} @ epoch {o["epoch"]}{flag}'
            )

    notes = []
    stuck_shown = [d for d in dropped_info if drawn(d)]
    if stuck_shown:
        runs = '; '.join(
            _fmt(
                texts['stuck_run'],
                who=_who(texts, ref_names, d['label'], d['jc'], d['lam']),
                seed=d['seed'],
                run=d['dir'].parent.parent.name,
            )
            for d in stuck_shown
        )
        notes.append(
            _fmt(texts['note_stuck'], runs=runs, threshold=threshold, within=within)
        )
    outliers_shown = [o for o in outlier_info if drawn(o)]
    if outliers_shown:
        runs = '; '.join(
            _fmt(
                texts['outlier_run'],
                who=_who(texts, ref_names, o['label'], o['jc'], o['lam']),
                seed=o['seed'],
                run=o['dir'].parent.parent.name,
                best=o['best'],
            )
            for o in outliers_shown
        )
        notes.append(_fmt(texts['note_outliers'], runs=runs, above=above))
    if hidden:
        lambdas = ', '.join(
            _fmt(
                texts['hidden_item'],
                who=_who(texts, ref_names, e.label, e.jc, e.lam),
                n=e.n_seeds,
            )
            for e in hidden
        )
        notes.append(_fmt(texts['note_hidden'], lambdas=lambdas, min_seeds=min_seeds))

    out_dir = args.output or Path(
        cfg.get('plot', {}).get('out_dir', 'plots/val_curves')
    )
    out_path = out_dir / bl.get('filename', 'best_vs_lambda.png')
    plot_best_vs_lambda(scan, refs, cfg, texts, error, notes, out_path)
    print(f'Saved {out_path}')

    csv_path = out_path.with_name(out_path.stem + '_summary.csv')
    with csv_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {csv_path}')

    if bl.get('curves', False):
        drawn = refs + scan
        curves_path = out_path.with_name(out_path.stem + '_curves' + out_path.suffix)
        psc.plot_seed_curves(
            [p.entry for p in drawn],
            [p.colour for p in drawn],
            cfg,
            metric,
            curves_path,
        )
        print(
            f'Saved {curves_path}  ({sum(p.entry.n_seeds for p in drawn)} seed curves, one star each)'
        )


if __name__ == '__main__':
    main()
