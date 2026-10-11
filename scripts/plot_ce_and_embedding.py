"""Train / validation cross-entropy and an embedding-quality metric, epoch by epoch, for the best run of each kind.

Two trainings are picked from the pool that ``plot_best_vs_lambda.py`` draws (read from the same config files:
same stuck-run list, same baseline outlier cut, same ``hide_references`` and ``min_seeds``, so the pool is the
trainings behind that figure):

  ``ce_only``  the jc == 0 training with the lowest best validation CE (cross-entropy only, no contrastive loss);
  ``cl``       the jc != 0 training with the lowest best validation CE (cross-entropy + jet-contrastive loss, any
               lambda of the pool).

"Best" is the minimum of the validation CE over the epochs of a training.  For each of the two and each metric of
``embedding_curves.metrics`` one figure is written:

  left axis   the training CE and the validation CE of every epoch, a star on the validation curve at its minimum;
  right axis  the metric of ``embedding_curves.embedding`` of every epoch.  By default that is the ``aggregator``
              output, the single jet embedding the classifier head reads and the jet-contrastive loss acts on;
              ``jets_embedding`` / ``tracks_embedding`` (the per-object projections) are logged as well.

The metrics come from the ``EmbeddingMetrics`` callback, which logs them at the end of each VALIDATION epoch on the
clean (no augmentation) embeddings of up to 8192 validation jets (``val/<embedding>/<metric>``); there is no
training-split value, so the right-axis curve is a validation curve like the validation CE.

Every word on the figures comes from ``embedding_curves.texts``; the placeholders of each text are documented next
to it in the config, a missing or mistyped key is an error, and nothing in this file puts words on a figure.

A ``run_selection.csv`` next to the figures lists every training that competed (ranked within its kind) and which
one was picked, so each figure can be traced back to its training directory.

Usage::

    python scripts/plot_ce_and_embedding.py --config \\
        scripts/configs/val_loss_ce_seed_curves_cmap.yaml \\
        scripts/configs/val_loss_ce_best_vs_lambda_no_dropout_min2.yaml \\
        scripts/configs/ce_and_embedding_curves.yaml

``--config`` takes several files and merges them in the order given, as ``plot_best_vs_lambda.py`` does: the base
config defines the trainings, the overlay of that figure its ``min_seeds`` / ``hide_references``, and the last file
the ``embedding_curves:`` block.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
from omegaconf import OmegaConf

# This script only writes PNG files, so never open a GUI window.  With DISPLAY set (an SSH X11 forward)
# matplotlib picks its Tk backend, and when the forward is dead plt.subplots() waits for it forever.
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402 -- after use()

sys.path.insert(0, str(Path(__file__).parent))
import plot_best_vs_lambda as pbl  # noqa: E402 -- the pool, and the text templating
import plot_val_curves as pvc  # noqa: E402 -- run + metric readers

# The two kinds of training.  They key ``texts.title`` and fill the ``{role}`` of the file names.
ROLES = ('ce_only', 'cl')

# Keys of ``embedding_curves.texts``: plain strings, one string per role, one string per metric.
_TEXTS_PLAIN = ('xlabel', 'ylabel_left', 'train_ce', 'val_ce', 'best')
_TEXTS_PER_ROLE = ('title',)
_TEXTS_PER_METRIC = ('ylabel_right', 'metric')


def _role(jc: float) -> str:
    return 'ce_only' if jc == 0.0 else 'cl'


def _texts(ec: dict, metrics: list[str]) -> dict:
    """``embedding_curves.texts``, checked: no unknown key, none missing, no role or metric left out."""
    given = ec.get('texts') or {}
    wanted = set(_TEXTS_PLAIN) | set(_TEXTS_PER_ROLE) | set(_TEXTS_PER_METRIC)
    if set(given) != wanted:
        raise KeyError(
            f'embedding_curves.texts: unknown key(s) {sorted(set(given) - wanted)}, '
            f'missing key(s) {sorted(wanted - set(given))}'
        )
    for key, names in [(k, ROLES) for k in _TEXTS_PER_ROLE] + [
        (k, metrics) for k in _TEXTS_PER_METRIC
    ]:
        absent = sorted(set(names) - set(given[key] or {}))
        if absent:
            raise KeyError(f'embedding_curves.texts.{key} has no entry for {absent}')
    return given


def candidates(cfg, metric: str, threshold: float) -> list[dict]:
    """Every training drawn on the best-vs-lambda figure of ``cfg``: a :func:`pbl._describe` record plus its role.

    The pool is ``pbl.select_pool`` (stuck runs and baseline outliers gone); on top of that a baseline named in
    ``best_vs_lambda.hide_references`` and a lambda with fewer than ``best_vs_lambda.min_seeds`` usable trainings
    are left out, exactly as that figure leaves them out.  jc and lambda are read from each run's own config.
    """
    bl = cfg.get('best_vs_lambda', {})
    min_seeds = int(bl.get('min_seeds', 1))
    cfg_used, _, _ = pbl.select_pool(cfg, metric, threshold)

    seed_status = list(cfg_used.get('seed_status', ['completed']))
    models = [
        [
            pbl._describe(d, metric, model.get('label'), threshold)
            for d in pvc.resolve_run_dirs(model, seed_status)
        ]
        for model in cfg_used['models']
    ]
    hide = pbl.hidden_references(
        bl, {m[0]['label'] for m in models if m[0]['jc'] == 0.0}
    )

    rows = []
    for infos in models:
        jc = infos[0]['jc']
        if jc == 0.0 and infos[0]['label'] in hide:
            continue
        if jc != 0.0 and len(infos) < min_seeds:
            continue
        rows += [{**info, 'role': _role(info['jc'])} for info in infos]
    return rows


def rank(rows: list[dict]) -> dict[str, list[dict]]:
    """The candidates of each role, best (lowest validation CE) first."""
    ranked = {
        role: sorted((r for r in rows if r['role'] == role), key=lambda r: r['best'])
        for role in ROLES
    }
    empty = [role for role, rs in ranked.items() if not rs]
    if empty:
        raise ValueError(
            f'no training left in the pool for {empty}; check hide_references / min_seeds'
        )
    return ranked


def _run_name(info: dict) -> str:
    """The sweep directory of a training (``<run>/<cell>/seed_<n>``)."""
    return info['dir'].parent.parent.name


def plot_run(
    info: dict, series: dict, metric: str, texts: dict, ec, out_path: Path
) -> None:
    """One figure: CE curves on the left axis, ``metric`` on a twin right axis, one legend."""
    colors = ec['colors']
    styles = ec['linestyles']
    width = float(ec['linewidth'])
    fields = {
        'lam': info['lam'],
        'jc': info['jc'],
        'seed': info['seed'],
        'run': _run_name(info),
        'best': info['best'],
        'epoch': info['epoch'],
    }

    fig, ax = plt.subplots(figsize=tuple(ec['figsize']))
    right = ax.twinx()

    train, val, emb = series['train_ce'], series['val_ce'], series[metric]
    handles = ax.plot(
        train['epoch'],
        train['value'],
        color=colors['train_ce'],
        linestyle=styles['train_ce'],
        linewidth=width,
        label=texts['train_ce'],
    )
    handles += ax.plot(
        val['epoch'],
        val['value'],
        color=colors['val_ce'],
        linestyle=styles['val_ce'],
        linewidth=width,
        label=texts['val_ce'],
    )
    if ec['best_marker']:
        star = ax.plot(
            [info['epoch']],
            [info['best']],
            marker='*',
            markersize=13,
            linestyle='none',
            zorder=6,
            color=colors['val_ce'],
            markeredgecolor='white',
            markeredgewidth=0.6,
            label=pbl._fmt(texts['best'], **fields) if texts['best'] else None,
        )
        if texts['best']:  # no text, a star without legend entry
            handles += star
    handles += right.plot(
        emb['epoch'],
        emb['value'],
        color=colors['metric'],
        linestyle=styles['metric'],
        linewidth=width,
        label=texts['metric'][metric],
    )

    ax.set_xlabel(texts['xlabel'])
    ax.set_ylabel(texts['ylabel_left'])
    right.set_ylabel(texts['ylabel_right'][metric], color=colors['metric'])
    right.tick_params(axis='y', colors=colors['metric'])
    right.spines['right'].set_color(colors['metric'])
    ax.spines['right'].set_visible(
        False
    )  # the left axes lies on top and would paint over the coloured spine
    title = texts['title'][info['role']]
    if title:
        ax.set_title(pbl._fmt(title, **fields))
    ax.grid(True, alpha=0.3)

    # The twin is drawn after (above) the first axes; swap them so the CE curves and the star stay visible.
    ax.set_zorder(right.get_zorder() + 1)
    ax.patch.set_visible(False)

    if ec.get('xlim'):
        ax.set_xlim(*ec['xlim'])
    if ec.get('ylim_left'):
        ax.set_ylim(*ec['ylim_left'])
    ylim_right = ec['ylim_right'].get(metric)
    if ylim_right:
        right.set_ylim(*ylim_right)

    legend = ec['legend']
    kwargs = {
        'loc': legend['loc'],
        'ncol': int(legend['ncol']),
        'fontsize': legend['fontsize'],
        'frameon': legend['frame'],
    }
    if legend.get('anchor'):
        kwargs['bbox_to_anchor'] = tuple(legend['anchor'])
    ax.legend(handles=handles, **kwargs)

    fig.tight_layout()
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
        help='YAML config; any further file is merged on top of it, in the order given',
    )
    parser.add_argument('--output', type=Path, help='Override embedding_curves.out_dir')
    args = parser.parse_args()

    cfg = OmegaConf.merge(*(OmegaConf.load(p) for p in args.config))
    ec = cfg.get('embedding_curves')
    if ec is None:
        raise KeyError(
            f'none of {[str(p) for p in args.config]} has an embedding_curves: block'
        )
    ec = OmegaConf.to_container(ec, resolve=True)
    metrics = list(ec['metrics'])
    texts = _texts(ec, metrics)

    val_metric = cfg.get('metric', 'val/head/loss_ce')
    threshold = float(
        cfg.get('best_vs_lambda', {}).get('stuck_check', {}).get('threshold', 0.610)
    )
    ranked = rank(candidates(cfg, val_metric, threshold))

    columns = {'train_ce': ec['train_metric'], 'val_ce': val_metric}
    columns |= {m: f'val/{ec["embedding"]}/{m}' for m in metrics}

    out_dir = args.output or Path(ec['out_dir'])
    rows = []
    for role in ROLES:
        print(
            f'\n{role}: {len(ranked[role])} candidate trainings, best validation CE first'
        )
        for k, r in enumerate(ranked[role], start=1):
            # lambda is the strength of the contrastive noise: it means nothing without a contrastive loss
            lam = f'{r["lam"]:g}' if r['jc'] != 0.0 else ''
            tag = '   <-- picked' if k == 1 else ''
            if k <= 5:
                print(
                    f'  {k:>2}. lambda {lam or "-":<5} seed {r["seed"]:<10} {_run_name(r):<22} '
                    f'best {r["best"]:.5f} @ epoch {r["epoch"]:<3}{tag}'
                )
            rows.append(
                {
                    'role': role,
                    'rank': k,
                    'picked': k == 1,
                    'label': r['label'] or '',
                    'lambda': lam,
                    'jc': r['jc'],
                    'seed': r['seed'],
                    'best_val_ce': r['best'],
                    'best_epoch': r['epoch'],
                    'run': _run_name(r),
                    'dir': str(r['dir'].resolve()),
                }
            )

    for role in ROLES:
        info = ranked[role][0]
        csv_path = pvc.find_metrics(info['dir'])
        series = {key: pvc.read_metric(csv_path, col) for key, col in columns.items()}
        for m in metrics:
            out_path = out_dir / ec['filename'].format(role=role, metric=m)
            plot_run(info, series, m, texts, ec, out_path)
            print(f'Saved {out_path}')

    sel_path = out_dir / 'run_selection.csv'
    with sel_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {sel_path}')


if __name__ == '__main__':
    main()
