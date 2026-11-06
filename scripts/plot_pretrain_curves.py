"""Pretraining runs epoch by epoch: the loss and its terms, and the loss next to an embedding metric.

For each run of ``pretrain_curves.runs`` and each entry of ``pretrain_curves.figures`` one figure is written:

  loss_terms   the total loss and each of its terms, training (epoch mean) and validation, one colour per term and
               one line style per split.  The terms and their weights come from the run's own .hydra config
               (``pretrain.loss_weights``): an embedding's term is drawn when its weight is > 0, and the legend
               writes the total as the weighted sum of the drawn terms.  The logged total is checked against that
               sum (``sum_tolerance``), so the legend cannot claim a sum the run did not optimise.
  <metric>     any other key (effective_rank, uniformity, ...): the layout of ``plot_ce_and_embedding.py``, the
               training and validation loss on the left axis and ``val/<embedding>/<metric>`` of every trained
               embedding on a twin right axis.  An embedding listed in ``trained_if`` is drawn only when that loss
               weight is > 0: P0 never trains the aggregator, so its metrics there describe random weights.

A figure entry may override any top-level style key (figsize, marker, linestyles, ylim_left, ylim_right, legend).
Every word on the figures comes from ``pretrain_curves.texts``; a missing key is an error, and nothing in this file
puts words on a figure.

Usage::

    python scripts/plot_pretrain_curves.py --config scripts/configs/pretrain_curves_final.yaml
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
import plot_arm_epochs as pae  # noqa: E402 -- .hydra config reader
import plot_val_curves as pvc  # noqa: E402 -- run + metric readers

SPLITS = ('train', 'val')
LOSS_TERMS = 'loss_terms'  # the figure kind that draws the terms; every other kind is an embedding metric


def run_meta(run_dir: Path, pc: dict) -> dict:
    """Loss weights and embedding sizes of one run, from its own .hydra config."""
    cfg = pae.run_config(run_dir)
    weights = {k: float(v) for k, v in cfg.pretrain.loss_weights.items()}
    dims = {}
    for emb, keys in pc['embedding_dim'].items():
        values = [OmegaConf.select(cfg, k) for k in keys]
        if any(v is None for v in values):
            raise KeyError(
                f'{run_dir}: its .hydra config has no value for one of {keys}'
            )
        dims[emb] = sum(int(v) for v in values)
    return {'weights': weights, 'dims': dims}


def term_weight(emb: str, meta: dict, pc: dict) -> float:
    return meta['weights'][pc['loss_weight'][emb]]


def is_trained(emb: str, meta: dict, pc: dict) -> bool:
    """An embedding listed in ``trained_if`` is trained only when that loss weight is > 0."""
    key = (pc.get('trained_if') or {}).get(emb)
    return key is None or meta['weights'][key] > 0


def check_total(
    csv_path: Path, split: str, terms: list[str], meta: dict, pc: dict
) -> None:
    """Fail unless the logged total equals the weighted sum of ``terms`` at every epoch."""
    total = pvc.read_metric(csv_path, pc['loss_metrics'][split])
    total = total.set_index('epoch')['value']
    weighted = sum(
        term_weight(e, meta, pc)
        * pvc.read_metric(
            csv_path, pc['term_metrics'][split].format(embedding=e)
        ).set_index('epoch')['value']
        for e in terms
    )
    # An epoch missing on one side gives NaN, which fails the check below.
    diff = (total - weighted).abs().max(skipna=False)
    if not diff <= float(pc['sum_tolerance']):
        raise ValueError(
            f'{csv_path}: the {split} total differs from the weighted sum of {terms} by {diff:.2e}'
        )
    print(f'  {split}: total = weighted sum of {terms} (largest difference {diff:.1e})')


def plot_curve(ax, s, st: dict, **kwargs) -> list:
    return ax.plot(
        s['epoch'] + int(st['epoch_offset']),
        s['value'],
        linewidth=float(st['linewidth']),
        marker=st.get('marker'),
        markersize=float(st['markersize']),
        **kwargs,
    )


def plot_loss_terms(
    csv_path: Path, meta: dict, st: dict, texts: dict, ax
) -> list[Line2D]:
    """Total and terms on ``ax``; returns the legend handles (terms by colour, splits by line style)."""
    colors, styles, ink = st['colors'], st['linestyles'], st['ink']
    terms = [e for e in st['embeddings'] if term_weight(e, meta, st) > 0]
    for split in SPLITS:
        check_total(csv_path, split, terms, meta, st)
        curves = [('total', st['loss_metrics'][split])]
        curves += [(e, st['term_metrics'][split].format(embedding=e)) for e in terms]
        for name, column in curves:
            plot_curve(
                ax,
                pvc.read_metric(csv_path, column),
                st,
                color=colors[name],
                linestyle=styles[split],
            )
    ax.set_ylabel(texts['ylabel'])

    formula = texts['formula_join'].join(
        texts['formula_term'].format(
            w=term_weight(e, meta, st), symbol=texts['symbol'][e]
        )
        for e in terms
    )
    width = float(st['linewidth'])
    handles = [
        Line2D(
            [],
            [],
            color=colors['total'],
            lw=width,
            label=texts['total'].format(formula=formula),
        )
    ]
    handles += [
        Line2D([], [], color=colors[e], lw=width, label=texts['term'][e]) for e in terms
    ]
    handles += [
        Line2D([], [], color=ink, ls=styles[s], lw=width, label=texts[s])
        for s in SPLITS
    ]
    return handles


def plot_loss_and_metric(
    csv_path: Path, meta: dict, st: dict, texts: dict, metric: str, ax
) -> list:
    """Losses on ``ax``, ``metric`` of every trained embedding on a twin right axis; returns the handles."""
    colors, styles = st['colors'], st['linestyles']
    right = ax.twinx()
    handles = []
    for split in SPLITS:
        s = pvc.read_metric(csv_path, st['loss_metrics'][split])
        handles += plot_curve(
            ax, s, st, color=colors[split], linestyle=styles[split], label=texts[split]
        )
    for emb in st['embeddings']:
        if not is_trained(emb, meta, st):
            continue
        s = pvc.read_metric(
            csv_path, st['embedding_metric'].format(embedding=emb, metric=metric)
        )
        label = texts['embedding'][emb].format(dim=meta['dims'][emb])
        handles += plot_curve(
            right, s, st, color=colors[emb], linestyle=styles['right'], label=label
        )
    ax.set_ylabel(texts['ylabel_left'])
    right.set_ylabel(texts['ylabel_right'])
    if st.get('ylim_right'):
        right.set_ylim(*st['ylim_right'])
    # The twin is drawn above the first axes; swap them so the loss curves stay on top.
    ax.set_zorder(right.get_zorder() + 1)
    ax.patch.set_visible(False)
    return handles


def plot_figure(
    key: str, kind: str, run_dir: Path, meta: dict, st: dict, out_path: Path
) -> None:
    texts = st['texts'][kind]
    csv_path = pvc.find_metrics(run_dir)
    fig, ax = plt.subplots(figsize=tuple(st['figsize']))
    if kind == LOSS_TERMS:
        handles = plot_loss_terms(csv_path, meta, st, texts, ax)
    else:
        handles = plot_loss_and_metric(csv_path, meta, st, texts, kind, ax)

    ax.set_xlabel(st['texts']['xlabel'])
    title = texts['title'].format(run=st['texts']['run'][key])
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    if st.get('ylim_left'):
        ax.set_ylim(*st['ylim_left'])

    legend = st['legend']
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
    parser.add_argument('--config', type=Path, required=True, help='YAML config')
    parser.add_argument('--output', type=Path, help='Override pretrain_curves.out_dir')
    args = parser.parse_args()

    pc = OmegaConf.to_container(
        OmegaConf.load(args.config)['pretrain_curves'], resolve=True
    )
    out_dir = args.output or Path(pc['out_dir'])
    for key, run_dir in pc['runs'].items():
        meta = run_meta(Path(run_dir), pc)
        print(f'{key}: loss weights {meta["weights"]}, embedding sizes {meta["dims"]}')
        for kind, fig_cfg in pc['figures'].items():
            st = {**pc, **fig_cfg}  # a figure entry overrides the top-level style keys
            out_path = out_dir / fig_cfg['filename'].format(key=key)
            plot_figure(key, kind, Path(run_dir), meta, st, out_path)
            print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
