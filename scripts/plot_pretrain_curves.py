"""Training / validation loss and the effective rank of the embeddings, epoch by epoch, for pretraining runs.

The pretraining counterpart of ``plot_ce_and_embedding.py`` (same layout: losses on the left axis, an embedding
metric on a twin right axis), for an explicit list of runs instead of a pool picked from the lambda scan.  For each
entry of ``pretrain_curves.runs`` one figure is written:

  left axis   ``train_metric`` (epoch mean) and ``val_metric`` of every epoch;
  right axis  ``val/<embedding>/<right_metric>`` of every embedding listed for the run, one dashed curve each.

Every word on the figures comes from ``pretrain_curves.texts``; a missing key is an error, and nothing in this file
puts words on a figure.

Usage::

    python scripts/plot_pretrain_curves.py --config scripts/configs/pretrain_curves.yaml
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

sys.path.insert(0, str(Path(__file__).parent))
import plot_val_curves as pvc  # noqa: E402 -- run + metric readers


def plot_run(key: str, run: dict, pc: dict, out_path: Path) -> None:
    """One figure: the two losses on the left axis, one rank curve per embedding on the right."""
    texts, colors, styles = pc['texts'], pc['colors'], pc['linestyles']
    width, marker = float(pc['linewidth']), pc.get('marker')
    offset = int(pc.get('epoch_offset', 0))
    csv_path = pvc.find_metrics(Path(run['dir']))

    fig, ax = plt.subplots(figsize=tuple(pc['figsize']))
    right = ax.twinx()
    handles = []
    for name, column in (('train', pc['train_metric']), ('val', pc['val_metric'])):
        s = pvc.read_metric(csv_path, column)
        handles += ax.plot(
            s['epoch'] + offset,
            s['value'],
            color=colors[name],
            linestyle=styles[name],
            linewidth=width,
            marker=marker,
            markersize=4,
            label=texts[name],
        )
    for emb in run['embeddings']:
        s = pvc.read_metric(csv_path, f'val/{emb}/{pc["right_metric"]}')
        handles += right.plot(
            s['epoch'] + offset,
            s['value'],
            color=colors[emb],
            linestyle=styles['right'],
            linewidth=width,
            marker=marker,
            markersize=4,
            label=texts['embedding'][emb],
        )

    ax.set_xlabel(texts['xlabel'])
    ax.set_ylabel(texts['ylabel_left'])
    right.set_ylabel(texts['ylabel_right'])
    title = texts['title'][key]
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    # The twin is drawn above the first axes; swap them so the loss curves stay on top.
    ax.set_zorder(right.get_zorder() + 1)
    ax.patch.set_visible(False)
    if pc.get('ylim_left'):
        ax.set_ylim(*pc['ylim_left'])
    if pc.get('ylim_right'):
        right.set_ylim(*pc['ylim_right'])

    legend = pc['legend']
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
    for key, run in pc['runs'].items():
        out_path = out_dir / pc['filename'].format(key=key)
        plot_run(key, run, pc, out_path)
        print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
