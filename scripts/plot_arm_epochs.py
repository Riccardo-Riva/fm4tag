"""Cross-entropy and embedding quality epoch by epoch, for every run of the compared training methods (arms).

Every completed run under ``arm_epochs.stages`` is put in an arm from its own ``.hydra`` config, never from its
directory name:

  encoder          ``scratch`` (no ``encoder_ckpt``), ``finetune`` (``encoder_ckpt`` set, encoders trained) or
                   ``frozen`` (``finetune.freeze_encoders``)
  pretraining      what the ``encoder_ckpt`` run optimised, from that run's own config: ``p0`` (contrastive on
                   the feature embeddings), ``p1`` (P0 + jet_contrastive), ``p2`` (jet_contrastive only)
  jet_contrastive  ``finetune.loss_weights.jet_contrastive`` of the run itself

Figures, one line per run, coloured by arm:

  ce_train_val      training CE (epoch mean, dashed) and validation CE (solid); optionally a copy with a
                    narrower y range (``ce.zoom_filename``, ``ce.zoom_ylim``)
  effective_rank    ``val/<embedding>/effective_rank``, one panel per embedding
  uniformity        ``val/<embedding>/uniformity``, one panel per embedding

plus ``run_selection.csv`` (the runs of each arm, their seed, pretraining run and best validation CE).  Every
word on the figures comes from ``arm_epochs.texts``; a missing key is an error.

Usage::

    python scripts/plot_arm_epochs.py --config scripts/configs/arm_epochs.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import yaml
from omegaconf import OmegaConf

# PNG only, never a GUI window (a dead X11 forward would hang plt.subplots()).
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402 -- after use()
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import plot_val_curves as pvc  # noqa: E402 -- run + metric readers


def run_config(run_dir: Path):
    """The newest resolved ``.hydra/config.yaml`` of a run."""
    configs = sorted(
        run_dir.glob('outputs/*/*/.hydra/config.yaml'), key=lambda p: p.stat().st_mtime
    )
    if not configs:
        raise FileNotFoundError(f'No .hydra/config.yaml under {run_dir}')
    return OmegaConf.load(configs[-1])


def pretraining_kind(encoder_ckpt: str) -> tuple[str, Path]:
    """``p0`` / ``p1`` / ``p2`` from the loss weights of the run that wrote ``encoder_ckpt``."""
    pre_dir = Path(encoder_ckpt).parents[3]  # <run>/<name>/version_0/checkpoints/<file>
    weights = run_config(pre_dir).pretrain.loss_weights
    contrastive, jet = float(weights.contrastive), float(weights.jet_contrastive)
    if contrastive > 0 and jet == 0:
        return 'p0', pre_dir
    if contrastive > 0 and jet > 0:
        return 'p1', pre_dir
    if contrastive == 0 and jet > 0:
        return 'p2', pre_dir
    raise ValueError(f'{pre_dir}: no contrastive term in its pretraining loss')


def classify(run_dir: Path) -> dict:
    """Arm-defining metadata of one downstream run, from its config."""
    cfg = run_config(run_dir)
    ckpt = cfg.get('encoder_ckpt')
    if ckpt:
        encoder = 'frozen' if cfg.finetune.freeze_encoders else 'finetune'
        pretraining, pre_dir = pretraining_kind(ckpt)
    else:
        encoder, pretraining, pre_dir = 'scratch', None, None
    return {
        'encoder': encoder,
        'pretraining': pretraining,
        'pretraining_run': str(pre_dir) if pre_dir else '',
        'jet_contrastive': float(cfg.finetune.loss_weights.jet_contrastive),
        'seed': int(cfg.seed),
    }


def select_runs(ac: dict) -> pd.DataFrame:
    """The completed runs of every arm in ``ac['arms']``, ``runs_per_arm`` each."""
    rows, skipped = [], 0
    for stage in ac['stages']:
        for run_dir in sorted(Path(stage).glob('*/seed_*')):
            info = yaml.safe_load((run_dir / 'run_info.yaml').read_text())
            if info.get('status') != 'completed':
                skipped += 1
                continue
            meta = classify(run_dir)
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
                skipped += 1
                continue
            val = pvc.read_metric(pvc.find_metrics(run_dir), ac['ce']['val_metric'])
            best = val.loc[val['value'].idxmin()]
            rows.append(
                {
                    'arm': arm,
                    'run_dir': str(run_dir),
                    **meta,
                    'best_val_ce': float(best['value']),
                    'best_epoch': int(best['epoch']) + int(ac['epoch_offset']),
                    'epochs_run': len(val),
                }
            )
    runs = pd.DataFrame(rows)
    counts = runs['arm'].value_counts() if len(runs) else pd.Series(dtype=int)
    for arm in ac['arms']:
        if counts.get(arm, 0) != int(ac['runs_per_arm']):
            raise ValueError(
                f'arm {arm}: {counts.get(arm, 0)} runs, expected {ac["runs_per_arm"]}'
            )
    print(
        f'{len(runs)} runs in {len(ac["arms"])} arms; {skipped} runs skipped (not completed or in no listed arm)'
    )
    order = {arm: i for i, arm in enumerate(ac['arms'])}
    return runs.sort_values(
        ['arm', 'seed'], key=lambda s: s.map(order) if s.name == 'arm' else s
    )


def arm_handles(ac: dict) -> list[Line2D]:
    """One legend entry per arm, drawn as a solid line in the arm's colour."""
    return [
        Line2D([], [], color=a['colour'], linewidth=2.0, label=ac['texts']['arm'][key])
        for key, a in ac['arms'].items()
    ]


def plot_ce(
    runs: pd.DataFrame,
    ac: dict,
    out_path: Path,
    ylim: list[float] | None,
    title: str,
) -> None:
    """Training (dashed) and validation (solid) CE of every run, one panel."""
    cc, texts = ac['ce'], ac['texts']
    offset = int(ac['epoch_offset'])
    fig, ax = plt.subplots(figsize=tuple(cc['figsize']))
    for run in runs.itertuples():
        csv_path = pvc.find_metrics(Path(run.run_dir))
        for split in ('train', 'val'):
            s = pvc.read_metric(csv_path, cc[f'{split}_metric'])
            ax.plot(
                s['epoch'] + offset,
                s['value'],
                color=ac['arms'][run.arm]['colour'],
                linestyle=ac['linestyles'][split],
                linewidth=float(ac['linewidth']),
                alpha=float(ac['alpha']),
            )
    split_handles = [
        Line2D(
            [],
            [],
            color=ac['ink'],
            linestyle=ac['linestyles'][split],
            linewidth=1.5,
            label=texts['ce'][split],
        )
        for split in ('val', 'train')
    ]
    ax.set_xlabel(texts['xlabel'])
    ax.set_ylabel(texts['ce']['ylabel'])
    if title:
        ax.set_title(title, fontsize=11)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=arm_handles(ac) + split_handles,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_embedding_metric(
    runs: pd.DataFrame, ac: dict, metric: str, out_path: Path
) -> None:
    """``val/<embedding>/<metric>`` of every run, one panel per embedding."""
    texts, embeddings = ac['texts'], ac['embeddings']
    offset = int(ac['epoch_offset'])
    fig, axes = plt.subplots(
        1, len(embeddings), figsize=tuple(ac['embedding_figsize']), sharex=True
    )
    for run in runs.itertuples():
        csv_path = pvc.find_metrics(Path(run.run_dir))
        for ax, emb in zip(axes, embeddings):
            s = pvc.read_metric(csv_path, f'val/{emb}/{metric}')
            ax.plot(
                s['epoch'] + offset,
                s['value'],
                color=ac['arms'][run.arm]['colour'],
                linewidth=float(ac['linewidth']),
                alpha=float(ac['alpha']),
            )
    for ax, emb in zip(axes, embeddings):
        ax.set_title(texts['panel'][emb], fontsize=10)
        ax.set_xlabel(texts['xlabel'])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(texts[metric]['ylabel'])
    if texts[metric]['title']:
        fig.suptitle(texts[metric]['title'], fontsize=11)
    fig.legend(
        handles=arm_handles(ac),
        loc='lower center',
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(ac['arms']),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=Path, required=True, help='YAML config')
    parser.add_argument('--output', type=Path, help='Override arm_epochs.out_dir')
    args = parser.parse_args()

    ac = OmegaConf.to_container(OmegaConf.load(args.config)['arm_epochs'], resolve=True)
    out_dir = args.output or Path(ac['out_dir'])
    runs = select_runs(ac)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_dir / 'run_selection.csv', index=False)
    print(
        runs[
            ['arm', 'seed', 'pretraining', 'best_val_ce', 'best_epoch', 'epochs_run']
        ].to_string(index=False)
    )

    cc, texts = ac['ce'], ac['texts']['ce']
    plot_ce(runs, ac, out_dir / cc['filename'], cc.get('ylim'), texts['title'])
    print(f'Saved {out_dir / cc["filename"]}')
    if cc.get('zoom_filename'):
        plot_ce(
            runs,
            ac,
            out_dir / cc['zoom_filename'],
            cc['zoom_ylim'],
            texts['zoom_title'],
        )
        print(f'Saved {out_dir / cc["zoom_filename"]}')
    for metric, filename in ac['embedding_figures'].items():
        plot_embedding_metric(runs, ac, metric, out_dir / filename)
        print(f'Saved {out_dir / filename}')


if __name__ == '__main__':
    main()
