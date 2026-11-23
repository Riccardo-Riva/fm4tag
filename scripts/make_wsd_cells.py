"""Write the cells of the WSD redo of the downstream campaign (rriva, 2026-11-16).

Every training of the dropout comparison again, with ONE batch size for all (the contrastive loss depends on it),
the warmup-stable-decay schedule and no early stopping:

    8 arms   sc, scjc (scratch, CE / CE + JC), ft_p0, ft_p1, ftjc_p1 (fine-tuned), fr_p0, fr_p1, frjc_p1 (frozen)
    x 2      no dropout / dropout 0.2 on every dropout site (cell suffix _d02)

``--kind lr``: the learning-rate search, three values a factor 2 apart around 7.5e-4 (the sqrt-B scaling of every
arm's previous lr to batch 12800 lands there), the fine-tuned arms with the pretrained parts at lr / 10 and lr / 30
(suffix _bb10 / _bb30): 66 cells, on the seed-42 pretrainings, to be run with SEEDS=42 (kept out of the finals).
``--kind final``: one cell per (arm, dropout, pretraining seed) at the lr given by ``--choice`` (csv: cell,lr,bb).

Per-cell ``@est`` (seconds per epoch) and ``@mem`` (GiB of GPU memory, for shared cards: docs/gpu_packing.md) come
from ``--measured`` (csv: arm,dropout,est,mem); without it the cells carry neither.

    python scripts/make_wsd_cells.py --kind lr --out experiments/2026-11_wsd-lr-search/configs/W_lr.cells
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

CAMPAIGN = '/storage3/DSIP/rriva/research/fm4tag/experiments/2026-10_ssl-campaign/outputs'  # where the P_full pretrainings live
PRETRAINED = {  # no-dropout pretrainings of P_full.cells
    'p0': f'{CAMPAIGN}/P_full_20261023_085735/p0/seed_{{seed}}',
    'p1': f'{CAMPAIGN}/P_full_20261026_094445/p1/seed_{{seed}}',
}
BATCH = 12800  # the largest that fits CE + JC with dropout (37.8 GiB), 25 x chunk 512
LRS = {'lr38': 3.75e-4, 'lr75': 7.5e-4, 'lr150': 1.5e-3}
BACKBONE = {'bb10': 10, 'bb30': 30}
# arm -> (hydra arm, pretraining or None, JC downstream)
ARMS = {
    'ftjc_p1': ('finetune', 'p1', True),
    'scjc': ('scratch', None, True),
    'ft_p1': ('finetune', 'p1', False),
    'ft_p0': ('finetune', 'p0', False),
    'sc': ('scratch', None, False),
    'frjc_p1': ('frozen', 'p1', True),
    'fr_p1': ('frozen', 'p1', False),
    'fr_p0': ('frozen', 'p0', False),
}
COMMON = (
    'encoders=rowcol_concat_v0 chunk_size=512 '
    f'datamodule.batch_size={BATCH} '
    # warmup 1.6 epochs (2% of the 80-epoch finals) for any run length, stable, linear decay over the last 20%
    'finetune.schedule=wsd finetune.warmup_epochs=1.6 finetune.decay_frac=0.2 '
    # no early stopping: the best-val-CE checkpoint of the full run is kept
    '~callbacks.early_stopping ~callbacks_finetune.early_stopping'
)
DROPOUT = ' '.join(
    f'{key}=0.2'
    for key in (
        'encoders.global_encoder.layers.0.attn_dropout',
        'encoders.global_encoder.layers.0.ff_dropout',
        'encoders.constituents.tracks.layers.0.attn_dropout',
        'encoders.constituents.tracks.layers.0.ff_dropout',
        'aggregator.attn_dropout',
        'aggregator.ff_dropout',
        'head.mlp_dropout',
    )
)


def cell(arm: str, dropout: bool, lr: float, bb: int | None, seed: int) -> str:
    hydra_arm, pre, jc = ARMS[arm]
    parts = [f'arm={hydra_arm}']
    if pre:
        parts.append(f'encoder_ckpt=@{PRETRAINED[pre].format(seed=seed)}')
    if jc:
        parts.append('finetune.loss_weights.jet_contrastive=0.3')
    parts.append(COMMON)
    parts.append(f'optimizer.lr={lr:g}')
    if hydra_arm == 'finetune':
        parts.append(f'optimizer.backbone_lr={lr / bb:.4g}')
    if arm == 'scjc':
        parts.append(
            '+trainer.gradient_clip_val=1.0'
        )  # from scratch + JC collapses without it
    if dropout:
        parts.append(DROPOUT)
    return ' '.join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--kind', choices=['lr', 'final'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--measured', type=Path, help='csv: arm,dropout,est,mem')
    parser.add_argument('--choice', type=Path, help='csv: cell,lr,bb (kind final)')
    args = parser.parse_args()

    measured = {}
    if args.measured:
        for row in csv.DictReader(args.measured.open()):
            measured[row['arm'], row['dropout'] == '1'] = (
                f' @est={int(float(row["est"]))} @mem={int(float(row["mem"]))}'
            )

    lines = []
    if args.kind == 'lr':
        for arm, (hydra_arm, _, _) in ARMS.items():
            for dropout in (True, False):
                for bb_tag, bb in (
                    BACKBONE.items() if hydra_arm == 'finetune' else [('', None)]
                ):
                    for lr_tag, lr in LRS.items():
                        name = '_'.join(
                            x
                            for x in (arm, 'd02' if dropout else '', lr_tag, bb_tag)
                            if x
                        )
                        lines.append(
                            f'{name}  {cell(arm, dropout, lr, bb, 42)}'
                            f'{measured.get((arm, dropout), "")}'
                        )
    else:
        if not args.choice:
            parser.error('--kind final needs --choice')
        chosen = {r['cell']: r for r in csv.DictReader(args.choice.open())}
        for arm, (hydra_arm, pre, _) in ARMS.items():
            for dropout in (True, False):
                base = arm + ('_d02' if dropout else '')
                row = chosen[base]
                bb = int(row['bb']) if hydra_arm == 'finetune' else None
                for seed in (42, 424) if pre else (None,):
                    name = base if seed is None else base.replace(pre, f'{pre}s{seed}')
                    lines.append(
                        f'{name}  {cell(arm, dropout, float(row["lr"]), bb, seed)}'
                        f'{measured.get((arm, dropout), "")}'
                    )
    header = [
        f'# GENERATED by scripts/make_wsd_cells.py --kind {args.kind} (see its docstring): batch {BATCH} for every',
        '# cell, WSD schedule (warmup 1.6 epochs, linear decay over the last 20% of the run), no early stopping.',
        '# Longest cells first.',
    ]
    args.out.write_text('\n'.join(header + lines) + '\n')
    print(f'wrote {args.out}: {len(lines)} cells')


if __name__ == '__main__':
    main()
