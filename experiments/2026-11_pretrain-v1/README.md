# P0 and P1 pretrainings with the protocol-v1 settings (2026-11-23 →)

**Status:** cancelled on 2026-11-25 by the owner (job 7387250, 1 h 40 min after its start): with the move to a smaller model, a 4-day pretraining of model B is not the next step. Successor: `2026-11_pretrain-saint-small`.

## Question
Pretrain P0 and P1 again with the same settings as the downstream protocol v1 (batch 12800, WSD, 80 epochs, no early
stopping), so that pretraining and downstream follow one protocol. Do longer, WSD-scheduled pretrainings give
better starting points than the 30-epoch cosine ones (`2026-10_ssl-campaign/outputs/P_full_*`)?

## Data
`data_parallel_preprocess_251112_152051`, the pretrain split (39.68 M jets, labels unused). Validation on the val split
(1.86 M). The splits share no jet (duplicate check of 2026-10-28).

## Model
**Model B (`encoders=rowcol_concat_v0`, `chunk_size=512`, 7.47 M parameters)**, the same as the SSL campaign:
- per-feature tokens of width 16 for the jet (2 features) and each track (19 features);
- one RowCol stack of depth 6 per object type:
  - column attention across the feature tokens of one object;
  - row (intersample) attention: full softmax attention among the objects of the same 512-jet chunk over the
    flattened token grid (`row_mode: concat`), with QK RMS-norm;
- projectors to 32 (jet) and 128 (tracks);
- `TransformerAggregator`: 6 layers over each jet's valid tracks, mean pool.

There is no classification head in pretraining.

## Training methods
- **P0:** per-object contrastive loss (InfoNCE, T 0.1) between the embeddings of an augmented view (CutMix + Mixup at
  noise λ 0.4) and the clean pass, on the track and the jet-feature embeddings. The aggregator is not trained.
- **P1:** P0 + jet-level contrastive (weight 1.0, T 0.1) on the aggregator output. The aggregator is trained and
  loaded downstream.
- **Protocol:**
  - batch 12800, AdamW lr 1e-3, gradient-norm clipping 1.0, no dropout;
  - WSD over 80 epochs: warmup 4 epochs (5 %), constant, linear decay over epochs 64–80 (20 %);
  - no early stopping; the downstream loads the last checkpoint.
- **Seeds:** see the launch command. The downstream uses the seed with the lowest pretraining val loss.

Differences from `P_full`: 80 instead of 30 epochs, WSD instead of warmup 2 % + cosine, batch 12800 instead of 12288,
no early stopping (P_full had it, but it never triggered).

## Stages and configs
| Stage | What |
|---|---|
| `P_v1_s42_<ts>`, `P_v1_s424_<ts>` | the pretrainings, one stage per seed (`configs/P_v1.cells`; launch command in its header) |

One pretraining per GPU: at batch 12800 a P0 needs about 34 GiB and a P1 about 42 GiB of the 44.4 GiB card (scaled
from the P_full peaks at 12288: 32.3 and 40.8 GiB), so two cannot share a card. A 2-GPU job runs one P0 and one P1.
L40S only: 80 epochs of P1 take about 95 h there, but about 148 h on an A40, too close to the 167 h a 7-day job can
use. A truncated WSD run never reaches its decay.

## Results
Pending.

## Folder layout
| Folder | Content | In git |
|---|---|---|
| `configs/` | `P_v1.cells` | yes |
| `slurm/` | (none specific; uses `slurm/ssl_pack.sh`, `slurm/dropout_batch_sweep.sh`) | yes |
| `outputs/` | stage folders, logs | no |
| `plots/` | pretraining curves, comparisons with P_full | no |

## Related experiments
- `2026-10_ssl-campaign`: the `P_full` pretrainings (30-epoch cosine) these replace.
- `2026-11_wsd-lr-search`: the protocol v1.
- `2026-11_dropout`: `P_drop` (dropout 0.1, no gain on the pretraining objective).
- Downstream from these checkpoints: fine-tuned / frozen runs of protocol v1 (not yet planned in detail).
