# P0 / P1 / P2 pretrainings of small_RC, small_C and normal_C (2026-11-25 →)

**Status:** being prepared. The GPU memory of each cell is being estimated on CPU (`experiments/dev/2026-11-26_cpu-mem-estimate`);
then 3 seeds of each cell, several runs per card.

## Question
First contrastive pretrainings of three models, to be fine-tuned and compared with the same models trained from scratch:
- **small_RC** (SAINT-small), the small model that replaces model B, close to GN2 in size;
- **small_C**, small_RC without the encoder's row (intersample) steps: GN2's structure and size;
- **normal_C**, model B without the row steps.

Together with model B (normal_RC, pretrained in `2026-10_ssl-campaign`), they ask how much of what pretraining gives
lives in the row steps, whether that depends on the model's size, and whether pretraining helps a GN2-shaped model at all.

## Data
`data_parallel_preprocess_251112_152051`, the pretrain split (39.68 M jets, labels unused). Validation on the val split
(1.86 M). The splits share no jet (duplicate check of 2026-10-28).

## Models
One base config per model (`src/fm4tag/configs/`); RC = column + row steps in the encoder, C = column steps only.

| Model | Base | Encoder | Aggregator, head | Parameters with the head |
|---|---|---|---|---|
| normal_RC (model B) | `default` | `rowcol_concat_v0` | `transformer_v0`, `mlp` | 7.5 M |
| normal_C | `saint_normal_C` | `col_v0_normal` | `transformer_v0`, `mlp` | 2.2 M |
| small_RC | `saint_small_RC` | `rowcol_concat_v1_small` | `transformer_v1_small`, `mlp_v1_small` | 3.5 M |
| small_C | `saint_small_C` | `col_v1_small` | `transformer_v1_small`, `mlp_v1_small` | 2.1 M |

(`scripts/model_summary.py --phase finetune --config-name <base>`; each also holds 14 K frozen parameters of the
denoising heads. GN2: 2.24 M.)

- **small_RC** (owner, 2026-11-25; inspection doc §14):
  - encoder: model B's per-feature tokens of width 16 (jet: 2 features, track: 19), one RowCol stack of **depth 2** per
    object (model B: 6). Column attention across the feature tokens of one object; row attention among the objects of
    the same 512-jet chunk over the flattened token grid (`row_mode: concat`), **4 row heads** for the tracks (model B:
    8), QK RMS-norm. Projectors to 128 (jet) and **304 → 256 → 256** (tracks; model B: 304 → 228 → 128).
  - aggregator, shaped like GN2's transformer: 4 layers at **width 256** over each jet's valid tracks, 8 heads × 32,
    GEGLU feed-forward with hidden 256 (`ff_mult: 1`), then per track a projection 256 → 128 and a LayerNorm (GN2's
    `out_proj` + `out_norm`, in its order), mean pool.
  - head (fine-tuning only): GN2's 128 → 64 → 32, no dropout, on z_jet = 128 (jet) + 128 (tracks).
- **small_C** and **normal_C** keep only the column half of each encoder step (the same blocks, depth 2 and 6) and the
  same projectors. No track sees another track and no jet sees another jet in the encoder: tracks mix only in the
  aggregator, within their jet. The contrastive negatives still come from the 512-jet chunk (`chunk_size` sets the
  loss's negatives as well as the row attention's context).

Parameters each pretraining trains: P0 small_RC 1.62 M, small_C 0.17 M, normal_C 0.15 M (the per-feature embeddings,
the column layers and the projectors; the aggregator is not trained); P1 and P2 small_RC 3.50 M, small_C 2.05 M,
normal_C 2.13 M.

## Training methods
- **P0:** per-object contrastive loss (InfoNCE, T 0.1) between the embeddings of an augmented view (CutMix + Mixup at
  noise λ 0.4) and the clean pass, on the track and the jet-feature embeddings. The aggregator is not trained.
- **P1:** P0 + jet-level contrastive (weight 1.0, T 0.1) on the aggregator output. The aggregator is trained and
  loaded downstream.
- **P2:** jet-level contrastive only (per-object contrastive weight 0).
- **Protocol** (all three models): P_v1's, shortened to **40 epochs** (owner, 2026-11-25).
  - batch 12800, AdamW lr 1e-3, gradient-norm clipping 1.0, no dropout;
  - WSD: warmup 2 epochs (5 %), constant, linear decay over epochs 32–40 (20 %);
  - no early stopping; the downstream loads the last checkpoint.
- **Seeds:** 42, 424, 4242 for now (5 per setting in the end).

## Stages and configs
| Stage | What |
|---|---|
| `P_small_s42_20261125_231824_CANCELLED_not_started` | P0 / P1 of small_RC, seed 42 (pack 7473289), cancelled on 2026-11-26 before it started: replaced by the stages below |
| (next) | `configs/P_small_RC.cells` (`CONFIG=saint_small_RC`), `P_small_C.cells` (`saint_small_C`), `P_normal_C.cells` (`saint_normal_C`), 3 seeds each; `@mem` / `@est` from the CPU estimate |

Before launch, each cells file was composed with its base, built, and run for one loss + backward step on a toy batch on
CPU (`cells_loss_check.py` in the session scratchpad): all 9 cells give a finite loss; P1 and P2 send gradients to both
encoders and the aggregator; P0 leaves the aggregator frozen.

## Results
Pending.

## Folder layout
| Folder | Content | In git |
|---|---|---|
| `configs/` | `P_small_RC.cells`, `P_small_C.cells`, `P_normal_C.cells` | yes |
| `slurm/` | (none specific; uses `slurm/ssl_pack.sh`) | yes |
| `outputs/` | stage folders, logs | no |
| `plots/` | pretraining curves | no |

## Related experiments
- `2026-10_ssl-campaign`: P0 / P1 / P2 of model B (normal_RC), 30 epochs, batch 12288, cosine.
- `2026-11_pretrain-v1`: the same recipe on model B, cancelled on 2026-11-25 after 1 h 40 min.
- `2026-11_four-way-jc-dropout`: model B from scratch (CE / CE+JC / CE+dropout / CE+JC+dropout), running.
- GN2 reference: salt `run_20261106_092332` (inspection doc §14).
