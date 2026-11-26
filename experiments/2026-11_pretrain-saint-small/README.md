# P0 and P1 pretrainings of SAINT-small and of its variant without row steps (2026-11-25 →)

**Status:** seed 42 submitted on 2026-11-25 (stage `P_small_s42_20261125_231824`, pack job 7473289, pending). The
other 4 seeds follow once its first epoch has measured the GPU memory and the time per epoch.

## Question
First contrastive pretrainings of the small model that replaces model B (`saint_reduced.yaml`, 3.56 M parameters,
close to GN2's 2.24 M): P0 (encoders only) and P1 (encoders + aggregator), to be fine-tuned and compared with the same
model trained from scratch.

Variant B (owner, 2026-11-26) is the same model without the encoder's row steps. Comparing the two asks how much of
what pretraining gives lives in the row steps, and whether pretraining helps a GN2-shaped model at all (B has GN2's
structure and size).

## Data
`data_parallel_preprocess_251112_152051`, the pretrain split (39.68 M jets, labels unused). Validation on the val split
(1.86 M). The splits share no jet (duplicate check of 2026-10-28).

## Model
**SAINT-small (`CONFIG=saint_reduced`, `chunk_size=512`)**, chosen by the owner on 2026-11-25 (inspection doc §14):
- **encoder** `rowcol_concat_v1_small`: model B's per-feature tokens of width 16 (jet: 2 features, track: 19), one
  RowCol stack of **depth 2** per object (model B: 6):
  - column attention across the feature tokens of one object;
  - row (intersample) attention: full softmax attention among the objects of the same 512-jet chunk over the
    flattened token grid (`row_mode: concat`), **4 row heads** for the tracks (model B: 8), QK RMS-norm;
  - projectors to 128 (jet) and **304 → 256 → 256** (tracks; model B: 304 → 228 → 128).
- **aggregator** `transformer_v1_small`, shaped like GN2's transformer: 4 layers at **width 256** over each jet's valid
  tracks, 8 heads × 32, GEGLU feed-forward with hidden 256 (`ff_mult: 1`), then per track a projection 256 → 128 and a
  LayerNorm (GN2's `out_proj` + `out_norm`, in its order), mean pool.
- **head** `mlp_v1_small` (fine-tuning only): GN2's 128 → 64 → 32, no dropout, on z_jet = 128 (jet) + 128 (tracks).

Parameters (`scripts/model_summary.py --phase finetune --config-name saint_reduced`): track encoder 1.60 M (two rowcol
steps 1.43 M), jet encoder 0.04 M, aggregator 1.88 M, head 0.04 M; total 3.56 M (14 K of it the denoising heads, frozen).
The per-track contrastive embedding of pretraining is the track projector's output, now 256-wide (model B: 128).

**Variant B, no row steps (`encoder=col_v1_small`, same base, aggregator and head):**
- The encoder keeps only the column half of each step: attention across the feature tokens of one object (depth 2,
  1 head × 16, `ff_mult` 2, the same blocks as in SAINT-small) and the same projectors.
- No track sees another track and no jet sees another jet in the encoder. Tracks mix only in the aggregator, within
  their jet. The model is GN2-shaped: per-track embedding → 4-layer transformer over the jet's tracks → projection +
  norm → pool → head.
- Parameters: track encoder 0.17 M, jet encoder 0.02 M, aggregator 1.88 M, head 0.04 M; total 2.11 M (GN2: 2.24 M).
  In pretraining, P0 trains only 0.17 M of it (embeddings, column layers, projectors) and P1 trains 2.05 M.
- The contrastive negatives still come from the 512-jet chunk: `chunk_size` sets the loss's negatives as well as the
  row attention's context.

## Training methods
- **P0:** per-object contrastive loss (InfoNCE, T 0.1) between the embeddings of an augmented view (CutMix + Mixup at
  noise λ 0.4) and the clean pass, on the track and the jet-feature embeddings. The aggregator is not trained.
- **P1:** P0 + jet-level contrastive (weight 1.0, T 0.1) on the aggregator output. The aggregator is trained and
  loaded downstream.
- **Protocol:** P_v1's, shortened to **40 epochs** (owner, 2026-11-25).
  - batch 12800, AdamW lr 1e-3, gradient-norm clipping 1.0, no dropout;
  - WSD: warmup 2 epochs (5 %), constant, linear decay over epochs 32–40 (20 %);
  - no early stopping; the downstream loads the last checkpoint.
- **Seeds:** 5 per setting in the end. Seed 42 first, one pretraining per card. The other 4 seeds go two per card if a
  P0 and a P1 fit on one card together.

## Stages and configs
| Stage | What |
|---|---|
| `P_small_s42_20261125_231824` | seed 42 of both cells, one per card (`configs/P_small.cells`, `CONFIG=saint_reduced`; launch command in its header) |
| (next) | the other 4 seeds, from the memory and time measured in seed 42's first epoch |
| (not launched) | variant B, `configs/P_col.cells` (`p1_col`, `p0_col`), same recipe |

Before launch, both cells files were composed with `saint_reduced`, built, and run for one loss + backward step on a
toy batch on CPU (`cells_loss_check.py` in the session scratchpad):
- all 4 cells give a finite loss;
- P1 sends gradients to both encoders and the aggregator;
- P0 leaves the aggregator frozen.

A measurement job (7470396, `scripts/dropout_batch_sweep.py` at batch 12800) was submitted first but cancelled while it
was still waiting in the queue: seed 42's first epoch measures the same thing, and measuring first meant queueing twice.
The cells' `@est` (6650 s/epoch) is model B's P1 on an A40, a deliberate upper bound.

## Results
Pending.

## Folder layout
| Folder | Content | In git |
|---|---|---|
| `configs/` | `P_small.cells` (SAINT-small), `P_col.cells` (variant B) | yes |
| `slurm/` | (none specific; uses `slurm/ssl_pack.sh`, `slurm/dropout_batch_sweep.sh`) | yes |
| `outputs/` | the measurement, stage folders, logs | no |
| `plots/` | pretraining curves | no |

## Related experiments
- `2026-11_pretrain-v1`: the same recipe on model B, cancelled on 2026-11-25 after 1 h 40 min.
- `2026-11_four-way-jc-dropout`: model B from scratch (CE / CE+JC / CE+dropout / CE+JC+dropout), running.
- GN2 reference: salt `run_20261106_092332` (inspection doc §14).
