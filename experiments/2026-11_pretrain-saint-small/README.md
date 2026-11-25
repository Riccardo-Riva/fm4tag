# P0 and P1 pretrainings of SAINT-small (2026-11-25 →)

**Status:** planned: measuring GPU memory and time per step at batch 12800 before the launch.

## Question
First contrastive pretrainings of the small model that replaces model B (`saint_reduced.yaml`, 3.56 M parameters,
close to GN2's 2.24 M): P0 (encoders only) and P1 (encoders + aggregator), to be fine-tuned and compared with the same
model trained from scratch.

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

## Training methods
- **P0:** per-object contrastive loss (InfoNCE, T 0.1) between the embeddings of an augmented view (CutMix + Mixup at
  noise λ 0.4) and the clean pass, on the track and the jet-feature embeddings. The aggregator is not trained.
- **P1:** P0 + jet-level contrastive (weight 1.0, T 0.1) on the aggregator output. The aggregator is trained and
  loaded downstream.
- **Protocol** (P_v1's): batch 12800, AdamW lr 1e-3, gradient-norm clipping 1.0, no dropout; WSD with warmup 4 epochs,
  constant, linear decay over the last 20 %; no early stopping; the downstream loads the last checkpoint. Epochs and
  seeds: see the launch command.

## Stages and configs
| Stage | What |
|---|---|
| `P_small_sweep_<job>` | peak GPU memory and s/step of both cells at batch 12800 (`scripts/dropout_batch_sweep.py`) |
| `P_small_<ts>` | the pretrainings (`configs/P_small.cells`, `CONFIG=saint_reduced`) |

## Results
Pending.

## Folder layout
| Folder | Content | In git |
|---|---|---|
| `configs/` | `P_small.cells` | yes |
| `slurm/` | (none specific; uses `slurm/ssl_pack.sh`, `slurm/dropout_batch_sweep.sh`) | yes |
| `outputs/` | the measurement, stage folders, logs | no |
| `plots/` | pretraining curves | no |

## Related experiments
- `2026-11_pretrain-v1`: the same recipe on model B, cancelled on 2026-11-25 after 1 h 40 min.
- `2026-11_four-way-jc-dropout`: model B from scratch (CE / CE+JC / CE+dropout / CE+JC+dropout), running.
- GN2 reference: salt `run_20261106_092332` (inspection doc §14).
