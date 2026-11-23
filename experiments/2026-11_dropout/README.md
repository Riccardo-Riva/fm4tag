# Dropout downstream and in pretraining (2026-11-12 → 2026-11-22)

**Status:** closed.

## Question
Does dropout (0.2 everywhere downstream; 0.1 in pretraining) reduce the strong overfitting and change the P0 vs P1 comparison?

## Data
`data_parallel_preprocess_251112_152051`: pretrain 39.68 M (labels unused) / train 18.6 M / val 1.86 M / test 1.86 M jets, disjoint by event number (duplicate check of 2026-10-28: no jet shared between these splits; about 2 % of training rows are resampled copies, 24 % for tau).

## Model
**Model B (`encoders=rowcol_concat_v0`, `chunk_size=512`, 7.47 M parameters).** As model A, but the row
(intersample) attention is **full softmax attention among the objects of the same 512-jet chunk** over the flattened
token grid (`row_mode: concat`, SAINT style), with **QK RMS-norm** (added 2026-10-22 against logit blow-ups) and a
zero-initialised out-projection.  Jets: attention among the 512 jets of a chunk; tracks: among all valid tracks of
those jets (about 3.7k).  Same aggregator (6 layers over each jet's tracks, mean pool) and MLP head (hidden 256).

## Training methods
D_drop: the six P0 vs P1 downstream cases (SC, FT←P0, FR←P0, SC+JC, FT+JC←P1, FR+JC←P1) with dropout 0.2 in every encoder / aggregator attention and FF and in the head (frozen: aggregator + head only), 80-epoch cosine, lrs unchanged, batch reduced where needed (sweep), 4 runs each. P_drop: P0, P1, P2 pretrained again with dropout 0.1 (P1 at batch 10752), seeds 42 / 424 (P2 s424 collapsed → s4242; P1 resumed after the node81 cancel, see `outputs/P_drop_*/CHANGES.txt`).

## Stages and configs
Cells in `configs/` (`D_drop*`, `P_drop*`); `slurm/launch_drop_stage.sh`, `slurm/launch_pretrain_drop_stage.sh` (chained after the generic `slurm/dropout_batch_sweep.sh`); plot configs `configs/dropout_*.yaml`.

## Results
FT←P0 + dropout best (val CE 0.6105, test c-rej 12.1 = 1.25 x SC at the same light / tau); every JC + dropout case fails (0.74–0.79); dropout pretraining did not improve the pretraining objective (P_drop never used downstream yet).

## Folder layout
| Folder | Content | In git |
|---|---|---|
| `configs/` | cells files (one per stage) and the YAML configs of the plot / analysis scripts | yes |
| `slurm/` | launchers specific to this experiment (generic ones are in the repo's `slurm/` and `scripts/`) | yes |
| `notes/`, `diagnostics/` | design notes, reports, one-off diagnostic scripts (where present) | yes |
| `outputs/` | run folders (`<STAGE>_<timestamp>/<cell>/seed_<s>/`: `.hydra/`, `metrics.csv`, checkpoints, `run_info.yaml`) and logs | no |
| `plots/` | every figure and metric table of the experiment | no |

## Related experiments
Uses the `P_full_*` pretrainings of `2026-10_ssl-campaign`; its confounds (schedule, batch, lr) motivated `2026-11_wsd-lr-search`.
