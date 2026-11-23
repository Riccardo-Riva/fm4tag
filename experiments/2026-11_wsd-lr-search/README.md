# WSD protocol and learning-rate search (2026-11-16 → 2026-11-23)

**Status:** closed; defines protocol v1.

## Question
With one batch size for all arms (12800), a warmup-stable-decay schedule and no early stopping, which learning rate does each case want, and does the FT←P0 + dropout gain survive lr tuning?

## Data
`data_parallel_preprocess_251112_152051`: pretrain 39.68 M (labels unused) / train 18.6 M / val 1.86 M / test 1.86 M jets, disjoint by event number (duplicate check of 2026-10-28: no jet shared between these splits; about 2 % of training rows are resampled copies, 24 % for tau).

## Model
**Model B (`encoders=rowcol_concat_v0`, `chunk_size=512`, 7.47 M parameters).** As model A, but the row
(intersample) attention is **full softmax attention among the objects of the same 512-jet chunk** over the flattened
token grid (`row_mode: concat`, SAINT style), with **QK RMS-norm** (added 2026-10-22 against logit blow-ups) and a
zero-initialised out-projection.  Jets: attention among the 512 jets of a chunk; tracks: among all valid tracks of
those jets (about 3.7k).  Same aggregator (6 layers over each jet's tracks, mean pool) and MLP head (hidden 256).

## Training methods
16 cases (SC, FT←P0, FT←P1, FR←P0, FR←P1 with CE; SC, FT←P1, FR←P1 with CE + JC; each without / with dropout 0.2) x lr 3.75e-4 / 7.5e-4 / 1.5e-3 (fine-tuned: backbone lr/10 and lr/30) = 66 runs, 40 epochs, WSD (warmup 1.6 epochs, linear decay over the last 20 %), best-val checkpoint, seed 42 on the seed-42 pretrainings; unweighted CE. Memory / speed measured alone and two to three per card (`W_measure`, `W_share_measure`).

## Stages and configs
Cells `configs/W_lr.cells` (written by `scripts/make_wsd_cells.py`); plot config `configs/wsd_lr_search.yaml` (`scripts/plot_lr_search.py`); `slurm/share_measure.sh`.

## Results
Most cases flat in lr (best val CE within ~0.003 = seed noise); the new protocol is as good or better than the old per-arm one in every case; FT←P0 + dropout 0.608–0.610 at all six settings vs 0.6305 for the best tuned no-dropout FT←P0; without dropout SC / FT←P0 / FR←P0 peak at epochs 5–7 and then overfit. `plots/wsd_lr_search/selection.csv` = best lr per case.

## Folder layout
| Folder | Content | In git |
|---|---|---|
| `configs/` | cells files (one per stage) and the YAML configs of the plot / analysis scripts | yes |
| `slurm/` | launchers specific to this experiment (generic ones are in the repo's `slurm/` and `scripts/`) | yes |
| `notes/`, `diagnostics/` | design notes, reports, one-off diagnostic scripts (where present) | yes |
| `outputs/` | run folders (`<STAGE>_<timestamp>/<cell>/seed_<s>/`: `.hydra/`, `metrics.csv`, checkpoints, `run_info.yaml`) and logs | no |
| `plots/` | every figure and metric table of the experiment | no |

## Related experiments
Uses `P_full_*` of `2026-10_ssl-campaign`; protocol v1 used by `2026-11_four-way-jc-dropout`.
