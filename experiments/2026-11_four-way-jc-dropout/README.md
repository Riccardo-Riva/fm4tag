# Four-way test: CE / CE + JC / CE + dropout / CE + JC + dropout (2026-11-23 →)

**Status:** running (see `outputs/`).

**2026-11-25, t4 (CE + JC + dropout) stopped early.** The 2 seeds on L40S (job 7385104: 4242, 42424) ran to
epoch 80. Best val CE 0.737 and 0.786, last 1.23 and 1.21: they fail.

The owner cancelled the other 4 seeds (jobs 7385105/106 on A40: 424242, 4242424, 42424242, 424242424) at epoch 35.
- Best val CE 0.79–0.84, last 0.94–1.02.
- Their curves stop at epoch 35.
- Their queue tasks are marked `.failed`, with a `.cancelled` note, so no worker restarts them.

## Question
Is the gain of JC seen in the λ scan a contrastive effect, or the regularisation that dropout gives as well?

## Data
`data_parallel_preprocess_251112_152051`: pretrain 39.68 M (labels unused) / train 18.6 M / val 1.86 M / test 1.86 M jets, disjoint by event number (duplicate check of 2026-10-28: no jet shared between these splits; about 2 % of training rows are resampled copies, 24 % for tau).

## Model
**Model B (`encoders=rowcol_concat_v0`, `chunk_size=512`, 7.47 M parameters).** As model A, but the row
(intersample) attention is **full softmax attention among the objects of the same 512-jet chunk** over the flattened
token grid (`row_mode: concat`, SAINT style), with **QK RMS-norm** (added 2026-10-22 against logit blow-ups) and a
zero-initialised out-projection.  Jets: attention among the 512 jets of a chunk; tracks: among all valid tracks of
those jets (about 3.7k).  Same aggregator (6 layers over each jet's tracks, mean pool) and MLP head (hidden 256).

## Training methods
**Protocol v1**: from scratch, batch 12800, WSD 80 epochs (warmup 4 epochs = 5 %, constant, linear decay over epochs 64–80), no early stopping, best-val-CE checkpoint, lr 1e-3 (backbone lr/10 for runs with pretrained parts), gradient clip 1.0, **class-weighted CE** (b 1.5, c 1.5, light 1.0, tau 4.0, as GN2). Cells: t1 CE (every dropout 0, head included); t2 CE + JC (0.3, λ 0.4, T 0.1); t3 CE + dropout 0.2 everywhere; t4 CE + JC + dropout. Seeds 4242, 42424, 424242, 4242424, 42424242, 424242424, paired across cells.

## Stages and configs
`configs/T4way.cells` (launch commands in its header): JC cells on full cards (`T4way_full_*`), CE cells two per L40S card (`T4way_shared_*`).

## Results
Pending. The class-weighted val CE is directly comparable with the GN2 runs on the same data (salt `run_20261106_092332`, 8 seeds, 0.6137–0.6154).

## Folder layout
| Folder | Content | In git |
|---|---|---|
| `configs/` | cells files (one per stage) and the YAML configs of the plot / analysis scripts | yes |
| `slurm/` | launchers specific to this experiment (generic ones are in the repo's `slurm/` and `scripts/`) | yes |
| `notes/`, `diagnostics/` | design notes, reports, one-off diagnostic scripts (where present) | yes |
| `outputs/` | run folders (`<STAGE>_<timestamp>/<cell>/seed_<s>/`: `.hydra/`, `metrics.csv`, checkpoints, `run_info.yaml`) and logs | no |
| `plots/` | every figure and metric table of the experiment | no |

## Related experiments
Tests the claim of `2026-08_lambda-scan`; protocol from `2026-11_wsd-lr-search`; FT / FR variants not included yet.
