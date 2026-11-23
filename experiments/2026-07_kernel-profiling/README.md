# Attention-kernel profiling (2026-07 → 2026-10)

**Status:** closed.

## Question
Speed and memory of the row-attention kernels (ISAB per_token vs concat, flash varlen, chunk sizes) and the S0 benchmark that chose chunk 512 for the SSL campaign.

## Data
Synthetic batches and the 152051 / 112022 training files.

## Model
**Model A (`encoders=rowcol_v0`, 2.2 M parameters).** Per-feature tokens of width 16 for the jet (2 features) and
for each track (19 features); one RowCol stack of depth 6 per object type: column attention across the feature tokens
of one object (2 heads x 8) and **row (intersample) attention through ISAB** (16 learned inducing points per feature
token, `row_mode: per_token`) over the rank-local batch; projectors to 32 (jet) and 128 (tracks).
`TransformerAggregator`: a 6-layer transformer over the valid tracks of each jet (4 heads x 64, GEGLU x4), mean pool;
jet vector z_jet = [z_g, pooled z_t] (160 dims) -> MLP head (hidden 256, dropout 0.02) -> 4 classes (b, c, light, tau).

**Model B (`encoders=rowcol_concat_v0`, `chunk_size=512`, 7.47 M parameters).** As model A, but the row
(intersample) attention is **full softmax attention among the objects of the same 512-jet chunk** over the flattened
token grid (`row_mode: concat`, SAINT style), with **QK RMS-norm** (added 2026-10-22 against logit blow-ups) and a
zero-initialised out-projection.  Jets: attention among the 512 jets of a chunk; tracks: among all valid tracks of
those jets (about 3.7k).  Same aggregator (6 layers over each jet's tracks, mean pool) and MLP head (hidden 256).

## Training methods
Short timed trainings (forward + backward), `torch.profiler` traces.

## Stages and configs
Launchers in `slurm/` (`submit_profile_timing.sh`, `submit_rowmode_final.sh`, `submit_flash_attn_verify.sh`, `submit_s0_benchmark.sh`); summaries with `scripts/profile_table.py`. `outputs/s0_*` = the S0 benchmark (code: tag `archive/s0-snapshot`).

## Results
concat measured 11–23 % faster than per_token at batch 1024–2048 (a different model); the 2.4x backward/forward ratio is flash tile padding, not a bug; chunk 512 chosen for the SSL campaign.

## Folder layout
| Folder | Content | In git |
|---|---|---|
| `configs/` | cells files (one per stage) and the YAML configs of the plot / analysis scripts | yes |
| `slurm/` | launchers specific to this experiment (generic ones are in the repo's `slurm/` and `scripts/`) | yes |
| `notes/`, `diagnostics/` | design notes, reports, one-off diagnostic scripts (where present) | yes |
| `outputs/` | run folders (`<STAGE>_<timestamp>/<cell>/seed_<s>/`: `.hydra/`, `metrics.csv`, checkpoints, `run_info.yaml`) and logs | no |
| `plots/` | every figure and metric table of the experiment | no |

## Related experiments
Feeds `2026-10_ssl-campaign` (S0, S1).
