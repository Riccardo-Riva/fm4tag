# CE + jet-contrastive grids (2026-07 → 2026-08)

**Status:** closed.

## Question
Which learning rate and which weight of the jet-level contrastive term (JC) work when training from scratch with CE + JC; concat vs per_token row attention; first noise-level (λ) sweeps.

## Data
`data_parallel_preprocess_251112_112022`: train 52.0 M / val 5.0 M / test 5.0 M jets (ttbar, 20 < pT < 250 GeV, |eta| < 2.5).

## Model
**Model A (`encoders=rowcol_v0`, 2.2 M parameters).** Per-feature tokens of width 16 for the jet (2 features) and
for each track (19 features); one RowCol stack of depth 6 per object type: column attention across the feature tokens
of one object (2 heads x 8) and **row (intersample) attention through ISAB** (16 learned inducing points per feature
token, `row_mode: per_token`) over the rank-local batch; projectors to 32 (jet) and 128 (tracks).
`TransformerAggregator`: a 6-layer transformer over the valid tracks of each jet (4 heads x 64, GEGLU x4), mean pool;
jet vector z_jet = [z_g, pooled z_t] (160 dims) -> MLP head (hidden 256, dropout 0.02) -> 4 classes (b, c, light, tau). (also concat variants).

## Training methods
From scratch, loss = CE + w_JC x L_JC: CE on the clean pass; L_JC = InfoNCE (MultiViewSupCon, T 0.1) between z_jet of an augmented view (CutMix + Mixup at noise λ) and the clean one; DDP 2 x 1024 jets; AdamW 3e-4, cosine, early stopping on val CE; unweighted CE.

## Stages and configs
Launcher `slurm/classify_from_scratch.sh` → `outputs/classify_from_scratch/run_<TS>/`; guide `docs/param_scan_guide.md`.

## Results
lr 3e-4 and w_JC 0.3 kept; JC needs a noise level λ (with the old λ definition the contrastive task collapsed). The ROC comparisons of these runs vs GN2 are in `2026-08_lambda-scan/configs/roc_comparison_run_*.yaml`.

## Folder layout
| Folder | Content | In git |
|---|---|---|
| `configs/` | cells files (one per stage) and the YAML configs of the plot / analysis scripts | yes |
| `slurm/` | launchers specific to this experiment (generic ones are in the repo's `slurm/` and `scripts/`) | yes |
| `notes/`, `diagnostics/` | design notes, reports, one-off diagnostic scripts (where present) | yes |
| `outputs/` | run folders (`<STAGE>_<timestamp>/<cell>/seed_<s>/`: `.hydra/`, `metrics.csv`, checkpoints, `run_info.yaml`) and logs | no |
| `plots/` | every figure and metric table of the experiment | no |

## Related experiments
Continued by `2026-08_lambda-scan`.
