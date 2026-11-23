# Early runs (2026-03 → 2026-07)

**Status:** closed; exploratory, superseded by everything after it.

## Question
First pretraining / fine-tuning / from-scratch tests of the fm4tag framework (SAINT-like column and row-column transformers), the DDP and intersample-attention issues found there, jet-only models.

## Data
Various early slices of the ttbar samples (see the hydra configs inside `outputs/`).

## Model
Early column / row-column transformers (`model_0_col`, `model_0_colrow`); 9.7 M → 2.3 M parameters after the July size reduction. Code: git history before tag `exp/lambda-scan-final`.

## Training methods
Contrastive + denoising pretraining (SAINT structure), fine-tuning with frozen / unfrozen encoders, training from scratch, CE.

## Stages and configs
Launchers in `slurm/` (`pretrain.sh`, `finetune.sh`, `jet_only/*`); they now write to `outputs/{pretraining,finetuning,jet_only}/run_<TS>`.

## Results
No headline result kept; history in `paper/factsheets/history_presentations.md`.

## Folder layout
| Folder | Content | In git |
|---|---|---|
| `configs/` | cells files (one per stage) and the YAML configs of the plot / analysis scripts | yes |
| `slurm/` | launchers specific to this experiment (generic ones are in the repo's `slurm/` and `scripts/`) | yes |
| `notes/`, `diagnostics/` | design notes, reports, one-off diagnostic scripts (where present) | yes |
| `outputs/` | run folders (`<STAGE>_<timestamp>/<cell>/seed_<s>/`: `.hydra/`, `metrics.csv`, checkpoints, `run_info.yaml`) and logs | no |
| `plots/` | every figure and metric table of the experiment | no |

## Related experiments
Led to the CE + JC grids (`2026-07_ce-jc-grids`) and the λ scan (`2026-08_lambda-scan`).

`outputs/hydra_outputs/` = the former root `outputs/` (hydra outputs of ad-hoc runs); `outputs/old_tests/`, `outputs/finetuning/` = the former `slurm/old_tests`, `slurm/finetuning`.
