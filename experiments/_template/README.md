# <Title> (<start date> →)

**Status:** planned | running | closed

## Question
One or two sentences: what will this experiment decide?

## Data
Dataset path and split sizes.

## Model
Architecture (encoders config, chunk size, parameter count), changes vs the previous experiment.

## Training methods
Pretraining / downstream methods, loss terms and weights, protocol (batch, schedule, epochs, stopping, lr, clipping, class weights, seeds).

## Stages and configs
One line per stage: cells file, what it varies, launch command.

## Results
Headline numbers with their uncertainty, and the plots that show them.

## Folder layout
| Folder | Content | In git |
|---|---|---|
| `configs/` | cells files (one per stage) and the YAML configs of the plot / analysis scripts | yes |
| `slurm/` | launchers specific to this experiment (generic ones are in the repo's `slurm/` and `scripts/`) | yes |
| `notes/`, `diagnostics/` | design notes, reports, one-off diagnostic scripts (where present) | yes |
| `outputs/` | run folders (`<STAGE>_<timestamp>/<cell>/seed_<s>/`: `.hydra/`, `metrics.csv`, checkpoints, `run_info.yaml`) and logs | no |
| `plots/` | every figure and metric table of the experiment | no |

## Related experiments
Which experiments this one builds on (paths of the checkpoints it uses) and which ones follow from it.

Copy this folder to `experiments/<yyyy-mm>_<short-name>/`, fill the README before the first launch, add a row to `experiments/README.md`.
