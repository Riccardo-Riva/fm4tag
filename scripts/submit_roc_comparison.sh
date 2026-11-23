#!/bin/bash
# Submit scripts/roc_comparison.py as a single-GPU slurm job.
#
# Usage:
#   scripts/submit_roc_comparison.sh <config.yaml> [extra roc_comparison.py args...]
#
# Examples:
#   scripts/submit_roc_comparison.sh experiments/2026-08_lambda-scan/configs/roc_comparison_run_20260708_045439.yaml
#   scripts/submit_roc_comparison.sh experiments/<campaign>/configs/my_comparison.yaml --recompute
#
# Plots go to slurm/plots/run_<timestamp> (passed to roc_comparison.py via
# --output); logs go to slurm/roc_comparison/roc_comparison_<jobid>.log.
# The prediction cache lives under the output directory, so each submission
# (fresh timestamp) re-runs inference.

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=$(realpath "${1:?usage: $0 <config.yaml> [extra roc_comparison.py args...]}")
shift
EXTRA_ARGS="$*"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# outputs go to the plots/ of the experiment whose configs/ holds CONFIG (or EXPERIMENT=<campaign>)
source "$REPO/slurm/experiment_of.sh"
EXP=$(experiment_of "$CONFIG")
[[ -n "$EXP" && -d "$REPO/experiments/$EXP" ]] || { echo "config is not under experiments/<campaign>/: set EXPERIMENT=<campaign>" >&2; exit 1; }
OUT_DIR=$REPO/experiments/$EXP/plots/run_$TIMESTAMP

LOG_DIR=$REPO/experiments/2026-08_lambda-scan/outputs/script_logs/roc_comparison
mkdir -p "$LOG_DIR"

echo "Plot output directory: $OUT_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=roc_comparison
#SBATCH --partition=gpu-L40S-open,gpu-A40,cliffjumper,gpu-1080
#SBATCH --gres=shard:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=$LOG_DIR/roc_comparison_${TIMESTAMP}.log

nvidia-smi -L
source $REPO/.venv/bin/activate
cd $REPO
python scripts/roc_comparison.py --config "$CONFIG" --output "$OUT_DIR" $EXTRA_ARGS
EOF
