#!/bin/bash
# Submit scripts/fraction_constrained_search.py as a multi-core CPU slurm job.
#
# No GPU is used — the grid sweep is pure numpy/CPU work over cached model
# predictions, parallelised across --workers processes. Runs on cpu-all.
#
# Usage:
#   scripts/submit_fraction_constrained_search.sh <config.yaml> [cpus] [extra args...]
#
# Examples:
#   scripts/submit_fraction_constrained_search.sh scripts/configs/fraction_two_stage_search_lambda_scan.yaml
#   scripts/submit_fraction_constrained_search.sh scripts/configs/my_search.yaml 36 --floor-frac 0.9
#
# Outputs go to scripts/plots/run_<timestamp>; logs go to
# scripts/logs/fraction_constrained_search/fraction_constrained_search_<jobid>.log.

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=$(realpath "${1:?usage: $0 <config.yaml> [cpus] [extra fraction_constrained_search.py args...]}")
shift
CPUS="${1:-32}"
[[ $# -gt 0 ]] && shift
EXTRA_ARGS="$*"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=$REPO/scripts/plots/run_$TIMESTAMP

LOG_DIR=$REPO/scripts/logs/fraction_constrained_search
mkdir -p "$LOG_DIR"

echo "Output directory: $OUT_DIR (cpus=$CPUS)"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fraction_constrained_search
#SBATCH --partition=cpu-all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=$LOG_DIR/fraction_constrained_search_${TIMESTAMP}.log

source $REPO/.venv/bin/activate
cd $REPO
python scripts/fraction_constrained_search.py --config "$CONFIG" --output "$OUT_DIR" --workers $CPUS $EXTRA_ARGS
EOF
