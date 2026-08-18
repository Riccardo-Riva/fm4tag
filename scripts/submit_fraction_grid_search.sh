#!/bin/bash
# Submit scripts/fraction_grid_search.py as a single-GPU slurm job.
#
# Usage:
#   scripts/submit_fraction_grid_search.sh <config.yaml> [extra fraction_grid_search.py args...]
#
# Examples:
#   scripts/submit_fraction_grid_search.sh scripts/configs/fraction_grid_search_best_ce.yaml
#   scripts/submit_fraction_grid_search.sh scripts/configs/my_grid.yaml --recompute
#
# Outputs go to slurm/plots/run_<timestamp> (passed to fraction_grid_search.py
# via --output); logs go to slurm/fraction_grid_search/fraction_grid_search_<jobid>.log.
# The prediction cache lives under the output directory, so each submission
# (fresh timestamp) re-runs inference unless you symlink an existing cache/
# directory into the new output dir first.

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=$(realpath "${1:?usage: $0 <config.yaml> [extra fraction_grid_search.py args...]}")
shift
EXTRA_ARGS="$*"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=$REPO/scripts/plots/run_$TIMESTAMP

LOG_DIR=$REPO/scripts/logs/fraction_grid_search
mkdir -p "$LOG_DIR"

echo "Output directory: $OUT_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fraction_grid_search
#SBATCH --partition=gpu-L40S-open,gpu-A40,cliffjumper,gpu-1080
#SBATCH --gres=shard:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=$LOG_DIR/fraction_grid_search_${TIMESTAMP}.log

nvidia-smi -L
source $REPO/.venv/bin/activate
cd $REPO
python scripts/fraction_grid_search.py --config "$CONFIG" --output "$OUT_DIR" $EXTRA_ARGS
EOF
