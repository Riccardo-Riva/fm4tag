#!/bin/bash
# Submit scripts/permutation_importance.py as a single-GPU slurm job.
#
# Usage:
#   scripts/submit_permutation_importance.sh <config.yaml> [extra args...]
#
# Examples:
#   scripts/submit_permutation_importance.sh scripts/configs/permutation_importance_lambda_scan.yaml
#   scripts/submit_permutation_importance.sh scripts/configs/... --max-jets 20000 --repeats 2
#
# Cost scales as  n_models x n_jets x (1 + n_variants x repeats)  forward
# passes; the default config is 7 models x 200k jets x (1 + 25 x 5) = 176 M
# jet-forwards.  Trim with --max-jets / --repeats if the walltime bites.
#
# Results go to scripts/plots/permutation_importance_<timestamp>/ (passed via
# --output); logs to scripts/logs/permutation_importance/.

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=$(realpath "${1:?usage: $0 <config.yaml> [extra args...]}")
shift
EXTRA_ARGS="$*"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=$REPO/scripts/plots/permutation_importance_$TIMESTAMP

LOG_DIR=$REPO/scripts/logs/permutation_importance
mkdir -p "$LOG_DIR"

echo "Output directory: $OUT_DIR"

sbatch <<EOF2
#!/bin/bash
#SBATCH --job-name=perm_importance
#SBATCH --partition=gpu-L40S-open,gpu-A40
#SBATCH --exclude=node82
#SBATCH --gres=shard:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=$LOG_DIR/permutation_importance_${TIMESTAMP}.log

nvidia-smi -L
source $REPO/.venv/bin/activate
cd $REPO
python scripts/permutation_importance.py --config "$CONFIG" --output "$OUT_DIR" $EXTRA_ARGS
EOF2
