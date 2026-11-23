#!/bin/bash
# Render plots on a CPU node of cpu-all.  Each argument is <script>:<config>, run in this worktree as
#
#   .venv/bin/python <script> --config <config>
#
# one after the other; a failing one does not stop the others, and the job fails if any did.  The log lists each
# command, its output and the failed ones at the end.
#
#   sbatch slurm/plot_job.sh scripts/plot_pretrain_curves.py:scripts/configs/pretrain_curves_final.yaml [...]
#
#SBATCH --job-name=fm4tag-plots
#SBATCH --partition=cpu-all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/storage3/DSIP/rriva/research/fm4tag/slurm/plot_logs/plot_job_%j.log

set -uo pipefail

REPO=/storage3/DSIP/rriva/research/fm4tag
cd "$REPO" || exit 1
export MPLBACKEND=Agg  # PNG only: a dead X11 forward would hang matplotlib

echo "node $(hostname)  git $(git rev-parse --short HEAD)  $(date)"
failed=()
for pair in "$@"; do
    script=${pair%%:*}
    config=${pair#*:}
    echo "=== $(date +%T)  $script --config $config"
    if ! .venv/bin/python "$script" --config "$config"; then
        failed+=("$pair")
    fi
done
echo "=== $(date +%T)  done: $# command(s), ${#failed[@]} failed"
for pair in "${failed[@]}"; do
    echo "FAILED $pair"
done
[ ${#failed[@]} -eq 0 ]
