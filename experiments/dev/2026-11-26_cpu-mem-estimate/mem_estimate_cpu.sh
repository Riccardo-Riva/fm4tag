#!/bin/bash
# CPU memory of one pretraining step per cell (README in this folder): model B's calibration cells, then P0 / P1 / P2
# of small_RC, small_C and normal_C, then two cells at half the batch (is the memory linear in the batch?).
# Results: outputs/results_<job>.jsonl; log: outputs/job_<job>.log.
#
#   mkdir -p experiments/dev/2026-11-26_cpu-mem-estimate/outputs
#   sbatch experiments/dev/2026-11-26_cpu-mem-estimate/mem_estimate_cpu.sh
#
#SBATCH --job-name=fm4tag-dev-mem-cpu
#SBATCH --partition=cpu-all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/storage3/DSIP/rriva/research/fm4tag/experiments/dev/2026-11-26_cpu-mem-estimate/outputs/job_%j.log

set -euo pipefail

REPO=/storage3/DSIP/rriva/research/fm4tag
HERE=${REPO}/experiments/dev/2026-11-26_cpu-mem-estimate
CELLS=${REPO}/experiments/2026-11_pretrain-saint-small/configs
OUT=${HERE}/outputs/results_${SLURM_JOB_ID}.jsonl
cd "${REPO}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# Freed tensors go back to the OS at once, so the sampled resident memory follows the live tensors.
export MALLOC_MMAP_THRESHOLD_=65536

echo "node $(hostname)  git $(git rev-parse --short HEAD)  $(date)"
git status --short -- src/fm4tag/configs
run() { echo "=== $(date +%T)  $*"; .venv/bin/python "${HERE}/mem_estimate_cpu.py" --out "${OUT}" "$@"; }

run --cells "${HERE}/calibration_model_b.cells" --config-name default --batch 2560
run --cells "${CELLS}/P_small_RC.cells" --config-name saint_small_RC --batch 2560
run --cells "${CELLS}/P_small_C.cells" --config-name saint_small_C --batch 2560
run --cells "${CELLS}/P_normal_C.cells" --config-name saint_normal_C --batch 2560
run --cells "${HERE}/calibration_model_b.cells" --config-name default --batch 1280 --only p1_B
run --cells "${CELLS}/P_small_RC.cells" --config-name saint_small_RC --batch 1280 --only p1_small_RC
echo "=== $(date +%T)  done"
