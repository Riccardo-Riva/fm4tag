#!/bin/bash
# Test-set rejections of every run of the compared methods (scripts/arm_test_metrics.py), one 2-GPU job:
#   GPU 0 and GPU 1: val + test inference of half the runs each (cached under <out_dir>/cache)
#   then (CPU):      per-run fraction search on val and test rejections, per_run.csv / per_arm.csv
#
#   sbatch slurm/arm_test_metrics_job.sh [<config>]   (default: experiments/2026-10_ssl-campaign/configs/arm_test_metrics.yaml)
#
#SBATCH --job-name=arm_test_metrics
#SBATCH --partition=gpu-L40S-open,gpu-A40
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/storage3/DSIP/rriva/research/fm4tag/slurm/logs/arm_test_metrics_%j.log   # default; pass --output=<experiment>/outputs/... for a campaign

set -uo pipefail

REPO=/storage3/DSIP/rriva/research/fm4tag
DATA=/storage3/DSIP/rriva/datasets/transforming-jet-flavor/data_parallel_preprocess_251112_152051
CFG=${1:-$REPO/experiments/2026-10_ssl-campaign/configs/arm_test_metrics.yaml}
PY=$REPO/.venv/bin/python

cd "$REPO"
echo "node $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  $(date)"
nvidia-smi -L

bash slurm/gpu_preflight.sh || { echo 'allocated GPUs still occupied'; exit 1; }

# One sequential read into the page cache: cold random NFS reads are ~8x slower.
cat "$DATA/val/output/pp_output_val.h5" "$DATA/test/output/pp_output_test.h5" > /dev/null
echo "page cache warm  $(date)"

IFS=, read -r GPU_A GPU_B <<< "$CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES=$GPU_A $PY scripts/arm_test_metrics.py --config "$CFG" --infer --part 0/2 &
PID_A=$!
CUDA_VISIBLE_DEVICES=$GPU_B $PY scripts/arm_test_metrics.py --config "$CFG" --infer --part 1/2 &
PID_B=$!
wait $PID_A; RC_A=$?
wait $PID_B; RC_B=$?
echo "inference rc=$RC_A/$RC_B  $(date)"
[[ $RC_A -eq 0 && $RC_B -eq 0 ]] || exit 1

$PY scripts/arm_test_metrics.py --config "$CFG" --metrics --workers 16
echo "metrics rc=$?  $(date)"
