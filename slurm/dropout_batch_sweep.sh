#!/bin/bash
# Batch-size sweep of the dropout cells on one 2-GPU node (scripts/dropout_batch_sweep.py): the largest batch per
# cell that fits with dropout, and the cost of dropout per step.  Submit from the repo root:
#
#   sbatch slurm/dropout_batch_sweep.sh
#
#SBATCH --job-name=ssl_drop_sweep
#SBATCH --partition=gpu-L40S-open,gpu-A40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/storage3/DSIP/rriva/research/fm4tag-chunk/slurm/ssl_campaign/D_drop_sweep_%j.out

REPO=/storage3/DSIP/rriva/research/fm4tag-chunk
cd "$REPO" || exit 1
nvidia-smi
export GPU_FREE_MIB_REQUIRED=44000 GPU_WAIT_SECONDS=1800
if ! bash slurm/gpu_preflight.sh; then
    # another tenant on an allocated card: back to the queue (as the pack jobs do), up to 3 times
    if (( ${SLURM_RESTART_COUNT:-0} < 3 )); then
        scontrol requeue "${SLURM_JOB_ID}" && sleep 600
    fi
    echo "[gpu] cards stayed occupied" >&2
    exit 1
fi
DATA=/storage3/DSIP/rriva/datasets/transforming-jet-flavor/data_parallel_preprocess_251112_152051
cat $DATA/train/output/pp_output_train.h5 $DATA/val/output/pp_output_val.h5 > /dev/null
export ENCODERS=rowcol_concat_v0 CHUNK=512
.venv/bin/python scripts/dropout_batch_sweep.py --cells scripts/configs/campaign/D_drop.cells \
    --out "$REPO/slurm/ssl_campaign/D_drop_sweep_${SLURM_JOB_ID}"
