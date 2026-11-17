#!/bin/bash
# Slowdown of a training when other trainings share its card (scripts/share_measure.py).  Submit from the repo root:
#
#   sbatch slurm/share_measure.sh
#
#SBATCH --job-name=ssl_share_measure
#SBATCH --partition=gpu-L40S-open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/storage3/DSIP/rriva/research/fm4tag-chunk/slurm/ssl_campaign/W_share_measure_%j.out

REPO=/storage3/DSIP/rriva/research/fm4tag-chunk
cd "$REPO" || exit 1
nvidia-smi
export GPU_FREE_MIB_REQUIRED=44000 GPU_WAIT_SECONDS=1800
if ! bash slurm/gpu_preflight.sh; then
    if (( ${SLURM_RESTART_COUNT:-0} < 3 )); then
        scontrol requeue "${SLURM_JOB_ID}" && sleep 600
    fi
    echo "[gpu] cards stayed occupied" >&2
    exit 1
fi
DATA=/storage3/DSIP/rriva/datasets/transforming-jet-flavor/data_parallel_preprocess_251112_152051
cat $DATA/train/output/pp_output_train.h5 $DATA/val/output/pp_output_val.h5 > /dev/null
.venv/bin/python scripts/share_measure.py --cells scripts/configs/campaign/W_lr.cells \
    --alone slurm/ssl_campaign/W_measure_sweep_7126988/results.csv \
    --extra finetune.warmup_epochs=0.2 \
    --out "$REPO/slurm/ssl_campaign/W_share_measure_${SLURM_JOB_ID}"
