#!/bin/bash
#SBATCH --job-name=fr_nan_diag
#SBATCH --partition=gpu-L40S-open,gpu-A40,cliffjumper
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/storage3/DSIP/rriva/research/fm4tag/experiments/2026-10_ssl-campaign/diagnostics/outputs/frnan/job_%j.out
#SBATCH --error=/storage3/DSIP/rriva/research/fm4tag/experiments/2026-10_ssl-campaign/diagnostics/outputs/frnan/job_%j.err
# Frozen arm NaN at batch 94208 (S3_lr_20261027_093035): which module, and is the cut at 65535 jets?
REPO=/storage3/DSIP/rriva/research/fm4tag
CFG=$(ls ${REPO}/experiments/2026-10_ssl-campaign/outputs/S3_lr_20261027_093035/fr_p0_lr1e-3/seed_42/outputs/*/*/.hydra/config.yaml)
source "${REPO}/.venv/bin/activate"
cd ${REPO}/slurm/smoke
srun --exclusive -n 1 --gres=gpu:1 --mem=22G python fr_nan_diag.py "$CFG" 94208 66048 > frnan/gpu0.log 2>&1 &
srun --exclusive -n 1 --gres=gpu:1 --mem=22G python fr_nan_diag.py "$CFG" 65024 32768 > frnan/gpu1.log 2>&1 &
wait
