#!/bin/bash
#SBATCH --job-name=fm4tag-gpu-pytest
#SBATCH --partition=gpu-A40,gpu-L40S-open
#SBATCH --exclude=node82
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
# Usage: sbatch --wait --output=<log> slurm/gpu_pytest.sh <pytest args...>
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES"
cd /storage3/DSIP/rriva/research/fm4tag
.venv/bin/python -m pytest -q -p no:cacheprovider -rs "$@"
