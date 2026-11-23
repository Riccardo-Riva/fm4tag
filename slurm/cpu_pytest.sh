#!/bin/bash
#SBATCH --job-name=fm4tag-cpu-pytest
#SBATCH --partition=cpu-all
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
# Usage: sbatch --wait --output=<log> slurm/cpu_pytest.sh <pytest args...>
cd /storage3/DSIP/rriva/research/fm4tag
.venv/bin/python -m pytest -q -p no:cacheprovider "$@"
