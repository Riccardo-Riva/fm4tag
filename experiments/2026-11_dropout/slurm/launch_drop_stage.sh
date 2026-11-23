#!/bin/bash
# Launch the dropout stage (D_drop: 24 runs, 8 two-GPU packs) from the sweep's batch sizes.  Meant to be
# submitted with a dependency on the sweep, from the repo root:
#
#   sbatch --dependency=afterok:<sweep job id> experiments/2026-11_dropout/slurm/launch_drop_stage.sh <abs path of the D_drop_sweep_<id> dir>
#
# Writes experiments/2026-11_dropout/configs/D_drop_final.cells (scripts/make_drop_cells.py), checks that every dropout of every
# cell is 0.2, then runs slurm/ssl_pack.sh twice as the D_drop.cells header says.  Stops without submitting anything
# if the configs are not committed, the sweep has no batch size for an arm, or a check fails.
#
#SBATCH --job-name=ssl_drop_launch
#SBATCH --partition=cpu-all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/storage3/DSIP/rriva/research/fm4tag/experiments/2026-11_dropout/outputs/D_drop_launch_%j.out

set -euo pipefail
REPO=/storage3/DSIP/rriva/research/fm4tag
SWEEP_DIR=${1:?abs path of the sweep dir}
cd "$REPO"
echo "node $(hostname)  git $(git rev-parse --short HEAD)  $(date)"

# config drift: the cells and the configs must be what is committed
if [[ -n "$(git status --porcelain -- src/fm4tag/configs experiments/2026-11_dropout/configs/D_drop.cells slurm/ssl_pack.sh slurm/pack_worker.sh)" ]]; then
    echo "ERROR: uncommitted changes in configs/cells/pack scripts — nothing submitted" >&2
    git status --short -- src/fm4tag/configs experiments/2026-11_dropout/configs slurm >&2
    exit 1
fi
CELLS=experiments/2026-11_dropout/configs/D_drop_final.cells
.venv/bin/python scripts/make_drop_cells.py --cells experiments/2026-11_dropout/configs/D_drop.cells \
    --sweep "$SWEEP_DIR/results.csv" --out "$CELLS"
export ENCODERS=rowcol_concat_v0 CHUNK=512
.venv/bin/python scripts/check_cell_dropout.py --cells "$CELLS" --expected 0.2

# a nested sbatch must not inherit this job's allocation
for v in $(env | grep -o '^SLURM_[A-Za-z0-9_]*'); do unset "$v"; done
export STAGE=D_drop MAX_EPOCHS=80 WALLTIME=7-00:00:00 EST_SECONDS_PER_EPOCH=1250 CELLS="$CELLS"
CELL_FILTER='^f[tr]' SEEDS='4242 42424' PACKS=5 MEM_PER_GPU_GB=40 bash slurm/ssl_pack.sh
sleep 3
CELL_FILTER='^sc' SEEDS='4242 42424 424242 4242424' PACKS=3 MEM_PER_GPU_GB=22 bash slurm/ssl_pack.sh
echo "launched $(date)"
