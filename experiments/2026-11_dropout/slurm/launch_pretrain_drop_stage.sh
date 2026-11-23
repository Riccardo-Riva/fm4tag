#!/bin/bash
# Launch the pretraining dropout stage (P_drop: P0, P1, P2 with dropout 0.1, 2 seeds = 6 runs, 3 two-GPU packs)
# from the sweep's batch sizes.  Meant to be submitted with a dependency on the sweep, from the repo root:
#
#   sbatch --dependency=afterok:<sweep job id> experiments/2026-11_dropout/slurm/launch_pretrain_drop_stage.sh <abs path of the P_drop_sweep_<id> dir>
#
# Writes experiments/2026-11_dropout/configs/P_drop_final.cells (scripts/make_drop_cells.py), checks that every dropout of every
# cell is 0.1 and that every run fits the walltime, then runs slurm/ssl_pack.sh once.  Stops without submitting
# anything if the configs are not committed, the sweep has no batch size for an arm, or a check fails.
#
#SBATCH --job-name=ssl_pdrop_launch
#SBATCH --partition=cpu-all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/storage3/DSIP/rriva/research/fm4tag/experiments/2026-11_dropout/outputs/P_drop_launch_%j.out

set -euo pipefail
REPO=/storage3/DSIP/rriva/research/fm4tag
SWEEP_DIR=${1:?abs path of the sweep dir}
SEEDS_PRE='42 424'          # the seeds of P_full, so every run is paired with its no-dropout twin
EPOCHS=30
WALLTIME=7-00:00:00         # partition limit
cd "$REPO"
echo "node $(hostname)  git $(git rev-parse --short HEAD)  $(date)"

# config drift: the cells and the configs must be what is committed
if [[ -n "$(git status --porcelain -- src/fm4tag/configs experiments/2026-11_dropout/configs/P_drop.cells slurm/ssl_pack.sh slurm/pack_worker.sh)" ]]; then
    echo "ERROR: uncommitted changes in configs/cells/pack scripts — nothing submitted" >&2
    git status --short -- src/fm4tag/configs experiments/2026-11_dropout/configs slurm >&2
    exit 1
fi
CELLS=experiments/2026-11_dropout/configs/P_drop_final.cells
.venv/bin/python scripts/make_drop_cells.py --cells experiments/2026-11_dropout/configs/P_drop.cells \
    --sweep "$SWEEP_DIR/results.csv" --out "$CELLS"
.venv/bin/python scripts/check_cell_dropout.py --cells "$CELLS" --expected 0.1 --skip head

# a task whose EPOCHS x @est (+ the worker's 1200 s margin and the job's 1800 s) exceeds the walltime is never started
.venv/bin/python - "$CELLS" "$EPOCHS" "$WALLTIME" <<'PY'
import re, sys
cells, epochs, wall = sys.argv[1], int(sys.argv[2]), sys.argv[3]
days, hms = wall.split('-')
limit = int(days) * 86400 + sum(int(x) * m for x, m in zip(hms.split(':'), (3600, 60, 1))) - 1800
bad = 0
for line in open(cells):
    if not line.strip() or line.startswith('#'):
        continue
    name = line.split()[0]
    est = int(re.search(r'@est=(\d+)', line).group(1))
    need = epochs * est + 1200
    print(f'{name}: {epochs} x {est} s = {need / 3600:.1f} h of {limit / 3600:.1f} h usable')
    bad += need > limit
sys.exit(1 if bad else 0)
PY

# a nested sbatch must not inherit this job's allocation
for v in $(env | grep -o '^SLURM_[A-Za-z0-9_]*'); do unset "$v"; done
# EST_SECONDS_PER_EPOCH is only the default of cells without @est (all three have one)
STAGE=P_drop MAX_EPOCHS=$EPOCHS WALLTIME=$WALLTIME CELLS="$CELLS" SEEDS="$SEEDS_PRE" PACKS=3 MEM_PER_GPU_GB=22 \
    EST_SECONDS_PER_EPOCH=6300 bash slurm/ssl_pack.sh
echo "launched $(date)"
