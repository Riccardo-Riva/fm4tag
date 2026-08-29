#!/bin/bash
# Constrained f_c/f_tau search + ROC-vs-GN2 for the per-token JC=0.3 noise
# (lambda) scan, using the best-val/head/loss_ce SEED of each lambda cell.
#
#   scripts/submit_lambda_seed_fraction_roc.sh
#   PRIME_CACHE=/path/to/existing/cache  scripts/submit_lambda_seed_fraction_roc.sh
#
# Grid: floor_u in {0.90, 0.95, 1.00} x floor_tau in {0.80, 0.90, 1.00}
#       x working_point in {0.77, 0.80}  = 18 constrained searches + 18 ROC plots.
# "floor" = required fraction of GN2 only-JT's own u / tau rejection at wp.
#
# Step 0 (1 GPU job, skipped if PRIME_CACHE is set): roc_comparison.py on the
#   nominal-fraction config -> a nominal ROC plot AND cache/probs_*.npz for all
#   8 checkpoints.
# Step 1 (2 CPU "grid-primer" jobs, one per wp, afterok step 0): the first
#   (u,tau) combo of each wp; computes the shared per-model (f_c,f_tau)
#   rejection grid ($ROOT/_grid/<slug>_grid_wp<wp>.csv) and its own ROC.
# Step 2 (16 CPU jobs, afterok the matching grid-primer): the other 8 combos
#   per wp; reuse the shared grid, so each is just a floor filter + ROC plot.
#
# The joint rejection grid depends only on (checkpoint, wp), not on the floors,
# and predictions depend only on (checkpoint, test-file) — so only step 0 needs
# a GPU and only the 2 grid-primers pay the sweep cost.
#
# Everything lands under scripts/plots/lambda_seed_bestval_<TS>/:
#   nominal/roc.png                       nominal-fraction comparison
#   <wpXXX_uXXX_tauXXX>/optimal_fractions.csv
#   <wpXXX_uXXX_tauXXX>/roc.png           constrained comparison for that floor set

set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=$REPO/.venv
BASE_CFG=$REPO/scripts/configs/fraction_constrained_lambda_seed_bestval.yaml
NOMINAL_CFG=$REPO/scripts/configs/roc_comparison_lambda_seed_bestval_nominal.yaml
PRIME_CACHE=${PRIME_CACHE:-}

FLOOR_U=(0.90 0.95 1.00)
FLOOR_TAU=(0.80 0.90 1.00)
WPS=(0.77 0.80)
CPUS=64

TS=$(date +%Y%m%d_%H%M%S)
ROOT=$REPO/scripts/plots/lambda_seed_bestval_$TS
CACHE=$ROOT/_cache
GRID=$ROOT/_grid
LOG=$REPO/scripts/logs/lambda_seed_bestval
mkdir -p "$ROOT" "$CACHE" "$GRID" "$LOG"
echo "Pipeline root: $ROOT"

# ── Step 0 — GPU prime (or reuse an existing prediction cache) ──────────────
DEP0=""
if [[ -n "$PRIME_CACHE" ]]; then
  n=$(ls "$PRIME_CACHE"/probs_*.npz 2>/dev/null | wc -l)
  [[ "$n" -ge 1 ]] || { echo "PRIME_CACHE has no probs_*.npz: $PRIME_CACHE" >&2; exit 1; }
  for f in "$PRIME_CACHE"/probs_*.npz; do ln -sf "$(readlink -f "$f")" "$CACHE/"; done
  echo "  step 0: reusing $n cached prediction files from $PRIME_CACHE (no GPU job)"
else
  PRIME=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=lsbv_prime
# L40S (sm_89) / A40 (sm_86) only — the venv torch has no kernels for the
# gpu-1080 / cliffjumper cards (sm_61), CUDA "no kernel image" at inference.
#SBATCH --partition=gpu-L40S-open,gpu-A40
#SBATCH --gres=shard:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=$LOG/prime_${TS}.log
set -euo pipefail
source $VENV/bin/activate
cd $REPO
python scripts/roc_comparison.py --config "$NOMINAL_CFG" --output "$ROOT/nominal"
for f in "$ROOT/nominal/cache"/probs_*.npz; do ln -sf "\$f" "$CACHE/"; done
ls -la "$CACHE"
EOF
)
  DEP0="--dependency=afterok:$PRIME"
  echo "  step 0 (GPU prime + nominal ROC): job $PRIME"
fi

# ── Steps 1 & 2 — CPU: constrained search + ROC per (u, tau, wp) ────────────
submit_combo() {  # $1=wp $2=u $3=t  $4=dep-flag  -> prints jobid
  local wp=$1 u=$2 t=$3 dep=$4
  local combo=wp$(tr -d . <<<"$wp")_u$(tr -d . <<<"$u")_tau$(tr -d . <<<"$t")
  local cdir=$ROOT/$combo wall=$5
  mkdir -p "$cdir/cache"
  sbatch --parsable $dep <<EOF
#!/bin/bash
#SBATCH --job-name=lsbv_$combo
#SBATCH --partition=cpu-all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=32G
#SBATCH --time=$wall
#SBATCH --output=$LOG/${combo}_${TS}.log
set -euo pipefail
source $VENV/bin/activate
cd $REPO
for f in "$CACHE"/probs_*.npz; do ln -sf "\$(readlink -f "\$f")" "$cdir/cache/"; done
python scripts/fraction_constrained_search.py --config "$BASE_CFG" --output "$cdir" \\
    --grid-dir "$GRID" --floor-u $u --floor-tau $t --working-point $wp \\
    --workers $CPUS --device cpu
python scripts/roc_comparison.py --config "$cdir/roc_comparison_config.yaml" \\
    --output "$cdir" --device cpu
echo "done: $combo"
EOF
}

for wp in "${WPS[@]}"; do
  first=1
  primer_jid=""
  for u in "${FLOOR_U[@]}"; do
  for t in "${FLOOR_TAU[@]}"; do
    if [[ $first -eq 1 ]]; then
      primer_jid=$(submit_combo "$wp" "$u" "$t" "$DEP0" "12:00:00")
      echo "  grid-primer wp=$wp -> job $primer_jid  (u=$u tau=$t)"
      first=0
    else
      jid=$(submit_combo "$wp" "$u" "$t" "--dependency=afterok:$primer_jid" "03:00:00")
      echo "    combo wp=$wp u=$u tau=$t -> job $jid"
    fi
  done
  done
done

echo
echo "Results: $ROOT/<combo>/roc.png and optimal_fractions.csv"
echo "Progress:  tail -f $LOG/*_${TS}.log"
