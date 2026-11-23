#!/bin/bash
# Test ROC of the best seed of each compared arm against GN2v01, one 2-GPU job:
#   GPU 0: constrained (f_c, f_tau) search on the VALIDATION split (inference + grid sweep)
#   GPU 1: test inference + ROC at the nominal fractions (fills the test prediction cache)
#   then:  from the cached test predictions (no GPU), the ROC at each model's own validation optimum, and both
#          ROCs without GN2v01 (ratio panels relative to the first model), configs written by roc_variants.py
#
#   sbatch slurm/arm_roc_job.sh [<search_cfg> <roc_cfg> <out_dir>]   (default: the P0 arms)
#
#SBATCH --job-name=arm_roc
#SBATCH --partition=gpu-L40S-open,gpu-A40
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/storage3/DSIP/rriva/research/fm4tag/slurm/logs/arm_roc_%j.log   # default; pass --output=<experiment>/outputs/... for a campaign

set -uo pipefail

REPO=/storage3/DSIP/rriva/research/fm4tag
DATA=/storage3/DSIP/rriva/datasets/transforming-jet-flavor/data_parallel_preprocess_251112_152051
SEARCH_CFG=${1:-$REPO/experiments/2026-10_ssl-campaign/configs/fraction_constrained_arms_final.yaml}
ROC_CFG=${2:-$REPO/experiments/2026-10_ssl-campaign/configs/roc_comparison_arms_final.yaml}
OUT=${3:-$REPO/experiments/2026-10_ssl-campaign/plots/arm_comparison}
PY=$REPO/.venv/bin/python

cd "$REPO"
export MPLBACKEND=Agg
mkdir -p "$OUT"
echo "node $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  $(date)"
echo "search $SEARCH_CFG  roc $ROC_CFG  out $OUT"
nvidia-smi -L

bash slurm/gpu_preflight.sh || { echo 'allocated GPUs still occupied'; exit 1; }

# One sequential read into the page cache: cold random NFS reads are ~8x slower.
cat "$DATA/val/output/pp_output_val.h5" "$DATA/test/output/pp_output_test.h5" > /dev/null
echo "page cache warm  $(date)"

IFS=, read -r GPU_A GPU_B <<< "$CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES=$GPU_A $PY scripts/fraction_constrained_search.py --config "$SEARCH_CFG" \
    --output "$OUT/fraction_search_val" --floor-frac 1.0 --workers 8 --device cuda \
    > "$OUT/fraction_search_val.log" 2>&1 &
SEARCH_PID=$!
CUDA_VISIBLE_DEVICES=$GPU_B $PY scripts/roc_comparison.py --config "$ROC_CFG" --output "$OUT/roc_test_nominal" \
    --device cuda > "$OUT/roc_test_nominal.log" 2>&1 &
ROC_PID=$!

wait $SEARCH_PID; SEARCH_RC=$?
wait $ROC_PID; ROC_RC=$?
echo "search rc=$SEARCH_RC  nominal roc rc=$ROC_RC  $(date)"
[[ $SEARCH_RC -eq 0 && $ROC_RC -eq 0 ]] || exit 1

$PY experiments/2026-10_ssl-campaign/outputs/arm_roc_logs/roc_variants.py "$SEARCH_CFG" "$ROC_CFG" "$OUT/fraction_search_val/optimal_fractions.csv" "$OUT" \
    || exit 1
for variant in optimal nominal_noref optimal_noref; do
    mkdir -p "$OUT/roc_test_$variant"
    ln -sfn "$OUT/roc_test_nominal/cache" "$OUT/roc_test_$variant/cache"
    $PY scripts/roc_comparison.py --config "$OUT/roc_comparison_$variant.yaml" --device cpu \
        > "$OUT/roc_test_$variant.log" 2>&1
    echo "$variant roc rc=$?  $(date)"
done
