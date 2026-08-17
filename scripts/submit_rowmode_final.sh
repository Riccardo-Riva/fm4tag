#!/bin/bash
# Final numbers on shipping code (the small-attention math gate was measured to
# be a 2-3% regression and reverted, so this runs plain rowcol_v0/rowcol_concat_v0).
#
# Covers what the earlier jobs could not:
#   * pretrain per_token vs concat.  The first attempt OOMed at bs 1024 on one
#     A40 (constituent-level SupCon over ~20k packed tracks) and the second used
#     the pre-fix --synthetic path, which rebuilt the batch every step and put
#     ~90 ms of CPU work inside the measured step.  Both fixed here.
#   * whether dim_head is free: PyTorch's flash kernels compile for head dims
#     32, 64, 96, ..., so dim_head 8 is padded to 32 and buys nothing.  Raising
#     it to 32 should cost only the wider qkv/out projections, not the attention.
#
#   scripts/submit_rowmode_final.sh
#
# Results: slurm/profiling/final_<TS>/{out.txt, <tag>/profile_*.json}

REPO=/storage3/DSIP/rriva/research/fm4tag
TS=$(date +%Y%m%d_%H%M%S)
OUT=${REPO}/slurm/profiling/final_${TS}

mkdir -pv "${OUT}"

cat > "${OUT}/study.sh" << 'OUTER'
#!/bin/bash
#SBATCH --job-name=rowmode_final
#SBATCH --partition=gpu-A40
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=__OUT__/out.txt
#SBATCH --error=__OUT__/err.txt

REPO=__REPO__
OUT=__OUT__
source ${REPO}/.venv/bin/activate
nvidia-smi -L

run () {
  local tag=$1; shift
  local phase=$1; shift
  local prof=() hyd=() seen=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then seen=1; continue; fi
    if [ ${seen} -eq 0 ]; then prof+=("$a"); else hyd+=("$a"); fi
  done
  echo
  echo "################################################################"
  echo "### ${tag}   (phase=${phase})   ${prof[*]}   ${hyd[*]}"
  echo "################################################################"
  python ${REPO}/scripts/profile_timing.py \
      --phase ${phase} --steps 60 --warmup 15 "${prof[@]}" \
      --out ${OUT}/${tag} "${hyd[@]}"
}

PT=encoders=rowcol_v0
CC=encoders=rowcol_concat_v0
TRK=encoders.constituents.tracks.layers.0
GLB=encoders.global_encoder.layers.0

# ── 1. pretrain head-to-head, A/B/A/B (synthetic: base_path has no pretrain
#       split; bs 512: 1024 OOMs on one A40) ───────────────────────────────
run pre_per_token_1 pretrain --synthetic -- ${PT} datamodule.batch_size=512
run pre_concat_1    pretrain --synthetic -- ${CC} datamodule.batch_size=512
run pre_per_token_2 pretrain --synthetic -- ${PT} datamodule.batch_size=512
run pre_concat_2    pretrain --synthetic -- ${CC} datamodule.batch_size=512

# ── 2. Is dim_head free below flash's 32-wide tile? ──────────────────────
run head8_baseline finetune --data-batches 5 -- ${PT} datamodule.num_workers=8
run head32         finetune --data-batches 5 -- ${PT} datamodule.num_workers=8 \
    ${TRK}.dim_head=32 ${TRK}.dim_row_head=32 ${GLB}.dim_head=32 ${GLB}.dim_row_head=32

echo
echo "### final study complete"
OUTER

sed -i "s|__OUT__|${OUT}|g; s|__REPO__|${REPO}|g" "${OUT}/study.sh"

cd "${OUT}" && sbatch study.sh
