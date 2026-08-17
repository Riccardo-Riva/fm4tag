#!/bin/bash
# Before/after for the small-attention math gate (_prefer_math_backend in
# fm4tag/models/attention.py), plus the row_mode head-to-head under the same
# protocol.
#
# Protocol notes — the earlier study measured the SAME config at 239 and 264
# ms/step in two different jobs (~12% spread, node contention rather than data),
# so a 10% effect needs care:
#   * 60 measured steps instead of 20.
#   * each pair run A,B,A,B back to back in ONE job, so both arms see the same
#     node conditions and the repeat exposes drift.
#   * real data, matching the phase-1 numbers this is compared against.
#
#   scripts/submit_mathgate_ab.sh
#
# Results: slurm/profiling/mathgate_<TS>/{out.txt, <tag>/profile_*.json}

REPO=/storage3/DSIP/rriva/research/fm4tag
TS=$(date +%Y%m%d_%H%M%S)
OUT=${REPO}/slurm/profiling/mathgate_${TS}

mkdir -pv "${OUT}"

cat > "${OUT}/study.sh" << 'OUTER'
#!/bin/bash
#SBATCH --job-name=mathgate
#SBATCH --partition=gpu-A40
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=__OUT__/out.txt
#SBATCH --error=__OUT__/err.txt

REPO=__REPO__
OUT=__OUT__
source ${REPO}/.venv/bin/activate
nvidia-smi -L

FLAGS="--steps 60 --warmup 15 --data-batches 5"
HYDRA_COMMON="datamodule.num_workers=8"

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
      --phase ${phase} ${FLAGS} "${prof[@]}" --out ${OUT}/${tag} \
      ${HYDRA_COMMON} "${hyd[@]}"
}

PT=encoders=rowcol_v0
CC=encoders=rowcol_concat_v0
TRK=encoders.constituents.tracks.layers.0
GLB=encoders.global_encoder.layers.0

# ── 1. The fix, A/B/A/B on the default (per_token) finetune config ──────
run pt_before_1 finetune --small-attn-math off --kernel-census 5 -- ${PT}
run pt_after_1  finetune --small-attn-math on  --kernel-census 5 -- ${PT}
run pt_before_2 finetune --small-attn-math off -- ${PT}
run pt_after_2  finetune --small-attn-math on  -- ${PT}

# ── 2. Same A/B on concat, to check the fix is not per_token-specific ───
run cc_before finetune --small-attn-math off -- ${CC}
run cc_after  finetune --small-attn-math on  -- ${CC}

# ── 3. Controlled row_mode A/B (rowcol_v0, only row_mode flipped), fix ON
#       — the configuration that would actually ship ─────────────────────
run ctrl_per_token finetune -- ${PT}
run ctrl_concat    finetune -- ${PT} ${TRK}.row_mode=concat ${GLB}.row_mode=concat

# ── 4. pretrain. bs 1024 OOMs on one A40 (constituent-level SupCon over
#       ~20k packed tracks), so halve it. ──────────────────────────────
run pt_pre_before pretrain --synthetic --small-attn-math off -- ${PT} datamodule.batch_size=512
run pt_pre_after  pretrain --synthetic --small-attn-math on  -- ${PT} datamodule.batch_size=512
run cc_pre_after  pretrain --synthetic --small-attn-math on  -- ${CC} datamodule.batch_size=512

echo
echo "### mathgate A/B complete"
OUTER

sed -i "s|__OUT__|${OUT}|g; s|__REPO__|${REPO}|g" "${OUT}/study.sh"

cd "${OUT}" && sbatch study.sh
