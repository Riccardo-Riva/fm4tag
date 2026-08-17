#!/bin/bash
# Study part 2 — reruns of what part 1 could not answer:
#
#  * the forced-SDPA-backend A/B (part 1 crashed on a reused sdpa_kernel()
#    context manager, now fixed in profile_timing.py)
#  * a CONTROLLED row_mode A/B.  rowcol_v0 and rowcol_concat_v0 also differ in
#    col_heads/dim_head/row_heads/num_inds, so part 1's head-to-head does not
#    isolate row_mode.  Here rowcol_v0 is held fixed and ONLY row_mode is
#    flipped, on both encoders.
#  * the pretrain head-to-head on --synthetic batches: the configured
#    pretrain_dataset_path does not exist under the default base_path
#    (that preprocess dir has train/val/test only).
#
#   scripts/submit_rowmode_study2.sh
#
# Results: slurm/profiling/rowmode2_<TS>/{out.txt, <tag>/profile_*.json}

REPO=/storage3/DSIP/rriva/research/fm4tag
TS=$(date +%Y%m%d_%H%M%S)
OUT=${REPO}/slurm/profiling/rowmode2_${TS}

mkdir -pv "${OUT}"

cat > "${OUT}/study.sh" << 'OUTER'
#!/bin/bash
#SBATCH --job-name=rowmode2
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

FLAGS="--steps 20 --warmup 8 --data-batches 5"
HYDRA_COMMON="datamodule.num_workers=8"

# run <tag> <phase> [profiler flags...] -- [hydra overrides...]
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

# ── 1. Which SDPA backend, and does the choice explain the backward? ────
run per_token_auto      finetune                          -- ${PT}
run per_token_math      finetune --sdpa-backend math      -- ${PT}
run per_token_efficient finetune --sdpa-backend efficient -- ${PT}

# ── 2. CONTROLLED row_mode A/B — rowcol_v0 with only row_mode flipped ───
run ctrl_per_token finetune -- ${PT}
run ctrl_concat    finetune -- ${PT} ${TRK}.row_mode=concat ${GLB}.row_mode=concat

# ── 3. pretrain head-to-head (synthetic: no pretrain split on base_path) ─
run per_token_pt pretrain --synthetic -- ${PT}
run concat_pt    pretrain --synthetic -- ${CC}

# ── 4. concat under a forced backend, for symmetry ──────────────────────
run concat_math finetune --sdpa-backend math -- ${CC}

echo
echo "### study 2 complete"
OUTER

sed -i "s|__OUT__|${OUT}|g; s|__REPO__|${REPO}|g" "${OUT}/study.sh"

cd "${OUT}" && sbatch study.sh
