#!/bin/bash
# One-off study: compute efficiency of RowColTransformer's two row_mode variants
# (per_token vs concat), plus the SDPA-backend diagnosis for the tracks encoder's
# lopsided backward/forward ratio.
#
# Bundles every configuration into ONE slurm job (shared cluster — a dozen
# 2-minute profiles do not deserve a dozen job slots).  Each run is
# scripts/profile_timing.py; results stream to out.txt in submission order, so a
# job killed partway still leaves the earlier (most important) runs readable.
#
#   scripts/submit_rowmode_study.sh
#
# Results: slurm/profiling/rowmode_<TS>/{out.txt, <tag>/profile_*.json}

REPO=/storage3/DSIP/rriva/research/fm4tag
TS=$(date +%Y%m%d_%H%M%S)
OUT=${REPO}/slurm/profiling/rowmode_${TS}

mkdir -pv "${OUT}"

cat > "${OUT}/study.sh" << 'OUTER'
#!/bin/bash
#SBATCH --job-name=rowmode
#SBATCH --partition=gpu-L40S-open,gpu-A40,cliffjumper
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

# Short but past warmup; the dataloader is not the subject here, so keep
# --data-batches small.
FLAGS="--steps 20 --warmup 8 --data-batches 5"
HYDRA_COMMON="datamodule.num_workers=8"

# run <tag> <phase> [profiler flags...] -- [hydra overrides...]
# argparse takes the trailing `overrides` positional only as one contiguous
# run at the very end, so every flag must precede every override.
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
  echo "### ${tag}   (phase=${phase})   ${hyd[*]}"
  echo "################################################################"
  python ${REPO}/scripts/profile_timing.py \
      --phase ${phase} ${FLAGS} "${prof[@]}" --out ${OUT}/${tag} \
      ${HYDRA_COMMON} "${hyd[@]}"
}

PT=encoders=rowcol_v0
CC=encoders=rowcol_concat_v0
TRK=encoders.constituents.tracks.layers.0

# ── 1. Head-to-head, finetune, with kernel census ───────────────────────
run per_token_ft   finetune --kernel-census 5 -- ${PT}
run concat_ft      finetune --kernel-census 5 -- ${CC}

# ── 2. Which SDPA backend, and does the choice explain the backward? ────
run per_token_math      finetune --sdpa-backend math      -- ${PT}
run per_token_efficient finetune --sdpa-backend efficient -- ${PT}
run concat_math         finetune --sdpa-backend math      -- ${CC}

# ── 3. Head-to-head, pretrain ───────────────────────────────────────────
run per_token_pt   pretrain -- ${PT}
run concat_pt      pretrain -- ${CC}

# ── 4. num_inds axis (tracks encoder only — it dominates the step) ──────
run per_token_ni16  finetune -- ${PT} ${TRK}.num_inds=16
run concat_ni64     finetune -- ${CC} ${TRK}.num_inds=64

# ── 5. batch-size axis ──────────────────────────────────────────────────
run per_token_bs512  finetune -- ${PT} datamodule.batch_size=512
run concat_bs512     finetune -- ${CC} datamodule.batch_size=512
run per_token_bs2048 finetune -- ${PT} datamodule.batch_size=2048
run concat_bs2048    finetune -- ${CC} datamodule.batch_size=2048

echo
echo "### study complete"
OUTER

sed -i "s|__OUT__|${OUT}|g; s|__REPO__|${REPO}|g" "${OUT}/study.sh"

cd "${OUT}" && sbatch study.sh
