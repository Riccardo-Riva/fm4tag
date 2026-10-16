#!/bin/bash
# S0 benchmark of the SSL campaign: step time, jets/s and peak GPU memory of
# every workload vs intersample chunk size C and batch size B, on ONE L40S with
# real data (campaign_ssl, 152051 splits).  One slurm job, one profile_timing.py
# process per point (an OOM only loses that point).
#
#   A. C sweep at B=4096:  C in {64,128,256,512,1024}  x  {SC+JC, P1}  x  {per_token, concat}
#   B. B sweep at C=256:   B in {4096,8192,16384}       x  6 workloads  x  {per_token, concat}
#
# Workloads: SC (scratch, CE), SCJC (+ JC 0.3), P0 (pretrain, contrastive),
# P1 (+ JC 1.0), FR (frozen encoders, CE), FRJC (+ JC 0.3).
#
#   scripts/submit_s0_benchmark.sh            # submit
#   DRY_RUN=1 scripts/submit_s0_benchmark.sh  # write the job script only
#
# Results: slurm/profiling/s0_<TS>/<point>/profile_<phase>.json; summarise with
#   scripts/profile_table.py slurm/profiling/s0_<TS>
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# The job imports fm4tag from ${REPO}/src (PYTHONPATH wins over the venv's
# editable install), so a detached snapshot worktree can be benchmarked with
# another checkout's interpreter:  PY=<other checkout>/.venv/bin/python
PY=${PY:-${REPO}/.venv/bin/python}
TS=$(date +%Y%m%d_%H%M%S)
OUT=${REPO}/slurm/profiling/s0_${TS}
mkdir -p "${OUT}"

declare -A WORKLOAD=(
    [SC]="--phase finetune arm=scratch"
    [SCJC]="--phase finetune arm=scratch finetune.loss_weights.jet_contrastive=0.3"
    [P0]="--phase pretrain arm=pretrain"
    [P1]="--phase pretrain arm=pretrain pretrain.loss_weights.jet_contrastive=1.0"
    [FR]="--phase finetune arm=frozen"
    [FRJC]="--phase finetune arm=frozen finetune.loss_weights.jet_contrastive=0.3"
)
declare -A MODE=([pt]=rowcol_v0 [cc]=rowcol_concat_v0)

POINTS=()
for mode in pt cc; do
    for w in SCJC P1; do
        for c in 64 128 256 512 1024; do
            POINTS+=("A_${w}_${mode}_c${c}_b4096|${w}|${mode}|${c}|4096")
        done
    done
    for w in SC SCJC P0 P1 FR FRJC; do
        for b in 4096 8192 16384; do
            POINTS+=("B_${w}_${mode}_c256_b${b}|${w}|${mode}|256|${b}")
        done
    done
done

{
cat <<HDR
#!/bin/bash
#SBATCH --job-name=fm4tag-s0
#SBATCH --partition=gpu-L40S-open
#SBATCH --exclude=node82
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=${OUT}/out.txt
#SBATCH --error=${OUT}/err.txt
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader -i \$CUDA_VISIBLE_DEVICES
GPU_FREE_MIB_REQUIRED=44000 bash ${REPO}/slurm/gpu_preflight.sh || exit 1
export PYTHONPATH=${REPO}/src
run() { name=\$1; shift; echo "=============== \$name  \$(date -Is)"; mkdir -p ${OUT}/\$name
  ${PY} ${REPO}/scripts/profile_timing.py --config-name campaign_ssl \\
      --steps 20 --warmup 8 --data-batches 10 --out ${OUT}/\$name "\$@" \\
      > ${OUT}/\$name/log.txt 2>&1 || echo "  FAILED (see \$name/log.txt)"; }
HDR
for p in "${POINTS[@]}"; do
    IFS='|' read -r name w mode c b <<< "${p}"
    echo "run ${name} ${WORKLOAD[$w]} encoders=${MODE[$mode]} chunk_size=${c} datamodule.batch_size=${b} datamodule.num_workers=10"
done
} > "${OUT}/job.sh"

echo "${#POINTS[@]} points -> ${OUT}/job.sh"
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
for v in $(env | grep -o '^SLURM_[A-Z_]*'); do unset "$v"; done
sbatch "${OUT}/job.sh"
