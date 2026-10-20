#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# SSL campaign launcher for a cluster that allows at most MAX_GPU_JOBS (10) GPU
# jobs per user, running + pending: GPU PACKS.
#
# A stage = the cells of a cell file x SEEDS = a list of TASKS (one 1-GPU
# training each). It is run by PACKS slurm jobs of GPUS_PER_PACK GPUs on one
# node; each GPU runs a worker (slurm/pack_worker.sh, its own srun step) that
# takes the next task from the stage's shared queue when it is free. Every
# training is exactly a 1-GPU training as before (batch fills one card).
#
# Cell file: one cell per line,   <name>  <hydra override> ...  [@token ...]
#   @epochs=N     trainer.max_epochs of this cell (default: MAX_EPOCHS)
#   @est=S        estimated seconds per epoch on the slowest card (default:
#                 EST_SECONDS_PER_EPOCH); a task starts only if it fits the
#                 time left in its job
#   encoder_ckpt=@<cell>       last.ckpt of that cell of THIS stage, same seed;
#                              the task waits until that pretraining is done
#   encoder_ckpt=@/abs/<cell dir>/seed_<s>   a pretraining of ANOTHER stage
#   ${VAR}        filled from the environment (unset = error)
#
# Usage:
#   CELLS=scripts/configs/campaign/S2.cells SEEDS=42 PACKS=4 \
#     MAX_EPOCHS=10 EST_SECONDS_PER_EPOCH=6000 bash slurm/ssl_pack.sh
#   DRY_RUN=1 ...                 write everything, submit nothing
#   CELL_FILTER=<regex> ...       only the cells whose name matches (e.g. the
#                                 probes whose pretrainings have ended)
#   DEPENDENCY=afterany:<ids> ...  hold the packs until those jobs have ended
#                                 (e.g. probes of single-GPU pretrainings)
#   STAGE_DIR=<stage> PACKS=1 bash slurm/ssl_pack.sh
#                                 add packs to an existing stage (e.g. after a
#                                 node failure: stale claims are released)
#
# Layout: slurm/ssl_campaign/<stage>_<TS>/{tasks.tsv, pack.env, pack_job.sh,
#   pack_<job>.out, pack_<job>_w<i>.log, .queue/, <cell>/seed_<s>/...} — every
#   seed dir has the usual run_info.yaml / .hydra / metrics.csv layout.
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${REPO}/.venv
CONFIG_DIR=${REPO}/src/fm4tag/configs
CONFIG=campaign_ssl

PACKS=${PACKS:?set PACKS (number of GPU-pack jobs to submit)}
GPUS_PER_PACK=${GPUS_PER_PACK:-2}
MAX_GPU_JOBS=${MAX_GPU_JOBS:-10}
GPU_NODE=${GPU_NODE:-gpu-L40S-open,gpu-A40}
EXCLUDE_NODES=${EXCLUDE_NODES:-}    # node82 GPU 5 works again (checked 2026-10-20)
# 7 loader workers per GPU (9 CPUs): node82 often has GPUs free but few CPUs
# (2026-10-20: 3 GPUs, 18 CPUs); 7 still feed a probe at B 32768 several times over.
NUM_WORKERS=${NUM_WORKERS:-7}
CPUS_PER_GPU=$((NUM_WORKERS + 2))
# Host RAM per GPU: a pretraining uses ~11 GiB anon (10 workers, measured S2);
# the rest holds the page cache of the train + val files (17 GB).
MEM_PER_GPU_GB=${MEM_PER_GPU_GB:-32}
WALLTIME=${WALLTIME:-7-00:00:00}
WANDB_MODE=${WANDB_MODE:-}
DEPENDENCY=${DEPENDENCY:-}      # e.g. afterany:<pretraining job ids> — the packs
                                # then start only when those jobs have ended
DRY_RUN=${DRY_RUN:-0}
GPU_FREE_MIB_REQUIRED=${GPU_FREE_MIB_REQUIRED:-44000}
GPU_WAIT_SECONDS=${GPU_WAIT_SECONDS:-1800}
GPU_MAX_REQUEUES=${GPU_MAX_REQUEUES:-3}
MIN_SANE_SECONDS=${MIN_SANE_SECONDS:-1200}

walltime_to_seconds() {
    local t=$1 d=0 a b c
    [[ "${t}" == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
    IFS=: read -r a b c <<< "${t}"
    echo $(( 10#${d} * 86400 + 10#${a} * 3600 + 10#${b} * 60 + 10#${c:-0} ))
}
WALLTIME_SECONDS=$(walltime_to_seconds "${WALLTIME}")

check_job_limit() {
    local queued
    queued=$(squeue -u "${USER}" -h -o '%b' | grep -c 'gpu' || true)
    if (( queued + PACKS > MAX_GPU_JOBS )); then
        echo "ERROR: ${queued} GPU jobs already queued + ${PACKS} packs > the limit of" \
             "${MAX_GPU_JOBS} — nothing submitted" >&2
        exit 1
    fi
    echo "GPU jobs: ${queued} queued + ${PACKS} new <= ${MAX_GPU_JOBS}"
}

submit_packs() {
    local i jid
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] would submit ${PACKS} x ${STAGE_DIR}/pack_job.sh"
        return
    fi
    check_job_limit
    for v in $(env | grep -o '^SLURM_[A-Z_]*'); do unset "$v"; done
    for (( i = 0; i < PACKS; i++ )); do
        jid=$(cd "${STAGE_DIR}" && sbatch --parsable ${DEPENDENCY:+--dependency=${DEPENDENCY}} pack_job.sh)
        echo "${jid}" >> "${STAGE_DIR}/pack_jobs"
        echo "Submitted pack job ${jid}${DEPENDENCY:+ (after ${DEPENDENCY})}"
    done
    echo "Progress: ls ${STAGE_DIR}/.queue/   (or grep -H '^status:' ${STAGE_DIR}/*/seed_*/run_info.yaml)"
}

# ── Adding packs to an existing stage ─────────────────────────────────────────
if [[ -n "${STAGE_DIR:-}" ]]; then
    [[ -f "${STAGE_DIR}/pack_job.sh" ]] || { echo "ERROR: ${STAGE_DIR} is not a pack stage" >&2; exit 1; }
    submit_packs
    exit 0
fi

# ── A new stage ───────────────────────────────────────────────────────────────
CELLS=${CELLS:?set CELLS=<cell file>}
STAGE=${STAGE:-$(basename "${CELLS}" .cells)}
MAX_EPOCHS=${MAX_EPOCHS:?set MAX_EPOCHS (default for cells without @epochs)}
EST_SECONDS_PER_EPOCH=${EST_SECONDS_PER_EPOCH:?set EST_SECONDS_PER_EPOCH (default for cells without @est)}
read -r -a SEEDS <<< "${SEEDS:-42}"
CELL_FILTER=${CELL_FILTER:-}

[[ -f "${CELLS}" ]] || { echo "ERROR: cell file ${CELLS} not found" >&2; exit 1; }
if ! git -C "${REPO}" diff --quiet -- src/fm4tag/configs; then
    echo "ERROR: uncommitted changes under src/fm4tag/configs — commit first" >&2
    exit 1
fi
for var in $(grep -vE '^\s*#' "${CELLS}" | grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*\}' \
             | tr -d '${}' | sort -u); do
    [[ -n "${!var:-}" ]] || { echo "ERROR: cell file uses \${${var}}, which is not set" >&2; exit 1; }
done
mapfile -t CELL_LINES < <(grep -vE '^\s*(#|$)' "${CELLS}" | envsubst)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STAGE_DIR=${REPO}/slurm/ssl_campaign/${STAGE}_${TIMESTAMP}
WANDB_GROUP=${WANDB_GROUP:-ssl_${STAGE}_${TIMESTAMP}}
mkdir -p "${STAGE_DIR}/.queue"
cp "${CELLS}" "${STAGE_DIR}/cells"
[[ -n "${CELL_FILTER}" ]] && echo "${CELL_FILTER}" > "${STAGE_DIR}/cell_filter"

declare -A DATA_SEEN=()
DATA_FILES=()
TASK_LINES=()
CELL_NAMES=" "
for LINE in "${CELL_LINES[@]}"; do
    read -r CELL TOKENS <<< "${LINE}"
    [[ -n "${CELL_FILTER}" && ! "${CELL}" =~ ${CELL_FILTER} ]] && continue
    EPOCHS=${MAX_EPOCHS} EST=${EST_SECONDS_PER_EPOCH} AFTER_CELL="" OVS=()
    for tok in ${TOKENS}; do
        case "${tok}" in
            @epochs=*) EPOCHS=${tok#@epochs=} ;;
            @est=*)    EST=${tok#@est=} ;;
            @*) echo "ERROR: ${CELL}: unknown token ${tok}" >&2; exit 1 ;;
            encoder_ckpt=@/*) OVS+=("${tok}") ;;
            encoder_ckpt=@*)
                AFTER_CELL=${tok#encoder_ckpt=@}
                [[ "${CELL_NAMES}" == *" ${AFTER_CELL} "* ]] || {
                    echo "ERROR: ${CELL}: ${tok} names no earlier cell of this stage" >&2; exit 1; }
                OVS+=("${tok}") ;;
            *) OVS+=("${tok}") ;;
        esac
    done
    CELL_NAMES+="${CELL} "
    # Compose now: a typo fails here; it also names the data files to warm.
    if ! FILES=$("${VENV}/bin/python" "${REPO}/scripts/campaign_cell_files.py" "${OVS[@]}"); then
        echo "ERROR: cell ${CELL} does not compose — nothing submitted" >&2
        exit 1
    fi
    for f in ${FILES}; do
        [[ -z "${DATA_SEEN[$f]:-}" ]] && { DATA_SEEN[$f]=1; DATA_FILES+=("$f"); }
    done
    for SEED in "${SEEDS[@]}"; do
        AFTER=-
        [[ -n "${AFTER_CELL}" ]] && AFTER=${AFTER_CELL}__s${SEED}
        TASK_LINES+=("${CELL}__s${SEED}"$'\t'"${CELL}"$'\t'"${SEED}"$'\t'"${EPOCHS}"$'\t'"${EST}"$'\t'"${AFTER}"$'\t'"${OVS[*]}")
        echo "  task ${CELL}__s${SEED}: ${EPOCHS} epochs x ${EST} s ($(( (EPOCHS * EST + 1200) / 3600 )) h)${AFTER_CELL:+, after ${AFTER}}"
    done
done
(( ${#TASK_LINES[@]} )) || { echo "ERROR: no cell matches CELL_FILTER=${CELL_FILTER}" >&2; exit 1; }
printf '%s\n' "${TASK_LINES[@]}" > "${STAGE_DIR}/tasks.tsv"

cat > "${STAGE_DIR}/pack.env" << EOF
REPO=${REPO}
VENV=${VENV}
CONFIG_DIR=${CONFIG_DIR}
CONFIG=${CONFIG}
STAGE=${STAGE}
WANDB_GROUP=${WANDB_GROUP}
NUM_WORKERS=${NUM_WORKERS}
MIN_SANE_SECONDS=${MIN_SANE_SECONDS}
EOF
[[ -n "${WANDB_MODE}" ]] && echo "export WANDB_MODE=${WANDB_MODE}" >> "${STAGE_DIR}/pack.env"

{
    echo "stage:        ${STAGE}"
    echo "submitted:    $(date -Is)"
    echo "repo:         ${REPO}  (git $(git -C "${REPO}" rev-parse --short HEAD))"
    echo "config:       ${CONFIG}"
    echo "packs:        ${PACKS} x ${GPUS_PER_PACK} GPUs (${GPU_NODE}, exclude ${EXCLUDE_NODES:-none}), ${WALLTIME}"
    echo "seeds:        ${SEEDS[*]}"
    echo "wandb group:  ${WANDB_GROUP}"
    echo "tasks:        ${#TASK_LINES[@]}"
    printf '  %s\n' "${CELL_LINES[@]}"
} > "${STAGE_DIR}/sweep_info.txt"

cat > "${STAGE_DIR}/pack_job.sh" << EOF
#!/bin/bash
#SBATCH --job-name=ssl_${STAGE}_pack
#SBATCH --partition=${GPU_NODE}
$( [[ -n "${EXCLUDE_NODES}" ]] && echo "#SBATCH --exclude=${EXCLUDE_NODES}" )
#SBATCH --nodes=1
#SBATCH --ntasks=${GPUS_PER_PACK}
#SBATCH --gres=gpu:${GPUS_PER_PACK}
#SBATCH --cpus-per-task=${CPUS_PER_GPU}
#SBATCH --mem=$(( MEM_PER_GPU_GB * GPUS_PER_PACK ))G
#SBATCH --time=${WALLTIME}
#SBATCH --output=${STAGE_DIR}/pack_%j.out
#SBATCH --error=${STAGE_DIR}/pack_%j.err
# Generated by slurm/ssl_pack.sh — edit that, not this.
nvidia-smi
END_TIME=\$(scontrol show job "\${SLURM_JOB_ID}" -o 2>/dev/null \\
           | grep -oE 'EndTime=[^ ]+' | head -1 | cut -d= -f2)
if [[ -n "\${END_TIME}" && "\${END_TIME}" != "Unknown" ]]; then
    export DEADLINE=\$(date -d "\${END_TIME}" +%s)
else
    export DEADLINE=\$(( \$(date +%s) + ${WALLTIME_SECONDS} ))
fi
export GPU_FREE_MIB_REQUIRED=${GPU_FREE_MIB_REQUIRED} GPU_WAIT_SECONDS=${GPU_WAIT_SECONDS}
if ! bash ${REPO}/slurm/gpu_preflight.sh; then
    if (( \${SLURM_RESTART_COUNT:-0} < ${GPU_MAX_REQUEUES} )); then
        scontrol requeue "\${SLURM_JOB_ID}" && sleep 600
    fi
    echo "[gpu] cards stayed occupied — no task consumed" >&2
    exit 1
fi
WARM_START=\$(date +%s)
cat ${DATA_FILES[*]} > /dev/null
echo "[data] page cache warmed in \$(( \$(date +%s) - WARM_START )) s"
for (( i = 0; i < ${GPUS_PER_PACK}; i++ )); do
    srun --exact --ntasks=1 --cpus-per-task=${CPUS_PER_GPU} --gres=gpu:1 \\
         --mem=${MEM_PER_GPU_GB}G \\
         --output=${STAGE_DIR}/pack_%j_w\${i}.log \\
         bash ${REPO}/slurm/pack_worker.sh ${STAGE_DIR} w\${i} &
done
wait
echo "=== pack done"
grep -H '^status:' ${STAGE_DIR}/*/seed_*/run_info.yaml 2>/dev/null || true
EOF

echo "${#TASK_LINES[@]} tasks, ${PACKS} packs x ${GPUS_PER_PACK} GPUs -> ${STAGE_DIR}"
submit_packs
