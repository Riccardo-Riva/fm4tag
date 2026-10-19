#!/bin/bash
# One GPU of a pack job (slurm/ssl_pack.sh): take tasks from the stage's queue
# and run them, one at a time, until none is left that this worker can run.
#
#   pack_worker.sh <stage dir> <worker name>      (inside a 1-GPU srun step)
#
# Queue protocol, all under <stage dir>/.queue/ (NFS-safe: mkdir is atomic):
#   <task>.claim/   claimed; owner = "<job id> <host> <worker>"
#   <task>.done     the training completed
#   <task>.failed   it failed, was truncated, or its prerequisite failed
# A claim whose owner job is no longer in the queue is released and retried.
set -uo pipefail

STAGE_DIR=$1
WORKER=$2
# shellcheck source=/dev/null
source "${STAGE_DIR}/pack.env"   # REPO VENV CONFIG_DIR CONFIG STAGE WANDB_GROUP NUM_WORKERS MIN_SANE_SECONDS
TASKS=${STAGE_DIR}/tasks.tsv
Q=${STAGE_DIR}/.queue
mkdir -p "${Q}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

log() { echo "[$(date -Is)] [${WORKER}] $*"; }
fmt_dhms() {
    local s=$1
    printf '%02d:%02d:%02d:%02d' $((s / 86400)) $(((s % 86400) / 3600)) \
                                 $(((s % 3600) / 60)) $((s % 60))
}
# 0 if the job that claimed the task is still queued or running
claim_alive() {
    local owner
    owner=$(cut -d' ' -f1 "$1/owner" 2>/dev/null) || return 1
    [[ -n "${owner}" ]] && squeue -h -j "${owner}" 2>/dev/null | grep -q .
}
# Pretraining of ANOTHER stage (encoder_ckpt=@/abs/seed dir):
# 0 = completed, 1 = still to come (its job is queued), 2 = failed / missing
external_state() {
    local src=$1 jid
    if grep -q '^status: completed$' "${src}/run_info.yaml" 2>/dev/null; then
        return 0
    fi
    jid=$(cat "$(dirname "${src}")/job_id" 2>/dev/null)
    if [[ -n "${jid}" ]] && squeue -h -j "${jid}" 2>/dev/null | grep -q .; then
        return 1
    fi
    return 2
}

QUICK_FAILS=0
while true; do
    NEXT="" WAIT=0
    while IFS=$'\t' read -r TASK CELL SEED EPOCHS EST AFTER OVS; do
        [[ -z "${TASK}" || "${TASK}" == \#* ]] && continue
        [[ -e "${Q}/${TASK}.done" || -e "${Q}/${TASK}.failed" ]] && continue
        if [[ -d "${Q}/${TASK}.claim" ]]; then
            claim_alive "${Q}/${TASK}.claim" && continue
            log "releasing the stale claim of ${TASK} ($(cat "${Q}/${TASK}.claim/owner" 2>/dev/null))"
            rm -rf "${Q}/${TASK}.claim"
        fi
        if [[ "${AFTER}" != "-" ]]; then
            if [[ -e "${Q}/${AFTER}.failed" ]]; then
                touch "${Q}/${TASK}.failed"
                log "${TASK}: prerequisite ${AFTER} failed — skipped"
                continue
            fi
            if [[ ! -e "${Q}/${AFTER}.done" ]]; then
                [[ -d "${Q}/${AFTER}.claim" ]] && WAIT=1
                continue
            fi
        fi
        EXT=0
        for ov in ${OVS}; do
            if [[ "${ov}" == encoder_ckpt=@/* ]]; then
                external_state "${ov#encoder_ckpt=@}"
                EXT=$?
            fi
        done
        if (( EXT == 1 )); then WAIT=1; continue; fi
        if (( EXT == 2 )); then
            touch "${Q}/${TASK}.failed"
            log "${TASK}: its pretraining in another stage failed or is missing — skipped"
            continue
        fi
        NEED=$(( EPOCHS * EST + 1200 ))
        USABLE=$(( DEADLINE - $(date +%s) - 1800 ))
        (( USABLE < NEED )) && continue   # does not fit what is left of this job
        if mkdir "${Q}/${TASK}.claim" 2>/dev/null; then
            echo "${SLURM_JOB_ID} $(hostname) ${WORKER}" > "${Q}/${TASK}.claim/owner"
            NEXT=${TASK}
            break
        fi
    done < "${TASKS}"

    if [[ -z "${NEXT}" ]]; then
        if (( WAIT )); then
            sleep 120
            continue
        fi
        log "no task left that this worker can run — exiting"
        break
    fi

    RUN_DIR=${STAGE_DIR}/${CELL}/seed_${SEED}
    mkdir -p "${RUN_DIR}"
    read -r -a OV_ARR <<< "${OVS}"
    ARM=scratch
    CKPT_OK=1
    for i in "${!OV_ARR[@]}"; do
        ov=${OV_ARR[$i]}
        [[ "${ov}" == arm=* ]] && ARM=${ov#arm=}
        if [[ "${ov}" == encoder_ckpt=@* ]]; then
            src=${ov#encoder_ckpt=@}
            [[ "${src}" != /* ]] && src=${STAGE_DIR}/${src}/seed_${SEED}
            ckpt=$(ls -t "${src}"/*/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
            if [[ -z "${ckpt}" ]]; then CKPT_OK=0; break; fi
            OV_ARR[$i]="encoder_ckpt=${ckpt}"
            log "${TASK}: encoder_ckpt -> ${ckpt}"
        fi
    done
    if (( ! CKPT_OK )); then
        log "${TASK}: no last.ckpt for ${ov} — failed"
        touch "${Q}/${TASK}.failed"
        continue
    fi

    USABLE=$(( DEADLINE - $(date +%s) - 1800 ))
    MAX_TIME=$(fmt_dhms "${USABLE}")
    log "${TASK}: start (max_time ${MAX_TIME}, GPU ${CUDA_VISIBLE_DEVICES:-?})"
    RUN_START=$(date +%s)
    cd "${RUN_DIR}" || exit 1
    fm4tag --config-path="${CONFIG_DIR}" --config-name="${CONFIG}" \
        action=fit \
        trainer.max_epochs="${EPOCHS}" \
        +trainer.max_time="${MAX_TIME}" \
        datamodule.num_workers="${NUM_WORKERS}" \
        "${OV_ARR[@]}" \
        seed="${SEED}" \
        experiment_name="${CELL}" \
        output_dir="${RUN_DIR}" \
        loggers.wandb.group="${WANDB_GROUP}" \
        loggers.wandb.job_type="${ARM}" \
        loggers.wandb.name="${CELL}_seed${SEED}" \
        > "${RUN_DIR}/rank_0.out" 2> "${RUN_DIR}/rank_0.err"
    RC=$?
    RUN_SECONDS=$(( $(date +%s) - RUN_START ))

    if (( RC != 0 )); then
        STATUS=failed
    elif (( RUN_SECONDS * 100 >= USABLE * 98 )); then
        STATUS=truncated
    else
        STATUS=completed
    fi
    cat > "${RUN_DIR}/run_info.yaml" << INFO
status: ${STATUS}
exit_code: ${RC}
seed: ${SEED}
elapsed_seconds: ${RUN_SECONDS}
budget_seconds: ${USABLE}
started: "$(date -Is -d "@${RUN_START}")"
finished: "$(date -Is)"
job_id: "${SLURM_JOB_ID}"
node: "$(hostname)"
worker: ${WORKER}
stage: ${STAGE}
cell: ${CELL}
task: ${TASK}
INFO
    if [[ "${STATUS}" == completed ]]; then
        touch "${Q}/${TASK}.done"
        QUICK_FAILS=0
    else
        touch "${Q}/${TASK}.failed"
        if (( RUN_SECONDS < MIN_SANE_SECONDS )); then
            QUICK_FAILS=$((QUICK_FAILS + 1))
        fi
    fi
    log "${TASK}: ${STATUS} (rc ${RC}) in $((RUN_SECONDS / 60)) min"
    if (( QUICK_FAILS >= 2 )); then
        log "two tasks failed within ${MIN_SANE_SECONDS}s each — a configuration error; stopping"
        break
    fi
done
