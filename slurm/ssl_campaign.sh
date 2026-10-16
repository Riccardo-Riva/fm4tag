#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# SSL campaign launcher: one slurm job per CELL, seeds back-to-back inside it.
#
# A cell is one line of a cell file:   <name>  <hydra override> <hydra override> ...
# Every training composes --config-name=campaign_ssl, then the cell's overrides,
# then the per-seed bookkeeping overrides.  Blank lines and # comments are
# skipped.  Example (scripts/configs/campaign/*.cells hold the real stages):
#
#   sc_pt_c256   arm=scratch chunk_size=256 datamodule.batch_size=8192 optimizer.lr=6e-4
#   ft_p0        arm=finetune encoder_ckpt=@/path/to/<p0 cell>/seed_42
#
# ${VAR} in a cell line is filled from the environment (unset = error), e.g.
# PRE=<pretraining stage dir> for  encoder_ckpt=@${PRE}/p0/seed_42.
# encoder_ckpt=@<pretrain seed dir> is resolved when the training starts, to
# that run's checkpoints/last.ckpt (the pretraining budget is fixed), so
# downstream cells can be queued behind their pretraining with
# DEPENDENCY=afterok:<jobid>.
#
# Usage:
#   CELLS=scripts/configs/campaign/S1_arch.cells EST_SECONDS_PER_EPOCH=900 bash slurm/ssl_campaign.sh
#   DRY_RUN=1 ...   writes the job scripts and prints the admission arithmetic only
#
# Output (one self-contained run dir per seed, the shape the plot scripts read):
#   slurm/ssl_campaign/<stage>_<TS>/
#     sweep_info.txt, cells (copy of the cell file)
#     <cell>/cell_job.sh, job_out.txt, job_err.txt
#     <cell>/seed_<s>/run_info.yaml, rank_0.{out,err}, outputs/.../.hydra/,
#                     <cell>/version_0/{metrics.csv,checkpoints/}
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${REPO}/.venv
CONFIG_DIR=${REPO}/src/fm4tag/configs
CONFIG=campaign_ssl

CELLS=${CELLS:?set CELLS=<cell file>}
STAGE=${STAGE:-$(basename "${CELLS}" .cells)}
GPU_NODE=${GPU_NODE:-gpu-L40S-open}
EXCLUDE_NODES=${EXCLUDE_NODES:-node82}
NUM_WORKERS=${NUM_WORKERS:-10}
CPUS_PER_TASK=$((NUM_WORKERS + 2))
# Host RAM: ~2.7 GiB per rank + ~0.5 GiB per worker (train + val loaders are both
# persistent) + slack — see the training RAM budget notes.
MEM=${MEM:-32G}
WALLTIME=${WALLTIME:-7-00:00:00}
MAX_EPOCHS=${MAX_EPOCHS:-100}
read -r -a SEEDS <<< "${SEEDS:-42 424 4242 42424}"
DEPENDENCY=${DEPENDENCY:-}
WANDB_MODE=${WANDB_MODE:-}      # e.g. offline for smoke tests
DRY_RUN=${DRY_RUN:-0}
RESUME_SKIP_COMPLETED=${RESUME_SKIP_COMPLETED:-1}

# No default on purpose: a wrong prior silently admits zero (or too many) seeds.
EST_SECONDS_PER_EPOCH=${EST_SECONDS_PER_EPOCH:?set EST_SECONDS_PER_EPOCH (measured s/epoch for this stage)}
EST_STARTUP_SECONDS=1200
EST_RUN_SECONDS=$((MAX_EPOCHS * EST_SECONDS_PER_EPOCH + EST_STARTUP_SECONDS))
TAIL_MARGIN_SECONDS=1800
MIN_SANE_SECONDS=${MIN_SANE_SECONDS:-1200}
GPU_FREE_MIB_REQUIRED=${GPU_FREE_MIB_REQUIRED:-44000}
GPU_WAIT_SECONDS=${GPU_WAIT_SECONDS:-1800}
GPU_MAX_REQUEUES=${GPU_MAX_REQUEUES:-3}

walltime_to_seconds() {
    local t=$1 d=0 a b c
    [[ "${t}" == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
    IFS=: read -r a b c <<< "${t}"
    echo $(( 10#${d} * 86400 + 10#${a} * 3600 + 10#${b} * 60 + 10#${c:-0} ))
}
WALLTIME_SECONDS=$(walltime_to_seconds "${WALLTIME}")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE=${REPO}/slurm/ssl_campaign/${STAGE}_${TIMESTAMP}
WANDB_GROUP=${WANDB_GROUP:-ssl_${STAGE}_${TIMESTAMP}}

# ── Preflight ─────────────────────────────────────────────────────────────────
[[ -f "${CELLS}" ]] || { echo "ERROR: cell file ${CELLS} not found" >&2; exit 1; }
[[ -e "${VENV}/bin/activate" ]] || { echo "ERROR: ${VENV} missing" >&2; exit 1; }
if ! git -C "${REPO}" diff --quiet -- src/fm4tag/configs; then
    echo "ERROR: uncommitted changes under src/fm4tag/configs — the runs would" >&2
    echo "       train a config that sweep_info.txt cannot record. Commit first." >&2
    exit 1
fi
USABLE_FIRST=$(( WALLTIME_SECONDS - TAIL_MARGIN_SECONDS ))
SEEDS_FIT=$(( USABLE_FIRST / EST_RUN_SECONDS ))
echo "admission: ${MAX_EPOCHS} epochs x ${EST_SECONDS_PER_EPOCH} s + ${EST_STARTUP_SECONDS} s" \
     "= $((EST_RUN_SECONDS / 3600)) h per seed; ${WALLTIME} admits at most ${SEEDS_FIT} seed(s)"
if (( SEEDS_FIT < 1 )); then
    echo "ERROR: not even one seed fits the walltime — lower MAX_EPOCHS or raise WALLTIME" >&2
    exit 1
fi

# ${VAR} in a cell line is taken from the environment (e.g. the directory of
# the pretraining stage); an unset one is an error, never an empty string.
for var in $(grep -vE '^\s*#' "${CELLS}" | grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*\}' \
             | tr -d '${}' | sort -u); do
    [[ -n "${!var:-}" ]] || { echo "ERROR: cell file uses \${${var}}, which is not set" >&2; exit 1; }
done
mapfile -t CELL_LINES < <(grep -vE '^\s*(#|$)' "${CELLS}" | envsubst)
mkdir -p "${OUTPUT_BASE}"
cp "${CELLS}" "${OUTPUT_BASE}/cells"
{
    echo "stage:        ${STAGE}"
    echo "submitted:    $(date -Is)"
    echo "repo:         ${REPO}  (git $(git -C "${REPO}" rev-parse --short HEAD))"
    echo "config:       ${CONFIG}"
    echo "partition:    ${GPU_NODE}  (exclude ${EXCLUDE_NODES:-none}), 1 GPU, ${WALLTIME}"
    echo "max_epochs:   ${MAX_EPOCHS}   est s/epoch: ${EST_SECONDS_PER_EPOCH}"
    echo "seeds:        ${SEEDS[*]}"
    echo "dependency:   ${DEPENDENCY:-none}"
    echo "wandb mode:   ${WANDB_MODE:-default}"
    echo "wandb group:  ${WANDB_GROUP}"
    echo "cells:"
    printf '  %s\n' "${CELL_LINES[@]}"
} > "${OUTPUT_BASE}/sweep_info.txt"

# ── One job per cell ──────────────────────────────────────────────────────────
for LINE in "${CELL_LINES[@]}"; do
read -r CELL_NAME CELL_OVERRIDES <<< "${LINE}"
CELL_DIR=${OUTPUT_BASE}/${CELL_NAME}
mkdir -p "${CELL_DIR}"
for ov in ${CELL_OVERRIDES}; do
    if [[ "${ov}" == encoder_ckpt=@* && "${ov}" != encoder_ckpt=@/* ]]; then
        echo "ERROR: ${CELL_NAME}: ${ov} — the @ path must be absolute" >&2
        exit 1
    fi
done
ARM=$(grep -oE '(^| )arm=[a-z]+' <<< " ${CELL_OVERRIDES}" | cut -d= -f2 || true)
ARM=${ARM:-scratch}
# Compose the cell's config now: a typo in an override fails here, before
# anything is submitted; it also names the data files the training reads.
# shellcheck disable=SC2086
if ! DATA_FILES=$("${VENV}/bin/python" "${REPO}/scripts/campaign_cell_files.py" ${CELL_OVERRIDES}); then
    echo "ERROR: cell ${CELL_NAME} does not compose — nothing more submitted" >&2
    exit 1
fi
DATA_FILES=$(echo ${DATA_FILES})
# Shell-quoted for the job script, so Hydra interpolations such as
# ${optimizer.lr} reach Hydra literally instead of being expanded by bash.
# shellcheck disable=SC2086
CELL_OVERRIDES_Q=$(printf '%q ' ${CELL_OVERRIDES})

cat > "${CELL_DIR}/cell_job.sh" << EOF
#!/bin/bash
#SBATCH --job-name=ssl_${STAGE}_${CELL_NAME}
#SBATCH --partition=${GPU_NODE}
$( [[ -n "${EXCLUDE_NODES}" ]] && echo "#SBATCH --exclude=${EXCLUDE_NODES}" )
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${WALLTIME}
#SBATCH --output=${CELL_DIR}/job_out.txt
#SBATCH --error=${CELL_DIR}/job_err.txt
# Generated by slurm/ssl_campaign.sh — edit that, not this.

JOB_START=\$(date +%s)
nvidia-smi
source ${VENV}/bin/activate
$( [[ -n "${WANDB_MODE}" ]] && echo "export WANDB_MODE=${WANDB_MODE}" )

END_TIME=\$(scontrol show job "\${SLURM_JOB_ID}" -o 2>/dev/null \\
           | grep -oE 'EndTime=[^ ]+' | head -1 | cut -d= -f2)
if [[ -n "\${END_TIME}" && "\${END_TIME}" != "Unknown" ]]; then
    DEADLINE=\$(date -d "\${END_TIME}" +%s)
else
    DEADLINE=\$(( JOB_START + ${WALLTIME_SECONDS} ))
fi
fmt_dhms() {
    local s=\$1
    printf '%02d:%02d:%02d:%02d' \$((s / 86400)) \$(((s % 86400) / 3600)) \\
                                 \$(((s % 3600) / 60)) \$((s % 60))
}

# encoder_ckpt=@<pretrain seed dir> → that run's last.ckpt
OVERRIDES=()
for ov in ${CELL_OVERRIDES_Q}; do
    if [[ "\${ov}" == encoder_ckpt=@* ]]; then
        src=\${ov#encoder_ckpt=@}
        ckpt=\$(ls -t "\${src}"/*/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
        if [[ -z "\${ckpt}" ]]; then
            echo "[ckpt] no last.ckpt under \${src} — is the pretraining done?" >&2
            exit 1
        fi
        echo "[ckpt] \${ov} -> \${ckpt}"
        ov="encoder_ckpt=\${ckpt}"
    fi
    OVERRIDES+=("\${ov}")
done

# Warm the node's page cache with one sequential read: the loader's random
# cold reads from NFS are ~8x slower than cached ones.
WARM_START=\$(date +%s)
cat ${DATA_FILES} > /dev/null
echo "[data] page cache warmed in \$(( \$(date +%s) - WARM_START )) s: ${DATA_FILES}"

export GPU_FREE_MIB_REQUIRED=${GPU_FREE_MIB_REQUIRED}
export GPU_WAIT_SECONDS=${GPU_WAIT_SECONDS}
EST_SECONDS=${EST_RUN_SECONDS}
STARTED=0
COMPLETED=0

for SEED in ${SEEDS[*]}; do
    RUN_DIR=${CELL_DIR}/seed_\${SEED}
    INFO=\${RUN_DIR}/run_info.yaml
    if [[ "${RESUME_SKIP_COMPLETED}" == "1" && -f "\${INFO}" ]] \\
       && grep -q '^status: completed\$' "\${INFO}"; then
        echo "[budget] seed \${SEED}: already completed — skipping"
        COMPLETED=\$((COMPLETED + 1))
        continue
    fi
    if ! bash ${REPO}/slurm/gpu_preflight.sh; then
        if (( \${SLURM_RESTART_COUNT:-0} < ${GPU_MAX_REQUEUES} )); then
            echo "[gpu] requeueing for a different allocation"
            scontrol requeue "\${SLURM_JOB_ID}" && sleep 600
        fi
        echo "[gpu] giving up: the card stayed occupied; no seed consumed." >&2
        exit 1
    fi
    USABLE=\$(( DEADLINE - \$(date +%s) - ${TAIL_MARGIN_SECONDS} ))
    if (( USABLE < EST_SECONDS )); then
        echo "[budget] stopping before seed \${SEED}: \$((USABLE / 60)) min usable" \\
             "< \$((EST_SECONDS / 60)) min needed"
        break
    fi
    MAX_TIME=\$(fmt_dhms \${USABLE})
    mkdir -p "\${RUN_DIR}"
    cd "\${RUN_DIR}" || exit 1
    echo "=== seed \${SEED} — max_time \${MAX_TIME}, started \$(date -Is)"
    RUN_START=\$(date +%s)
    STARTED=\$((STARTED + 1))

    srun --kill-on-bad-exit=1 \\
        --output=\${RUN_DIR}/rank_%t.out --error=\${RUN_DIR}/rank_%t.err \\
        fm4tag --config-path=${CONFIG_DIR} --config-name=${CONFIG} \\
        action=fit \\
        trainer.max_epochs=${MAX_EPOCHS} \\
        +trainer.max_time="\${MAX_TIME}" \\
        datamodule.num_workers=${NUM_WORKERS} \\
        "\${OVERRIDES[@]}" \\
        seed=\${SEED} \\
        experiment_name=${CELL_NAME} \\
        output_dir=\${RUN_DIR} \\
        loggers.wandb.group=${WANDB_GROUP} \\
        loggers.wandb.job_type=${ARM} \\
        loggers.wandb.name=${CELL_NAME}_seed\${SEED}
    RC=\$?
    RUN_SECONDS=\$(( \$(date +%s) - RUN_START ))

    if (( RC != 0 )); then
        STATUS=failed
    elif (( RUN_SECONDS * 100 >= USABLE * 98 )); then
        STATUS=truncated
    else
        STATUS=completed
        COMPLETED=\$((COMPLETED + 1))
    fi
    cat > "\${INFO}" << INFO
status: \${STATUS}
exit_code: \${RC}
seed: \${SEED}
elapsed_seconds: \${RUN_SECONDS}
budget_seconds: \${USABLE}
started: "\$(date -Is -d "@\${RUN_START}")"
finished: "\$(date -Is)"
job_id: "\${SLURM_JOB_ID}"
node: "\$(hostname)"
stage: ${STAGE}
cell: ${CELL_NAME}
INFO
    echo "=== seed \${SEED} — \${STATUS} (rc \${RC}) in \$((RUN_SECONDS / 60)) min"

    if (( RC != 0 && RUN_SECONDS < ${MIN_SANE_SECONDS} )); then
        echo "[budget] aborting the cell: failed after \${RUN_SECONDS}s (configuration error)" >&2
        exit 1
    fi
    MEASURED=\$(( RUN_SECONDS * 115 / 100 ))
    if (( MEASURED > EST_SECONDS )); then
        EST_SECONDS=\${MEASURED}
    fi
done

echo "=== cell done: \${STARTED} started, \${COMPLETED} completed"
grep -H '^status:' ${CELL_DIR}/seed_*/run_info.yaml 2>/dev/null || true
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${CELL_NAME}: ${CELL_OVERRIDES}"
else
    for v in $(env | grep -o '^SLURM_[A-Z_]*'); do unset "$v"; done
    ( cd "${CELL_DIR}" && sbatch ${DEPENDENCY:+--dependency=${DEPENDENCY}} cell_job.sh )
fi
done

echo
echo "${#CELL_LINES[@]} cells under ${OUTPUT_BASE}"
echo "Progress:  grep -H '^status:' ${OUTPUT_BASE}/*/seed_*/run_info.yaml"
