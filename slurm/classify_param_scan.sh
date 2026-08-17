#!/bin/bash
#SBATCH --partition=cpu-all
#SBATCH --mem=4G

# ══════════════════════════════════════════════════════════════════════════════
# Parameter scan WITH SEED REPETITIONS — classify from scratch.
#
# Same experiment as classify_from_scratch.sh (phase=finetune, encoder_ckpt=null,
# so backbone + aggregator + head all train jointly from random init), with one
# structural difference:
#
#   classify_from_scratch.sh :  one slurm job = one training.
#   this script              :  one slurm job = one GRID CELL, inside which
#                               trainings are repeated with different SEEDS
#                               back-to-back until the walltime runs out.
#
# The point of the repetitions is the ERROR BAR: several independent runs of the
# same configuration give a spread on the best validation loss, so a plot like
# notebooks/val_loss_curves.ipynb can show mean ± spread per cell instead of a
# single sample that may be a lucky (or unlucky) seed.
#
# ── The scan is generic ───────────────────────────────────────────────────────
# The scanned quantity is not hard-coded to the augmentation strength: SCAN_PARAM
# is any Hydra key and SCAN_VALUES its values (see "Grid definitions").  A second
# axis (SCAN2_*, defaulted to the aggregator's contrastive temperature) is
# crossed with the first.  Cells = |SCAN_VALUES| x |SCAN2_VALUES|, one slurm job
# each.
#
# ── Output layout ─────────────────────────────────────────────────────────────
#   classify_param_scan/run_<TS>/
#     sweep_info.txt              settings + the cell list, for later reference
#     <cell_name>/                one slurm job (one grid cell)
#       cell_job.sh               the generated job script
#       job_out.txt, job_err.txt  the seed loop's own bookkeeping
#       seed_<seed>/              ONE training — a self-contained run directory
#         run_info.yaml           status / elapsed / exit code / hyper-params
#         rank_*.out, rank_*.err
#         outputs/<date>/<time>/.hydra/config.yaml
#         <experiment_name>/version_0/metrics.csv
#
# `seed_<seed>/` has exactly the shape scripts/plot_val_curves.py expects from a
# run directory (it globs `outputs/*/*/.hydra/config.yaml` and
# `*/version_*/metrics.csv`).  That script takes a whole cell as one entry —
# `cell: <cell_dir>` averages its seeds and turns the spread into a band plus an
# error bar on the best value.  Template config, ready for the `root:` of this
# sweep:  scripts/configs/val_loss_lambda_seed_scan.yaml
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash slurm/classify_param_scan.sh            # submit
#   DRY_RUN=1 bash slurm/classify_param_scan.sh  # write job scripts, submit nothing
#
# TRANSFORMER_TYPE, MAX_EPOCHS, WALLTIME, SEEDS and SCAN_VALUES can be overridden
# from the environment (defaults below are unchanged), which is how a short pilot
# is run without editing this file.  Note the admission test needs
# MAX_EPOCHS x EST_SECONDS_PER_EPOCH of headroom before it will start a seed, so
# a short WALLTIME needs a small MAX_EPOCHS or no training will ever start:
#
#   TRANSFORMER_TYPE=rowcol_concat SCAN_VALUES="0.1" SEEDS="42 424" \
#     MAX_EPOCHS=3 WALLTIME=12:00:00 bash slurm/classify_param_scan.sh
#
# Resubmitting the SAME cell directory (e.g. after a node failure) reuses the
# seeds that already completed instead of redoing them — see RESUME_SKIP_COMPLETED.
# ══════════════════════════════════════════════════════════════════════════════

# ── Settings ──────────────────────────────────────────────────────────────────

# Partition(s) the cells may land on.  Both available pools are 2 nodes x 8 GPUs
# with a 7-day limit; A40 nodes have 80 CPUs against L40S's 192, which at
# CPUS_PER_TASK x GPU_NUM = 24 caps A40 at 3 concurrent cells per node before
# CPUs rather than GPUs become the binding constraint.
#
# A COMMA-SEPARATED LIST lets slurm place each cell on whichever partition frees
# up first, which is what you want when one pool is saturated and the grid would
# otherwise sit pending for days.  The cost, and it is a real one: the sweep then
# straddles two hardware classes, so the seed spread this script exists to
# measure picks up a hardware term, and the per-epoch prior below (measured on
# L40S) no longer holds on the A40 cells.  Loss-vs-epoch curves are unaffected —
# the arithmetic is the same on both — but wall-clock, and therefore how many
# seeds a cell completes, is not.  Prefer a single partition when a comparison
# depends on timing; use the list when it depends only on the curves.
GPU_NODE=${GPU_NODE:-gpu-L40S-open}

# GPUs per TRAINING, all on one node.  Lightning selects DDP for devices > 1 and
# srun launches one rank per GPU (each rank binds to CUDA_VISIBLE_DEVICES[local_rank]
# — the scheduler's variable is read, never overwritten, as DICLUB requires).
#
# The seeds run SEQUENTIALLY over these GPUs, one training at a time, rather than
# one independent single-GPU training per card.  With the dataloader as the
# bottleneck, 2-rank DDP scales at ~1.85x (two ranks = two independent dataloaders),
# so the GPU-hours per training are nearly the same either way — but the DDP form
# finishes a seed every ~24-44 h instead of every ~45-82 h, which
#   * packs better into a fixed walltime (smaller quantum = less wasted tail),
#   * delivers the first seeds days earlier, and
#   * keeps the gradient batch at GPU_NUM x BATCH_SIZE = 2048, identical to the
#     earlier scans (run_20260811_105149 et al.), so the new points are directly
#     comparable to them.
GPU_NUM=2

# Dataloader workers PER RANK (see classify_from_scratch.sh for the profiling
# that sets this: a lone job is already data-bound at 12 workers, and 10 trades a
# little of that margin for less contention when jobs share a node — which, with
# up to 6 cells of this sweep on 2 nodes, is exactly what happens here).
NUM_WORKERS=10

REPO=/storage3/DSIP/rriva/research/fm4tag
VENV=${REPO}/.venv
CONFIG_DIR=${REPO}/src/fm4tag/configs

CONFIG=default.yaml
MAX_EPOCHS=${MAX_EPOCHS:-100}

# Walltime per CELL (not per training).  The seed loop keeps starting trainings
# until the time left is smaller than one training's estimated duration, so this
# is what buys the seeds: at ~24-44 h per training, 7 days gives 3-7 seeds.
WALLTIME=${WALLTIME:-7-00:00:00}

# Derived from WALLTIME so the two can never drift apart.
# Accepts [D-]HH:MM:SS, slurm's own format.
walltime_to_seconds() {
    local t=$1 d=0 a b c
    [[ "${t}" == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
    IFS=: read -r a b c <<< "${t}"
    echo $(( 10#${d} * 86400 + 10#${a} * 3600 + 10#${b} * 60 + 10#${c:-0} ))
}
WALLTIME_SECONDS=$(walltime_to_seconds "${WALLTIME}")

MEM=32G

CPUS_PER_TASK=$((NUM_WORKERS + 2))
if (( GPU_NUM * CPUS_PER_TASK > 80 )); then
    echo "NOTE: ${GPU_NUM} ranks x ${CPUS_PER_TASK} cpus = $((GPU_NUM * CPUS_PER_TASK)) CPUs"
    echo "      > the 80 CPUs of a gpu-A40 node.  Fine while GPU_NODE is L40S-only"
    echo "      (192 CPUs), but adding gpu-A40 to GPU_NODE would make the job"
    echo "      A40-ineligible and it would just sit pending."
fi

# Global timestamp for this sweep (seconds precision)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE=${REPO}/slurm/classify_param_scan/run_${TIMESTAMP}

# wandb: one group per invocation of this script; every training of every cell —
# all cells, all seeds — clusters into one expandable row in the wandb UI.
# job_type is the fixed phase label ("scratch", not "finetune", even though the
# underlying hydra phase is finetune).
WANDB_GROUP="scan_${TIMESTAMP}"
WANDB_JOB_TYPE="scratch"

# ── Seed repetitions and the time budget ──────────────────────────────────────

# Seeds are consumed IN ORDER until the walltime runs out, so the list is an
# upper bound, not a promise.  `seed` is a single top-level config key: the runner
# passes it to L.seed_everything(workers=True) (model init, augmentation RNG,
# dataloader worker seeding) and it is interpolated into datamodule.seed (batch
# composition / shuffling), so one override reseeds the entire training.
read -r -a SEEDS <<< "${SEEDS:-42 424 4242 42424 424242 4242424 42424242 424242424}"

# Duration prior for a training, used to decide whether the next seed still fits.
# It stays a FLOOR for the whole job even after real durations are known: an
# early-stopped 24 h run must not tempt the loop into starting a seed that then
# needs 44 h.
# Re-measure this if MAX_EPOCHS, BATCH_SIZE, the encoder or the partition change.
#
# 2800 s (~47 min) = the 2772 s/epoch measured on rowcol_CONCAT, bs 2048, 2 L40S
# (run_20260817_171303, both seeds, epoch boundaries 2775 / 2770 s), plus ~1%.
# Startup measured at ~20 s there, but 1200 is kept as slack for a cold FS cache.
#
# NOTE this prior is encoder-specific and the two encoders differ by ~16%:
#   rowcol_concat_v0  2772 s/epoch  (measured, see above)
#   rowcol_v0         ~3300 s/epoch (the previous value, per_token)
# Getting it WRONG IN THE HIGH DIRECTION silently costs seeds rather than
# failing: at 3300 with MAX_EPOCHS=100 a 7-day cell admits ONE seed, because
# after the first training the remaining 90.5 h is just under the 92 h the prior
# demands — and one seed produces no error bar, which is the point of this script.
EST_SECONDS_PER_EPOCH=${EST_SECONDS_PER_EPOCH:-2800}
EST_STARTUP_SECONDS=1200
EST_RUN_SECONDS=$((MAX_EPOCHS * EST_SECONDS_PER_EPOCH + EST_STARTUP_SECONDS))

# Reserved at the end of the allocation for the last training's teardown
# (checkpoint write + wandb sync).  Also kept clear of the deadline when a
# training's max_time is computed.
TAIL_MARGIN_SECONDS=1800

# A training that exits non-zero faster than this is a configuration error, not a
# transient one: aborting the cell avoids burning the whole seed list on it.
MIN_SANE_SECONDS=600

# 1 = a resubmitted cell skips seeds whose run_info.yaml says `status: completed`.
RESUME_SKIP_COMPLETED=${RESUME_SKIP_COMPLETED:-1}

# ── Grid definitions ──────────────────────────────────────────────────────────

# Encoder config group: the CLI gets `encoders=${TRANSFORMER_TYPE}_v0`, so this
# must name a file in src/fm4tag/configs/encoders/ minus the _v0 suffix.
# e.g. rowcol | rowcol_concat | col | row
TRANSFORMER_TYPE=${TRANSFORMER_TYPE:-rowcol}

# PER-RANK batch size.  Also a MODEL hyper-parameter, not just a throughput knob:
# the row (intersample) attention makes every sample's output depend on its
# batch-mates, and the ISAB summarises only the rank's own shard (no all-gather),
# so this is the intersample context each sample actually sees.
BATCH_SIZE=2048

# Learning rate for the whole model (`finetune.lr` interpolates `optimizer.lr`
# and backbone_lr is forced to null below — everything is randomly initialised).
LR=3e-4

# CE on the head output, SupCon on the aggregator output.
#
# JET_CONTRASTIVE=0 is a structurally different run, not just a reweighting:
# FinetuneModule._compute_loss only calls _encode_jet_views() when the weight is
# > 0, and that is the sole consumer of `views`.  So with 0 the augmentation
# pipeline never executes and SCAN_PARAM=lambda (which sets CutMix/Mixup on
# views[0]) becomes INERT — a lambda scan would submit N identical trainings.
# Scan a single dummy value and spend the budget on seeds instead.  The same
# goes for SCAN2_PARAM=...jet_contrastive_loss.temperature.
# It is also cheaper per step: one encoder pass instead of two (see
# EST_SECONDS_PER_EPOCH, which is calibrated for the jet_contrastive>0 path).
CROSS_ENTROPY=${CROSS_ENTROPY:-1.0}
JET_CONTRASTIVE=${JET_CONTRASTIVE:-1.0}

# ── Scan axis A (one slurm job per value, crossed with axis B) ────────────────
# Any Hydra key.  SCAN_LABEL is the short tag used in directory and wandb names.
#
# Default: `lambda`, the augmentation strength, which feeds BOTH CutMix.lam and
# Mixup.lam of views[0] (views[1] is Identity and takes no parameter).  It is a
# Python keyword, but Hydra parses overrides as strings, so `lambda=<x>` is fine.
#
# To scan something else, edit these three lines, e.g.
#   SCAN_PARAM="optimizer.lr";  SCAN_LABEL="lr";  SCAN_VALUES=(1e-4 3e-4 1e-3)
SCAN_PARAM="lambda"
SCAN_LABEL="lam"
read -r -a SCAN_VALUES <<< "${SCAN_VALUES:-0.1 0.2 0.25 0.3 0.35 0.4 0.5}"

# ── Scan axis B ───────────────────────────────────────────────────────────────
# Temperature of the aggregator's (jet-level) contrastive loss — the logit scale
# in MultiViewSupConLoss.  One value for now, so the grid is 6 cells; adding
# values here multiplies the number of jobs.
SCAN2_PARAM="finetune.jet_contrastive_loss.temperature"
SCAN2_LABEL="temp"
SCAN2_VALUES=(0.1)

# Extra Hydra overrides appended verbatim to every training, e.g.
#   EXTRA_OVERRIDES=("finetune.warmup_frac=0.05" "trainer.gradient_clip_val=1.0")
EXTRA_OVERRIDES=()

# DRY_RUN=1 writes every job script but submits nothing.
DRY_RUN=${DRY_RUN:-0}

# ── Preflight ─────────────────────────────────────────────────────────────────

for path in "${VENV}/bin/activate" "${CONFIG_DIR}/${CONFIG}"; do
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: ${path} not found — nothing submitted." >&2
        exit 1
    fi
done
if (( ${#SEEDS[@]} == 0 )); then
    echo "ERROR: SEEDS is empty — nothing to run." >&2
    exit 1
fi

N_CELLS=$(( ${#SCAN_VALUES[@]} * ${#SCAN2_VALUES[@]} ))
MAX_SEEDS=${#SEEDS[@]}

mkdir -pv "${OUTPUT_BASE}"

echo "Grid: ${#SCAN_VALUES[@]} ${SCAN_PARAM} x ${#SCAN2_VALUES[@]} ${SCAN2_PARAM} = ${N_CELLS} cells"
echo "      each cell = 1 job, ${GPU_NUM} GPUs, ${WALLTIME} walltime,"
echo "      up to ${MAX_SEEDS} seeds run back-to-back (~$((EST_RUN_SECONDS / 3600)) h estimated each)"

# ── Sweep-level record ────────────────────────────────────────────────────────

SWEEP_INFO=${OUTPUT_BASE}/sweep_info.txt
{
    echo "sweep:        ${TIMESTAMP}"
    echo "submitted:    $(date -Is)"
    echo "repo:         ${REPO}  (git $(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo unknown))"
    echo "config:       ${CONFIG}"
    echo "encoders:     ${TRANSFORMER_TYPE}_v0"
    echo "partition:    ${GPU_NODE}   gpus/training: ${GPU_NUM}   walltime: ${WALLTIME}"
    echo "max_epochs:   ${MAX_EPOCHS}"
    echo "batch_size:   ${BATCH_SIZE} per rank (gradient batch $((GPU_NUM * BATCH_SIZE)))"
    echo "lr:           ${LR}"
    echo "loss:         cross_entropy=${CROSS_ENTROPY}  jet_contrastive=${JET_CONTRASTIVE}"
    echo "scan A:       ${SCAN_PARAM} = ${SCAN_VALUES[*]}"
    echo "scan B:       ${SCAN2_PARAM} = ${SCAN2_VALUES[*]}"
    echo "seeds:        ${SEEDS[*]}  (in order, until the walltime runs out)"
    echo "extra:        ${EXTRA_OVERRIDES[*]-}"
    echo "wandb group:  ${WANDB_GROUP}"
    echo
    echo "cells:"
} > "${SWEEP_INFO}"

# ── Submit one job per cell ───────────────────────────────────────────────────

CELL_ID=0

for SCAN_VALUE in "${SCAN_VALUES[@]}"; do
for SCAN2_VALUE in "${SCAN2_VALUES[@]}"; do

CELL_NAME="scratch_bs_${BATCH_SIZE}_lr_${LR}_ce_${CROSS_ENTROPY}_jc_${JET_CONTRASTIVE}_${SCAN_LABEL}_${SCAN_VALUE}_${SCAN2_LABEL}_${SCAN2_VALUE}"
CELL_DIR=${OUTPUT_BASE}/${CELL_NAME}

mkdir -pv "${CELL_DIR}"
echo "  ${CELL_NAME}" >> "${SWEEP_INFO}"

# unfreeze_at_epoch=0: train the backbone + aggregator from step 0 — freezing
# would be pointless (and harmful) with random init.
# finetune.backbone_lr=null: fall back to the head lr; the reduced backbone_lr is
# meant for pretrained weights only.
cat > "${CELL_DIR}/cell_job.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${CELL_NAME}
#SBATCH --partition=${GPU_NODE}
#SBATCH --nodes=1
#SBATCH --gres=gpu:${GPU_NUM}
#SBATCH --ntasks-per-node=${GPU_NUM}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${WALLTIME}
#SBATCH --output=${CELL_DIR}/job_out.txt
#SBATCH --error=${CELL_DIR}/job_err.txt

# One grid cell: the same configuration trained once per seed, sequentially,
# for as many seeds as the walltime allows.  Generated by
# slurm/classify_param_scan.sh — edit that, not this.

JOB_START=\$(date +%s)
nvidia-smi

source ${VENV}/bin/activate

# P2P broken on this cluster; use shared memory transport instead.
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ── Wall-clock deadline ───────────────────────────────────────────────────────
# The scheduler's EndTime is authoritative (it accounts for the queue time
# already spent, and for any admin adjustment of the limit); the requested
# walltime is only the fallback.
END_TIME=\$(scontrol show job "\${SLURM_JOB_ID}" -o 2>/dev/null \\
           | grep -oE 'EndTime=[^ ]+' | head -1 | cut -d= -f2)
if [[ -n "\${END_TIME}" && "\${END_TIME}" != "Unknown" ]]; then
    DEADLINE=\$(date -d "\${END_TIME}" +%s)
else
    DEADLINE=\$(( JOB_START + ${WALLTIME_SECONDS} ))
fi
echo "[budget] deadline \$(date -d "@\${DEADLINE}" -Is), \$(( (DEADLINE - JOB_START) / 3600 )) h from now"

# Lightning's trainer.max_time wants DD:HH:MM:SS.
fmt_dhms() {
    local s=\$1
    printf '%02d:%02d:%02d:%02d' \$((s / 86400)) \$(((s % 86400) / 3600)) \\
                                 \$(((s % 3600) / 60)) \$((s % 60))
}

SEEDS=(${SEEDS[*]})
EXTRA_OVERRIDES=(${EXTRA_OVERRIDES[*]-})
RESUME_SKIP_COMPLETED=${RESUME_SKIP_COMPLETED}

# Per-training duration estimate.  Starts at the prior from the submitter and is
# raised — never lowered — by what the completed trainings of this cell actually
# took, so a run longer than expected tightens the admission test for the next.
EST_SECONDS=${EST_RUN_SECONDS}

STARTED=0
COMPLETED=0

for SEED in "\${SEEDS[@]}"; do

    RUN_DIR=${CELL_DIR}/seed_\${SEED}
    INFO=\${RUN_DIR}/run_info.yaml

    if [[ "\${RESUME_SKIP_COMPLETED}" == "1" && -f "\${INFO}" ]] \\
       && grep -q '^status: completed\$' "\${INFO}"; then
        echo "[budget] seed \${SEED}: already completed in a previous job — skipping"
        COMPLETED=\$((COMPLETED + 1))
        continue
    fi

    USABLE=\$(( DEADLINE - \$(date +%s) - ${TAIL_MARGIN_SECONDS} ))
    if (( USABLE < EST_SECONDS )); then
        echo "[budget] stopping before seed \${SEED}: \$((USABLE / 60)) min usable" \\
             "< \$((EST_SECONDS / 60)) min needed for one training"
        break
    fi
    echo "[budget] seed \${SEED}: \$((USABLE / 60)) min usable," \\
         "\$((EST_SECONDS / 60)) min estimated per training"

    mkdir -p "\${RUN_DIR}"
    cd "\${RUN_DIR}" || exit 1

    # Hard cap so the LAST training of the cell ends gracefully (checkpoint +
    # metrics.csv + wandb sync) instead of being SIGKILLed at the walltime.
    # Under the admission test above it should never bind — if it does, the run
    # is truncated and run_info.yaml records it as such.
    MAX_TIME=\$(fmt_dhms \${USABLE})

    echo "=== seed \${SEED} — training \$((STARTED + 1)), max_time \${MAX_TIME}, started \$(date -Is)"
    RUN_START=\$(date +%s)
    STARTED=\$((STARTED + 1))

    srun \\
        --kill-on-bad-exit=1 \\
        --output=\${RUN_DIR}/rank_%t.out \\
        --error=\${RUN_DIR}/rank_%t.err \\
        fm4tag \\
        --config-path=${CONFIG_DIR} \\
        --config-name=${CONFIG} \\
        phase=finetune action=fit \\
        encoder_ckpt=null \\
        finetune.unfreeze_at_epoch=0 \\
        finetune.backbone_lr=null \\
        encoders=${TRANSFORMER_TYPE}_v0 \\
        trainer.devices=${GPU_NUM} \\
        trainer.max_epochs=${MAX_EPOCHS} \\
        +trainer.max_time="\${MAX_TIME}" \\
        datamodule.num_workers=${NUM_WORKERS} \\
        datamodule.batch_size=${BATCH_SIZE} \\
        optimizer.lr=${LR} \\
        finetune.loss_weights.cross_entropy=${CROSS_ENTROPY} \\
        finetune.loss_weights.jet_contrastive=${JET_CONTRASTIVE} \\
        ${SCAN_PARAM}=${SCAN_VALUE} \\
        ${SCAN2_PARAM}=${SCAN2_VALUE} \\
        seed=\${SEED} \\
        experiment_name=${CELL_NAME} \\
        output_dir=\${RUN_DIR} \\
        loggers.wandb.group=${WANDB_GROUP} \\
        loggers.wandb.job_type=${WANDB_JOB_TYPE} \\
        loggers.wandb.name=${SCAN_LABEL}${SCAN_VALUE}_${SCAN2_LABEL}${SCAN2_VALUE}_seed\${SEED} \\
        "\${EXTRA_OVERRIDES[@]}"
    RC=\$?

    RUN_SECONDS=\$(( \$(date +%s) - RUN_START ))

    # A run that used (almost) all the time it was given hit max_time rather than
    # max_epochs or early stopping: its curve is cut short, so mark it and let
    # the analysis decide whether to keep it.
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
max_time: "\${MAX_TIME}"
started: "\$(date -Is -d "@\${RUN_START}")"
finished: "\$(date -Is)"
job_id: "\${SLURM_JOB_ID}"
node: "\$(hostname)"
cell: ${CELL_NAME}
${SCAN_PARAM}: ${SCAN_VALUE}
${SCAN2_PARAM}: ${SCAN2_VALUE}
batch_size: ${BATCH_SIZE}
gpus: ${GPU_NUM}
lr: ${LR}
max_epochs: ${MAX_EPOCHS}
INFO

    echo "=== seed \${SEED} — \${STATUS} (rc \${RC}) in" \\
         "\$((RUN_SECONDS / 3600))h \$(((RUN_SECONDS / 60) % 60))m"

    if (( RC != 0 && RUN_SECONDS < ${MIN_SANE_SECONDS} )); then
        echo "[budget] aborting the cell: the training failed after only" \\
             "\${RUN_SECONDS}s — that is a configuration error, not a transient." >&2
        exit 1
    fi

    # Raise the estimate if this run was slower than assumed (15% headroom for
    # the seed-to-seed spread in how late early stopping fires).
    MEASURED=\$(( RUN_SECONDS * 115 / 100 ))
    if (( MEASURED > EST_SECONDS )); then
        echo "[budget] raising the per-training estimate to \$((MEASURED / 60)) min"
        EST_SECONDS=\${MEASURED}
    fi

done

echo "=== cell done: \${STARTED} trainings started, \${COMPLETED} completed," \\
     "\$(( (\$(date +%s) - JOB_START) / 3600 ))h used"
grep -H '^status:' ${CELL_DIR}/seed_*/run_info.yaml 2>/dev/null
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] cell ${CELL_ID}: ${CELL_NAME}"
else
    echo "Submitting cell ${CELL_ID}: ${CELL_NAME}"
    cd "${CELL_DIR}"
    sbatch cell_job.sh
fi

((CELL_ID++))

done
done

echo
echo "Submitted ${CELL_ID} cells (up to $((CELL_ID * MAX_SEEDS)) trainings) under ${OUTPUT_BASE}"
echo "Progress:  grep -H '^status:' ${OUTPUT_BASE}/*/seed_*/run_info.yaml"
