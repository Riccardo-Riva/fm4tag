#!/bin/bash
#SBATCH --partition=cpu-all
#SBATCH --mem=4G

# ── Settings ──────────────────────────────────────────────────────────────────
# Supervised classification from scratch: phase=finetune with no pretrained
# encoder (encoder_ckpt=null), so the whole backbone + head is trained jointly.

GPU_NODE=gpu-L40S-open,gpu-A40

# DDP world size: GPUs per run, all on one node.  Both GPU partitions have 8
# physical GPUs per node (gpu-A40 = node81-82, gpu-L40S-open = node101-102), so 8
# is the ceiling.  Lightning selects DDP automatically for devices > 1; srun
# launches one rank per GPU, and each rank binds to CUDA_VISIBLE_DEVICES[local_rank]
# — the scheduler's variable is read, never overwritten, as DICLUB requires.
#
# We request whole GPUs (--gres=gpu:N) rather than the cluster's preferred
# --gres=shard:N.  DICLUB allows this "only if really needed", and DDP is that
# case: each rank needs its own dedicated device.  A shard is 1/12 of a physical
# GPU (shard:tesla:96 over 8 GPUs), and shard allocation gives no guarantee that N
# shards map to N *distinct* GPUs — two ranks landing on one card would contend for
# the same SMs and memory, which defeats the point of data-parallel training.
# For single-GPU runs, prefer --gres=shard:12 (= one GPU's worth) to be a good
# citizen on the shared queues.
#
# 2 (not 3+) divides evenly into each node's 8 GPUs, so grid jobs from the same
# sweep pack without stranding GPUs (3+3 leaves 2 idle -- observed on node102 in
# run_20260803_162005, where it also meant 2 of our own jobs' dataloaders were
# competing on one node, see NUM_WORKERS below).
GPU_NUM=2

# Dataloader workers PER RANK (each DDP rank runs its own dataloader, so the node
# carries GPU_NUM x NUM_WORKERS worker processes).
#
# Single-job profiling (slurm/profiling/run_20260713_161724) puts one training
# step at ~319 ms of compute (fwd 117 + bwd 183 + clip 17 + opt 1) while the
# dataloader with 12 workers delivers a batch every ~422 ms -- i.e. a LONE job is
# already DATA-BOUND at 12 workers, and fewer would starve it further.
#
# That profiling assumed no node-mate. In run_20260803_162005, two of our own
# GPU_NUM=3 grid jobs landed on the same physical node (72 combined dataloader
# workers hammering the same storage3 mount), and one rank measurably stalled
# waiting on data while its DDP peers busy-spun on the collective. 10 trades a
# little of a solo job's margin for less contention when jobs (ours or another
# user's) share a node -- which on this cluster is the common case, not the
# exception.
NUM_WORKERS=10

REPO=/storage3/DSIP/rriva/research/fm4tag
VENV=${REPO}/.venv
CONFIG_DIR=${REPO}/src/fm4tag/configs

#CONFIG=default.yaml
CONFIG=rowcol_concat_test.yaml
MAX_EPOCHS=50

# gpu-A40 nodes have only 80 CPUs (gpu-L40S-open have 192).  Slurm never schedules
# a job whose ntasks x cpus-per-task exceeds the node's CPU count, so a too-greedy
# NUM_WORKERS silently makes the job A40-ineligible and it just sits pending.
CPUS_PER_TASK=$((NUM_WORKERS + 2))
if (( GPU_NUM * CPUS_PER_TASK > 80 )); then
    echo "WARNING: ${GPU_NUM} ranks x ${CPUS_PER_TASK} cpus = $((GPU_NUM * CPUS_PER_TASK)) CPUs"
    echo "         > the 80 CPUs on gpu-A40 — this job can only land on gpu-L40S-open."
    echo "         Set NUM_WORKERS=$((80 / GPU_NUM - 2)) or lower to keep gpu-A40 eligible."
fi

# Global timestamp for this sweep (seconds precision)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE=${REPO}/slurm/classify_from_scratch/run_${TIMESTAMP}

mkdir -pv "${OUTPUT_BASE}"

# wandb: one group per invocation of this script, so every run this sweep
# submits clusters into a single expandable row in the wandb UI. job_type is
# a fixed phase label independent of the timestamp — this is "scratch", not
# "finetune", even though the underlying hydra phase is finetune (see below).
WANDB_GROUP="scratch_${TIMESTAMP}"
WANDB_JOB_TYPE="scratch"

# ── Grid definitions ──────────────────────────────────────────────────────────

# Encoder config group: the CLI gets `encoders=${TRANSFORMER_TYPE}_v0`, so this
# must name a file in src/fm4tag/configs/encoders/ minus the _v0 suffix.
#   rowcol         -> rowcol_v0         (per-token ISAB, dim 16)
#   rowcol_concat  -> rowcol_concat_v0  (one ISAB over the concatenated N*dim
#                                        vector; measured 2.17M params in 43.2 ms
#                                        against rowcol_v0's 0.06M in 54.8 ms —
#                                        strictly more model for less time)
# Keep CONFIG in sync: rowcol_concat_test.yaml already defaults to the concat
# encoder group, but the CLI override above is what actually decides.
TRANSFORMER_TYPES=("rowcol_concat")

# PER-RANK batch size.  Lightning injects a DistributedSampler under DDP, so each
# rank's dataloader yields BATCH_SIZE samples and the gradient batch is
# GPU_NUM x BATCH_SIZE.  Two things follow, and both matter here:
#
#  1. Intersample (row) attention does NOT all-gather across ranks — each rank's
#     ISAB summarises only its own shard.  Keeping BATCH_SIZE fixed therefore keeps
#     the intersample context identical to a single-GPU run, which is what we want.
#  2. The gradient batch still grows with GPU_NUM, so the best learning rate shifts.
#     A sweep at GPU_NUM=4 is NOT directly comparable to one at GPU_NUM=1.
BATCH_SIZES=(1024)

# Learning rates to compare.  `finetune.lr` interpolates from `optimizer.lr`, and
# backbone_lr is forced to null below (everything is randomly initialised here),
# so this single value is the learning rate for the whole model.
#LEARNING_RATES=(3e-5 1e-4 3e-4)
LEARNING_RATES=(3e-4)

# "cross_entropy jet_contrastive": CE on the head output, SupCon on the
# aggregator output.  Weights are normalised to unit sum; 0 disables the term.
LOSS_CONFIGS=(
#"1.0 0.0"
#"1.0 0.3"
"1.0 1.0"
)

# Augmentation strength for the contrastive view.  `lambda` in the config feeds
# BOTH CutMix.lam and Mixup.lam of views[0]; views[1] is Identity and takes no
# parameter.  Overridden on the CLI as `lambda=<x>` (a Python keyword, but Hydra
# parses overrides as strings, so this is fine — verified end to end).
#NOISE_LAMBDAS=(0.03 0.1 0.3)
NOISE_LAMBDAS=(0.15 0.2 0.25 0.35 0.4 0.45 0.5 0.6 0.75 0.9)

# `lambda` only ever reaches the model through views[0], and FinetuneModule only
# builds the views when jet_contrastive > 0 (see _compute_loss).  So at jc=0.0
# every NOISE_LAMBDAS value produces a bit-identical run — same seed, same data
# order, same weights.  A literal 3x3 grid would therefore submit 3 copies of the
# same baseline and burn ~2 x 50 epochs of GPU on nothing.
#
# 1 = collapse those duplicates to a single jc=0.0 baseline (3x3 -> 7 real runs).
# 0 = submit the literal cartesian product, duplicates included.
SKIP_REDUNDANT_LAMBDA=${SKIP_REDUNDANT_LAMBDA:-1}

RUN_ID=0

# DRY_RUN=1 writes every job script but submits nothing.
DRY_RUN=${DRY_RUN:-0}

N_CELLS=$(( ${#TRANSFORMER_TYPES[@]} * ${#BATCH_SIZES[@]} * ${#LEARNING_RATES[@]} \
            * ${#LOSS_CONFIGS[@]} * ${#NOISE_LAMBDAS[@]} ))
echo "Grid: ${#TRANSFORMER_TYPES[@]} arch x ${#BATCH_SIZES[@]} bs x ${#LEARNING_RATES[@]} lr" \
     "x ${#LOSS_CONFIGS[@]} loss x ${#NOISE_LAMBDAS[@]} lambda = ${N_CELLS} cells"
if [[ "${SKIP_REDUNDANT_LAMBDA}" == "1" ]]; then
    echo "       (lambda is inert at jet_contrastive=0 — duplicate baselines will be skipped;"
    echo "        set SKIP_REDUNDANT_LAMBDA=0 to submit the literal product)"
fi

# ── Submit grid ───────────────────────────────────────────────────────────────

for TRANSFORMER_TYPE in "${TRANSFORMER_TYPES[@]}"; do
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
for LR in "${LEARNING_RATES[@]}"; do
for LOSS_CONF in "${LOSS_CONFIGS[@]}"; do
for LAMBDA in "${NOISE_LAMBDAS[@]}"; do

read CROSS_ENTROPY JET_CONTRASTIVE <<< "${LOSS_CONF}"

# See SKIP_REDUNDANT_LAMBDA above: with the contrastive term off, the views
# (and hence lambda) never run, so keep only the first lambda as the baseline.
if [[ "${SKIP_REDUNDANT_LAMBDA}" == "1" ]] \
   && awk "BEGIN{exit !(${JET_CONTRASTIVE} == 0)}" \
   && [[ "${LAMBDA}" != "${NOISE_LAMBDAS[0]}" ]]; then
    echo "  skip (duplicate of the jc=0.0 baseline): lambda=${LAMBDA}"
    continue
fi

#RUN_NAME="scratch_${TIMESTAMP}_${TRANSFORMER_TYPE}_ddp${GPU_NUM}_bs_${BATCH_SIZE}_lr_${LR}_ce_${CROSS_ENTROPY}_jc_${JET_CONTRASTIVE}_lam_${LAMBDA}"
RUN_NAME="scratch_${TRANSFORMER_TYPE}_bs_${BATCH_SIZE}_lr_${LR}_ce_${CROSS_ENTROPY}_jc_${JET_CONTRASTIVE}_lam_${LAMBDA}"
OUTPUT_DIR=${OUTPUT_BASE}/${RUN_NAME}_${RUN_ID}

# Short wandb display name: only the axes that vary WITHIN this sweep.
# Phase + timestamp are already carried by WANDB_GROUP, so repeating them
# here would just be noise in the runs table.
WANDB_NAME="${TRANSFORMER_TYPE}_ddp${GPU_NUM}_bs${BATCH_SIZE}_lr${LR}_ce${CROSS_ENTROPY}_jc${JET_CONTRASTIVE}_lam${LAMBDA}_${RUN_ID}"

mkdir -pv "${OUTPUT_DIR}"

# unfreeze_at_epoch=0: train the backbone + aggregator from step 0 — freezing
# would be pointless (and harmful) with random init.
# finetune.backbone_lr=null: fall back to the head lr; the reduced backbone_lr
# is meant for pretrained weights only.
cat > "${OUTPUT_DIR}/classify_from_scratch_run.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --partition=${GPU_NODE}
# node82's GPU5 (0000:D1:00.0) fails NVML init (run_20260811_103403) — exclude until fixed.
#SBATCH --exclude=node82
#SBATCH --nodes=1
#SBATCH --gres=gpu:${GPU_NUM}
#SBATCH --ntasks-per-node=${GPU_NUM}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=64G
#SBATCH --output=${OUTPUT_DIR}/out.txt
#SBATCH --error=${OUTPUT_DIR}/err.txt

SECONDS=0
nvidia-smi

source ${VENV}/bin/activate

# P2P broken on this cluster; use shared memory transport instead.
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

srun \\
    --kill-on-bad-exit=1 \\
    --output=${OUTPUT_DIR}/rank_%t.out \\
    --error=${OUTPUT_DIR}/rank_%t.err \\
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
    datamodule.num_workers=${NUM_WORKERS} \\
    datamodule.batch_size=${BATCH_SIZE} \\
    optimizer.lr=${LR} \\
    finetune.loss_weights.cross_entropy=${CROSS_ENTROPY} \\
    finetune.loss_weights.jet_contrastive=${JET_CONTRASTIVE} \\
    lambda=${LAMBDA} \\
    experiment_name=${RUN_NAME} \\
    output_dir=${OUTPUT_DIR} \\
    loggers.wandb.group=${WANDB_GROUP} \\
    loggers.wandb.job_type=${WANDB_JOB_TYPE} \\
    loggers.wandb.name=${WANDB_NAME}

echo "Elapsed: \$((SECONDS/3600))h \$(((SECONDS/60)%60))m \$((SECONDS%60))s"
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] run ${RUN_ID}: ${RUN_NAME}"
else
    echo "Submitting run ${RUN_ID}: ${RUN_NAME}"
    cd "${OUTPUT_DIR}"
    sbatch classify_from_scratch_run.sh
fi

((RUN_ID++))

done
done
done
done
done

echo "Submitted ${RUN_ID} runs under ${OUTPUT_BASE}"
