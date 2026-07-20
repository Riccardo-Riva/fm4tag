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
GPU_NUM=2

# Dataloader workers PER RANK (each DDP rank runs its own dataloader, so the node
# carries GPU_NUM x NUM_WORKERS worker processes).
#
# Do not lower this.  Profiling (slurm/profiling/run_20260713_161724) puts one
# training step at ~319 ms of compute (fwd 117 + bwd 183 + clip 17 + opt 1) while
# the dataloader with 12 workers delivers a batch every ~422 ms — i.e. training is
# already DATA-BOUND and the GPU idles ~25% of the time.  Halving the workers would
# roughly halve dataloader throughput and starve the GPU for most of the step.
NUM_WORKERS=12

REPO=/storage3/DSIP/rriva/research/fm4tag
VENV=${REPO}/.venv
CONFIG_DIR=${REPO}/src/fm4tag/configs

CONFIG=default.yaml
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

# ── Grid definitions ──────────────────────────────────────────────────────────

TRANSFORMER_TYPES=("rowcol")

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
LEARNING_RATES=(3e-5 1e-4 3e-4)

# "cross_entropy jet_contrastive": CE on the head output, SupCon on the
# aggregator output.  Weights are normalised to unit sum; 0 disables the term.
LOSS_CONFIGS=(
"1.0 0.0"
"1.0 0.3"
"1.0 1.0"
)

RUN_ID=0

# DRY_RUN=1 writes every job script but submits nothing.
DRY_RUN=${DRY_RUN:-0}

N_RUNS=$(( ${#TRANSFORMER_TYPES[@]} * ${#BATCH_SIZES[@]} * ${#LEARNING_RATES[@]} * ${#LOSS_CONFIGS[@]} ))
echo "Grid: ${#TRANSFORMER_TYPES[@]} arch x ${#BATCH_SIZES[@]} bs x ${#LEARNING_RATES[@]} lr x ${#LOSS_CONFIGS[@]} loss = ${N_RUNS} runs"

# ── Submit grid ───────────────────────────────────────────────────────────────

for TRANSFORMER_TYPE in "${TRANSFORMER_TYPES[@]}"; do
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
for LR in "${LEARNING_RATES[@]}"; do
for LOSS_CONF in "${LOSS_CONFIGS[@]}"; do

read CROSS_ENTROPY JET_CONTRASTIVE <<< "${LOSS_CONF}"

RUN_NAME="scratch_${TIMESTAMP}_${TRANSFORMER_TYPE}_ddp${GPU_NUM}_bs_${BATCH_SIZE}_lr_${LR}_ce_${CROSS_ENTROPY}_jc_${JET_CONTRASTIVE}"
OUTPUT_DIR=${OUTPUT_BASE}/${RUN_NAME}_${RUN_ID}

mkdir -pv "${OUTPUT_DIR}"

# unfreeze_at_epoch=0: train the backbone + aggregator from step 0 — freezing
# would be pointless (and harmful) with random init.
# finetune.backbone_lr=null: fall back to the head lr; the reduced backbone_lr
# is meant for pretrained weights only.
cat > "${OUTPUT_DIR}/classify_from_scratch_run.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --partition=${GPU_NODE}
#SBATCH --nodes=1
#SBATCH --gres=gpu:${GPU_NUM}
#SBATCH --ntasks-per-node=${GPU_NUM}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=128G
#SBATCH --output=${OUTPUT_DIR}/out.txt
#SBATCH --error=${OUTPUT_DIR}/err.txt

SECONDS=0
nvidia-smi

source ${VENV}/bin/activate

# P2P broken on this cluster; use shared memory transport instead.
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

srun \\
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
    experiment_name=${RUN_NAME} \\
    output_dir=${OUTPUT_DIR}

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

echo "Submitted ${RUN_ID} runs under ${OUTPUT_BASE}"
