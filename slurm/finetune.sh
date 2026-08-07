#!/bin/bash
#SBATCH --partition=cpu-all
#SBATCH --mem=4G

# ── Settings ──────────────────────────────────────────────────────────────────
# Fine-tuning from pretrained encoders: phase=finetune with encoder_ckpt set.
# finetune.unfreeze_at_epoch (default 3) holds the pretrained parts at lr=0 for
# a short linear-probe phase, then ramps them onto the cosine schedule at
# optimizer.backbone_lr — DDP-safe (all params in the optimiser from step 0).

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
GPU_NUM=2

# Dataloader workers PER RANK (each DDP rank runs its own dataloader, so the node
# carries GPU_NUM x NUM_WORKERS worker processes).
#
# Do not lower this.  Profiling (slurm/profiling/run_20260713_161724) puts one
# training step at ~319 ms of compute while the dataloader with 12 workers delivers
# a batch every ~422 ms — training is already DATA-BOUND.
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
OUTPUT_BASE=${REPO}/slurm/finetuning/run_${TIMESTAMP}

mkdir -pv "${OUTPUT_BASE}"

# wandb: one group per invocation of this script, so every run this sweep
# submits clusters into a single expandable row in the wandb UI. job_type is
# a fixed phase label independent of the timestamp, so "all finetune runs
# ever" is one filter away regardless of which sweep they came from.
WANDB_GROUP="finetune_${TIMESTAMP}"
WANDB_JOB_TYPE="finetune"

# ── Grid definitions ──────────────────────────────────────────────────────────

# "transformer_type checkpoint" pairs: the architecture must match the one the
# checkpoint was pretrained with.  Checkpoints from pretrain.sh sweeps land in
#   ${REPO}/slurm/pretraining/run_<TS>/<RUN_NAME>_<ID>/<RUN_NAME>/version_0/checkpoints/epochXXX-stepYYY.ckpt
ENCODERS=(
"rowcol ${REPO}/slurm/pretraining/run_20260710_152049/pretrain_20260710_152049_rowcol_bs_1024_c_1.0_jc_1.0_0/pretrain_20260710_152049_rowcol_bs_1024_c_1.0_jc_1.0/version_0/checkpoints/epoch018-step368125.ckpt"
#"col    ${REPO}/slurm/pretraining/run_XXXXXXXX_XXXXXX/CHANGE_ME.ckpt"
)

# PER-RANK batch size.  Under DDP the gradient batch is GPU_NUM x BATCH_SIZE, but
# intersample (row) attention does NOT all-gather across ranks — each rank's ISAB
# summarises only its own shard — so holding BATCH_SIZE fixed keeps the intersample
# context identical to a single-GPU run.  The larger gradient batch does shift the
# best LR, so a GPU_NUM=2 sweep is not directly comparable to a GPU_NUM=1 one.
BATCH_SIZES=(2048)

# Finetuning has TWO learning rates, and they are swept independently:
#   optimizer.lr          — the head (and anything randomly initialised)
#   optimizer.backbone_lr — the pretrained encoders + aggregator, held at 0 until
#                           finetune.unfreeze_at_epoch, then ramped onto the cosine
#                           schedule.  Deliberately much smaller than the head lr.
# finetune.lr / finetune.backbone_lr interpolate from these, so overriding the
# optimizer keys is enough.  BACKBONE_LRS has one entry by default — add more only
# if you want the 2-D grid (it multiplies the run count).
LEARNING_RATES=(1e-5 3e-4 1e-4)
BACKBONE_LRS=(1e-5)

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

N_RUNS=$(( ${#ENCODERS[@]} * ${#BATCH_SIZES[@]} * ${#LEARNING_RATES[@]} * ${#BACKBONE_LRS[@]} * ${#LOSS_CONFIGS[@]} ))
echo "Grid: ${#ENCODERS[@]} ckpt x ${#BATCH_SIZES[@]} bs x ${#LEARNING_RATES[@]} lr x ${#BACKBONE_LRS[@]} blr x ${#LOSS_CONFIGS[@]} loss = ${N_RUNS} runs"

# ── Submit grid ───────────────────────────────────────────────────────────────

for ENCODER in "${ENCODERS[@]}"; do
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
for LR in "${LEARNING_RATES[@]}"; do
for BACKBONE_LR in "${BACKBONE_LRS[@]}"; do
for LOSS_CONF in "${LOSS_CONFIGS[@]}"; do

read TRANSFORMER_TYPE ENCODER_CKPT <<< "${ENCODER}"
read CROSS_ENTROPY JET_CONTRASTIVE <<< "${LOSS_CONF}"

if [[ ! -f "${ENCODER_CKPT}" ]]; then
    echo "Skipping: checkpoint not found: ${ENCODER_CKPT}"
    continue
fi

RUN_NAME="finetune_${TIMESTAMP}_${TRANSFORMER_TYPE}_ddp${GPU_NUM}_bs_${BATCH_SIZE}_lr_${LR}_blr_${BACKBONE_LR}_ce_${CROSS_ENTROPY}_jc_${JET_CONTRASTIVE}"
OUTPUT_DIR=${OUTPUT_BASE}/${RUN_NAME}_${RUN_ID}

# Short wandb display name: only the axes that vary WITHIN this sweep.
# Phase + timestamp are already carried by WANDB_GROUP, so repeating them
# here would just be noise in the runs table.
WANDB_NAME="${TRANSFORMER_TYPE}_ddp${GPU_NUM}_bs${BATCH_SIZE}_lr${LR}_blr${BACKBONE_LR}_ce${CROSS_ENTROPY}_jc${JET_CONTRASTIVE}_${RUN_ID}"

mkdir -pv "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/finetune_run.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --partition=${GPU_NODE}
#SBATCH --nodes=1
#SBATCH --gres=gpu:${GPU_NUM}
#SBATCH --ntasks-per-node=${GPU_NUM}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=96G
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
    "encoder_ckpt='${ENCODER_CKPT}'" \\
    encoders=${TRANSFORMER_TYPE}_v0 \\
    trainer.devices=${GPU_NUM} \\
    trainer.max_epochs=${MAX_EPOCHS} \\
    datamodule.num_workers=${NUM_WORKERS} \\
    datamodule.batch_size=${BATCH_SIZE} \\
    optimizer.lr=${LR} \\
    optimizer.backbone_lr=${BACKBONE_LR} \\
    finetune.loss_weights.cross_entropy=${CROSS_ENTROPY} \\
    finetune.loss_weights.jet_contrastive=${JET_CONTRASTIVE} \\
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
    sbatch finetune_run.sh
fi

((RUN_ID++))

done
done
done
done
done

echo "Submitted ${RUN_ID} runs under ${OUTPUT_BASE}"
