#!/bin/bash
#SBATCH --partition=cpu-all
#SBATCH --mem=4G

# ── Settings ──────────────────────────────────────────────────────────────────
# Supervised classification from scratch: phase=finetune with no pretrained
# encoder (encoder_ckpt=null), so the whole backbone + head is trained jointly.

GPU_NODE=gpu-L40S-open,gpu-A40
GPU_NUM=1
NUM_WORKERS=12

REPO=/storage3/DSIP/rriva/research/fm4tag-main
VENV=${REPO}/.venv
CONFIG_DIR=${REPO}/src/fm4tag/configs

CONFIG=default.yaml
MAX_EPOCHS=50

# Global timestamp for this sweep (seconds precision)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE=${REPO}/slurm/classify_from_scratch/run_${TIMESTAMP}

mkdir -pv "${OUTPUT_BASE}"

# ── Grid definitions ──────────────────────────────────────────────────────────

TRANSFORMER_TYPES=("rowcol")
BATCH_SIZES=(1024)

# "cross_entropy jet_contrastive": CE on the head output, SupCon on the
# aggregator output.  Weights are normalised to unit sum; 0 disables the term.
LOSS_CONFIGS=(
"1.0 0.0"
"1.0 0.3"
"1.0 1.0"
)

RUN_ID=0

# ── Submit grid ───────────────────────────────────────────────────────────────

for TRANSFORMER_TYPE in "${TRANSFORMER_TYPES[@]}"; do
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
for LOSS_CONF in "${LOSS_CONFIGS[@]}"; do

read CROSS_ENTROPY JET_CONTRASTIVE <<< "${LOSS_CONF}"

RUN_NAME="scratch_${TIMESTAMP}_${TRANSFORMER_TYPE}_bs_${BATCH_SIZE}_ce_${CROSS_ENTROPY}_jc_${JET_CONTRASTIVE}"
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
#SBATCH --gres=gpu:${GPU_NUM}
#SBATCH --ntasks-per-node=${GPU_NUM}
#SBATCH --cpus-per-task=$((NUM_WORKERS + 2))
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
    finetune.loss_weights.cross_entropy=${CROSS_ENTROPY} \\
    finetune.loss_weights.jet_contrastive=${JET_CONTRASTIVE} \\
    experiment_name=${RUN_NAME} \\
    output_dir=${OUTPUT_DIR}

echo "Elapsed: \$((SECONDS/3600))h \$(((SECONDS/60)%60))m \$((SECONDS%60))s"
EOF

echo "Submitting run ${RUN_ID}: ${RUN_NAME}"
cd "${OUTPUT_DIR}"
sbatch classify_from_scratch_run.sh

((RUN_ID++))

done
done
done

echo "Submitted ${RUN_ID} runs under ${OUTPUT_BASE}"
