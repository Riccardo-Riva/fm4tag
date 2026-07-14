#!/bin/bash
#SBATCH --partition=cpu-all
#SBATCH --mem=4G

# ── Settings ──────────────────────────────────────────────────────────────────

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
OUTPUT_BASE=${REPO}/slurm/pretraining/run_${TIMESTAMP}

mkdir -pv "${OUTPUT_BASE}"

# ── Grid definitions ──────────────────────────────────────────────────────────

TRANSFORMER_TYPES=("rowcol" "col")
BATCH_SIZES=(1024)

LOSS_CONFIGS=(
"1.0 1.0"
"0.0 1.0"
"1.0 0.0"
)

# Fixed denoising weights

DENOISING_CAT=0.4
DENOISING_CON=0.4

RUN_ID=0

# ── Submit grid ───────────────────────────────────────────────────────────────

for TRANSFORMER_TYPE in "${TRANSFORMER_TYPES[@]}"; do
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
for LOSS_CONF in "${LOSS_CONFIGS[@]}"; do


read CONTRASTIVE JET_CONTRASTIVE <<< "${LOSS_CONF}"

RUN_NAME="pretrain_${TIMESTAMP}_${TRANSFORMER_TYPE}_bs_${BATCH_SIZE}_c_${CONTRASTIVE}_jc_${JET_CONTRASTIVE}"
OUTPUT_DIR=${OUTPUT_BASE}/${RUN_NAME}_${RUN_ID}

mkdir -pv "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/pretrain_run.sh" << EOF
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

export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

srun \
--output=${OUTPUT_DIR}/rank_%t.out \
--error=${OUTPUT_DIR}/rank_%t.err \
fm4tag \
--config-path=${CONFIG_DIR} \
--config-name=${CONFIG} \
phase=pretrain action=fit \
trainer.devices=${GPU_NUM} \
trainer.max_epochs=${MAX_EPOCHS} \
datamodule.num_workers=${NUM_WORKERS} \
datamodule.batch_size=${BATCH_SIZE} \
encoders=${TRANSFORMER_TYPE}_v0 \
pretrain.loss_weights.contrastive=${CONTRASTIVE} \
pretrain.loss_weights.denoising_cat=${DENOISING_CAT} \
pretrain.loss_weights.denoising_con=${DENOISING_CON} \
pretrain.loss_weights.jet_contrastive=${JET_CONTRASTIVE} \
experiment_name=${RUN_NAME} \
output_dir=${OUTPUT_DIR}

echo "Elapsed: \$((SECONDS/3600))h \$(((SECONDS/60)%60))m \$((SECONDS%60))s"
EOF

echo "Submitting run ${RUN_ID}: ${RUN_NAME}"
cd "${OUTPUT_DIR}"
sbatch pretrain_run.sh

((RUN_ID++))


done
done
done
