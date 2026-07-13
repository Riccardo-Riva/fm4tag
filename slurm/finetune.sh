#!/bin/bash
#SBATCH --partition=cpu-all
#SBATCH --mem=4G

# ── Settings ──────────────────────────────────────────────────────────────────
# Fine-tuning from pretrained encoders: phase=finetune with encoder_ckpt set.
# finetune.unfreeze_at_epoch (default 3) holds the pretrained parts at lr=0 for
# a short linear-probe phase, then ramps them onto the cosine schedule at
# optimizer.backbone_lr — DDP-safe (all params in the optimiser from step 0).

GPU_NODE=gpu-L40S-open,gpu-A40
GPU_NUM=1
NUM_WORKERS=16

REPO=/storage3/DSIP/rriva/research/fm4tag-review
VENV=${REPO}/.venv
CONFIG_DIR=${REPO}/src/fm4tag/configs

CONFIG=default.yaml
MAX_EPOCHS=100

# Global timestamp for this sweep (seconds precision)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE=${REPO}/slurm/finetuning/run_${TIMESTAMP}

mkdir -pv "${OUTPUT_BASE}"

# ── Grid definitions ──────────────────────────────────────────────────────────

# "transformer_type checkpoint" pairs: the architecture must match the one the
# checkpoint was pretrained with.  Checkpoints from pretrain.sh sweeps land in
#   ${REPO}/slurm/pretraining/run_<TS>/<RUN_NAME>_<ID>/<RUN_NAME>/version_0/checkpoints/epochXXX-stepYYY.ckpt
ENCODERS=(
"rowcol ${REPO}/slurm/pretraining/run_20260710_152049/pretrain_20260710_152049_rowcol_bs_1024_c_1.0_jc_1.0_0/pretrain_20260710_152049_rowcol_bs_1024_c_1.0_jc_1.0/version_0/checkpoints/epoch018-step368125.ckpt"
#"col    ${REPO}/slurm/pretraining/run_XXXXXXXX_XXXXXX/CHANGE_ME.ckpt"
)

BATCH_SIZES=(2048)

# "cross_entropy jet_contrastive": CE on the head output, SupCon on the
# aggregator output.  Weights are normalised to unit sum; 0 disables the term.
LOSS_CONFIGS=(
"1.0 0.0"
"1.0 0.3"
"1.0 1.0"
)

RUN_ID=0

# ── Submit grid ───────────────────────────────────────────────────────────────

for ENCODER in "${ENCODERS[@]}"; do
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
for LOSS_CONF in "${LOSS_CONFIGS[@]}"; do

read TRANSFORMER_TYPE ENCODER_CKPT <<< "${ENCODER}"
read CROSS_ENTROPY JET_CONTRASTIVE <<< "${LOSS_CONF}"

if [[ ! -f "${ENCODER_CKPT}" ]]; then
    echo "Skipping: checkpoint not found: ${ENCODER_CKPT}"
    continue
fi

RUN_NAME="finetune_${TIMESTAMP}_${TRANSFORMER_TYPE}_bs_${BATCH_SIZE}_ce_${CROSS_ENTROPY}_jc_${JET_CONTRASTIVE}"
OUTPUT_DIR=${OUTPUT_BASE}/${RUN_NAME}_${RUN_ID}

mkdir -pv "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/finetune_run.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --partition=${GPU_NODE}
#SBATCH --gres=gpu:${GPU_NUM}
#SBATCH --ntasks-per-node=${GPU_NUM}
#SBATCH --cpus-per-task=$((NUM_WORKERS + 2))
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
    finetune.loss_weights.cross_entropy=${CROSS_ENTROPY} \\
    finetune.loss_weights.jet_contrastive=${JET_CONTRASTIVE} \\
    experiment_name=${RUN_NAME} \\
    output_dir=${OUTPUT_DIR}

echo "Elapsed: \$((SECONDS/3600))h \$(((SECONDS/60)%60))m \$((SECONDS%60))s"
EOF

echo "Submitting run ${RUN_ID}: ${RUN_NAME}"
cd "${OUTPUT_DIR}"
sbatch finetune_run.sh

((RUN_ID++))

done
done
done

echo "Submitted ${RUN_ID} runs under ${OUTPUT_BASE}"
