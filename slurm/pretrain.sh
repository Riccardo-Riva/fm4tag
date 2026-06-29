#!/bin/bash
#SBATCH --partition=cpu-all
#SBATCH --mem=4G

# ── Settings ──────────────────────────────────────────────────────────────────
GPU_NODE=gpu-L40S-open,gpu-A40
GPU_NUM=2
NUM_WORKERS=8
REPO=/storage3/DSIP/rriva/research/fm4tag
VENV=${REPO}/.venv
CONFIG_DIR=${REPO}/src/fm4tag/configs
OUTPUT_BASE=${REPO}/slurm/pretraining

# Config and overrides
CONFIG=default          # src/fm4tag/configs/default.yaml
BATCH_SIZE=512          # per-GPU batch size (×GPU_NUM effective batch)
MAX_EPOCHS=100

# ── Timestamped output directory ──────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_BASE}/run_${TIMESTAMP}
mkdir -pv "${OUTPUT_DIR}"

# ── Write inner GPU job script ─────────────────────────────────────────────────
cat > "${OUTPUT_DIR}/run.sh" << EOF
#!/bin/bash
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
    phase=pretrain action=fit \\
    trainer.devices=${GPU_NUM} \\
    trainer.max_epochs=${MAX_EPOCHS} \\
    dataloader.num_workers=${NUM_WORKERS} \\
    dataloader.batch_size=${BATCH_SIZE} \\
    output_dir=${OUTPUT_DIR}

echo "Elapsed: \$((SECONDS/3600))h \$(((SECONDS/60)%60))m \$((SECONDS%60))s"
EOF

cd "${OUTPUT_DIR}"
sbatch run.sh
