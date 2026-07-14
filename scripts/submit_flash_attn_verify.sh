#!/bin/bash
# One-off verification job for the FlashAttention-2 fast path added to
# fm4tag.models.attention.sdpa (see tests/test_sdpa_flash.py). Installs
# flash-attn into the repo's .venv, runs the GPU-gated unit tests, then
# re-runs the per-module timing profile so it can be compared against the
# pre-flash-attn baseline in slurm/profiling/run_20260713_161724/.
#
# Usage: scripts/submit_flash_attn_verify.sh
#
# Results: slurm/profiling/flash_verify_<TS>/{out.txt, err.txt, profile_*.json}

REPO=/storage3/DSIP/rriva/research/fm4tag
TS=$(date +%Y%m%d_%H%M%S)
OUT=${REPO}/slurm/profiling/flash_verify_${TS}

mkdir -pv "${OUT}"

cat > "${OUT}/verify_run.sh" << EOF
#!/bin/bash
#SBATCH --job-name=flash_verify_${TS}
#SBATCH --partition=gpu-L40S-open,gpu-A40
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=${OUT}/out.txt
#SBATCH --error=${OUT}/err.txt

source ${REPO}/.venv/bin/activate
nvidia-smi -L
python -c "import torch; print('torch', torch.__version__, 'cap', torch.cuda.get_device_capability(0))"

echo
echo "=============== installing flash-attn (via uv, venv has no pip) ==============="
export MAX_JOBS=4
uv pip install flash-attn --no-build-isolation 2>&1 | tail -80
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" || echo "flash-attn import FAILED — GPU tests below will skip, profiling will still run on the SDPA fallback"

echo
echo "=============== tests/test_sdpa_flash.py ==============="
python -m pytest ${REPO}/tests/test_sdpa_flash.py -v

echo
echo "=============== PRETRAIN profile (compare vs slurm/profiling/run_20260713_161724/) ==============="
python ${REPO}/scripts/profile_timing.py --phase pretrain \\
    --steps 30 --warmup 10 --data-batches 50 \\
    --out ${OUT} datamodule.num_workers=8

echo
echo "=============== FINETUNE profile ==============="
python ${REPO}/scripts/profile_timing.py --phase finetune \\
    --steps 30 --warmup 10 --data-batches 50 \\
    --out ${OUT} datamodule.num_workers=8
EOF

cd "${OUT}" && sbatch verify_run.sh
