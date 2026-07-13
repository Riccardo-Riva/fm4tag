#!/bin/bash
# Submit the per-module timing profile (scripts/profile_timing.py) as a 1-GPU
# slurm job. Profiles BOTH phases on real data with the current default config
# (plus any extra hydra overrides passed to this script), e.g.
#
#   scripts/submit_profile_timing.sh                    # current config (rowcol_v0)
#   scripts/submit_profile_timing.sh encoders=col_v0    # compare architectures
#
# Results: slurm/profiling/run_<TS>/{out.txt, profile_pretrain.json, profile_finetune.json}

REPO=/storage3/DSIP/rriva/research/fm4tag-review
TS=$(date +%Y%m%d_%H%M%S)
OUT=${REPO}/slurm/profiling/run_${TS}
OVERRIDES="$@"

mkdir -pv "${OUT}"

cat > "${OUT}/profile_run.sh" << EOF
#!/bin/bash
#SBATCH --job-name=profile_${TS}
#SBATCH --partition=gpu-L40S-open,gpu-A40
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --output=${OUT}/out.txt
#SBATCH --error=${OUT}/err.txt

source ${REPO}/.venv/bin/activate
nvidia-smi -L

echo "=============== PRETRAIN profile ==============="
python ${REPO}/scripts/profile_timing.py --phase pretrain \\
    --steps 30 --warmup 10 --data-batches 50 \\
    --out ${OUT} datamodule.num_workers=8 ${OVERRIDES}

echo
echo "=============== FINETUNE profile ==============="
python ${REPO}/scripts/profile_timing.py --phase finetune \\
    --steps 30 --warmup 10 --data-batches 50 \\
    --out ${OUT} datamodule.num_workers=8 ${OVERRIDES}
EOF

cd "${OUT}" && sbatch profile_run.sh
