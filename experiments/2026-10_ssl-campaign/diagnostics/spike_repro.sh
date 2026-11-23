#!/bin/bash
#SBATCH --job-name=spike_repro
#SBATCH --partition=gpu-L40S-open,gpu-A40
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/storage3/DSIP/rriva/research/fm4tag/experiments/2026-10_ssl-campaign/diagnostics/outputs/repro/job_%j.out
#SBATCH --error=/storage3/DSIP/rriva/research/fm4tag/experiments/2026-10_ssl-campaign/diagnostics/outputs/repro/job_%j.err
# Resume two stopped full pretrainings from their last healthy checkpoint, a
# little before a recorded gradient blowup, with SpikeDump on: per-parameter
# gradient norms every step, batch + state dumped at the first spikes.
#   w0: P0 seed 42 from epoch 4 (step 16145); full collapse at step ~17399
#   w1: P1 seed 42 from epoch 7 (step 25832); blowup (max norm 25756) at ~27699
REPO=/storage3/DSIP/rriva/research/fm4tag
OUT=${REPO}/experiments/2026-10_ssl-campaign/diagnostics/outputs/repro
DATA=/storage3/DSIP/rriva/datasets/transforming-jet-flavor/data_parallel_preprocess_251112_152051
source "${REPO}/.venv/bin/activate"
nvidia-smi
cat "${DATA}/pretrain/output/pp_output_pretrain.h5" "${DATA}/val/output/pp_output_val.h5" > /dev/null

COMMON=(--config-path="${REPO}/src/fm4tag/configs" --config-name=campaign_ssl
        action=fit trainer.max_epochs=30
        trainer.log_every_n_steps=10 datamodule.num_workers=6
        arm=pretrain encoders=rowcol_concat_v0 chunk_size=512 datamodule.batch_size=12288
        optimizer.lr=1e-3 seed=42 '~loggers.wandb'
        +callbacks.spike_dump._target_=spike_dump.SpikeDump)

run() {  # name ckpt stop_step extra-overrides...
    local name=$1 ckpt=$2 stop=$3; shift 3
    mkdir -p "${OUT}/${name}"
    cd "${OUT}/${name}" || return 1
    PYTHONPATH=${REPO}/slurm/smoke fm4tag "${COMMON[@]}" "$@" \
        experiment_name="repro_${name}" output_dir="${OUT}/${name}" ckpt_path="${ckpt}" \
        +callbacks.spike_dump.out_dir="${OUT}/${name}/dump" +callbacks.spike_dump.stop_step="${stop}" \
        > "${OUT}/${name}/rank_0.out" 2> "${OUT}/${name}/rank_0.err"
    echo "${name}: rc $?"
}
P0=${REPO}/experiments/2026-10_ssl-campaign/outputs/P_full_20261021_112456/p0/seed_42/p0/version_0/checkpoints/epoch004-step16145-val_loss6.2696.ckpt
P1=${REPO}/experiments/2026-10_ssl-campaign/outputs/P_full_20261021_161637/p1/seed_42/p1/version_0/checkpoints/epoch007-step25832-val_loss6.0010.ckpt
export -f run; export OUT REPO
srun --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=24G \
    bash -c "$(declare -p COMMON); run p0_s42 ${P0} 17900" &
srun --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=24G \
    bash -c "$(declare -p COMMON); run p1_s42 ${P1} 28200 pretrain.loss_weights.jet_contrastive=1.0 pretrain.jet_contrastive_loss.temperature=0.1" &
wait
