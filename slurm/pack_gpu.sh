#!/bin/bash
# One GPU of a pack job that runs several trainings at a time (SLOTS_PER_GPU > 1 in
# slurm/ssl_pack.sh; how it works: docs/gpu_packing.md).
#
#   pack_gpu.sh <stage dir> <gpu name, e.g. w0> <slots>     (inside a 1-GPU srun step)
#
# Starts <slots> workers (slurm/pack_worker.sh) that all see the card slurm gave this
# step, and gives them one ledger directory on the node: each worker writes there the
# estimated GPU memory of the training it runs and starts a new one only if the sum
# stays within GPU_BUDGET_GIB (pack.env).  The ledger is per job and per card, so two
# cards of one job never count each other's trainings.
set -uo pipefail

STAGE_DIR=$1
GPU=$2
SLOTS=$3
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# node-local on purpose: every worker of the card is on this node, and flock is only
# reliable on a local file system
export PACK_LEDGER=${PACK_LEDGER_ROOT:-/tmp}/fm4tag_pack_${SLURM_JOB_ID:-$$}_${GPU}
rm -rf "${PACK_LEDGER}"
mkdir -p "${PACK_LEDGER}"
trap 'rm -rf "${PACK_LEDGER}"' EXIT

for (( s = 0; s < SLOTS; s++ )); do
    bash "${HERE}/pack_worker.sh" "${STAGE_DIR}" "${GPU}s${s}" &
    sleep "${PACK_STAGGER_SECONDS:-20}"   # the first training allocates before the next starts
done
wait
