#!/usr/bin/env bash
# Reproduce the global-only (no-constituent) smoke test:
#   10-epoch pretrain  ->  checkpoint  ->  10-epoch finetune
# on configs/jets_only.yaml.
#
# Pins to a single GPU.  Resolution order for which GPU to use:
#   1. first CLI arg            ->  scripts/run_jets_only_test.sh 2
#   2. $CUDA_VISIBLE_DEVICES    ->  CUDA_VISIBLE_DEVICES=2 scripts/run_jets_only_test.sh
#   3. auto-detect from SLURM   ->  scontrol GRES IDX of $SLURM_JOB_ID
#
# Wipes outputs/jets_only_test/ first so pretrain is version_0 and finetune is
# version_1 (the python runner picks up version_0/last.ckpt as encoder_ckpt).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-${CUDA_VISIBLE_DEVICES:-}}"
if [ -z "$GPU" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  GPU="$(scontrol show job -d "$SLURM_JOB_ID" 2>/dev/null \
         | grep -oP 'IDX:\K[0-9]+' | head -n1 || true)"
fi
if [ -z "$GPU" ]; then
  echo "ERROR: could not determine the GPU to use." >&2
  echo "       Pass it explicitly, e.g.: scripts/run_jets_only_test.sh 2" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

rm -rf outputs/jets_only_test

PY="${PYTHON:-.venv/bin/python}"
exec "$PY" scripts/run_jets_only_test.py
