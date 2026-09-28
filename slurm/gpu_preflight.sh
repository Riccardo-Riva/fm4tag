#!/bin/bash
# Wait until every GPU this job was allocated has enough free memory to hold a
# training rank.
#
# Why this is needed: gpu-L40S-open publishes each physical card TWICE in
# gres.conf — once as a whole GPU and once as 12 shards:
#
#     NodeName=node101 Name=gpu   Type=rtx File=/dev/nvidia[0-7]
#     Nodename=node101 Name=shard Count=12 File=/dev/nvidia6      # same card
#
# Slurm will hand a card out whole while a shard tenant is already on it, and
# shards carry no memory enforcement, so `--gres=gpu:2` can give us a card with
# under 100 MiB free.  The rank that lands there dies with CUDA OOM, its peer
# then hangs for the full 1800 s NCCL watchdog and aborts (rc 134), and the seed
# is burned for nothing.  run_20260915_110720 lost five seeds to this in two
# hours on 2026-09-15, all on GPU 6 of node101 under someone else's shard:6 job.
#
# Waiting costs minutes.  Starting anyway costs a seed.
#
#   exit 0 — the allocated GPUs are usable now
#   exit 1 — they were still occupied when the wait ran out
#
# Env: GPU_FREE_MIB_REQUIRED (20480), GPU_WAIT_SECONDS (1800),
#      GPU_POLL_SECONDS (120).

set -u

REQUIRED_MIB=${GPU_FREE_MIB_REQUIRED:-20480}
WAIT_SECONDS=${GPU_WAIT_SECONDS:-1800}
POLL_SECONDS=${GPU_POLL_SECONDS:-120}

# The indices Slurm gave us.  nvidia-smi is an NVML tool and reports every card
# on the node regardless of CUDA_VISIBLE_DEVICES, so its `index` column is the
# node-global one and these have to be matched against it by hand.  An empty
# list means "check every card nvidia-smi reports", which is the safe direction.
ALLOCATED=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}

# Prints one "index free_mib" line per allocated GPU that is short of memory.
short_gpus() {
    local idx used total free
    while IFS=', ' read -r idx used total; do
        [[ -n "${idx}" ]] || continue
        if [[ -n "${ALLOCATED}" ]] && [[ ",${ALLOCATED}," != *",${idx},"* ]]; then
            continue
        fi
        free=$(( total - used ))
        if (( free < REQUIRED_MIB )); then
            echo "${idx} ${free}"
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total \
                        --format=csv,noheader,nounits 2>/dev/null)
}

DEADLINE=$(( $(date +%s) + WAIT_SECONDS ))

while :; do
    if ! nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
        # Never block a training because the monitoring tool hiccuped — a node
        # where nvidia-smi does not answer has bigger problems than contention.
        echo "[gpu] WARNING: nvidia-smi did not answer; starting without the check" >&2
        exit 0
    fi

    SHORT=$(short_gpus)

    if [[ -z "${SHORT}" ]]; then
        echo "[gpu] allocated GPUs (${ALLOCATED:-all}) have >= $((REQUIRED_MIB / 1024)) GiB free"
        exit 0
    fi

    LEFT=$(( DEADLINE - $(date +%s) ))
    if (( LEFT <= 0 )); then
        echo "[gpu] still short after $((WAIT_SECONDS / 60)) min:" >&2
        echo "${SHORT}" | while read -r idx free; do
            echo "[gpu]   GPU ${idx}: only $((free)) MiB free" >&2
        done
        exit 1
    fi

    echo "[gpu] waiting $((POLL_SECONDS))s — another tenant is on:" \
         "$(echo "${SHORT}" | awk '{printf "GPU %s (%s MiB free) ", $1, $2}')" \
         "— $((LEFT / 60)) min of patience left"
    sleep "${POLL_SECONDS}"
done
