#!/bin/bash
# gpu_monitor.sh — Monitor GPU utilisation for a running SLURM job.
#
# Usage:
#   ./gpu_monitor.sh <JOBID> [interval_seconds]
#
# Arguments:
#   JOBID    — SLURM job ID to monitor
#   interval — refresh interval in seconds (default: 30)
#
# What it checks:
#   • Which nodes are allocated to the job
#   • Per-GPU: utilisation %, VRAM used / total, temperature
#   • Per-GPU: which compute processes are running (PID + user + VRAM)
#   • Whether the number of GPUs with active processes matches
#     the number requested by the job (--gres=gpu:N)
#
# Requirements:
#   • ssh passwordless access to the compute nodes (standard on HPC clusters)
#   • nvidia-smi available on the compute nodes

set -euo pipefail

JOBID=${1:?"Usage: $0 <JOBID> [interval_seconds]"}
INTERVAL=${2:-30}

# ─── colours (disabled when not a tty) ──────────────────────────────────────
if [ -t 1 ]; then
    RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
else
    RED=''; YELLOW=''; GREEN=''; CYAN=''; BOLD=''; RESET=''
fi

# ─── helpers ────────────────────────────────────────────────────────────────

die() { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }

# Expand GPU index strings that may contain ranges: "0-2,5" → "0,1,2,5"
expand_gpu_indices() {
    local s="$1" result="" part
    IFS=',' read -ra parts <<< "$s"
    for part in "${parts[@]}"; do
        part=$(echo "$part" | xargs)
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            for (( i=${BASH_REMATCH[1]}; i<=${BASH_REMATCH[2]}; i++ )); do
                [[ -n "$result" ]] && result+=","
                result+="$i"
            done
        elif [[ -n "$part" ]]; then
            [[ -n "$result" ]] && result+=","
            result+="$part"
        fi
    done
    echo "$result"
}

# Run nvidia-smi queries on a remote node in one SSH round-trip.
# Prints a structured report to stdout.
query_node() {
    local node=$1
    local job_user=$2

    ssh -o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=no \
        "$node" bash <<'REMOTE'
set -euo pipefail

# ── GPU summary (one line per GPU) ──────────────────────────────────────────
echo "===GPU_SUMMARY==="
nvidia-smi \
    --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null || echo "NO_GPUS"

# ── Compute processes (one line per process) ─────────────────────────────────
echo "===COMPUTE_PROCS==="
# Primary: nvidia-smi compute apps (includes VRAM per process)
_nsmi=$(nvidia-smi --query-compute-apps=gpu_index,pid,used_gpu_memory,process_name \
    --format=csv,noheader,nounits 2>/dev/null || true)
if [[ -n "$_nsmi" ]]; then
    echo "$_nsmi"
else
    # Fallback: scan /proc for processes with open /dev/nvidiaX file descriptors.
    # Works even when nvidia-smi restricts cross-user process visibility.
    for pid_dir in /proc/[0-9]*/fd; do
        pid=$(basename "$(dirname "$pid_dir")")
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        gpu_indices=$(ls -la "$pid_dir" 2>/dev/null \
            | grep -oP '/dev/nvidia\K[0-9]+\b' \
            | sort -un 2>/dev/null) || continue
        [[ -z "$gpu_indices" ]] && continue
        cmd=$(ps -o comm= -p "$pid" 2>/dev/null | head -1 || echo "?")
        while IFS= read -r gidx; do
            echo "$gidx, $pid, 0, $cmd"
        done <<< "$gpu_indices"
    done
fi

# ── Map PID → username ───────────────────────────────────────────────────────
echo "===PID_USERS==="
# Collect PIDs from nvidia-smi if available, else from /proc scan above
{
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true
    # Also scan /proc in case nvidia-smi returned nothing
    for pid_dir in /proc/[0-9]*/fd; do
        pid=$(basename "$(dirname "$pid_dir")")
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        ls -la "$pid_dir" 2>/dev/null | grep -q '/dev/nvidia[0-9]' || continue
        echo "$pid"
    done
} | sort -u \
  | while read -r pid; do
        user=$(ps -o user= -p "$pid" 2>/dev/null | head -1 || echo "?")
        echo "$pid $user"
    done
REMOTE
}

# ─── parse a node's report ───────────────────────────────────────────────────
print_node_report() {
    local node=$1
    local job_user=$2
    local gpu_requested=$3
    local raw_output=$4
    local allocated_gpus=$5   # comma-separated physical indices from scontrol, e.g. "0,1,2"

    # Split sections
    local gpu_section proc_section pid_section
    gpu_section=$(echo "$raw_output" | awk '/===GPU_SUMMARY===/,/===COMPUTE_PROCS===/' \
        | grep -v '===')
    proc_section=$(echo "$raw_output" | awk '/===COMPUTE_PROCS===/,/===PID_USERS===/' \
        | grep -v '===')
    pid_section=$(echo "$raw_output"  | awk '/===PID_USERS===/,0' \
        | grep -v '===')

    if echo "$gpu_section" | grep -q "NO_GPUS"; then
        echo -e "  ${YELLOW}[!] No GPUs or nvidia-smi unavailable on ${node}${RESET}"
        return
    fi

    # Build set of SLURM-allocated GPU indices for this job
    declare -A alloc_set=()
    if [[ -n "$allocated_gpus" ]]; then
        IFS=',' read -ra _alloc_arr <<< "$allocated_gpus"
        for _g in "${_alloc_arr[@]}"; do
            _g=$(echo "$_g" | xargs)
            [[ -n "$_g" ]] && alloc_set["$_g"]=1
        done
    fi
    local have_alloc_info=false
    [[ ${#alloc_set[@]} -gt 0 ]] && have_alloc_info=true

    # Build PID→user map
    declare -A pid_user=()
    while read -r pid user; do
        [[ -n "$pid" ]] && pid_user["$pid"]="$user"
    done <<< "$pid_section"

    # Build GPU index → list of "pid:user:vram_MiB" entries
    declare -A gpu_procs=()
    while IFS=',' read -r gpu_idx pid vram pname; do
        gpu_idx=$(echo "$gpu_idx" | xargs)
        pid=$(echo "$pid"      | xargs)
        vram=$(echo "$vram"    | xargs)
        [[ -z "$gpu_idx" || -z "$pid" ]] && continue
        local user="${pid_user[$pid]:-?}"
        gpu_procs["$gpu_idx"]+="${pid}:${user}:${vram}MiB "
    done <<< "$proc_section"

    # ── Header row ─────────────────────────────────────────────────────────
    printf "  ${BOLD}%-2s %-4s  %-22s  %6s  %10s  %10s  %5s  %s${RESET}\n" \
        "" "GPU" "Name" "Util%" "MemUsed" "MemTotal" "Temp" "Processes (pid:user:vram)"
    printf "  %-2s %-4s  %-22s  %6s  %10s  %10s  %5s  %s\n" \
        "--" "----" "----------------------" "------" "----------" "----------" "-----" \
        "------------------------------------"

    local gpus_with_job_proc=0
    local gpu_count=0

    while IFS=',' read -r idx name util mem_used mem_total temp; do
        idx=$(echo "$idx"       | xargs)
        name=$(echo "$name"     | xargs)
        util=$(echo "$util"     | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        temp=$(echo "$temp"     | xargs)
        [[ -z "$idx" ]] && continue
        (( ++gpu_count ))

        local procs="${gpu_procs[$idx]:-}"
        local has_job=false
        local proc_str="—"

        if [[ -n "$procs" ]]; then
            proc_str="$procs"
            # Check if any process belongs to the job user
            if echo "$procs" | grep -qw "$job_user"; then
                has_job=true
                (( ++gpus_with_job_proc ))
            fi
        fi

        # Is this GPU SLURM-allocated to our job?
        local is_alloc=false
        [[ ${alloc_set[$idx]+x} ]] && is_alloc=true

        # Colour the utilisation figure
        local util_col="$RESET"
        if [[ "$util" =~ ^[0-9]+$ ]]; then
            (( util >= 70 )) && util_col="$GREEN"
            (( util >= 1  && util < 70 )) && util_col="$YELLOW"
            (( util == 0 )) && util_col="$RED"
        fi

        # Two-char status indicator:
        #   char 1 — SLURM allocation: ★ (yours) or · (not yours / unknown)
        #   char 2 — active process:   ✓ (running) or · (none yet) or space
        local alloc_col proc_col alloc_char proc_char
        if $is_alloc; then
            alloc_col="$GREEN"; alloc_char="★"
        elif $have_alloc_info; then
            alloc_col="$RESET"; alloc_char="·"
        else
            alloc_col="$RESET"; alloc_char="?"
        fi
        if $has_job; then
            proc_col="$GREEN"; proc_char="✓"
        elif $is_alloc; then
            proc_col="$YELLOW"; proc_char="·"   # allocated but no process yet
        else
            proc_col="$RESET";  proc_char=" "
        fi

        local mem_pct=""
        if [[ "$mem_total" =~ ^[0-9]+$ && "$mem_total" -gt 0 ]]; then
            mem_pct=$(( mem_used * 100 / mem_total ))
            mem_pct="(${mem_pct}%)"
        fi

        printf "  ${alloc_col}%s${RESET}${proc_col}%s${RESET} %-4s  %-22s  ${util_col}%5s%%${RESET}  %6sMiB  %6sMiB%-6s  %4s°C  %s\n" \
            "$alloc_char" "$proc_char" \
            "$idx" "${name:0:22}" \
            "$util" "$mem_used" "$mem_total" "$mem_pct" \
            "$temp" "${proc_str:0:60}"
    done <<< "$gpu_section"

    # ── Legend + summary ───────────────────────────────────────────────────
    echo ""
    if $have_alloc_info; then
        echo -e "  Legend: ${GREEN}★${RESET} = SLURM-allocated to job  ${GREEN}✓${RESET} = your process running  ${YELLOW}·${RESET} = allocated, no process yet  · = not allocated"
    else
        echo -e "  Legend: ${GREEN}✓${RESET} = your process running  (SLURM GPU indices unavailable — scontrol returned no IDX)"
    fi
    echo ""
    if [[ "$gpu_requested" =~ ^[0-9]+$ ]]; then
        if (( gpus_with_job_proc >= gpu_requested )); then
            echo -e "  ${GREEN}[OK]${RESET} ${BOLD}${gpus_with_job_proc} / ${gpu_requested}${RESET} requested GPUs have active processes from user '${job_user}'"
        elif (( gpus_with_job_proc > 0 )); then
            echo -e "  ${YELLOW}[WARN]${RESET} Only ${BOLD}${gpus_with_job_proc} / ${gpu_requested}${RESET} requested GPUs active for user '${job_user}'"
        else
            echo -e "  ${RED}[WARN]${RESET} ${BOLD}0 / ${gpu_requested}${RESET} GPUs have processes from user '${job_user}' — job may not have started yet or is using CPU only"
        fi
    else
        echo -e "  ${CYAN}INFO${RESET}: ${gpus_with_job_proc} GPU(s) have processes from user '${job_user}' (requested count unknown)"
    fi
}

# ─── main monitoring loop ────────────────────────────────────────────────────
iteration=0
while true; do
    # ── Fetch job metadata ────────────────────────────────────────────────
    job_raw=$(squeue -j "$JOBID" \
        --format="%N %b %u %T %P %l %D" \
        --noheader 2>/dev/null || true)

    # Clear screen after the first iteration so first output is always visible
    (( iteration > 0 )) && clear

    if [[ -z "$job_raw" ]]; then
        echo -e "${YELLOW}Job ${JOBID} is not in the queue.${RESET}"
        echo ""
        echo "Last known info from accounting:"
        sacct -j "$JOBID" \
            --format=JobID,JobName,State,NodeList,ReqGRES,Elapsed,Start,End \
            --noheader 2>/dev/null | head -5 || echo "  (sacct not available)"
        break
    fi

    read -r nodelist gres job_user state partition timelimit nnodes <<< "$job_raw"

    # Parse requested GPU count from gres string (e.g. "gpu:4", "gpu:L40S:2")
    gpu_requested=$(echo "$gres" | grep -oP '(?<=gpu:)(\w+:)?\K\d+' | tail -1)
    gpu_requested=${gpu_requested:-"?"}

    # Expand hostlist (handles "node[01-03]" notation)
    nodes=$(scontrol show hostnames "$nodelist" 2>/dev/null || echo "$nodelist")

    # Parse SLURM-allocated GPU indices from scontrol
    # Handles both comma-separated "IDX:0,1,2" and range "IDX:0-2" formats
    allocated_gpus=$(scontrol show job "$JOBID" 2>/dev/null \
        | grep -oP 'IDX:\K[0-9,\-]+' | head -1 || true)
    allocated_gpus=${allocated_gpus:-""}
    [[ -n "$allocated_gpus" ]] && allocated_gpus=$(expand_gpu_indices "$allocated_gpus")

    # ── Print header ──────────────────────────────────────────────────────
    echo -e "${BOLD}${CYAN}══ SLURM GPU Monitor ══════════════════════════════════════════════${RESET}"
    printf "  ${BOLD}Job:${RESET} %-10s  ${BOLD}User:${RESET} %-12s  ${BOLD}State:${RESET} %s\n" \
        "$JOBID" "$job_user" "$state"
    printf "  ${BOLD}Nodes:${RESET} %-18s  ${BOLD}Partition:${RESET} %-10s  ${BOLD}GPUs requested:${RESET} %s\n" \
        "$nodelist" "$partition" "$gpu_requested"
    printf "  ${BOLD}Time limit:${RESET} %-10s  ${BOLD}Num nodes:${RESET} %s\n" \
        "$timelimit" "$nnodes"
    printf "  ${BOLD}Timestamp:${RESET} %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════════${RESET}"

    # ── Query each node ───────────────────────────────────────────────────
    for node in $nodes; do
        echo ""
        echo -e "  ${BOLD}Node: ${CYAN}${node}${RESET}"
        echo "  ──────────────────────────────────────────────────────────────────"

        raw=$(query_node "$node" "$job_user" 2>/dev/null) || {
            echo -e "  ${RED}[!] SSH to ${node} failed or timed out${RESET}"
            continue
        }
        print_node_report "$node" "$job_user" "$gpu_requested" "$raw" "$allocated_gpus"
    done

    echo ""
    (( ++iteration ))
    echo -e "  ${CYAN}[iteration ${iteration}]${RESET} Next refresh in ${INTERVAL}s — Ctrl+C to stop"
    sleep "$INTERVAL"
done
