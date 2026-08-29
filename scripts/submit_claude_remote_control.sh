#!/bin/bash
# Hold a cpu-all node and run `claude --remote-control` inside it, so the
# session stays reachable from Remote Control (e.g. the mobile app) for as
# long as the Slurm allocation lasts. Mirrors the existing "vscode-node" job
# on node1 (fm4tag-node1), which holds a node the same way.
#
# claude's interactive UI needs a real PTY, which a plain batch script does
# not provide, so it runs inside a detached tmux session; `sleep infinity`
# then just holds the Slurm allocation open for the tmux session to live in.
#
# Usage:
#   scripts/submit_claude_remote_control.sh [time_limit] [cpus] [mem]
#
# Examples:
#   scripts/submit_claude_remote_control.sh                # defaults: 3 days, 4 cpus, 8G
#   scripts/submit_claude_remote_control.sh 7-00:00:00 8 16G

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TIME_LIMIT="${1:-3-00:00:00}"
CPUS="${2:-4}"
MEM="${3:-8G}"

LOG_DIR=$REPO/scripts/logs/claude_remote
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Requesting cpu-all node: time=$TIME_LIMIT cpus=$CPUS mem=$MEM"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=remote
#SBATCH --partition=cpu-all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME_LIMIT
#SBATCH --output=$LOG_DIR/remote_${TIMESTAMP}.log

cd $REPO
# Use a job-unique tmux session name: the per-user tmux server on a node can
# outlive the job that spawned it, so a fixed name collides with stale
# sessions from earlier jobs and \`tmux new-session\` silently no-ops.
SESSION=claude_remote_\${SLURM_JOB_ID}
tmux kill-session -t "\$SESSION" 2>/dev/null || true
tmux new-session -d -s "\$SESSION" \
  "claude --remote-control --remote-control-session-name-prefix fm4tag"
echo "tmux session '\$SESSION' started on \$(hostname) at \$(date)"
sleep infinity
EOF
