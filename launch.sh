#!/usr/bin/env bash
# Usage: ./launch.sh [job_name]
# Submits slurm_scripts/<job_name>.sbatch, then starts the relay worker in a
# detached tmux session. The relay is stopped automatically once the job ends.
set -euo pipefail

JOB_NAME=${2:-${1:-current_job}}
POLL_SECONDS=${POLL_SECONDS:-30}   # how often to check the job state
GRACE_SECONDS=${GRACE_SECONDS:-60} # time the relay gets to drain after the job ends

# ---------------------------------------------------------------------------
# Inner mode: runs INSIDE tmux. Starts the relay, watches the Slurm job,
# then shuts the relay down when the job is gone.
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--inner" ]]; then
  JOB_NAME=$2
  ID=$3

  # setsid puts the relay in its own process group so we can stop
  # `time` and `python` together.
  setsid /usr/bin/time -v python scripts/file_api_worker.py \
      --queue-root api_queue \
      --job-name "$JOB_NAME" \
      --job-id "$ID" \
      --env-file .env >> "logs/relay_${ID}.log" 2>&1 &
  RELAY_PID=$!
  echo "[$(date '+%F %T')] relay started (pid ${RELAY_PID}) for job ${ID}"

  stop_relay() {
    if kill -0 "$RELAY_PID" 2>/dev/null; then
      echo "[$(date '+%F %T')] stopping relay"
      kill -TERM -- "-${RELAY_PID}" 2>/dev/null || true
      sleep 5
      kill -KILL -- "-${RELAY_PID}" 2>/dev/null || true
    fi
  }
  trap stop_relay EXIT INT TERM

  while true; do
    # Relay died on its own: nothing left to watch.
    if ! kill -0 "$RELAY_PID" 2>/dev/null; then
      echo "[$(date '+%F %T')] relay exited by itself"
      break
    fi

    # If squeue itself fails (controller busy), skip this round
    # rather than wrongly concluding the job is finished.
    if ! STATE=$(squeue -h -j "$ID" -o %T 2>/dev/null); then
      sleep "$POLL_SECONDS"
      continue
    fi

    # Empty output = job is no longer in the queue (finished, failed, cancelled).
    if [[ -z "$STATE" ]]; then
      FINAL=$(sacct -X -n -j "$ID" -o State%20 2>/dev/null | head -n1 | xargs || true)
      echo "[$(date '+%F %T')] job ${ID} finished (${FINAL:-unknown state}); grace ${GRACE_SECONDS}s"
      sleep "$GRACE_SECONDS"
      break
    fi

    sleep "$POLL_SECONDS"
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# Outer mode: runs on the login node.
# ---------------------------------------------------------------------------
mkdir -p logs

ID=$(sbatch --parsable "slurm_scripts/${JOB_NAME}.sbatch")
ID=${ID%%;*}   # strip ";cluster" on multi-cluster setups

SESSION="relay_${ID}"
tmux new-session -d -s "$SESSION" \
  "bash '$(readlink -f "$0")' --inner '${JOB_NAME}' '${ID}' 2>&1 | tee -a logs/watch_${ID}.log"

echo "Submitted job ${ID}"
echo "Relay + watcher running in tmux session: ${SESSION}  (on $(hostname))"
echo "Attach:      tmux attach -t ${SESSION}"
echo "Relay log:   tail -f logs/relay_${ID}.log"