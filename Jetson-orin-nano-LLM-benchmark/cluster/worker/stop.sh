#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PORT="${PORT:-8000}"
PID_FILE="$PROJECT_ROOT/.run/cluster/worker_server_${PORT}.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[INFO] no worker API PID file for port $PORT"
  exit 0
fi

worker_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -n "$worker_pid" ]] && kill -0 "$worker_pid" 2>/dev/null; then
  kill "$worker_pid"
  for _ in $(seq 1 40); do
    if ! kill -0 "$worker_pid" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done
  if kill -0 "$worker_pid" 2>/dev/null; then
    echo "[WARN] worker API did not stop after SIGTERM (PID=$worker_pid)" >&2
    exit 1
  fi
  echo "[OK] worker API stopped (PID=$worker_pid)"
else
  echo "[INFO] stale worker API PID file removed"
fi
rm -f "$PID_FILE"
