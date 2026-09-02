#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/.venv" ]]; then
  PROJECT_ROOT="$SCRIPT_DIR"
else
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"

PORT="${PORT:-8000}"
RUN_DIR="$PROJECT_ROOT/.run"
PID_FILE="$RUN_DIR/chat_server.pid"

stopped=0
stopped_pid=""

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" || true
    echo "[OK] Sent stop signal to PID=$pid"
    stopped=1
    stopped_pid="$pid"
  fi
  rm -f "$PID_FILE"
fi

fallback_pids="$(pgrep -f "uvicorn web.app:app --host .* --port $PORT" || true)"
if [[ -n "$fallback_pids" ]]; then
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    [[ -n "$stopped_pid" && "$pid" == "$stopped_pid" ]] && continue
    kill "$pid" || true
    echo "[OK] Sent stop signal to PID=$pid (fallback)"
    stopped=1
  done <<< "$fallback_pids"
fi

if [[ "$stopped" -eq 0 ]]; then
  echo "[INFO] No running server found on port $PORT"
else
  echo "[OK] Server stop requested"
fi
