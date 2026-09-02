#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/.venv" && -d "$SCRIPT_DIR/web" ]]; then
  PROJECT_ROOT="$SCRIPT_DIR"
else
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RUN_DIR="$PROJECT_ROOT/.run"
PID_FILE="$RUN_DIR/chat_server.pid"
LOG_FILE="$RUN_DIR/chat_server.log"

mkdir -p "$RUN_DIR"

if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
  echo "[ERROR] .venv not found in $PROJECT_ROOT"
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "[INFO] Server already running (PID=$existing_pid)"
    echo "[INFO] URL: http://$HOST:$PORT"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

source "$PROJECT_ROOT/.venv/bin/activate"

nohup uvicorn web.app:app --host "$HOST" --port "$PORT" >"$LOG_FILE" 2>&1 &
server_pid="$!"
echo "$server_pid" >"$PID_FILE"

if kill -0 "$server_pid" 2>/dev/null; then
  if python - "$HOST" "$PORT" <<'PY'
import json
import sys
import time
import urllib.request

host = sys.argv[1]
port = int(sys.argv[2])
if host == "0.0.0.0":
    host = "127.0.0.1"

url = f"http://{host}:{port}/health"
for _ in range(30):
    try:
        with urllib.request.urlopen(url, timeout=1.0) as r:
            body = json.loads(r.read().decode("utf-8"))
            if body.get("ok") is True:
                raise SystemExit(0)
    except Exception:
        time.sleep(0.2)

raise SystemExit(1)
PY
  then
    echo "[OK] Server started (PID=$server_pid)"
    echo "[OK] URL: http://$HOST:$PORT"
    echo "[OK] Log: $LOG_FILE"
  else
    echo "[ERROR] Server process started but health check failed."
    tail -n 40 "$LOG_FILE" || true
    rm -f "$PID_FILE"
    exit 1
  fi
else
  echo "[ERROR] Failed to start server. Check log: $LOG_FILE"
  tail -n 40 "$LOG_FILE" || true
  rm -f "$PID_FILE"
  exit 1
fi
