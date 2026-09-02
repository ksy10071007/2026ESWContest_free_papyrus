#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RUN_DIR="$PROJECT_ROOT/.run/cluster"
PID_FILE="$RUN_DIR/worker_server_${PORT}.pid"
LOG_FILE="$RUN_DIR/worker_server_${PORT}.log"
TOKEN_FILE="$RUN_DIR/worker.token"
SETTINGS_FILE="$RUN_DIR/settings.json"

mkdir -p "$RUN_DIR"
if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  echo "[ERROR] virtual environment missing: $PROJECT_ROOT/.venv" >&2
  exit 1
fi
if [[ -z "${CLUSTER_WORKER_AUTH:-}" ]]; then
  CLUSTER_WORKER_AUTH="false"
  if [[ -f "$SETTINGS_FILE" ]]; then
    CLUSTER_WORKER_AUTH="$($PROJECT_ROOT/.venv/bin/python - "$SETTINGS_FILE" <<'PY'
import json
import sys
try:
    value = bool(json.load(open(sys.argv[1], encoding="utf-8")).get("worker_api_auth", False))
except (OSError, ValueError):
    value = False
print("true" if value else "false")
PY
)"
  fi
fi
if [[ "$CLUSTER_WORKER_AUTH" == "true" && ! -f "$TOKEN_FILE" ]]; then
  "$PROJECT_ROOT/.venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))' >"$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
fi
CLUSTER_API_TOKEN=""
if [[ -f "$TOKEN_FILE" ]]; then
  CLUSTER_API_TOKEN="$(tr -d '\r\n' <"$TOKEN_FILE")"
fi
export CLUSTER_API_TOKEN
export CLUSTER_WORKER_AUTH

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    health_host="$HOST"
    [[ "$health_host" == "0.0.0.0" ]] && health_host="127.0.0.1"
    authenticated_health="$(curl -fsS -H "X-Cluster-Worker-Token: $CLUSTER_API_TOKEN" "http://$health_host:$PORT/cluster/health" 2>/dev/null || true)"
    unauthenticated_code="$(curl -sS -o /dev/null -w '%{http_code}' "http://$health_host:$PORT/cluster/health" 2>/dev/null || true)"
    if { [[ "$CLUSTER_WORKER_AUTH" == "true" && "$unauthenticated_code" == "401" ]] || [[ "$CLUSTER_WORKER_AUTH" != "true" && "$unauthenticated_code" == "200" ]]; } \
      && printf '%s' "$authenticated_health" | grep -q "\"worker_api_auth\":$CLUSTER_WORKER_AUTH"; then
      echo "[INFO] worker API already running and authenticated (PID=$existing_pid, port=$PORT)"
      exit 0
    fi
    echo "[INFO] replacing stale or unauthenticated worker API (PID=$existing_pid)"
    kill "$existing_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$existing_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$existing_pid" 2>/dev/null; then
      echo "[ERROR] stale worker API did not stop; refusing to start a duplicate process" >&2
      exit 1
    fi
  fi
  rm -f "$PID_FILE"
fi

cd "$PROJECT_ROOT"
PYTHONDONTWRITEBYTECODE=1 nohup "$PROJECT_ROOT/.venv/bin/python" -m uvicorn cluster.worker.app:app \
  --host "$HOST" --port "$PORT" >"$LOG_FILE" 2>&1 &
worker_pid="$!"
echo "$worker_pid" > "$PID_FILE"

health_host="$HOST"
[[ "$health_host" == "0.0.0.0" ]] && health_host="127.0.0.1"
for _ in $(seq 1 60); do
  if curl -fsS -H "X-Cluster-Worker-Token: $CLUSTER_API_TOKEN" "http://$health_host:$PORT/cluster/health" >/dev/null 2>&1; then
    echo "[OK] worker API started (PID=$worker_pid, port=$PORT)"
    echo "[OK] log=$LOG_FILE"
    exit 0
  fi
  if ! kill -0 "$worker_pid" 2>/dev/null; then
    break
  fi
  sleep 0.25
done

echo "[ERROR] worker API health check failed" >&2
tail -n 50 "$LOG_FILE" >&2 || true
rm -f "$PID_FILE"
exit 1
