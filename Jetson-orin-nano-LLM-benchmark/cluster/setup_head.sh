#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUNTIME_DIR="$PROJECT_ROOT/.run/cluster"
INVENTORY_FILE="$RUNTIME_DIR/nodes.local.csv"
IDENTITY_FILE="${CLUSTER_IDENTITY_FILE:-$HOME/.ssh/id_ed25519_llm_cluster}"
WORKER_TOKEN_FILE="$RUNTIME_DIR/worker.token"
SETTINGS_FILE="$RUNTIME_DIR/settings.json"
HEAD_NAME="${CLUSTER_HEAD_NAME:-edge-head}"
HEAD_API_PORT="${CLUSTER_WORKER_PORT:-8000}"

detect_platform() {
  local board=""
  if [[ -r /proc/device-tree/model ]]; then
    board="$(tr -d '\000' </proc/device-tree/model)"
  fi
  if [[ -f /etc/nv_tegra_release || -d /etc/nv_tegra_release.d ]] || command -v nvpmodel >/dev/null 2>&1; then
    printf 'jetson'
  elif [[ "${board,,}" == *"raspberry pi"* ]]; then
    printf 'raspberry-pi'
  else
    printf 'auto'
  fi
}

HEAD_PLATFORM="$(detect_platform)"
HEAD_USER="$(id -un)"

if [[ "$PROJECT_ROOT" == *","* || "$HEAD_USER" == *","* || "$HEAD_NAME" == *","* ]]; then
  echo "[ERROR] project path, user and head name cannot contain commas" >&2
  exit 2
fi

echo "[INFO] preparing head runtime platform=$HEAD_PLATFORM user=$HEAD_USER"
"$SCRIPT_DIR/worker_setup.sh" --install --project-dir "$PROJECT_ROOT"

mkdir -p "$RUNTIME_DIR" "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

if [[ ! -f "$INVENTORY_FILE" ]]; then
  {
    printf 'name,role,host,user,ssh_port,api_port,project_dir,enabled,identity_file,platform\n'
    printf '%s,head,127.0.0.1,%s,22,%s,%s,true,,%s\n' \
      "$HEAD_NAME" "$HEAD_USER" "$HEAD_API_PORT" "$PROJECT_ROOT" "$HEAD_PLATFORM"
  } >"$INVENTORY_FILE"
  chmod 600 "$INVENTORY_FILE"
  echo "[OK] created platform-aware inventory: $INVENTORY_FILE"
else
  echo "[INFO] inventory already exists and was not changed: $INVENTORY_FILE"
fi

if [[ ! -f "$IDENTITY_FILE" ]]; then
  ssh-keygen -t ed25519 -N "" -C "llm-cluster-head@$(hostname)" -f "$IDENTITY_FILE"
  echo "[OK] created cluster SSH identity: $IDENTITY_FILE"
else
  echo "[INFO] cluster SSH identity already exists: $IDENTITY_FILE"
fi

if [[ ! -f "$SETTINGS_FILE" ]]; then
  printf '{\n  "worker_api_auth": false,\n  "dashboard_token_auth": false\n}\n' >"$SETTINGS_FILE"
  echo "[OK] created cluster settings (worker and dashboard token auth disabled by default)"
fi
chmod 600 "$SETTINGS_FILE"

chmod 600 "$IDENTITY_FILE"
chmod 644 "$IDENTITY_FILE.pub"

CLUSTER_NODE_NAME="$HEAD_NAME" CLUSTER_NODE_ROLE="head" CLUSTER_PLATFORM="$HEAD_PLATFORM" \
  PORT="$HEAD_API_PORT" "$SCRIPT_DIR/worker/start.sh"

echo
echo "Worker onboarding public key:"
cat "$IDENTITY_FILE.pub"
echo
echo "Install this key in each worker's ~/.ssh/authorized_keys, then use"
echo "the dashboard LAN device search to register and prepare the worker."
