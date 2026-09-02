#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
VENV_PATH="${PROJECT_ROOT}/venv"
TORCH_WHEEL="${NVIDIA_TORCH_WHEEL:-}"
TORCHVISION_WHEEL="${NVIDIA_TORCHVISION_WHEEL:-}"
SKIP_APT=0
NO_ENV_CREATE=0
SKIP_PREFLIGHT=0
ALLOW_NO_CAMERA=0

info() { printf '[INFO] %s\n' "$*"; }
ok() { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: bash scripts/install_jetson.sh [options]

Options:
  --skip-apt                    Do not install Ubuntu packages
  --venv <path>                 Virtual environment path (default: ./venv)
  --torch-wheel <path-or-url>   User-supplied JetPack-compatible torch wheel
  --torchvision-wheel <value>   User-supplied compatible torchvision wheel
  --no-env-create               Do not create .env when it is absent
  --skip-preflight              Skip the final hardware preflight
  --allow-no-camera             Allow preflight to skip absent USB cameras
  -h, --help                    Show this help

The script never enables a boot service and never guesses NVIDIA wheel URLs.
EOF
}

while (($#)); do
  case "$1" in
    --skip-apt) SKIP_APT=1; shift ;;
    --venv) (($# >= 2)) || die '--venv requires a path'; VENV_PATH="$2"; shift 2 ;;
    --torch-wheel) (($# >= 2)) || die '--torch-wheel requires a value'; TORCH_WHEEL="$2"; shift 2 ;;
    --torchvision-wheel) (($# >= 2)) || die '--torchvision-wheel requires a value'; TORCHVISION_WHEEL="$2"; shift 2 ;;
    --no-env-create) NO_ENV_CREATE=1; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --allow-no-camera) ALLOW_NO_CAMERA=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ ${EUID} -ne 0 ]] || die 'run this installer as the project user, not root'
[[ "$(uname -m)" == 'aarch64' ]] || die "unsupported architecture: $(uname -m); aarch64 Jetson required"
[[ -r /etc/nv_tegra_release ]] || dpkg-query -W nvidia-l4t-core >/dev/null 2>&1 \
  || die 'NVIDIA L4T was not detected; run this on a Jetson with JetPack installed'
command -v python3 >/dev/null || die 'python3 is required'

info "project=${PROJECT_ROOT}"
info "architecture=$(uname -m)"
info "python=$(python3 --version 2>&1)"
[[ -r /etc/nv_tegra_release ]] && info "L4T=$(head -n 1 /etc/nv_tegra_release)"
dpkg-query -W -f='[INFO] nvidia-l4t-core=${Version}\n' nvidia-l4t-core 2>/dev/null || true

available_kb="$(df -Pk "${PROJECT_ROOT}" | awk 'NR==2 {print $4}')"
((available_kb >= 5 * 1024 * 1024)) || die 'at least 5 GiB of free storage is required'
ok 'Jetson environment detected'

if ((SKIP_APT == 0)); then
  info 'Installing minimal Ubuntu runtime packages (sudo is limited to apt-get)'
  sudo apt-get update
  sudo apt-get install -y \
    git curl python3-pip python3-venv python3-dev build-essential \
    libopenblas-dev libgl1 libglib2.0-0 v4l-utils fontconfig fonts-noto-cjk
  ok 'Ubuntu runtime packages installed'
else
  warn 'Ubuntu package installation skipped by request'
fi

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  info "Creating virtual environment with system site packages: ${VENV_PATH}"
  python3 -m venv --system-site-packages "${VENV_PATH}"
else
  info "Reusing virtual environment: ${VENV_PATH}"
fi
PYTHON="${VENV_PATH}/bin/python"
PIP=("${PYTHON}" -m pip)

verify_torch() {
  "${PYTHON}" - <<'PY'
import torch
import torchvision
x = torch.ones(4, device='cuda')
assert float((x * 2).sum().cpu()) == 8.0
torch.cuda.synchronize()
torchvision.models.efficientnet_b0(weights=None)
print(f'[OK] torch={torch.__version__} torchvision={torchvision.__version__} cuda={torch.version.cuda}')
PY
}

if verify_torch; then
  ok 'Existing JetPack-compatible PyTorch stack passed the CUDA probe'
else
  [[ -n "${TORCH_WHEEL}" && -n "${TORCHVISION_WHEEL}" ]] || die \
    'PyTorch CUDA verification failed. Supply verified JetPack-compatible --torch-wheel and --torchvision-wheel values.'
  info 'Installing only the user-supplied torch and torchvision wheels'
  "${PIP[@]}" install "${TORCH_WHEEL}" "${TORCHVISION_WHEEL}"
  verify_torch || die 'User-supplied torch/torchvision wheels failed CUDA verification'
fi

info 'Installing pinned application dependencies'
"${PIP[@]}" install -r "${PROJECT_ROOT}/requirements_jetson.txt"
"${PYTHON}" - <<'PY'
import cv2
import dotenv
import flask
import mediapipe
import numpy
import PIL
import pytorch_grad_cam
import qrcode
import reportlab
import requests
print(f'[OK] application imports passed; cv2={cv2.__version__}, mediapipe={mediapipe.__version__}, numpy={numpy.__version__}')
PY

ENV_PATH="${PROJECT_ROOT}/.env"
if [[ ! -e "${ENV_PATH}" ]]; then
  if ((NO_ENV_CREATE)); then
    warn '.env creation skipped by request'
  else
    cp "${PROJECT_ROOT}/.env.example" "${ENV_PATH}"
    ENV_PATH="${ENV_PATH}" "${PYTHON}" - <<'PY'
import os
import secrets
from pathlib import Path

path = Path(os.environ['ENV_PATH'])
lines = path.read_text(encoding='utf-8').splitlines()
replacements = {
    'HASH_PEPPER': secrets.token_hex(32),
    'EYE_APP_SECRET_KEY': secrets.token_hex(32),
    'MODEL_DEVICE': 'jetson',
    'TORCH_DEVICE': 'cuda',
    'CUDA_DEVICE_INDEX': '0',
}
updated = []
for line in lines:
    key = line.split('=', 1)[0].strip() if '=' in line else ''
    if key in replacements:
        updated.append(f'{key}={replacements[key]}')
    else:
        updated.append(line)
path.write_text('\n'.join(updated) + '\n', encoding='utf-8')
PY
    chmod 600 "${ENV_PATH}"
    ok 'Created protected .env with one-time persistent secrets (values not displayed)'
  fi
else
  chmod 600 "${ENV_PATH}"
  warn 'Existing .env preserved without changing any value'
  "${PYTHON}" - "${ENV_PATH}" <<'PY'
import sys
from dotenv import dotenv_values
required = ('MODEL_DEVICE', 'TORCH_DEVICE', 'CUDA_DEVICE_INDEX', 'HASH_PEPPER', 'EYE_APP_SECRET_KEY')
values = dotenv_values(sys.argv[1])
missing = [key for key in required if not str(values.get(key) or '').strip()]
if missing:
    print('[WARN] Existing .env has missing or empty deployment keys: ' + ', '.join(missing))
PY
fi

chmod +x "${PROJECT_ROOT}/scripts/mediflow-kiosk" "${PROJECT_ROOT}/scripts/jetson_preflight.sh"
mkdir -p "${HOME}/.local/bin"
COMMAND_LINK="${HOME}/.local/bin/mediflow-kiosk"
if [[ -L "${COMMAND_LINK}" && "$(readlink -f "${COMMAND_LINK}")" == "$(readlink -f "${PROJECT_ROOT}/scripts/mediflow-kiosk")" ]]; then
  ok 'Existing mediflow-kiosk command link is correct'
elif [[ -e "${COMMAND_LINK}" || -L "${COMMAND_LINK}" ]]; then
  die "refusing to replace existing path: ${COMMAND_LINK}"
else
  ln -s "${PROJECT_ROOT}/scripts/mediflow-kiosk" "${COMMAND_LINK}"
  ok "Installed command link: ${COMMAND_LINK}"
fi
case ":${PATH}:" in
  *":${HOME}/.local/bin:"*) ;;
  *) warn "Add ${HOME}/.local/bin to PATH; shell profile was not modified" ;;
esac

if ((SKIP_PREFLIGHT == 0)); then
  preflight_args=()
  ((ALLOW_NO_CAMERA)) && preflight_args+=(--allow-no-camera)
  "${PROJECT_ROOT}/scripts/jetson_preflight.sh" "${preflight_args[@]}"
else
  warn 'Final preflight skipped by request'
fi

ok 'Jetson installation completed; no boot-time service was enabled'
