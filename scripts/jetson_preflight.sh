#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

if [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/venv/bin/python"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
  printf '[ERROR] Python virtual environment not found under %s\n' "${PROJECT_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON}" "${SCRIPT_DIR}/jetson_preflight.py" "$@"
