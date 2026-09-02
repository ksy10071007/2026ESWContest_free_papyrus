#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MANAGER="${PROJECT_ROOT}/scripts/mediflow-kiosk"

"${MANAGER}" start

if [[ "${OPEN_KIOSK_BROWSER:-0}" != '1' ]]; then
  printf '[INFO] Browser launch skipped. Set OPEN_KIOSK_BROWSER=1 for the optional local kiosk browser.\n'
  exit 0
fi

if [[ -n "${BROWSER_COMMAND:-}" ]]; then
  printf '[ERROR] BROWSER_COMMAND is not accepted; choose from the detected fixed browser executables.\n' >&2
  exit 1
fi

BROWSER_URL="${BROWSER_URL:-http://127.0.0.1:5000/}"
LOG_DIR="${PROJECT_ROOT}/runtime/log"
mkdir -p "${LOG_DIR}"
chmod 700 "${PROJECT_ROOT}/runtime" "${LOG_DIR}" 2>/dev/null || true

browser=()
for candidate in firefox epiphany-browser epiphany chromium-browser chromium; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    browser=("$(command -v "${candidate}")")
    break
  fi
done

if ((${#browser[@]} == 0)); then
  printf '[WARN] No supported local browser was found; open %s manually.\n' "${BROWSER_URL}" >&2
  exit 0
fi

browser_args=()
browser_name="$(basename -- "${browser[0]}")"
if [[ "${browser_name}" == 'firefox' ]]; then
  export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
  export MOZ_WEBRENDER="${MOZ_WEBRENDER:-0}"
  browser_args=(--new-window)
fi

printf '[INFO] Opening optional kiosk browser (%s): %s\n' "${browser_name}" "${BROWSER_URL}"
nohup "${browser[@]}" "${browser_args[@]}" "${BROWSER_URL}" >"${LOG_DIR}/browser.log" 2>&1 </dev/null &
printf '[OK] Browser launched with PID=%s; service lifecycle remains managed separately.\n' "$!"
