#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODE="check"
PLAN_ONLY=0
PLATFORM_OVERRIDE=""
REPORT_JSON=""

usage() {
  cat <<'EOF'
Usage: cluster/worker_setup.sh [--check-only|--install] [--project-dir PATH]
       [--report-json PATH]
       cluster/worker_setup.sh --plan-only --platform jetson|raspberry-pi

Detects NVIDIA Jetson Orin or Raspberry Pi 5, checks system dependencies and
verifies the matching llama-cpp-python backend. --install may install a fixed
apt package allowlist only when root or passwordless sudo is available. Python
packages are installed only in PROJECT_DIR/.venv. The script never accepts or
stores a sudo/SSH password and never installs JetPack, CUDA, or an OS image.

--report-json writes an atomic, mode-0600 machine-readable readiness report.
A single-line CLUSTER_READINESS_JSON marker is also printed to stdout.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only) MODE="check"; shift ;;
    --install) MODE="install"; shift ;;
    --project-dir)
      [[ $# -ge 2 ]] || { echo "[ERROR] --project-dir requires a path" >&2; exit 2; }
      PROJECT_DIR="$2"; shift 2 ;;
    --report-json)
      [[ $# -ge 2 ]] || { echo "[ERROR] --report-json requires a path" >&2; exit 2; }
      REPORT_JSON="$2"; shift 2 ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    --platform)
      [[ $# -ge 2 ]] || { echo "[ERROR] --platform requires a value" >&2; exit 2; }
      PLATFORM_OVERRIDE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$MODE" != "check" && "$MODE" != "install" ]]; then
  echo "[ERROR] invalid mode" >&2
  exit 2
fi
if [[ "$PROJECT_DIR" != /* ]]; then
  echo "[ERROR] project directory must be absolute" >&2
  exit 2
fi
if [[ -n "$REPORT_JSON" && "$REPORT_JSON" != /* ]]; then
  echo "[ERROR] report path must be absolute" >&2
  exit 2
fi

read_board_model() {
  if [[ -r /proc/device-tree/model ]]; then
    tr -d '\000' </proc/device-tree/model
  else
    uname -m
  fi
}

detect_platform() {
  local override="${PLATFORM_OVERRIDE:-${CLUSTER_PLATFORM_OVERRIDE:-}}"
  local board
  board="$(read_board_model)"
  if [[ "$override" == "jetson" || "$override" == "raspberry-pi" ]]; then
    printf '%s' "$override"
  elif [[ -f /etc/nv_tegra_release || -d /etc/nv_tegra_release.d ]] || command -v nvpmodel >/dev/null 2>&1; then
    printf 'jetson'
  elif [[ "${board,,}" == *"raspberry pi"* ]]; then
    printf 'raspberry-pi'
  else
    printf 'unsupported'
  fi
}

PLATFORM_KIND="$(detect_platform)"
BOARD_MODEL="$(read_board_model)"
ARCH="$(uname -m)"
HOSTNAME_VALUE="$(hostname)"
CHECKS_FILE="$(mktemp)"
MANUAL_FILE="$(mktemp)"
MISSING_FILE="$(mktemp)"
REPORT_EMITTED=0
LOCK_FD_OPEN=0
failures=0
manual_failures=0
blocked_failures=0
repairable_failures=0
backend_kind="unknown"
backend_verified="false"
python_version=""
model_count=0
disk_free_gb=""

sanitize_field() {
  printf '%s' "$1" | tr '\t\r\n' '   '
}

add_check() {
  local check_id="$1" label="$2" check_status="$3" auto_fixable="$4" detail="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(sanitize_field "$check_id")" \
    "$(sanitize_field "$label")" \
    "$(sanitize_field "$check_status")" \
    "$(sanitize_field "$auto_fixable")" \
    "$(sanitize_field "$detail")" >>"$CHECKS_FILE"
}

add_manual_command() {
  printf '%s\n' "$(sanitize_field "$1")" >>"$MANUAL_FILE"
}

record_failure() {
  local class="$1" message="$2"
  echo "[FAIL] $message" >&2
  failures=$((failures + 1))
  case "$class" in
    manual) manual_failures=$((manual_failures + 1)) ;;
    blocked) blocked_failures=$((blocked_failures + 1)) ;;
    repairable) repairable_failures=$((repairable_failures + 1)) ;;
  esac
}

emit_report() {
  [[ "$REPORT_EMITTED" -eq 0 ]] || return 0
  REPORT_EMITTED=1
  local readiness_status="ready"
  if (( blocked_failures > 0 )); then
    readiness_status="blocked"
  elif (( manual_failures > 0 )); then
    readiness_status="manual"
  elif (( repairable_failures > 0 || failures > 0 )); then
    readiness_status="repairable"
  fi
  local checked_at
  checked_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  CLUSTER_REPORT_STATUS="$readiness_status" \
  CLUSTER_REPORT_NODE="$HOSTNAME_VALUE" \
  CLUSTER_REPORT_CHECKED_AT="$checked_at" \
  CLUSTER_REPORT_PLATFORM="$PLATFORM_KIND" \
  CLUSTER_REPORT_BOARD="$BOARD_MODEL" \
  CLUSTER_REPORT_ARCH="$ARCH" \
  CLUSTER_REPORT_PROJECT="$PROJECT_DIR" \
  CLUSTER_REPORT_VENV="$PROJECT_DIR/.venv" \
  CLUSTER_REPORT_MODE="$MODE" \
  CLUSTER_REPORT_BACKEND="$backend_kind" \
  CLUSTER_REPORT_BACKEND_VERIFIED="$backend_verified" \
  CLUSTER_REPORT_PYTHON="$python_version" \
  CLUSTER_REPORT_MODEL_COUNT="$model_count" \
  CLUSTER_REPORT_DISK_GB="$disk_free_gb" \
  CLUSTER_REPORT_CHECKS_FILE="$CHECKS_FILE" \
  CLUSTER_REPORT_MANUAL_FILE="$MANUAL_FILE" \
  CLUSTER_REPORT_MISSING_FILE="$MISSING_FILE" \
  CLUSTER_REPORT_OUTPUT="$REPORT_JSON" \
  python3 - <<'PY'
import json
import os
import pathlib
import tempfile

def lines(path):
    try:
        return pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

checks = []
for line in lines(os.environ["CLUSTER_REPORT_CHECKS_FILE"]):
    parts = line.split("\t", 4)
    if len(parts) != 5:
        continue
    check_id, label, status, auto_fixable, detail = parts
    checks.append({
        "id": check_id,
        "label": label,
        "status": status,
        "detail": detail,
        "auto_fixable": auto_fixable == "true",
    })

raw_disk = os.environ.get("CLUSTER_REPORT_DISK_GB", "")
try:
    disk_free_gb = round(float(raw_disk), 2)
except ValueError:
    disk_free_gb = None

report = {
    "schema_version": 1,
    "node": os.environ["CLUSTER_REPORT_NODE"],
    "status": os.environ["CLUSTER_REPORT_STATUS"],
    "checked_at": os.environ["CLUSTER_REPORT_CHECKED_AT"],
    "mode": os.environ["CLUSTER_REPORT_MODE"],
    "platform": os.environ["CLUSTER_REPORT_PLATFORM"],
    "board_model": os.environ["CLUSTER_REPORT_BOARD"],
    "architecture": os.environ["CLUSTER_REPORT_ARCH"],
    "project_dir": os.environ["CLUSTER_REPORT_PROJECT"],
    "venv_path": os.environ["CLUSTER_REPORT_VENV"],
    "python": os.environ.get("CLUSTER_REPORT_PYTHON", ""),
    "backend": {
        "kind": os.environ.get("CLUSTER_REPORT_BACKEND", "unknown"),
        "verified": os.environ.get("CLUSTER_REPORT_BACKEND_VERIFIED") == "true",
    },
    "model_count": int(os.environ.get("CLUSTER_REPORT_MODEL_COUNT", "0") or 0),
    "disk_free_gb": disk_free_gb,
    "checks": checks,
    "missing_system_packages": lines(os.environ["CLUSTER_REPORT_MISSING_FILE"]),
    "manual_commands": lines(os.environ["CLUSTER_REPORT_MANUAL_FILE"]),
}
compact = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
output = os.environ.get("CLUSTER_REPORT_OUTPUT", "")
if output:
    target = pathlib.Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        target.chmod(0o600)
    finally:
        try:
            pathlib.Path(temporary).unlink()
        except FileNotFoundError:
            pass
print("CLUSTER_READINESS_JSON=" + compact)
PY
}

cleanup() {
  local original_status=$?
  emit_report || true
  rm -f "$CHECKS_FILE" "$MANUAL_FILE" "$MISSING_FILE"
  if [[ "$LOCK_FD_OPEN" -eq 1 ]]; then
    flock -u 9 2>/dev/null || true
  fi
  return "$original_status"
}
trap cleanup EXIT

echo "[INFO] node=$HOSTNAME_VALUE platform=$PLATFORM_KIND arch=$ARCH"
echo "[INFO] board=$BOARD_MODEL"
echo "[INFO] project=$PROJECT_DIR mode=$MODE"

common_packages=(
  ca-certificates curl git rsync openssh-client iproute2 util-linux build-essential cmake ninja-build pkg-config
  python3 python3-dev python3-venv
)
required_packages=("${common_packages[@]}")
if [[ "$PLATFORM_KIND" == "raspberry-pi" ]]; then
  required_packages+=(libopenblas-dev)
fi

if [[ "$PLAN_ONLY" -eq 1 ]]; then
  echo "[PLAN] platform=$PLATFORM_KIND"
  echo "[PLAN] apt=${required_packages[*]}"
  if [[ "$PLATFORM_KIND" == "jetson" ]]; then
    echo "[PLAN] backend=cuda"
    echo "[PLAN] cmake=-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87"
  elif [[ "$PLATFORM_KIND" == "raspberry-pi" ]]; then
    echo "[PLAN] backend=openblas n_gpu_layers=0"
    echo "[PLAN] cmake=-DGGML_NATIVE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DCMAKE_BUILD_TYPE=Release"
  else
    echo "[FAIL] unsupported platform plan" >&2
    exit 1
  fi
  REPORT_EMITTED=1
  exit 0
fi

if [[ "$PLATFORM_KIND" == "unsupported" ]]; then
  add_check "platform" "지원 하드웨어" "fail" "false" "NVIDIA Jetson Orin 또는 Raspberry Pi 5가 필요합니다."
  record_failure blocked "unsupported board; expected NVIDIA Jetson Orin or Raspberry Pi 5"
else
  add_check "platform" "지원 하드웨어" "pass" "false" "$BOARD_MODEL · $PLATFORM_KIND"
fi
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  add_check "architecture" "64-bit ARM OS" "fail" "false" "감지: $ARCH"
  record_failure blocked "64-bit ARM OS is required (detected $ARCH)"
else
  add_check "architecture" "64-bit ARM OS" "pass" "false" "$ARCH"
fi

if [[ "$MODE" == "install" ]]; then
  if mkdir -p "$PROJECT_DIR/.run/cluster" 2>/dev/null; then
    chmod 700 "$PROJECT_DIR/.run/cluster" 2>/dev/null || true
    if command -v flock >/dev/null 2>&1; then
      exec 9>"$PROJECT_DIR/.run/cluster/environment-setup.lock"
      chmod 600 "$PROJECT_DIR/.run/cluster/environment-setup.lock" 2>/dev/null || true
      if flock -w 30 9; then
        LOCK_FD_OPEN=1
      else
        add_check "install_lock" "환경 구성 잠금" "fail" "true" "다른 설치 작업이 실행 중입니다."
        record_failure repairable "another environment setup is already running"
      fi
    else
      add_check "install_lock" "환경 구성 잠금" "fail" "false" "flock 명령이 없습니다."
      record_failure blocked "flock is required for safe environment installation"
    fi
  else
    add_check "project_writable" "프로젝트 쓰기 권한" "fail" "false" "$PROJECT_DIR를 만들거나 쓸 수 없습니다."
    record_failure blocked "project directory is not writable: $PROJECT_DIR"
  fi
fi

missing_packages=()
venv_probe_dir=""
venv_works=0
if command -v python3 >/dev/null 2>&1; then
  venv_probe_dir="$(mktemp -d 2>/dev/null || true)"
  if [[ -n "$venv_probe_dir" ]] && python3 -m venv "$venv_probe_dir/check" >/dev/null 2>&1; then
    venv_works=1
  fi
  [[ -n "$venv_probe_dir" ]] && rm -rf "$venv_probe_dir"
fi
if command -v dpkg-query >/dev/null 2>&1; then
  for package_name in "${required_packages[@]}"; do
    if [[ "$package_name" == "python3-venv" && "$venv_works" -eq 1 ]]; then
      continue
    fi
    if ! dpkg-query -W -f='${db:Status-Abbrev}' "$package_name" 2>/dev/null | grep -q '^ii'; then
      missing_packages+=("$package_name")
      printf '%s\n' "$package_name" >>"$MISSING_FILE"
    fi
  done
else
  add_check "system_packages" "시스템 빌드 패키지" "fail" "false" "dpkg-query가 없어 지원하는 Ubuntu/Debian 환경으로 확인할 수 없습니다."
  record_failure blocked "dpkg-query is unavailable; Debian/Ubuntu based 64-bit OS is required"
fi

if (( ${#missing_packages[@]} > 0 )); then
  can_sudo=0
  apt_prefix=()
  if [[ "$(id -u)" -eq 0 ]]; then
    can_sudo=1
  elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    can_sudo=1
    apt_prefix=(sudo -n)
  fi
  manual_command="sudo apt-get update && sudo apt-get install -y ${missing_packages[*]}"
  if [[ "$MODE" == "install" && "$can_sudo" -eq 1 && "$LOCK_FD_OPEN" -eq 1 ]]; then
    echo "[INFO] installing fixed system package allowlist: ${missing_packages[*]}"
    if "${apt_prefix[@]}" apt-get update && \
       "${apt_prefix[@]}" apt-get install -y --no-install-recommends "${missing_packages[@]}"; then
      : >"$MISSING_FILE"
      missing_packages=()
      add_check "system_packages" "시스템 빌드 패키지" "pass" "false" "고정 허용 패키지를 설치하고 확인했습니다."
    else
      add_check "system_packages" "시스템 빌드 패키지" "fail" "true" "apt 설치가 실패했습니다. 작업 로그를 확인하세요."
      record_failure repairable "fixed allowlist apt installation failed"
    fi
  elif [[ "$can_sudo" -eq 1 ]]; then
    add_check "system_packages" "시스템 빌드 패키지" "fail" "true" "누락: ${missing_packages[*]}"
    record_failure repairable "missing system packages: ${missing_packages[*]}"
  else
    add_manual_command "$manual_command"
    add_check "system_packages" "시스템 빌드 패키지" "fail" "false" "수동 sudo 1회 필요: ${missing_packages[*]}"
    record_failure manual "system packages require sudo. Run once on the node: $manual_command"
  fi
elif command -v dpkg-query >/dev/null 2>&1; then
  echo "[OK] system build dependencies"
  add_check "system_packages" "시스템 빌드 패키지" "pass" "false" "필수 apt 패키지가 모두 설치되어 있습니다."
fi

if [[ "$PLATFORM_KIND" == "jetson" ]]; then
  backend_kind="cuda"
  if [[ "${BOARD_MODEL,,}" != *"orin"* ]]; then
    add_check "jetson_generation" "Jetson 세대" "fail" "false" "현재 자동 빌드는 Orin(sm_87)만 지원합니다: $BOARD_MODEL"
    record_failure blocked "automatic CUDA build currently supports Jetson Orin only"
  else
    add_check "jetson_generation" "Jetson 세대" "pass" "false" "Orin · CUDA architecture sm_87"
  fi
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    cuda_detail="$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -n 1 || true)"
    echo "[OK] CUDA compiler: $cuda_detail"
    add_check "platform_runtime" "JetPack / CUDA" "pass" "false" "${cuda_detail:-CUDA compiler available}"
  else
    add_check "platform_runtime" "JetPack / CUDA" "fail" "false" "JetPack/CUDA는 자동 설치하지 않습니다. NVIDIA 이미지 설치가 필요합니다."
    record_failure manual "JetPack/CUDA is missing; install the matching NVIDIA JetPack image manually"
  fi
  if command -v nvpmodel >/dev/null 2>&1; then
    power_mode="$(nvpmodel -q 2>/dev/null | head -n 1 || true)"
    add_check "power_runtime" "Jetson 전력 런타임" "pass" "false" "${power_mode:-nvpmodel available}"
  else
    add_check "power_runtime" "Jetson 전력 런타임" "fail" "false" "nvpmodel이 없습니다. JetPack 설치를 확인하세요."
    record_failure manual "nvpmodel is missing; verify the JetPack installation"
  fi
elif [[ "$PLATFORM_KIND" == "raspberry-pi" ]]; then
  backend_kind="openblas"
  if [[ "${BOARD_MODEL,,}" == *"raspberry pi 5"* ]]; then
    add_check "pi_generation" "Raspberry Pi 세대" "pass" "false" "Raspberry Pi 5 · CPU/OpenBLAS"
  else
    add_check "pi_generation" "Raspberry Pi 세대" "fail" "false" "현재 최적화 대상은 Raspberry Pi 5입니다: $BOARD_MODEL"
    record_failure blocked "optimized target is Raspberry Pi 5; detected $BOARD_MODEL"
  fi
fi

if [[ -d "$PROJECT_DIR" ]]; then
  add_check "project" "프로젝트 폴더" "pass" "false" "$PROJECT_DIR"
  disk_free_kb="$(df -Pk "$PROJECT_DIR" 2>/dev/null | awk 'NR==2 {print $4}' || true)"
  if [[ "$disk_free_kb" =~ ^[0-9]+$ ]]; then
    disk_free_gb="$(python3 -c "print(round($disk_free_kb / 1024 / 1024, 2))" 2>/dev/null || true)"
    if (( disk_free_kb < 2097152 )); then
      add_check "disk" "프로젝트 디스크 여유" "fail" "false" "${disk_free_gb:-0} GB · 최소 2 GB 이상의 여유 공간을 권장합니다."
      record_failure manual "less than 2 GB free disk space is available for the runtime"
    else
      add_check "disk" "프로젝트 디스크 여유" "pass" "false" "${disk_free_gb} GB free"
    fi
  fi
else
  add_check "project" "프로젝트 폴더" "fail" "true" "코드 동기화로 생성할 수 있습니다: $PROJECT_DIR"
  record_failure repairable "project directory is missing: $PROJECT_DIR"
fi

python_bin="$PROJECT_DIR/.venv/bin/python"
if [[ "$MODE" == "install" && ! -x "$python_bin" && -d "$PROJECT_DIR" && "$LOCK_FD_OPEN" -eq 1 && "$blocked_failures" -eq 0 && "$manual_failures" -eq 0 ]]; then
  echo "[INFO] creating project-local Python virtual environment"
  venv_build="$PROJECT_DIR/.venv.build-$$"
  if [[ -e "$PROJECT_DIR/.venv" ]]; then
    add_check "venv" "프로젝트 가상환경" "fail" "false" "기존 .venv가 불완전합니다. 데이터 보호를 위해 자동 덮어쓰지 않습니다."
    record_failure manual "an incomplete .venv exists; move it aside and run automatic setup again"
  elif python3 -m venv "$venv_build" && mv "$venv_build" "$PROJECT_DIR/.venv"; then
    add_check "venv" "프로젝트 가상환경" "pass" "false" "$PROJECT_DIR/.venv 생성 완료"
  else
    rm -rf "$venv_build"
    add_check "venv" "프로젝트 가상환경" "fail" "true" "python3 -m venv 실행이 실패했습니다."
    record_failure repairable "failed to create the project virtual environment"
  fi
elif [[ -x "$python_bin" ]]; then
  python_version="$($python_bin --version 2>&1 || true)"
  echo "[OK] virtual environment: $python_version"
  add_check "venv" "프로젝트 가상환경" "pass" "false" "$PROJECT_DIR/.venv · $python_version"
elif [[ ! -e "$PROJECT_DIR/.venv" ]]; then
  add_check "venv" "프로젝트 가상환경" "fail" "true" "$PROJECT_DIR/.venv가 없습니다."
  record_failure repairable "virtual environment is missing: $PROJECT_DIR/.venv"
else
  add_check "venv" "프로젝트 가상환경" "fail" "false" "기존 .venv의 Python을 실행할 수 없습니다."
  record_failure manual "the existing project virtual environment is incomplete or not executable"
fi

runtime_requirements="$PROJECT_DIR/cluster/requirements-runtime.txt"
runtime_versions_ok() {
  "$python_bin" - "$runtime_requirements" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    package, expected = line.split("==", 1)
    try:
        actual = version(package)
    except PackageNotFoundError:
        raise SystemExit(1)
    if actual != expected:
        print(f"{package}: expected {expected}, found {actual}", file=sys.stderr)
        raise SystemExit(1)
PY
}

if [[ -x "$python_bin" && -f "$runtime_requirements" ]]; then
  python_version="$($python_bin --version 2>&1 || true)"
  common_imports='import fastapi, uvicorn, psutil, jinja2, sse_starlette'
  if "$python_bin" -c "$common_imports" >/dev/null 2>&1 && runtime_versions_ok; then
    echo "[OK] common Python runtime packages"
    add_check "python_runtime" "Python 실험 패키지" "pass" "false" "requirements-runtime.txt 고정 버전 일치"
  elif [[ "$MODE" == "install" && "$LOCK_FD_OPEN" -eq 1 && "$blocked_failures" -eq 0 && "$manual_failures" -eq 0 ]]; then
    echo "[INFO] installing pinned common Python dependencies into $PROJECT_DIR/.venv"
    if "$python_bin" -m pip install --upgrade pip setuptools wheel && \
       "$python_bin" -m pip install --requirement "$runtime_requirements" && \
       "$python_bin" -c "$common_imports" >/dev/null 2>&1 && runtime_versions_ok; then
      add_check "python_runtime" "Python 실험 패키지" "pass" "false" "프로젝트 .venv에 고정 버전 설치 완료"
    else
      add_check "python_runtime" "Python 실험 패키지" "fail" "true" "고정 Python 패키지 설치 또는 검증이 실패했습니다."
      record_failure repairable "failed to install pinned Python runtime packages"
    fi
  else
    add_check "python_runtime" "Python 실험 패키지" "fail" "true" "requirements-runtime.txt의 고정 패키지가 없거나 버전이 다릅니다."
    record_failure repairable "required Python packages are missing or differ from cluster/requirements-runtime.txt"
  fi
elif [[ -x "$python_bin" ]]; then
  add_check "python_runtime" "Python 실험 패키지" "fail" "true" "$runtime_requirements 파일이 없습니다. 코드 동기화가 필요합니다."
  record_failure repairable "runtime requirements file is missing"
else
  add_check "python_runtime" "Python 실험 패키지" "fail" "true" "프로젝트 .venv를 만든 뒤 고정 패키지를 설치해야 합니다."
  record_failure repairable "project Python runtime is unavailable"
fi

if [[ "$PLATFORM_KIND" == "jetson" && -x "$python_bin" ]]; then
  if "$python_bin" -c 'import jtop' >/dev/null 2>&1; then
    add_check "telemetry_package" "Jetson 원격 측정 패키지" "pass" "false" "jetson-stats Python 패키지 사용 가능"
  elif [[ "$MODE" == "install" && "$LOCK_FD_OPEN" -eq 1 && "$blocked_failures" -eq 0 && "$manual_failures" -eq 0 ]] && \
       "$python_bin" -m pip install 'jetson-stats==4.3.2'; then
    add_check "telemetry_package" "Jetson 원격 측정 패키지" "pass" "false" "jetson-stats 4.3.2 설치 완료"
  else
    add_check "telemetry_package" "Jetson 원격 측정 패키지" "warn" "true" "jtop 없이도 psutil 기본 지표와 LLM 실험은 가능합니다."
  fi
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet jtop.service; then
    add_check "telemetry_service" "jtop 고급 측정 서비스" "pass" "false" "jtop.service active"
  else
    add_check "telemetry_service" "jtop 고급 측정 서비스" "warn" "false" "고급 GPU/전력 지표는 system-level jtop 서비스와 권한을 수동 구성해야 합니다."
  fi
fi

verify_backend() {
  CLUSTER_EXPECTED_PLATFORM="$PLATFORM_KIND" "$python_bin" - <<'PY'
import os
import subprocess
from pathlib import Path
import llama_cpp as llama_package
from llama_cpp import llama_cpp

expected = os.environ["CLUSTER_EXPECTED_PLATFORM"]
info = llama_cpp.llama_print_system_info().decode("utf-8", errors="replace")
gpu = bool(llama_cpp.llama_supports_gpu_offload())
print(" ".join(info.split())[:800])
if getattr(llama_package, "__version__", "") != "0.3.20":
    raise SystemExit(1)
if expected == "jetson":
    raise SystemExit(0 if gpu and "CUDA" in info.upper() else 1)
if expected == "raspberry-pi":
    library_root = Path(llama_package.__file__).resolve().parent
    linked_openblas = False
    for candidate in library_root.rglob("*.so"):
        try:
            linked = subprocess.check_output(["ldd", str(candidate)], text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            continue
        if "openblas" in linked.lower():
            linked_openblas = True
            break
    normalized = " ".join(info.upper().split())
    arm_optimized = "NEON = 1" in normalized and "ARM_FMA = 1" in normalized
    raise SystemExit(0 if not gpu and linked_openblas and arm_optimized else 1)
raise SystemExit(1)
PY
}

if [[ -x "$python_bin" && "$PLATFORM_KIND" != "unsupported" ]]; then
  backend_log="$(mktemp)"
  if verify_backend >"$backend_log" 2>&1; then
    backend_verified="true"
    echo "[OK] llama-cpp-python backend verified for $PLATFORM_KIND"
    add_check "llm_backend" "LLM 네이티브 백엔드" "pass" "false" "llama-cpp-python 0.3.20 · $backend_kind 검증 완료"
  elif [[ "$MODE" == "install" && "$LOCK_FD_OPEN" -eq 1 && "$blocked_failures" -eq 0 && "$manual_failures" -eq 0 ]]; then
    echo "[INFO] building llama-cpp-python 0.3.20 for $PLATFORM_KIND inside project .venv"
    build_ok=0
    if [[ "$PLATFORM_KIND" == "jetson" ]]; then
      if PATH="/usr/local/cuda/bin:$PATH" \
         CUDACXX="/usr/local/cuda/bin/nvcc" \
         CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}" \
         CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87" \
         FORCE_CMAKE=1 "$python_bin" -m pip install --force-reinstall --no-cache-dir \
           --no-binary=llama-cpp-python 'llama-cpp-python==0.3.20'; then
        build_ok=1
      fi
    elif [[ "$PLATFORM_KIND" == "raspberry-pi" ]]; then
      if CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}" \
         CMAKE_ARGS="-DGGML_NATIVE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DCMAKE_BUILD_TYPE=Release" \
         FORCE_CMAKE=1 "$python_bin" -m pip install --force-reinstall --no-cache-dir \
           --no-binary=llama-cpp-python 'llama-cpp-python==0.3.20'; then
        build_ok=1
      fi
    fi
    if [[ "$build_ok" -eq 1 ]] && verify_backend >"$backend_log" 2>&1; then
      backend_verified="true"
      add_check "llm_backend" "LLM 네이티브 백엔드" "pass" "false" "llama-cpp-python 0.3.20 · $backend_kind 빌드 및 검증 완료"
    else
      add_check "llm_backend" "LLM 네이티브 백엔드" "fail" "true" "소스 빌드 후 $backend_kind 검증에 실패했습니다."
      record_failure repairable "llama-cpp-python backend verification failed after build"
      sed -n '1,12p' "$backend_log" >&2 || true
    fi
  else
    add_check "llm_backend" "LLM 네이티브 백엔드" "fail" "true" "플랫폼에 맞는 llama-cpp-python 0.3.20 $backend_kind 백엔드가 필요합니다."
    record_failure repairable "llama-cpp-python backend is missing or does not match $PLATFORM_KIND"
    sed -n '1,12p' "$backend_log" >&2 || true
  fi
  rm -f "$backend_log"
elif [[ "$PLATFORM_KIND" != "unsupported" ]]; then
  add_check "llm_backend" "LLM 네이티브 백엔드" "fail" "true" "프로젝트 .venv 생성 후 $backend_kind 백엔드를 설치해야 합니다."
  record_failure repairable "LLM backend cannot be verified without a working project Python"
fi

# A native llama build may resolve transitive dependencies after the common
# runtime check above. Revalidate the pinned dashboard/worker packages last so
# every completed setup leaves a reproducible project environment.
if [[ -x "$python_bin" && -f "$runtime_requirements" ]]; then
  if "$python_bin" -c "$common_imports" >/dev/null 2>&1 && runtime_versions_ok; then
    add_check "python_runtime_final" "Python 고정 버전 재검증" "pass" "false" "네이티브 백엔드 확인 후에도 고정 버전 유지"
  elif [[ "$MODE" == "install" && "$LOCK_FD_OPEN" -eq 1 && "$blocked_failures" -eq 0 && "$manual_failures" -eq 0 ]] && \
       "$python_bin" -m pip install --requirement "$runtime_requirements" && \
       "$python_bin" -c "$common_imports" >/dev/null 2>&1 && runtime_versions_ok; then
    add_check "python_runtime_final" "Python 고정 버전 재검증" "pass" "false" "고정 버전 복구 및 재검증 완료"
  else
    add_check "python_runtime_final" "Python 고정 버전 재검증" "fail" "true" "네이티브 백엔드 처리 후 패키지 버전이 다릅니다."
    record_failure repairable "pinned Python runtime changed after native backend handling"
  fi
fi

if [[ -d "$PROJECT_DIR/models" ]]; then
  model_count="$(find "$PROJECT_DIR/models" -type f -name '*.gguf' 2>/dev/null | wc -l | tr -d ' ')"
fi
if (( model_count > 0 )); then
  add_check "models" "GGUF 모델" "pass" "false" "$model_count개 모델 발견"
else
  add_check "models" "GGUF 모델" "warn" "true" "모델이 없습니다. 일반 분산 실험 전 모델 동기화가 필요합니다."
fi
echo "[INFO] GGUF models: $model_count"

if (( failures > 0 )); then
  echo "[FAIL] worker preflight found $failures problem(s)" >&2
  exit 1
fi
echo "[OK] worker is ready platform=$PLATFORM_KIND backend=$backend_kind"
exit 0
