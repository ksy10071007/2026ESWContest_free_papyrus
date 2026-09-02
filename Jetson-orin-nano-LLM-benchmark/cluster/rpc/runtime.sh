#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR="$PROJECT_ROOT/.run/cluster"
SOURCE_DIR="$RUN_DIR/llama.cpp-src"
BUILD_DIR="$RUN_DIR/llama.cpp-rpc"
BIN_DIR="$BUILD_DIR/bin"
PINNED_COMMIT="f49e9178767d557a522618b16ce8694f9ddac628"

rpc_server_bin="$BIN_DIR/rpc-server"
llama_server_bin="$BIN_DIR/llama-server"

die() { echo "[ERROR] $*" >&2; exit 1; }

platform_kind() {
  if [[ -f /etc/nv_tegra_release ]] || command -v nvpmodel >/dev/null 2>&1; then
    printf 'jetson'
  elif [[ -r /proc/device-tree/model ]] && tr -d '\000' </proc/device-tree/model | grep -qi 'raspberry pi'; then
    printf 'raspberry-pi'
  else
    printf 'unsupported'
  fi
}

check_runtime() {
  [[ -x "$rpc_server_bin" ]] || die "rpc-server missing: run prepare-rpc"
  [[ -x "$llama_server_bin" ]] || die "llama-server missing: run prepare-rpc"
  [[ -d "$SOURCE_DIR/.git" ]] || die "pinned llama.cpp source missing"
  actual="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
  [[ "$actual" == "$PINNED_COMMIT" ]] || die "runtime commit mismatch: $actual"
  "$rpc_server_bin" --help >/dev/null 2>&1
  server_help="$("$llama_server_bin" --help 2>&1)"
  [[ "$server_help" == *"--rpc SERVERS"* ]] || die "llama-server was built without RPC"
  echo "[OK] llama.cpp RPC commit=$actual platform=$(platform_kind)"
}

prepare_runtime() {
  kind="$(platform_kind)"
  [[ "$kind" != unsupported ]] || die "only Jetson and Raspberry Pi are supported"
  command -v git >/dev/null || die "git is required"
  command -v cmake >/dev/null || die "cmake is required"
  if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    mkdir -p "$SOURCE_DIR"
    git -C "$SOURCE_DIR" init
    git -C "$SOURCE_DIR" remote add origin https://github.com/ggml-org/llama.cpp.git
  fi
  git -C "$SOURCE_DIR" fetch --depth 1 origin "$PINNED_COMMIT"
  git -C "$SOURCE_DIR" checkout --detach "$PINNED_COMMIT"
  common=(
    -S "$SOURCE_DIR" -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release -DGGML_RPC=ON -DLLAMA_CURL=OFF
    -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
  )
  if [[ "$kind" == jetson ]]; then
    PATH="/usr/local/cuda/bin:$PATH" CUDACXX="/usr/local/cuda/bin/nvcc" \
      cmake "${common[@]}" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
    jobs="${RPC_BUILD_JOBS:-6}"
  else
    cmake "${common[@]}" -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
    jobs="${RPC_BUILD_JOBS:-4}"
  fi
  cmake --build "$BUILD_DIR" --config Release --target rpc-server llama-server -j "$jobs"
  check_runtime
}

stop_pid_file() {
  pid_file=$1
  expected=${2:-}
  if [[ ! -f "$pid_file" ]]; then return 0; fi
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    if [[ -n "$expected" && -r "/proc/$pid/cmdline" ]]; then
      command_line="$(tr '\000' ' ' <"/proc/$pid/cmdline")"
      if [[ "$command_line" != *"$expected"* ]]; then
        rm -f "$pid_file"
        die "refusing to stop unrelated process from stale PID file: $pid"
      fi
    fi
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pid_file"
}

start_worker() {
  port=$1
  host=${2:-0.0.0.0}
  check_runtime >/dev/null
  pid_file="$RUN_DIR/rpc_worker_${port}.pid"
  log_file="$RUN_DIR/rpc_worker_${port}.log"
  stop_pid_file "$pid_file" rpc-server
  device=CPU
  [[ "$(platform_kind)" == jetson ]] && device=CUDA0
  "$rpc_server_bin" --host "$host" --port "$port" --cache --device "$device" >"$log_file" 2>&1 &
  pid=$!
  echo "$pid" >"$pid_file"
  for _ in $(seq 1 100); do
    listen_sockets="$(ss -ltnH 2>/dev/null || true)"
    if [[ "$listen_sockets" == *":$port "* ]] && kill -0 "$pid" 2>/dev/null; then
      echo "[OK] RPC device started pid=$pid host=$host port=$port device=$device"
      exit 0
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  tail -n 80 "$log_file" >&2 || true
  stop_pid_file "$pid_file" rpc-server
  die "RPC device failed to start"
}

start_coordinator() {
  port=$1 model=$2 context=$3 gpu_layers=$4 endpoints=$5 split_mode=$6 split_csv=$7
  check_runtime >/dev/null
  pid_file="$RUN_DIR/rpc_coordinator_${port}.pid"
  log_file="$RUN_DIR/rpc_coordinator_${port}.log"
  stop_pid_file "$pid_file" llama-server
  command=(
    "$llama_server_bin" --host 127.0.0.1 --port "$port"
    --model "$model" --ctx-size "$context" --gpu-layers "$gpu_layers"
    --rpc "$endpoints" --split-mode "$split_mode" --metrics
    --parallel 1 --cont-batching --no-webui
  )
  if [[ "$split_csv" != - ]]; then command+=(--tensor-split "$split_csv"); fi
  "${command[@]}" >"$log_file" 2>&1 &
  pid=$!
  echo "$pid" >"$pid_file"
  for _ in $(seq 1 1800); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "[OK] RPC coordinator started pid=$pid port=$port"
      exit 0
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
  tail -n 120 "$log_file" >&2 || true
  stop_pid_file "$pid_file" llama-server
  die "RPC coordinator failed to load the model"
}

action="${1:-}"
case "$action" in
  prepare) prepare_runtime ;;
  check) check_runtime ;;
  start-worker) start_worker "${2:?port required}" "${3:-0.0.0.0}" ;;
  stop-worker) stop_pid_file "$RUN_DIR/rpc_worker_${2:?port required}.pid" rpc-server ;;
  start-coordinator) start_coordinator "${2:?}" "${3:?}" "${4:?}" "${5:?}" "${6:?}" "${7:?}" "${8:--}" ;;
  stop-coordinator) stop_pid_file "$RUN_DIR/rpc_coordinator_${2:?port required}.pid" llama-server ;;
  *) die "usage: runtime.sh prepare|check|start-worker|stop-worker|start-coordinator|stop-coordinator" ;;
esac
