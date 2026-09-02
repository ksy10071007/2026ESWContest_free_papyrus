#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/.venv" && -d "$SCRIPT_DIR/bench" ]]; then
  PROJECT_ROOT="$SCRIPT_DIR"
else
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"

if [[ ! -d ".venv" ]]; then
  echo "[ERROR] .venv not found. Create it first in $PROJECT_ROOT"
  exit 1
fi

source .venv/bin/activate

AUTO_MAX_POWER="${AUTO_MAX_POWER:-1}"
RESTORE_POWER_ON_EXIT="${RESTORE_POWER_ON_EXIT:-1}"
MAX_POWER_MODE="${MAX_POWER_MODE:-auto}"

NUM_PROMPTS="${NUM_PROMPTS:-1}"
MAX_TOKENS="${MAX_TOKENS:-128}"
# If not set, probe uses each model's effective max_tokens.
PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-}"
N_CTX="${N_CTX:-1024}"
N_THREADS="${N_THREADS:-6}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
SEED="${SEED:-42}"
PROBE_LAYERS="${PROBE_LAYERS:-35,32,30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0}"
PROBE_CTXS="${PROBE_CTXS:-}"
GPU_LAYER_MARGIN="${GPU_LAYER_MARGIN:-2}"
MODEL_CSV_PATH="${MODEL_CSV_PATH:-$PROJECT_ROOT/models.example.csv}"
LIST_MODELS_ONLY="${LIST_MODELS_ONLY:-0}"
SKIP_UNLOADABLE_MODELS="${SKIP_UNLOADABLE_MODELS:-1}"

GENERAL_OUTPUT_DIR="${GENERAL_OUTPUT_DIR:-outputs/general_benchmark}"
OUTPUT_CSV="${OUTPUT_CSV:-$GENERAL_OUTPUT_DIR/comparison_results.csv}"
RANKED_CSV="${RANKED_CSV:-$GENERAL_OUTPUT_DIR/comparison_ranked.csv}"
PLOT_PNG="${PLOT_PNG:-$GENERAL_OUTPUT_DIR/comparison_results.png}"
PLOT_SORT_BY="${PLOT_SORT_BY:-avg_tps}"

RANK_W_TPS="${RANK_W_TPS:-0.50}"
RANK_W_RSS="${RANK_W_RSS:-0.30}"
RANK_W_TTFT="${RANK_W_TTFT:-0.20}"
RANK_W_GPU_UTIL="${RANK_W_GPU_UTIL:-0.00}"
RANK_W_POWER="${RANK_W_POWER:-0.00}"
RANK_W_JETSON_RAM="${RANK_W_JETSON_RAM:-0.00}"
RANK_W_GPU_TEMP="${RANK_W_GPU_TEMP:-0.00}"
RANK_W_CPU_TEMP="${RANK_W_CPU_TEMP:-0.00}"

TMP_CSV=""
PREV_NVP_MODE_ID=""
POWER_MODE_CHANGED=0
JETSON_CLOCKS_APPLIED=0
JETSON_CLOCKS_STORED=0
JETSON_CLOCKS_STATE_FILE="$PROJECT_ROOT/.run/jetson_clocks.conf"

warn() {
  echo "[WARN] $*" >&2
}

run_privileged() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

get_current_nvp_mode_id() {
  nvpmodel -q 2>/dev/null | awk 'match($0, /^[0-9]+$/) { mode=$0 } END { if (mode != "") print mode }'
}

detect_max_power_mode_id() {
  if [[ "$MAX_POWER_MODE" != "auto" ]]; then
    echo "$MAX_POWER_MODE"
    return 0
  fi

  local maxn_id
  maxn_id="$(nvpmodel -p --verbose 2>/dev/null | awk '
    /POWER_MODEL: ID=/ {
      id=""; name="";
      if (match($0, /ID=[0-9]+/)) {
        id = substr($0, RSTART + 3, RLENGTH - 3)
      }
      if (match($0, /NAME=[^ ]+/)) {
        name = substr($0, RSTART + 5, RLENGTH - 5)
      }
      if (name ~ /MAXN/) {
        print id
        exit
      }
    }
  ')"
  if [[ -n "$maxn_id" ]]; then
    echo "$maxn_id"
    return 0
  fi

  local best_watt_id
  best_watt_id="$(nvpmodel -p --verbose 2>/dev/null | awk '
    BEGIN { best_watt=-1; best_id="" }
    /POWER_MODEL: ID=/ {
      id=""; name=""; watt=0;
      if (match($0, /ID=[0-9]+/)) {
        id = substr($0, RSTART + 3, RLENGTH - 3)
      }
      if (match($0, /NAME=[^ ]+/)) {
        name = substr($0, RSTART + 5, RLENGTH - 5)
      }
      if (match(name, /[0-9]+W/)) {
        watt = substr(name, RSTART, RLENGTH - 1) + 0
      }
      if (watt > best_watt) {
        best_watt = watt
        best_id = id
      }
    }
    END {
      if (best_id != "") {
        print best_id
      }
    }
  ')"
  if [[ -n "$best_watt_id" ]]; then
    echo "$best_watt_id"
    return 0
  fi

  echo "0"
}

apply_max_power_mode() {
  if [[ "$AUTO_MAX_POWER" != "1" ]]; then
    echo "[INFO] AUTO_MAX_POWER=0, skipping nvpmodel/jetson_clocks"
    return 0
  fi

  if ! command -v nvpmodel >/dev/null 2>&1; then
    warn "nvpmodel not found; skipping max power setup"
    return 0
  fi

  PREV_NVP_MODE_ID="$(get_current_nvp_mode_id || true)"
  local target_mode
  target_mode="$(detect_max_power_mode_id)"

  echo "[INFO] Applying max power mode: nvpmodel -m $target_mode"
  if run_privileged nvpmodel -m "$target_mode"; then
    POWER_MODE_CHANGED=1
  else
    warn "failed to apply nvpmodel mode $target_mode"
    return 0
  fi

  if command -v jetson_clocks >/dev/null 2>&1; then
    echo "[INFO] Applying jetson_clocks"
    mkdir -p "$PROJECT_ROOT/.run"
    if run_privileged jetson_clocks --store "$JETSON_CLOCKS_STATE_FILE" >/dev/null 2>&1; then
      JETSON_CLOCKS_STORED=1
    else
      warn "failed to store current jetson_clocks state; restore may be skipped"
    fi

    if run_privileged jetson_clocks; then
      JETSON_CLOCKS_APPLIED=1
    else
      warn "failed to apply jetson_clocks"
    fi
  else
    warn "jetson_clocks not found; skipping clock lock"
  fi
}

restore_power_mode() {
  if [[ "$RESTORE_POWER_ON_EXIT" != "1" ]]; then
    return 0
  fi

  if (( JETSON_CLOCKS_APPLIED == 1 )) && command -v jetson_clocks >/dev/null 2>&1; then
    if (( JETSON_CLOCKS_STORED == 1 )) && [[ -f "$JETSON_CLOCKS_STATE_FILE" ]]; then
      echo "[INFO] Restoring jetson_clocks"
      if ! run_privileged jetson_clocks --restore "$JETSON_CLOCKS_STATE_FILE"; then
        warn "failed to restore jetson_clocks"
      fi
    else
      echo "[INFO] Skipping jetson_clocks restore (no stored state file)"
    fi
  fi

  if (( POWER_MODE_CHANGED == 1 )) && [[ -n "$PREV_NVP_MODE_ID" ]] && command -v nvpmodel >/dev/null 2>&1; then
    echo "[INFO] Restoring previous nvpmodel mode: $PREV_NVP_MODE_ID"
    if ! run_privileged nvpmodel -m "$PREV_NVP_MODE_ID"; then
      warn "failed to restore nvpmodel mode $PREV_NVP_MODE_ID"
    fi
  fi
}

cleanup() {
  if [[ -n "$TMP_CSV" && -f "$TMP_CSV" ]]; then
    rm -f "$TMP_CSV"
  fi
  restore_power_mode
}

trap cleanup EXIT

csv_lookup_field() {
  local model_path="$1"
  local column_name="$2"

  if [[ ! -f "$MODEL_CSV_PATH" ]]; then
    return 1
  fi

  awk -F',' -v p="$model_path" -v c="$column_name" '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        if ($i == c) {
          col = i
        }
      }
      next
    }

    $2 == p {
      if (col > 0 && col <= NF) {
        print $col
        found = 1
        exit
      }
    }

    END {
      if (!found) {
        exit 1
      }
    }
  ' "$MODEL_CSV_PATH"
}

infer_model_name() {
  local model_path="$1"
  local csv_name=""
  local rel_path
  local inferred_name

  if csv_name="$(csv_lookup_field "$model_path" "name" 2>/dev/null)"; then
    if [[ -n "$csv_name" ]]; then
      echo "$csv_name"
      return 0
    fi
  fi

  rel_path="${model_path#$PROJECT_ROOT/models/}"
  inferred_name="$(echo "$rel_path" | tr '[:upper:]' '[:lower:]' | sed -E 's/\.gguf$//; s/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
  echo "${inferred_name:-model}"
}

infer_model_max_tokens() {
  local model_path="$1"
  local csv_max_tokens=""
  local filename

  if csv_max_tokens="$(csv_lookup_field "$model_path" "max_tokens" 2>/dev/null)"; then
    if [[ "$csv_max_tokens" =~ ^[0-9]+$ ]] && (( csv_max_tokens > 0 )); then
      echo "$csv_max_tokens"
      return 0
    fi
  fi

  filename="$(basename "$model_path")"
  if [[ "$filename" == *"DeepSeek-R1-Distill"* ]]; then
    echo "256"
    return 0
  fi

  echo "$MAX_TOKENS"
}

MODEL_SPECS=()
while IFS= read -r model_path; do
  [[ -z "$model_path" ]] && continue
  model_name="$(infer_model_name "$model_path")"
  model_max_tokens="$(infer_model_max_tokens "$model_path")"
  MODEL_SPECS+=("$model_name|$model_path|$model_max_tokens")
done < <(find "$PROJECT_ROOT/models" -type f -name "*.gguf" | sort)

if [[ ${#MODEL_SPECS[@]} -eq 0 ]]; then
  echo "[ERROR] No .gguf models found under $PROJECT_ROOT/models"
  exit 1
fi

echo "[INFO] Discovered ${#MODEL_SPECS[@]} model(s) under $PROJECT_ROOT/models"
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r name path model_max_tokens <<< "$spec"
  echo "[INFO] - $name | max_tokens=$model_max_tokens | $path"
done

if [[ "$LIST_MODELS_ONLY" == "1" ]]; then
  echo "[INFO] LIST_MODELS_ONLY=1, exiting before benchmark run"
  exit 0
fi

apply_max_power_mode

build_ctx_schedule() {
  local seen=""
  local raw="${PROBE_CTXS// /}"

  if [[ -n "$raw" ]]; then
    IFS=',' read -r -a ctx_candidates <<< "$raw"
  else
    ctx_candidates=("$N_CTX" 768 512 384 256 192 128)
  fi

  for candidate in "${ctx_candidates[@]}"; do
    if [[ ! "$candidate" =~ ^[0-9]+$ ]]; then
      continue
    fi
    if (( candidate < 128 || candidate > N_CTX )); then
      continue
    fi
    if [[ ",$seen," == *",$candidate,"* ]]; then
      continue
    fi
    seen+="${seen:+,}$candidate"
    echo "$candidate"
  done
}

find_safe_config() {
  local model_path="$1"
  local probe_max_tokens="$2"
  local candidate_ctx
  local adjusted
  IFS=',' read -r -a layer_candidates <<< "$PROBE_LAYERS"

  while IFS= read -r candidate_ctx; do
    for n_gpu_layers in "${layer_candidates[@]}"; do
      if python "$PROJECT_ROOT/bench/benchmark.py" \
        --model "$model_path" \
        --model-name probe \
        --num-prompts 1 \
        --max-tokens "$probe_max_tokens" \
        --n-ctx "$candidate_ctx" \
        --n-gpu-layers "$n_gpu_layers" \
        --n-threads "$N_THREADS" \
        --temperature 0.0 \
        --top-p 1.0 \
        --seed "$SEED" \
        --warmup \
        >/dev/null 2>&1
      then
        adjusted=$(( n_gpu_layers - GPU_LAYER_MARGIN ))
        if (( adjusted < 0 )); then
          adjusted=0
        fi
        echo "$adjusted,$candidate_ctx"
        return 0
      fi
    done
  done < <(build_ctx_schedule)

  echo "-1,-1"
}

TMP_CSV="$(mktemp "${TMPDIR:-/tmp}/models_autotuned.XXXXXX.csv")"
mkdir -p "$GENERAL_OUTPUT_DIR" "$(dirname "$OUTPUT_CSV")" "$(dirname "$RANKED_CSV")" "$(dirname "$PLOT_PNG")"

echo "name,path,n_gpu_layers,n_ctx,max_tokens" > "$TMP_CSV"

echo "[INFO] Probing safe n_gpu_layers for each model..."
selected_count=0
skipped_count=0
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r name path model_max_tokens <<< "$spec"
  probe_tokens="${PROBE_MAX_TOKENS:-$model_max_tokens}"
  safe_config="$(find_safe_config "$path" "$probe_tokens")"
  IFS=',' read -r safe_layers safe_ctx <<< "$safe_config"

  if [[ "$safe_layers" == "-1" ]]; then
    if [[ "$SKIP_UNLOADABLE_MODELS" == "1" ]]; then
      warn "Skipping unloadable model: $name"
      skipped_count=$(( skipped_count + 1 ))
      continue
    fi
    echo "[ERROR] Could not load model even with n_gpu_layers=0 and reduced n_ctx: $name"
    exit 1
  fi

  if (( safe_ctx != N_CTX )); then
    echo "[INFO] $name -> n_gpu_layers=$safe_layers, n_ctx=$safe_ctx (requested $N_CTX), max_tokens=$model_max_tokens (probe=$probe_tokens)"
  else
    echo "[INFO] $name -> n_gpu_layers=$safe_layers, max_tokens=$model_max_tokens (probe=$probe_tokens)"
  fi

  echo "$name,$path,$safe_layers,$safe_ctx,$model_max_tokens" >> "$TMP_CSV"
  selected_count=$(( selected_count + 1 ))
done

if (( selected_count == 0 )); then
  echo "[ERROR] No models were loadable after probing."
  exit 1
fi

if (( skipped_count > 0 )); then
  warn "Skipped $skipped_count unloadable model(s); continuing with $selected_count model(s)"
fi

echo "[INFO] Running compare_models.py"
python "$PROJECT_ROOT/bench/compare_models.py" \
  --models-file "$TMP_CSV" \
  --output-dir "$GENERAL_OUTPUT_DIR" \
  --num-prompts "$NUM_PROMPTS" \
  --max-tokens "$MAX_TOKENS" \
  --n-ctx "$N_CTX" \
  --n-threads "$N_THREADS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --seed "$SEED" \
  --warmup \
  --output-csv "$OUTPUT_CSV"

echo "[INFO] Running plot_results.py"
if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("matplotlib") else 1)
PY
then
  python "$PROJECT_ROOT/bench/plot_results.py" \
    --output-dir "$GENERAL_OUTPUT_DIR" \
    --sort-by "$PLOT_SORT_BY"
else
  echo "[WARN] matplotlib is not installed; skipping PNG plot generation"
fi

echo "[INFO] Running rank_models.py"
if python - "$OUTPUT_CSV" <<'PY'
import csv
import sys

ok_count = 0
with open(sys.argv[1], "r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        if row.get("status") == "ok":
            ok_count += 1

raise SystemExit(0 if ok_count > 0 else 1)
PY
then
  python "$PROJECT_ROOT/bench/rank_models.py" \
    --output-dir "$GENERAL_OUTPUT_DIR" \
    --w-tps "$RANK_W_TPS" \
    --w-rss "$RANK_W_RSS" \
    --w-ttft "$RANK_W_TTFT" \
    --w-gpu-util "$RANK_W_GPU_UTIL" \
    --w-power "$RANK_W_POWER" \
    --w-jetson-ram "$RANK_W_JETSON_RAM" \
    --w-gpu-temp "$RANK_W_GPU_TEMP" \
    --w-cpu-temp "$RANK_W_CPU_TEMP"
else
  echo "[WARN] No successful rows in $OUTPUT_CSV; skipping ranking"
fi

echo "[DONE] All steps completed"
echo "[DONE] Output dir   : $GENERAL_OUTPUT_DIR"
echo "[DONE] Results CSV  : $OUTPUT_CSV"
if [[ -f "$RANKED_CSV" ]]; then
  echo "[DONE] Ranked CSV   : $RANKED_CSV"
else
  echo "[DONE] Ranked CSV   : skipped"
fi
if [[ -f "$PLOT_PNG" ]]; then
  echo "[DONE] Plot PNG     : $PLOT_PNG"
else
  echo "[DONE] Plot PNG     : skipped"
fi
