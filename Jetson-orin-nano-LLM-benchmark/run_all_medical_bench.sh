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

MODEL_CSV_PATH="${MODEL_CSV_PATH:-$PROJECT_ROOT/models.example.csv}"
LIST_MODELS_ONLY="${LIST_MODELS_ONLY:-0}"
SKIP_UNLOADABLE_MODELS="${SKIP_UNLOADABLE_MODELS:-1}"

N_CTX="${N_CTX:-1024}"
N_THREADS="${N_THREADS:-6}"
MAX_TOKENS="${MAX_TOKENS:-128}"
PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-}"
PROBE_LAYERS="${PROBE_LAYERS:-35,32,30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0}"
PROBE_CTXS="${PROBE_CTXS:-}"
GPU_LAYER_MARGIN="${GPU_LAYER_MARGIN:-2}"
SEED="${SEED:-42}"

MEDICAL_OUTPUT_DIR="${MEDICAL_OUTPUT_DIR:-outputs/medical_benchmark}"
MEDICAL_SUMMARY_CSV="${MEDICAL_SUMMARY_CSV:-$MEDICAL_OUTPUT_DIR/medical_compare_summary.csv}"
MEDICAL_DETAILS_JSON="${MEDICAL_DETAILS_JSON:-$MEDICAL_OUTPUT_DIR/medical_compare_details.json}"
MEDICAL_RANKED_CSV="${MEDICAL_RANKED_CSV:-$MEDICAL_OUTPUT_DIR/medical_compare_ranked.csv}"
MEDICAL_LIMIT="${MEDICAL_LIMIT:-0}"

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

TMP_CSV="$(mktemp "${TMPDIR:-/tmp}/medical_models_autotuned.XXXXXX.csv")"
mkdir -p "$MEDICAL_OUTPUT_DIR" "$(dirname "$MEDICAL_SUMMARY_CSV")" "$(dirname "$MEDICAL_DETAILS_JSON")" "$(dirname "$MEDICAL_RANKED_CSV")"

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

echo "[INFO] Running compare_medical_models.py"
python "$PROJECT_ROOT/bench/compare_medical_models.py" \
  --models-file "$TMP_CSV" \
  --limit "$MEDICAL_LIMIT" \
  --output-dir "$MEDICAL_OUTPUT_DIR" \
  --output-csv "$MEDICAL_SUMMARY_CSV" \
  --details-json "$MEDICAL_DETAILS_JSON"

echo "[INFO] Building ranked summary"
python - "$MEDICAL_SUMMARY_CSV" "$MEDICAL_RANKED_CSV" <<'PY'
import csv
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

rows = []
with open(input_path, 'r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        row['accuracy'] = float(row.get('accuracy') or 0.0)
        row['correct_answers'] = int(row.get('correct_answers') or 0)
        row['total_questions'] = int(row.get('total_questions') or 0)
        rows.append(row)

rows.sort(key=lambda item: (-item['accuracy'], -item['correct_answers'], item['model_name']))

with open(output_path, 'w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['rank', 'model_name', 'model_path', 'total_questions', 'correct_answers', 'accuracy'])
    writer.writeheader()
    for index, row in enumerate(rows, start=1):
        writer.writerow(
            {
                'rank': index,
                'model_name': row['model_name'],
                'model_path': row['model_path'],
                'total_questions': row['total_questions'],
                'correct_answers': row['correct_answers'],
                'accuracy': f"{row['accuracy']:.2f}",
            }
        )
PY

echo "[DONE] Medical run-all completed"
echo "[DONE] Summary CSV : $MEDICAL_SUMMARY_CSV"
echo "[DONE] Details JSON : $MEDICAL_DETAILS_JSON"
echo "[DONE] Ranked CSV   : $MEDICAL_RANKED_CSV"
