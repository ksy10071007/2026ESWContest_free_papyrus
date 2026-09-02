#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PATH="${VENV_PATH:-$PROJECT_DIR/.venv}"
MEDIAPIPE_PT_PATH="${MEDIAPIPE_PT_PATH:-}"
CLASSIFIER_PTH_PATH="${CLASSIFIER_PTH_PATH:-}"
MEDIAPIPE_ONNX_PATH="${MEDIAPIPE_ONNX_PATH:-models/yolo.onnx}"
CLASSIFIER_ONNX_PATH="${CLASSIFIER_ONNX_PATH:-models/efficientnet.onnx}"
IMGSZ="${IMGSZ:-640}"
OPSET="${OPSET:-12}"
SKIP_INSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --yolo-pt)
      MEDIAPIPE_PT_PATH="$2"
      shift 2
      ;;
    --classifier-pth)
      CLASSIFIER_PTH_PATH="$2"
      shift 2
      ;;
    --yolo-onnx)
      MEDIAPIPE_ONNX_PATH="$2"
      shift 2
      ;;
    --classifier-onnx)
      CLASSIFIER_ONNX_PATH="$2"
      shift 2
      ;;
    --imgsz)
      IMGSZ="$2"
      shift 2
      ;;
    --opset)
      OPSET="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    -h|--help)
      cat <<'HELP'
Usage: bash scripts/export_onnx_rpi.sh [options]

Options:
  --venv <path>             Python venv path (default: .venv)
  --yolo-pt <path>          Source YOLO .pt file (default: newest models/*.pt)
  --classifier-pth <path>   Source classifier .pth file (default: models/Augmented_EffNet_V1_B0_best.pth or newest models/*.pth)
  --yolo-onnx <path>        Output YOLO ONNX path (default: models/yolo.onnx)
  --classifier-onnx <path>  Output classifier ONNX path (default: models/efficientnet.onnx)
  --imgsz <int>             YOLO export image size (default: 640)
  --opset <int>             ONNX opset (default: 12)
  --skip-install            Skip dependency installation
HELP
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$MEDIAPIPE_PT_PATH" ]]; then
  MEDIAPIPE_PT_PATH="$(ls -t models/*.pt 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$CLASSIFIER_PTH_PATH" ]]; then
  if [[ -f "models/Augmented_EffNet_V1_B0_best.pth" ]]; then
    CLASSIFIER_PTH_PATH="models/Augmented_EffNet_V1_B0_best.pth"
  else
    CLASSIFIER_PTH_PATH="$(ls -t models/*.pth 2>/dev/null | head -n 1 || true)"
  fi
fi

if [[ ! -d "$VENV_PATH" ]]; then
  echo "[ERROR] venv not found: $VENV_PATH"
  echo "        Create it first: python3 -m venv .venv"
  exit 1
fi

if [[ -z "$MEDIAPIPE_PT_PATH" || ! -f "$MEDIAPIPE_PT_PATH" ]]; then
  echo "[ERROR] YOLO PT file not found: $MEDIAPIPE_PT_PATH"
  exit 1
fi

if [[ -z "$CLASSIFIER_PTH_PATH" || ! -f "$CLASSIFIER_PTH_PATH" ]]; then
  echo "[ERROR] Classifier PTH file not found: $CLASSIFIER_PTH_PATH"
  exit 1
fi

mkdir -p "$(dirname "$MEDIAPIPE_ONNX_PATH")" "$(dirname "$CLASSIFIER_ONNX_PATH")"

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  echo "[INFO] Installing export dependencies in venv"
  pip install -q --upgrade pip setuptools wheel
  pip install -q torch torchvision ultralytics onnx onnxscript
fi

echo "[INFO] Exporting YOLO ONNX from: $MEDIAPIPE_PT_PATH"
yolo export \
  model="$MEDIAPIPE_PT_PATH" \
  format=onnx \
  imgsz="$IMGSZ" \
  opset="$OPSET" \
  dynamic=False \
  simplify=False \
  device=cpu

GENERATED_YOLO_ONNX="${MEDIAPIPE_PT_PATH%.pt}.onnx"
if [[ ! -f "$GENERATED_YOLO_ONNX" ]]; then
  echo "[ERROR] Expected generated file not found: $GENERATED_YOLO_ONNX"
  exit 1
fi

if [[ "$(readlink -f "$MEDIAPIPE_ONNX_PATH" 2>/dev/null || echo "$MEDIAPIPE_ONNX_PATH")" != "$(readlink -f "$GENERATED_YOLO_ONNX")" ]]; then
  cp -f "$GENERATED_YOLO_ONNX" "$MEDIAPIPE_ONNX_PATH"
else
  echo "[INFO] YOLO output already points to generated ONNX: $MEDIAPIPE_ONNX_PATH"
fi

echo "[INFO] Exporting classifier ONNX from: $CLASSIFIER_PTH_PATH"
python - "$CLASSIFIER_PTH_PATH" "$CLASSIFIER_ONNX_PATH" "$OPSET" <<'PY'
import sys
import torch
import torch.nn as nn
import torchvision.models as models

src = sys.argv[1]
out = sys.argv[2]
opset = int(sys.argv[3])

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
checkpoint = torch.load(src, map_location='cpu')

if isinstance(checkpoint, dict):
    state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict') or checkpoint
else:
    state_dict = checkpoint

cleaned = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
missing, unexpected = model.load_state_dict(cleaned, strict=False)
if missing:
    print(f"[WARN] Missing keys: {len(missing)}")
if unexpected:
    print(f"[WARN] Unexpected keys: {len(unexpected)}")

model.eval()
dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    out,
    export_params=True,
    opset_version=opset,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['logits'],
)
print(f"[OK] Classifier ONNX exported: {out}")
PY

echo "[INFO] Validating ONNX runtime load"
python - "$MEDIAPIPE_ONNX_PATH" "$CLASSIFIER_ONNX_PATH" <<'PY'
import sys
import onnxruntime as ort

for path in sys.argv[1:]:
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    in_names = [i.name for i in session.get_inputs()]
    out_names = [o.name for o in session.get_outputs()]
    print(f"[OK] {path} (inputs={in_names}, outputs={out_names})")
PY

echo "[DONE] ONNX export completed"
echo "       YOLO ONNX: $MEDIAPIPE_ONNX_PATH"
echo "       Classifier ONNX: $CLASSIFIER_ONNX_PATH"
