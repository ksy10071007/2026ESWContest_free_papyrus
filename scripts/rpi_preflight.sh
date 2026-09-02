#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "[INFO] Project dir: $PROJECT_DIR"

echo "[CHECK] Python"
python3 -V

echo "[CHECK] Required Python packages"
python3 - <<'PY'
mods = [
    "flask",
    "numpy",
    "cv2",
    "onnxruntime",
    "PIL",
]
missing = []
for m in mods:
    try:
        __import__(m)
        print(f"  - OK: {m}")
    except Exception:
        missing.append(m)
        print(f"  - MISSING: {m}")
if missing:
    raise SystemExit(f"Missing modules: {', '.join(missing)}")
PY

MODEL_DEVICE="${MODEL_DEVICE:-rpi}"
MEDIAPIPE_ONNX_PATH="${MEDIAPIPE_ONNX_PATH:-models/yolo.onnx}"
CLASSIFIER_ONNX_PATH="${CLASSIFIER_ONNX_PATH:-models/efficientnet.onnx}"

echo "[CHECK] Env"
echo "  - MODEL_DEVICE=$MODEL_DEVICE"
echo "  - MEDIAPIPE_ONNX_PATH=$MEDIAPIPE_ONNX_PATH"
echo "  - CLASSIFIER_ONNX_PATH=$CLASSIFIER_ONNX_PATH"

if [[ "$MODEL_DEVICE" != "rpi" ]]; then
  echo "[WARN] MODEL_DEVICE is not 'rpi' (current: $MODEL_DEVICE)"
fi

echo "[CHECK] Model files"
[[ -f "$MEDIAPIPE_ONNX_PATH" ]] && echo "  - OK: $MEDIAPIPE_ONNX_PATH" || { echo "  - MISSING: $MEDIAPIPE_ONNX_PATH"; exit 1; }
[[ -f "$CLASSIFIER_ONNX_PATH" ]] && echo "  - OK: $CLASSIFIER_ONNX_PATH" || { echo "  - MISSING: $CLASSIFIER_ONNX_PATH"; exit 1; }

echo "[DONE] RPi preflight checks passed"
