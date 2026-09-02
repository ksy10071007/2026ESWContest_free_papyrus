from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort

from inference.base import InferenceBackend


class _FakeScalar(float):
    def item(self) -> float:
        return float(self)


class _FakeConfTensor:
    def __init__(self, values: List[float]):
        self._values = np.array(values, dtype=np.float32)

    def numel(self) -> int:
        return int(self._values.size)

    def max(self) -> _FakeScalar:
        if self._values.size == 0:
            return _FakeScalar(0.0)
        return _FakeScalar(float(np.max(self._values)))


class _FakeBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, conf: float):
        self.xyxy = [np.array([x1, y1, x2, y2], dtype=np.float32)]
        self.conf = _FakeScalar(conf)


class _FakeBoxes(list):
    @property
    def conf(self) -> _FakeConfTensor:
        return _FakeConfTensor([float(box.conf) for box in self])


class _FakeDetections:
    def __init__(self, boxes: Optional[List[_FakeBox]] = None):
        self.boxes = _FakeBoxes(boxes or [])


class _NoopLogger:
    def log_result(self, _payload: Dict[str, Any]) -> None:
        return None


class _RPiDetectorAdapter:
    """Legacy detector interface for existing eye_server pipeline."""

    def detect(self, image, conf_threshold=None):
        # Keep status polling conservative (no forced positive detection).
        _ = image
        _ = conf_threshold
        return _FakeDetections([])

    def crop_eyes(self, image, detections):
        # ONNX detector parsing is export-dependent.
        # For compatibility, use full frame as a single eye ROI fallback.
        _ = detections
        if image is None or getattr(image, "size", 0) == 0:
            return []
        h, w = image.shape[:2]
        return [{"image": image, "bbox": (0, 0, w, h), "confidence": 0.99}]


class _RPiClassifierAdapter:
    """Legacy classifier interface for existing eye_server pipeline."""

    def __init__(self, backend: "RPiONNXBackend"):
        self.backend = backend

    def classify_with_details(self, image, generate_cam=True):
        _ = generate_cam
        logits, probs = self.backend._run_classifier(image)
        cls_idx = int(np.argmax(probs))
        conf = float(probs[cls_idx])
        disease = self.backend.class_names[cls_idx] if cls_idx < len(self.backend.class_names) else str(cls_idx)

        return {
            "class": cls_idx,
            "disease": disease,
            "confidence": conf,
            "probabilities": probs.tolist(),
            "heatmap_image": None,
            "logits": logits.tolist(),
        }

    def classify(self, image, generate_cam=True):
        details = self.classify_with_details(image, generate_cam=generate_cam)
        return details["disease"], details["confidence"], details["heatmap_image"]


class _RPiAnalyzerAdapter:
    """Reuse 기존 EyeAnalyzer 로직 (cv2/numpy 기반)"""

    def __init__(self):
        from modules.analyzer import EyeAnalyzer

        self._analyzer = EyeAnalyzer()

    def analyze(self, image):
        return self._analyzer.analyze(image)


class RPiONNXBackend(InferenceBackend):
    """ONNX Runtime backend for Raspberry Pi (CPU only)."""

    def __init__(
        self,
        yolo_onnx_path: str,
        classifier_onnx_path: str,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.yolo_onnx_path = yolo_onnx_path
        self.classifier_onnx_path = classifier_onnx_path
        self.class_names = class_names or ["normal", "conjunctivitis", "uveitis", "cataract", "stye"]

        if not os.path.exists(self.yolo_onnx_path):
            raise FileNotFoundError(f"MediaPipe/YOLO ONNX model not found: {self.yolo_onnx_path}")
        if not os.path.exists(self.classifier_onnx_path):
            raise FileNotFoundError(f"Classifier ONNX model not found: {self.classifier_onnx_path}")

        providers = ["CPUExecutionProvider"]
        self.yolo_session = ort.InferenceSession(self.yolo_onnx_path, providers=providers)
        self.classifier_session = ort.InferenceSession(self.classifier_onnx_path, providers=providers)

        self.classifier_input_name = self.classifier_session.get_inputs()[0].name

        # Legacy-compatible objects used by existing eye_server.py
        self.detector = _RPiDetectorAdapter()
        self.classifier = _RPiClassifierAdapter(self)
        self.analyzer = _RPiAnalyzerAdapter()
        self.logger = _NoopLogger()

    def _preprocess_classifier(self, image_bgr: np.ndarray, size: int = 224) -> np.ndarray:
        resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std

        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def _run_classifier(self, image_bgr: Any):
        inp = self._preprocess_classifier(image_bgr)
        logits = self.classifier_session.run(None, {self.classifier_input_name: inp})[0]
        probs = self._softmax(logits[0])
        return logits[0], probs

    def predict_eye_disease(self, image_bgr: Any) -> Dict[str, Any]:
        # YOLO session is loaded for parity with Jetson pipeline.
        # Output parsing differs by export format, so classification path is kept deterministic here.
        _ = self.yolo_session

        _logits, probs = self._run_classifier(image_bgr)
        cls_idx = int(np.argmax(probs))
        conf = float(probs[cls_idx])
        disease = self.class_names[cls_idx] if cls_idx < len(self.class_names) else str(cls_idx)

        return {
            "status": "success",
            "device": "rpi",
            "disease": disease,
            "confidence": conf,
            "redness": None,
            "eyes_detected": None,
            "bboxes": [],
            "heatmap_available": False,
        }
