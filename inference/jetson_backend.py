from __future__ import annotations

from typing import Any, Dict, List

import config as config
from utils.logger import ResultLogger
from modules import EyeDetector, DiseaseClassifier, EyeAnalyzer
from inference.base import InferenceBackend


class JetsonBackend(InferenceBackend):
    """PyTorch-based backend for Jetson devices."""

    def __init__(self) -> None:
        # Lazy import by environment is handled by factory (model_loader.py).
        # [LEGACY YOLO] self.detector = EyeDetector(config.YOLO_MODEL_PATH)
        # MediaPipe Face Mesh initialization (no model_path needed)
        self.detector = EyeDetector()
        self.classifier = DiseaseClassifier(config.CLASSIFIER_MODEL_PATH)
        self.analyzer = EyeAnalyzer()
        self.logger = ResultLogger(config.LOG_FORMAT)

    def _pick_best_crop(self, image_bgr: Any, eye_crops: List[Dict[str, Any]]) -> Any:
        if not eye_crops:
            return image_bgr
        eye_crops = sorted(eye_crops, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        return eye_crops[0]["image"]

    def predict_eye_disease(self, image_bgr: Any) -> Dict[str, Any]:
        detections = self.detector.detect(image_bgr)
        eye_crops = self.detector.crop_eyes(image_bgr, detections)

        best_crop = self._pick_best_crop(image_bgr, eye_crops)
        disease, confidence, heatmap = self.classifier.classify(best_crop, generate_cam=True)
        analysis = self.analyzer.analyze(best_crop)

        bboxes = [list(c.get("bbox", [])) for c in eye_crops]

        return {
            "status": "success",
            "device": "jetson",
            "disease": disease,
            "confidence": float(confidence),
            "redness": float(analysis.get("redness", 0.0)),
            "eyes_detected": len(eye_crops),
            "bboxes": bboxes,
            "heatmap_available": bool(heatmap is not None),
        }
