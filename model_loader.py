from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv


load_dotenv()


SUPPORTED_DEVICES = {"jetson", "rpi"}


@dataclass
class LoaderConfig:
    device: str
    yolo_onnx_path: str
    classifier_onnx_path: str


def resolve_device(explicit_device: Optional[str] = None) -> str:
    device = (explicit_device or os.getenv("MODEL_DEVICE") or "jetson").strip().lower()
    if device not in SUPPORTED_DEVICES:
        raise ValueError(f"Unsupported MODEL_DEVICE '{device}'. Use one of: {sorted(SUPPORTED_DEVICES)}")
    return device


def build_loader_config(explicit_device: Optional[str] = None) -> LoaderConfig:
    device = resolve_device(explicit_device)
    return LoaderConfig(
        device=device,
        yolo_onnx_path=os.getenv("MEDIAPIPE_ONNX_PATH", "models/yolo.onnx"),
        classifier_onnx_path=os.getenv("CLASSIFIER_ONNX_PATH", "models/efficientnet.onnx"),
    )


class ModelLoader:
    """Factory facade exposing a hardware-agnostic predict function."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.config = build_loader_config(device)
        self.device = self.config.device
        self.backend = self._create_backend(self.config)

    def _create_backend(self, cfg: LoaderConfig):
        if cfg.device == "jetson":
            # Import torch only in Jetson path.
            import torch  # noqa: F401
            from inference.jetson_backend import JetsonBackend

            return JetsonBackend()

        # cfg.device == "rpi"
        import onnxruntime  # noqa: F401
        from inference.rpi_backend import RPiONNXBackend

        return RPiONNXBackend(
            yolo_onnx_path=cfg.yolo_onnx_path,
            classifier_onnx_path=cfg.classifier_onnx_path,
        )

    def predict_eye_disease(self, image_bgr: Any) -> Dict[str, Any]:
        return self.backend.predict_eye_disease(image_bgr)


# -------- Optional compatibility layer for existing eye_server.py --------
class _LegacyManagerProxy:
    def __init__(self, loader: ModelLoader):
        self._loader = loader

    def get_detector(self):
        return getattr(self._loader.backend, "detector", None)

    def get_classifier(self):
        return getattr(self._loader.backend, "classifier", None)

    def get_analyzer(self):
        return getattr(self._loader.backend, "analyzer", None)

    def get_logger(self):
        return getattr(self._loader.backend, "logger", None)


_model_loader_singleton: Optional[ModelLoader] = None
_legacy_manager_singleton: Optional[_LegacyManagerProxy] = None


def initialize_model_loader(device: Optional[str] = None) -> ModelLoader:
    global _model_loader_singleton
    if _model_loader_singleton is None:
        _model_loader_singleton = ModelLoader(device=device)
    return _model_loader_singleton


def initialize_models(device: Optional[str] = None):
    global _model_loader_singleton, _legacy_manager_singleton
    if _model_loader_singleton is None:
        _model_loader_singleton = initialize_model_loader(device=device)
        _legacy_manager_singleton = _LegacyManagerProxy(_model_loader_singleton)
    return _legacy_manager_singleton


def get_models():
    global _legacy_manager_singleton
    if _legacy_manager_singleton is None:
        initialize_models()
    return _legacy_manager_singleton


def predict_eye_disease(image_bgr: Any, device: Optional[str] = None) -> Dict[str, Any]:
    """Unified top-level inference API for server code."""
    loader = initialize_model_loader(device=device)
    return loader.predict_eye_disease(image_bgr)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ModelLoader factory")
    parser.add_argument("--device", choices=sorted(SUPPORTED_DEVICES), default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    loader = ModelLoader(device=args.device)
    print(
        {
            "status": "ok",
            "device": loader.device,
            "backend": loader.backend.__class__.__name__,
        }
    )
