from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class InferenceBackend(ABC):
    """Common interface for hardware-specific inference backends."""

    @abstractmethod
    def predict_eye_disease(self, image_bgr: Any) -> Dict[str, Any]:
        """Run eye disease prediction and return a unified payload."""
        raise NotImplementedError
