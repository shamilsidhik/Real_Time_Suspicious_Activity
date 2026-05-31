from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _default_device() -> str | int:
    configured = os.environ.get("YOLO_DEVICE", "").strip()
    if configured:
        return configured

    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


class ActivityPredictor:
    """
    YOLO-based activity predictor.

    Supports both:
      - YOLO classify models (results.probs)
      - YOLO detect models (results.boxes)
    """

    def __init__(self, model_path: str | None = None, conf: float = 0.25, imgsz: int = 640) -> None:
        self.model_path = model_path or os.environ.get(
            "ACTIVITY_MODEL_PATH",
            "ml/models/activity_yolov8/best.pt",
        )
        self.conf = conf
        self.imgsz = imgsz
        self.device = _default_device()

        self._lock = threading.Lock()
        self._model = None
        self.model_missing = True

        self._load()

    def _load(self) -> None:
        if not Path(self.model_path).exists():
            logger.warning("Activity model not found: %s", self.model_path)
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
            self.model_missing = False
            logger.info("Activity model loaded: %s", self.model_path)
        except Exception:
            logger.exception("Failed to load activity model from %s", self.model_path)

    def is_model_available(self) -> bool:
        return not self.model_missing

    def predict_sequence(self, frames) -> dict[str, Any]:
        """
        Backward-compatible adapter for code that still passes frame sequences.
        """
        if frames is None or len(frames) == 0:
            return {"status": "error", "label": "unknown", "confidence": 0.0}

        return self.predict_frame(frames[-1])

    def predict(self, frames) -> dict[str, Any]:
        return self.predict_sequence(frames)

    def predict_frame(self, frame) -> dict[str, Any]:
        if self.model_missing or self._model is None:
            return {"status": "unavailable", "label": "unknown", "confidence": 0.0}

        try:
            with self._lock:
                result = self._model.predict(
                    source=frame,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )[0]

            # Classification model path
            if getattr(result, "probs", None) is not None and result.probs is not None:
                class_id = int(result.probs.top1)
                confidence = float(result.probs.top1conf.item())
                return {
                    "status": "ok",
                    "label": _class_name(result.names, class_id),
                    "confidence": confidence,
                    "task": "classify",
                }

            # Detection model path: pick the highest-confidence detection
            if getattr(result, "boxes", None) is not None and result.boxes is not None and len(result.boxes) > 0:
                best_conf = -1.0
                best_label = "unknown"

                for box in result.boxes:
                    conf = float(box.conf[0].item())
                    class_id = int(box.cls[0].item())
                    if conf > best_conf:
                        best_conf = conf
                        best_label = _class_name(result.names, class_id)

                return {
                    "status": "ok",
                    "label": best_label,
                    "confidence": max(best_conf, 0.0),
                    "task": "detect",
                }

            # No activity box / no classification output: treat as normal if your
            # model is event-centric and only emits fight/violence boxes.
            return {
                "status": "no_detection",
                "label": "normal",
                "confidence": 0.0,
                "task": getattr(self._model, "task", "unknown"),
            }

        except Exception:
            logger.exception("Activity prediction failed")
            return {"status": "error", "label": "unknown", "confidence": 0.0}
