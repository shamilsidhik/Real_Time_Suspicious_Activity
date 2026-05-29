"""Activity prediction with a YOLOv8-first interface.

The live webcam path uses the checked-in YOLOv8 activity detector when
``ml/models/activity_yolov8/best.pt`` exists. The older MobileNet/LSTM model
remains a documented fallback for offline sequence callers, but the public
return value is always normalized.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

ML_DIR = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_PATH = ML_DIR / "models" / "activity_yolov8" / "best.pt"
DEFAULT_LSTM_PATH = ML_DIR / "models" / "activity_mobilenet_lstm" / "activity_model.keras"
DEFAULT_CLASS_NAMES = ML_DIR / "models" / "activity_mobilenet_lstm" / "class_names.json"


class ActivityPredictor:
    def __init__(self, model_path: str | None = None):
        env_path = os.environ.get("ACTIVITY_MODEL_PATH")
        self.model_path = Path(model_path or env_path or DEFAULT_YOLO_PATH)
        self.backend = "unavailable"
        self._model = None
        self._class_names: List[str] = []
        self.last_error = ""
        self._load()

    def _load(self) -> None:
        if self.model_path.exists() and self.model_path.suffix.lower() == ".pt":
            self._load_yolo(self.model_path)
            return

        if DEFAULT_YOLO_PATH.exists():
            self.model_path = DEFAULT_YOLO_PATH
            self._load_yolo(DEFAULT_YOLO_PATH)
            return

        if DEFAULT_LSTM_PATH.exists():
            self.model_path = DEFAULT_LSTM_PATH
            self._load_lstm(DEFAULT_LSTM_PATH)
            return

        self.last_error = f"No activity model found at {self.model_path}"
        logger.warning(self.last_error)

    def _load_yolo(self, path: Path) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(path))
            self.backend = "yolov8"
            self.last_error = ""
            logger.info("Activity YOLOv8 model loaded from %s", path)
        except Exception as exc:
            self.backend = "unavailable"
            self.last_error = str(exc)
            logger.exception("Activity YOLOv8 model load failed: %s", exc)

    def _load_lstm(self, path: Path) -> None:
        try:
            from tensorflow.keras.models import load_model

            self._model = load_model(str(path))
            self.backend = "mobilenet_lstm_fallback"
            if DEFAULT_CLASS_NAMES.exists():
                with open(DEFAULT_CLASS_NAMES, "r", encoding="utf-8") as fh:
                    names = json.load(fh)
                if isinstance(names, dict):
                    self._class_names = [names[str(i)] for i in range(len(names))]
                elif isinstance(names, list):
                    self._class_names = [str(item) for item in names]
            self.last_error = ""
            logger.info("Fallback activity LSTM model loaded from %s", path)
        except Exception as exc:
            self.backend = "unavailable"
            self.last_error = str(exc)
            logger.exception("Fallback activity model load failed: %s", exc)

    def is_model_available(self) -> bool:
        return self._model is not None

    def model_status(self) -> Dict[str, Any]:
        return {
            "available": self.is_model_available(),
            "backend": self.backend,
            "path": str(self.model_path),
            "last_error": self.last_error,
        }

    def predict_sequence(self, frames) -> Dict[str, Any]:
        return self.predict(frames)

    def predict(self, frames) -> Dict[str, Any]:
        if not self.is_model_available():
            return self._result("unavailable", "unknown", 0.0, [])

        frame = self._select_frame(frames)
        if frame is None:
            return self._result("no_frame", "unknown", 0.0, [])

        if self.backend == "yolov8":
            return self._predict_yolo(frame)

        return self._predict_lstm(frames)

    def _predict_yolo(self, frame) -> Dict[str, Any]:
        try:
            result = self._model(frame, verbose=False)[0]
            detections = []

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = str(result.names.get(cls_id, cls_id))
                detections.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "conf": conf,
                        "label": label,
                    }
                )

            if detections:
                best = max(detections, key=lambda item: item["conf"])
                return self._result("ok", best["label"], best["conf"], detections)

            return self._result("ok", "normal", 0.0, [])
        except Exception as exc:
            logger.exception("Activity YOLOv8 prediction failed: %s", exc)
            self.last_error = str(exc)
            return self._result("error", "error", 0.0, [])

    def _predict_lstm(self, frames) -> Dict[str, Any]:
        try:
            arr = np.asarray(frames, dtype=np.float32)
            if arr.ndim == 4:
                arr = np.expand_dims(arr, axis=0)
            arr = arr / 255.0

            pred = self._model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(pred))
            conf = float(pred[idx])
            label = self._class_names[idx] if idx < len(self._class_names) else str(idx)
            return self._result("ok", label, conf, [])
        except Exception as exc:
            logger.exception("Fallback activity prediction failed: %s", exc)
            self.last_error = str(exc)
            return self._result("error", "error", 0.0, [])

    @staticmethod
    def _select_frame(frames):
        if frames is None:
            return None
        if hasattr(frames, "shape") and len(frames.shape) == 3:
            return frames
        if hasattr(frames, "__len__") and len(frames) > 0:
            return frames[-1]
        return None

    @staticmethod
    def _result(status: str, label: str, confidence: float, detections: List[dict]) -> Dict[str, Any]:
        return {
            "status": status,
            "label": str(label or "unknown"),
            "confidence": float(confidence or 0.0),
            "detections": detections,
        }


_predictor = None
_predictor_lock = Lock()


def get_activity_predictor() -> ActivityPredictor:
    global _predictor
    with _predictor_lock:
        if _predictor is None:
            _predictor = ActivityPredictor()
        return _predictor
