from __future__ import annotations

import json
import logging
import threading
from collections import deque
from typing import Any

import numpy as np

from .common import cpu_device, resolve_model_path

logger = logging.getLogger(__name__)

_MODEL = None
_BACKEND = "unavailable"
_CLASS_NAMES: list[str] = []
_LOAD_ERROR = ""
_MODEL_LOCK = threading.Lock()


def _preferred_path():
    return resolve_model_path(
        "ACTIVITY_MODEL_PATH",
        [
            "ml/models/activity_yolov8/best_openvino_model",
            "ml/models/activity_yolov8/best.onnx",
            "ml/models/activity_yolov8/best.pt",
            "ml/models/activity_mobilenet_lstm/activity_model.keras",
        ],
    )


def _load_model():
    global _MODEL, _BACKEND, _CLASS_NAMES, _LOAD_ERROR
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        path = _preferred_path()
        if not path.exists():
            _LOAD_ERROR = f"Activity model not found: {path}"
            logger.warning(_LOAD_ERROR)
            return None
        try:
            if path.suffix.lower() == ".keras":
                from tensorflow.keras.models import load_model

                _MODEL = load_model(str(path))
                _BACKEND = "mobilenet_lstm"
                names_path = path.with_name("class_names.json")
                if names_path.exists():
                    with open(names_path, "r", encoding="utf-8") as fh:
                        names = json.load(fh)
                    _CLASS_NAMES = list(names.values()) if isinstance(names, dict) else list(names)
            else:
                from ultralytics import YOLO

                _MODEL = YOLO(str(path))
                _BACKEND = path.suffix.lower().lstrip(".") or "openvino"
            _LOAD_ERROR = ""
            logger.info("Activity model loaded from %s", path)
        except Exception as exc:
            _LOAD_ERROR = str(exc)
            logger.exception("Activity model load failed: %s", exc)
        return _MODEL


def is_model_available() -> bool:
    return _load_model() is not None


def model_status() -> dict[str, Any]:
    path = _preferred_path()
    return {
        "available": is_model_available(),
        "backend": _BACKEND,
        "path": str(path),
        "last_error": _LOAD_ERROR,
    }


class ActivityPredictor:
    def __init__(self, conf: float = 0.32, iou: float = 0.45, imgsz: int = 480, hold_frames: int = 16) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = cpu_device()
        self._model = _load_model()
        self._infer_lock = threading.Lock()
        self._labels: deque[tuple[str, float]] = deque(maxlen=3)
        self._held_label = "normal"
        self._held_conf = 0.0
        self._held_ttl = 0
        self.hold_frames = max(hold_frames, 1)

    def is_model_available(self) -> bool:
        return self._model is not None

    def model_status(self) -> dict[str, Any]:
        return model_status()

    def predict_frame(self, frame: np.ndarray) -> dict[str, Any]:
        if self._model is None:
            return self._result("unavailable", self._held_label, self._held_conf, [])
        if _BACKEND == "mobilenet_lstm":
            return self._predict_lstm([frame])
        return self._predict_yolo(frame)

    def predict(self, frames) -> dict[str, Any]:
        frame = _select_frame(frames)
        if frame is None:
            return self._result("no_frame", "unknown", 0.0, [])
        return self.predict_frame(frame)

    predict_sequence = predict

    def _predict_yolo(self, frame: np.ndarray) -> dict[str, Any]:
        try:
            with self._infer_lock:
                result = self._model.predict(
                    source=frame,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )[0]
        except Exception as exc:
            logger.exception("Activity inference failed: %s", exc)
            return self._result("error", self._held_label, self._held_conf, [])

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0].item())
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(box.conf[0].item()),
                    "label": str(result.names.get(cls_id, cls_id)),
                }
            )

        if detections:
            best = max(detections, key=lambda item: item["conf"])
            self._labels.append((best["label"], best["conf"]))
            label, conf = self._smooth_label()
            self._held_label, self._held_conf, self._held_ttl = label, conf, self.hold_frames
        elif self._held_ttl > 0:
            self._held_ttl -= 1
        else:
            self._held_label, self._held_conf = "normal", 0.0
        return self._result("ok", self._held_label, self._held_conf, detections)

    def _predict_lstm(self, frames) -> dict[str, Any]:
        try:
            arr = np.asarray(frames, dtype=np.float32)
            if arr.ndim == 4:
                arr = np.expand_dims(arr, axis=0)
            arr = arr / 255.0
            pred = self._model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(pred))
            label = _CLASS_NAMES[idx] if idx < len(_CLASS_NAMES) else str(idx)
            conf = float(pred[idx])
            self._held_label, self._held_conf, self._held_ttl = label, conf, self.hold_frames
            return self._result("ok", label, conf, [])
        except Exception as exc:
            logger.exception("Activity LSTM inference failed: %s", exc)
            return self._result("error", self._held_label, self._held_conf, [])

    def _smooth_label(self) -> tuple[str, float]:
        scores: dict[str, list[float]] = {}
        for label, conf in self._labels:
            scores.setdefault(label, []).append(conf)
        label = max(scores, key=lambda key: (len(scores[key]), sum(scores[key]) / len(scores[key])))
        return label, float(sum(scores[label]) / len(scores[label]))

    @staticmethod
    def _result(status: str, label: str, confidence: float, detections: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "status": status,
            "label": str(label or "unknown"),
            "confidence": float(confidence or 0.0),
            "detections": detections,
        }


def _select_frame(frames):
    if frames is None:
        return None
    if hasattr(frames, "shape") and len(frames.shape) == 3:
        return frames
    if hasattr(frames, "__len__") and len(frames) > 0:
        return frames[-1]
    return None


_predictor = None
_predictor_lock = threading.Lock()


def get_activity_predictor() -> ActivityPredictor:
    global _predictor
    with _predictor_lock:
        if _predictor is None:
            _predictor = ActivityPredictor()
        return _predictor
