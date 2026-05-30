from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any

import numpy as np

from .common import cpu_device, resolve_model_path

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_LOCK = threading.Lock()
_LOAD_ERROR = ""


def _preferred_path():
    return resolve_model_path(
        "WEAPON_MODEL_PATH",
        [
            "ml/models/weapons_yolov8/best_openvino_model",
            "ml/models/weapons_yolov8/best.onnx",
            "ml/models/weapons_yolov8/best.pt",
        ],
    )


def _load_model():
    global _MODEL, _LOAD_ERROR
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        path = _preferred_path()
        if not path.exists():
            _LOAD_ERROR = f"Weapon model not found: {path}"
            logger.warning(_LOAD_ERROR)
            return None
        try:
            from ultralytics import YOLO

            _MODEL = YOLO(str(path))
            _LOAD_ERROR = ""
            logger.info("Weapon model loaded from %s", path)
        except Exception as exc:
            _LOAD_ERROR = str(exc)
            logger.exception("Weapon model load failed: %s", exc)
        return _MODEL


def is_model_available() -> bool:
    return _load_model() is not None


def model_status() -> dict[str, Any]:
    path = _preferred_path()
    return {
        "available": is_model_available(),
        "backend": path.suffix.lower().lstrip(".") or "openvino",
        "path": str(path),
        "last_error": _LOAD_ERROR,
    }


class WeaponDetector:
    def __init__(self, conf: float = 0.35, iou: float = 0.45, imgsz: int = 480, hold_frames: int = 12) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = cpu_device()
        self._model = _load_model()
        self._infer_lock = threading.Lock()
        self._held: list[dict[str, Any]] = []
        self._held_ttl = 0
        self._recent: deque[list[dict[str, Any]]] = deque(maxlen=2)
        self.hold_frames = max(hold_frames, 1)

    def is_model_available(self) -> bool:
        return self._model is not None

    def model_status(self) -> dict[str, Any]:
        return model_status()

    def detect_frame(self, frame: np.ndarray) -> dict[str, Any]:
        if self._model is None:
            return {"status": "unavailable", "detections": self._held}
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
            logger.exception("Weapon inference failed: %s", exc)
            return {"status": "error", "detections": self._held, "error": str(exc)}

        detections = []
        h, w = frame.shape[:2]
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            cls_id = int(box.cls[0].item())
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(box.conf[0].item()),
                    "label": str(result.names.get(cls_id, cls_id)),
                }
            )

        self._recent.append(detections)
        if detections:
            self._held = detections
            self._held_ttl = self.hold_frames
        elif self._held_ttl > 0:
            self._held_ttl -= 1
        else:
            self._held = []
        return {"status": "ok", "detections": list(self._held)}

    detect_image = detect_frame
