from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any

import cv2
import numpy as np

from .common import cpu_device, resolve_model_path

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_LOCK = threading.Lock()
_LOAD_ERROR = ""


def _preferred_path():
    return resolve_model_path(
        "ID_CARD_MODEL_PATH",
        [
            "ml/models/id_card_yolov5/best_openvino_model",
            "ml/models/id_card_yolov5/best.onnx",
            "ml/models/id_card_yolov5/best.pt",
            "ml/models/id_card_yolov5/idcard/weights/best.pt",
        ],
    )


def _load_model():
    global _MODEL, _LOAD_ERROR
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        path = _preferred_path()
        if not path.exists():
            _LOAD_ERROR = f"ID-card model not found: {path}"
            logger.warning(_LOAD_ERROR)
            return None

        try:
            from ultralytics import YOLO

            _MODEL = YOLO(str(path))
            _LOAD_ERROR = ""
            logger.info("ID-card model loaded from %s", path)
        except Exception as exc:
            _LOAD_ERROR = str(exc)
            logger.exception("ID-card model load failed: %s", exc)
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


class IDDetector:
    def __init__(
        self,
        conf: float = 0.55,
        iou: float = 0.45,
        imgsz: int = 480,
        confirm_frames: int = 2,
        hold_frames: int = 10,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = cpu_device()
        self.confirm_frames = max(confirm_frames, 1)
        self.hold_frames = max(hold_frames, 1)
        self._history: deque[list[dict[str, Any]]] = deque(maxlen=self.confirm_frames)
        self._held: list[dict[str, Any]] = []
        self._held_ttl = 0
        self._infer_lock = threading.Lock()
        self._model = _load_model()

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
            logger.exception("ID-card inference failed: %s", exc)
            return {"status": "error", "detections": self._held, "error": str(exc)}

        raw = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0].item())
            raw.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(box.conf[0].item()),
                    "label": str(result.names.get(cls_id, cls_id)),
                }
            )

        filtered = [det for det in raw if self._passes_geometry(frame, det)]
        self._history.append(filtered)
        confirmed = self._confirm(filtered)

        if confirmed:
            self._held = confirmed
            self._held_ttl = self.hold_frames
        elif self._held_ttl > 0:
            self._held_ttl -= 1
        else:
            self._held = []

        return {"status": "ok", "detections": list(self._held), "raw_detections": raw}

    detect_image = detect_frame

    def _passes_geometry(self, frame: np.ndarray, det: dict[str, Any]) -> bool:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = _clip_bbox(det.get("bbox", []), w, h)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return False

        conf = float(det.get("conf", 0.0))
        area_ratio = (bw * bh) / float(max(w * h, 1))
        aspect = bw / float(max(bh, 1))
        if conf < self.conf or area_ratio < 0.012:
            return False
        if not (1.35 <= aspect <= 2.35):
            return False

        crop = frame[y1:y2, x1:x2]
        quality = _rectangle_quality(crop)
        if quality < 0.18:
            return False
        return True

    def _confirm(self, current: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.confirm_frames <= 1:
            return current
        confirmed = []
        for det in current:
            hits = 1
            for prior in list(self._history)[:-1]:
                if any(_iou(det["bbox"], other["bbox"]) >= 0.35 for other in prior):
                    hits += 1
            if hits >= self.confirm_frames:
                confirmed.append(det)
        return confirmed


def _clip_bbox(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        return 0, 0, 0, 0
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def _rectangle_quality(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    area = crop.shape[0] * crop.shape[1]
    best = 0.0
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        contour_area = cv2.contourArea(contour)
        if len(approx) == 4:
            best = max(best, contour_area / float(max(area, 1)))
    edge_density = float(np.count_nonzero(edges)) / float(max(area, 1))
    return max(best, min(edge_density * 2.0, 0.4))


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(max(area_a + area_b - inter, 1))
