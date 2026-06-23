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


class IDDetector:
    def __init__(self, weights_path: str | None = None, conf: float = 0.50, imgsz: int = 640) -> None:
        self.weights_path = weights_path or os.environ.get(
            "ID_CARD_MODEL_PATH",
            "ml/models/id_card_yolov5/best.pt",
        )
        self.conf = conf
        self.imgsz = imgsz
        self.device = _default_device()

        self._lock = threading.Lock()
        self._model = None
        self.model_missing = True

        self._load()

    def _load(self) -> None:
        if not Path(self.weights_path).exists():
            logger.warning("ID detector weights not found: %s", self.weights_path)
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(self.weights_path)
            self.model_missing = False
            logger.info("ID detector loaded: %s", self.weights_path)
        except Exception:
            logger.exception("Failed to load ID detector from %s", self.weights_path)

    def is_model_available(self) -> bool:
        return not self.model_missing

    def detect_image(self, frame) -> dict[str, Any]:
        if self.model_missing or self._model is None:
            return {"status": "unavailable", "detections": []}

        try:
            with self._lock:
                result = self._model.predict(
                    source=frame,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )[0]

            detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                class_id = int(box.cls[0].item())
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(box.conf[0].item()),
                        "label": _class_name(result.names, class_id),
                    }
                )

            return {"status": "ok", "detections": detections}
        except Exception:
            logger.exception("ID detection failed")
            return {"status": "error", "detections": []}
