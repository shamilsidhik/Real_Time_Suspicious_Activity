"""ml/inference/id_detector.py"""
import os, logging
import numpy as np
logger = logging.getLogger(__name__)

class IDDetector:
    def __init__(self, weights_path=None):
        self.weights_path = weights_path or os.environ.get("ID_CARD_MODEL_PATH","ml/models/id_card_yolov5/best.pt")
        self._model = None
        self.model_missing = True
        self._load()

    def _load(self):
        if not os.path.exists(self.weights_path):
            logger.warning("ID model not found: %s", self.weights_path)
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.weights_path)
            self.model_missing = False
            logger.info("ID model loaded OK")
        except Exception as e:
            logger.error("ID model load error: %s", e)

    def is_model_available(self):
        return not self.model_missing

    def detect_image(self, frame):
        if self.model_missing:
            return {"status":"unavailable","detections":[]}
        try:
            results = self._model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [int(x1),int(y1),int(x2),int(y2)],
                    "conf": float(box.conf[0]),
                    "label": results.names[int(box.cls[0])],
                })
            return {"status":"ok","detections":detections}
        except Exception as e:
            logger.error("ID detect error: %s", e)
            return {"status":"error","detections":[]}
