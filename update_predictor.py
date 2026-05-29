# Update activity predictor to use YOLOv8 instead of Keras
import os

new_content = """
import os, logging
import numpy as np
logger = logging.getLogger(__name__)

class ActivityPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.environ.get("ACTIVITY_MODEL_PATH","ml/models/activity_yolov8/best.pt")
        self._model = None
        self.model_missing = True
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            logger.warning("Activity model not found: %s", self.model_path)
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self.model_missing = False
            logger.info("Activity model loaded OK")
        except Exception as e:
            logger.error("Activity model load error: %s", e)

    def is_model_available(self):
        return not self.model_missing

    def predict_sequence(self, frames):
        return self.predict(frames)

    def predict(self, frames):
        if self.model_missing:
            return {"label":"unknown","confidence":0.0,"status":"unavailable"}
        try:
            if hasattr(frames, '__len__') and len(frames) > 0:
                frame = frames[-1] if hasattr(frames[0], 'shape') else frames
            else:
                return {"label":"unknown","confidence":0.0,"status":"no_frame"}
            import cv2
            if hasattr(frame, 'shape'):
                results = self._model(frame, verbose=False)[0]
            else:
                return {"label":"unknown","confidence":0.0,"status":"invalid_frame"}
            if len(results.boxes) == 0:
                return {"label":"normal","confidence":0.8,"status":"ok"}
            best = max(results.boxes, key=lambda b: float(b.conf[0]))
            label = results.names[int(best.cls[0])]
            conf  = float(best.conf[0])
            return {"label":label,"confidence":conf,"status":"ok"}
        except Exception as e:
            logger.error("Activity predict error: %s", e)
            return {"label":"error","confidence":0.0,"status":"error"}
"""

with open("ml/inference/activity_predictor.py", "w") as f:
    f.write(new_content)
print("Done!")
