"""ml/inference/activity_predictor.py"""
import os, json, logging
import numpy as np
logger = logging.getLogger(__name__)

class ActivityPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.environ.get("ACTIVITY_MODEL_PATH","ml/models/activity_mobilenet_lstm/activity_model.keras")
        self.names_path = os.environ.get("ACTIVITY_CLASS_NAMES_PATH","ml/models/activity_mobilenet_lstm/class_names.json")
        self._model = None
        self.model_missing = True
        self._class_names = ["normal","suspicious"]
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            logger.warning("Activity model not found: %s", self.model_path)
            return
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(self.model_path)
            if os.path.exists(self.names_path):
                with open(self.names_path) as f:
                    self._class_names = json.load(f)
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
            return {"label":"unknown","confidence":0.0}
        try:
            import cv2
            resized = [cv2.resize(f,(224,224)) for f in frames]
            while len(resized) < 30:
                resized.append(resized[-1])
            seq = np.array(resized[:30],dtype="float32")/255.0
            seq = np.expand_dims(seq,0)
            probs = self._model.predict(seq,verbose=0)[0]
            idx = int(np.argmax(probs))
            return {"label":self._class_names[idx],"confidence":float(probs[idx])}
        except Exception as e:
            logger.error("Activity predict error: %s", e)
            return {"label":"error","confidence":0.0}
