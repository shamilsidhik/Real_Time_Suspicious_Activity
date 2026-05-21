"""
Safe wrapper for the activity detection model.
Usage:
    detector = ActivityDetector()
    label, confidence = detector.predict(frames)   # frames: list of np arrays
"""
import os, json, logging
import numpy as np

logger = logging.getLogger(__name__)


class ActivityDetector:
    def __init__(self):
        self._model = None
        self._class_names = ["normal", "suspicious"]
        self._loaded = False
        self._try_load()

    def _try_load(self):
        model_path = os.environ.get("ACTIVITY_MODEL_PATH", "ml/models/activity_mobilenet_lstm/activity_model.keras")
        names_path = os.environ.get("ACTIVITY_CLASS_NAMES_PATH", "ml/models/activity_mobilenet_lstm/class_names.json")

        if not os.path.exists(model_path):
            logger.warning(f"Activity model not found at {model_path}. Run training first.")
            return

        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(model_path)

            if os.path.exists(names_path):
                with open(names_path) as f:
                    self._class_names = json.load(f)

            self._loaded = True
            logger.info(f"Activity model loaded. Classes: {self._class_names}")
        except Exception as e:
            logger.error(f"Failed to load activity model: {e}")

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def is_model_available(self) -> bool:
        return self._loaded

    def predict(self, frames: list) -> tuple:
        """
        frames: list of numpy arrays (H, W, 3) — must be 30 frames
        Returns: (label: str, confidence: float)
        """
        if not self._loaded:
            return "unknown", 0.0

        try:
            seq = np.array([
                __import__("cv2").resize(f, (224, 224)) for f in frames
            ], dtype="float32") / 255.0   # (30, 224, 224, 3)

            seq = np.expand_dims(seq, 0)   # (1, 30, 224, 224, 3)
            probs = self._model.predict(seq, verbose=0)[0]
            idx = int(np.argmax(probs))
            return self._class_names[idx], float(probs[idx])
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return "error", 0.0


# Singleton — loaded once per Django worker
_detector = None

def get_activity_detector() -> ActivityDetector:
    global _detector
    if _detector is None:
        _detector = ActivityDetector()
    return _detector