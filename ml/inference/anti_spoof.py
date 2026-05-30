from __future__ import annotations

from typing import Any

import numpy as np


def is_model_available() -> bool:
    return False


def model_status() -> dict[str, Any]:
    return {
        "available": False,
        "backend": "disabled_live_mode",
        "path": "",
        "last_error": "Anti-spoof is disabled in live mode until a real direct-camera liveness model is installed.",
    }


class AntiSpoofEngine:
    mode = "disabled"

    def is_model_available(self) -> bool:
        return False

    def model_status(self) -> dict[str, Any]:
        return model_status()

    def predict_frame(self, frame) -> dict[str, Any]:
        return self.evaluate(frame)

    def evaluate(self, frame) -> dict[str, Any]:
        return {"status": "DISABLED", "spoof": False, "confidence": 0.0}


def _as_float_gray(frame):
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr.astype(np.float32)


def frame_difference_score(frames):
    if len(frames) < 2:
        return 0.0
    diffs = []
    previous = _as_float_gray(frames[0])
    for frame in frames[1:]:
        current = _as_float_gray(frame)
        diffs.append(float(np.abs(current - previous).mean()))
        previous = current
    return float(np.mean(diffs)) if diffs else 0.0


def repeated_frame_ratio(frames, threshold=0.25):
    if len(frames) < 2:
        return 0.0
    repeated = 0
    comparisons = 0
    previous = _as_float_gray(frames[0])
    for frame in frames[1:]:
        current = _as_float_gray(frame)
        comparisons += 1
        if float(np.abs(current - previous).mean()) < threshold:
            repeated += 1
        previous = current
    return repeated / comparisons if comparisons else 0.0


def is_spoof_sequence(frames):
    return {
        "status": "DISABLED",
        "spoof": False,
        "difference_score": frame_difference_score(frames) if len(frames) > 1 else 0.0,
        "repeated_ratio": repeated_frame_ratio(frames) if len(frames) > 1 else 0.0,
    }
