from __future__ import annotations

import os
from collections import deque
from typing import Any

import cv2
import numpy as np


def frame_difference_score(frames) -> float:
    if frames is None or len(frames) < 2:
        return 0.0

    diffs = []
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)).mean()
        diffs.append(float(diff))

    return float(np.mean(diffs)) if diffs else 0.0


def repeated_frame_ratio(frames, repeated_threshold: float = 1.0) -> float:
    if frames is None or len(frames) < 2:
        return 0.0

    repeated = 0
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)).mean()
        if diff < repeated_threshold:
            repeated += 1

    return repeated / float(max(len(frames) - 1, 1))


class AntiSpoofEngine:
    """
    Passive anti-spoof is OFF by default.

    Why:
      - The old implementation marked low-motion scenes as spoof.
      - That is unsafe for live webcam streaming, especially when a real person
        is still or when frames are smoothed/compressed.

    Modes:
      - off      : always returns DISABLED
      - passive  : returns LOW_MOTION or LIVE_LIKE, but never hard-fails as SPOOF
    """

    def __init__(self, mode: str | None = None, history: int = 12) -> None:
        self.mode = (mode or os.environ.get("ANTI_SPOOF_MODE", "off")).strip().lower()
        self.history = max(history, 4)
        self._gray_frames: deque[np.ndarray] = deque(maxlen=self.history)

    def evaluate(self, frame) -> dict[str, Any]:
        if self.mode in {"", "off", "disabled", "none"}:
            return {
                "enabled": False,
                "status": "DISABLED",
                "confidence": 0.0,
                "reason": "Passive anti-spoof is disabled to avoid false positives on live MJPEG streams.",
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._gray_frames.append(gray)

        if len(self._gray_frames) < self.history:
            return {
                "enabled": True,
                "status": "WARMUP",
                "confidence": 0.0,
                "reason": "Collecting temporal history.",
            }

        frames = list(self._gray_frames)
        diff_score = frame_difference_score(frames)
        repeat_ratio = repeated_frame_ratio(frames)

        # Heuristic-only signal. Low motion is not equivalent to spoof.
        if diff_score < 2.0 or repeat_ratio > 0.70:
            return {
                "enabled": True,
                "status": "LOW_MOTION",
                "confidence": max(0.0, min(1.0, 1.0 - (diff_score / 10.0))),
                "reason": "Low temporal variation detected; treat as a prompt for active liveness, not as an automatic spoof verdict.",
            }

        return {
            "enabled": True,
            "status": "LIVE_LIKE",
            "confidence": max(0.0, min(1.0, diff_score / 20.0)),
            "reason": "Temporal variation is consistent with live motion, but this is not a certified PAD result.",
        }
