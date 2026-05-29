"""Lightweight liveness helpers for optional live anti-spoof checks.

The direct webcam pipeline disables anti-spoof by default because repeated
frames can be caused by transport, CPU pressure, or camera driver buffering.
These helpers are intentionally conservative and never suppress activity
overlays on their own.
"""
from __future__ import annotations

import numpy as np


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
    """Return a conservative normalized anti-spoof result."""
    if len(frames) < 12:
        return {
            "status": "warming",
            "spoof": False,
            "difference_score": 0.0,
            "repeated_ratio": 0.0,
        }

    diff_score = frame_difference_score(frames)
    repeat_ratio = repeated_frame_ratio(frames)

    spoof = diff_score < 0.35 and repeat_ratio > 0.9
    return {
        "status": "ok",
        "spoof": bool(spoof),
        "difference_score": diff_score,
        "repeated_ratio": repeat_ratio,
    }

