"""ml/inference/anti_spoof.py"""
import numpy as np

def frame_difference_score(frames):
    if len(frames) < 2:
        return 0.0
    diffs = []
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float)).mean()
        diffs.append(diff)
    return float(np.mean(diffs))

def repeated_frame_ratio(frames):
    if len(frames) < 2:
        return 0.0
    repeated = 0
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float)).mean()
        if diff < 1.0:
            repeated += 1
    return repeated / len(frames)
