from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceRecognizer:
    def __init__(
        self,
        known_dir: str | None = None,
        threshold: float = 0.30,
        min_size: int = 70,
    ) -> None:
        self.known_dir = Path(known_dir or os.environ.get("KNOWN_FACES_DIR", "ml/known_faces"))
        self.threshold = float(os.environ.get("FACE_MATCH_THRESHOLD", threshold))
        self.min_size = int(os.environ.get("FACE_MIN_SIZE", min_size))
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._known: list[dict[str, Any]] = []
        self._load_known_faces()

    def _load_known_faces(self) -> None:
        self._known = []
        if not self.known_dir.exists():
            logger.warning("Known faces folder not found: %s", self.known_dir)
            return

        for person_dir in sorted(p for p in self.known_dir.iterdir() if p.is_dir()):
            samples = []
            for image_path in sorted(person_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                faces = self._detect_faces(image)
                if not faces:
                    continue
                x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                samples.append(self._signature(image[y:y + h, x:x + w]))
            if samples:
                self._known.append({"name": person_dir.name, "signatures": samples})

        logger.info("Loaded %d known face profile(s) from %s", len(self._known), self.known_dir)

    def is_model_available(self) -> bool:
        return not self._cascade.empty()

    def known_count(self) -> int:
        return len(self._known)

    def _detect_faces(self, frame) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(self.min_size, self.min_size),
        )
        return self._suppress_overlaps([tuple(int(v) for v in face) for face in faces])

    def _suppress_overlaps(self, faces: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        kept: list[tuple[int, int, int, int]] = []
        for face in sorted(faces, key=lambda box: box[2] * box[3], reverse=True):
            if all(self._iou(face, other) < 0.35 for other in kept):
                kept.append(face)
        return kept

    def _iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
        inter = iw * ih
        union = (aw * ah) + (bw * bh) - inter
        return inter / union if union else 0.0

    def _signature(self, face_img) -> np.ndarray:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(cv2.resize(gray, (96, 96)))

        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        hist = hist / max(float(hist.sum()), 1.0)

        center = gray[12:84, 12:84]
        lbp = self._lbp(center)
        lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
        lbp_hist = lbp_hist / max(float(lbp_hist.sum()), 1.0)

        small = cv2.resize(gray, (12, 12)).astype("float32").flatten() / 255.0
        signature = np.concatenate([hist * 0.5, lbp_hist * 1.5, small * 0.25]).astype("float32")
        norm = float(np.linalg.norm(signature))
        return signature / norm if norm else signature

    def _lbp(self, gray: np.ndarray) -> np.ndarray:
        center = gray[1:-1, 1:-1]
        code = np.zeros_like(center, dtype=np.uint8)
        neighbors = [
            gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:],
            gray[1:-1, 2:], gray[2:, 2:], gray[2:, 1:-1],
            gray[2:, :-2], gray[1:-1, :-2],
        ]
        for bit, neighbor in enumerate(neighbors):
            code |= ((neighbor >= center).astype(np.uint8) << bit)
        return code

    def _match(self, signature: np.ndarray) -> tuple[str, float]:
        if not self._known:
            return "unknown_person", 0.0

        best_name = "unknown_person"
        best_score = -1.0
        for known in self._known:
            score = max(
                float(np.dot(signature, sample))
                for sample in known["signatures"]
            )
            if score > best_score:
                best_name = known["name"]
                best_score = score

        if best_score >= self.threshold:
            return best_name, best_score
        return "unknown_person", best_score

    def detect_frame(self, frame) -> dict[str, Any]:
        if not self.is_model_available():
            return {"status": "unavailable", "detections": []}

        detections = []
        for x, y, w, h in self._detect_faces(frame):
            signature = self._signature(frame[y:y + h, x:x + w])
            name, score = self._match(signature)
            detections.append({
                "bbox": [x, y, x + w, y + h],
                "label": name,
                "conf": max(score, 0.0),
                "unknown": name == "unknown_person",
            })

        return {"status": "ok", "detections": detections, "known_count": self.known_count()}

    detect_image = detect_frame
