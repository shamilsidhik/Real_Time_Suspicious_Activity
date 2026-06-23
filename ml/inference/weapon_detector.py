from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


class WeaponDetector:
    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parents[2]

        # Use your existing yolov8s.pt from project root
        self.model_path = os.environ.get(
            "WEAPON_MODEL_PATH",
            str(base_dir / "yolov8s.pt")
        )

        self.model = None
        self.available = False

        # Lower confidence because weapon is inside phone screen
        self.conf_threshold = float(os.environ.get("WEAPON_CONF", "0.15"))

        # YOLO may return knife, gun, pistol, etc.
        self.weapon_classes = {
            "knife",
            "gun",
            "pistol",
            "rifle",
            "weapon",
            "handgun",
            "firearm"
        }

        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                self.available = True
            else:
                print(f"Weapon model not found: {self.model_path}")
        except Exception as e:
            print("Weapon model loading failed:", e)
            self.model = None
            self.available = False

    def is_model_available(self) -> bool:
        return self.available and self.model is not None

    def detect_image(self, frame) -> dict[str, Any]:
        if not self.is_model_available():
            return {
                "status": "model_unavailable",
                "detections": []
            }

        detections = []

        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = str(self.model.names[cls_id]).lower()

                    # Main fix: count knife/gun/pistol as WEAPON
                    if label not in self.weapon_classes:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    detections.append({
                        "label": label,
                        "conf": round(conf, 3),
                        "bbox": [x1, y1, x2, y2]
                    })

            return {
                "status": "ok",
                "detections": detections
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "detections": []
            }