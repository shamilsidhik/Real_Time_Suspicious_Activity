from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the live weapon detector.")
    parser.add_argument("--data", default="ml/datasets/weapons_yolo/weapon_data.yaml")
    parser.add_argument("--base", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    data = Path(args.data)
    if not data.is_absolute():
        data = ROOT / data
    model = YOLO(args.base)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device="cpu", project=str(ROOT / "runs" / "weapons"), name="yolov8")
    print("Copy the best weights to ml/models/weapons_yolov8/best.pt when validation is acceptable.")


if __name__ == "__main__":
    main()
