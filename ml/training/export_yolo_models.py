from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]


def export(weights: Path, fmt: str, imgsz: int) -> None:
    model = YOLO(str(weights))
    model.export(format=fmt, imgsz=imgsz, device="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO .pt models to CPU-friendly ONNX/OpenVINO artifacts.")
    parser.add_argument("--weights", default="", help="Optional single weights path relative to project root.")
    parser.add_argument("--format", choices=["onnx", "openvino"], default="openvino")
    parser.add_argument("--imgsz", type=int, default=480)
    args = parser.parse_args()

    candidates = [Path(args.weights)] if args.weights else [
        Path("ml/models/id_card_yolov5/best.pt"),
        Path("ml/models/id_card_yolov5/idcard/weights/best.pt"),
        Path("ml/models/activity_yolov8/best.pt"),
        Path("ml/models/weapons_yolov8/best.pt"),
    ]
    for rel in candidates:
        path = rel if rel.is_absolute() else ROOT / rel
        if path.exists():
            print(f"Exporting {path} -> {args.format}")
            export(path, args.format, args.imgsz)
        else:
            print(f"Skipping missing weights: {path}")


if __name__ == "__main__":
    main()
