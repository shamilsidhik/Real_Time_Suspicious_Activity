"""
Trains YOLOv5 for ID card detection.
Requires: pip install ultralytics
Run: python ml/training/train_yolo_id.py
"""
import os
from pathlib import Path


DATA_YAML   = "ml/datasets/id_card/data.yaml"
MODEL_OUT   = "ml/models/id_card_yolov5"
EPOCHS      = 50
BATCH_SIZE  = 16
IMG_SIZE    = 640


def check_dataset():
    yaml = Path(DATA_YAML)
    if not yaml.exists():
        print(f"❌ {DATA_YAML} not found. Download a dataset first (see guide).")
        return False

    img_dir = Path("ml/datasets/id_card/images/train")
    if not img_dir.exists() or not list(img_dir.glob("*")):
        print(f"❌ No training images found in {img_dir}")
        return False

    label_dir = Path("ml/datasets/id_card/labels/train")
    if not label_dir.exists() or not list(label_dir.glob("*.txt")):
        print(f"❌ No YOLO label .txt files found in {label_dir}")
        return False

    print(f"✅ Dataset OK: {len(list(img_dir.glob('*')))} training images")
    return True


def train():
    if not check_dataset():
        return

    Path(MODEL_OUT).mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO

        model = YOLO("yolov5su.pt")   # YOLOv5 small via ultralytics
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            project=MODEL_OUT,
            name="idcard",
            exist_ok=True,
            patience=10,
            save=True,
            plots=True,
        )

        # Copy best.pt to expected location
        best = Path(MODEL_OUT) / "idcard" / "weights" / "best.pt"
        dest = Path(MODEL_OUT) / "best.pt"
        if best.exists():
            import shutil
            shutil.copy2(best, dest)
            print(f"\n✅ Best model copied → {dest}")
        print("\n✅ Training complete!")

    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")


if __name__ == "__main__":
    train()