from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["gun", "handgun", "pistol", "knife", "weapon"]


def ensure_yolo_tree(out_dir: Path) -> None:
    for split in ["train", "val", "test", "phone_val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_yolo_split(src: Path, out_dir: Path, split: str) -> None:
    for image_dir_name in ["images", "Images", "JPEGImages"]:
        image_dir = src / image_dir_name / split
        if image_dir.exists():
            break
    else:
        image_dir = src / split / "images"
    label_dir = src / "labels" / split if (src / "labels" / split).exists() else src / split / "labels"

    if not image_dir.exists():
        return
    for image in image_dir.glob("*.*"):
        shutil.copy2(image, out_dir / "images" / split / image.name)
        label = label_dir / f"{image.stem}.txt"
        if label.exists():
            shutil.copy2(label, out_dir / "labels" / split / label.name)


def write_data_yaml(out_dir: Path, classes: list[str]) -> None:
    payload = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(classes)},
    }
    with open(out_dir / "weapon_data.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public/Kaggle/Roboflow-style weapon datasets for YOLO training.")
    parser.add_argument("--source", default=os.environ.get("WEAPON_DATASET_DIR", ""), help="Existing dataset directory.")
    parser.add_argument("--out", default="ml/datasets/weapons_yolo")
    parser.add_argument("--classes", nargs="*", default=CLASSES)
    args = parser.parse_args()

    out_dir = ROOT / args.out
    ensure_yolo_tree(out_dir)

    if args.source:
        src = Path(args.source)
        if not src.is_absolute():
            src = ROOT / src
        for split in ["train", "val", "test"]:
            copy_yolo_split(src, out_dir, split)

    write_data_yaml(out_dir, args.classes)
    print(f"Dataset scaffold ready: {out_dir}")
    print("For Kaggle or Roboflow, download with their CLIs using env vars, then pass --source to this script.")


if __name__ == "__main__":
    main()
