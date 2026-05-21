"""
Organizes downloaded activity videos into train/val/test splits.
Run: python ml/datasets/organize_activity_dataset.py --src <downloaded_folder>
"""
import os, shutil, random, argparse
from pathlib import Path

SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}

def organize(src: str, dst: str, classes=("normal", "suspicious")):
    src = Path(src)
    dst = Path(dst)
    random.seed(42)

    for cls in classes:
        cls_dir = src / cls
        if not cls_dir.exists():
            print(f"⚠️  {cls_dir} not found, skipping")
            continue

        files = list(cls_dir.glob("*.*"))
        random.shuffle(files)
        n = len(files)
        cuts = [int(n * SPLITS["train"]), int(n * (SPLITS["train"] + SPLITS["val"]))]

        for split, chunk in zip(
            ["train", "val", "test"],
            [files[:cuts[0]], files[cuts[0]:cuts[1]], files[cuts[1]:]],
        ):
            out = dst / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in chunk:
                shutil.copy2(f, out / f.name)
            print(f"  {split}/{cls}: {len(chunk)} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source folder with normal/ and suspicious/ subfolders")
    parser.add_argument("--dst", default="ml/datasets/activity", help="Output destination")
    args = parser.parse_args()
    organize(args.src, args.dst)
    print("✅ Done organizing dataset")