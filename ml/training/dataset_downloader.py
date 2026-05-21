"""
Downloads datasets from Kaggle or Roboflow.
Run: python ml/training/dataset_downloader.py --type activity
     python ml/training/dataset_downloader.py --type id_card
     python ml/training/dataset_downloader.py --type both
"""
import argparse, os, subprocess, sys
from pathlib import Path


ACTIVITY_DATASETS = [
    # (kaggle_id, destination, description)
    ("mohamedmustafa/real-life-violence-situations-dataset", "ml/datasets/activity_raw",
     "Real-life violence vs non-violence videos"),
    ("rresma7/ucf-crime-dataset-subset", "ml/datasets/activity_raw",
     "UCF Crime surveillance subset"),
]

ID_CARD_DATASETS = [
    ("muhammadehsanansari/national-identity-card-detection", "ml/datasets/id_card",
     "Pakistani ID card detection with YOLO labels"),
    ("humansintheloop/id-card-segmentation", "ml/datasets/id_card_raw",
     "ID card segmentation dataset"),
]


def download_kaggle(dataset_id: str, dest: str):
    Path(dest).mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset_id, "-p", dest, "--unzip"]
    print(f"\n⬇️  Downloading: {dataset_id}")
    print(f"   → {dest}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"❌ Failed to download {dataset_id}")
        print("   Make sure kaggle.json is set up (~/.kaggle/kaggle.json)")
        return False
    print(f"✅ Downloaded → {dest}")
    return True


def download_activity():
    print("\n=== DOWNLOADING ACTIVITY DATASETS ===")
    success = False
    for dataset_id, dest, desc in ACTIVITY_DATASETS:
        print(f"\nTrying: {desc}")
        if download_kaggle(dataset_id, dest):
            success = True
            break  # stop after first successful download

    if not success:
        print("\n⚠️  All Kaggle downloads failed. Manual alternative:")
        print("   1. Go to https://universe.roboflow.com")
        print("   2. Search 'suspicious activity' or 'violence detection'")
        print("   3. Export in YOLOv5 format")
        print("   4. Place in ml/datasets/activity/")


def download_id_card():
    print("\n=== DOWNLOADING ID CARD DATASETS ===")
    success = False
    for dataset_id, dest, desc in ID_CARD_DATASETS:
        print(f"\nTrying: {desc}")
        if download_kaggle(dataset_id, dest):
            success = True
            break

    # Ensure data.yaml exists
    yaml_path = Path("ml/datasets/id_card/data.yaml")
    if not yaml_path.exists():
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(
            "path: ml/datasets/id_card\n"
            "train: images/train\nval: images/val\ntest: images/test\n"
            "nc: 1\nnames: ['id_card']\n"
        )
        print(f"✅ Created default data.yaml → {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["activity", "id_card", "both"], default="both")
    args = parser.parse_args()

    if args.type in ("activity", "both"):
        download_activity()
    if args.type in ("id_card", "both"):
        download_id_card()
        