"""
Extracts fixed-length frame sequences from videos for LSTM training.
Run: python ml/training/preprocess_activity_videos.py \
         --src ml/datasets/activity \
         --out ml/datasets/activity_processed \
         --seq-len 30
"""
import os, cv2, argparse
import numpy as np
from pathlib import Path

IMG_SIZE = (224, 224)

def extract_frames(video_path: str, seq_len: int) -> list:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []

    # Sample evenly-spaced frames
    indices = np.linspace(0, total - 1, seq_len, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # Pad if short
    while len(frames) < seq_len:
        frames.append(np.zeros((*IMG_SIZE, 3), dtype=np.uint8))

    return frames[:seq_len]


def process_split(src_root: Path, out_root: Path, split: str, seq_len: int):
    split_dir = src_root / split
    if not split_dir.exists():
        print(f"  Skipping {split} (not found)")
        return

    for cls_dir in split_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        out_cls = out_root / split / cls_dir.name
        out_cls.mkdir(parents=True, exist_ok=True)

        video_files = list(cls_dir.glob("*.mp4")) + list(cls_dir.glob("*.avi")) + \
                      list(cls_dir.glob("*.mov")) + list(cls_dir.glob("*.mkv"))

        for i, vf in enumerate(video_files):
            frames = extract_frames(str(vf), seq_len)
            if not frames:
                print(f"    ⚠️  Could not read {vf.name}")
                continue
            arr = np.array(frames, dtype=np.uint8)  # (seq_len, H, W, 3)
            out_path = out_cls / f"{vf.stem}.npy"
            np.save(str(out_path), arr)
            if i % 20 == 0:
                print(f"    [{split}/{cls_dir.name}] {i+1}/{len(video_files)} processed")

        print(f"  ✅ {split}/{cls_dir.name}: {len(video_files)} videos → .npy sequences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="ml/datasets/activity")
    parser.add_argument("--out", default="ml/datasets/activity_processed")
    parser.add_argument("--seq-len", type=int, default=30)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        process_split(src, out, split, args.seq_len)

    print("\n✅ Preprocessing complete →", out)