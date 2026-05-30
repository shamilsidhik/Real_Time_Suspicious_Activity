from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def augment(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    canvas = np.full((h + 80, w + 50, 3), 18, dtype=np.uint8)
    phone = canvas.copy()
    cv2.rectangle(phone, (10, 10), (phone.shape[1] - 10, phone.shape[0] - 10), (5, 5, 7), -1)
    cv2.rectangle(phone, (24, 32), (phone.shape[1] - 24, phone.shape[0] - 34), (35, 35, 42), -1)
    resized = cv2.resize(image, (phone.shape[1] - 60, phone.shape[0] - 90))
    phone[42:42 + resized.shape[0], 30:30 + resized.shape[1]] = resized

    src = np.float32([[0, 0], [phone.shape[1], 0], [phone.shape[1], phone.shape[0]], [0, phone.shape[0]]])
    jitter = 34
    dst = np.float32([
        [random.randint(0, jitter), random.randint(0, jitter)],
        [phone.shape[1] - random.randint(0, jitter), random.randint(0, jitter)],
        [phone.shape[1] - random.randint(0, jitter), phone.shape[0] - random.randint(0, jitter)],
        [random.randint(0, jitter), phone.shape[0] - random.randint(0, jitter)],
    ])
    warped = cv2.warpPerspective(phone, cv2.getPerspectiveTransform(src, dst), (phone.shape[1], phone.shape[0]))

    glare = np.zeros_like(warped)
    cv2.ellipse(glare, (warped.shape[1] // 2, warped.shape[0] // 4), (warped.shape[1] // 3, 24), -18, 0, 360, (255, 255, 255), -1)
    warped = cv2.addWeighted(warped, 0.88, glare, 0.12, 0)
    warped = cv2.GaussianBlur(warped, (3, 3), 0)
    warped[:, ::4] = (warped[:, ::4] * 0.82).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(45, 75)]
    ok, buf = cv2.imencode(".jpg", warped, encode_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else warped


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate phone-screen weapon presentation augmentations.")
    parser.add_argument("--images", required=True, help="Source weapon image directory.")
    parser.add_argument("--out", default="ml/datasets/weapons_yolo/images/phone_val")
    parser.add_argument("--copies", type=int, default=2)
    args = parser.parse_args()

    src = Path(args.images)
    if not src.is_absolute():
        src = ROOT / src
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    for image_path in src.glob("*.*"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for idx in range(args.copies):
            cv2.imwrite(str(out / f"{image_path.stem}_phone_{idx}.jpg"), augment(image))
    print(f"Phone-screen augmentations written to {out}")


if __name__ == "__main__":
    main()
