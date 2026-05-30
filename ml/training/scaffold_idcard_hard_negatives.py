from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NEGATIVE_CLASSES = ["faces", "portraits", "passports_non_target", "phone_screens", "printed_photos"]


def main() -> None:
    base = ROOT / "ml" / "datasets" / "id_card_hard_negatives"
    for name in NEGATIVE_CLASSES:
        (base / name).mkdir(parents=True, exist_ok=True)
    readme = base / "README.md"
    readme.write_text(
        "# ID-card hard negatives\n\n"
        "Add non-card lookalikes here and include them as empty-label negative images when retraining the ID-card YOLO model.\n"
        "Target folders: faces, portraits, passports_non_target, phone_screens, printed_photos.\n",
        encoding="utf-8",
    )
    print(f"Hard-negative scaffold ready: {base}")


if __name__ == "__main__":
    main()
