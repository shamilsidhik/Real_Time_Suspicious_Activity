from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "ml"
MODELS_ROOT = ML_ROOT / "models"


def project_path(*parts: str | os.PathLike[str]) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def resolve_model_path(env_name: str, candidates: Iterable[str | os.PathLike[str]]) -> Path:
    configured = os.environ.get(env_name, "").strip()
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else PROJECT_ROOT / path

    candidate_list = list(candidates)
    for candidate in candidate_list:
        path = Path(candidate)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            return path

    first = Path(candidate_list[0])
    return first if first.is_absolute() else PROJECT_ROOT / first


def cpu_device() -> str:
    return os.environ.get("YOLO_DEVICE", "cpu").strip() or "cpu"
