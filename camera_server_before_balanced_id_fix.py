# -*- coding: utf-8 -*-
"""
camera_server.py
================
Fast real-time camera server for the Django live monitoring page.

Speed improvements in this version:
- No fixed 0.35-second delay between detections.
- Inference always processes the newest camera frame and skips stale frames.
- The custom weapon model runs on every inference cycle.
- ID-card/activity models run less often so they do not block weapon detection.
- COCO knife fallback runs only occasionally when the custom model misses.
- Roboflow fallback runs in a separate thread and never blocks local YOLO.
- Camera buffering is reduced and MJPG capture is requested when supported.
- CUDA/half precision is used automatically when available.
- UTF-8 source encoding avoids the previous Windows non-UTF-8 SyntaxError.

Environment variables that can be changed in PowerShell before start.ps1:
    $env:WEAPON_IMGSZ="416"       # 320 = faster, 512/640 = more accurate
    $env:WEAPON_CONF="0.60"
    $env:ID_MODEL_EVERY="8"
    $env:COCO_FALLBACK_EVERY="6"
    $env:ENABLE_COCO_FALLBACK="1"
    $env:ROBOFLOW_API_KEY="YOUR_KEY"
    $env:WEAPON_EVENT_RESET_SECONDS="1.5"
    $env:ID_EVENT_RESET_SECONDS="1.5"
    $env:ENABLE_ACTIVITY_MODEL="1"
    $env:ACTIVITY_CONF="0.45"
    $env:ACTIVITY_MODEL_PATH="ml/models/activity_yolov8/best.pt"

Weapon events are stored permanently in weapon_detection_store.json.
ID-card events are stored permanently in id_card_detection_store.json.
The `weapons` and `id_cards` fields are permanent event counts.
The `weapons_live` and `id_cards_live` fields are visible-now counts.
"""

from __future__ import annotations

import base64
import collections
import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2


# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------
def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


# -----------------------------------------------------------------------------
# Paths and logging
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CAM] %(message)s",
)
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Camera and stream settings
# -----------------------------------------------------------------------------
CAMERA_WIDTH = env_int("CAMERA_WIDTH", 640)
CAMERA_HEIGHT = env_int("CAMERA_HEIGHT", 480)
CAMERA_FPS = env_int("CAMERA_FPS", 30)
STREAM_FPS = env_int("STREAM_FPS", 25)
JPEG_QUALITY = min(95, env_int("JPEG_QUALITY", 78))
CAMERA_INDEX_MAX = env_int("CAMERA_INDEX_MAX", 3)


# -----------------------------------------------------------------------------
# Model scheduling and performance settings
# -----------------------------------------------------------------------------
# Weapon detection runs on every inference cycle.
WEAPON_IMGSZ = env_int("WEAPON_IMGSZ", 416, minimum=160)
WEAPON_CONF = env_float("WEAPON_CONF", 0.60)

# Slower secondary models run only every N weapon inference cycles.
ID_IMGSZ = env_int("ID_IMGSZ", 640, minimum=160)
ID_MODEL_EVERY = env_int("ID_MODEL_EVERY", 3)
ID_CONF = env_float("ID_CONF", 0.45)
ID_HOLD_SECONDS = env_float("ID_HOLD_SECONDS", 0.45)
ID_CONFIRM_HITS = env_int("ID_CONFIRM_HITS", 2)
ID_CONTOUR_FALLBACK = env_bool("ID_CONTOUR_FALLBACK", True)

ACTIVITY_IMGSZ = env_int("ACTIVITY_IMGSZ", 416, minimum=160)
# Run activity often enough for responsive detection while leaving the
# custom weapon model as the highest-priority model.
ACTIVITY_MODEL_EVERY = env_int("ACTIVITY_MODEL_EVERY", 4)
ACTIVITY_CONF = env_float("ACTIVITY_CONF", 0.75)
ACTIVITY_HOLD_SECONDS = env_float("ACTIVITY_HOLD_SECONDS", 0.70)
ENABLE_ACTIVITY_MODEL = env_bool("ENABLE_ACTIVITY_MODEL", True)
ACTIVITY_MODEL_PATH = os.environ.get(
    "ACTIVITY_MODEL_PATH",
    "ml/models/activity_yolov8/best.pt",
).strip()

ENABLE_COCO_FALLBACK = env_bool("ENABLE_COCO_FALLBACK", True)
COCO_IMGSZ = env_int("COCO_IMGSZ", 416, minimum=160)
COCO_FALLBACK_EVERY = env_int("COCO_FALLBACK_EVERY", 6)
COCO_WEAPON_CONF = env_float("COCO_WEAPON_CONF", 0.75)

# Keep a detection briefly when one frame is missed to reduce box flicker.
WEAPON_HOLD_SECONDS = env_float("WEAPON_HOLD_SECONDS", 0.22)
MAX_DETECTIONS = env_int("MAX_DETECTIONS", 20)

# A weapon that remains visible is counted once. After it has been absent
# for this period, showing it again creates a new permanent event.
WEAPON_EVENT_RESET_SECONDS = env_float("WEAPON_EVENT_RESET_SECONDS", 1.5)
WEAPON_EVENT_HISTORY_LIMIT = env_int("WEAPON_EVENT_HISTORY_LIMIT", 5000)
WEAPON_STORE_PATH = BASE_DIR / "weapon_detection_store.json"

# An ID card that remains visible is one event. After it has been absent
# for this period, showing an ID card again creates another saved event.
ID_EVENT_RESET_SECONDS = env_float("ID_EVENT_RESET_SECONDS", 1.5)
ID_EVENT_HISTORY_LIMIT = env_int("ID_EVENT_HISTORY_LIMIT", 5000)
ID_STORE_PATH = BASE_DIR / "id_card_detection_store.json"


# -----------------------------------------------------------------------------
# Roboflow weapon API settings
# -----------------------------------------------------------------------------
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
ROBOFLOW_MODEL_ID = os.environ.get(
    "ROBOFLOW_MODEL_ID",
    "weapon-detection-f1lih/1",
).strip()
ROBOFLOW_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}"
ROBOFLOW_CONF = env_float("ROBOFLOW_CONF", 0.60)
WEAPON_API_EVERY = env_int("WEAPON_API_EVERY", 20)
ROBOFLOW_RESULT_MAX_AGE = env_float("ROBOFLOW_RESULT_MAX_AGE", 0.75)
ROBOFLOW_ENABLED = bool(ROBOFLOW_API_KEY)


# -----------------------------------------------------------------------------
# Detection filtering and labels
# -----------------------------------------------------------------------------
ID_AREA_MIN = env_int("ID_AREA_MIN", 900)
# Real ID cards are normally landscape rectangles. These limits reject
# tall body/arm boxes while still allowing perspective distortion.
ID_RATIO_MIN = env_float("ID_RATIO_MIN", 1.10)
ID_RATIO_MAX = env_float("ID_RATIO_MAX", 2.45)
ID_AREA_MAX_RATIO = env_float("ID_AREA_MAX_RATIO", 0.22)
ID_MAX_WIDTH_RATIO = env_float("ID_MAX_WIDTH_RATIO", 0.78)
ID_MAX_HEIGHT_RATIO = env_float("ID_MAX_HEIGHT_RATIO", 0.55)
ID_MIN_RECTANGULARITY = env_float("ID_MIN_RECTANGULARITY", 0.45)
ID_MIN_EDGE_DENSITY = env_float("ID_MIN_EDGE_DENSITY", 0.025)
ID_MAX_EDGE_DENSITY = env_float("ID_MAX_EDGE_DENSITY", 0.35)

SMOOTH_WINDOW = env_int("SMOOTH_WINDOW", 8)
SMOOTH_MIN_VOTE = env_int("SMOOTH_MIN_VOTE", 4)

# Common safe and suspicious activity labels. Labels are normalized before
# comparison, so "Non-Violence", "non_violence", and "non violence" work.
SAFE_ACTIVITY_LABELS = {
    "normal",
    "safe",
    "idle",
    "standing",
    "sitting",
    "walking",
    "talking",
    "working",
    "eating",
    "sleeping",
    "using phone",
    "non violence",
    "nonviolent",
    "no violence",
    "no fight",
    "person",
}

ACTIVITY_ALERT: set[str] = {
    "violence",
    "violent",
    "fight",
    "fighting",
    "assault",
    "attack",
    "theft",
    "stealing",
    "robbery",
    "suspicious",
    "suspicious activity",
    "fall",
    "falling",
    "running",
    "chasing",
    "punching",
    "kicking",
    "abuse",
    "vandalism",
}


def normalize_activity_label(label: str) -> str:
    value = str(label or "").strip().lower()
    for character in ("_", "-", "/", "\\"):
        value = value.replace(character, " ")
    return " ".join(value.split())


def is_safe_activity(label: str) -> bool:
    normalized = normalize_activity_label(label)
    if not normalized:
        return True

    if normalized in SAFE_ACTIVITY_LABELS:
        return True

    # Prevent labels such as "non violence" from being treated as violence
    # merely because the word "violence" appears in the class name.
    safe_phrases = (
        "non violence",
        "no violence",
        "not violent",
        "nonviolent",
        "no fight",
        "normal",
        "safe",
    )
    return any(phrase in normalized for phrase in safe_phrases)


def is_alert_activity(label: str) -> bool:
    normalized = normalize_activity_label(label)

    if not normalized or is_safe_activity(normalized):
        return False

    if normalized in ACTIVITY_ALERT:
        return True

    alert_words = (
        "violence",
        "violent",
        "fight",
        "assault",
        "attack",
        "theft",
        "steal",
        "robbery",
        "suspicious",
        "fall",
        "run",
        "chase",
        "punch",
        "kick",
        "abuse",
        "vandal",
    )
    return any(word in normalized for word in alert_words)

PISTOL_LABELS = {"pistol", "gun", "handgun", "firearm", "revolver"}
KNIFE_LABELS = {"knife", "blade"}
RIFLE_LABELS = {"rifle", "shotgun", "long gun"}
UNCERTAIN_WEAPON_LABELS = {"missile", "grenade", "weapon", "unknown"}
COCO_WEAPON_LABELS = {"knife"}

WEAPON_MODEL_LABELS = (
    PISTOL_LABELS
    | KNIFE_LABELS
    | RIFLE_LABELS
    | UNCERTAIN_WEAPON_LABELS
)

WEAPON_MIN_AREA = env_float("WEAPON_MIN_AREA", 0.003)
WEAPON_MAX_AREA = env_float("WEAPON_MAX_AREA", 0.30)
WEAPON_MAX_WIDTH_RATIO = env_float("WEAPON_MAX_WIDTH_RATIO", 0.70)
WEAPON_MAX_HEIGHT_RATIO = env_float("WEAPON_MAX_HEIGHT_RATIO", 0.70)


# -----------------------------------------------------------------------------
# Torch device selection
# -----------------------------------------------------------------------------
TORCH_AVAILABLE = False
CUDA_AVAILABLE = False
USE_HALF = False
DEVICE: Any = "cpu"
DEVICE_NAME = "cpu"

try:
    import torch

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = bool(torch.cuda.is_available())

    if CUDA_AVAILABLE:
        DEVICE = 0
        USE_HALF = True
        DEVICE_NAME = str(torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        # Prevent excessive CPU thread contention with camera/JPEG threads.
        requested_threads = env_int(
            "TORCH_NUM_THREADS",
            max(1, min(4, (os.cpu_count() or 4) // 2)),
        )
        torch.set_num_threads(requested_threads)
        DEVICE_NAME = f"cpu ({requested_threads} torch threads)"
except Exception as exc:
    log.warning("Torch device setup skipped: %s", exc)


# -----------------------------------------------------------------------------
# Shared state
# -----------------------------------------------------------------------------
latest_jpeg: bytes | None = None
latest_jpeg_seq = 0
latest_meta: dict[str, Any] = {
    "fps": 0.0,
    "inference_fps": 0.0,
    "inference_ms": 0.0,
    "activity": "warming",
    "activity_conf": 0.0,
    "id_cards": 0,
    "id_cards_current": 0,
    "id_cards_live": 0,
    "id_cards_detected_now": 0,
    "id_card_total": 0,
    "id_event_count": 0,
    "weapons": 0,
    "weapons_current": 0,
    "weapons_live": 0,
    "weapons_detected_now": 0,
    "weapon_total": 0,
    "weapon_event_count": 0,
    "weapon_list": [],
    "resolution": "--",
    "frame_age_ms": 0,
    "camera_ok": False,
    "reconnects": 0,
    "device": DEVICE_NAME,
    "models": {
        "activity": False,
        "id_card": False,
        "weapon": False,
        "coco_weapon": False,
        "roboflow_weapon": ROBOFLOW_ENABLED,
    },
}
state_lock = threading.Lock()

raw_frame = None
raw_frame_seq = 0
raw_frame_time = 0.0
raw_frame_lock = threading.Lock()

infer_meta: dict[str, Any] = {
    "activity": "normal",
    "activity_conf": 0.0,
    "id_cards": 0,
    "id_cards_current": 0,
    "id_cards_live": 0,
    "id_cards_detected_now": 0,
    "id_card_total": 0,
    "id_event_count": 0,
    "weapons": 0,
    "weapons_current": 0,
    "weapons_live": 0,
    "weapons_detected_now": 0,
    "weapon_total": 0,
    "weapon_event_count": 0,
    "weapon_list": [],
    "inference_fps": 0.0,
    "inference_ms": 0.0,
}
infer_dets: list[dict[str, Any]] = []
infer_lock = threading.Lock()

current_fps = 0.0
shutdown_event = threading.Event()


# -----------------------------------------------------------------------------
# Helper classes
# -----------------------------------------------------------------------------
class ActivitySmoother:
    def __init__(self) -> None:
        self._history = collections.deque(maxlen=SMOOTH_WINDOW)
        self._stable = ("normal", 0.0)

    def update(self, label: str, conf: float) -> tuple[str, float]:
        label = str(label or "normal")
        lower = normalize_activity_label(label)

        if is_alert_activity(lower) or lower == "weapon":
            self._stable = (label, conf)
            self._history.append((label, conf))
            return self._stable

        self._history.append((label, conf))
        counts = collections.Counter(
            normalize_activity_label(item[0])
            for item in self._history
        )

        if not counts:
            return self._stable

        top, count = counts.most_common(1)[0]
        if count >= SMOOTH_MIN_VOTE:
            avg = sum(
                value
                for current_label, value in self._history
                if normalize_activity_label(current_label) == top
            ) / count
            original = next(
                current_label
                for current_label, _ in self._history
                if normalize_activity_label(current_label) == top
            )
            self._stable = (original, round(avg, 2))

        return self._stable

    def force(self, label: str, conf: float) -> tuple[str, float]:
        self._stable = (label, conf)
        return self._stable


class InferenceCache:
    """Stores results from secondary models between scheduled executions."""

    def __init__(self) -> None:
        self.activity_label = "normal"
        self.activity_conf = 0.0
        self.activity_dets: list[dict[str, Any]] = []
        self.last_activity_time = 0.0

        self.id_count = 0
        self.id_dets: list[dict[str, Any]] = []
        self.last_id_time = 0.0
        self.id_candidate_box: list[int] | None = None
        self.id_candidate_hits = 0

        self.last_weapon_dets: list[dict[str, Any]] = []
        self.last_weapon_time = 0.0



class WeaponEventStore:
    """Store every distinct weapon appearance permanently.

    The same weapon is not counted on every video frame. One continuous
    appearance is one event. After the weapon is absent for the configured
    reset period, the next appearance becomes a new event.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.event_count = 0
        self.total_detections = 0
        self.events: list[dict[str, Any]] = []

        self._active = False
        self._last_seen = 0.0
        self._active_peak_count = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.event_count = max(0, int(data.get("event_count", 0)))
            self.total_detections = max(
                0,
                int(data.get("total_detections", 0)),
            )
            loaded_events = data.get("events", [])
            if isinstance(loaded_events, list):
                self.events = loaded_events[-WEAPON_EVENT_HISTORY_LIMIT:]

            log.info(
                "Loaded permanent weapon events: events=%d detections=%d",
                self.event_count,
                self.total_detections,
            )
        except Exception as exc:
            log.warning("Could not read %s: %s", self.path.name, exc)

    def _save_locked(self) -> None:
        payload = {
            "event_count": self.event_count,
            "total_detections": self.total_detections,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "events": self.events[-WEAPON_EVENT_HISTORY_LIMIT:],
        }

        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except Exception as exc:
            log.warning("Could not save weapon event file: %s", exc)
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass

    def update(self, weapon_list: list[dict[str, Any]]) -> dict[str, int]:
        current_count = len(weapon_list)
        now = time.monotonic()

        with self._lock:
            if current_count == 0:
                if (
                    self._active
                    and now - self._last_seen >= WEAPON_EVENT_RESET_SECONDS
                ):
                    self._active = False
                    self._active_peak_count = 0

                return self._snapshot_locked()

            self._last_seen = now

            if not self._active:
                self._active = True
                self._active_peak_count = current_count
                self.event_count += 1
                self.total_detections += current_count
                self._append_event_locked(weapon_list, current_count)
                self._save_locked()

                log.info(
                    "Permanent weapon event #%d stored (+%d weapon(s))",
                    self.event_count,
                    current_count,
                )

            elif current_count > self._active_peak_count:
                added = current_count - self._active_peak_count
                self._active_peak_count = current_count
                self.total_detections += added
                self._append_event_locked(
                    weapon_list,
                    added,
                    additional=True,
                )
                self._save_locked()

            return self._snapshot_locked()

    def _append_event_locked(
        self,
        weapon_list: list[dict[str, Any]],
        added_count: int,
        additional: bool = False,
    ) -> None:
        detections = []
        for item in weapon_list:
            detections.append(
                {
                    "label": str(item.get("label", "WEAPON")),
                    "confidence": round(
                        float(item.get("confidence", 0.0)),
                        3,
                    ),
                    "source": str(item.get("source", "")),
                    "original_label": str(
                        item.get("original_label", "")
                    ),
                }
            )

        self.events.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "event_number": self.event_count,
                "added_count": added_count,
                "additional_in_same_event": additional,
                "detections": detections,
            }
        )
        self.events = self.events[-WEAPON_EVENT_HISTORY_LIMIT:]

    def _snapshot_locked(self) -> dict[str, int]:
        return {
            "weapon_event_count": self.event_count,
            "weapon_total": self.total_detections,
        }

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return self._snapshot_locked()

    def recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(item)
                for item in self.events[-max(1, limit):]
            ]


weapon_event_store = WeaponEventStore(WEAPON_STORE_PATH)


class IdCardEventStore:
    """Permanently store distinct ID-card detection events.

    A card shown continuously is counted once. When no ID card has been
    detected for ID_EVENT_RESET_SECONDS, the next detection becomes a new
    event. This avoids increasing the dashboard count on every video frame.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.event_count = 0
        self.total_detections = 0
        self.events: list[dict[str, Any]] = []

        self._active = False
        self._last_seen = 0.0
        self._active_peak_count = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.event_count = max(0, int(data.get("event_count", 0)))
            self.total_detections = max(
                0,
                int(data.get("total_detections", 0)),
            )

            loaded_events = data.get("events", [])
            if isinstance(loaded_events, list):
                self.events = loaded_events[-ID_EVENT_HISTORY_LIMIT:]

            log.info(
                "Loaded permanent ID-card events: events=%d detections=%d",
                self.event_count,
                self.total_detections,
            )
        except Exception as exc:
            log.warning("Could not read %s: %s", self.path.name, exc)

    def _save_locked(self) -> None:
        payload = {
            "event_count": self.event_count,
            "total_detections": self.total_detections,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "events": self.events[-ID_EVENT_HISTORY_LIMIT:],
        }

        temporary = self.path.with_suffix(self.path.suffix + ".tmp")

        try:
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except Exception as exc:
            log.warning("Could not save ID-card event file: %s", exc)
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass

    def update(
        self,
        id_detections: list[dict[str, Any]],
    ) -> dict[str, int]:
        current_count = len(id_detections)
        now = time.monotonic()

        with self._lock:
            if current_count == 0:
                if (
                    self._active
                    and now - self._last_seen >= ID_EVENT_RESET_SECONDS
                ):
                    self._active = False
                    self._active_peak_count = 0

                return self._snapshot_locked()

            self._last_seen = now

            if not self._active:
                self._active = True
                self._active_peak_count = current_count
                self.event_count += 1
                self.total_detections += current_count
                self._append_event_locked(
                    id_detections,
                    current_count,
                )
                self._save_locked()

                log.info(
                    "Permanent ID-card event #%d stored (+%d card(s))",
                    self.event_count,
                    current_count,
                )

            elif current_count > self._active_peak_count:
                added = current_count - self._active_peak_count
                self._active_peak_count = current_count
                self.total_detections += added
                self._append_event_locked(
                    id_detections,
                    added,
                    additional=True,
                )
                self._save_locked()

            return self._snapshot_locked()

    def _append_event_locked(
        self,
        id_detections: list[dict[str, Any]],
        added_count: int,
        additional: bool = False,
    ) -> None:
        detections = []

        for item in id_detections:
            detections.append(
                {
                    "label": str(item.get("plain_label", "ID-CARD")),
                    "confidence": round(
                        float(item.get("conf", 0.0)),
                        3,
                    ),
                    "source": str(item.get("src", "id_card")),
                    "original_label": str(
                        item.get("original_label", "ID-card")
                    ),
                }
            )

        self.events.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "event_number": self.event_count,
                "added_count": added_count,
                "additional_in_same_event": additional,
                "detections": detections,
            }
        )
        self.events = self.events[-ID_EVENT_HISTORY_LIMIT:]

    def _snapshot_locked(self) -> dict[str, int]:
        return {
            "id_event_count": self.event_count,
            "id_card_total": self.total_detections,
        }

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return self._snapshot_locked()

    def recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(item)
                for item in self.events[-max(1, limit):]
            ]


id_card_event_store = IdCardEventStore(ID_STORE_PATH)


class RoboflowWorker:
    """Runs remote API requests separately so local inference never waits."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending_frame = None
        self._pending_seq = -1
        self._result_dets: list[dict[str, Any]] = []
        self._result_time = 0.0
        self._result_seq = -1
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not ROBOFLOW_ENABLED:
            return

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="roboflow",
        )
        self._thread.start()

    def submit(self, frame, frame_seq: int) -> None:
        if not ROBOFLOW_ENABLED:
            return

        # Keep only the newest pending request.
        with self._condition:
            self._pending_frame = frame.copy()
            self._pending_seq = frame_seq
            self._condition.notify()

    def get_recent(self, max_age: float) -> list[dict[str, Any]]:
        with self._condition:
            if time.monotonic() - self._result_time <= max_age:
                return [dict(item) for item in self._result_dets]
        return []

    def _run(self) -> None:
        while not shutdown_event.is_set():
            with self._condition:
                while self._pending_frame is None and not shutdown_event.is_set():
                    self._condition.wait(timeout=0.5)

                if shutdown_event.is_set():
                    return

                frame = self._pending_frame
                frame_seq = self._pending_seq
                self._pending_frame = None

            detections = call_roboflow_weapon_api(frame)

            with self._condition:
                self._result_dets = detections
                self._result_time = time.monotonic()
                self._result_seq = frame_seq


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def resolve_model_path(relative_path: str) -> Path:
    """Prefer the script directory, then preserve compatibility with CWD."""
    script_path = BASE_DIR / relative_path
    if script_path.exists():
        return script_path
    return Path(relative_path)


def model_class_names(model) -> list[str]:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return [str(value) for value in names.values()]
    if isinstance(names, (list, tuple)):
        return [str(value) for value in names]
    return []


def load_models() -> dict[str, Any]:
    models: dict[str, Any] = {}

    paths: dict[str, str] = {
        "id_card": "ml/models/id_card_yolov5/best.pt",
        "weapon": "ml/models/weapons_yolov8/best.pt",
    }

    if ENABLE_ACTIVITY_MODEL:
        activity_candidates = [
            ACTIVITY_MODEL_PATH,
            "ml/models/activity_yolov8/best.pt",
            "ml/models/activity/best.pt",
            "models/activity_yolov8/best.pt",
            "models/activity/best.pt",
            "activity_yolov8/best.pt",
        ]

        selected_activity_path = ACTIVITY_MODEL_PATH
        for candidate in activity_candidates:
            if candidate and resolve_model_path(candidate).exists():
                selected_activity_path = candidate
                break

        paths["activity"] = selected_activity_path

    if ENABLE_COCO_FALLBACK:
        paths["coco_weapon"] = "yolov8s.pt"

    try:
        from ultralytics import YOLO
    except Exception as exc:
        log.error("Ultralytics import failed: %s", exc)
        return models

    for key, relative_path in paths.items():
        path = resolve_model_path(relative_path)

        if not path.exists():
            log.warning("%-12s NOT FOUND: %s", key, path)
            continue

        try:
            model = YOLO(str(path))

            # Fusing Conv + BatchNorm can reduce inference overhead.
            try:
                model.fuse()
            except Exception:
                pass

            models[key] = model
            class_names = model_class_names(model)

            if key == "activity":
                # Treat unknown non-safe activity classes as alert candidates.
                for class_name in class_names:
                    normalized = normalize_activity_label(class_name)
                    if normalized and not is_safe_activity(normalized):
                        ACTIVITY_ALERT.add(normalized)

            log.info(
                "%-12s LOADED: %s classes=%s",
                key,
                path,
                class_names,
            )
        except Exception as exc:
            log.error("%-12s FAILED: %s - %s", key, path, exc)

    return models


# -----------------------------------------------------------------------------
# Detection helpers
# -----------------------------------------------------------------------------
def normalize_weapon_label(original_label: str, confidence: float) -> str:
    del confidence  # Kept in the signature for compatibility and future rules.
    label = str(original_label or "").lower().strip()

    if label in PISTOL_LABELS:
        return "PISTOL"
    if label in KNIFE_LABELS:
        return "KNIFE"
    if label in RIFLE_LABELS:
        return "RIFLE"
    return "WEAPON"


def box_area_ratio(box: list[int], frame_w: int, frame_h: int) -> float:
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return area / float(max(frame_w * frame_h, 1))


def valid_weapon_box(box: list[int], frame_w: int, frame_h: int) -> bool:
    if len(box) != 4:
        return False

    x1, y1, x2, y2 = box
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)

    area_ratio = box_area_ratio(box, frame_w, frame_h)
    width_ratio = box_w / float(max(frame_w, 1))
    height_ratio = box_h / float(max(frame_h, 1))

    return (
        WEAPON_MIN_AREA <= area_ratio <= WEAPON_MAX_AREA
        and width_ratio <= WEAPON_MAX_WIDTH_RATIO
        and height_ratio <= WEAPON_MAX_HEIGHT_RATIO
    )


def iou(box_a: list[int], box_b: list[int]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def remove_duplicate_boxes(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    sorted_dets = sorted(
        detections,
        key=lambda item: float(item.get("conf", 0.0)),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []

    for detection in sorted_dets:
        box = detection.get("box", [])
        if len(box) != 4:
            continue

        if any(
            iou(box, kept_detection.get("box", [])) >= iou_threshold
            for kept_detection in kept
        ):
            continue

        kept.append(detection)

    return kept


def yolo_predict(model, frame, imgsz: int, confidence: float):
    """Run Ultralytics on the original frame without manual resize/scaling."""
    results = model.predict(
        source=frame,
        imgsz=imgsz,
        conf=confidence,
        device=DEVICE,
        half=USE_HALF,
        max_det=MAX_DETECTIONS,
        verbose=False,
    )
    return results[0]


def _detect_weapon_with_model(
    model,
    frame,
    size: int,
    conf: float,
    source: str,
    allowed_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    frame_h, frame_w = frame.shape[:2]

    result = yolo_predict(model, frame, size, conf)
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        score = float(box.conf[0])
        class_id = int(box.cls[0])
        original_label = str(result.names[class_id])
        lower_label = original_label.lower().strip()

        if allowed_labels is not None and lower_label not in allowed_labels:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        scaled_box = [int(x1), int(y1), int(x2), int(y2)]

        if not valid_weapon_box(scaled_box, frame_w, frame_h):
            continue

        display_label = normalize_weapon_label(original_label, score)
        detections.append(
            {
                "box": scaled_box,
                "label": f"{display_label} {score:.0%}",
                "plain_label": display_label,
                "original_label": original_label,
                "conf": score,
                "color": (0, 0, 255),
                "src": source,
            }
        )

    return remove_duplicate_boxes(detections)


def _weapon_list_from_detections(
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    weapon_list: list[dict[str, Any]] = []

    for detection in detections:
        source = str(detection.get("src", ""))
        if not source.startswith("weapon") and source != "coco_weapon":
            continue

        weapon_list.append(
            {
                "label": str(detection.get("plain_label", "WEAPON")),
                "confidence": round(float(detection.get("conf", 0.0)), 2),
                "box": detection.get("box", []),
                "source": source,
                "original_label": detection.get("original_label", ""),
            }
        )

    return weapon_list


# -----------------------------------------------------------------------------
# Roboflow API
# -----------------------------------------------------------------------------
def call_roboflow_weapon_api(frame) -> list[dict[str, Any]]:
    if not ROBOFLOW_ENABLED or frame is None:
        return []

    try:
        original_h, original_w = frame.shape[:2]
        api_size = 416
        small = cv2.resize(frame, (api_size, api_size))

        ok, buffer = cv2.imencode(
            ".jpg",
            small,
            [cv2.IMWRITE_JPEG_QUALITY, 72],
        )
        if not ok:
            return []

        encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
        url = (
            f"{ROBOFLOW_URL}"
            f"?api_key={urllib.parse.quote(ROBOFLOW_API_KEY)}"
            f"&confidence={int(ROBOFLOW_CONF * 100)}"
            "&overlap=30"
        )

        request = urllib.request.Request(
            url,
            data=encoded.encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=3) as response:
            result = json.loads(response.read().decode("utf-8"))

        scale_x = original_w / api_size
        scale_y = original_h / api_size
        detections: list[dict[str, Any]] = []

        for prediction in result.get("predictions", []):
            confidence = float(prediction.get("confidence", 0.0))
            original_label = str(prediction.get("class", "weapon"))
            display_label = normalize_weapon_label(original_label, confidence)

            center_x = float(prediction.get("x", 0.0))
            center_y = float(prediction.get("y", 0.0))
            width = float(prediction.get("width", 0.0))
            height = float(prediction.get("height", 0.0))

            box = [
                int((center_x - width / 2) * scale_x),
                int((center_y - height / 2) * scale_y),
                int((center_x + width / 2) * scale_x),
                int((center_y + height / 2) * scale_y),
            ]

            if not valid_weapon_box(box, original_w, original_h):
                continue

            detections.append(
                {
                    "box": box,
                    "label": f"{display_label} {confidence:.0%}",
                    "plain_label": display_label,
                    "original_label": original_label,
                    "conf": confidence,
                    "color": (0, 0, 255),
                    "src": "weapon_api",
                }
            )

        return remove_duplicate_boxes(detections)

    except Exception as exc:
        log.debug("Roboflow API error: %s", exc)
        return []


# -----------------------------------------------------------------------------
# Secondary model inference
# -----------------------------------------------------------------------------
def run_activity_model(
    frame,
    model,
    smoother: ActivitySmoother,
) -> tuple[str, float, list[dict[str, Any]]]:
    """Run either an activity detection model or classification model."""
    detections: list[dict[str, Any]] = []
    raw_label = "normal"
    raw_conf = 0.0

    result = yolo_predict(model, frame, ACTIVITY_IMGSZ, ACTIVITY_CONF)
    boxes = getattr(result, "boxes", None)

    # Object-detection activity model.
    if boxes is not None and len(boxes) > 0:
        best = max(boxes, key=lambda item: float(item.conf[0]))
        class_id = int(best.cls[0])
        raw_label = str(result.names[class_id])
        raw_conf = float(best.conf[0])

        if raw_conf >= ACTIVITY_CONF:
            x1, y1, x2, y2 = best.xyxy[0].tolist()
            alert = is_alert_activity(raw_label)

            # Show an activity box for all meaningful non-normal activities.
            if normalize_activity_label(raw_label) not in {"", "normal"}:
                detections.append(
                    {
                        "box": [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                        ],
                        "label": f"{raw_label.upper()} {raw_conf:.0%}",
                        "plain_label": raw_label,
                        "original_label": raw_label,
                        "conf": raw_conf,
                        "color": (
                            (0, 0, 255)
                            if alert
                            else (0, 180, 80)
                        ),
                        "src": "activity",
                    }
                )

    # Classification activity model.
    else:
        probabilities = getattr(result, "probs", None)
        if probabilities is not None:
            try:
                class_id = int(probabilities.top1)
                raw_conf = float(probabilities.top1conf)
                raw_label = str(result.names[class_id])
            except Exception:
                raw_label = "normal"
                raw_conf = 0.0

    stable_label, stable_conf = smoother.update(raw_label, raw_conf)
    return stable_label, stable_conf, detections


def _id_crop_features(frame, box: list[int]) -> tuple[float, float, float]:
    """Return rectangularity, edge density, and bright-pixel ratio."""
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return 0.0, 0.0, 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)

    crop_area = float(max(gray.shape[0] * gray.shape[1], 1))
    edge_density = float((edges > 0).sum()) / crop_area
    bright_ratio = float((gray > 145).sum()) / crop_area

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best_rectangularity = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < crop_area * 0.08:
            continue

        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]
        rect_area = float(max(rect_w * rect_h, 1.0))
        rectangularity = min(1.0, area / rect_area)
        best_rectangularity = max(best_rectangularity, rectangularity)

    return best_rectangularity, edge_density, bright_ratio


def valid_id_card_box(
    frame,
    box: list[int],
    frame_width: int,
    frame_height: int,
) -> bool:
    """Reject body/hand boxes and keep card-shaped textured rectangles."""
    if len(box) != 4:
        return False

    x1, y1, x2, y2 = box
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)

    if box_width < 35 or box_height < 24:
        return False

    frame_area = max(frame_width * frame_height, 1)
    area = box_width * box_height
    area_ratio = area / float(frame_area)
    ratio = box_width / float(max(box_height, 1))

    if area < ID_AREA_MIN:
        return False
    if area_ratio > ID_AREA_MAX_RATIO:
        return False
    if not (ID_RATIO_MIN <= ratio <= ID_RATIO_MAX):
        return False
    if box_width / float(max(frame_width, 1)) > ID_MAX_WIDTH_RATIO:
        return False
    if box_height / float(max(frame_height, 1)) > ID_MAX_HEIGHT_RATIO:
        return False

    margin_x = max(4, int(frame_width * 0.015))
    margin_y = max(4, int(frame_height * 0.015))
    touching_edges = sum(
        (
            x1 <= margin_x,
            y1 <= margin_y,
            x2 >= frame_width - margin_x,
            y2 >= frame_height - margin_y,
        )
    )
    if touching_edges >= 2:
        return False

    rectangularity, edge_density, bright_ratio = _id_crop_features(
        frame,
        box,
    )

    # Cards contain text/photo edges and usually a substantial light region.
    if not (ID_MIN_EDGE_DENSITY <= edge_density <= ID_MAX_EDGE_DENSITY):
        return False
    if bright_ratio < 0.18:
        return False

    # A loose YOLO box may include fingers, so rectangularity is a soft but
    # useful filter rather than an extremely strict requirement.
    if rectangularity < ID_MIN_RECTANGULARITY:
        return False

    return True


def find_id_card_contour(frame) -> dict[str, Any] | None:
    """Classical rectangle fallback for a visible landscape ID card.

    This is used only when YOLO does not produce a valid card box. It searches
    the whole frame for a bright, textured, rectangular card-like contour.
    """
    frame_height, frame_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 55, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    frame_area = float(max(frame_width * frame_height, 1))
    candidates: list[tuple[float, list[int]]] = []

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        area_ratio = contour_area / frame_area
        if not (0.006 <= area_ratio <= ID_AREA_MAX_RATIO):
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]
        if rect_w < 1 or rect_h < 1:
            continue

        long_side = max(rect_w, rect_h)
        short_side = min(rect_w, rect_h)
        rotated_ratio = long_side / max(short_side, 1.0)
        if not (ID_RATIO_MIN <= rotated_ratio <= ID_RATIO_MAX):
            continue

        x, y, width, height = cv2.boundingRect(approx)
        box = [
            max(0, x),
            max(0, y),
            min(frame_width - 1, x + width),
            min(frame_height - 1, y + height),
        ]

        box_ratio = width / float(max(height, 1))
        # Bounding boxes can become portrait when the card is strongly tilted;
        # still reject very tall person/arm-shaped regions.
        if box_ratio < 0.85 or box_ratio > 2.60:
            continue

        rectangularity, edge_density, bright_ratio = _id_crop_features(
            frame,
            box,
        )
        if rectangularity < 0.55:
            continue
        if not (ID_MIN_EDGE_DENSITY <= edge_density <= ID_MAX_EDGE_DENSITY):
            continue
        if bright_ratio < 0.24:
            continue

        score = (
            rectangularity * 0.45
            + min(edge_density / 0.16, 1.0) * 0.25
            + min(bright_ratio / 0.60, 1.0) * 0.20
            + min(area_ratio / 0.06, 1.0) * 0.10
        )
        candidates.append((score, box))

    if not candidates:
        return None

    score, best_box = max(candidates, key=lambda item: item[0])
    confidence = max(0.45, min(0.85, score))

    return {
        "box": best_box,
        "label": f"ID-CARD {confidence:.0%}",
        "plain_label": "ID-CARD",
        "original_label": "card-contour",
        "conf": confidence,
        "color": (0, 165, 255),
        "src": "id_card",
    }


def run_id_card_model(frame, model) -> tuple[int, list[dict[str, Any]]]:
    detections: list[dict[str, Any]] = []
    frame_height, frame_width = frame.shape[:2]

    result = yolo_predict(model, frame, ID_IMGSZ, ID_CONF)
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            original_label = str(result.names[class_id])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            candidate_box = [
                max(0, int(x1)),
                max(0, int(y1)),
                min(frame_width - 1, int(x2)),
                min(frame_height - 1, int(y2)),
            ]

            if not valid_id_card_box(
                frame,
                candidate_box,
                frame_width,
                frame_height,
            ):
                continue

            detections.append(
                {
                    "box": candidate_box,
                    "label": f"ID-CARD {confidence:.0%}",
                    "plain_label": "ID-CARD",
                    "original_label": original_label,
                    "conf": confidence,
                    "color": (0, 165, 255),
                    "src": "id_card",
                }
            )

    detections = remove_duplicate_boxes(detections, iou_threshold=0.35)

    # This project expects one card to be shown at a time. Keeping only the
    # strongest valid box prevents two false ID boxes on the same person.
    if detections:
        best = max(detections, key=lambda item: float(item.get("conf", 0.0)))
        return 1, [best]

    if ID_CONTOUR_FALLBACK:
        contour_detection = find_id_card_contour(frame)
        if contour_detection is not None:
            return 1, [contour_detection]

    return 0, []


# -----------------------------------------------------------------------------
# Main inference pipeline
# -----------------------------------------------------------------------------
def run_inference(
    frame,
    frame_seq: int,
    models: dict[str, Any],
    smoother: ActivitySmoother,
    cache: InferenceCache,
    cycle: int,
    roboflow_worker: RoboflowWorker,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    now = time.monotonic()

    # Run activity regularly and reuse the latest result between runs.
    if "activity" in models and (
        cycle == 1 or cycle % ACTIVITY_MODEL_EVERY == 0
    ):
        try:
            (
                new_activity_label,
                new_activity_conf,
                new_activity_dets,
            ) = run_activity_model(frame, models["activity"], smoother)

            cache.activity_label = new_activity_label
            cache.activity_conf = new_activity_conf

            if new_activity_dets:
                cache.activity_dets = new_activity_dets
                cache.last_activity_time = now
            elif now - cache.last_activity_time > ACTIVITY_HOLD_SECONDS:
                cache.activity_dets = []
        except Exception as exc:
            log.debug("Activity model: %s", exc)

    # Run ID-card detection frequently enough to detect a small card displayed
    # on a phone. Keep the last valid result briefly to avoid box flicker.
    if "id_card" in models and (cycle == 1 or cycle % ID_MODEL_EVERY == 0):
        try:
            new_id_count, new_id_dets = run_id_card_model(
                frame,
                models["id_card"],
            )

            if new_id_count > 0 and new_id_dets:
                best_det = max(
                    new_id_dets,
                    key=lambda item: float(item.get("conf", 0.0)),
                )
                best_box = list(best_det.get("box", []))

                same_candidate = (
                    cache.id_candidate_box is not None
                    and len(best_box) == 4
                    and iou(cache.id_candidate_box, best_box) >= 0.25
                )

                if same_candidate:
                    cache.id_candidate_hits += 1
                else:
                    cache.id_candidate_box = best_box
                    cache.id_candidate_hits = 1

                if cache.id_candidate_hits >= ID_CONFIRM_HITS:
                    cache.id_count = 1
                    cache.id_dets = [best_det]
                    cache.last_id_time = now
            else:
                cache.id_candidate_box = None
                cache.id_candidate_hits = 0

                if now - cache.last_id_time > ID_HOLD_SECONDS:
                    cache.id_count = 0
                    cache.id_dets = []
        except Exception as exc:
            log.debug("ID-card model: %s", exc)

    detections = list(cache.activity_dets) + list(cache.id_dets)

    # Highest priority: custom local weapon model on every newest frame.
    weapon_dets: list[dict[str, Any]] = []
    if "weapon" in models:
        try:
            weapon_dets = _detect_weapon_with_model(
                models["weapon"],
                frame,
                size=WEAPON_IMGSZ,
                conf=WEAPON_CONF,
                source="weapon",
                allowed_labels=WEAPON_MODEL_LABELS,
            )
        except Exception as exc:
            log.debug("Weapon local model: %s", exc)

    # COCO is larger/slower, so run it occasionally only after a local miss.
    if (
        not weapon_dets
        and "coco_weapon" in models
        and (cycle == 1 or cycle % COCO_FALLBACK_EVERY == 0)
    ):
        try:
            weapon_dets = _detect_weapon_with_model(
                models["coco_weapon"],
                frame,
                size=COCO_IMGSZ,
                conf=COCO_WEAPON_CONF,
                source="coco_weapon",
                allowed_labels=COCO_WEAPON_LABELS,
            )
        except Exception as exc:
            log.debug("COCO weapon fallback: %s", exc)

    # Submit API work without blocking this inference cycle.
    if (
        not weapon_dets
        and ROBOFLOW_ENABLED
        and cycle % WEAPON_API_EVERY == 0
    ):
        roboflow_worker.submit(frame, frame_seq)

    # Use a recent asynchronous API result only when local models missed.
    if not weapon_dets and ROBOFLOW_ENABLED:
        weapon_dets = roboflow_worker.get_recent(ROBOFLOW_RESULT_MAX_AGE)

    # Hold the last weapon box very briefly to reduce one-frame flicker.
    if weapon_dets:
        cache.last_weapon_dets = weapon_dets
        cache.last_weapon_time = now
    elif now - cache.last_weapon_time <= WEAPON_HOLD_SECONDS:
        weapon_dets = [dict(item) for item in cache.last_weapon_dets]
    else:
        cache.last_weapon_dets = []

    meta: dict[str, Any] = {
        "activity": cache.activity_label,
        "activity_conf": cache.activity_conf,
        "id_cards": cache.id_count,
        "weapons": 0,
        "weapon_list": [],
    }

    if weapon_dets:
        # Avoid showing an activity warning box together with a weapon box.
        detections = [
            item for item in detections if item.get("src") != "activity"
        ]
        detections.extend(weapon_dets)

        weapon_list = _weapon_list_from_detections(weapon_dets)
        meta["weapon_list"] = weapon_list
        meta["weapons"] = len(weapon_list)

        best_detection = max(
            weapon_dets,
            key=lambda item: float(item.get("conf", 0.0)),
        )
        best_conf = float(best_detection.get("conf", 0.0))
        best_label = str(best_detection.get("plain_label", "WEAPON"))

        meta["activity"] = best_label.lower()
        meta["activity_conf"] = round(best_conf, 2)
        smoother.force(best_label.lower(), best_conf)

    return meta, detections


# -----------------------------------------------------------------------------
# Overlay drawing
# -----------------------------------------------------------------------------
def draw_overlays(
    frame,
    meta: dict[str, Any],
    detections: list[dict[str, Any]],
    fps: float,
):
    frame_h, frame_w = frame.shape[:2]
    activity = str(meta.get("activity", "normal")).lower()
    confidence = float(meta.get("activity_conf", 0.0))
    # Use only the current live count for the red camera warning.
    # `weapons` is the permanently stored event count used by the dashboard.
    weapon_count = int(
        meta.get(
            "weapons_live",
            meta.get("weapons_detected_now", 0),
        )
    )

    alert = (
        weapon_count > 0
        or is_alert_activity(activity)
        or activity in {"weapon", "pistol", "knife", "rifle"}
    )

    cv2.rectangle(
        frame,
        (0, 0),
        (frame_w, 44),
        (0, 0, 180) if alert else (0, 80, 0),
        -1,
    )

    if weapon_count > 0:
        best_label = "WEAPON"
        weapon_list = meta.get("weapon_list", [])
        if weapon_list:
            best_weapon = max(
                weapon_list,
                key=lambda item: float(item.get("confidence", 0.0)),
            )
            best_label = str(best_weapon.get("label", "WEAPON")).upper()

        top_text = f"!! {best_label} DETECTED !!  {confidence:.0%}"
        top_color = (30, 30, 255)
    elif is_alert_activity(activity):
        top_text = f"{activity.upper()}  {confidence:.0%}"
        top_color = (30, 30, 255)
    else:
        safe_label = activity.upper() if activity else "NORMAL"
        top_text = f"{safe_label}  {confidence:.0%}"
        top_color = (0, 255, 120)

    cv2.putText(
        frame,
        top_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.88,
        top_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS:{fps:.1f}",
        (max(8, frame_w - 115), 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    for detection in detections:
        box = detection.get("box", [])
        if len(box) != 4:
            continue

        x1, y1, x2, y2 = [int(value) for value in box]
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))

        color = tuple(detection.get("color", (0, 0, 255)))
        label = str(detection.get("label", "DETECTED"))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            1,
        )
        label_top = max(44, y1 - text_height - baseline - 6)
        label_right = min(frame_w - 1, x1 + text_width + 8)

        cv2.rectangle(
            frame,
            (x1, label_top),
            (label_right, label_top + text_height + baseline + 6),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_top + text_height + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
        (8, frame_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            f"ID:{meta.get('id_cards_live', 0)}  "
            f"WPN:{meta.get('weapons_live', 0)}"
        ),
        (max(8, frame_w - 180), frame_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    return frame


# -----------------------------------------------------------------------------
# Camera handling
# -----------------------------------------------------------------------------
def configure_camera(cap) -> None:
    # MJPG often gives smoother Windows webcam capture with less buffering.
    try:
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
    except Exception:
        pass

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def open_camera():
    backends = [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_ANY, "ANY"),
    ]

    for index in range(CAMERA_INDEX_MAX):
        for backend, backend_name in backends:
            cap = None

            try:
                cap = cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue

                configure_camera(cap)
                time.sleep(0.25)

                # Discard startup frames and verify that the camera works.
                for _ in range(12):
                    success, frame = cap.read()
                    if success and frame is not None and frame.size > 0:
                        log.info(
                            "Camera: idx=%d %s %dx%d",
                            index,
                            backend_name,
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        )
                        return cap
                    time.sleep(0.04)

                cap.release()
            except Exception as exc:
                log.debug(
                    "Camera open idx=%d backend=%s: %s",
                    index,
                    backend_name,
                    exc,
                )
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

    return None


# -----------------------------------------------------------------------------
# Worker threads
# -----------------------------------------------------------------------------
def capture_thread(cap_holder: list[Any]) -> None:
    global raw_frame, raw_frame_seq, raw_frame_time

    reconnects = 0

    try:
        while not shutdown_event.is_set():
            cap = cap_holder[0]

            if cap is None:
                log.info("Opening camera attempt %d...", reconnects + 1)
                cap = open_camera()
                cap_holder[0] = cap

                if cap is None:
                    reconnects += 1
                    with state_lock:
                        latest_meta["camera_ok"] = False
                        latest_meta["reconnects"] = reconnects
                    shutdown_event.wait(2.0)
                    continue

            success, frame = cap.read()

            if not success or frame is None or frame.size == 0:
                log.warning("Camera read failed. Reopening camera.")
                try:
                    cap.release()
                except Exception:
                    pass

                cap_holder[0] = None
                reconnects += 1
                with state_lock:
                    latest_meta["camera_ok"] = False
                    latest_meta["reconnects"] = reconnects
                shutdown_event.wait(0.5)
                continue

            # Store only the latest frame. Old frames are intentionally dropped.
            with raw_frame_lock:
                raw_frame = frame
                raw_frame_seq += 1
                raw_frame_time = time.monotonic()

    except Exception:
        log.exception("Capture thread crashed")


def inference_thread(
    models: dict[str, Any],
    model_status: dict[str, bool],
    roboflow_worker: RoboflowWorker,
) -> None:
    global infer_meta, infer_dets

    smoother = ActivitySmoother()
    cache = InferenceCache()
    last_processed_seq = -1
    cycle = 0

    with state_lock:
        latest_meta["models"] = dict(model_status)
        latest_meta["device"] = DEVICE_NAME

    while not shutdown_event.is_set():
        with raw_frame_lock:
            frame_seq = raw_frame_seq
            frame = (
                None
                if raw_frame is None or frame_seq == last_processed_seq
                else raw_frame.copy()
            )

        # There is no fixed inference delay. Wait only for a newer camera frame.
        if frame is None:
            shutdown_event.wait(0.001)
            continue

        last_processed_seq = frame_seq
        cycle += 1
        start = time.perf_counter()

        try:
            meta, detections = run_inference(
                frame=frame,
                frame_seq=frame_seq,
                models=models,
                smoother=smoother,
                cache=cache,
                cycle=cycle,
                roboflow_worker=roboflow_worker,
            )

            # run_inference() returns live counts for the current frame.
            live_id_count = int(meta.get("id_cards", 0))
            live_weapon_count = int(meta.get("weapons", 0))

            id_detections = [
                item
                for item in detections
                if str(item.get("src", "")) == "id_card"
            ]

            stored_ids = id_card_event_store.update(id_detections)
            id_event_count = int(
                stored_ids.get("id_event_count", 0)
            )
            id_card_total = int(
                stored_ids.get("id_card_total", 0)
            )

            stored_weapons = weapon_event_store.update(
                list(meta.get("weapon_list", []))
            )
            weapon_event_count = int(
                stored_weapons.get("weapon_event_count", 0)
            )
            weapon_total = int(
                stored_weapons.get("weapon_total", 0)
            )

            # Existing dashboard JavaScript reads `id_cards` and `weapons`.
            # Return permanent event counts through those fields.
            meta["id_cards_live"] = live_id_count
            meta["id_cards_detected_now"] = live_id_count
            meta["id_cards"] = id_event_count
            meta["id_cards_current"] = id_event_count
            meta["id_event_count"] = id_event_count
            meta["id_card_total"] = id_card_total

            meta["weapons_live"] = live_weapon_count
            meta["weapons_detected_now"] = live_weapon_count
            meta["weapons"] = weapon_event_count
            meta["weapons_current"] = weapon_event_count
            meta["weapon_event_count"] = weapon_event_count
            meta["weapon_total"] = weapon_total

            elapsed = time.perf_counter() - start
            meta["inference_ms"] = round(elapsed * 1000.0, 1)
            meta["inference_fps"] = round(1.0 / elapsed, 1) if elapsed > 0 else 0.0

            with infer_lock:
                infer_meta = meta
                infer_dets = detections

            with state_lock:
                latest_meta["models"] = dict(model_status)
                latest_meta["inference_ms"] = meta["inference_ms"]
                latest_meta["inference_fps"] = meta["inference_fps"]

        except Exception:
            log.exception("Inference cycle failed")


def encode_thread(cap_holder: list[Any]) -> None:
    global latest_jpeg, latest_jpeg_seq, latest_meta, current_fps

    last_encoded_seq = -1
    fps_counter = 0
    fps_timer = time.monotonic()
    frame_interval = 1.0 / max(STREAM_FPS, 1)
    next_encode_time = 0.0

    while not shutdown_event.is_set():
        now = time.monotonic()
        if now < next_encode_time:
            shutdown_event.wait(min(0.005, next_encode_time - now))
            continue

        with raw_frame_lock:
            frame_seq = raw_frame_seq
            captured_at = raw_frame_time
            frame = (
                None
                if raw_frame is None or frame_seq == last_encoded_seq
                else raw_frame.copy()
            )

        if frame is None:
            shutdown_event.wait(0.003)
            continue

        last_encoded_seq = frame_seq
        next_encode_time = now + frame_interval

        fps_counter += 1
        if now - fps_timer >= 1.0:
            current_fps = fps_counter / max(now - fps_timer, 1e-6)
            fps_counter = 0
            fps_timer = now

        with infer_lock:
            meta = dict(infer_meta)
            detections = [dict(item) for item in infer_dets]

        display = draw_overlays(frame, meta, detections, current_fps)
        success, jpeg = cv2.imencode(
            ".jpg",
            display,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )

        if not success:
            continue

        cap = cap_holder[0]
        actual_width = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap else display.shape[1]
        )
        actual_height = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap else display.shape[0]
        )
        frame_age_ms = max(0, int((time.monotonic() - captured_at) * 1000))

        with state_lock:
            latest_jpeg = jpeg.tobytes()
            latest_jpeg_seq = frame_seq
            latest_meta.update(
                {
                    **meta,
                    "fps": round(current_fps, 1),
                    "resolution": f"{actual_width}x{actual_height}",
                    "frame_age_ms": frame_age_ms,
                    "camera_ok": True,
                }
            )


# -----------------------------------------------------------------------------
# HTTP server
# -----------------------------------------------------------------------------
BOUNDARY = b"--frame"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]

        if path in ("/stream.mjpg", "/stream.mjpeg", "/"):
            self._stream()
        elif path == "/status":
            self._status()
        elif path in ("/healthz", "/health"):
            self._healthz()
        elif path == "/snapshot.jpg":
            self._snapshot()
        elif path == "/weapon-events":
            self._weapon_events()
        elif path == "/id-events":
            self._id_events()
        else:
            self.send_error(404)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        last_sent_seq = -1

        try:
            while not shutdown_event.is_set():
                with state_lock:
                    frame = latest_jpeg
                    frame_seq = latest_jpeg_seq

                if (
                    frame is not None
                    and len(frame) > 1000
                    and frame_seq != last_sent_seq
                ):
                    self.wfile.write(
                        BOUNDARY
                        + b"\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                        + frame
                        + b"\r\n"
                    )
                    self.wfile.flush()
                    last_sent_seq = frame_seq
                else:
                    shutdown_event.wait(0.005)

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _status(self) -> None:
        with state_lock:
            data = dict(latest_meta)
            data["models"] = dict(latest_meta.get("models", {}))
            data["weapon_list"] = list(latest_meta.get("weapon_list", []))

        stored_ids = id_card_event_store.snapshot()
        live_id_count = int(
            data.get(
                "id_cards_live",
                data.get("id_cards_detected_now", 0),
            )
        )
        id_event_count = int(
            stored_ids.get("id_event_count", 0)
        )
        id_card_total = int(
            stored_ids.get("id_card_total", 0)
        )

        stored_weapons = weapon_event_store.snapshot()
        live_weapon_count = int(
            data.get(
                "weapons_live",
                data.get("weapons_detected_now", 0),
            )
        )
        weapon_event_count = int(
            stored_weapons.get("weapon_event_count", 0)
        )
        weapon_total = int(
            stored_weapons.get("weapon_total", 0)
        )

        data["id_cards_live"] = live_id_count
        data["id_cards_detected_now"] = live_id_count
        data["id_cards"] = id_event_count
        data["id_cards_current"] = id_event_count
        data["id_event_count"] = id_event_count
        data["id_card_total"] = id_card_total

        data["weapons_live"] = live_weapon_count
        data["weapons_detected_now"] = live_weapon_count
        data["weapons"] = weapon_event_count
        data["weapons_current"] = weapon_event_count
        data["weapon_event_count"] = weapon_event_count
        data["weapon_total"] = weapon_total

        body = json.dumps(data).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


    def _weapon_events(self) -> None:
        stored = weapon_event_store.snapshot()
        body = json.dumps(
            {
                **stored,
                "events": weapon_event_store.recent_events(100),
            },
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


    def _id_events(self) -> None:
        stored = id_card_event_store.snapshot()
        body = json.dumps(
            {
                **stored,
                "events": id_card_event_store.recent_events(100),
            },
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _healthz(self) -> None:
        with state_lock:
            ready = latest_jpeg is not None

        body = b"ok" if ready else b"warming"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _snapshot(self) -> None:
        with state_lock:
            frame = latest_jpeg

        if frame is None:
            self.send_error(503, "Camera is warming up")
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(frame)

    def log_message(self, *args: Any) -> None:
        # Silence the default per-request console logging.
        return


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    log.info("Inference device: %s", DEVICE_NAME)
    log.info(
        "Fast mode: weapon imgsz=%d, ID every=%d, COCO every=%d",
        WEAPON_IMGSZ,
        ID_MODEL_EVERY,
        COCO_FALLBACK_EVERY,
    )

    models = load_models()
    model_status = {
        "activity": "activity" in models,
        "id_card": "id_card" in models,
        "weapon": (
            "weapon" in models
            or "coco_weapon" in models
            or ROBOFLOW_ENABLED
        ),
        "coco_weapon": "coco_weapon" in models,
        "roboflow_weapon": ROBOFLOW_ENABLED,
    }
    log.info("Models: %s", model_status)
    log.info(
        "ID settings: imgsz=%d conf=%.2f ratio=%.2f..%.2f confirm=%d",
        ID_IMGSZ,
        ID_CONF,
        ID_RATIO_MIN,
        ID_RATIO_MAX,
        ID_CONFIRM_HITS,
    )
    log.info(
        "Activity settings: enabled=%s path=%s imgsz=%d conf=%.2f every=%d",
        ENABLE_ACTIVITY_MODEL,
        ACTIVITY_MODEL_PATH,
        ACTIVITY_IMGSZ,
        ACTIVITY_CONF,
        ACTIVITY_MODEL_EVERY,
    )
    log.info(
        "Permanent counts: ID=%d, weapon=%d",
        id_card_event_store.snapshot().get("id_event_count", 0),
        weapon_event_store.snapshot().get("weapon_event_count", 0),
    )

    cap_holder: list[Any] = [None]
    roboflow_worker = RoboflowWorker()
    roboflow_worker.start()

    capture_worker = threading.Thread(
        target=capture_thread,
        args=(cap_holder,),
        daemon=True,
        name="capture",
    )
    inference_worker = threading.Thread(
        target=inference_thread,
        args=(models, model_status, roboflow_worker),
        daemon=True,
        name="inference",
    )
    encode_worker = threading.Thread(
        target=encode_thread,
        args=(cap_holder,),
        daemon=True,
        name="encode",
    )

    capture_worker.start()
    inference_worker.start()
    encode_worker.start()

    server = ReusableThreadingHTTPServer(("0.0.0.0", 8765), Handler)
    server_thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="http",
    )
    server_thread.start()

    log.info("HTTP server: http://localhost:8765")
    log.info("Stream     : http://localhost:8765/stream.mjpg")
    log.info("Status     : http://localhost:8765/status")
    log.info("Health     : http://localhost:8765/healthz")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopping camera server...")
    finally:
        shutdown_event.set()
        server.shutdown()
        server.server_close()

        cap = cap_holder[0]
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


if __name__ == "__main__":
    main()
