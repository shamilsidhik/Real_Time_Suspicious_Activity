from __future__ import annotations

import copy
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


@dataclass
class OverlayState:
    activity_label: str = "warming_up"
    activity_confidence: float = 0.0
    spoof_status: str = "DISABLED"
    spoof_confidence: float = 0.0
    id_detections: list[dict[str, Any]] = field(default_factory=list)
    weapon_detections: list[dict[str, Any]] = field(default_factory=list)
    updated_at: float = 0.0


class FPSMeter:
    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self._last_t: float | None = None
        self._fps: float = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last_t is not None:
            dt = max(now - self._last_t, 1e-6)
            instant = 1.0 / dt
            if self._fps <= 0:
                self._fps = instant
            else:
                self._fps = (self.smoothing * self._fps) + ((1.0 - self.smoothing) * instant)
        self._last_t = now
        return self._fps


class CameraService:
    def __init__(self) -> None:
        self.camera_index = _env_int("CAMERA_INDEX", 0)
        self.api_preference = cv2.CAP_DSHOW
        self.capture_width = _env_int("CAMERA_WIDTH", 640)
        self.capture_height = _env_int("CAMERA_HEIGHT", 480)
        self.target_fps = _env_float("CAMERA_TARGET_FPS", 18.0)
        self.jpeg_quality = _env_int("MJPEG_JPEG_QUALITY", 80)
        self.object_every_n = max(_env_int("OBJECT_DETECT_EVERY_N", 2), 1)
        self.activity_every_n = max(_env_int("ACTIVITY_DETECT_EVERY_N", 4), 1)
        self.reconnect_delay = _env_float("CAMERA_RECONNECT_DELAY_SEC", 1.0)

        self._state_lock = threading.Lock()
        self._frame_condition = threading.Condition(self._state_lock)
        self._stop_event = threading.Event()

        self._started = False
        self._capture_thread: threading.Thread | None = None
        self._object_thread: threading.Thread | None = None
        self._activity_thread: threading.Thread | None = None

        self._object_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._activity_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)

        self._latest_frame_id = 0
        self._latest_jpeg: bytes = self._render_status_jpeg("Camera service starting…")
        self._latest_status = "starting"
        self._last_frame_ts = 0.0
        self._capture_fps = 0.0
        self._overlay_state = OverlayState()
        self._fps_meter = FPSMeter()

        self._model_state = {
            "id_available": False,
            "weapon_available": False,
            "activity_available": False,
            "anti_spoof_mode": "off",
        }

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                return
            self._started = True

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="surveillance-camera-capture",
            daemon=True,
        )
        self._object_thread = threading.Thread(
            target=self._object_loop,
            name="surveillance-object-worker",
            daemon=True,
        )
        self._activity_thread = threading.Thread(
            target=self._activity_loop,
            name="surveillance-activity-worker",
            daemon=True,
        )

        self._capture_thread.start()
        self._object_thread.start()
        self._activity_thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def get_jpeg(self, last_frame_id: int = -1, timeout: float = 1.0) -> tuple[int, bytes]:
        deadline = time.monotonic() + timeout
        with self._frame_condition:
            while (
                self._latest_frame_id == last_frame_id
                and not self._stop_event.is_set()
                and time.monotonic() < deadline
            ):
                remaining = deadline - time.monotonic()
                self._frame_condition.wait(timeout=max(remaining, 0.01))

            return self._latest_frame_id, self._latest_jpeg

    def get_status(self) -> dict[str, Any]:
        with self._state_lock:
            age = None
            if self._last_frame_ts > 0:
                age = round(time.time() - self._last_frame_ts, 3)

            return {
                "camera_index": self.camera_index,
                "backend": "CAP_DSHOW",
                "capture_width": self.capture_width,
                "capture_height": self.capture_height,
                "target_fps": self.target_fps,
                "capture_fps": round(self._capture_fps, 2),
                "last_frame_age_sec": age,
                "status": self._latest_status,
                "frame_id": self._latest_frame_id,
                "models": copy.deepcopy(self._model_state),
                "overlay": {
                    "activity_label": self._overlay_state.activity_label,
                    "activity_confidence": round(self._overlay_state.activity_confidence, 3),
                    "spoof_status": self._overlay_state.spoof_status,
                    "spoof_confidence": round(self._overlay_state.spoof_confidence, 3),
                    "id_count": len(self._overlay_state.id_detections),
                    "weapon_count": len(self._overlay_state.weapon_detections),
                    "updated_at": self._overlay_state.updated_at,
                },
            }

    def _open_camera(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self.camera_index, self.api_preference)
        if not cap.isOpened():
            return None

        # Backend-dependent hints; OpenCV documents that actual behavior varies
        # by device, driver, and backend.
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            logger.debug("One or more capture-property hints were ignored by the backend", exc_info=True)

        return cap

    def _capture_loop(self) -> None:
        frame_counter = 0
        sleep_interval = max(1.0 / max(self.target_fps, 1.0), 0.001)

        while not self._stop_event.is_set():
            cap = self._open_camera()
            if cap is None:
                logger.warning("Failed to open camera index %s with CAP_DSHOW; retrying", self.camera_index)
                self._set_status("camera_open_failed")
                self._publish_status_frame("Camera open failed. Retrying…")
                self._stop_event.wait(self.reconnect_delay)
                continue

            logger.info("Camera opened successfully with CAP_DSHOW")

            try:
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None or frame.size == 0:
                        logger.warning("Camera read failed; reopening camera")
                        self._set_status("camera_read_failed")
                        self._publish_status_frame("Camera read failed. Reconnecting…")
                        break

                    frame_counter += 1
                    capture_fps = self._fps_meter.tick()
                    snapshot = self._snapshot_overlay()

                    display = self._draw_overlay(frame, snapshot, capture_fps)
                    ok, buffer = cv2.imencode(
                        ".jpg",
                        display,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
                    )
                    if ok:
                        self._publish_jpeg(buffer.tobytes(), "camera_ok")
                    else:
                        logger.warning("JPEG encode failed for current frame")

                    if frame_counter % self.object_every_n == 0:
                        self._offer_latest(self._object_queue, frame.copy())

                    if frame_counter % self.activity_every_n == 0:
                        self._offer_latest(self._activity_queue, frame.copy())

                    time.sleep(sleep_interval)
            finally:
                cap.release()

            self._stop_event.wait(self.reconnect_delay)

    def _object_loop(self) -> None:
        try:
            from ml.inference.id_detector import IDDetector
            from ml.inference.weapon_detector import WeaponDetector

            id_detector = IDDetector()
            weapon_detector = WeaponDetector()

            self._set_model_state("id_available", id_detector.is_model_available())
            self._set_model_state("weapon_available", weapon_detector.is_model_available())

            while not self._stop_event.is_set():
                try:
                    frame = self._object_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                id_detections: list[dict[str, Any]] = []
                weapon_detections: list[dict[str, Any]] = []

                if id_detector.is_model_available():
                    out = id_detector.detect_image(frame)
                    if out.get("status") == "ok":
                        id_detections = self._filter_id_detections(out.get("detections", []), frame.shape)

                if weapon_detector.is_model_available():
                    out = weapon_detector.detect_image(frame)
                    if out.get("status") == "ok":
                        weapon_detections = out.get("detections", [])

                self._update_overlay(
                    id_detections=id_detections,
                    weapon_detections=weapon_detections,
                )
        except Exception:
            logger.exception("Object worker crashed")
            self._set_model_state("id_available", False)
            self._set_model_state("weapon_available", False)

    def _activity_loop(self) -> None:
        try:
            from ml.inference.activity_predictor import ActivityPredictor
            from ml.inference.anti_spoof import AntiSpoofEngine

            activity_predictor = ActivityPredictor()
            anti_spoof = AntiSpoofEngine()

            self._set_model_state("activity_available", activity_predictor.is_model_available())
            self._set_model_state("anti_spoof_mode", anti_spoof.mode)

            while not self._stop_event.is_set():
                try:
                    frame = self._activity_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                activity_label = "warming_up"
                activity_confidence = 0.0

                if activity_predictor.is_model_available():
                    out = activity_predictor.predict_frame(frame)
                    if out.get("status") == "ok":
                        activity_label = str(out.get("label", "unknown"))
                        activity_confidence = float(out.get("confidence", 0.0))
                    elif out.get("status") == "no_detection":
                        activity_label = "normal"
                        activity_confidence = 0.0
                    else:
                        activity_label = "model_unavailable"

                spoof_out = anti_spoof.evaluate(frame)
                spoof_status = str(spoof_out.get("status", "DISABLED"))
                spoof_confidence = float(spoof_out.get("confidence", 0.0))

                self._update_overlay(
                    activity_label=activity_label,
                    activity_confidence=activity_confidence,
                    spoof_status=spoof_status,
                    spoof_confidence=spoof_confidence,
                )
        except Exception:
            logger.exception("Activity worker crashed")
            self._set_model_state("activity_available", False)

    def _filter_id_detections(
        self,
        detections: list[dict[str, Any]],
        frame_shape: tuple[int, int, int],
    ) -> list[dict[str, Any]]:
        """
        Lightweight post-filter to reduce face-vs-ID false positives.

        Real ID cards are usually rectangular, non-tiny, and not almost-square.
        This does not replace better training, but it helps immediately.
        """
        frame_h, frame_w = frame_shape[:2]
        filtered: list[dict[str, Any]] = []

        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            box_w = max(x2 - x1, 1)
            box_h = max(y2 - y1, 1)
            area_ratio = (box_w * box_h) / float(max(frame_w * frame_h, 1))
            aspect_ratio = max(box_w, box_h) / float(max(min(box_w, box_h), 1))

            if box_w < 40 or box_h < 25:
                continue
            if area_ratio < 0.01:
                continue
            if not (1.2 <= aspect_ratio <= 2.5):
                continue

            filtered.append(det)

        return filtered

    def _snapshot_overlay(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "capture_fps": self._capture_fps,
                "status": self._latest_status,
                "models": copy.deepcopy(self._model_state),
                "overlay": copy.deepcopy(self._overlay_state),
            }

    def _update_overlay(self, **kwargs: Any) -> None:
        with self._state_lock:
            for key, value in kwargs.items():
                setattr(self._overlay_state, key, value)
            self._overlay_state.updated_at = time.time()

    def _set_model_state(self, key: str, value: Any) -> None:
        with self._state_lock:
            self._model_state[key] = value

    def _set_status(self, status: str) -> None:
        with self._state_lock:
            self._latest_status = status

    def _offer_latest(self, q: queue.Queue[np.ndarray], frame: np.ndarray) -> None:
        try:
            q.put_nowait(frame)
            return
        except queue.Full:
            pass

        try:
            q.get_nowait()
        except queue.Empty:
            pass

        try:
            q.put_nowait(frame)
        except queue.Full:
            pass

    def _publish_jpeg(self, jpeg_bytes: bytes, status: str) -> None:
        with self._frame_condition:
            self._latest_frame_id += 1
            self._latest_jpeg = jpeg_bytes
            self._latest_status = status
            self._last_frame_ts = time.time()
            self._capture_fps = self._fps_meter._fps
            self._frame_condition.notify_all()

    def _publish_status_frame(self, message: str) -> None:
        self._publish_jpeg(self._render_status_jpeg(message), "status_frame")

    def _render_status_jpeg(self, message: str) -> bytes:
        frame = np.full((self.capture_height, self.capture_width, 3), 38, dtype=np.uint8)

        cv2.putText(
            frame,
            "Real-Time Suspicious Activity Detection",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            message,
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
        )
        cv2.putText(
            frame,
            "The stream stays alive while the camera reconnects.",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
        )

        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if not ok:
            return b""
        return buffer.tobytes()

    def _draw_overlay(self, frame: np.ndarray, snapshot: dict[str, Any], capture_fps: float) -> np.ndarray:
        display = frame.copy()
        overlay: OverlayState = snapshot["overlay"]

        # Semi-transparent top bar.
        bar = display.copy()
        cv2.rectangle(bar, (0, 0), (display.shape[1], 120), (0, 0, 0), -1)
        display = cv2.addWeighted(bar, 0.45, display, 0.55, 0.0)

        self._draw_boxes(display, overlay.id_detections, (0, 255, 255), "ID")
        self._draw_boxes(display, overlay.weapon_detections, (0, 0, 255), "WEAPON")

        line1 = f"Activity: {overlay.activity_label} ({overlay.activity_confidence:.2f})"
        line2 = f"Anti-spoof: {overlay.spoof_status} ({overlay.spoof_confidence:.2f})"
        line3 = (
            f"FPS: {capture_fps:.1f} | IDs: {len(overlay.id_detections)} | "
            f"Weapons: {len(overlay.weapon_detections)}"
        )

        cv2.putText(display, line1, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
        cv2.putText(display, line2, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 210, 0), 2)
        cv2.putText(display, line3, (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            display,
            timestamp,
            (16, display.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        return display

    def _draw_boxes(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        color: tuple[int, int, int],
        prefix: str,
    ) -> None:
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            conf = float(det.get("conf", 0.0))
            label = str(det.get("label", prefix))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{prefix}: {label} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            top = max(y1 - th - baseline - 6, 0)
            cv2.rectangle(frame, (x1, top), (x1 + tw + 8, top + th + baseline + 6), color, -1)
            cv2.putText(
                frame,
                text,
                (x1 + 4, top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )


camera_service = CameraService()
