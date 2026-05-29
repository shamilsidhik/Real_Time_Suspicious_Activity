"""
Shared Windows webcam manager for the Django live MJPEG pipeline.

Only this module opens cv2.VideoCapture. Request handlers read the latest
frame through get_camera_manager(), which keeps camera ownership out of
Django request threads and avoids stale-frame buildup.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


@dataclass
class CameraState:
    camera_open: bool = False
    fps: float = 0.0
    frame_timestamp: float = 0.0
    reconnect_count: int = 0
    last_error: str = ""
    frame_width: int = 640
    frame_height: int = 480


class CameraManager:
    """Dedicated capture-thread manager using Windows DirectShow."""

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: float = 20.0,
        reconnect_delay: float = 1.5,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.reconnect_delay = reconnect_delay

        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._latest_processed_frame = None
        self._state = CameraState(frame_width=width, frame_height=height)
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started and self._thread and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="django-webcam-capture",
                daemon=True,
            )
            self._thread.start()
            self._started = True
            logger.info("Camera manager thread started")

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

    def read(self) -> Tuple[bool, Optional[object]]:
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def set_processed_frame(self, frame) -> None:
        with self._lock:
            self._latest_processed_frame = None if frame is None else frame.copy()

    def read_processed(self) -> Tuple[bool, Optional[object]]:
        with self._lock:
            if self._latest_processed_frame is None:
                return False, None
            return True, self._latest_processed_frame.copy()

    def status(self) -> dict:
        with self._lock:
            return asdict(self._state)

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _open_camera(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            logger.debug("CAP_PROP_BUFFERSIZE is not supported by this backend")

        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Unable to open camera index 0 with CAP_DSHOW")

        warmed_frame = None
        for _ in range(20):
            ok, frame = cap.read()
            if ok and frame is not None:
                warmed_frame = frame
            time.sleep(0.02)

        if warmed_frame is None:
            cap.release()
            raise RuntimeError("Camera opened but did not return warm-up frames")

        with self._lock:
            self._latest_frame = warmed_frame.copy()
            self._state.camera_open = True
            self._state.frame_timestamp = time.time()
            self._state.last_error = ""

        return cap

    def _capture_loop(self) -> None:
        cap = None
        frame_counter = 0
        fps_started_at = time.monotonic()
        read_failures = 0
        frame_interval = 1.0 / max(self.target_fps, 1.0)

        while not self._stop_event.is_set():
            try:
                if cap is None or not cap.isOpened():
                    with self._lock:
                        self._state.camera_open = False
                    cap = self._open_camera()
                    read_failures = 0
                    frame_counter = 0
                    fps_started_at = time.monotonic()
                    logger.info("Camera opened with DirectShow at index %s", self.index)

                loop_started_at = time.monotonic()
                ok, frame = cap.read()

                if not ok or frame is None:
                    read_failures += 1
                    if read_failures >= 10:
                        raise RuntimeError("Failed to read camera frames repeatedly")
                    time.sleep(0.03)
                    continue

                read_failures = 0
                now = time.time()
                frame_counter += 1

                elapsed = time.monotonic() - fps_started_at
                if elapsed >= 1.0:
                    fps = frame_counter / elapsed
                    frame_counter = 0
                    fps_started_at = time.monotonic()
                else:
                    fps = None

                with self._lock:
                    self._latest_frame = frame.copy()
                    self._state.camera_open = True
                    self._state.frame_timestamp = now
                    self._state.last_error = ""
                    if fps is not None:
                        self._state.fps = round(float(fps), 1)

                sleep_for = frame_interval - (time.monotonic() - loop_started_at)
                if sleep_for > 0:
                    time.sleep(sleep_for)

            except Exception as exc:
                logger.warning("Camera capture will reconnect: %s", exc)
                if cap is not None:
                    cap.release()
                    cap = None

                with self._lock:
                    self._state.camera_open = False
                    self._state.reconnect_count += 1
                    self._state.last_error = str(exc)

                time.sleep(self.reconnect_delay)

        if cap is not None:
            cap.release()

        with self._lock:
            self._state.camera_open = False


_manager: Optional[CameraManager] = None
_manager_lock = threading.Lock()


def get_camera_manager(start: bool = False) -> CameraManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = CameraManager()
        if start:
            _manager.start()
        return _manager

