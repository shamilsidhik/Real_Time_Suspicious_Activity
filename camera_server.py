from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np


logging.basicConfig(level=os.environ.get("CAMERA_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("camera_server")


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
    id_detections: list[dict[str, Any]] = field(default_factory=list)
    weapon_detections: list[dict[str, Any]] = field(default_factory=list)
    updated_at: float = 0.0


class FPSMeter:
    def __init__(self) -> None:
        self._last: float | None = None
        self.value = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            instant = 1.0 / max(now - self._last, 1e-6)
            self.value = instant if self.value <= 0 else (self.value * 0.88) + (instant * 0.12)
        self._last = now
        return self.value


class CameraService:
    def __init__(self) -> None:
        self.host = os.environ.get("CAMERA_SERVER_HOST", "127.0.0.1")
        self.port = _env_int("CAMERA_SERVER_PORT", 8765)
        self.camera_index = _env_int("CAMERA_INDEX", 0)
        self.width = _env_int("CAMERA_WIDTH", 640)
        self.height = _env_int("CAMERA_HEIGHT", 480)
        self.target_fps = _env_float("CAMERA_TARGET_FPS", 18.0)
        self.jpeg_quality = _env_int("MJPEG_JPEG_QUALITY", 78)
        self.reconnect_delay = _env_float("CAMERA_RECONNECT_DELAY_SEC", 1.0)
        self.id_every = max(_env_int("ID_DETECT_EVERY_N", 4), 1)
        self.weapon_every = max(_env_int("WEAPON_DETECT_EVERY_N", 5), 1)
        self.activity_every = max(_env_int("ACTIVITY_DETECT_EVERY_N", 9), 1)

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fps = FPSMeter()
        self._frame_id = 0
        self._last_frame_ts = 0.0
        self._latest_jpeg = self._status_jpeg("Camera service starting")
        self._status = "starting"
        self._last_error = ""
        self._camera_backend = "CAP_DSHOW"
        self._actual_width = 0
        self._actual_height = 0
        self._actual_fps = 0.0
        self._overlay = OverlayState()
        self._models = {
            "activity": {"available": False, "backend": "unloaded", "path": "", "last_error": ""},
            "id_card": {"available": False, "backend": "unloaded", "path": "", "last_error": ""},
            "weapon": {"available": False, "backend": "unloaded", "path": "", "last_error": ""},
            "anti_spoof": {"available": False, "backend": "disabled_live_mode", "path": "", "last_error": ""},
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=2.0)

    def wait_for_jpeg(self, last_frame_id: int, timeout: float = 2.0) -> tuple[int, bytes]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._frame_id == last_frame_id and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return self._frame_id, self._latest_jpeg

    def snapshot_jpeg(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    def status(self) -> dict[str, Any]:
        with self._lock:
            age = round(time.time() - self._last_frame_ts, 3) if self._last_frame_ts else None
            return {
                "ok": self._status == "camera_ok",
                "status": self._status,
                "camera_open": self._status == "camera_ok",
                "camera_index": self.camera_index,
                "backend": self._camera_backend,
                "capture_width": self._actual_width or self.width,
                "capture_height": self._actual_height or self.height,
                "target_fps": self.target_fps,
                "capture_fps": round(self._fps.value, 2),
                "last_frame_age_sec": age,
                "frame_id": self._frame_id,
                "last_error": self._last_error,
                "stream_url": f"http://{self.host}:{self.port}/stream.mjpg",
                "snapshot_url": f"http://{self.host}:{self.port}/snapshot.jpg",
                "models": copy.deepcopy(self._models),
                "overlay": {
                    "activity_label": self._overlay.activity_label,
                    "activity_confidence": round(self._overlay.activity_confidence, 3),
                    "id_count": len(self._overlay.id_detections),
                    "weapon_count": len(self._overlay.weapon_detections),
                    "updated_at": self._overlay.updated_at,
                },
            }

    def _capture_loop(self) -> None:
        self._load_detectors()
        frame_no = 0
        sleep_interval = 1.0 / max(self.target_fps, 1.0)
        while not self._stop.is_set():
            cap = self._open_camera()
            if cap is None:
                self._publish_status("Camera unavailable. Retrying...", "camera_open_failed", "open failed")
                self._stop.wait(self.reconnect_delay)
                continue
            try:
                while not self._stop.is_set():
                    for _ in range(2):
                        cap.grab()
                    ok, frame = cap.retrieve()
                    if not ok or frame is None or frame.size == 0:
                        self._publish_status("Camera read failed. Reconnecting...", "camera_read_failed", "read failed")
                        break

                    frame_no += 1
                    self._run_scheduled_inference(frame, frame_no)
                    fps = self._fps.tick()
                    display = self._draw_overlay(frame, fps)
                    ok, buf = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                    if ok:
                        self._publish_jpeg(buf.tobytes(), "camera_ok", "")
                    elapsed_sleep = max(0.0, sleep_interval)
                    if elapsed_sleep:
                        self._stop.wait(elapsed_sleep)
            finally:
                cap.release()
            self._stop.wait(self.reconnect_delay)

    def _open_camera(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        backend = cap.getBackendName() if hasattr(cap, "getBackendName") else "CAP_DSHOW"
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        logger.info("Camera opened backend=%s width=%s height=%s fps=%.2f", backend, actual_w, actual_h, actual_fps)
        with self._lock:
            self._camera_backend = backend
            self._actual_width = actual_w
            self._actual_height = actual_h
            self._actual_fps = actual_fps
        return cap

    def _load_detectors(self) -> None:
        self.id_detector = None
        self.weapon_detector = None
        self.activity_predictor = None
        try:
            from ml.inference.id_detector import IDDetector

            self.id_detector = IDDetector()
            self._set_model("id_card", self.id_detector.model_status())
        except Exception as exc:
            self._set_model("id_card", {"available": False, "backend": "error", "path": "", "last_error": str(exc)})
            logger.exception("ID detector init failed")
        try:
            from ml.inference.weapon_detector import WeaponDetector

            self.weapon_detector = WeaponDetector()
            self._set_model("weapon", self.weapon_detector.model_status())
        except Exception as exc:
            self._set_model("weapon", {"available": False, "backend": "error", "path": "", "last_error": str(exc)})
            logger.exception("Weapon detector init failed")
        try:
            from ml.inference.activity_predictor import ActivityPredictor

            self.activity_predictor = ActivityPredictor()
            self._set_model("activity", self.activity_predictor.model_status())
        except Exception as exc:
            self._set_model("activity", {"available": False, "backend": "error", "path": "", "last_error": str(exc)})
            logger.exception("Activity predictor init failed")
        try:
            from ml.inference.anti_spoof import model_status

            self._set_model("anti_spoof", model_status())
        except Exception:
            pass

    def _run_scheduled_inference(self, frame: np.ndarray, frame_no: int) -> None:
        if self.id_detector and frame_no % self.id_every == 0:
            out = self.id_detector.detect_frame(frame)
            if out.get("status") in {"ok", "unavailable"}:
                self._update_overlay(id_detections=out.get("detections", []))
        if self.weapon_detector and frame_no % self.weapon_every == 0:
            out = self.weapon_detector.detect_frame(frame)
            if out.get("status") in {"ok", "unavailable"}:
                self._update_overlay(weapon_detections=out.get("detections", []))
        if self.activity_predictor and frame_no % self.activity_every == 0:
            out = self.activity_predictor.predict_frame(frame)
            if out.get("status") in {"ok", "unavailable", "error"}:
                self._update_overlay(
                    activity_label=str(out.get("label", "normal")),
                    activity_confidence=float(out.get("confidence", 0.0)),
                )

    def _draw_overlay(self, frame: np.ndarray, fps: float) -> np.ndarray:
        display = frame.copy()
        with self._lock:
            overlay = copy.deepcopy(self._overlay)
            status = self._status

        alert = overlay.weapon_detections or overlay.activity_label.lower() in {"fight", "violence", "suspicious", "weapon"}
        color = (0, 0, 255) if alert else ((0, 190, 255) if overlay.id_detections else (0, 255, 136))
        bar = display.copy()
        cv2.rectangle(bar, (0, 0), (display.shape[1], 112), (10, 10, 15), -1)
        display = cv2.addWeighted(bar, 0.72, display, 0.28, 0)

        self._draw_boxes(display, overlay.id_detections, (0, 190, 255), "ID")
        self._draw_boxes(display, overlay.weapon_detections, (0, 0, 255), "WEAPON")

        lines = [
            f"FPS {fps:.1f}  |  Camera {status}",
            f"Activity {overlay.activity_label} ({overlay.activity_confidence:.2f})",
            f"ID cards {len(overlay.id_detections)}  |  Weapons {len(overlay.weapon_detections)}  |  Anti-spoof DISABLED",
        ]
        for idx, text in enumerate(lines):
            cv2.putText(display, text, (16, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color if idx == 1 else (235, 245, 240), 2)
        cv2.putText(display, time.strftime("%Y-%m-%d %H:%M:%S"), (16, display.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 245, 240), 1)
        return display

    def _draw_boxes(self, frame: np.ndarray, detections: list[dict[str, Any]], color: tuple[int, int, int], prefix: str) -> None:
        h, w = frame.shape[:2]
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{prefix}: {det.get('label', prefix)} {float(det.get('conf', 0.0)):.2f}"
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            top = max(0, y1 - th - base - 8)
            cv2.rectangle(frame, (x1, top), (min(w - 1, x1 + tw + 8), top + th + base + 8), color, -1)
            cv2.putText(frame, text, (x1 + 4, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (5, 5, 8), 2)

    def _status_jpeg(self, message: str) -> bytes:
        frame = np.full((self.height, self.width, 3), (15, 15, 22), dtype=np.uint8)
        cv2.putText(frame, "Real-Time Surveillance", (24, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 136), 2)
        cv2.putText(frame, message, (24, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 190, 255), 2)
        cv2.putText(frame, "Stream remains active while the camera reconnects.", (24, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 225, 225), 1)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        return buf.tobytes() if ok else b""

    def _publish_status(self, message: str, status: str, error: str) -> None:
        self._publish_jpeg(self._status_jpeg(message), status, error)

    def _publish_jpeg(self, jpeg: bytes, status: str, error: str) -> None:
        with self._condition:
            self._frame_id += 1
            self._latest_jpeg = jpeg
            self._status = status
            self._last_error = error
            self._last_frame_ts = time.time()
            self._condition.notify_all()

    def _update_overlay(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._overlay, key, value)
            self._overlay.updated_at = time.time()

    def _set_model(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._models[key] = value


service = CameraService()


class Handler(BaseHTTPRequestHandler):
    server_version = "CameraServer/1.0"

    def do_GET(self) -> None:
        if self.path.startswith("/stream.mjpg"):
            self._stream()
        elif self.path.startswith("/status"):
            self._json(service.status(), HTTPStatus.OK)
        elif self.path.startswith("/healthz"):
            self._json({"ok": True, "status": service.status()["status"]}, HTTPStatus.OK)
        elif self.path.startswith("/snapshot.jpg"):
            self._jpeg(service.snapshot_jpeg())
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _headers(self, status: HTTPStatus, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")

    def _json(self, payload: dict[str, Any], status: HTTPStatus) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._headers(status, "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _jpeg(self, body: bytes) -> None:
        self._headers(HTTPStatus.OK, "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        self._headers(HTTPStatus.OK, "multipart/x-mixed-replace; boundary=frame")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        last_id = -1
        try:
            while True:
                last_id, body = service.wait_for_jpeg(last_id, timeout=2.0)
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                self.wfile.write(str(len(body)).encode("ascii"))
                self.wfile.write(b"\r\n\r\n")
                self.wfile.write(body)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return


def main() -> None:
    service.start()
    httpd = ThreadingHTTPServer((service.host, service.port), Handler)
    logger.info("Camera server listening on http://%s:%s", service.host, service.port)
    try:
        httpd.serve_forever()
    finally:
        service.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
