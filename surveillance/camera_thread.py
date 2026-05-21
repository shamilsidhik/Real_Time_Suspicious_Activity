"""
surveillance/camera_thread.py
Windows-safe camera capture using COM initialization for DSHOW.
"""
import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)

class CameraStream:
    def __init__(self, index=0):
        self.index = index
        self.frame = None
        self.running = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait up to 10 seconds for first frame
        for _ in range(100):
            if self.frame is not None:
                logger.info("Camera ready!")
                break
            time.sleep(0.1)

    def _run(self):
        # Initialize COM for this thread (required for DSHOW on Windows)
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass

        cap = None
        # Try each backend
        for backend_name, backend in [
            ("DSHOW", cv2.CAP_DSHOW),
            ("ANY",   cv2.CAP_ANY),
            ("MSMF",  cv2.CAP_MSMF),
        ]:
            try:
                c = cv2.VideoCapture(self.index, backend)
                time.sleep(0.3)
                if c.isOpened():
                    ret, frm = c.read()
                    if ret and frm is not None:
                        cap = c
                        logger.info("Camera opened with %s", backend_name)
                        break
                    c.release()
            except Exception as e:
                logger.warning("Backend %s failed: %s", backend_name, e)

        if cap is None:
            logger.error("All backends failed")
            return

        self.running = True
        while self.running:
            try:
                ret, frame = cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self.frame = frame.copy()
                else:
                    time.sleep(0.01)
            except Exception as e:
                logger.error("Camera read error: %s", e)
                time.sleep(0.1)

        cap.release()

    def read(self):
        with self._lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def isOpened(self):
        return self.frame is not None

    def release(self):
        self.running = False


_stream = None
_lock = threading.Lock()

def get_camera_stream(index=0) -> CameraStream:
    global _stream
    with _lock:
        if _stream is None or not _stream.isOpened():
            _stream = CameraStream(index)
    return _stream
