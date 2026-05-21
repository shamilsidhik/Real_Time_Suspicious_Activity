import cv2
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

latest_frame = None
lock = threading.Lock()

def capture_loop():
    global latest_frame

    while True:
        ret, frame = cap.read()

        if ret and frame is not None:
            success, jpg = cv2.imencode(".jpg", frame)

            if success:
                with lock:
                    latest_frame = jpg.tobytes()
        else:
            print("Failed to read camera")

        time.sleep(0.033)

threading.Thread(target=capture_loop, daemon=True).start()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()

        while True:
            with lock:
                frame = latest_frame

            if frame:
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.033)
                except:
                    break

    def log_message(self, *args):
        pass

print("Camera server running on http://localhost:8765")
HTTPServer(("localhost", 8765), Handler).serve_forever()