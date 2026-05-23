import cv2, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

latest_frame = None
lock = threading.Lock()

def capture_loop():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if ret and frame is not None:
            _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with lock:
                latest_frame = jpg.tobytes()
        time.sleep(0.033)

threading.Thread(target=capture_loop, daemon=True).start()
print("Camera started, waiting for first frame...")
time.sleep(2)
print("Camera server running on http://localhost:8765")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with lock:
            frame = latest_frame
        if frame:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(frame)
            self.wfile.flush()
        else:
            self.send_response(503)
            self.end_headers()
    def log_message(self, *args):
        pass

server = HTTPServer(("localhost", 8765), Handler)
server.serve_forever()
