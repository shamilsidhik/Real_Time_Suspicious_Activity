with open("surveillance/views.py") as f:
    content = f.read()

old = """            # Read frame from camera_server.py (with race condition protection)
            _fpath = 'surveillance/static/live_frame.jpg'
            frame = None
            success = False
            if os.path.exists(_fpath):
                try:
                    _bytes = open(_fpath, 'rb').read()
                    if len(_bytes) > 1000:
                        import numpy as np
                        _arr = np.frombuffer(_bytes, dtype=np.uint8)
                        _f = cv2.imdecode(_arr, cv2.IMREAD_COLOR)
                        if _f is not None:
                            frame = _f
                            success = True
                except Exception:
                    pass"""

new = """            # Fetch frame from camera HTTP server
            frame = None
            success = False
            try:
                import urllib.request
                import numpy as np
                _resp = urllib.request.urlopen('http://localhost:8765', timeout=1)
                _bytes = _resp.read()
                if len(_bytes) > 1000:
                    _arr = np.frombuffer(_bytes, dtype=np.uint8)
                    _f = cv2.imdecode(_arr, cv2.IMREAD_COLOR)
                    if _f is not None:
                        frame = _f
                        success = True
            except Exception:
                pass"""

content = content.replace(old, new)
with open("surveillance/views.py", "w") as f:
    f.write(content)
print("Done!")
