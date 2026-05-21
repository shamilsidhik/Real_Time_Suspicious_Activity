with open("surveillance/views.py") as f:
    content = f.read()

old = """            # Read frame from camera_server.py
            _fpath = 'surveillance/static/live_frame.jpg'
            if os.path.exists(_fpath):
                frame = cv2.imread(_fpath)
                success = frame is not None
            else:
                success = False
                frame = None"""

new = """            # Read frame from camera_server.py (with race condition protection)
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

content = content.replace(old, new)
with open("surveillance/views.py", "w") as f:
    f.write(content)
print("Done!")
