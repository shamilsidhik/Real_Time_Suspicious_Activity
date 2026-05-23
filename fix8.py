with open("surveillance/views.py") as f:
    lines = f.readlines()

# Find and fix the urllib block - add proper indentation
new_block = """            # Fetch frame from camera HTTP server
            frame = None
            success = False
            try:
                import urllib.request, numpy as np
                _resp = urllib.request.urlopen("http://localhost:8765", timeout=1)
                _bytes = _resp.read()
                if len(_bytes) > 5000:
                    _arr = np.frombuffer(_bytes, dtype=np.uint8)
                    _f = cv2.imdecode(_arr, cv2.IMREAD_COLOR)
                    if _f is not None:
                        frame = _f
                        success = True
            except Exception:
                pass
"""

lines[120] = new_block

with open("surveillance/views.py", "w") as f:
    f.writelines(lines)
print("Fixed!")
