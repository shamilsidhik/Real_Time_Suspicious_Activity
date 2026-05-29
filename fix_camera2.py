with open("surveillance/views.py") as f:
    lines = f.readlines()

new_block = """            # Fetch frame from camera HTTP server
            frame = None
            success = False
            try:
                import urllib.request, numpy as np
                _resp = urllib.request.urlopen("http://127.0.0.1:8765", timeout=1)
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

lines[132] = new_block

with open("surveillance/views.py", "w") as f:
    f.writelines(lines)

print("Fixed!")
with open("surveillance/views.py") as f:
    lines2 = f.readlines()
for i,l in enumerate(lines2[130:150],131):
    print(i, l, end="")
