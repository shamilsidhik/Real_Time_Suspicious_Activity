with open("surveillance/views.py") as f:
    lines = f.readlines()

print("Line 131-137 before fix:")
for i,l in enumerate(lines[130:137],131):
    print(i, repr(l))

lines[132] = """            # Fetch frame from camera HTTP server
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

with open("surveillance/views.py", "w") as f:
    f.writelines(lines)
print("Fixed! Verifying...")
with open("surveillance/views.py") as f:
    lines2 = f.readlines()
for i,l in enumerate(lines2[130:148],131):
    print(i, l.strip())
