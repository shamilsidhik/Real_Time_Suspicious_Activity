with open("surveillance/views.py") as f:
    lines = f.readlines()

print("Line 119-125 before fix:")
for i,l in enumerate(lines[118:125],119):
    print(i, repr(l))

# Replace line 121 (index 120)
lines[120] = '''            # Fetch frame from camera HTTP server
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
            except Exception as _e:
                pass
'''

with open("surveillance/views.py", "w") as f:
    f.writelines(lines)

print("Done! Verifying...")
with open("surveillance/views.py") as f:
    lines2 = f.readlines()
for i,l in enumerate(lines2[118:130],119):
    print(i, l.strip())
