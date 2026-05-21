with open("surveillance/views.py") as f:
    content = f.read()

# Find the generate_camera_frames function and replace the entire camera section
old = """    # Use file-based camera (camera_server.py writes frames to this file)
    FRAME_PATH = 'surveillance/static/live_frame.jpg'
    camera = 'file_based'"""

new = """    # File-based camera reading
    FRAME_PATH = 'surveillance/static/live_frame.jpg'
    camera = 'file_based'
    
    # Wait for first frame
    import time as _time
    for _ in range(20):
        if os.path.exists(FRAME_PATH) and os.path.getsize(FRAME_PATH) > 1000:
            break
        _time.sleep(0.5)"""

content = content.replace(old, new)
with open("surveillance/views.py", "w") as f:
    f.write(content)
print("Done!")
