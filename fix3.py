with open("surveillance/views.py") as f:
    content = f.read()

old = """    # Use file-based camera (camera_server.py writes frames)
    camera = None
    FRAME_PATH = 'surveillance/static/live_frame.jpg' 

    if camera is None:

        err_msg = 'Camera unavailable (try laptop index 0/1 or set CAMERA_SOURCE to IP/RTSP)'
        if last_open_err is not None:
            err_msg = f"{err_msg}: {last_open_err}"
        frame_bytes = generate_unavailable_frame(err_msg)


        while True:
            yield (
                b"--frame\\r\\n"
                b"Content-Type: image/jpeg\\r\\n\\r\\n" + frame_bytes + b"\\r\\n"
            )
            time.sleep(1)"""

new = """    # Use file-based camera (camera_server.py writes frames to this file)
    FRAME_PATH = 'surveillance/static/live_frame.jpg'
    camera = 'file_based'"""

content = content.replace(old, new)
with open("surveillance/views.py", "w") as f:
    f.write(content)
print("Done! Replaced:", "file_based" in content)
