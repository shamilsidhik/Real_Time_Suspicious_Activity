content = open('surveillance/views.py').read()
content = content.replace(
    "if camera is None or (hasattr(camera, 'isOpened') and not camera.isOpened()):",
    "if camera is None:"
)
open('surveillance/views.py','w').write(content)
print('Fixed!')
