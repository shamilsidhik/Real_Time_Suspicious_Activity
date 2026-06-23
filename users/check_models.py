import os
import numpy as np

# Check all model paths
paths = [
    'ml/models/weapons_yolov8/best.pt',
    'ml/models/activity_yolov8/best.pt',
    'ml/models/id_card_yolov5/best.pt',
]

for p in paths:
    exists = os.path.exists(p)
    size = os.path.getsize(p) if exists else 0
    status = "EXISTS" if exists else "MISSING"
    print(f'{p}: {status} ({size // 1024} KB)')

# Test weapon model if present
wpn = 'ml/models/weapons_yolov8/best.pt'
if os.path.exists(wpn):
    from ultralytics import YOLO
    model = YOLO(wpn)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    r = model(frame, verbose=False, conf=0.1)[0]
    print('Weapon model test OK, classes:', model.names)
else:
    print('\nWeapon model is MISSING - please add best.pt to ml/models/weapons_yolov8/')