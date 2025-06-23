from ultralytics import YOLO
import os
import json
import re
import numpy as np

IMG_DIR = '../dataset/mocap_8.v1i.coco/train'
MODEL_PATH = 'yolov8s-pose.pt'
OUTPUT_JSON = 'yolo2d_for_triangulation.json'

model = YOLO(MODEL_PATH)
results = model.predict(source=IMG_DIR, save=False, stream=True, verbose=False)

# Struttura: frame_data[frame][cam] = {joint_idx: (x, y)}
frame_data = {}

for r in results:
    fname = os.path.basename(r.path)
    m = re.match(r'out(\d+)_frame_(\d+)_png', fname)
    if not m:
        continue
    cam = int(m.group(1))
    frame = int(m.group(2))
    # Prendi la prima persona rilevata (o scegli la migliore se più persone)
    if len(r.keypoints.xy) == 0:
        continue
    kps = r.keypoints.xy[0]  # shape (17,2)
    pts2d = {i: (float(x), float(y)) for i, (x, y) in enumerate(kps)}
    frame_data.setdefault(frame, {})[cam] = pts2d

with open(OUTPUT_JSON, 'w') as f:
    json.dump(frame_data, f, indent=2)

print(f"YOLO 2D keypoints per triangolazione salvati in {OUTPUT_JSON}")