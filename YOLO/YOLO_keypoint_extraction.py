from ultralytics import YOLO
import os
import json
import re
import numpy as np

# Directory containing the multi-view images
IMG_DIR = '../dataset/mocap_8.v1i.coco/train'
# Path to the YOLOv8-Pose model weights
MODEL_PATH = 'yolov8s-pose.pt'
# Output JSON file for storing 2D keypoints per frame and camera
OUTPUT_JSON = 'yolo2d_for_triangulation.json'

# Initialize the YOLOv8-Pose model
model = YOLO(MODEL_PATH)
# Run prediction in streaming mode over the input directory without saving images
results = model.predict(source=IMG_DIR, save=False, stream=True, verbose=False)

# Prepare a dictionary to accumulate 2D keypoints:
# frame_data[frame][cam] = { joint_index: (x, y) }
frame_data = {}

for r in results:
    # Extract the filename from the result's image path
    fname = os.path.basename(r.path)
    # Match filenames of the form 'out<cam>_frame_<frame>_png...'
    m = re.match(r'out(\d+)_frame_(\d+)_png', fname)
    if not m:
        # Skip any files that do not match the expected naming pattern
        continue

    # Parse camera ID and frame number from the filename
    cam   = int(m.group(1))
    frame = int(m.group(2))

    # If no person was detected in this frame, skip it
    if len(r.keypoints.xy) == 0:
        continue

    # Take the first detected person's keypoints (shape: [num_keypoints, 2])
    kps = r.keypoints.xy[0]

    # Convert keypoints into a dict: { joint_index: (x, y) }
    pts2d = {i: (float(x), float(y)) for i, (x, y) in enumerate(kps)}

    # Store under the corresponding frame and camera
    frame_data.setdefault(frame, {})[cam] = pts2d

# Write the aggregated 2D keypoints to the output JSON file
with open(OUTPUT_JSON, 'w') as f:
    json.dump(frame_data, f, indent=2)

print(f"✅ Saved YOLO 2D keypoints for triangulation to {OUTPUT_JSON}")
