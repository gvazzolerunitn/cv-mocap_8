#!/usr/bin/env python3
"""
Extract a 3D pose clip from the motion capture .mat file and save it in JSON format.

This script reads the motion capture data in 'Nick_3.mat', selects frames
corresponding to a specified time interval (79–83 seconds), and writes the
3D joint positions (in COCO keypoint order) to a JSON file.
"""

import json
import numpy as np
from scipy.io import loadmat

# Path to the .mat file containing MoCap data
MAT_FILE     = "Nick_3.mat"
# Path to the output JSON for the extracted clip
OUTPUT_JSON  = "mocap_clip_79_83s.json"

# Define the COCO keypoint order (18 joints)
COCO_KPTS = [
    "Hips",  "RHip", "RKnee", "RAnkle", "RFoot",
    "LHip",  "LKnee", "LAnkle", "LFoot",
    "Spine", "Neck",  "Head",
    "RShoulder", "RElbow", "RHand",
    "LShoulder", "LElbow", "LHand"
]

# Map each COCO keypoint to the corresponding label in the MoCap skeleton
COCO_TO_MATLAB = {
    "Hips":       "Hips",
    "RHip":       "RightUpLeg",
    "RKnee":      "RightLeg",
    "RAnkle":     "RightFoot",
    "RFoot":      "RightToeBase",
    "LHip":       "LeftUpLeg",
    "LKnee":      "LeftLeg",
    "LAnkle":     "LeftFoot",
    "LFoot":      "LeftToeBase",
    "Spine":      "Spine",
    "Neck":       "Neck",
    "Head":       "Head",
    "RShoulder":  "RightShoulder",
    "RElbow":     "RightForeArm",
    "RHand":      "RightHand",
    "LShoulder":  "LeftShoulder",
    "LElbow":     "LeftForeArm",
    "LHand":      "LeftHand"
}

# ----- 1. Load the MATLAB file -----
# - struct_as_record=False: keep MATLAB structs as native Python objects
# - squeeze_me=True: remove singleton dimensions
mat = loadmat(MAT_FILE, struct_as_record=False, squeeze_me=True)
mo  = mat["Nick_3"]            # Top-level struct
skel = mo.Skeletons            # Nested 'Skeletons' struct

# ----- 2. Retrieve joint labels -----
# The skeleton may list joint names under 'SegmentLabels' or 'JointNames'
if hasattr(skel, "SegmentLabels"):
    labels = np.asarray(skel.SegmentLabels).astype(str).tolist()
elif hasattr(skel, "JointNames"):
    labels = np.asarray(skel.JointNames).astype(str).tolist()
else:
    raise ValueError("Neither 'SegmentLabels' nor 'JointNames' found in Skeletons")

# Build a mapping from label name to its index
label_to_idx = {name: i for i, name in enumerate(labels)}

# ----- 3. Rearrange PositionData to shape (T, J, 3) -----
P = skel.PositionData  # Original shape could be (3,J,T), (T,J,3), or (J,3,T)
if P.ndim == 3:
    if P.shape[0] == 3:
        # Convert from (3, J, T) to (T, J, 3)
        P = np.transpose(P, (2, 1, 0))
    elif P.shape[2] == 3:
        # Already in (T, J, 3)
        pass
    elif P.shape[1] == 3:
        # Convert from (J, 3, T) to (T, J, 3)
        P = np.transpose(P, (2, 0, 1))
    else:
        raise ValueError("Unexpected 3-D shape for PositionData")
elif P.ndim == 2:
    # Flattened shape (T, J*3), reshape into (T, J, 3)
    if P.shape[1] % 3 != 0:
        raise ValueError("PositionData flattened length not divisible by 3")
    P = P.reshape((-1, P.shape[1] // 3, 3))
else:
    raise ValueError("Unhandled PositionData dimensionality")

# Extract total frames (T) and joint count (J)
T, J, _ = P.shape

# ----- 4. Define time interval and convert to frame indices -----
fps          = int(mo.FrameRate)  # e.g., 100 fps
START_SEC    = 79
END_SEC      = 83
start_frame  = int(START_SEC * fps)        # inclusive start frame
end_frame    = int(END_SEC   * fps) - 1    # inclusive end frame
if end_frame >= T:
    raise ValueError("Requested end time exceeds recording duration")

print(f"Extracting frames {start_frame}–{end_frame} inclusive "
      f"({end_frame - start_frame + 1} frames)")

# ----- 5. Build output JSON dictionary -----
out = {}
for f in range(start_frame, end_frame + 1):
    # Skip if entire frame is empty (all NaNs)
    if np.isnan(P[f]).all():
        continue

    joints = {}
    # For each COCO keypoint, map to the MoCap label and extract coordinates
    for coco_idx, coco_name in enumerate(COCO_KPTS):
        mcap_name = COCO_TO_MATLAB[coco_name]
        j_idx = label_to_idx.get(mcap_name)
        if j_idx is None:
            # This joint label is not present in the MoCap data
            continue

        xyz = P[f, j_idx]
        if np.isnan(xyz).all():
            # All coordinates are NaN → skip this joint
            continue

        # Store the 3D position (rounded to 6 decimal places)
        joints[str(coco_idx)] = xyz.astype(float).round(6).tolist()

    if joints:
        # Use the absolute frame index as the JSON key
        out[str(f)] = joints

# ----- 6. Save to JSON -----
with open(OUTPUT_JSON, "w") as fp:
    json.dump(out, fp, indent=2)

print(f"✅ Saved {len(out)} valid frames to '{OUTPUT_JSON}'")
