import json
import numpy as np
import csv

# Paths to the YOLO-triangulated 3D output and the ground-truth triangulated positions
YOLO_3D_PATH = 'triangulated_positions_yolo.json'
GT_3D_PATH   = '../Triangulation and reprojection and 3D/triangulated_positions_v2.json'

# Number of joints expected in each pose (17 for COCO, 18 if there is one extra)
NUM_JOINTS = 17

# Load the YOLO-based 3D reconstructions
with open(YOLO_3D_PATH) as f:
    yolo_3d = json.load(f)

# Load the ground-truth 3D triangulations
with open(GT_3D_PATH) as f:
    gt_3d = json.load(f)

# Prepare lists to collect per-joint errors and overall MPJPE
mpjpe_list = []
per_joint_errors = [[] for _ in range(NUM_JOINTS)]

# Find the set of frames present in both predictions and ground truth
common_frames = set(yolo_3d.keys()) & set(gt_3d.keys())
print(f"Common frames: {sorted(common_frames)}")

# Iterate over each common frame
for frame in common_frames:
    # For each joint index in the pose
    for j in range(NUM_JOINTS):
        # Attempt to retrieve the predicted and ground-truth 3D coordinates
        yolo_joint = yolo_3d[frame].get(str(j)) or yolo_3d[frame].get(j)
        gt_joint   = gt_3d[frame].get(str(j)) or gt_3d[frame].get(j)

        # Skip if either is missing or contains NaN/inf
        if yolo_joint is None or gt_joint is None:
            continue
        if np.any(np.isnan(yolo_joint)) or np.any(np.isnan(gt_joint)):
            continue
        if np.any(np.isinf(yolo_joint)) or np.any(np.isinf(gt_joint)):
            continue

        # Compute the Euclidean distance (MPJPE) for this joint
        err = np.linalg.norm(np.array(yolo_joint) - np.array(gt_joint))
        mpjpe_list.append(err)
        per_joint_errors[j].append(err)

# If we collected any errors, print summary statistics
if mpjpe_list:
    print(f"\nEvaluation over {len(common_frames)} frames and {len(mpjpe_list)} joint-pairs:")
    print(f"Mean MPJPE:   {np.mean(mpjpe_list):.2f} mm")
    print(f"Median MPJPE: {np.median(mpjpe_list):.2f} mm")
    print(f"Min / Max:    {np.min(mpjpe_list):.2f} / {np.max(mpjpe_list):.2f} mm")
    print("\nAverage error per joint:")
    for j, errs in enumerate(per_joint_errors):
        if errs:
            print(f"  Joint {j}: {np.mean(errs):.2f} mm")
        else:
            print(f"  Joint {j}: N/A")
else:
    print("No comparable joint-pairs found!")

# Write detailed per-frame, per-joint MPJPE values to CSV
with open('mpjpe_3d_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['frame', 'joint', 'mpjpe_mm'])
    for frame in common_frames:
        for j in range(NUM_JOINTS):
            yolo_joint = yolo_3d[frame].get(str(j)) or yolo_3d[frame].get(j)
            gt_joint   = gt_3d[frame].get(str(j)) or gt_3d[frame].get(j)
            if yolo_joint is None or gt_joint is None:
                continue
            if np.any(np.isnan(yolo_joint)) or np.any(np.isnan(gt_joint)):
                continue
            if np.any(np.isinf(yolo_joint)) or np.any(np.isinf(gt_joint)):
                continue
            err = np.linalg.norm(np.array(yolo_joint) - np.array(gt_joint))
            writer.writerow([frame, j, err])

print("✅ Results saved to mpjpe_3d_results.csv")
