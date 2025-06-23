import json
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import os
from scipy.optimize import linear_sum_assignment
import csv

# === CONFIGURATION ===
# Path to the COCO-format ground truth annotations for the train split
COCO_ANN = '../dataset/mocap_8.v1i.coco/train/_annotations.coco.json'
# Directory containing the multiple camera images
IMG_DIR = '../dataset/mocap_8.v1i.coco/train'
# YOLOv8-Pose model weights
YOLO_MODEL = 'yolov8s-pose.pt'

# --- 1. Load COCO annotations into memory ---
with open(COCO_ANN) as f:
    coco = json.load(f)

# Build a map from image filename to image ID
fname2imgid = {img['file_name']: img['id'] for img in coco['images']}

# Build a dict: image_id -> list of ground-truth keypoint arrays
gt_kps = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    # reshape flat list [x,y,v,...] into (num_joints, 3)
    kps = np.array(ann['keypoints']).reshape(-1, 3)
    gt_kps.setdefault(img_id, []).append(kps)

# --- 2. Initialize YOLOv8-Pose model and run inference in streaming mode ---
model = YOLO(YOLO_MODEL)
results = model.predict(source=IMG_DIR, save=False, stream=True, verbose=False)

# Prepare lists to collect MPJPE values
mpjpe_list = []
mpjpe_norm_list = []

# --- 3. Iterate over each detection result and compute MPJPE ---
for r in tqdm(results, desc="Evaluating"):
    # Normalize filename to lowercase and extract basename
    fname = os.path.basename(r.path).lower()
    # Skip if this file is not in our ground truth mapping
    if fname not in fname2imgid:
        continue
    img_id = fname2imgid[fname]
    # Skip if no ground-truth keypoints for this image
    if img_id not in gt_kps or len(gt_kps[img_id]) == 0:
        continue
    gt_list = gt_kps[img_id]
    preds = r.keypoints.xy  # predicted keypoints for each detection

    if len(preds) == 0:
        continue  # no pose detected

    # Build cost matrices for Hungarian matching:
    #  - cost: raw MPJPE between each GT-person and each prediction
    #  - cost_norm: MPJPE normalized by the person's bbox height
    cost = np.zeros((len(gt_list), len(preds)))
    cost_norm = np.zeros((len(gt_list), len(preds)))

    for i, gt in enumerate(gt_list):
        gt_pts = gt[:17, :2]  # take only x,y for first 17 COCO keypoints
        # compute bounding‐box height for normalization
        y_valid = gt_pts[:, 1][~np.isnan(gt_pts[:, 1])]
        if len(y_valid) == 0:
            continue
        bbox_height = y_valid.max() - y_valid.min()
        if bbox_height < 1e-3:
            bbox_height = 1.0

        for j, pred in enumerate(preds):
            pred_pts = pred[:17, :2]
            # compute mean per-joint Euclidean distance
            mpjpe = np.linalg.norm(pred_pts - gt_pts, axis=1).mean()
            cost[i, j] = mpjpe
            cost_norm[i, j] = mpjpe / bbox_height

    # Solve assignment to match each GT person to one detection
    row_ind, col_ind = linear_sum_assignment(cost_norm)
    for i, j in zip(row_ind, col_ind):
        mpjpe_list.append(cost[i, j])
        mpjpe_norm_list.append(cost_norm[i, j])

# --- 4. Write per-person MPJPE results and summary to CSV ---
with open('mpjpe_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Header for individual matches
    writer.writerow(['MPJPE_pixel', 'MPJPE_normalized'])
    for px, norm in zip(mpjpe_list, mpjpe_norm_list):
        writer.writerow([px, norm])
    # Blank line separator
    writer.writerow([])
    # Summary statistics
    writer.writerow(['STATISTIC', 'MPJPE_pixel', 'MPJPE_normalized'])
    writer.writerow(['mean', np.mean(mpjpe_list), np.mean(mpjpe_norm_list)])
    writer.writerow(['median', np.median(mpjpe_list), np.median(mpjpe_norm_list)])
    writer.writerow(['min', np.min(mpjpe_list), np.min(mpjpe_norm_list)])
    writer.writerow(['max', np.max(mpjpe_list), np.max(mpjpe_norm_list)])

# --- 5. Print a concise evaluation summary ---
print(f"\nEvaluated {len(mpjpe_list)} matched persons:")
print(f"Mean MPJPE: {np.mean(mpjpe_list):.2f} px | normalized: {np.mean(mpjpe_norm_list):.3f}")
print(f"Median MPJPE: {np.median(mpjpe_list):.2f} px | normalized: {np.median(mpjpe_norm_list):.3f}")
print(f"MPJPE min/max: {np.min(mpjpe_list):.2f}/{np.max(mpjpe_list):.2f} px | normalized: {np.min(mpjpe_norm_list):.3f}/{np.max(mpjpe_norm_list):.3f}")
print("✅ Results saved in mpjpe_results.csv")
