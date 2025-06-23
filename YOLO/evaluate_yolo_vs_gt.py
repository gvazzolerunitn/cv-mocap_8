import json
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import os
from scipy.optimize import linear_sum_assignment
import csv

# === CONFIG ===
COCO_ANN = '../dataset/mocap_8.v1i.coco/train/_annotations.coco.json'
IMG_DIR = '../dataset/mocap_8.v1i.coco/train'
YOLO_MODEL = 'yolov8s-pose.pt'

# --- 1. Carica annotazioni COCO ---
with open(COCO_ANN) as f:
    coco = json.load(f)

fname2imgid = {img['file_name']: img['id'] for img in coco['images']}
gt_kps = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    kps = np.array(ann['keypoints']).reshape(-1,3)
    gt_kps.setdefault(img_id, []).append(kps)

model = YOLO(YOLO_MODEL)
results = model.predict(source=IMG_DIR, save=False, stream=True, verbose=False)

mpjpe_list = []
mpjpe_norm_list = []

for r in tqdm(results, desc="Evaluating"):
    fname = os.path.basename(r.path).lower()
    if fname not in fname2imgid:
        continue
    img_id = fname2imgid[fname]
    if img_id not in gt_kps:
        continue
    gt_list = gt_kps[img_id]
    preds = r.keypoints.xy
    if len(gt_list) == 0 or len(preds) == 0:
        continue

    cost = np.zeros((len(gt_list), len(preds)))
    cost_norm = np.zeros((len(gt_list), len(preds)))
    for i, gt in enumerate(gt_list):
        gt_pts = gt[:17, :2]
        y_valid = gt_pts[:, 1][~np.isnan(gt_pts[:, 1])]
        if len(y_valid) == 0:
            continue
        bbox_height = y_valid.max() - y_valid.min()
        if bbox_height < 1e-3:
            bbox_height = 1.0
        for j, pred in enumerate(preds):
            pred_pts = pred[:17, :2]
            mpjpe = np.linalg.norm(pred_pts - gt_pts, axis=1).mean()
            cost[i, j] = mpjpe
            cost_norm[i, j] = mpjpe / bbox_height

    row_ind, col_ind = linear_sum_assignment(cost_norm)
    for i, j in zip(row_ind, col_ind):
        mpjpe_list.append(cost[i, j])
        mpjpe_norm_list.append(cost_norm[i, j])

# Salva in CSV
with open('mpjpe_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['MPJPE_pixel', 'MPJPE_normalized'])
    for px, norm in zip(mpjpe_list, mpjpe_norm_list):
        writer.writerow([px, norm])
    # Riga vuota per separare
    writer.writerow([])
    # Statistiche
    writer.writerow(['STATISTIC', 'MPJPE_pixel', 'MPJPE_normalized'])
    writer.writerow(['mean', np.mean(mpjpe_list), np.mean(mpjpe_norm_list)])
    writer.writerow(['median', np.median(mpjpe_list), np.median(mpjpe_norm_list)])
    writer.writerow(['min', np.min(mpjpe_list), np.min(mpjpe_norm_list)])
    writer.writerow(['max', np.max(mpjpe_list), np.max(mpjpe_norm_list)])

print(f"\nValutazione su {len(mpjpe_list)} persone:")
print(f"MPJPE medio: {np.mean(mpjpe_list):.2f} px | normalizzato: {np.mean(mpjpe_norm_list):.3f}")
print(f"MPJPE mediano: {np.median(mpjpe_list):.2f} px | normalizzato: {np.median(mpjpe_norm_list):.3f}")
print(f"MPJPE min/max: {np.min(mpjpe_list):.2f} / {np.max(mpjpe_list):.2f} px | normalizzato: {np.min(mpjpe_norm_list):.3f} / {np.max(mpjpe_norm_list):.3f}")
print("Risultati salvati in mpjpe_results.csv")