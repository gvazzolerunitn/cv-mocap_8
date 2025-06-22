import json
import numpy as np
import csv

YOLO_3D_PATH = 'triangulated_positions_yolo.json'
GT_3D_PATH   = '../Triangulation and reprojection and 3D/triangulated_positions_v2.json'  # aggiorna se serve
NUM_JOINTS = 17  # o 18 se il GT ne ha uno in più

with open(YOLO_3D_PATH) as f:
    yolo_3d = json.load(f)
with open(GT_3D_PATH) as f:
    gt_3d = json.load(f)

mpjpe_list = []
per_joint_errors = [[] for _ in range(NUM_JOINTS)]

common_frames = set(yolo_3d.keys()) & set(gt_3d.keys())
print(f"Frame in comune: {sorted(common_frames)}")

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
        mpjpe_list.append(err)
        per_joint_errors[j].append(err)

if mpjpe_list:
    print(f"\nValutazione su {len(common_frames)} frame e {len(mpjpe_list)} joint:")
    print(f"MPJPE medio: {np.mean(mpjpe_list):.2f} mm")
    print(f"MPJPE mediano: {np.median(mpjpe_list):.2f} mm")
    print(f"MPJPE min/max: {np.min(mpjpe_list):.2f} / {np.max(mpjpe_list):.2f} mm")
    print("\nErrore medio per joint:")
    for j, errs in enumerate(per_joint_errors):
        if errs:
            print(f"  Joint {j}: {np.mean(errs):.2f} mm")
        else:
            print(f"  Joint {j}: N/A")
else:
    print("Nessun joint confrontabile trovato!")

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

print("Risultati salvati in mpjpe_3d_results.csv")