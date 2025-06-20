# reprojection.py

import json, re
import numpy as np
import cv2

# ——— utenti: percorsi dei file ———
CALIB_FOLDER  = '3D Pose Estimation Material/camera_data_with_Rvecs_2ndversion/Camera_config2'
ANN_PATH      = '_annotations_rectified.coco.json'  # Use rectified annotations
INPUT_3D      = 'triangulated_positions.json'
TARGET_CAMS   = [2,5,8,13]

# Import the updated function
from triangulation import load_calibrations

def main():
    # 1) Carica calibrazione
    calibs = load_calibrations(CALIB_FOLDER, TARGET_CAMS)
    Ks = {c:calibs[c]['K'] for c in TARGET_CAMS}
    dists = {c:calibs[c]['dist'] for c in TARGET_CAMS}
    rvecs = {c:calibs[c]['rvec'] for c in TARGET_CAMS}
    tvecs = {c:calibs[c]['tvec'] for c in TARGET_CAMS}

    # 2) Carica 3D e annotazioni
    with open(INPUT_3D) as f:
        points3d = json.load(f)
    with open(ANN_PATH) as f:
        coco = json.load(f)
    images = {img['id']:img for img in coco['images']}

    # ricostruisci frame_data come in triangulation.py
    frame_data = {}
    for ann in coco['annotations']:
        img = images[ann['image_id']]
        name = img['extra']['name']
        m = re.match(r'out(\d+)_frame_(\d+)\.png', name)
        if not m:
            continue
        cam = int(m.group(1)); frame = int(m.group(2))
        kps = np.array(ann['keypoints']).reshape(-1,3)
        pts2d = {i:(float(x),float(y)) for i,(x,y,v) in enumerate(kps) if v>0}
        frame_data.setdefault(frame, {})[cam] = pts2d

    # 3) Calcola errori di reproiezione
    errors = []
    for f_str, joints in points3d.items():
        f = int(f_str)  # Convert string key to int
        if f not in frame_data:
            continue
            
        for j_str, X in joints.items():
            j = int(j_str)  # Convert string key to int
            X = np.array(X, dtype=float).reshape(1,3)
            
            for cam in TARGET_CAMS:
                if cam in frame_data[f] and j in frame_data[f][cam]:
                    uv_true = np.array(frame_data[f][cam][j])
                    uv_proj,_ = cv2.projectPoints(
                        X, rvecs[cam], tvecs[cam], Ks[cam], None # Changed from dists[cam] (it applied          
                    )                                            # distortion twice this way)
                    uvp = uv_proj.reshape(2)
                    err = np.linalg.norm(uvp - uv_true)
                    errors.append(err)

    errors = np.array(errors)
    if len(errors) > 0:
        mpjpe = np.mean(errors)
        mse   = np.mean(errors**2)
        rmse  = np.sqrt(mse)

        print(f"Reprojection errors over {len(errors)} points:")
        print(f" • MPJPE = {mpjpe:.3f} px")
        print(f" • RMSE  = {rmse:.3f} px")
        print(f" • MSE   = {mse:.3f} px²")
        print(f" • Max   = {np.max(errors):.3f} px")
        print(f" • Min   = {np.min(errors):.3f} px")
    else:
        print("No valid reprojection points found!")

if __name__ == '__main__':
    main()