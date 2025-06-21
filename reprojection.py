# reprojection.py

import json, re
import numpy as np
import cv2
import csv
from datetime import datetime
import os

# ——— utenti: percorsi dei file ———
CALIB_FOLDER  = '3D Pose Estimation Material/camera_data_with_Rvecs_2ndversion/Camera_config2'
ANN_PATH      = '_annotations_rectified_v2.coco.json'  # Use rectified annotations
INPUT_3D      = 'triangulated_positions_v2.json'
TARGET_CAMS   = [2,5,8,13]

# Import the updated function
from triangulation import load_calibrations

def save_results_to_csv(errors, target_cams, calib_folder, ann_path, input_3d, output_file='reprojection_results.csv'):
    """
    Save reprojection analysis results to a well-formatted CSV file.
    
    Args:
        errors: numpy array of reprojection errors
        target_cams: list of camera IDs used
        calib_folder: path to calibration folder
        ann_path: path to annotations file
        input_3d: path to 3D points file
        output_file: output CSV filename
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header section
        writer.writerow(['REPROJECTION ERROR ANALYSIS RESULTS'])
        writer.writerow(['Generated on:', timestamp])
        writer.writerow([])
        
        # Parameters section
        writer.writerow(['PARAMETERS'])
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['Calibration Folder', calib_folder])
        writer.writerow(['Annotations File', ann_path])
        writer.writerow(['3D Points File', input_3d])
        writer.writerow(['Target Cameras', ', '.join(map(str, target_cams))])
        writer.writerow(['Number of Cameras', len(target_cams)])
        writer.writerow([])
        
        # Data statistics section
        writer.writerow(['DATA STATISTICS'])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Points Analyzed', len(errors)])
        writer.writerow(['Points per Camera (avg)', f"{len(errors) / len(target_cams):.1f}"])
        writer.writerow([])
        
        # Results section
        if len(errors) > 0:
            mpjpe = np.mean(errors)
            mse = np.mean(errors**2)
            rmse = np.sqrt(mse)
            std_err = np.std(errors)
            median_err = np.median(errors)
            q25 = np.percentile(errors, 25)
            q75 = np.percentile(errors, 75)
            
            writer.writerow(['REPROJECTION ERROR RESULTS'])
            writer.writerow(['Metric', 'Value (pixels)', 'Description'])
            writer.writerow(['MPJPE', f"{mpjpe:.4f}", 'Mean Per Joint Position Error'])
            writer.writerow(['RMSE', f"{rmse:.4f}", 'Root Mean Square Error'])
            writer.writerow(['MSE', f"{mse:.4f}", 'Mean Square Error'])
            writer.writerow(['Standard Deviation', f"{std_err:.4f}", 'Standard deviation of errors'])
            writer.writerow(['Median Error', f"{median_err:.4f}", 'Median reprojection error'])
            writer.writerow(['25th Percentile', f"{q25:.4f}", '25% of errors below this value'])
            writer.writerow(['75th Percentile', f"{q75:.4f}", '75% of errors below this value'])
            writer.writerow(['Maximum Error', f"{np.max(errors):.4f}", 'Largest reprojection error'])
            writer.writerow(['Minimum Error', f"{np.min(errors):.4f}", 'Smallest reprojection error'])
            writer.writerow([])
            
            # Error distribution section
            writer.writerow(['ERROR DISTRIBUTION'])
            writer.writerow(['Error Range (px)', 'Count', 'Percentage'])
            
            # Define error ranges
            ranges = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, float('inf'))]
            for min_err, max_err in ranges:
                if max_err == float('inf'):
                    count = np.sum(errors >= min_err)
                    range_str = f"≥ {min_err}"
                else:
                    count = np.sum((errors >= min_err) & (errors < max_err))
                    range_str = f"{min_err} - {max_err}"
                
                percentage = (count / len(errors)) * 100
                writer.writerow([range_str, count, f"{percentage:.1f}%"])
            
        else:
            writer.writerow(['RESULTS'])
            writer.writerow(['Status', 'No valid reprojection points found'])
        
        writer.writerow([])
        writer.writerow(['END OF REPORT'])
    
    print(f"Results saved to: {output_file}")

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
        # ...existing code...
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
        
        # Save results to CSV
        save_results_to_csv(errors, TARGET_CAMS, CALIB_FOLDER, ANN_PATH, INPUT_3D)
    else:
        print("No valid reprojection points found!")
        # Save empty results to CSV
        save_results_to_csv(np.array([]), TARGET_CAMS, CALIB_FOLDER, ANN_PATH, INPUT_3D)

if __name__ == '__main__':
    main()