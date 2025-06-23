import os
import json
import numpy as np
import cv2

# --- CONFIGURATION ---

# Path to the folder containing camera calibration subfolders (cam_2, cam_5, etc.)
CALIB_FOLDER = '../3D Pose Estimation Material/camera_data_with_Rvecs_2ndversion/Camera_config2'
# Path to the JSON file with 2D keypoints produced by YOLOv8-pose
YOLO_2D_PATH = 'yolo2d_for_triangulation.json'
# Output file where we will save the reconstructed 3D points
OUTPUT_3D    = 'triangulated_positions_yolo.json'
# List of camera IDs to use for triangulation
TARGET_CAMS  = [2, 5, 8, 13]

def load_calibrations(folder_path, target_cams):
    """
    Read each cam_<id> folder under folder_path, find camera_calib.json,
    and load the intrinsic (K), distortion, rotation (rvec), and translation (tvec)
    parameters for each camera in target_cams.
    Returns a dict: { cam_id: { 'K', 'dist', 'rvec', 'tvec' } }.
    """
    calibs = {}
    for item in os.listdir(folder_path):
        if not item.startswith('cam_'):
            continue
        cam_id = int(item.split('_')[1])
        if cam_id not in target_cams:
            continue

        # Search recursively for camera_calib.json
        item_path = os.path.join(folder_path, item)
        calib_file = None
        for root, _, files in os.walk(item_path):
            if 'camera_calib.json' in files:
                calib_file = os.path.join(root, 'camera_calib.json')
                break
        if calib_file is None:
            print(f"⚠️ Calibration file not found for cam_{cam_id}")
            continue

        # Load calibration values
        with open(calib_file, 'r') as f:
            data = json.load(f)
        K    = np.array(data['mtx'])                    # Camera intrinsic matrix
        dist = np.array(data['dist']).flatten()         # Distortion coefficients
        # Rotation and translation vectors may be in 'rvec' or 'rvecs'
        rvec = np.array(data.get('rvecs', data.get('rvec'))).reshape(3)
        tvec = np.array(data.get('tvecs', data.get('tvec'))).reshape(3)

        calibs[cam_id] = {'K': K, 'dist': dist, 'rvec': rvec, 'tvec': tvec}
    return calibs

def build_projection_matrices(calibs, cams):
    """
    Given calibration parameters for each camera, build the 3x4 projection matrix P = K [R | T].
    Returns dict: { cam_id: P }.
    """
    Ps = {}
    for cam in cams:
        C = calibs[cam]
        # Convert Rodrigues vector to rotation matrix
        R, _ = cv2.Rodrigues(C['rvec'])
        # Column vector for translation
        T = C['tvec'].reshape(3,1)
        # Projection matrix: K * [R | T]
        Ps[cam] = C['K'].dot(np.hstack((R, T)))
    return Ps

def triangulate_point(Ps_list, pts2d_list):
    """
    Given a list of projection matrices (Ps_list) and corresponding 2D points (pts2d_list),
    solve for the 3D point X by linear least-squares (SVD).
    Returns homogeneous 4-vector normalized to 3D.
    """
    A = []
    for P, (u, v) in zip(Ps_list, pts2d_list):
        # Each view contributes two equations: u*P[2] - P[0], and v*P[2] - P[1]
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.vstack(A)
    # Solve AX = 0 by SVD; solution is last row of V^T
    _, _, Vt = np.linalg.svd(A)
    X_hom = Vt[-1]
    # Convert homogeneous coordinate to 3D
    return X_hom / X_hom[3]

def main():
    # 1) Load camera calibrations
    calibs = load_calibrations(CALIB_FOLDER, TARGET_CAMS)
    print(f"Available camera IDs: {sorted(calibs.keys())}")
    print(f"Looking for cameras: {TARGET_CAMS}")

    missing = [c for c in TARGET_CAMS if c not in calibs]
    if missing:
        print(f"Missing cameras: {missing}")
        return

    # Build the projection matrices for each target camera
    Ps = build_projection_matrices(calibs, TARGET_CAMS)

    # 2) Load YOLOv8 2D keypoints for each frame and camera
    with open(YOLO_2D_PATH) as f:
        frame_data = json.load(f)
    # frame_data: { frame_str: { cam_str: { joint_str: [x,y], ... }, ... }, ... }

    num_joints = 17  # COCO-format keypoints from YOLOv8-pose

    out_3d = {}
    total_points = 0

    # Iterate over every frame in the YOLO output
    for f in frame_data.keys():
        out_3d[f] = {}
        frame_cams = frame_data[f]
        # For each possible joint index
        for j in range(num_joints):
            # Identify which cameras detected this joint in this frame
            available_cams = [
                cam for cam in TARGET_CAMS
                if str(cam) in frame_cams and (
                   str(j) in frame_cams[str(cam)] or j in frame_cams[str(cam)])
            ]
            if len(available_cams) >= 2:
                # Collect the valid 2D points and corresponding projection matrices
                pts2d_list = []
                P_list     = []
                for cam in available_cams:
                    joints = frame_cams[str(cam)]
                    pt = joints.get(str(j)) or joints.get(j)
                    if pt is not None:
                        pts2d_list.append(pt)
                        P_list.append(Ps[cam])
                # Only triangulate if at least two valid observations
                if len(pts2d_list) >= 2:
                    X = triangulate_point(P_list, pts2d_list)
                    out_3d[f][j] = X[:3].tolist()
                    total_points += 1
                else:
                    out_3d[f][j] = None
            else:
                out_3d[f][j] = None

    # 4) Save the reconstructed 3D positions to JSON
    with open(OUTPUT_3D, 'w') as f:
        json.dump(out_3d, f, indent=2)

    print(f"Triangulated 3D saved to {OUTPUT_3D}")
    print(f"Frames processed: {len(out_3d)}")
    print(f"Total 3D points: {total_points}")
    if len(out_3d) > 0:
        print(f"Average points per frame: {total_points / len(out_3d):.1f}")

if __name__ == '__main__':
    main()
