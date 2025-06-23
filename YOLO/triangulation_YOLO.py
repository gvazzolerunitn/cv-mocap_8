import os, json
import numpy as np
import cv2

# --- CONFIG ---
CALIB_FOLDER = '../3D Pose Estimation Material/camera_data_with_Rvecs_2ndversion/Camera_config2'
YOLO_2D_PATH = 'yolo2d_for_triangulation.json'  # <-- file prodotto dallo script YOLO
OUTPUT_3D    = 'triangulated_positions_yolo.json'
TARGET_CAMS  = [2,5,8,13]

def load_calibrations(folder_path, target_cams):
    calibs = {}
    for item in os.listdir(folder_path):
        if not item.startswith('cam_'):
            continue
        cam_id = int(item.split('_')[1])
        if cam_id not in target_cams:
            continue
        item_path = os.path.join(folder_path, item)
        calib_file = None
        for root, _, files in os.walk(item_path):
            if 'camera_calib.json' in files:
                calib_file = os.path.join(root, 'camera_calib.json')
                break
        if calib_file is None:
            print(f"⚠️ calibrazione non trovata per cam_{cam_id}")
            continue
        with open(calib_file, 'r') as f:
            data = json.load(f)
        K    = np.array(data['mtx'])
        dist = np.array(data['dist']).flatten()
        rvec = np.array(data.get('rvecs', data.get('rvec'))).reshape(3)
        tvec = np.array(data.get('tvecs', data.get('tvec'))).reshape(3)
        calibs[cam_id] = {'K':K, 'dist':dist, 'rvec':rvec, 'tvec':tvec}
    return calibs

def build_projection_matrices(calibs, cams):
    Ps = {}
    for cam in cams:
        C = calibs[cam]
        R,_ = cv2.Rodrigues(C['rvec'])
        T    = C['tvec'].reshape(3,1)
        Ps[cam] = C['K'].dot( np.hstack((R,T)) )
    return Ps

def triangulate_point(Ps_list, pts2d_list):
    A = []
    for P, (u,v) in zip(Ps_list, pts2d_list):
        A.append( u*P[2] - P[0] )
        A.append( v*P[2] - P[1] )
    A = np.vstack(A)
    _,_,Vt = np.linalg.svd(A)
    X_hom  = Vt[-1]
    return X_hom / X_hom[3]

def main():
    # 1) Carica calibrazioni
    calibs = load_calibrations(CALIB_FOLDER, TARGET_CAMS)
    print(f"Available camera IDs: {sorted(calibs.keys())}")
    print(f"Looking for cameras: {TARGET_CAMS}")

    missing = [c for c in TARGET_CAMS if c not in calibs]
    if missing:
        print(f"Missing cameras: {missing}")
        return

    Ps = build_projection_matrices(calibs, TARGET_CAMS)

    # 2) Carica keypoint YOLO 2D
    with open(YOLO_2D_PATH) as f:
        frame_data = json.load(f)
    # frame_data[frame][cam][joint] = [x, y]

    num_joints = 17  # YOLOv8-pose COCO format

    out_3d = {}
    total_points = 0

    for f in frame_data.keys():
        out_3d[f] = {}
        frame_cams = frame_data[f]
        for j in range(num_joints):
            # Trova telecamere che hanno questo joint
            available_cams = [int(cam) for cam in TARGET_CAMS if str(cam) in frame_cams and (str(j) in frame_cams[str(cam)] or j in frame_cams[str(cam)])]
            if len(available_cams) >= 2:
                pts2d_list = []
                P_list = []
                for cam in available_cams:
                    joints = frame_cams[str(cam)]
                    pt = joints.get(str(j)) or joints.get(j)
                    if pt is not None:
                        pts2d_list.append(pt)
                        P_list.append(Ps[cam])
                if len(pts2d_list) >= 2:
                    X = triangulate_point(P_list, pts2d_list)
                    out_3d[f][j] = X[:3].tolist()
                    total_points += 1
                else:
                    out_3d[f][j] = None
            else:
                out_3d[f][j] = None

    # 4) Salva con statistiche migliori
    with open(OUTPUT_3D, 'w') as f:
        json.dump(out_3d, f, indent=2)

    print(f"Triangulated 3D saved to {OUTPUT_3D}")
    print(f"Frames processed: {len(out_3d)}")
    print(f"Total 3D points: {total_points}")
    if len(out_3d) > 0:
        print(f"Average points per frame: {total_points/len(out_3d):.1f}")

if __name__=='__main__':
    main()