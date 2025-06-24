import os
import json
import numpy as np
import cv2

def triangulate_with_corrected_data():
    """Triangola usando i dati corretti"""
    # Usa i dati YOLO riscalati
    with open('yolo2d_for_triangulation_rescaled.json') as f:
        frame_data = json.load(f)
    
    # Carica calibrazioni originali (ora compatibili)
    calib_folder = '../3D Pose Estimation Material/camera_data_with_Rvecs_2ndversion/Camera_config2'
    TARGET_CAMS = [2, 5, 8, 13]
    
    # Carica matrici di proiezione
    Ps = {}
    for cam_id in TARGET_CAMS:
        cam_path = os.path.join(calib_folder, f'cam_{cam_id}')
        for root, dirs, files in os.walk(cam_path):
            if 'camera_calib.json' in files:
                calib_file = os.path.join(root, 'camera_calib.json')
                with open(calib_file) as f:
                    calib = json.load(f)
                
                K = np.array(calib['mtx'])
                rvec = np.array(calib.get('rvecs', calib.get('rvec', []))).flatten()
                tvec = np.array(calib.get('tvecs', calib.get('tvec', []))).flatten()
                
                R, _ = cv2.Rodrigues(rvec)
                P = K @ np.hstack([R, tvec.reshape(-1, 1)])
                Ps[cam_id] = P
                print(f"Loaded calibration for cam {cam_id}")
    
    # Triangolazione migliorata
    def triangulate_point_robust(Ps_list, pts2d_list):
        if len(pts2d_list) < 2:
            return None
        
        A = []
        for pt2d, P in zip(pts2d_list, Ps_list):
            x, y = pt2d
            # Fix: Costruisci correttamente le righe della matrice A
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        
        A = np.array(A)
        
        # SVD per trovare la soluzione
        try:
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1, :]
            
            # Fix: gestione corretta dell'elemento scalare
            if len(X) < 4:
                return None
                
            w = X[3]
            if isinstance(w, np.ndarray):
                w = w.item()  # Converte array 0D a scalare
            
            if abs(w) < 1e-8:
                return None
            
            point_3d = X[:3] / w
            
            # Validazione: rigetta punti troppo lontani
            if np.linalg.norm(point_3d) > 5000:  # 5 metri
                return None
            
            return point_3d.tolist()
            
        except Exception as e:
            print(f"Errore nella triangolazione: {e}")
            return None
    
    # Esegui triangolazione
    out_3d = {}
    total_points = 0
    
    for frame_id in frame_data.keys():
        out_3d[frame_id] = {}
        frame_cams = frame_data[frame_id]
        
        for joint_id in range(17):  # 17 keypoints COCO
            pts2d_list = []
            P_list = []
            
            for cam_id in TARGET_CAMS:
                cam_str = str(cam_id)
                joint_str = str(joint_id)
                
                if (cam_str in frame_cams and
                    joint_str in frame_cams[cam_str] and
                    frame_cams[cam_str][joint_str] is not None):
                    
                    pt = frame_cams[cam_str][joint_str]
                    if len(pt) == 2:
                        pts2d_list.append(pt)
                        P_list.append(Ps[cam_id])
            
            # Triangola se hai almeno 2 osservazioni
            if len(pts2d_list) >= 2:
                pt3d = triangulate_point_robust(P_list, pts2d_list)
                if pt3d is not None:
                    out_3d[frame_id][joint_str] = pt3d
                    total_points += 1
    
    # Salva risultati
    with open('triangulated_positions_corrected.json', 'w') as f:
        json.dump(out_3d, f, indent=2)
    
    print(f"Triangolazione completata. {total_points} punti 3D generati.")
    print("Risultati salvati in 'triangulated_positions_corrected.json'")

if __name__ == "__main__":
    triangulate_with_corrected_data()