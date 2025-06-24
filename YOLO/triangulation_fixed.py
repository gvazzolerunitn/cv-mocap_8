import os
import json
import numpy as np
import cv2

def triangulate_with_detailed_debug():
    """Triangolazione con debug dettagliato e soglie più permissive"""
    
    # Carica dati rescalati
    with open('yolo2d_for_triangulation_rescaled.json') as f:
        frame_data = json.load(f)
    
    print(f"✅ Caricati {len(frame_data)} frame")
    
    # Carica calibrazioni
    calib_folder = '../3D Pose Estimation Material/camera_data_with_Rvecs_2ndversion/Camera_config2'
    TARGET_CAMS = [2, 5, 8, 13]
    
    Ps = {}
    for cam_id in TARGET_CAMS:
        cam_path = os.path.join(calib_folder, f'cam_{cam_id}')
        calib_file = None
        
        for root, dirs, files in os.walk(cam_path):
            if 'camera_calib.json' in files:
                calib_file = os.path.join(root, 'camera_calib.json')
                break
        
        if calib_file and os.path.exists(calib_file):
            with open(calib_file) as f:
                calib = json.load(f)
            
            K = np.array(calib['mtx'])
            rvec = np.array(calib.get('rvecs', calib.get('rvec', []))).flatten()
            tvec = np.array(calib.get('tvecs', calib.get('tvec', []))).flatten()
            
            R, _ = cv2.Rodrigues(rvec)
            P = K @ np.hstack([R, tvec.reshape(-1, 1)])
            Ps[cam_id] = P
            print(f"✅ Calibrazione caricata per cam {cam_id}")
        else:
            print(f"❌ Calibrazione non trovata per cam {cam_id}")
    
    print(f"📊 Calibrazioni disponibili: {list(Ps.keys())}")
    
    # Funzione di triangolazione più robusta
    def triangulate_point_permissive(Ps_list, pts2d_list):
        """Triangolazione più permissiva"""
        if len(pts2d_list) < 2:
            return None
        
        A = []
        for pt2d, P in zip(pts2d_list, Ps_list):
            x, y = pt2d
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        
        A = np.array(A)
        
        try:
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1, :]
            
            if len(X) < 4:
                return None
            
            w = X[3]
            if isinstance(w, np.ndarray):
                w = w.item()
            
            if abs(w) < 1e-10:  # Soglia più permissiva
                return None
            
            point_3d = X[:3] / w
            
            # Validazione meno restrittiva
            norm = np.linalg.norm(point_3d)
            if norm > 20000 or norm < 0.1:  # 20 metri max, 10cm min
                return None
            
            return point_3d.tolist()
            
        except Exception as e:
            return None
    
    # Analizza frame di esempio
    sample_frame = "1"
    print(f"\n🔍 Analisi dettagliata frame {sample_frame}:")
    
    frame_cams = frame_data[sample_frame]
    available_cams = list(frame_cams.keys())
    print(f"Camere disponibili nel frame: {available_cams}")
    
    # Conta keypoint validi per camera
    for cam_str in available_cams:
        if cam_str in frame_cams:
            cam_data = frame_cams[cam_str]
            valid_joints = []
            for joint_str, coords in cam_data.items():
                if coords and len(coords) == 2 and not (coords[0] == 0.0 and coords[1] == 0.0):
                    valid_joints.append(joint_str)
            print(f"  Cam {cam_str}: {len(valid_joints)}/17 keypoint validi: {valid_joints}")
    
    # Triangolazione per ogni keypoint
    print(f"\n🎯 Tentativo triangolazione keypoint:")
    
    triangulated_count = 0
    out_3d = {}
    
    # Processa tutti i frame
    for frame_id in frame_data.keys():
        out_3d[frame_id] = {}
        frame_cams = frame_data[frame_id]
        
        for joint_id in range(17):
            pts2d_list = []
            P_list = []
            cam_list = []
            
            for cam_id in TARGET_CAMS:
                cam_str = str(cam_id)
                joint_str = str(joint_id)
                
                # Verifica se abbiamo dati per questa camera e joint
                if (cam_str in frame_cams and 
                    joint_str in frame_cams[cam_str] and
                    frame_cams[cam_str][joint_str] is not None and
                    cam_id in Ps):
                    
                    coords = frame_cams[cam_str][joint_str]
                    
                    # Filtra coordinate [0.0, 0.0] - non rilevate
                    if (isinstance(coords, list) and 
                        len(coords) == 2 and
                        not (coords[0] == 0.0 and coords[1] == 0.0)):
                        
                        pts2d_list.append(coords)
                        P_list.append(Ps[cam_id])
                        cam_list.append(cam_id)
            
            # Triangola se abbiamo almeno 2 osservazioni valide
            if len(pts2d_list) >= 2:
                pt3d = triangulate_point_permissive(P_list, pts2d_list)
                if pt3d is not None:
                    out_3d[frame_id][str(joint_id)] = pt3d
                    triangulated_count += 1
                    
                    # Debug per il primo frame
                    if frame_id == sample_frame:
                        print(f"  Joint {joint_id}: ✅ triangolato con {len(cam_list)} cams {cam_list}")
                else:
                    if frame_id == sample_frame:
                        print(f"  Joint {joint_id}: ❌ triangolazione fallita ({len(cam_list)} cams: {cam_list})")
            else:
                if frame_id == sample_frame:
                    print(f"  Joint {joint_id}: ❌ insufficienti osservazioni ({len(pts2d_list)} cams)")
    
    # Salva risultati
    with open('triangulated_positions_corrected.json', 'w') as f:
        json.dump(out_3d, f, indent=2)
    
    print(f"\n📊 Risultati finali:")
    print(f"   Total punti 3D triangolati: {triangulated_count}")
    print(f"   Media per frame: {triangulated_count/len(frame_data):.1f}")
    print(f"   File salvato: triangulated_positions_corrected.json")
    
    # Verifica risultato
    frame_with_data = 0
    for frame_id, frame_joints in out_3d.items():
        if len(frame_joints) > 0:
            frame_with_data += 1
    
    print(f"   Frame con dati: {frame_with_data}/{len(frame_data)}")
    
    return out_3d

if __name__ == "__main__":
    result = triangulate_with_detailed_debug()