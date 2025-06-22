import os, re, json
import numpy as np
import cv2

# ——— utenti: aggiorna questi percorsi se serve ———
CALIB_FOLDER = '../3D Pose Estimation Material/camera_data_with_Rvecs_2ndversion/Camera_config2'
ANN_PATH     = '../Annotations/_annotations_rectified_v2.coco.json'
OUTPUT_3D    = 'triangulated_positions_window.json'
TARGET_CAMS  = [2,5,8,13]

def load_calibrations(folder_path, target_cams):
    """Legge tutti i file camera_calib.json dentro le cartelle cam_<id> (anche dentro calib/) 
       e restituisce { cam_id: {'K','dist','rvec','tvec'} } per le cam in target_cams."""
    calibs = {}

    for item in os.listdir(folder_path):
        if not item.startswith('cam_'):
            continue

        cam_id = int(item.split('_')[1])
        if cam_id not in target_cams:
            continue

        # scorri ricorsivamente per trovare camera_calib.json
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
        # rvec/tvec possono trovarsi come singoli o liste
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

def analyze_annotations(coco, images):
    """Analizza le annotazioni per capire il formato dei keypoints"""
    print("\n=== ANALISI ANNOTAZIONI ===")
    
    debug_count = 0
    for ann in coco['annotations'][:5]:  # Primi 5 per debug
        img = images[ann['image_id']]
        name = img['extra']['name']
        kps = np.array(ann['keypoints']).reshape(-1,3)
        
        print(f"\nImmagine: {name}")
        for j, (x, y, v) in enumerate(kps[:5]):  # Primi 5 joint
            print(f"  Joint {j}: x={x:.1f}, y={y:.1f}, visibility={v}")
        debug_count += 1

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

    # 2) Carica annotazioni COCO
    with open(ANN_PATH) as f:
        coco = json.load(f)
    images = {img['id']:img for img in coco['images']}

    # ricava num_joints: prima prova da categories, altrimenti da annotazioni
    if 'categories' in coco and coco['categories'] and 'keypoints' in coco['categories'][0]:
        num_joints = len(coco['categories'][0]['keypoints'])
    else:
        num_joints = len(coco['annotations'][0]['keypoints']) // 3

    # Aggiungi analisi debug
    analyze_annotations(coco, images)

    frame_data = {}
    for ann in coco['annotations']:
        img = images[ann['image_id']]
        name = img['extra']['name']
        m = re.match(r'out(\d+)_frame_(\d+)\.png', name)
        if not m:
            continue
        cam   = int(m.group(1))
        frame = int(m.group(2))
        kps = np.array(ann['keypoints']).reshape(-1,3)
        
        # NUOVA STRATEGIA: Usa tutti i keypoint
        # Dato che hai annotato manualmente tutto, sono tutti validi
        pts2d = {i:(float(x),float(y)) for i,(x,y,v) in enumerate(kps)}
        
        frame_data.setdefault(frame, {})[cam] = pts2d 

    # 3) STRATEGIA PIÙ FLESSIBILE: usa almeno 2 telecamere
    out_3d = {}
    total_points = 0
    
    for f in frame_data.keys():
        out_3d[f] = {}
        frame_cams = frame_data[f]
        
        for j in range(num_joints):
            # Trova telecamere che hanno questo joint
            available_cams = [cam for cam in TARGET_CAMS 
                            if cam in frame_cams and j in frame_cams[cam]]
            
            # Triangola con almeno 2 telecamere
            if len(available_cams) >= 2:
                pts2d_list = [frame_cams[cam][j] for cam in available_cams]
                P_list = [Ps[cam] for cam in available_cams]
                X = triangulate_point(P_list, pts2d_list)
                out_3d[f][j] = X[:3].tolist()
                total_points += 1

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
