# rectify_annotations.py

import json
import re
import zipfile

import cv2
import numpy as np

# ------------------------------------------------------------------
# 1) Carica tutte le calibrazioni dal tuo ZIP
# ------------------------------------------------------------------
def load_calibrations_from_zip(zip_path):
    """
    Restituisce un dict:
      { cam_id: { 'mtx': (3×3), 'dist': (5,) } }
    estraendo ogni camera_calib.json dentro il ZIP.
    """
    calibs = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        for fname in z.namelist():
            if fname.endswith('camera_calib.json'):
                m = re.search(r'cam_(\d+)', fname)
                if not m:
                    continue
                cam_id = int(m.group(1))
                data = json.loads(z.read(fname).decode())
                mtx  = np.array(data['mtx'],  dtype=np.float32)
                # dist arriva come [[k1,k2,p1,p2,k3]]
                dist = np.array(data['dist'], dtype=np.float32).reshape(-1)[0:5]
                calibs[cam_id] = {'mtx': mtx, 'dist': dist}
    return calibs


# ------------------------------------------------------------------
# 2) Rettifica le annotazioni COCO
# ------------------------------------------------------------------
def rectify_annotations(coco_in, calib_zip, coco_out):
    # carica file COCO
    coco = json.load(open(coco_in, 'r'))
    calibs = load_calibrations_from_zip(calib_zip)

    # index rapido per le immagini
    img_index = {img['id']: img for img in coco['images']}

    for ann in coco['annotations']:
        img = img_index[ann['image_id']]
        name = img['extra']['name']  # es. "out5_frame_0016.png"
        m = re.match(r'out(\d+)_frame_\d+\.png', name)
        if not m:
            continue
        cam = int(m.group(1))

        # prendi calibrazione di questa camera
        K    = calibs[cam]['mtx']
        dist = calibs[cam]['dist']

        # estrai keypoints in Nx3 (x, y, visibility)
        kps = np.array(ann['keypoints'], dtype=float).reshape(-1, 3)
        pts = kps[:, :2]
        vis = kps[:, 2].astype(bool)

        # rettifica solo i punti visibili
        if vis.any():
            pts_vis = pts[vis].astype(np.float32).reshape(-1, 1, 2)
            # undistortPoints + P=K riporta in pixel
            und = cv2.undistortPoints(pts_vis, K, dist, P=K)
            pts[vis] = und.reshape(-1, 2)

        # riscrivi nel formato COCO [x,y,v,...]
        new_kps = []
        for (x, y), v_flag in zip(pts, vis):
            new_kps.extend([float(x), float(y), int(v_flag)])
        ann['keypoints'] = new_kps

    # salva nuovo file COCO rettificato
    with open(coco_out, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"Annotazioni rettificate salvate in: {coco_out}")


# ------------------------------------------------------------------
# 3) Punto di ingresso
# ------------------------------------------------------------------
if __name__ == '__main__':
    rectify_annotations(
        coco_in   = 'dataset/mocap_8.v1i.coco/train/_annotations.coco.json',  # Updated path
        calib_zip = '3D Pose Estimation Material/camera_data_with_Rvecs_2ndversion.zip',  # Updated path
        coco_out  = '_annotations_rectified.coco.json'
    )
