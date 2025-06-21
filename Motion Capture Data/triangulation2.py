import os, re, json, argparse
import numpy as np
import cv2

# ——— Parse CLI arguments ———
def parse_args():
    p = argparse.ArgumentParser(description="Triangulate 3D points from multi-view annotations")
    p.add_argument('--calib_folder', required=True, help='Path to calibration folder')
    p.add_argument('--ann_path',    required=True, help='Path to COCO annotations')
    p.add_argument('--output_3d',   required=True, help='Path to save output JSON')
    p.add_argument('--cams',        required=True, type=int, nargs='+', help='List of camera IDs to use')
    p.add_argument('--start-frame', type=int, help='Only triangulate frames >= START_FRAME')
    p.add_argument('--end-frame',   type=int, help='Only triangulate frames <= END_FRAME')
    p.add_argument('--verbose',     action='store_true', help='Enable debug prints')
    return p.parse_args()

# ——— Calibration loader ———
def load_calibrations(folder_path, target_cams):
    calibs = {}
    for item in os.listdir(folder_path):
        if not item.startswith('cam_'): continue
        cam_id = int(item.split('_')[1])
        if cam_id not in target_cams: continue
        # find camera_calib.json
        calib_file = None
        for root, _, files in os.walk(os.path.join(folder_path, item)):
            if 'camera_calib.json' in files:
                calib_file = os.path.join(root, 'camera_calib.json')
                break
        if calib_file is None:
            print(f"⚠️ calibrazione non trovata per cam_{cam_id}")
            continue
        data = json.load(open(calib_file))
        K    = np.array(data['mtx'])
        dist = np.array(data['dist']).flatten()
        rvec = np.array(data.get('rvecs', data.get('rvec'))).reshape(3)
        tvec = np.array(data.get('tvecs', data.get('tvec'))).reshape(3)
        calibs[cam_id] = {'K':K, 'dist':dist, 'rvec':rvec, 'tvec':tvec}
    return calibs

# ——— Build projection matrices ———
def build_projection_matrices(calibs, cams):
    Ps = {}
    for cam in cams:
        C = calibs[cam]
        R,_ = cv2.Rodrigues(C['rvec'])
        T    = C['tvec'].reshape(3,1)
        Ps[cam] = C['K'] @ np.hstack((R, T))
    return Ps

# ——— Triangulate one point ———
def triangulate_point(Ps_list, pts2d_list):
    A = []
    for P, (u,v) in zip(Ps_list, pts2d_list):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.vstack(A)
    _,_,Vt = np.linalg.svd(A)
    X_hom  = Vt[-1]
    return X_hom / X_hom[3]

# ——— Debug print of annotations ———
def analyze_annotations(coco, images):
    print("\n=== ANALISI ANNOTAZIONI ===")
    for ann in coco['annotations'][:5]:
        img = images[ann['image_id']]
        name = img['extra']['name']
        kps = np.array(ann['keypoints']).reshape(-1,3)
        print(f"\nImmagine: {name}")
        for j,(x,y,v) in enumerate(kps[:5]):
            print(f"  Joint {j}: x={x:.1f}, y={y:.1f}, visibility={v}")

# ——— Main triangulation ———
def main(args):
    # load calibrations
    calibs = load_calibrations(args.calib_folder, args.cams)
    print(f"Available camera IDs: {sorted(calibs.keys())}")
    print(f"Looking for cameras: {args.cams}")
    missing = [c for c in args.cams if c not in calibs]
    if missing:
        print(f"Missing cameras: {missing}"); return
    Ps = build_projection_matrices(calibs, args.cams)
    # load annotations
    coco = json.load(open(args.ann_path))
    images = {img['id']:img for img in coco['images']}
    if args.verbose: analyze_annotations(coco, images)
    # prepare frame_data
    frame_data = {}
    for ann in coco['annotations']:
        img = images[ann['image_id']]
        m = re.match(r'out(\d+)_frame_(\d+)\.png', img['extra']['name'])
        if not m: continue
        cam = int(m.group(1)); frame = int(m.group(2))
        # filter by frame range
        if args.start_frame and frame < args.start_frame: continue
        if args.end_frame   and frame > args.end_frame:   continue
        kps = np.array(ann['keypoints']).reshape(-1,3)
        pts2d = {i:(float(x),float(y)) for i,(x,y,v) in enumerate(kps)}
        frame_data.setdefault(frame, {})[cam] = pts2d
    # triangulate
    out_3d = {}
    total_points = 0
    for f, cams_dict in sorted(frame_data.items()):
        if len(cams_dict) >= 2:
            out_3d[f] = {}
            for j in range(len(cams_dict[args.cams[0]])):
                cams_avail = [c for c in args.cams if j in cams_dict[c]]
                if len(cams_avail) >= 2:
                    pts2d_list = [cams_dict[c][j] for c in cams_avail]
                    P_list     = [Ps[c] for c in cams_avail]
                    X = triangulate_point(P_list, pts2d_list)
                    out_3d[f][j] = X[:3].tolist()
                    total_points += 1
    # save
    with open(args.output_3d, 'w') as f:
        json.dump(out_3d, f, indent=2)
    print(f"Triangulated 3D saved to {args.output_3d}")
    print(f"Frames processed: {len(out_3d)}")
    print(f"Total 3D points: {total_points}")
    if len(out_3d)>0:
        print(f"Average points/frame: {total_points/len(out_3d):.1f}")

if __name__=='__main__':
    args = parse_args()
    main(args)

#run using:
# python triangulation2.py --calib_folder "../3D Pose Estimation Material/camera_data_with_Rvecs_2ndversion/Camera_config2" --ann_path "../Annotations/_annotations_rectified_v2.coco.json" --cams 2 5 8 13 --start-frame 0 --end-frame 47 --output_3d triangulated_positions_window.json --verbose