#!/usr/bin/env python3
"""
Rectify COCO key‑points using camera calibration parameters.

Differenze rispetto alla prima versione
--------------------------------------
* Stesso core (visibilità intatta, rettifica completa, bbox/area aggiornati).
* **Nessuna gestione/estrazione di ZIP**: `--calib_dir` deve puntare a una cartella già scompattata
  in cui esistono sottocartelle `cam_<id>/calib/camera_calib.json`.
* Docstring e messaggi di aiuto aggiornati di conseguenza.
* **Usa cv2.remap** per consistenza con la rettifica video.

Utilizzo
--------
```bash
python rectify_annotationsV2.py \
    --coco _annotations.coco.json \
    --calib_dir camera_data_with_Rvecs \
    --output _annotations_rectified_v2.coco.json
```

Puoi lanciare lo script più volte puntando, di volta in volta, alla cartella estratta
di ciascuna calibrazione (prima versione, seconda versione, …) per confrontare l'errore
medio di riproiezione.
"""

import json
import os
import re
import argparse
import warnings
from typing import Dict, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def infer_cam_id(filename: str) -> int:
    """Infer the camera id from an image filename or path.

    Supported patterns (case-insensitive):
        …/cam_8/…          → 8
        out5_frame_012.png → 5
    Raises if no id can be found so the dataset can be fixed early.
    """
    patterns = [r"cam[_-]?(\d+)", r"out(\d+)_"]
    for p in patterns:
        m = re.search(p, filename, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    raise ValueError(f"Cannot infer camera id from filename: {filename}")


def load_calibrations(root: str) -> Dict[int, Dict[str, np.ndarray]]:
    """Recursively load every `camera_calib.json` found under *root*.

    Returns
    -------
    dict
        Mapping *cam_id → {"K": 3×3 float32, "dist": (N,) float32}*
    """
    calibs: Dict[int, Dict[str, np.ndarray]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        if "camera_calib.json" not in filenames:
            continue
        try:
            cam_id = int(re.search(r"cam[_-]?(\d+)", dirpath, re.IGNORECASE).group(1))
        except AttributeError:
            warnings.warn(f"Cannot parse camera id from path: {dirpath}. Skipped.")
            continue
        with open(os.path.join(dirpath, "camera_calib.json"), "r") as f:
            data = json.load(f)
        K = np.asarray(data["mtx"], dtype=np.float32)
        dist = np.asarray(data["dist"], dtype=np.float32).reshape(-1)
        calibs[cam_id] = {"K": K, "dist": dist}
    if not calibs:
        raise RuntimeError(f"No camera_calib.json found under {root}.")
    return calibs


def create_undistortion_map(width: int, height: int, K: np.ndarray, dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create undistortion maps for a given image size and camera parameters.
    
    This matches the approach used in rectified_videos.py and fix_rectified_videos.py
    """
    # Create a grid of (x,y) coordinates for each pixel
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    pts = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)  # shape requested by cv2.undistortPoints
    
    # Get undistorted points
    undistorted_pts = cv2.undistortPoints(pts, K, dist, P=K)
    undistorted_map = undistorted_pts.reshape(height, width, 2)
    
    map_x = undistorted_map[:, :, 0]  # new x-coordinates
    map_y = undistorted_map[:, :, 1]  # new y-coordinates
    
    return map_x, map_y


def rectify_points_with_map(points: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Rectify points using precomputed undistortion maps via bilinear interpolation."""
    rectified_points = []
    
    for x, y in points:
        # Convert to integer coordinates for map lookup
        x_int, y_int = int(x), int(y)
        
        # Check bounds
        if 0 <= x_int < map_x.shape[1] and 0 <= y_int < map_x.shape[0]:
            # Get the fractional parts for bilinear interpolation
            x_frac = x - x_int
            y_frac = y - y_int
            
            # Get the four surrounding pixels
            x0, y0 = x_int, y_int
            x1, y1 = min(x_int + 1, map_x.shape[1] - 1), min(y_int + 1, map_x.shape[0] - 1)
            
            # Bilinear interpolation
            new_x = ((1 - x_frac) * (1 - y_frac) * map_x[y0, x0] +
                     x_frac * (1 - y_frac) * map_x[y0, x1] +
                     (1 - x_frac) * y_frac * map_x[y1, x0] +
                     x_frac * y_frac * map_x[y1, x1])
            
            new_y = ((1 - x_frac) * (1 - y_frac) * map_y[y0, x0] +
                     x_frac * (1 - y_frac) * map_y[y0, x1] +
                     (1 - x_frac) * y_frac * map_y[y1, x0] +
                     x_frac * y_frac * map_y[y1, x1])
            
            rectified_points.append([new_x, new_y])
        else:
            # Point is outside image bounds, keep original
            rectified_points.append([x, y])
    
    return np.array(rectified_points)


# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------

def rectify_coco(coco: dict, calibs: Dict[int, Dict[str, np.ndarray]]) -> dict:
    img_lookup = {img["id"]: img for img in coco["images"]}
    
    # Cache undistortion maps per camera and image size
    undistortion_maps = {}

    for ann in tqdm(coco["annotations"], desc="Rectifying key‑points", unit="ann"):
        img = img_lookup[ann["image_id"]]
        try:
            cam_id = infer_cam_id(img["file_name"])
        except ValueError as e:
            warnings.warn(str(e) + f"  (annotation id {ann['id']})")
            continue

        if cam_id not in calibs:
            warnings.warn(f"Calibration for camera {cam_id} not found → ann id {ann['id']} skipped")
            continue

        K = calibs[cam_id]["K"]
        dist = calibs[cam_id]["dist"]
        
        # Get image dimensions
        width = img["width"]
        height = img["height"]
        
        # Create or get cached undistortion maps
        map_key = (cam_id, width, height)
        if map_key not in undistortion_maps:
            map_x, map_y = create_undistortion_map(width, height, K, dist)
            undistortion_maps[map_key] = (map_x, map_y)
        else:
            map_x, map_y = undistortion_maps[map_key]

        kp = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        xy = kp[:, :2]
        
        # Only rectify visible keypoints
        visible_mask = kp[:, 2] > 0
        if visible_mask.any():
            visible_xy = xy[visible_mask]
            rectified_xy = rectify_points_with_map(visible_xy, map_x, map_y)
            xy[visible_mask] = rectified_xy
        
        kp[:, :2] = xy

        # Save back – keep original visibility (0/1/2) untouched
        ann["keypoints"] = kp.flatten().round(3).tolist()

        # ─────────────────────────── bbox + area ────────────────────────────
        vis_mask = kp[:, 2] > 0  # labelled (1 or 2)
        if vis_mask.any():
            vis_xy = kp[vis_mask, :2]
            x_min, y_min = vis_xy.min(axis=0)
            x_max, y_max = vis_xy.max(axis=0)
            w, h = x_max - x_min, y_max - y_min
            ann["bbox"] = [float(x_min), float(y_min), float(w), float(h)]
            ann["area"] = float(w * h)
        # else: keep original bbox/area (completely unlabelled person)

    return coco


# -----------------------------------------------------------------------------
# CLI entry‑point
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rectify all key‑points in a COCO file using per‑camera calibrations (directories, not ZIPs).")
    parser.add_argument("--coco", required=True, help="Input COCO annotations (JSON)")
    parser.add_argument("--calib_dir", required=True, help="Root directory containing cam_*/calib/camera_calib.json")
    parser.add_argument("--output", default="_annotations_rectified_v2.coco.json",
                        help="Output COCO JSON (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading COCO…", flush=True)
    with open(args.coco, "r") as f:
        coco = json.load(f)

    print("Scanning calibrations…", flush=True)
    calibs = load_calibrations(args.calib_dir)
    print(f"Found {len(calibs)} calibrated cameras: {sorted(calibs.keys())}")

    coco = rectify_coco(coco, calibs)

    with open(args.output, "w") as f:
        json.dump(coco, f)
    print(f"Saved rectified annotations → {args.output}")


if __name__ == "__main__":
    main()