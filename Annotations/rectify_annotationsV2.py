"""
This script corrects lens distortion in 2D keypoint coordinates by applying camera calibration parameters, 
making the keypoints match what they would look like in rectified (undistorted) images.

The process involves:
1. Loading camera calibration parameters (intrinsic matrix K and distortion coefficients)
2. Creating undistortion maps for each camera/image size combination
3. Applying lens distortion correction to keypoint coordinates using bilinear interpolation
4. Updating bounding boxes and areas based on rectified keypoints

To run the code:
python rectify_annotationsV2.py \
    --coco <path_to_coco_annotations.json> \
    --calib_dir <path_to_calibrations_folder> \
    --output <output_annotations_name.json>
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
    """
    Extract camera ID from image filename using regex patterns.
    
    Supports common naming patterns:
    - cam_8/image.jpg → camera ID 8
    - cam-5_frame.png → camera ID 5  
    - out3_sequence.jpg → camera ID 3
    
    Args:
        filename: Image filename or path
        
    Returns:
        Camera ID as integer
        
    Raises:
        ValueError: If no camera ID pattern is found in filename
    """
    patterns = [r"cam[_-]?(\d+)", r"out(\d+)_"]
    for pattern in patterns:
        match = re.search(pattern, filename, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise ValueError(f"Cannot infer camera id from filename: {filename}")


def load_calibrations(root: str) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Recursively search for camera_calib.json files and load calibration data.
    
    Expected directory structure:
    root/
    ├── cam_1/calib/camera_calib.json
    ├── cam_2/calib/camera_calib.json
    └── ...
    
    Args:
        root: Root directory to search for calibration files
        
    Returns:
        Dictionary mapping camera_id to calibration parameters:
        {camera_id: {"K": intrinsic_matrix_3x3, "dist": distortion_coefficients_1xN}}
        
    Raises:
        RuntimeError: If no calibration files are found
    """
    calibs: Dict[int, Dict[str, np.ndarray]] = {}
    
    for dirpath, _dirnames, filenames in os.walk(root):
        if "camera_calib.json" not in filenames:
            continue
            
        # Extract camera ID from directory path
        try:
            cam_id = int(re.search(r"cam[_-]?(\d+)", dirpath, re.IGNORECASE).group(1))
        except AttributeError:
            warnings.warn(f"Cannot parse camera id from path: {dirpath}. Skipped.")
            continue
            
        # Load calibration data
        calib_path = os.path.join(dirpath, "camera_calib.json")
        with open(calib_path, "r") as f:
            data = json.load(f)
            
        # Convert to numpy arrays with consistent data types
        K = np.asarray(data["mtx"], dtype=np.float32)  # 3x3 intrinsic matrix
        dist = np.asarray(data["dist"], dtype=np.float32).reshape(-1)  # distortion coefficients
        
        calibs[cam_id] = {"K": K, "dist": dist}
        
    if not calibs:
        raise RuntimeError(f"No camera_calib.json found under {root}.")
        
    return calibs


def create_undistortion_map(width: int, height: int, K: np.ndarray, dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create pixel-wise mapping from distorted to undistorted coordinates.
    
    This function generates lookup tables that map each pixel in the distorted image
    to its corresponding location in the undistorted image space.
    
    Process:
    1. Create a grid of all pixel coordinates (x,y) in the image
    2. Apply cv2.undistortPoints to get corrected coordinates for each pixel
    3. Return separate maps for x and y coordinates
    
    Args:
        width: Image width in pixels
        height: Image height in pixels  
        K: 3x3 camera intrinsic matrix
        dist: Distortion coefficients array
        
    Returns:
        Tuple of (map_x, map_y) where:
        - map_x: Array of shape (height, width) containing corrected x-coordinates
        - map_y: Array of shape (height, width) containing corrected y-coordinates
    """
    # Create coordinate grid for all pixels
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    pts = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)  # Reshape to format required by cv2.undistortPoints
    
    # Apply undistortion to get corrected coordinates
    undistorted_pts = cv2.undistortPoints(pts, K, dist, P=K)
    undistorted_map = undistorted_pts.reshape(height, width, 2)
    
    # Separate x and y coordinate maps
    map_x = undistorted_map[:, :, 0]  # Corrected x-coordinates
    map_y = undistorted_map[:, :, 1]  # Corrected y-coordinates
    
    return map_x, map_y


def rectify_points_with_map(points: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """
    Apply lens distortion correction to keypoints using bilinear interpolation.
    
    For each keypoint coordinate, this function:
    1. Finds the four surrounding pixels in the undistortion maps
    2. Uses bilinear interpolation to get sub-pixel accurate corrected coordinates
    3. Handles out-of-bounds points by keeping original coordinates
    
    Args:
        points: Array of shape (N, 2) containing [x, y] coordinates
        map_x: Undistortion map for x-coordinates, shape (height, width)
        map_y: Undistortion map for y-coordinates, shape (height, width)
        
    Returns:
        Array of shape (N, 2) containing rectified [x, y] coordinates
    """
    rectified_points = []
    
    for x, y in points:
        # Convert to integer pixel coordinates for map lookup
        x_int, y_int = int(x), int(y)
        
        # Check if point is within image bounds
        if 0 <= x_int < map_x.shape[1] and 0 <= y_int < map_x.shape[0]:
            # Calculate fractional parts for bilinear interpolation
            x_frac = x - x_int
            y_frac = y - y_int
            
            # Define the four surrounding pixel coordinates
            x0, y0 = x_int, y_int
            x1, y1 = min(x_int + 1, map_x.shape[1] - 1), min(y_int + 1, map_x.shape[0] - 1)
            
            # Bilinear interpolation for x-coordinate
            new_x = ((1 - x_frac) * (1 - y_frac) * map_x[y0, x0] +
                     x_frac * (1 - y_frac) * map_x[y0, x1] +
                     (1 - x_frac) * y_frac * map_x[y1, x0] +
                     x_frac * y_frac * map_x[y1, x1])
            
            # Bilinear interpolation for y-coordinate
            new_y = ((1 - x_frac) * (1 - y_frac) * map_y[y0, x0] +
                     x_frac * (1 - y_frac) * map_y[y0, x1] +
                     (1 - x_frac) * y_frac * map_y[y1, x0] +
                     x_frac * y_frac * map_y[y1, x1])
            
            rectified_points.append([new_x, new_y])
        else:
            # Point is outside image bounds, keep original coordinates
            rectified_points.append([x, y])
    
    return np.array(rectified_points)


# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------

def rectify_coco(coco: dict, calibs: Dict[int, Dict[str, np.ndarray]]) -> dict:
    """
    Process all annotations in COCO dataset to apply lens distortion correction.
    
    This function assumes all keypoints are visible (visibility = 2) and processes them
    without filtering. For each annotation:
    1. Identifies camera from image filename
    2. Loads or retrieves cached undistortion maps for that camera/image size
    3. Applies rectification to all keypoints 
    4. Updates bounding box and area based on rectified keypoints
    5. Preserves original visibility values
    
    Args:
        coco: COCO dataset dictionary with 'images' and 'annotations'
        calibs: Camera calibration parameters per camera ID
        
    Returns:
        Modified COCO dataset with rectified keypoint coordinates
    """
    # Create lookup table for fast image metadata access
    img_lookup = {img["id"]: img for img in coco["images"]}
    
    # Cache undistortion maps to avoid recomputation for same camera/image size
    undistortion_maps = {}

    for ann in tqdm(coco["annotations"], desc="Rectifying keypoints", unit="ann"):
        img = img_lookup[ann["image_id"]]
        
        # Extract camera ID from image filename
        try:
            cam_id = infer_cam_id(img["file_name"])
        except ValueError as e:
            warnings.warn(str(e) + f"  (annotation id {ann['id']})")
            continue

        # Check if calibration data exists for this camera
        if cam_id not in calibs:
            warnings.warn(f"Calibration for camera {cam_id} not found → ann id {ann['id']} skipped")
            continue

        # Get calibration parameters
        K = calibs[cam_id]["K"]
        dist = calibs[cam_id]["dist"]
        
        # Get image dimensions
        width = img["width"]
        height = img["height"]
        
        # Create or retrieve cached undistortion maps
        map_key = (cam_id, width, height)
        if map_key not in undistortion_maps:
            map_x, map_y = create_undistortion_map(width, height, K, dist)
            undistortion_maps[map_key] = (map_x, map_y)
        else:
            map_x, map_y = undistortion_maps[map_key]

        # Parse keypoints from COCO format [x1,y1,v1, x2,y2,v2, ...]
        kp = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        xy = kp[:, :2]  # Extract x,y coordinates only
        
        # Apply lens distortion correction to all keypoints
        rectified_xy = rectify_points_with_map(xy, map_x, map_y)
        kp[:, :2] = rectified_xy  # Update coordinates, preserve visibility

        # Save back to annotation (round to 3 decimal places for file size)
        ann["keypoints"] = kp.flatten().round(3).tolist()

        # Update bounding box based on rectified keypoint positions
        x_min, y_min = rectified_xy.min(axis=0)
        x_max, y_max = rectified_xy.max(axis=0)
        w, h = x_max - x_min, y_max - y_min
        ann["bbox"] = [float(x_min), float(y_min), float(w), float(h)]
        ann["area"] = float(w * h)

    return coco


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments for the rectification script."""
    parser = argparse.ArgumentParser(
        description="Rectify all keypoints in a COCO file using per-camera calibrations.")
    parser.add_argument("--coco", required=True, 
                       help="Input COCO annotations (JSON)")
    parser.add_argument("--calib_dir", required=True, 
                       help="Root directory containing cam_*/calib/camera_calib.json")
    parser.add_argument("--output", default="_annotations_rectified_v2.coco.json",
                       help="Output COCO JSON (default: %(default)s)")
    return parser.parse_args()


def main():
    """Main function to execute the rectification process."""
    args = parse_args()

    print("Loading COCO annotations...", flush=True)
    with open(args.coco, "r") as f:
        coco = json.load(f)

    print("Scanning for calibration files...", flush=True)
    calibs = load_calibrations(args.calib_dir)
    print(f"Found {len(calibs)} calibrated cameras: {sorted(calibs.keys())}")

    print("Starting rectification process...", flush=True)
    coco = rectify_coco(coco, calibs)

    print("Saving rectified annotations...", flush=True)
    with open(args.output, "w") as f:
        json.dump(coco, f)
    print(f"Saved rectified annotations → {args.output}")


if __name__ == "__main__":
    main()