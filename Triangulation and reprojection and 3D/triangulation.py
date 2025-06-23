import os, re, json
import numpy as np
import cv2

# ——— Configuration: Update these paths if needed ———
CALIB_FOLDER = '../3D Pose Estimation Material/camera_data_with_Rvecs_2ndversion/Camera_config2'
ANN_PATH     = '../Annotations/_annotations_rectified_v2.coco.json'
OUTPUT_3D    = 'triangulated_positions_v2.json'
TARGET_CAMS  = [2,5,8,13]  # Camera IDs to use for triangulation

def load_calibrations(folder_path, target_cams):
    """
    Load camera calibration data from JSON files.
    
    Searches recursively for 'camera_calib.json' files within cam_<id> folders.
    Each calibration file should contain:
    - 'mtx': 3x3 camera intrinsic matrix (K)
    - 'dist': distortion coefficients
    - 'rvecs'/'rvec': rotation vector (camera pose)
    - 'tvecs'/'tvec': translation vector (camera position)
    
    Args:
        folder_path (str): Path to the folder containing cam_<id> subdirectories
        target_cams (list): List of camera IDs to load calibrations for
        
    Returns:
        dict: Dictionary mapping camera_id -> {'K', 'dist', 'rvec', 'tvec'}
    """
    calibrations = {}

    # Iterate through all items in the calibration folder
    for item in os.listdir(folder_path):
        if not item.startswith('cam_'):
            continue

        # Extract camera ID from folder name (e.g., 'cam_2' -> 2)
        cam_id = int(item.split('_')[1])
        if cam_id not in target_cams:
            continue

        # Search recursively for camera_calib.json file
        item_path = os.path.join(folder_path, item)
        calib_file = None
        for root, _, files in os.walk(item_path):
            if 'camera_calib.json' in files:
                calib_file = os.path.join(root, 'camera_calib.json')
                break

        if calib_file is None:
            print(f"⚠️ Calibration not found for cam_{cam_id}")
            continue

        # Load and parse calibration data
        with open(calib_file, 'r') as f:
            data = json.load(f)

        # Extract calibration parameters
        K    = np.array(data['mtx'])  # Intrinsic camera matrix
        dist = np.array(data['dist']).flatten()  # Distortion coefficients
        
        # Handle different naming conventions for rotation/translation vectors
        rvec = np.array(data.get('rvecs', data.get('rvec'))).reshape(3)
        tvec = np.array(data.get('tvecs', data.get('tvec'))).reshape(3)

        calibrations[cam_id] = {'K': K, 'dist': dist, 'rvec': rvec, 'tvec': tvec}

    return calibrations

def build_projection_matrices(calibrations, camera_ids):
    """
    Build 3x4 projection matrices for each camera.
    
    The projection matrix P = K * [R|t] maps 3D world points to 2D image coordinates.
    Where:
    - K is the 3x3 intrinsic matrix
    - R is the 3x3 rotation matrix (from rotation vector using Rodrigues formula)
    - t is the 3x1 translation vector
    
    Args:
        calibrations (dict): Camera calibration data
        camera_ids (list): List of camera IDs to build matrices for
        
    Returns:
        dict: Dictionary mapping camera_id -> 3x4 projection matrix
    """
    projection_matrices = {}
    
    for cam_id in camera_ids:
        calib = calibrations[cam_id]
        
        # Convert rotation vector to rotation matrix using Rodrigues formula
        R, _ = cv2.Rodrigues(calib['rvec'])
        
        # Reshape translation vector to column vector
        t = calib['tvec'].reshape(3, 1)
        
        # Build projection matrix: P = K * [R|t]
        Rt = np.hstack((R, t))  # Concatenate R and t horizontally
        P = calib['K'].dot(Rt)  # Matrix multiplication K * [R|t]
        
        projection_matrices[cam_id] = P

    return projection_matrices

def triangulate_point(projection_matrices_list, points_2d_list):
    """
    Triangulate a 3D point from multiple 2D observations using the DLT method.
    
    Uses the Direct Linear Transformation (DLT) algorithm to solve for the 3D point
    that best satisfies the reprojection constraints from multiple cameras.
    
    For each camera i with projection matrix P_i and observed point (u_i, v_i):
    u_i * P_i[2,:] - P_i[0,:] = 0  (constraint from x-coordinate)
    v_i * P_i[2,:] - P_i[1,:] = 0  (constraint from y-coordinate)
    
    These linear equations are stacked into matrix A and solved using SVD.
    
    Args:
        projection_matrices_list (list): List of 3x4 projection matrices
        points_2d_list (list): List of (u, v) 2D point coordinates
        
    Returns:
        np.array: 3D point coordinates [X, Y, Z]
    """
    A = []
    
    # Build the system of linear equations A * X = 0
    for P, (u, v) in zip(projection_matrices_list, points_2d_list):
        # Add constraints from x and y coordinates
        A.append(u * P[2] - P[0])  # u*P[2,:] - P[0,:] = 0
        A.append(v * P[2] - P[1])  # v*P[2,:] - P[1,:] = 0
    
    A = np.vstack(A)
    
    # Solve using SVD: the solution is the right singular vector corresponding
    # to the smallest singular value (last column of V^T)
    _, _, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[-1]  # Last row of V^T
    
    # Convert from homogeneous to 3D coordinates by dividing by w
    return X_homogeneous / X_homogeneous[3]

def analyze_annotations(coco_data, images_dict):
    """
    Debug function to analyze and display annotation structure.
    
    Prints the first few annotations to understand the keypoint format
    and verify that data is being parsed correctly.
    
    Args:
        coco_data (dict): COCO format annotation data
        images_dict (dict): Dictionary mapping image_id -> image info
    """
    print("\n=== ANNOTATION ANALYSIS ===")
    
    # Analyze first 5 annotations for debugging
    for ann in coco_data['annotations'][:5]:
        img = images_dict[ann['image_id']]
        image_name = img['extra']['name']
        
        # Reshape keypoints from flat array to (n_joints, 3) format
        # Each keypoint has [x, y, visibility] format
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)
        
        print(f"\nImage: {image_name}")
        # Show first 5 joints for brevity
        for joint_idx, (x, y, visibility) in enumerate(keypoints[:5]):
            print(f"  Joint {joint_idx}: x={x:.1f}, y={y:.1f}, visibility={visibility}")

def main():
    """
    Main triangulation pipeline:
    1. Load camera calibrations
    2. Build projection matrices
    3. Load and parse COCO annotations
    4. Group keypoints by frame
    5. Triangulate 3D positions for each joint
    6. Save results to JSON file
    """
    
    # ——— Step 1: Load Camera Calibrations ———
    print("Loading camera calibrations...")
    calibrations = load_calibrations(CALIB_FOLDER, TARGET_CAMS)
    print(f"Available camera IDs: {sorted(calibrations.keys())}")
    print(f"Target cameras: {TARGET_CAMS}")

    # Check if all required cameras are available
    missing_cameras = [cam for cam in TARGET_CAMS if cam not in calibrations]
    if missing_cameras:
        print(f"❌ Missing cameras: {missing_cameras}")
        return

    # Build projection matrices for triangulation
    projection_matrices = build_projection_matrices(calibrations, TARGET_CAMS)
    print("✅ Projection matrices built successfully")

    # ——— Step 2: Load COCO Annotations ———
    print("Loading COCO annotations...")
    with open(ANN_PATH) as f:
        coco_data = json.load(f)
    
    # Create lookup dictionary for image information
    images_dict = {img['id']: img for img in coco_data['images']}

    # Determine number of joints from categories or first annotation
    if 'categories' in coco_data and coco_data['categories'] and 'keypoints' in coco_data['categories'][0]:
        num_joints = len(coco_data['categories'][0]['keypoints'])
    else:
        # Fallback: infer from first annotation (keypoints array length / 3)
        num_joints = len(coco_data['annotations'][0]['keypoints']) // 3

    print(f"Number of joints detected: {num_joints}")

    # Debug: analyze annotation structure
    analyze_annotations(coco_data, images_dict)

    # ——— Step 3: Parse and Group Annotations by Frame ———
    print("Parsing annotations and grouping by frame...")
    frame_data = {}
    
    for annotation in coco_data['annotations']:
        img = images_dict[annotation['image_id']]
        image_name = img['extra']['name']
        
        # Parse image name to extract camera ID and frame number
        # Expected format: out<cam_id>_frame_<frame_num>.png
        match = re.match(r'out(\d+)_frame_(\d+)\.png', image_name)
        if not match:
            continue
            
        camera_id = int(match.group(1))
        frame_number = int(match.group(2))
        
        # Reshape keypoints from flat array to (n_joints, 3)
        keypoints = np.array(annotation['keypoints']).reshape(-1, 3)
        
        # Extract 2D coordinates for each joint (ignore visibility for now)
        # Since these are manually annotated, we assume all keypoints are valid
        points_2d = {joint_idx: (float(x), float(y)) 
                    for joint_idx, (x, y, visibility) in enumerate(keypoints)}
        
        # Group by frame, then by camera
        frame_data.setdefault(frame_number, {})[camera_id] = points_2d

    print(f"Parsed {len(frame_data)} frames")

    # ——— Step 4: Triangulate 3D Points ———
    print("Starting triangulation...")
    triangulated_3d = {}
    total_points = 0
    
    for frame_num in frame_data.keys():
        triangulated_3d[frame_num] = {}
        frame_cameras = frame_data[frame_num]
        
        # Triangulate each joint independently
        for joint_idx in range(num_joints):
            # Find cameras that have this joint annotated
            available_cameras = [cam for cam in TARGET_CAMS 
                               if cam in frame_cameras and joint_idx in frame_cameras[cam]]
            
            # Triangulate if we have at least 2 camera views
            # (minimum requirement for triangulation)
            if len(available_cameras) >= 2:
                # Get 2D points from all available cameras for this joint
                points_2d_list = [frame_cameras[cam][joint_idx] for cam in available_cameras]
                
                # Get corresponding projection matrices
                projection_matrices_list = [projection_matrices[cam] for cam in available_cameras]
                
                # Perform triangulation
                point_3d = triangulate_point(projection_matrices_list, points_2d_list)
                
                # Store only the 3D coordinates (first 3 elements, excluding homogeneous coordinate)
                triangulated_3d[frame_num][joint_idx] = point_3d[:3].tolist()
                total_points += 1

    # ——— Step 5: Save Results and Statistics ———
    print("Saving triangulated results...")
    with open(OUTPUT_3D, 'w') as f:
        json.dump(triangulated_3d, f, indent=2)
    
    # Print comprehensive statistics
    print(f"\n=== TRIANGULATION COMPLETE ===")
    print(f"✅ Results saved to: {OUTPUT_3D}")
    print(f"📊 Frames processed: {len(triangulated_3d)}")
    print(f"📊 Total 3D points triangulated: {total_points}")
    if len(triangulated_3d) > 0:
        print(f"📊 Average points per frame: {total_points/len(triangulated_3d):.1f}")
        print(f"📊 Expected points per frame: {num_joints}")
        coverage = (total_points/len(triangulated_3d)) / num_joints * 100
        print(f"📊 Joint coverage: {coverage:.1f}%")

if __name__ == '__main__':
    main()