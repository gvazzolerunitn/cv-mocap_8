"""
3D Pose Reprojection Error Analysis Tool

This script evaluates the quality of 3D triangulated poses by computing reprojection errors.
It takes triangulated 3D joint positions and projects them back onto the original camera images,
then compares these projections with the original 2D annotations to calculate error metrics.
"""

import json
import re
import numpy as np
import cv2
import csv
from datetime import datetime
import os

# ——— Configuration: File paths and parameters ———
CALIB_FOLDER = '../3D Pose Estimation Material/camera_data_with_Rvecs_2ndversion/Camera_config2'
ANN_PATH     = '../Annotations/_annotations_rectified_v2.coco.json'  # Use rectified annotations
INPUT_3D     = 'triangulated_positions_v2.json'
TARGET_CAMS  = [2, 5, 8, 13]  # Camera IDs to use for reprojection analysis

# Import the calibration loading function from triangulation module
from triangulation import load_calibrations

def save_results_to_csv(errors, target_cams, calib_folder, ann_path, input_3d, output_file='reprojection_results.csv'):
    """
    Save comprehensive reprojection error analysis results to a CSV file.
    
    Creates a detailed report including:
    - Experimental parameters and file paths
    - Dataset statistics (number of points, cameras, etc.)
    - Error metrics (MPJPE, RMSE, MSE, percentiles)
    - Error distribution analysis across different ranges
    
    Args:
        errors (np.ndarray): Array of reprojection errors in pixels
        target_cams (list): List of camera IDs used in analysis
        calib_folder (str): Path to camera calibration folder
        ann_path (str): Path to annotation file
        input_3d (str): Path to 3D triangulated points file
        output_file (str): Output CSV filename
    """
    # Generate timestamp for report header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # ——— Header Section ———
        writer.writerow(['REPROJECTION ERROR ANALYSIS RESULTS'])
        writer.writerow(['Generated on:', timestamp])
        writer.writerow([])
        
        # ——— Parameters Section ———
        writer.writerow(['PARAMETERS'])
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['Calibration Folder', calib_folder])
        writer.writerow(['Annotations File', ann_path])
        writer.writerow(['3D Points File', input_3d])
        writer.writerow(['Target Cameras', ', '.join(map(str, target_cams))])
        writer.writerow(['Number of Cameras', len(target_cams)])
        writer.writerow([])
        
        # ——— Data Statistics Section ———
        writer.writerow(['DATA STATISTICS'])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Points Analyzed', len(errors)])
        if len(target_cams) > 0:
            writer.writerow(['Points per Camera (avg)', f"{len(errors) / len(target_cams):.1f}"])
        writer.writerow([])
        
        # ——— Results Section ———
        if len(errors) > 0:
            # Calculate comprehensive error statistics
            mpjpe = np.mean(errors)                    # Mean Per Joint Position Error
            mse = np.mean(errors**2)                   # Mean Square Error
            rmse = np.sqrt(mse)                        # Root Mean Square Error
            std_err = np.std(errors)                   # Standard deviation
            median_err = np.median(errors)             # Median error
            q25 = np.percentile(errors, 25)            # 25th percentile
            q75 = np.percentile(errors, 75)            # 75th percentile
            
            writer.writerow(['REPROJECTION ERROR RESULTS'])
            writer.writerow(['Metric', 'Value (pixels)', 'Description'])
            writer.writerow(['MPJPE', f"{mpjpe:.4f}", 'Mean Per Joint Position Error'])
            writer.writerow(['RMSE', f"{rmse:.4f}", 'Root Mean Square Error'])
            writer.writerow(['MSE', f"{mse:.4f}", 'Mean Square Error'])
            writer.writerow(['Standard Deviation', f"{std_err:.4f}", 'Standard deviation of errors'])
            writer.writerow(['Median Error', f"{median_err:.4f}", 'Median reprojection error'])
            writer.writerow(['25th Percentile', f"{q25:.4f}", '25% of errors below this value'])
            writer.writerow(['75th Percentile', f"{q75:.4f}", '75% of errors below this value'])
            writer.writerow(['Maximum Error', f"{np.max(errors):.4f}", 'Largest reprojection error'])
            writer.writerow(['Minimum Error', f"{np.min(errors):.4f}", 'Smallest reprojection error'])
            writer.writerow([])
            
            # ——— Error Distribution Analysis ———
            writer.writerow(['ERROR DISTRIBUTION'])
            writer.writerow(['Error Range (px)', 'Count', 'Percentage'])
            
            # Define error ranges for distribution analysis
            # These ranges help understand the quality distribution of triangulation
            ranges = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, float('inf'))]
            for min_err, max_err in ranges:
                if max_err == float('inf'):
                    count = np.sum(errors >= min_err)
                    range_str = f"≥ {min_err}"
                else:
                    count = np.sum((errors >= min_err) & (errors < max_err))
                    range_str = f"{min_err} - {max_err}"
                
                percentage = (count / len(errors)) * 100
                writer.writerow([range_str, count, f"{percentage:.1f}%"])
            
        else:
            # Handle case with no valid data
            writer.writerow(['RESULTS'])
            writer.writerow(['Status', 'No valid reprojection points found'])
            
    print(f"✅ Results saved to: {output_file}")


def load_and_parse_annotations(ann_path):
    """
    Load and parse COCO format annotations to extract 2D keypoint data.
    
    Reconstructs the frame_data structure used in triangulation to match
    3D points with their corresponding 2D observations.
    
    Args:
        ann_path (str): Path to COCO format annotation file
        
    Returns:
        dict: Nested dictionary structure {frame_num: {camera_id: {joint_id: (x, y)}}}
    """
    # Load COCO annotations
    with open(ann_path) as f:
        coco_data = json.load(f)
    
    # Create lookup dictionary for image information
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Parse annotations and group by frame and camera
    frame_data = {}
    for annotation in coco_data['annotations']:
        # Get image information
        img = images_dict[annotation['image_id']]
        image_name = img['extra']['name']
        
        # Parse image name to extract camera ID and frame number
        # Expected format: out<cam_id>_frame_<frame_num>.png
        match = re.match(r'out(\d+)_frame_(\d+)\.png', image_name)
        if not match:
            continue
            
        camera_id = int(match.group(1))
        frame_number = int(match.group(2))
        
        # Reshape keypoints from flat array to (n_joints, 3) format
        keypoints = np.array(annotation['keypoints']).reshape(-1, 3)
        
        # Extract 2D coordinates for visible joints only
        # Only include keypoints with visibility > 0 (annotated and visible)
        points_2d = {joint_idx: (float(x), float(y)) 
                    for joint_idx, (x, y, visibility) in enumerate(keypoints) 
                    if visibility > 0}
        
        # Group by frame, then by camera
        frame_data.setdefault(frame_number, {})[camera_id] = points_2d
    
    return frame_data


def compute_reprojection_errors(points_3d, frame_data, calibrations, target_cams):
    """
    Compute reprojection errors for all valid 3D-2D point correspondences.
    
    For each triangulated 3D point, projects it back onto each camera image
    using the camera's calibration parameters and compares with the original
    2D annotation to compute the Euclidean distance error.
    
    Args:
        points_3d (dict): Dictionary of triangulated 3D points {frame: {joint: [x,y,z]}}
        frame_data (dict): Dictionary of 2D annotations {frame: {camera: {joint: (x,y)}}}
        calibrations (dict): Camera calibration parameters
        target_cams (list): List of camera IDs to analyze
        
    Returns:
        np.ndarray: Array of reprojection errors in pixels
    """
    # Extract calibration parameters for each camera
    camera_matrices = {cam: calibrations[cam]['K'] for cam in target_cams}
    rotation_vectors = {cam: calibrations[cam]['rvec'] for cam in target_cams}
    translation_vectors = {cam: calibrations[cam]['tvec'] for cam in target_cams}
    
    errors = []
    
    # Iterate through all triangulated 3D points
    for frame_str, joints_3d in points_3d.items():
        frame_number = int(frame_str)  # Convert string key to integer
        
        # Skip frames without corresponding 2D annotations
        if frame_number not in frame_data:
            continue
            
        # Process each joint in the current frame
        for joint_str, point_3d in joints_3d.items():
            joint_id = int(joint_str)  # Convert string key to integer
            
            # Convert 3D point to proper format for OpenCV
            point_3d_array = np.array(point_3d, dtype=float).reshape(1, 3)
            
            # Check reprojection for each target camera
            for camera_id in target_cams:
                # Verify that this camera has annotation for this joint in this frame
                if (camera_id in frame_data[frame_number] and 
                    joint_id in frame_data[frame_number][camera_id]):
                    
                    # Get the original 2D annotation (ground truth)
                    observed_2d = np.array(frame_data[frame_number][camera_id][joint_id])
                    
                    # Project 3D point back to 2D using camera parameters
                    # Note: Using None for distortion coefficients since annotations are already rectified
                    projected_2d, _ = cv2.projectPoints(
                        point_3d_array, 
                        rotation_vectors[camera_id], 
                        translation_vectors[camera_id], 
                        camera_matrices[camera_id], 
                        None  # No distortion correction (rectified images)
                    )
                    
                    # Reshape projected point to 2D vector
                    projected_2d_flat = projected_2d.reshape(2)
                    
                    # Calculate Euclidean distance between projected and observed points
                    reprojection_error = np.linalg.norm(projected_2d_flat - observed_2d)
                    errors.append(reprojection_error)
    
    return np.array(errors)


def main():
    """
    Main function that orchestrates the reprojection error analysis pipeline.
    
    Pipeline steps:
    1. Load camera calibrations
    2. Load triangulated 3D points
    3. Parse 2D annotations
    4. Compute reprojection errors
    5. Display and save results
    """
    
    print("="*60)
    print("3D POSE REPROJECTION ERROR ANALYSIS")
    print("="*60)
    
    # ——— Step 1: Load Camera Calibrations ———
    print("📋 Loading camera calibrations...")
    try:
        calibrations = load_calibrations(CALIB_FOLDER, TARGET_CAMS)
        print(f"✅ Loaded calibrations for cameras: {sorted(calibrations.keys())}")
    except Exception as e:
        print(f"❌ Error loading calibrations: {e}")
        return
    
    # Verify all target cameras are available
    missing_cameras = [cam for cam in TARGET_CAMS if cam not in calibrations]
    if missing_cameras:
        print(f"❌ Missing calibrations for cameras: {missing_cameras}")
        return
    
    # ——— Step 2: Load Triangulated 3D Points ———
    print("📊 Loading triangulated 3D points...")
    try:
        with open(INPUT_3D) as f:
            points_3d = json.load(f)
        print(f"✅ Loaded 3D points for {len(points_3d)} frames")
    except FileNotFoundError:
        print(f"❌ 3D points file not found: {INPUT_3D}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON format in: {INPUT_3D}")
        return
    
    # ——— Step 3: Parse 2D Annotations ———
    print("🎯 Loading and parsing 2D annotations...")
    try:
        frame_data = load_and_parse_annotations(ANN_PATH)
        total_annotations = sum(len(cameras.keys()) for cameras in frame_data.values())
        print(f"✅ Parsed annotations for {len(frame_data)} frames ({total_annotations} camera views)")
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return
    
    # ——— Step 4: Compute Reprojection Errors ———
    print("🔄 Computing reprojection errors...")
    errors = compute_reprojection_errors(points_3d, frame_data, calibrations, TARGET_CAMS)
    
    # ——— Step 5: Display and Save Results ———
    if len(errors) > 0:
        # Calculate key metrics
        mpjpe = np.mean(errors)           # Mean Per Joint Position Error
        mse = np.mean(errors**2)          # Mean Square Error
        rmse = np.sqrt(mse)               # Root Mean Square Error
        
        print(f"\n📈 REPROJECTION ERROR ANALYSIS RESULTS")
        print(f"{'='*50}")
        print(f"Total points analyzed: {len(errors)}")
        print(f"Target cameras: {TARGET_CAMS}")
        print(f"\n📊 Error Metrics:")
        print(f"  • MPJPE (Mean Per Joint Position Error): {mpjpe:.3f} px")
        print(f"  • RMSE  (Root Mean Square Error):       {rmse:.3f} px")
        print(f"  • MSE   (Mean Square Error):            {mse:.3f} px²")
        print(f"  • Maximum error:                        {np.max(errors):.3f} px")
        print(f"  • Minimum error:                        {np.min(errors):.3f} px")
        print(f"  • Standard deviation:                   {np.std(errors):.3f} px")
        print(f"  • Median error:                         {np.median(errors):.3f} px")
        
        # Save comprehensive results to CSV
        save_results_to_csv(errors, TARGET_CAMS, CALIB_FOLDER, ANN_PATH, INPUT_3D)
        
    else:
        print("❌ No valid reprojection points found!")
        print("   This could indicate:")
        print("   - Mismatch between 3D points and 2D annotations")
        print("   - Issues with frame/joint indexing")
        print("   - Problems with input data format")
        
        # Save empty results for documentation
        save_results_to_csv(np.array([]), TARGET_CAMS, CALIB_FOLDER, ANN_PATH, INPUT_3D)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")


if __name__ == '__main__':
    main()