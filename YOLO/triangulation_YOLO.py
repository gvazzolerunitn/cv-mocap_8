import os
import json
import numpy as np
import cv2

def triangulate_with_detailed_debug():
    """
    Performs 3D triangulation of 2D keypoints from multiple camera views.
    This version includes detailed debugging output and uses more permissive thresholds.
    """
    
    # Load the rescaled 2D keypoint data. This file contains coordinates
    # that have been adjusted to match the resolution of the calibration images.
    with open('yolo2d_for_triangulation_rescaled.json') as f:
        frame_data = json.load(f)
    
    print(f"✅ Loaded data for {len(frame_data)} frames")
    
    # --- Load Camera Calibrations ---
    calib_folder = '../3D Pose Estimation Material/camera_data_with_Rvecs_2ndversion/Camera_config2'
    # Define the specific cameras that will be used for triangulation.
    TARGET_CAMS = [2, 5, 8, 13]
    
    # Dictionary to store the projection matrix (P) for each camera.
    Ps = {}
    for cam_id in TARGET_CAMS:
        cam_path = os.path.join(calib_folder, f'cam_{cam_id}')
        calib_file = None
        
        # Walk through the directory to find the calibration file.
        for root, dirs, files in os.walk(cam_path):
            if 'camera_calib.json' in files:
                calib_file = os.path.join(root, 'camera_calib.json')
                break
        
        if calib_file and os.path.exists(calib_file):
            with open(calib_file) as f:
                calib = json.load(f)
            
            # Extract intrinsic (K), rotation (rvec), and translation (tvec) parameters.
            K = np.array(calib['mtx'])
            rvec = np.array(calib.get('rvecs', calib.get('rvec', []))).flatten()
            tvec = np.array(calib.get('tvecs', calib.get('tvec', []))).flatten()
            
            # Convert rotation vector to a rotation matrix.
            R, _ = cv2.Rodrigues(rvec)
            # Compute the projection matrix P = K * [R|t].
            P = K @ np.hstack([R, tvec.reshape(-1, 1)])
            Ps[cam_id] = P
            print(f"✅ Calibration loaded for cam {cam_id}")
        else:
            print(f"❌ Calibration not found for cam {cam_id}")
    
    print(f"📊 Calibrations available for cameras: {list(Ps.keys())}")
    
    # --- Robust Triangulation Function ---
    def triangulate_point_permissive(Ps_list, pts2d_list):
        """
        Triangulates a 3D point from a list of 2D points and their corresponding
        projection matrices using the Direct Linear Transformation (DLT) algorithm.
        """
        # Triangulation requires at least two different camera views.
        if len(pts2d_list) < 2:
            return None
        
        # Build the A matrix for the linear system Ax = 0.
        A = []
        for pt2d, P in zip(pts2d_list, Ps_list):
            x, y = pt2d
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        
        A = np.array(A)
        
        try:
            # Solve the system using Singular Value Decomposition (SVD).
            # The solution is the last row of the V^T matrix.
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1, :]
            
            if len(X) < 4:
                return None
            
            # Homogenize the 3D point by dividing by the 4th component.
            w = X[3]
            if isinstance(w, np.ndarray):
                w = w.item()
            
            if abs(w) < 1e-10:  # Permissive threshold for numerical stability.
                return None
            
            point_3d = X[:3] / w
            
            # Perform a sanity check on the reconstructed point's distance.
            # This helps filter out outliers far from the origin.
            norm = np.linalg.norm(point_3d)
            if norm > 20000 or norm < 0.1:  # Max 20 meters, min 0.1 meters (10cm)
                return None
            
            return point_3d.tolist()
            
        except Exception as e:
            # Catch any other potential errors during SVD or calculation.
            return None
    
    # --- Detailed Analysis of a Sample Frame for Debugging ---
    sample_frame = "1"
    print(f"\n🔍 Detailed analysis for frame {sample_frame}:")
    
    frame_cams = frame_data[sample_frame]
    available_cams = list(frame_cams.keys())
    print(f"Available cameras in this frame: {available_cams}")
    
    # Count the number of valid (non-zero) keypoints for each camera in the sample frame.
    for cam_str in available_cams:
        if cam_str in frame_cams:
            cam_data = frame_cams[cam_str]
            valid_joints = []
            for joint_str, coords in cam_data.items():
                if coords and len(coords) == 2 and not (coords[0] == 0.0 and coords[1] == 0.0):
                    valid_joints.append(joint_str)
            print(f"  Cam {cam_str}: {len(valid_joints)}/17 valid keypoints: {valid_joints}")
    
    # --- Main Triangulation Loop ---
    print(f"\n🎯 Attempting keypoint triangulation:")
    
    triangulated_count = 0
    # Dictionary to store the final 3D points: out_3d[frame_id][joint_id] = [x, y, z]
    out_3d = {}
    
    # Process all frames found in the input data.
    for frame_id in frame_data.keys():
        out_3d[frame_id] = {}
        frame_cams = frame_data[frame_id]
        
        # Iterate through each of the 17 possible joints.
        for joint_id in range(17):
            pts2d_list = []
            P_list = []
            cam_list = []
            
            # Collect all available 2D observations for the current joint from the target cameras.
            for cam_id in TARGET_CAMS:
                cam_str = str(cam_id)
                joint_str = str(joint_id)
                
                # Check if we have valid data for this camera and joint.
                if (cam_str in frame_cams and 
                    joint_str in frame_cams[cam_str] and
                    frame_cams[cam_str][joint_str] is not None and
                    cam_id in Ps):
                    
                    coords = frame_cams[cam_str][joint_str]
                    
                    # Filter out [0.0, 0.0] coordinates, which indicate an undetected keypoint.
                    if (isinstance(coords, list) and 
                        len(coords) == 2 and
                        not (coords[0] == 0.0 and coords[1] == 0.0)):
                        
                        pts2d_list.append(coords)
                        P_list.append(Ps[cam_id])
                        cam_list.append(cam_id)
            
            # If we have at least 2 valid observations, attempt triangulation.
            if len(pts2d_list) >= 2:
                pt3d = triangulate_point_permissive(P_list, pts2d_list)
                if pt3d is not None:
                    out_3d[frame_id][str(joint_id)] = pt3d
                    triangulated_count += 1
                    
                    # Print debug info for the sample frame.
                    if frame_id == sample_frame:
                        print(f"  Joint {joint_id}: ✅ triangulated with {len(cam_list)} cams {cam_list}")
                else:
                    if frame_id == sample_frame:
                        print(f"  Joint {joint_id}: ❌ triangulation failed ({len(cam_list)} cams: {cam_list})")
            else:
                if frame_id == sample_frame:
                    print(f"  Joint {joint_id}: ❌ insufficient observations ({len(pts2d_list)} cams)")
    
    # --- Save Results ---
    with open('triangulated_positions.json', 'w') as f:
        json.dump(out_3d, f, indent=2)
    
    # --- Final Summary ---
    print(f"\n📊 Final Results:")
    print(f"   Total 3D points triangulated: {triangulated_count}")
    print(f"   Average per frame: {triangulated_count/len(frame_data):.1f}")
    print(f"   File saved: triangulated_positions.json")
    
    # Verify how many frames have at least one triangulated point.
    frame_with_data = 0
    for frame_id, frame_joints in out_3d.items():
        if len(frame_joints) > 0:
            frame_with_data += 1
    
    print(f"   Frames with data: {frame_with_data}/{len(frame_data)}")
    
    return out_3d

# This block ensures the script runs only when executed directly.
if __name__ == "__main__":
    result = triangulate_with_detailed_debug()