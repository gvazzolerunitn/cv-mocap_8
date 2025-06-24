import json
import numpy as np
import os

def rescale_yolo_coordinates():
    """Rescales YOLO coordinates to match the existing camera calibrations."""
    
    # Load YOLO 2D keypoint data
    with open('yolo2d_for_triangulation.json') as f:
        yolo_2d = json.load(f)
    
    # Load camera calibrations to determine the scaling factors
    calib_folder = '../3D Pose Estimation Material/camera_data_with_Rvecs_2ndversion/Camera_config2'
    
    scale_factors = {}
    
    # Iterate through the target cameras used for triangulation
    for cam_id in [2, 5, 8, 13]:
        cam_path = os.path.join(calib_folder, f'cam_{cam_id}')
        
        if os.path.exists(cam_path):
            for root, dirs, files in os.walk(cam_path):
                if 'camera_calib.json' in files:
                    calib_file = os.path.join(root, 'camera_calib.json')
                    with open(calib_file) as f:
                        calib = json.load(f)
                    
                    # Extract the intrinsic matrix K
                    K = np.array(calib['mtx'])
                    calib_cx = K[0, 2]
                    calib_cy = K[1, 2]
                    
                    # The resolution expected by the calibration (derived from the principal point)
                    expected_width = calib_cx * 2
                    expected_height = calib_cy * 2
                    
                    # The actual resolution of the images processed by YOLO
                    actual_width = 3840
                    actual_height = 2160
                    
                    # Calculate scaling factors to convert YOLO coordinates to the calibration's coordinate system
                    scale_x = expected_width / actual_width
                    scale_y = expected_height / actual_height
                    
                    scale_factors[str(cam_id)] = {'x': scale_x, 'y': scale_y}
                    
                    print(f"Cam {cam_id}: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")
    
    # Apply the calculated scaling factors to the coordinates
    corrected_data = {}
    
    for frame in yolo_2d:
        corrected_data[frame] = {}
        
        for cam in yolo_2d[frame]:
            corrected_data[frame][cam] = {}
            
            if cam in scale_factors:
                scale_x = scale_factors[cam]['x']
                scale_y = scale_factors[cam]['y']
                
                for joint_id, (x, y) in yolo_2d[frame][cam].items():
                    # Scale the coordinates to match the calibration's system
                    corrected_x = x * scale_x
                    corrected_y = y * scale_y
                    corrected_data[frame][cam][joint_id] = [corrected_x, corrected_y]
            else:
                # If no calibration is found, keep the original coordinates
                corrected_data[frame][cam] = yolo_2d[frame][cam]
    
    # Save the rescaled data to a new JSON file
    with open('yolo2d_for_triangulation_rescaled.json', 'w') as f:
        json.dump(corrected_data, f, indent=2)
    
    print("\nRescaled coordinates saved to 'yolo2d_for_triangulation_rescaled.json'")

if __name__ == "__main__":
    rescale_yolo_coordinates()