import json
import numpy as np
import csv
from scipy.spatial import procrustes

# Paths to the YOLO-triangulated 3D output and the ground-truth triangulated positions
YOLO_3D_PATH = 'triangulated_positions.json'
GT_3D_PATH   = '../Motion Capture Data/mocap_clip_79_83s.json'

# Number of joints expected in each pose (17 for COCO)
NUM_JOINTS = 17

# Load the YOLO-based 3D reconstructions
with open(YOLO_3D_PATH) as f:
    yolo_3d = json.load(f)

# Load the ground-truth 3D triangulations
with open(GT_3D_PATH) as f:
    gt_3d = json.load(f)

# --- START OF FRAME ALIGNMENT LOGIC ---

# 1. Normalize frame numbers to integers for reliable calculations
yolo_frames_int = sorted([int(k) for k in yolo_3d.keys()])
gt_frames_int = sorted([int(k) for k in gt_3d.keys()])

if not yolo_frames_int or not gt_frames_int:
    print("Error: One of the two datasets contains no frames. Exiting.")
    exit()

# 2. Calculate the temporal offset between the two datasets
# It is assumed that the first frame of the YOLO sequence corresponds to the first of the MoCap sequence.
offset = gt_frames_int[0] - yolo_frames_int[0]
print(f"Initial frames -> YOLO: {yolo_frames_int[0]}, MoCap: {gt_frames_int[0]}")
print(f"Calculated frame offset: {offset}")

# 3. Create a map from the YOLO frame to the corresponding MoCap frame
# A YOLO frame 'f' is mapped to the MoCap frame 'f + offset' if the latter exists.
frame_mapping = {}
for yolo_frame in yolo_frames_int:
    gt_equivalent = yolo_frame + offset
    if gt_equivalent in gt_frames_int:
        # Use strings for keys for consistency with the JSON format
        frame_mapping[str(yolo_frame)] = str(gt_equivalent)

common_yolo_frames = sorted(frame_mapping.keys(), key=int)
print(f"Found {len(common_yolo_frames)} common frames after alignment with offset.")

# --- END OF FRAME ALIGNMENT LOGIC ---

# Prepare lists for statistics
mpjpe_list = []
per_joint_errors = [[] for _ in range(NUM_JOINTS)]
frame_joint_errors = {} # To save errors for the CSV file

# Iterate over each common frame to perform alignment and calculate the error
for yolo_frame_str in common_yolo_frames:
    gt_frame_str = frame_mapping[yolo_frame_str]
    
    yolo_points_for_frame = []
    gt_points_for_frame = []
    valid_joint_indices = []

    # 1. Collect all valid joint pairs for the current frame
    for j in range(NUM_JOINTS):
        yolo_joint = yolo_3d[yolo_frame_str].get(str(j))
        gt_joint   = gt_3d[gt_frame_str].get(str(j))

        # Check if both joints are valid (not None, NaN, or Inf)
        if (yolo_joint is not None and gt_joint is not None and
            not np.any(np.isnan(yolo_joint)) and not np.any(np.isnan(gt_joint)) and
            not np.any(np.isinf(yolo_joint)) and not np.any(np.isinf(gt_joint))):
            
            yolo_points_for_frame.append(yolo_joint)
            gt_points_for_frame.append(gt_joint)
            valid_joint_indices.append(j)

    # 2. Perform alignment if there are enough points (at least 3 for stability)
    if len(yolo_points_for_frame) < 3:
        print(f"Skipping frame {yolo_frame_str}: only found {len(yolo_points_for_frame)} valid joint pairs.")
        continue

    yolo_np = np.array(yolo_points_for_frame)
    gt_np = np.array(gt_points_for_frame)

    # 3. Align the YOLO points to the GT points using Procrustes analysis
    gt_aligned, yolo_aligned, disparity = procrustes(gt_np, yolo_np)

    # 4. Calculate errors on the aligned points
    errors = np.linalg.norm(gt_aligned - yolo_aligned, axis=1)
    
    # 5. Save the errors for statistics and CSV export
    frame_joint_errors[yolo_frame_str] = {}
    mpjpe_list.extend(errors)
    for i, err in enumerate(errors):
        joint_idx = valid_joint_indices[i]
        per_joint_errors[joint_idx].append(err)
        frame_joint_errors[yolo_frame_str][joint_idx] = err

# If errors were collected, print the summary statistics
if mpjpe_list:
    print(f"\nEvaluation on {len(frame_joint_errors)} frames and {len(mpjpe_list)} joint pairs (post-alignment):")
    print(f"Mean MPJPE:   {np.mean(mpjpe_list):.2f} mm")
    print(f"Median MPJPE: {np.median(mpjpe_list):.2f} mm")
    print(f"Min / Max:     {np.min(mpjpe_list):.2f} / {np.max(mpjpe_list):.2f} mm")
    print("\nMean error per joint:")
    for j, errs in enumerate(per_joint_errors):
        if errs:
            print(f"  Joint {j}: {np.mean(errs):.2f} mm ({len(errs)} measurements)")
        else:
            print(f"  Joint {j}: N/A")
else:
    print("No comparable joint pairs found after alignment!")

# Write the detailed results to a CSV file
with open('mpjpe_3d_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['frame', 'joint', 'mpjpe_mm_post_alignment'])
    for frame, errors_dict in sorted(frame_joint_errors.items(), key=lambda item: int(item[0])):
        for joint_idx, error in errors_dict.items():
            writer.writerow([frame, joint_idx, error])
    
    writer.writerow([])
    writer.writerow(['STATISTIC', 'VALUE_mm'])
    if mpjpe_list:
        writer.writerow(['mean_mpjpe', f"{np.mean(mpjpe_list):.2f}"])
        writer.writerow(['median_mpjpe', f"{np.median(mpjpe_list):.2f}"])
        writer.writerow(['min_mpjpe', f"{np.min(mpjpe_list):.2f}"])
        writer.writerow(['max_mpjpe', f"{np.max(mpjpe_list):.2f}"])
        writer.writerow(['total_joint_pairs', len(mpjpe_list)])
        writer.writerow(['total_frames', len(frame_joint_errors)])
        writer.writerow([])
        writer.writerow(['JOINT_INDEX', 'MEAN_ERROR_mm', 'NUM_MEASUREMENTS'])
        for j, errs in enumerate(per_joint_errors):
            if errs:
                writer.writerow([j, f"{np.mean(errs):.2f}", len(errs)])
            else:
                writer.writerow([j, 'N/A', 0])
    else:
        writer.writerow(['mean_mpjpe', 'N/A'])

print("\n✅ Results with Procrustes alignment and temporal offset saved to mpjpe_3d_results.csv")