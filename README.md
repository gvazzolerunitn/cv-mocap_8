# MoCap_8

**Computer Vision Project – MoCap Alignment**  

---

## 🔎 Overview

Pipeline for:

1. **Annotating** 2D skeletons (black‐suit player) over 4 camera views  
2. **Triangulating** from 2D to 3D & reprojection error analysis  
3. **Time-aligning** multi-view video with marker-based MoCap  
4. **Evaluating** accuracy vs. ground-truth annotations and MoCap  
5. **Bonus:** YOLOv8-Pose inference, 2D/3D evaluation

---

## 📁 Repository Structure

```graphql
cv-mocap_8/
├── Annotations/
│ ├── _annotations_rectified_v2.coco.json
│ └── rectify_annotationsV2.py
├── dataset/
│ └── mocap_8.v1i.coco/
│ └── README.dataset.txt/
│ └── README.roboflow.txt/
│ └── train/
│ └── annotations.coco.json ← original COCO train split
├── 3D Pose Estimation Material/
│ ├── rectified_videos.py
│ ├── camera_data/
│ ├── camera_data_with_Rvecs/
│ └── camera_data_with_Rvecs_2ndversion/
│     └── Camera_config2/
├── Motion Capture Data/
│ ├── compare_relative_procrustes.py
│ └── export_mocap_segment.py
│ └── skeleton_comparison.py
│ ├── accuracy_metrics.csv
│ └── mocap_clip_79_83s.json
│ └── frame_number_mat.py
│ └── timestamps_nick3.txt
│ └── timestamps.txt
│ └── Nick_3.mat
├── Triangulation and reprojection and 3D/
│ ├── triangulation.py
│ ├── reprojection.py
│ ├── triangulated_positions_v2.json
│ └── reprojection_results.csv
│ ├── visualize_3d_poses.py
├── YOLO/
│ ├── runs/
│ ├── 3d_confront.py
│ ├── evaluate_yolo_vs_gt.py
│ ├── fix_yolo_coordinates.py
│ └── mpjpe_3d_results.csv
│ └── mpjpe_results.csv
│ ├── triangulated_positions.json
│ ├── triangulation_yolo.py
│ ├── yolo_keypoint_extraction.py
│ ├── yolo_predict.py
│ ├── yolo2d_for_triangulation_rescaled.json
│ ├── yolo2d_for_triangulation.json
├── video/
│ ├── mocap_7/
│ ├── rectified/
│ ├── rectified_rvecs/
│ ├── rectified_rvecs_2nd/
├── requirements.txt
└── README.md

```
---
## ⚙️ Requirements & Installation

1. Create a virtual environment
```bash
python -m venv mocap_env
```
a. Linux/Mac
```bash
source mocap_env/bin/activate
```
b. Windows
```bash
mocap_env\Scripts\activate
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

> **⚠️ Requirements note:** The requirements.txt file specifies only the necessary packages without version constraints, allowing pip to install the most suitable versions for your Python environment.
---
> **⚠️ Note:** All commands below assume you are running them from the project root directory (`cv-mocap_8/`).
> ---
# 🚀 Usage
1. Annotate Skeletons (Roboflow upload)  
Rectify raw annotations and prepare COCO file:
```bash
cd Annotations
python rectify_annotationsV2.py \
  --coco dataset/mocap_8.v1i.coco/train \
  --calib_dir "3D Pose Estimation Material/camera_data_with_Rvecs_2ndversion/Camera_config2"
  --output Annotations/_annotations_rectified_v2.coco.json
```
2. 3D Player’s Position via Triangulation
Triangulate: 
```bash
cd Triangulation and reprojection and 3D
python triangulation.py
```
Analyze reprojection error:
```bash
cd Triangulation and reprojection and 3D
python reprojection.py
```
Visualize the skeleton in 3D:
```bash
cd Triangulation and reprojection and 3D
python visualize_3d_poses.py triangulated_positions_v2.json
```
---
3. Time-aligning with MoCap Data  
Align video to MoCap:
```bash
cd Motion Capture Data
python export_mocap_segment.py
```
Compute Procrustes errors:
```bash
cd Motion Capture Data
python compare_relative_procrustes.py
```
To graphically visualize the difference between the 2 skeleton:
```bash
cd Motion Capture Data
python skeleton_comparison.py
```
---
4. Bonus: YOLOv8-Pose Inference & Evaluation  
Run YOLOv8‑Pose on all views, evaluate 2D & triangulate 3D:
```bash
cd YOLO
python yolo_predict.py
python yolo_keypoint_extraction.py
python fix_yolo_coordinates.py
python triangulation_yolo.py
python evaluate_yolo_vs_gt.py 
python 3d_confront.py
```
---
## 💡 Tip for YOLO  
For more accurate—but more CPU‐intensive—inference, we suggest swapping out `yolov8s-pose.pt` (small) for:

`yolov8m-pose.pt` (medium)

`yolov8l-pose.pt` (large)

`yolov8x-pose.pt` (extra large)

---
## 📂 Data & Results

Annotations: `Annotations/_annotations_rectified_v2.coco.json`

3D Points: `Triangulation and reprojection and 3D/triangulated_positions_v2.json`

Reprojection errors: `Triangulation and reprojection and 3D/reprojection_results.csv`

Mocap alignment metrics: `Motion Capture Data/accuracy_metrics.csv`

2D YOLO evaluation: `YOLO/mpjpe_results.csv`

3D YOLO vs MoCap: `YOLO/mpjpe_3d_results.csv`

--- 

## Authors

- Antonio Di Lauro, Gianluigi Vazzoler
- Computer Vision course, University of Trento, 2024-2025

---
