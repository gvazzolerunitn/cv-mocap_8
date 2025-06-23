# cv-mocap_8

**Computer Vision Project – MoCap Alignment**  
Group member:
Antonio Di Lauro,
Gianluigi Vazzoler

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
│ ├── accuracy_metrics.csv
│ └── mocap_clip_79_83s.json
│ └── numero_frame_mat.py
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
│ ├── yolo_predict.py
│ ├── YOLO_keypoint_extraction.py
│ ├── triangulation_YOLO.py
│ ├── evaluate_yolo_vs_gt.py
│ ├── 3d_confront.py
│ ├── yolo2d_for_triangulation.json
│ ├── triangulated_positions_yolo.json
│ └── mpjpe_3d_results.csv
│ └── mpjpe_results.csv
├── video/
│ ├── og/
│ ├── rectified/
│ ├── rectified_rvecs/
│ └── fixed/rectified_rvecs_2nd/
├── requirements.txt
└── README.md

```
---
## ⚙️ Requirements & Installation

# Crea ambiente virtuale
```bash
python -m venv mocap_env
```
# Attiva ambiente (Linux/Mac)
```bash
source mocap_env/bin/activate
```
# Installa dipendenze
```bash
pip install -r requirements.txt
```
---
# 🚀 Usage
1. Annotate Skeletons
2. 3D Player’s Position
```bash

```
