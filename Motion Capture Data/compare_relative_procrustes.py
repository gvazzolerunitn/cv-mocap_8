#!/usr/bin/env python3
"""
Align and evaluate 3D triangulated poses against motion-capture data.
This script:
  1. Loads a short MoCap clip and the full triangulated sequence.
  2. Maps video frames to MoCap frames using their timestamps.
  3. Normalizes each 3D pose relative to the Hips joint and scales by Hip-Head distance.
  4. Matches corresponding joints between the two sequences.
  5. Computes rigid and similarity (scale + rotation + translation) Procrustes errors.
  6. Outputs summary statistics to the console and writes them to a CSV.
"""

import json
import numpy as np
import csv

# ——— Configuration ———
VIDEO_FPS = 12           # Frame rate of the video/triangulation data
MOCAP_FPS = 100          # Frame rate of the MoCap recording
FRAME_RATIO = MOCAP_FPS / VIDEO_FPS  # ≈ 8.33

# ——— Joint definitions ———
KP_NAMES = [
    "Hips","RHip","RKnee","RAnkle","RFoot",
    "LHip","LKnee","LAnkle","LFoot",
    "Spine","Neck","Head",
    "RShoulder","RElbow","RHand",
    "LShoulder","LElbow","LHand"
]
HIPS_IDX = KP_NAMES.index("Hips")  # Index of the Hips joint

# ——— File paths ———
MOCAP_JSON  = 'mocap_clip_79_83s.json'
TRIANG_JSON = "../Triangulation and reprojection and 3D/triangulated_positions_v2.json"

def umeyama_alignment(X, Y, with_scale=True):
    """
    Estimate a rigid or similarity transform (Umeyama) that aligns X to Y.
    X, Y: arrays of shape (N, 3)
    with_scale: if True, include isotropic scale in the transform.
    Returns (s, R, t) such that Y ≈ s * R @ X.T + t
    """
    N, dim = X.shape
    # Compute centroids
    muX, muY = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc   = X - muX, Y - muY
    # Covariance matrix
    SXY      = (Yc.T @ Xc) / N
    # SVD decomposition
    U, Dv, Vt = np.linalg.svd(SXY)
    # Reflection correction if needed
    S_mat    = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_mat[-1, -1] = -1
    # Rotation
    R = U @ S_mat @ Vt
    # Scale (optional)
    if with_scale:
        varX = (Xc**2).sum() / N
        s    = np.trace(np.diag(Dv) @ S_mat) / varX
    else:
        s = 1.0
    # Translation
    t = muY - s * (R @ muX)
    return s, R, t

def normalize_pose(joints_dict):
    """
    Convert a dict of joint positions to a normalized (18,3) array:
      - Origin at Hips joint.
      - Scaled so that the distance from Hips to Head = 1.
    Returns None if normalization fails (missing keypoints or zero scale).
    """
    pose = np.full((18, 3), np.nan)
    # Fill available joints
    for j_str, pos in joints_dict.items():
        idx = int(j_str)
        if isinstance(pos, list) and len(pos) == 3:
            pose[idx] = np.array(pos)
    # Require Hips to be present
    if np.isnan(pose[HIPS_IDX]).any():
        return None
    # Translate Hips to origin
    pose -= pose[HIPS_IDX]
    # Scale by Hip-Head distance
    HEAD_IDX = KP_NAMES.index("Head")
    if np.isnan(pose[HEAD_IDX]).any():
        return None
    scale = np.linalg.norm(pose[HEAD_IDX])
    if scale < 1e-6:
        return None
    return pose / scale

def main():
    # 1) Load and re-index triangulated poses relative to frame 0
    tri_raw   = json.load(open(TRIANG_JSON))
    tri_keys  = sorted(int(k) for k in tri_raw)
    start_tri = tri_keys[0]
    tri = { str(k - start_tri): tri_raw[str(k)] for k in tri_keys }

    # 2) Load and re-index MoCap clip poses relative to frame 0
    mocap_raw  = json.load(open(MOCAP_JSON))
    mocap_keys = sorted(int(k) for k in mocap_raw)
    start_m    = mocap_keys[0]
    mocap = { str(k - start_m): mocap_raw[str(k)] for k in mocap_keys }
    
    # Build frame→time mappings
    tri_frame_to_time = {int(k): int(k) / VIDEO_FPS for k in tri_raw}
    mocap_frame_to_time = {int(k): int(k) / MOCAP_FPS for k in mocap_raw}
    mocap_times = sorted(mocap_frame_to_time.values())
    mocap_time_to_frame = {t: k for k, t in mocap_frame_to_time.items()}

    # Print dataset info
    print("=== DATASET ANALYSIS ===")
    print(f"Triangulation frames: {min(tri_keys)}–{max(tri_keys)} ({len(tri_keys)} total)")
    print(f"MoCap frames: {min(mocap_keys)}–{max(mocap_keys)} ({len(mocap_keys)} total)")
    print(f"Video FPS: {VIDEO_FPS}, MoCap FPS: {MOCAP_FPS}, Ratio: {FRAME_RATIO:.2f}")

    # 3) Match each triangulated frame to the nearest MoCap frame in time
    Xs, Ys = [], []
    debug_pairs = []
    for tri_idx, tri_time in tri_frame_to_time.items():
        # Find closest MoCap timestamp
        closest_time = min(mocap_times, key=lambda t: abs(t - tri_time))
        mocap_idx = mocap_time_to_frame[closest_time]

        tri_pose_dict   = tri_raw[str(tri_idx)]
        mocap_pose_dict = mocap_raw[str(mocap_idx)]
        tri_pose = normalize_pose(tri_pose_dict)
        mocap_pose = normalize_pose(mocap_pose_dict)
        if tri_pose is None or mocap_pose is None:
            continue

        # Collect per-joint pairs
        for j in range(18):
            if not np.isnan(tri_pose[j]).any() and not np.isnan(mocap_pose[j]).any():
                Xs.append(mocap_pose[j])
                Ys.append(tri_pose[j])
                debug_pairs.append((tri_idx, mocap_idx, j))

    # Report matching results
    print(f"\n=== FINAL MATCHING (relative to Hips) ===")
    print(f"Matched {len(Xs)} joint-pairs (ideal: {len(tri_keys)*18})")
    if not Xs:
        print("❌ ERROR: No joint pairs found! Check data and offset.")
        return

    print("Sample matches:")
    for t_f, m_f, joint in debug_pairs[:5]:
        print(f"  Video frame {t_f} ↔ MoCap frame {m_f}, joint {joint}")

    X = np.array(Xs)
    Y = np.array(Ys)
    print(f"Final arrays: X.shape = {X.shape}, Y.shape = {Y.shape}")

    # 4) Procrustes alignment and error computation
    s0, R0, t0 = umeyama_alignment(X, Y, with_scale=False)
    Xr0 = (R0 @ X.T).T + t0
    e0  = np.linalg.norm(Xr0 - Y, axis=1)

    s1, R1, t1 = umeyama_alignment(X, Y, with_scale=True)
    Xr1 = (s1 * R1 @ X.T).T + t1
    e1  = np.linalg.norm(Xr1 - Y, axis=1)

    def stats(errs, label):
        """Print basic error statistics."""
        print(f"\n— {label} —")
        print(f" MPJPE = {errs.mean():.3f}")
        print(f" RMSE  = {np.sqrt((errs**2).mean()):.3f}")
        print(f" Max   = {errs.max():.3f}")
        print(f" Min   = {errs.min():.3f}")
        print(f" Count = {len(errs)}")

    stats(e0, "Rigid (no scale)")
    stats(e1, "Similarity (with scale)")

    # 5) Save detailed metrics to CSV
    metrics = [
        ("Count",           len(e0),                  len(e1)),
        ("MPJPE (px)",      e0.mean(),                e1.mean()),
        ("MSE (px^2)",      (e0**2).mean(),           (e1**2).mean()),
        ("RMSE (px)",       np.sqrt((e0**2).mean()),  np.sqrt((e1**2).mean())),
        ("Median (px)",     np.median(e0),            np.median(e1)),
        ("25th Percentile", np.percentile(e0, 25),    np.percentile(e1, 25)),
        ("75th Percentile", np.percentile(e0, 75),    np.percentile(e1, 75)),
        ("Max (px)",        e0.max(),                 e1.max()),
        ("Min (px)",        e0.min(),                 e1.min()),
    ]

    with open('accuracy_metrics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Rigid_no_scale", "Similarity_with_scale"])
        for name, r_val, s_val in metrics:
            writer.writerow([name, f"{r_val:.3f}", f"{s_val:.3f}"])

    print("✅ Saved accuracy metrics to accuracy_metrics.csv")

if __name__ == "__main__":
    main()
