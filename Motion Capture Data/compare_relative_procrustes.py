#!/usr/bin/env python3
import json
import numpy as np
import csv

# ——— Frame rate configuration ———
VIDEO_FPS = 12
MOCAP_FPS = 100
FRAME_RATIO = MOCAP_FPS / VIDEO_FPS  # ≈ 8.33

# ——— manual keypoint order ———
KP_NAMES = [
  "Hips","RHip","RKnee","RAnkle","RFoot",
  "LHip","LKnee","LAnkle","LFoot",
  "Spine","Neck","Head",
  "RShoulder","RElbow","RHand",
  "LShoulder","LElbow","LHand"
]
HIPS_IDX = KP_NAMES.index("Hips")

# ——— paths ———
MOCAP_JSON  = 'mocap_clip_79_83s.json'
TRIANG_JSON = "../Triangulation and reprojection and 3D/triangulated_positions_v2.json"

def umeyama_alignment(X, Y, with_scale=True):
    N, dim = X.shape
    muX, muY = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc   = X - muX, Y - muY
    SXY      = (Yc.T @ Xc) / N
    U, Dv, Vt = np.linalg.svd(SXY)
    S_mat    = np.eye(dim)
    if np.linalg.det(U)*np.linalg.det(Vt) < 0:
        S_mat[-1, -1] = -1
    R = U @ S_mat @ Vt
    if with_scale:
        varX = (Xc**2).sum()/N
        s    = np.trace(np.diag(Dv) @ S_mat) / varX
    else:
        s = 1.0
    t = muY - s*(R @ muX)
    return s, R, t

def normalize_pose(joints_dict):
    """Restituisce un array (18,3) normalizzato rispetto all'Hips e con scala Hips-Head=1."""
    pose = np.full((18, 3), np.nan)
    for j_str, pos in joints_dict.items():
        idx = int(j_str)
        if pos and len(pos) == 3:
            pose[idx] = np.array(pos)
    if np.isnan(pose[HIPS_IDX]).any():
        return None  # Non confrontabile
    pose = pose - pose[HIPS_IDX]  # Normalizza rispetto all'Hips
    # Normalizza la scala usando la distanza Hips-Head
    HEAD_IDX = KP_NAMES.index("Head")
    if np.isnan(pose[HEAD_IDX]).any():
        return None
    scale = np.linalg.norm(pose[HEAD_IDX])
    if scale < 1e-6:
        return None
    pose = pose / scale
    return pose

def main():
    # — 1) Carica e normalizza triangolazioni al proprio frame0
    tri_raw   = json.load(open(TRIANG_JSON))
    tri_keys  = sorted(int(k) for k in tri_raw)
    start_tri = tri_keys[0]
    tri = { str(k - start_tri): tri_raw[str(k)]
            for k in tri_keys }

    # — 2) Carica e normalizza MoCap al proprio frame0
    mocap_raw  = json.load(open(MOCAP_JSON))
    mocap_keys = sorted(int(k) for k in mocap_raw)
    start_m    = mocap_keys[0]
    mocap = { str(k - start_m): mocap_raw[str(k)]
              for k in mocap_keys }
    
    # Costruisci mapping: frame index → tempo (secondi)
    tri_frame_to_time = {int(k): int(k)/VIDEO_FPS for k in tri_raw}
    mocap_frame_to_time = {int(k): int(k)/MOCAP_FPS for k in mocap_raw}
    mocap_times = sorted(mocap_frame_to_time.values())
    mocap_time_to_frame = {v: k for k, v in mocap_frame_to_time.items()}

    print("=== DATASET ANALYSIS ===")
    print(f"Triangulation frames: {min(tri_keys)} to {max(tri_keys)} ({len(tri_keys)} total)")
    print(f"MoCap frames: {min(mocap_keys)} to {max(mocap_keys)} ({len(mocap_keys)} total)")
    print(f"Video FPS: {VIDEO_FPS}, MoCap FPS: {MOCAP_FPS}, Ratio: {FRAME_RATIO:.2f}")

    # === MATCHING NEAREST IN TIME + NORMALIZZAZIONE RELATIVA ===
    Xs = []
    Ys = []
    debug_pairs = []

    for tri_idx, tri_time in tri_frame_to_time.items():
        # Trova il frame MoCap più vicino in tempo
        mocap_time = min(mocap_times, key=lambda t: abs(t - tri_time))
        mocap_idx = mocap_time_to_frame[mocap_time]
        tri_joints = tri_raw[str(tri_idx)]
        mocap_joints = mocap_raw[str(mocap_idx)]
        tri_pose = normalize_pose(tri_joints)
        mocap_pose = normalize_pose(mocap_joints)
        if tri_pose is None or mocap_pose is None:
            continue
        for j in range(18):
            if not np.isnan(tri_pose[j]).any() and not np.isnan(mocap_pose[j]).any():
                Xs.append(mocap_pose[j])
                Ys.append(tri_pose[j])
                debug_pairs.append((tri_idx, mocap_idx, j))

    print(f"\n=== FINAL MATCHING (relative to Hips) ===")
    print(f"Matched {len(Xs)} joint-pairs (target: {len(tri_keys)*18})")
    if len(Xs) == 0:
        print("❌ ERROR: No joint pairs found! Check data format and offset.")
        return

    print("Sample matches:")
    for i in range(min(5, len(debug_pairs))):
        t_f, m_f, joint = debug_pairs[i]
        print(f"  Video frame {t_f} ↔ MoCap frame {m_f}, joint {joint}")

    X = np.array(Xs, dtype=float)
    Y = np.array(Ys, dtype=float)
    print(f"Final arrays: X.shape = {X.shape}, Y.shape = {Y.shape}")

    # — 5) Procrustes
    s0, R0, t0 = umeyama_alignment(X, Y, with_scale=False)
    Xr0 = (R0 @ X.T).T + t0
    e0  = np.linalg.norm(Xr0 - Y, axis=1)

    s1, R1, t1 = umeyama_alignment(X, Y, with_scale=True)
    Xr1 = (s1 * R1 @ X.T).T + t1
    e1  = np.linalg.norm(Xr1 - Y, axis=1)

    def stats(errs, label):
        print(f"\n— {label} —")
        print(f" MPJPE = {errs.mean():.3f}")
        print(f" RMSE  = {np.sqrt((errs**2).mean()):.3f}")
        print(f" Max   = {errs.max():.3f}")
        print(f" Min   = {errs.min():.3f}")
        print(f" Count = {len(errs)}")

    stats(e0, "Rigid (no scale)")
    stats(e1, "Similarity (with scale)")

    # 6) Salva su CSV con nome diverso
    metrics = [
        ("Count",           len(e0),                len(e1)),
        ("MPJPE (px)",      e0.mean(),              e1.mean()),
        ("MSE (px^2)",      (e0**2).mean(),         (e1**2).mean()),
        ("RMSE (px)",       np.sqrt((e0**2).mean()), np.sqrt((e1**2).mean())),
        ("Median (px)",     np.median(e0),         np.median(e1)),
        ("25th Percentile", np.percentile(e0, 25), np.percentile(e1, 25)),
        ("75th Percentile", np.percentile(e0, 75), np.percentile(e1, 75)),
        ("Max (px)",        e0.max(),               e1.max()),
        ("Min (px)",        e0.min(),               e1.min()),
    ]

    with open('accuracy_metrics.csv','w', newline='') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["Metric", "Rigid_no_scale", "Similarity_with_scale"])
        for name, r_val, s_val in metrics:
            w.writerow([name, f"{r_val:.3f}", f"{s_val:.3f}"])

    print("✅ Saved accuracy metrics to accuracy_metrics.csv")

if __name__ == "__main__":
    main()