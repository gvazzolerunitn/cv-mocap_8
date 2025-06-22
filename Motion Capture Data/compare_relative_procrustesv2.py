#!/usr/bin/env python3
import json
import numpy as np
import csv

# ——— README: fa lo stesso di procrustes_v1 ma usa le triangolazioni mie (triangulated_positions_v3.json) ———

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
HEAD_IDX = KP_NAMES.index("Head")
LW_IDX   = KP_NAMES.index("LHand")
RW_IDX   = KP_NAMES.index("RHand")

# ——— paths ———
MOCAP_JSON  = 'mocap_clip_81_85s.json'
TRIANG_JSON = "../Triangulation and reprojection and 3D/triangulated_positions_v3.json"

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

    # === DEBUG E SINCRONIZZAZIONE CON FRAME RATE ===
    print("=== DATASET ANALYSIS ===")
    print(f"Triangulation frames: {min(tri_keys)} to {max(tri_keys)} ({len(tri_keys)} total)")
    print(f"MoCap frames: {min(mocap_keys)} to {max(mocap_keys)} ({len(mocap_keys)} total)")
    print(f"Video FPS: {VIDEO_FPS}, MoCap FPS: {MOCAP_FPS}, Ratio: {FRAME_RATIO:.2f}")
    
    # Analizza contenuto triangolazione
    non_empty_frames = 0
    total_joints = 0
    for frame_id, joints in tri.items():
        if joints:  # frame non vuoto
            non_empty_frames += 1
            valid_joints = sum(1 for j_id, pos in joints.items() if pos and len(pos) == 3)
            total_joints += valid_joints
    
    print(f"Triangulation: {non_empty_frames} non-empty frames, avg {total_joints/max(1,non_empty_frames):.1f} valid joints/frame")
    
    # Analizza un frame di esempio
    sample_tri_frame = next(iter(tri.values()))
    sample_mocap_frame = next(iter(mocap.values()))
    
    print(f"\nSample triangulation joint: {list(sample_tri_frame.items())[:2]}")
    print(f"Sample mocap joint: {list(sample_mocap_frame.items())[:2]}")
    
    # === TROVA OFFSET OTTIMALE CON CORREZIONE FRAME RATE ===
    print(f"\n=== OFFSET OPTIMIZATION (with frame rate correction) ===")
    best_offset = 0
    best_matches = 0
    best_joint_pairs = 0
    
    # Test range di offset iniziali
    for initial_offset in range(0, 100, 10):  # Test offset iniziale 0-100
        frame_matches = 0
        joint_pairs = 0
        
        for t_str, tv in tri.items():
            video_frame = int(t_str)
            # CORREZIONE: Applica conversione frame rate + offset
            mocap_frame = int(video_frame * FRAME_RATIO) + initial_offset
            m_key = str(mocap_frame)
            
            mv = mocap.get(m_key, {})
            
            if mv:  # Frame MoCap esistente
                frame_matches += 1
                # Conta joint pairs validi
                for j_idx_str, pos_v in tv.items():
                    pos_m = mv.get(j_idx_str)
                    if pos_m is not None and pos_v and len(pos_v) == 3:
                        joint_pairs += 1
        
        if joint_pairs > best_joint_pairs:
            best_joint_pairs = joint_pairs
            best_offset = initial_offset
            best_matches = frame_matches
            
        print(f"Initial offset {initial_offset:2d}: {frame_matches:2d} frame matches, {joint_pairs:4d} joint pairs")
    
    print(f"\n🎯 BEST INITIAL OFFSET: {best_offset} (frame matches: {best_matches}, joint pairs: {best_joint_pairs})")
    
    # Usa l'offset ottimale
    offset = best_offset
    print(f"Using optimized initial offset = {offset} frames")

    # === CONTINUA CON IL CODICE ORIGINALE ===
    # — 4) Raccogli joint‐pairs corrispondenti CON CORREZIONE FRAME RATE
    Xs, Ys = [], []
    debug_pairs = []  # Per debug
    
    for t_str, tv in tri.items():
        video_frame = int(t_str)
        # CORREZIONE: Applica conversione frame rate + offset
        mocap_frame = int(video_frame * FRAME_RATIO) + offset
        m_key = str(mocap_frame)
        
        mv = mocap.get(m_key, {})
        
        for j_idx_str, pos_v in tv.items():
            pos_m = mv.get(j_idx_str)
            if pos_m is not None and pos_v and len(pos_v) == 3:
                Xs.append(pos_m)
                Ys.append(pos_v)
                debug_pairs.append((t_str, m_key, j_idx_str))

    print(f"\n=== FINAL MATCHING ===")
    print(f"Matched {len(Xs)} joint-pairs (target: {non_empty_frames*18})")
    
    if len(Xs) == 0:
        print("❌ ERROR: No joint pairs found! Check data format and offset.")
        print("Sample debug info:")
        sample_frame = list(tri.keys())[0]
        print(f"  Triangulation frame {sample_frame}: {tri[sample_frame]}")
        video_frame = int(sample_frame)
        corresponding_mocap = int(video_frame * FRAME_RATIO) + offset
        print(f"  Video frame {video_frame} → MoCap frame {corresponding_mocap}: exists = {str(corresponding_mocap) in mocap}")
        return
    
    # Mostra alcuni esempi di matching
    print("Sample matches:")
    for i in range(min(5, len(debug_pairs))):
        t_f, m_f, joint = debug_pairs[i]
        print(f"  Video frame {t_f} ↔ MoCap frame {m_f}, joint {joint}")

    # Converti le liste in array NumPy
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

    with open('accuracy_metrics_framerate_corrected.csv','w', newline='') as csvfile:
        w = csv.writer(csvfile)
        # Header
        w.writerow(["Metric", "Rigid_no_scale", "Similarity_with_scale"])
        # Rows
        for name, r_val, s_val in metrics:
            w.writerow([name, f"{r_val:.3f}", f"{s_val:.3f}"])

    print("✅ Saved accuracy metrics to accuracy_metrics_framerate_corrected.csv")

if __name__ == "__main__":
    main()