#!/usr/bin/env python3
import json
import numpy as np

# ——— manual keypoint order (stesso che in triangulation2.py) ———
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
TRIANG_JSON = 'triangulated_positions_window.json'

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

    # — 3) Offset manuale dal tiro (MoCap = Video + 285)
    offset = 285
    print(f"Using manual offset = {offset} frames  (MoCap = Video + {offset})")

    # — 4) Raccogli joint‐pairs corrispondenti
    Xs, Ys = [], []
    for t_str, tv in tri.items():
        m_key = str(int(t_str) + offset)
        mv    = mocap.get(m_key, {})
        for j_str, pos_v in tv.items():
            pos_m = mv.get(j_str)
            if pos_m is not None:
                Xs.append(pos_m)
                Ys.append(pos_v)

    X = np.array(Xs, dtype=float)
    Y = np.array(Ys, dtype=float)
    print(f"Matched {len(X)} joint-pairs  (idealmente 48×18=864)")

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

if __name__ == "__main__":
    main()
