import json
import numpy as np

# ——— FILE INPUT/OUTPUT ———
MOCAP_JSON   = 'mocap8_positions_window.json'                     # esportato da export_mocap_segment.py
TRIANG_JSON  = 'triangulated_positions_window.json'        # triangolato su 0–47

# ——— Procrustes (Umeyama) alignment ———
def umeyama_alignment(X, Y, with_scale=True):
    """
    Stima Rigid/Similarity transform che allinea X a Y.
    X, Y: array Nx3.
    with_scale: True per includere fattore di scala.
    Restituisce s, R, t.
    """
    N, dim = X.shape
    # centri
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc  = X - muX
    Yc  = Y - muY

    # matrice di covarianza
    SXY = (Yc.T @ Xc) / N

    # SVD
    U, D_vec, Vt = np.linalg.svd(SXY)

    # correzione in caso di riflessione
    S_mat = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_mat[-1, -1] = -1

    # rotazione
    R = U @ S_mat @ Vt

    # scala
    if with_scale:
        # varianza di X
        varX = (Xc ** 2).sum() / N
        # costruisci diag(D_vec)
        D_mat = np.diag(D_vec)
        # trace(D_mat @ S_mat)
        scale_num = np.trace(D_mat @ S_mat)
        s = scale_num / varX
    else:
        s = 1.0

    # traslazione
    t = muY - s * (R @ muX)
    return s, R, t

#calcolo offset tra frame MoCap e triangolazione
# cerca il primo frame valido in ciascun dataset
def find_first_valid_frame(data):
    for t_str in sorted(data, key=lambda x: int(x)):
        joints = data[t_str]
        for v in joints.values():
            if isinstance(v, list) and len(v) == 3:
                return int(t_str)
    return None

def main():
    # carica JSON MoCap e triangolazioni
    mocap = json.load(open(MOCAP_JSON, 'r'))
    tri   = json.load(open(TRIANG_JSON, 'r'))

    # Calcola offset automaticamente
    first_mocap = find_first_valid_frame(mocap)
    first_triang = find_first_valid_frame(tri)
    offset = first_mocap - first_triang
    print(f"Offset calcolato automaticamente: {offset} (frame MoCap = frame Triangolazione + {offset})")

    # costruisci liste di punti corrispondenti
    X_list, Y_list = [], []
    for t_str, tv in tri.items():
        mocap_frame = str(int(t_str) + offset)
        mv = mocap.get(mocap_frame, {})
        for j_str, pos_v in tv.items():
            pos_m = mv.get(j_str)
            if pos_m is not None:
                X_list.append(pos_m)
                Y_list.append(pos_v)

    X = np.array(X_list, dtype=float)
    Y = np.array(Y_list, dtype=float)
    print(f"Matched {len(X)} joint-pairs (should be ≈48*#joints)")

    # Rigid (no scale)
    s_r, R_r, t_r = umeyama_alignment(X, Y, with_scale=False)
    Xr = (R_r @ X.T).T + t_r
    errs_r = np.linalg.norm(Xr - Y, axis=1)

    # Similarity (with scale)
    s_s, R_s, t_s = umeyama_alignment(X, Y, with_scale=True)
    Xs = (s_s * R_s @ X.T).T + t_s
    errs_s = np.linalg.norm(Xs - Y, axis=1)

    # funzione per stampare statistiche
    def print_stats(errs, label):
        print(f"\n— {label} —")
        print(f" MPJPE = {errs.mean():.3f}")
        print(f" RMSE  = {np.sqrt((errs**2).mean()):.3f}")
        print(f" Max   = {errs.max():.3f}")
        print(f" Min   = {errs.min():.3f}")
        print(f" Count = {len(errs)}")

    print_stats(errs_r, "Rigid (no scale)")
    print_stats(errs_s, "Similarity (with scale)")

if __name__=='__main__':
    main()
