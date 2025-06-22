#!/usr/bin/env python3
# -----------------------------------------------------------
#  export_mocap_clip.py
#  Estrae dal file Nick_3.mat le posizioni 3-D dei joint
#  nel range 81-85 s e le salva in formato JSON.
# -----------------------------------------------------------

import json
import numpy as np
from scipy.io import loadmat

MAT_FILE     = "Nick_3.mat"
OUTPUT_JSON  = "mocap_clip_81_85s.json"

# Ordine COCO da mantenere nell'output (18 key-points)
COCO_KPTS = [
    "Hips",  "RHip", "RKnee", "RAnkle", "RFoot",
    "LHip",  "LKnee", "LAnkle", "LFoot",
    "Spine", "Neck",  "Head",
    "RShoulder", "RElbow", "RHand",
    "LShoulder", "LElbow", "LHand"
]

# Mappatura COCO → nomi del tuo scheletro MoCap
COCO_TO_MATLAB = {
    "Hips":       "Hips",
    "RHip":       "RightUpLeg",
    "RKnee":      "RightLeg",
    "RAnkle":     "RightFoot",
    "RFoot":      "RightToeBase",
    "LHip":       "LeftUpLeg",
    "LKnee":      "LeftLeg",
    "LAnkle":     "LeftFoot",
    "LFoot":      "LeftToeBase",
    "Spine":      "Spine",
    "Neck":       "Neck",
    "Head":       "Head",
    "RShoulder":  "RightShoulder",
    "RElbow":     "RightForeArm",
    "RHand":      "RightHand",
    "LShoulder":  "LeftShoulder",
    "LElbow":     "LeftForeArm",
    "LHand":      "LeftHand"
}

# ---------- 1. Carica il .mat ----------
mat = loadmat(MAT_FILE, struct_as_record=False, squeeze_me=True)
mo  = mat["Nick_3"]
skel = mo.Skeletons                               # 1×1 struct

# ---------- 2. Etichette dei joint ----------
if hasattr(skel, "SegmentLabels"):
    labels = np.asarray(skel.SegmentLabels).astype(str).tolist()
elif hasattr(skel, "JointNames"):
    labels = np.asarray(skel.JointNames).astype(str).tolist()
else:
    raise ValueError("Nel campo Skeletons non trovo né 'SegmentLabels' né 'JointNames'")
    
label_to_idx = {name: i for i, name in enumerate(labels)}

# ---------- 3. PositionData → shape (T, J, 3) ----------
P = skel.PositionData
if P.ndim == 3:               # (3,J,T) | (T,J,3) | (J,3,T)
    if P.shape[0] == 3:
        P = np.transpose(P, (2, 1, 0))
    elif P.shape[2] == 3:
        pass  # già (T,J,3)
    elif P.shape[1] == 3:
        P = np.transpose(P, (2, 0, 1))
    else:
        raise ValueError("Forma 3-D inattesa per PositionData")
elif P.ndim == 2:             # flatten (T, J*3)
    if P.shape[1] % 3 != 0:
        raise ValueError("PositionData flatten con dimensione incompatibile")
    P = P.reshape((-1, P.shape[1] // 3, 3))
else:
    raise ValueError("Dimensionalità PositionData non gestita")
T, J, _ = P.shape

# ---------- 4. Intervallo temporale ----------
fps          = int(mo.FrameRate)      # 100 fps
START_SEC    = 81
END_SEC      = 85
start_frame  = int(START_SEC * fps)        # 0-based → 8100
end_frame    = int(END_SEC   * fps) - 1    # 0-based → 8499
if end_frame >= T:
    raise ValueError("End-time richiesto oltre la durata della registrazione")

print(f"Estrazione frame {start_frame}–{end_frame} inclusivi "
      f"({end_frame-start_frame+1} frame)")

# ---------- 5. Costruisci il dizionario JSON ----------
out = {}
for f in range(start_frame, end_frame + 1):        # inclusivo
    if np.isnan(P[f]).all():                       # frame vuoto
        continue
    joints = {}
    for coco_idx, coco_name in enumerate(COCO_KPTS):
        mcap_name = COCO_TO_MATLAB[coco_name]
        j_idx = label_to_idx.get(mcap_name)
        if j_idx is None:
            continue                              # quel joint non esiste
        xyz = P[f, j_idx]
        if np.isnan(xyz).all():                   # joint NaN su tutte e 3
            continue
        joints[str(coco_idx)] = xyz.astype(float).round(6).tolist()
    if joints:
        out[str(f)] = joints

# ---------- 6. Salva ----------
with open(OUTPUT_JSON, "w") as fp:
    json.dump(out, fp, indent=2)

print(f"✅  Salvati {len(out)} frame validi in '{OUTPUT_JSON}'")
