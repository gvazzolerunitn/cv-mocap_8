import json
import numpy as np
from scipy.io import loadmat

MAT_FILE    = 'Nick_3.mat'
OUTPUT_JSON = 'mocap8_positions_window.json'

# Mappatura: COCO_KPTS → MATLAB_KPTS
COCO_TO_MATLAB = {
    "Hips": "Hips",
    "RHip": "RightUpLeg",
    "RKnee": "RightLeg",
    "RAnkle": "RightFoot",
    "RFoot": "RightToeBase",
    "LHip": "LeftUpLeg",
    "LKnee": "LeftLeg",
    "LAnkle": "LeftFoot",
    "LFoot": "LeftToeBase",
    "Spine": "Spine",
    "Neck": "Neck",
    "Head": "Head",
    "RShoulder": "RightShoulder",
    "RElbow": "RightForeArm",
    "RHand": "RightHand",
    "LShoulder": "LeftShoulder",
    "LElbow": "LeftForeArm",
    "LHand": "LeftHand"
}

MATLAB_KPTS = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftForeArmRoll', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightForeArmRoll', 'RightHand', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'
]
matlab_name_to_idx = {name: idx for idx, name in enumerate(MATLAB_KPTS)}

# Carica il .mat
mat = loadmat(MAT_FILE, struct_as_record=False, squeeze_me=True)
mo  = mat['Nick_3']

# Prendi i dati 3×J×T
skel = mo.Skeletons
P    = skel.PositionData    # shape (3, J, T)
_, J, T = P.shape

# Calcola fps effettivo
duration_sec = 125  # 2:05 = 125 secondi
fps = T / duration_sec

start_sec = 81
end_sec = 85
start_frame = int(start_sec * fps)
end_frame = int(end_sec * fps)

print(f"Estrazione frame da {start_frame} a {end_frame-1} (secondi {start_sec}-{end_sec})")

out = {}
for t in range(start_frame, end_frame):
    if np.all(np.isnan(P[:,:,t])):
        continue
    joints = {}
    for idx_out, (coco_kpt, matlab_kpt) in enumerate(COCO_TO_MATLAB.items()):
        idx = matlab_name_to_idx.get(matlab_kpt)
        if idx is not None and not np.isnan(P[0, idx, t]):
            joints[str(idx_out)] = [
                float(P[0, idx, t]),
                float(P[1, idx, t]),
                float(P[2, idx, t])
            ]
    out[str(t)] = joints

with open(OUTPUT_JSON, 'w') as f:
    json.dump(out, f, indent=2)

print(f"✅ Esportati {len(out)} frame MoCap validi in {OUTPUT_JSON}")