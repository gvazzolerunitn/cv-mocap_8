# export_mocap_segment.py

import json
import numpy as np
from scipy.io import loadmat

MAT_FILE    = 'Nick_3.mat'               
OUTPUT_JSON = 'mocap8_positions.json'

# 1) Carica il .mat
mat = loadmat(MAT_FILE, struct_as_record=False, squeeze_me=True)
mo  = mat['Nick_3']

# 2) Prendi i dati 3×J×T
skel = mo.Skeletons
P    = skel.PositionData    # shape (3, J, T)
_, J, T = P.shape

# 3) Filtra i soli t con dati validi (non tutti NaN) e salva con chiave = t
out = {}
for t in range(T):
    # se *tutti* i joint sono NaN per quel t, skip
    if np.all(np.isnan(P[:,:,t])):
        continue
    # altrimenti estrai tutti i joint j
    joints = { str(j): [float(P[0,j,t]), float(P[1,j,t]), float(P[2,j,t])] 
               for j in range(J)
               if not np.isnan(P[0,j,t])  # skip joint singoli NaN
             }
    out[str(t)] = joints

# 4) Scrivi JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(out, f, indent=2)

print(f"✅ Esportati {len(out)} frame MoCap validi in {OUTPUT_JSON}")
