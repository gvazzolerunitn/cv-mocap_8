# filter_triangulated.py

import json
from pathlib import Path

FPS_VIDEO = 25.007
START_TS  = '01:21'
END_TS    = '01:25'

def time_to_frame(ts):
    m, s = ts.split(':')
    total_s = int(m)*60 + float(s)
    return int(round(total_s * FPS_VIDEO))

# 1) calcola gli estremi in frame
start_f = time_to_frame(START_TS)
end_f   = time_to_frame(END_TS)
print(f"Filtering triangulated frames between {start_f} and {end_f}")

# 2) carica il JSON completo
tri_all = json.load(open('../Triangulation and reprojection and 3D/triangulated_positions_v2.json'))

# 3) crea il sottoinsieme
tri_sub = {f:tri_all[f] for f in tri_all
           if start_f <= int(f) <= end_f}

# 4) salva
with open('triangulated_positions_window.json','w') as out:
    json.dump(tri_sub, out, indent=2)

print(f"Saved {len(tri_sub)} frames to triangulated_positions_window.json")
