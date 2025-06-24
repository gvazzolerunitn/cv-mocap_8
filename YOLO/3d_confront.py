import json
import numpy as np
import csv
from scipy.spatial import procrustes

# Paths to the YOLO-triangulated 3D output and the ground-truth triangulated positions
YOLO_3D_PATH = 'triangulated_positions.json'
GT_3D_PATH   = '../Motion Capture Data/mocap_clip_79_83s.json'

# Number of joints expected in each pose (17 for COCO)
NUM_JOINTS = 17

# Load the YOLO-based 3D reconstructions
with open(YOLO_3D_PATH) as f:
    yolo_3d = json.load(f)

# Load the ground-truth 3D triangulations
with open(GT_3D_PATH) as f:
    gt_3d = json.load(f)

# --- INIZIO LOGICA DI ALLINEAMENTO FRAME ---

# 1. Normalizza i numeri di frame a interi per un calcolo affidabile
yolo_frames_int = sorted([int(k) for k in yolo_3d.keys()])
gt_frames_int = sorted([int(k) for k in gt_3d.keys()])

if not yolo_frames_int or not gt_frames_int:
    print("Errore: Uno dei due dataset non contiene frame. Uscita.")
    exit()

# 2. Calcola l'offset temporale tra i due set di dati
# Si assume che il primo frame della sequenza YOLO corrisponda al primo della sequenza MoCap.
offset = gt_frames_int[0] - yolo_frames_int[0]
print(f"Frame iniziali -> YOLO: {yolo_frames_int[0]}, MoCap: {gt_frames_int[0]}")
print(f"Offset calcolato tra i frame: {offset}")

# 3. Crea una mappa dal frame YOLO al corrispondente frame MoCap
# Un frame YOLO 'f' viene mappato al frame MoCap 'f + offset' se quest'ultimo esiste.
frame_mapping = {}
for yolo_frame in yolo_frames_int:
    gt_equivalent = yolo_frame + offset
    if gt_equivalent in gt_frames_int:
        # Usa stringhe per le chiavi, per coerenza con il formato JSON
        frame_mapping[str(yolo_frame)] = str(gt_equivalent)

common_yolo_frames = sorted(frame_mapping.keys(), key=int)
print(f"Trovati {len(common_yolo_frames)} frame comuni dopo l'allineamento con offset.")

# --- FINE LOGICA DI ALLINEAMENTO FRAME ---

# Prepara le liste per le statistiche
mpjpe_list = []
per_joint_errors = [[] for _ in range(NUM_JOINTS)]
frame_joint_errors = {} # Per salvare gli errori per il CSV

# Itera su ogni frame comune per eseguire l'allineamento e calcolare l'errore
for yolo_frame_str in common_yolo_frames:
    gt_frame_str = frame_mapping[yolo_frame_str]
    
    yolo_points_for_frame = []
    gt_points_for_frame = []
    valid_joint_indices = []

    # 1. Colleziona tutte le coppie di giunti valide per il frame corrente
    for j in range(NUM_JOINTS):
        yolo_joint = yolo_3d[yolo_frame_str].get(str(j))
        gt_joint   = gt_3d[gt_frame_str].get(str(j))

        # Controlla se entrambi i giunti sono validi
        if (yolo_joint is not None and gt_joint is not None and
            not np.any(np.isnan(yolo_joint)) and not np.any(np.isnan(gt_joint)) and
            not np.any(np.isinf(yolo_joint)) and not np.any(np.isinf(gt_joint))):
            
            yolo_points_for_frame.append(yolo_joint)
            gt_points_for_frame.append(gt_joint)
            valid_joint_indices.append(j)

    # 2. Esegui l'allineamento se ci sono abbastanza punti (almeno 3 per stabilità)
    if len(yolo_points_for_frame) < 3:
        print(f"Saltando frame {yolo_frame_str}: trovate solo {len(yolo_points_for_frame)} coppie di giunti valide.")
        continue

    yolo_np = np.array(yolo_points_for_frame)
    gt_np = np.array(gt_points_for_frame)

    # 3. Allinea i punti YOLO a quelli GT usando l'analisi di Procruste
    gt_aligned, yolo_aligned, disparity = procrustes(gt_np, yolo_np)

    # 4. Calcola gli errori sui punti allineati
    errors = np.linalg.norm(gt_aligned - yolo_aligned, axis=1)
    
    # 5. Salva gli errori per le statistiche e l'esportazione CSV
    frame_joint_errors[yolo_frame_str] = {}
    mpjpe_list.extend(errors)
    for i, err in enumerate(errors):
        joint_idx = valid_joint_indices[i]
        per_joint_errors[joint_idx].append(err)
        frame_joint_errors[yolo_frame_str][joint_idx] = err

# Se sono stati raccolti errori, stampa le statistiche riassuntive
if mpjpe_list:
    print(f"\nValutazione su {len(frame_joint_errors)} frame e {len(mpjpe_list)} coppie di giunti (post-allineamento):")
    print(f"MPJPE Medio:   {np.mean(mpjpe_list):.2f} mm")
    print(f"MPJPE Mediano: {np.median(mpjpe_list):.2f} mm")
    print(f"Min / Max:     {np.min(mpjpe_list):.2f} / {np.max(mpjpe_list):.2f} mm")
    print("\nErrore medio per giunto:")
    for j, errs in enumerate(per_joint_errors):
        if errs:
            print(f"  Giunto {j}: {np.mean(errs):.2f} mm ({len(errs)} misurazioni)")
        else:
            print(f"  Giunto {j}: N/A")
else:
    print("Nessuna coppia di giunti confrontabile trovata dopo l'allineamento!")

# Scrivi i risultati dettagliati su CSV
with open('mpjpe_3d_resultsV2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['frame', 'joint', 'mpjpe_mm_post_alignment'])
    for frame, errors_dict in sorted(frame_joint_errors.items(), key=lambda item: int(item[0])):
        for joint_idx, error in errors_dict.items():
            writer.writerow([frame, joint_idx, error])
    
    writer.writerow([])
    writer.writerow(['STATISTIC', 'VALUE_mm'])
    if mpjpe_list:
        writer.writerow(['mean_mpjpe', f"{np.mean(mpjpe_list):.2f}"])
        writer.writerow(['median_mpjpe', f"{np.median(mpjpe_list):.2f}"])
        writer.writerow(['min_mpjpe', f"{np.min(mpjpe_list):.2f}"])
        writer.writerow(['max_mpjpe', f"{np.max(mpjpe_list):.2f}"])
        writer.writerow(['total_joint_pairs', len(mpjpe_list)])
        writer.writerow(['total_frames', len(frame_joint_errors)])
        writer.writerow([])
        writer.writerow(['JOINT_INDEX', 'MEAN_ERROR_mm', 'NUM_MEASUREMENTS'])
        for j, errs in enumerate(per_joint_errors):
            if errs:
                writer.writerow([j, f"{np.mean(errs):.2f}", len(errs)])
            else:
                writer.writerow([j, 'N/A', 0])
    else:
        writer.writerow(['mean_mpjpe', 'N/A'])

print("\n✅ Risultati con allineamento Procrustes e offset temporale salvati su mpjpe_3d_resultsV2.csv")