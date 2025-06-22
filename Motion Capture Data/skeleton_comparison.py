#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ——— Configurazione ———
MOCAP_JSON = 'mocap_clip_81_85s.json'
TRIANG_JSON = "../Triangulation and reprojection and 3D/triangulated_positions_v3.json"

VIDEO_FPS = 12
MOCAP_FPS = 100
FRAME_RATIO = MOCAP_FPS / VIDEO_FPS  # ≈ 8.33

# Connessioni scheletro
SKELETON_CONNECTIONS = [
    (0, 1), (0, 5),  # Hips to RHip, LHip
    (1, 2), (2, 3), (3, 4),  # Right leg
    (5, 6), (6, 7), (7, 8),  # Left leg
    (0, 9), (9, 10), (10, 11),  # Spine, Neck, Head
    (10, 12), (12, 13), (13, 14),  # Right arm
    (10, 15), (15, 16), (16, 17),  # Left arm
]

def find_optimal_frame_mapping():
    """Trova il mapping ottimale frame per frame usando correlazione delle pose"""
    
    # Carica dati
    tri_raw = json.load(open(TRIANG_JSON))
    tri_keys = sorted(int(k) for k in tri_raw)
    start_tri = tri_keys[0]
    tri = {str(k - start_tri): tri_raw[str(k)] for k in tri_keys}
    
    mocap_raw = json.load(open(MOCAP_JSON))
    mocap_keys = sorted(int(k) for k in mocap_raw)
    start_m = mocap_keys[0]
    mocap = {str(k - start_m): mocap_raw[str(k)] for k in mocap_keys}
    
    print("🔍 Finding optimal frame mapping using pose correlation...")
    
    optimal_mapping = {}
    
    for t_str, tv in tri.items():
        video_frame = int(t_str)
        
        # Estrai pose triangolata
        tri_pose = []
        for j_idx in range(18):
            j_str = str(j_idx)
            pos = tv.get(j_str)
            if pos and len(pos) == 3:
                tri_pose.append(pos)
        
        if len(tri_pose) < 10:  # Skip se troppo pochi joint
            continue
            
        tri_pose = np.array(tri_pose)
        tri_centroid = np.mean(tri_pose, axis=0)
        
        # Cerca il frame MoCap più simile in un range ragionevole
        base_mocap_frame = int(video_frame * FRAME_RATIO)
        search_range = range(max(0, base_mocap_frame - 20), 
                           min(len(mocap), base_mocap_frame + 21))
        
        best_similarity = float('inf')
        best_mocap_frame = base_mocap_frame
        
        for mocap_frame in search_range:
            m_key = str(mocap_frame)
            mv = mocap.get(m_key, {})
            
            if not mv:
                continue
                
            # Estrai pose MoCap
            mocap_pose = []
            for j_idx in range(18):
                j_str = str(j_idx)
                pos = mv.get(j_str)
                if pos and len(pos) == 3:
                    mocap_pose.append(pos)
            
            if len(mocap_pose) < 10:
                continue
                
            mocap_pose = np.array(mocap_pose)
            mocap_centroid = np.mean(mocap_pose, axis=0)
            
            # Calcola similarità usando distanza centroidi + forma pose
            centroid_dist = np.linalg.norm(tri_centroid - mocap_centroid)
            
            # Aggiungi similarità della forma (distanze relative tra joint)
            if len(tri_pose) == len(mocap_pose):
                tri_relative = tri_pose - tri_centroid
                mocap_relative = mocap_pose - mocap_centroid
                shape_dist = np.mean(np.linalg.norm(tri_relative - mocap_relative, axis=1))
                total_similarity = centroid_dist + 0.1 * shape_dist  # Peso maggiore al centroide
            else:
                total_similarity = centroid_dist
            
            if total_similarity < best_similarity:
                best_similarity = total_similarity
                best_mocap_frame = mocap_frame
        
        optimal_mapping[video_frame] = {
            'mocap_frame': best_mocap_frame,
            'similarity': best_similarity,
            'expected_frame': base_mocap_frame,
            'offset': best_mocap_frame - base_mocap_frame
        }
        
        print(f"Video {video_frame:2d} → MoCap {best_mocap_frame:3d} (expected {base_mocap_frame:3d}, offset {best_mocap_frame - base_mocap_frame:+3d}, sim {best_similarity:.1f})")
    
    return optimal_mapping

def load_and_align_data_optimized():
    """Carica e allinea i dati usando il mapping ottimale"""
    
    optimal_mapping = find_optimal_frame_mapping()
    
    # Carica dati
    tri_raw = json.load(open(TRIANG_JSON))
    tri_keys = sorted(int(k) for k in tri_raw)
    start_tri = tri_keys[0]
    tri = {str(k - start_tri): tri_raw[str(k)] for k in tri_keys}
    
    mocap_raw = json.load(open(MOCAP_JSON))
    mocap_keys = sorted(int(k) for k in mocap_raw)
    start_m = mocap_keys[0]
    mocap = {str(k - start_m): mocap_raw[str(k)] for k in mocap_keys}
    
    frames_data = []
    
    for t_str, tv in tri.items():
        video_frame = int(t_str)
        
        if video_frame not in optimal_mapping:
            continue
            
        mocap_frame = optimal_mapping[video_frame]['mocap_frame']
        m_key = str(mocap_frame)
        mv = mocap.get(m_key, {})
        
        if mv:
            tri_joints = []
            mocap_joints = []
            
            for j_idx in range(18):
                j_str = str(j_idx)
                tri_pos = tv.get(j_str)
                mocap_pos = mv.get(j_str)
                
                if tri_pos and mocap_pos and len(tri_pos) == 3:
                    tri_joints.append(tri_pos)
                    mocap_joints.append(mocap_pos)
                else:
                    tri_joints.append([0, 0, 0])
                    mocap_joints.append([0, 0, 0])
            
            frames_data.append({
                'frame': video_frame,
                'mocap_frame': mocap_frame,
                'triangulation': np.array(tri_joints),
                'mocap': np.array(mocap_joints),
                'similarity': optimal_mapping[video_frame]['similarity'],
                'offset': optimal_mapping[video_frame]['offset']
            })
    
    return frames_data

def plot_skeleton(ax, joints, connections, color, label):
    """Disegna uno scheletro 3D"""
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
              c=color, s=50, alpha=0.8, label=label)
    
    for start, end in connections:
        if start < len(joints) and end < len(joints):
            if not (np.allclose(joints[start], [0, 0, 0]) or np.allclose(joints[end], [0, 0, 0])):
                line = np.array([joints[start], joints[end]])
                ax.plot(line[:, 0], line[:, 1], line[:, 2], 
                       color=color, linewidth=2, alpha=0.7)

def create_optimized_comparison():
    """Crea confronto con mapping ottimizzato"""
    frames_data = load_and_align_data_optimized()
    
    if not frames_data:
        print("❌ No frame data loaded!")
        return
    
    # Mostra statistiche del mapping
    offsets = [d['offset'] for d in frames_data]
    similarities = [d['similarity'] for d in frames_data]
    
    print(f"\n📊 MAPPING STATISTICS:")
    print(f"Average offset: {np.mean(offsets):.1f} frames")
    print(f"Offset std dev: {np.std(offsets):.1f} frames")
    print(f"Average similarity: {np.mean(similarities):.1f}")
    print(f"Best similarity: {np.min(similarities):.1f}")
    print(f"Worst similarity: {np.max(similarities):.1f}")
    
    # Overlay comparison con i migliori match
    fig = plt.figure(figsize=(20, 12))
    
    # Seleziona i frame con migliore similarità
    sorted_frames = sorted(frames_data, key=lambda x: x['similarity'])
    selected_frames = sorted_frames[:6]  # I 6 migliori
    
    for i, data in enumerate(selected_frames):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        
        plot_skeleton(ax, data['triangulation'], SKELETON_CONNECTIONS, 
                     'red', 'Triangulation')
        plot_skeleton(ax, data['mocap'], SKELETON_CONNECTIONS, 
                     'blue', 'MoCap')
        
        ax.set_title(f"Video {data['frame']} ↔ MoCap {data['mocap_frame']}\n"
                    f"Similarity: {data['similarity']:.1f}, Offset: {data['offset']:+d}")
        ax.legend()
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('skeleton_optimized_perfect.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return frames_data

def create_animation():
    """Crea animazione side-by-side per confrontare i movimenti"""
    frames_data = load_and_align_data_optimized()
    
    if not frames_data:
        print("❌ No frame data loaded!")
        return
    
    print(f"🎬 Creating animation with {len(frames_data)} frames...")
    
    # Setup figura con subplot side-by-side
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Inizializza plot vuoti
    tri_points = ax1.scatter([], [], [], c='red', s=50, alpha=0.8, label='Triangulation')
    tri_lines = []
    
    mocap_points = ax2.scatter([], [], [], c='blue', s=50, alpha=0.8, label='MoCap')
    mocap_lines = []
    
    # Setup assi
    def setup_axis(ax, title, data_list):
        # Calcola bounds da tutti i frame
        all_points = np.vstack([d[data_list] for d in frames_data])
        valid_points = all_points[~np.all(all_points == [0, 0, 0], axis=1)]
        
        if len(valid_points) > 0:
            margin = 500
            ax.set_xlim(valid_points[:, 0].min() - margin, valid_points[:, 0].max() + margin)
            ax.set_ylim(valid_points[:, 1].min() - margin, valid_points[:, 1].max() + margin)
            ax.set_zlim(valid_points[:, 2].min() - margin, valid_points[:, 2].max() + margin)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
    
    setup_axis(ax1, 'Triangulation', 'triangulation')
    setup_axis(ax2, 'MoCap', 'mocap')
    
    def animate(frame_idx):
        if frame_idx >= len(frames_data):
            return
        
        data = frames_data[frame_idx]
        
        # Clear previous lines
        for line in tri_lines:
            line.remove()
        for line in mocap_lines:
            line.remove()
        tri_lines.clear()
        mocap_lines.clear()
        
        # Update triangulation
        tri_joints = data['triangulation']
        valid_tri = tri_joints[~np.all(tri_joints == [0, 0, 0], axis=1)]
        
        if len(valid_tri) > 0:
            tri_points._offsets3d = (valid_tri[:, 0], valid_tri[:, 1], valid_tri[:, 2])
            
            # Draw connections
            for start, end in SKELETON_CONNECTIONS:
                if start < len(tri_joints) and end < len(tri_joints):
                    if not (np.allclose(tri_joints[start], [0, 0, 0]) or 
                           np.allclose(tri_joints[end], [0, 0, 0])):
                        line = ax1.plot([tri_joints[start][0], tri_joints[end][0]],
                                       [tri_joints[start][1], tri_joints[end][1]],
                                       [tri_joints[start][2], tri_joints[end][2]],
                                       'r-', linewidth=2, alpha=0.7)[0]
                        tri_lines.append(line)
        
        # Update MoCap
        mocap_joints = data['mocap']
        valid_mocap = mocap_joints[~np.all(mocap_joints == [0, 0, 0], axis=1)]
        
        if len(valid_mocap) > 0:
            mocap_points._offsets3d = (valid_mocap[:, 0], valid_mocap[:, 1], valid_mocap[:, 2])
            
            # Draw connections
            for start, end in SKELETON_CONNECTIONS:
                if start < len(mocap_joints) and end < len(mocap_joints):
                    if not (np.allclose(mocap_joints[start], [0, 0, 0]) or 
                           np.allclose(mocap_joints[end], [0, 0, 0])):
                        line = ax2.plot([mocap_joints[start][0], mocap_joints[end][0]],
                                       [mocap_joints[start][1], mocap_joints[end][1]],
                                       [mocap_joints[start][2], mocap_joints[end][2]],
                                       'b-', linewidth=2, alpha=0.7)[0]
                        mocap_lines.append(line)
        
        # Update titles with frame info
        ax1.set_title(f'Triangulation - Video Frame {data["frame"]}')
        ax2.set_title(f'MoCap - Frame {data["mocap_frame"]} (offset {data["offset"]:+d})')
        
        return [tri_points, mocap_points] + tri_lines + mocap_lines
    
    # Crea animazione
    anim = animation.FuncAnimation(fig, animate, frames=len(frames_data), 
                                 interval=500, blit=False, repeat=True)
    
    print("💾 Saving animation as 'skeleton_animation.gif'...")
    anim.save('skeleton_animation.gif', writer='pillow', fps=2, dpi=150)
    
    print("🎬 Showing animation...")
    plt.show()
    
    return anim

def create_overlay_animation():
    """Crea animazione overlay con entrambi gli scheletri nello stesso plot"""
    frames_data = load_and_align_data_optimized()
    
    if not frames_data:
        print("❌ No frame data loaded!")
        return
    
    print(f"🎬 Creating overlay animation with {len(frames_data)} frames...")
    
    # Setup figura
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Inizializza plot vuoti
    tri_points = ax.scatter([], [], [], c='red', s=50, alpha=0.8, label='Triangulation')
    mocap_points = ax.scatter([], [], [], c='blue', s=50, alpha=0.8, label='MoCap')
    tri_lines = []
    mocap_lines = []
    
    # Setup assi con bounds globali
    all_tri = np.vstack([d['triangulation'] for d in frames_data])
    all_mocap = np.vstack([d['mocap'] for d in frames_data])
    all_points = np.vstack([all_tri, all_mocap])
    valid_points = all_points[~np.all(all_points == [0, 0, 0], axis=1)]
    
    if len(valid_points) > 0:
        margin = 1000
        ax.set_xlim(valid_points[:, 0].min() - margin, valid_points[:, 0].max() + margin)
        ax.set_ylim(valid_points[:, 1].min() - margin, valid_points[:, 1].max() + margin)
        ax.set_zlim(valid_points[:, 2].min() - margin, valid_points[:, 2].max() + margin)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    def animate_overlay(frame_idx):
        if frame_idx >= len(frames_data):
            return
        
        data = frames_data[frame_idx]
        
        # Clear previous lines
        for line in tri_lines + mocap_lines:
            line.remove()
        tri_lines.clear()
        mocap_lines.clear()
        
        # Update points and connections
        tri_joints = data['triangulation']
        mocap_joints = data['mocap']
        
        # Triangulation
        valid_tri = tri_joints[~np.all(tri_joints == [0, 0, 0], axis=1)]
        if len(valid_tri) > 0:
            tri_points._offsets3d = (valid_tri[:, 0], valid_tri[:, 1], valid_tri[:, 2])
            
            for start, end in SKELETON_CONNECTIONS:
                if start < len(tri_joints) and end < len(tri_joints):
                    if not (np.allclose(tri_joints[start], [0, 0, 0]) or 
                           np.allclose(tri_joints[end], [0, 0, 0])):
                        line = ax.plot([tri_joints[start][0], tri_joints[end][0]],
                                      [tri_joints[start][1], tri_joints[end][1]],
                                      [tri_joints[start][2], tri_joints[end][2]],
                                      'r-', linewidth=2, alpha=0.7)[0]
                        tri_lines.append(line)
        
        # MoCap
        valid_mocap = mocap_joints[~np.all(mocap_joints == [0, 0, 0], axis=1)]
        if len(valid_mocap) > 0:
            mocap_points._offsets3d = (valid_mocap[:, 0], valid_mocap[:, 1], valid_mocap[:, 2])
            
            for start, end in SKELETON_CONNECTIONS:
                if start < len(mocap_joints) and end < len(mocap_joints):
                    if not (np.allclose(mocap_joints[start], [0, 0, 0]) or 
                           np.allclose(mocap_joints[end], [0, 0, 0])):
                        line = ax.plot([mocap_joints[start][0], mocap_joints[end][0]],
                                      [mocap_joints[start][1], mocap_joints[end][1]],
                                      [mocap_joints[start][2], mocap_joints[end][2]],
                                      'b-', linewidth=2, alpha=0.7)[0]
                        mocap_lines.append(line)
        
        # Update title
        ax.set_title(f'Overlay - Video {data["frame"]} ↔ MoCap {data["mocap_frame"]} (sim: {data["similarity"]:.1f})')
        
        return [tri_points, mocap_points] + tri_lines + mocap_lines
    
    # Crea animazione
    anim = animation.FuncAnimation(fig, animate_overlay, frames=len(frames_data), 
                                 interval=500, blit=False, repeat=True)
    
    print("💾 Saving overlay animation as 'skeleton_overlay_animation.gif'...")
    anim.save('skeleton_overlay_animation.gif', writer='pillow', fps=2, dpi=150)
    
    print("🎬 Showing overlay animation...")
    plt.show()
    
    return anim

if __name__ == "__main__":
    print("🎯 Creating PERFECT skeleton synchronization with ANIMATION...")
    
    # Statico
    print("\n📊 Creating static comparison...")
    frames_data = create_optimized_comparison()
    
    # Animazioni
    print("\n🎬 Creating side-by-side animation...")
    anim1 = create_animation()
    
    print("\n🎯 Creating overlay animation...")
    anim2 = create_overlay_animation()
    
    print(f"\n✅ Complete! Check the files:")
    print("📸 skeleton_optimized_perfect.png - Static comparison")
    print("🎬 skeleton_animation.gif - Side-by-side animation")
    print("🎯 skeleton_overlay_animation.gif - Overlay animation")