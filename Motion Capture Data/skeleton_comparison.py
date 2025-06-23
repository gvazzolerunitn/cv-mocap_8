"""
3D Skeleton Comparison Tool

This script compares 3D human poses from two different sources:
1. Motion Capture (MoCap) data - Ground truth reference from professional system
2. Triangulated poses - Reconstructed from multi-camera computer vision pipeline

The comparison involves:
- Temporal synchronization using pose correlation analysis
- 3D skeleton visualization with anatomical connections
- Static overlay plots and animated sequences
- Quality assessment through pose similarity metrics

The goal is to validate the accuracy of the computer vision triangulation
by comparing it against professional motion capture data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ——— Configuration Parameters ———
MOCAP_JSON = 'mocap_clip_79_83s.json'      # Ground truth MoCap data
TRIANG_JSON = "../Triangulation and reprojection and 3D/triangulated_positions_v2.json"  # Our triangulated results

# Frame rate synchronization parameters
VIDEO_FPS = 12          # Frame rate of input video sequence
MOCAP_FPS = 100         # Frame rate of motion capture system
FRAME_RATIO = MOCAP_FPS / VIDEO_FPS  # ≈ 8.33 - Expected frame offset between systems

# Human skeleton structure definition (18-joint model)
# Each tuple (a, b) represents a bone connecting joint a to joint b
# This follows the standard human pose estimation joint ordering
SKELETON_CONNECTIONS = [
    # Pelvis connections - root joint connects to both hips
    (0, 1), (0, 5),                    # Hips to RHip, LHip
    
    # Right leg kinematic chain
    (1, 2), (2, 3), (3, 4),            # RHip -> RKnee -> RAnkle -> RFoot
    
    # Left leg kinematic chain  
    (5, 6), (6, 7), (7, 8),            # LHip -> LKnee -> LAnkle -> LFoot
    
    # Torso and head chain
    (0, 9), (9, 10), (10, 11),         # Hips -> Spine -> Neck -> Head
    
    # Right arm kinematic chain
    (10, 12), (12, 13), (13, 14),      # Neck -> RShoulder -> RElbow -> RHand
    
    # Left arm kinematic chain
    (10, 15), (15, 16), (16, 17),      # Neck -> LShoulder -> LElbow -> LHand
]


def find_optimal_frame_mapping():
    """
    Find optimal temporal alignment between video and MoCap sequences.
    
    Since the video and MoCap were recorded independently, there's no guarantee
    of perfect temporal synchronization. This function uses pose correlation
    analysis to find the best frame-to-frame mapping.
    
    Algorithm:
    1. For each video frame, extract the 3D triangulated pose centroid
    2. Search in a reasonable range around the expected MoCap frame
    3. Compare pose similarity using centroid distance + shape matching
    4. Select the MoCap frame with highest similarity
    
    Returns:
        dict: Mapping from video_frame -> {mocap_frame, similarity, offset}
    """
    
    # Load triangulated data and normalize frame indices to start from 0
    triangulated_raw = json.load(open(TRIANG_JSON))
    tri_frame_keys = sorted(int(k) for k in triangulated_raw)
    start_tri_frame = tri_frame_keys[0]
    triangulated_data = {str(k - start_tri_frame): triangulated_raw[str(k)] for k in tri_frame_keys}
    
    # Load MoCap data and normalize frame indices to start from 0
    mocap_raw = json.load(open(MOCAP_JSON))
    mocap_frame_keys = sorted(int(k) for k in mocap_raw)
    start_mocap_frame = mocap_frame_keys[0]
    mocap_data = {str(k - start_mocap_frame): mocap_raw[str(k)] for k in mocap_frame_keys}
    
    print("🔍 Finding optimal frame mapping using pose correlation analysis...")
    
    optimal_mapping = {}
    
    # Process each triangulated frame to find its best MoCap match
    for tri_frame_str, tri_frame_data in triangulated_data.items():
        video_frame = int(tri_frame_str)
        
        # Extract 3D pose from triangulated data
        triangulated_pose = []
        for joint_idx in range(18):  # 18-joint human pose model
            joint_str = str(joint_idx)
            joint_position = tri_frame_data.get(joint_str)
            if joint_position and len(joint_position) == 3:
                triangulated_pose.append(joint_position)
        
        # Skip frames with insufficient joint detections (poor triangulation quality)
        if len(triangulated_pose) < 10:
            continue
            
        # Calculate pose centroid for similarity comparison
        triangulated_pose = np.array(triangulated_pose)
        tri_pose_centroid = np.mean(triangulated_pose, axis=0)
        
        # Determine search range for MoCap frames
        # Start with expected frame based on frame rate ratio, then search nearby
        expected_mocap_frame = int(video_frame * FRAME_RATIO)
        search_range = range(max(0, expected_mocap_frame - 20), 
                           min(len(mocap_data), expected_mocap_frame + 21))
        
        # Find MoCap frame with highest pose similarity
        best_similarity_score = float('inf')  # Lower is better
        best_matching_frame = expected_mocap_frame
        
        for candidate_mocap_frame in search_range:
            mocap_key = str(candidate_mocap_frame)
            mocap_frame_data = mocap_data.get(mocap_key, {})
            
            if not mocap_frame_data:
                continue
                
            # Extract 3D pose from MoCap data
            mocap_pose = []
            for joint_idx in range(18):
                joint_str = str(joint_idx)
                joint_position = mocap_frame_data.get(joint_str)
                if joint_position and len(joint_position) == 3:
                    mocap_pose.append(joint_position)
            
            # Skip frames with insufficient joint data
            if len(mocap_pose) < 10:
                continue
                
            # Calculate pose centroid and similarity metrics
            mocap_pose = np.array(mocap_pose)
            mocap_pose_centroid = np.mean(mocap_pose, axis=0)
            
            # Primary similarity metric: centroid distance
            # This captures overall pose position in 3D space
            centroid_distance = np.linalg.norm(tri_pose_centroid - mocap_pose_centroid)
            
            # Secondary similarity metric: pose shape matching
            # Compare relative joint positions (pose-invariant to translation)
            if len(triangulated_pose) == len(mocap_pose):
                tri_relative_positions = triangulated_pose - tri_pose_centroid
                mocap_relative_positions = mocap_pose - mocap_pose_centroid
                shape_distance = np.mean(np.linalg.norm(tri_relative_positions - mocap_relative_positions, axis=1))
                
                # Combine metrics with higher weight on centroid (global position)
                total_similarity = centroid_distance + 0.1 * shape_distance
            else:
                total_similarity = centroid_distance
            
            # Update best match if this frame is more similar
            if total_similarity < best_similarity_score:
                best_similarity_score = total_similarity
                best_matching_frame = candidate_mocap_frame
        
        # Store optimal mapping result
        optimal_mapping[video_frame] = {
            'mocap_frame': best_matching_frame,
            'similarity': best_similarity_score,
            'expected_frame': expected_mocap_frame,
            'offset': best_matching_frame - expected_mocap_frame
        }
        
        # Log mapping result for analysis
        print(f"Video {video_frame:2d} → MoCap {best_matching_frame:3d} "
              f"(expected {expected_mocap_frame:3d}, offset {best_matching_frame - expected_mocap_frame:+3d}, "
              f"similarity {best_similarity_score:.1f})")
    
    return optimal_mapping


def load_and_align_data_with_optimal_mapping():
    """
    Load and temporally align both datasets using the optimal frame mapping.
    
    This function applies the frame mapping computed by find_optimal_frame_mapping()
    to create synchronized pose pairs for comparison and visualization.
    
    Returns:
        list: List of synchronized frame data dictionaries containing:
              - frame: video frame number
              - mocap_frame: corresponding MoCap frame
              - triangulation: 18x3 array of triangulated joint positions
              - mocap: 18x3 array of MoCap joint positions
              - similarity: pose similarity score
              - offset: temporal offset from expected alignment
    """
    
    # Get optimal frame alignment
    optimal_mapping = find_optimal_frame_mapping()
    
    # Load both datasets with normalized frame indices
    triangulated_raw = json.load(open(TRIANG_JSON))
    tri_frame_keys = sorted(int(k) for k in triangulated_raw)
    start_tri_frame = tri_frame_keys[0]
    triangulated_data = {str(k - start_tri_frame): triangulated_raw[str(k)] for k in tri_frame_keys}
    
    mocap_raw = json.load(open(MOCAP_JSON))
    mocap_frame_keys = sorted(int(k) for k in mocap_raw)
    start_mocap_frame = mocap_frame_keys[0]
    mocap_data = {str(k - start_mocap_frame): mocap_raw[str(k)] for k in mocap_frame_keys}
    
    synchronized_frames = []
    
    # Create synchronized frame pairs using optimal mapping
    for tri_frame_str, tri_frame_data in triangulated_data.items():
        video_frame = int(tri_frame_str)
        
        # Skip frames without optimal mapping
        if video_frame not in optimal_mapping:
            continue
            
        # Get corresponding MoCap frame
        optimal_mocap_frame = optimal_mapping[video_frame]['mocap_frame']
        mocap_key = str(optimal_mocap_frame)
        mocap_frame_data = mocap_data.get(mocap_key, {})
        
        if mocap_frame_data:
            # Extract joint positions for both systems
            triangulated_joints = []
            mocap_joints = []
            
            # Process all 18 joints in the human pose model
            for joint_idx in range(18):
                joint_str = str(joint_idx)
                tri_position = tri_frame_data.get(joint_str)
                mocap_position = mocap_frame_data.get(joint_str)
                
                # Use actual positions if available, otherwise fill with zeros
                if tri_position and mocap_position and len(tri_position) == 3:
                    triangulated_joints.append(tri_position)
                    mocap_joints.append(mocap_position)
                else:
                    # Missing joint data - use origin as placeholder
                    triangulated_joints.append([0, 0, 0])
                    mocap_joints.append([0, 0, 0])
            
            # Store synchronized frame data
            synchronized_frames.append({
                'frame': video_frame,
                'mocap_frame': optimal_mocap_frame,
                'triangulation': np.array(triangulated_joints),
                'mocap': np.array(mocap_joints),
                'similarity': optimal_mapping[video_frame]['similarity'],
                'offset': optimal_mapping[video_frame]['offset']
            })
    
    return synchronized_frames


def draw_3d_skeleton(ax, joint_positions, skeleton_connections, color, label):
    """
    Render a 3D human skeleton on a matplotlib 3D axis.
    
    Draws both joint positions as scatter points and bone connections as lines
    to create a complete skeleton visualization.
    
    Args:
        ax: Matplotlib 3D axis object
        joint_positions: Nx3 array of 3D joint coordinates
        skeleton_connections: List of (joint_a, joint_b) bone connections
        color: Color for rendering joints and bones
        label: Legend label for this skeleton
    """
    # Draw joints as scatter points with distinctive styling
    ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], 
              c=color, s=50, alpha=0.8, label=label)
    
    # Draw bones as lines connecting joints according to skeleton structure
    for start_joint, end_joint in skeleton_connections:
        if start_joint < len(joint_positions) and end_joint < len(joint_positions):
            # Skip drawing bones connected to missing joints (at origin)
            if not (np.allclose(joint_positions[start_joint], [0, 0, 0]) or 
                   np.allclose(joint_positions[end_joint], [0, 0, 0])):
                
                # Create line segment between connected joints
                bone_line = np.array([joint_positions[start_joint], joint_positions[end_joint]])
                ax.plot(bone_line[:, 0], bone_line[:, 1], bone_line[:, 2], 
                       color=color, linewidth=2, alpha=0.7)


def create_static_comparison_plot():
    """
    Create static visualization comparing the best-matched pose pairs.
    
    Shows multiple pose comparisons in a grid layout, selecting frames
    with the highest similarity scores for optimal visual comparison.
    
    Returns:
        list: Synchronized frame data for further analysis
    """
    synchronized_frames = load_and_align_data_with_optimal_mapping()
    
    if not synchronized_frames:
        print("❌ No synchronized frame data available!")
        return
    
    # Analyze mapping quality statistics
    temporal_offsets = [frame_data['offset'] for frame_data in synchronized_frames]
    similarity_scores = [frame_data['similarity'] for frame_data in synchronized_frames]
    
    print(f"\n📊 TEMPORAL SYNCHRONIZATION ANALYSIS:")
    print(f"Average temporal offset: {np.mean(temporal_offsets):.1f} frames")
    print(f"Temporal offset std deviation: {np.std(temporal_offsets):.1f} frames")
    print(f"Average pose similarity: {np.mean(similarity_scores):.1f}")
    print(f"Best similarity score: {np.min(similarity_scores):.1f}")
    print(f"Worst similarity score: {np.max(similarity_scores):.1f}")
    
    # Create comparison plot with best-matched frames
    fig = plt.figure(figsize=(20, 12))
    
    # Select frames with highest similarity (lowest similarity scores)
    best_matched_frames = sorted(synchronized_frames, key=lambda x: x['similarity'])
    selected_frames = best_matched_frames[:6]  # Show top 6 matches
    
    # Create subplot grid for visual comparison
    for i, frame_data in enumerate(selected_frames):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        
        # Draw both skeletons with different colors for comparison
        draw_3d_skeleton(ax, frame_data['triangulation'], SKELETON_CONNECTIONS, 
                        'red', 'Computer Vision Triangulation')
        draw_3d_skeleton(ax, frame_data['mocap'], SKELETON_CONNECTIONS, 
                        'blue', 'Motion Capture Ground Truth')
        
        # Set informative title with synchronization details
        ax.set_title(f"Video Frame {frame_data['frame']} ↔ MoCap Frame {frame_data['mocap_frame']}\n"
                    f"Similarity: {frame_data['similarity']:.1f}, "
                    f"Temporal Offset: {frame_data['offset']:+d} frames")
        
        # Configure 3D plot appearance
        ax.legend()
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
    
    plt.tight_layout()
    plt.savefig('skeleton_comparison_best_matches.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return synchronized_frames


def create_side_by_side_animation():
    """
    Create side-by-side animated comparison of skeleton sequences.
    
    Shows triangulated and MoCap skeletons in separate subplots,
    synchronized using the optimal frame mapping for temporal alignment.
    
    Returns:
        matplotlib.animation.FuncAnimation: Animation object
    """
    synchronized_frames = load_and_align_data_with_optimal_mapping()
    
    if not synchronized_frames:
        print("❌ No synchronized frame data available!")
        return
    
    print(f"🎬 Creating side-by-side animation with {len(synchronized_frames)} synchronized frames...")
    
    # Setup figure with side-by-side 3D subplots
    fig = plt.figure(figsize=(16, 8))
    triangulation_ax = fig.add_subplot(121, projection='3d')
    mocap_ax = fig.add_subplot(122, projection='3d')
    
    # Initialize empty scatter plots for animation
    triangulation_points = triangulation_ax.scatter([], [], [], c='red', s=50, alpha=0.8, 
                                                   label='Computer Vision')
    triangulation_lines = []
    
    mocap_points = mocap_ax.scatter([], [], [], c='blue', s=50, alpha=0.8, 
                                   label='Motion Capture')
    mocap_lines = []
    
    def setup_3d_axis(ax, title, data_key):
        """Configure 3D axis with appropriate bounds and labels."""
        # Calculate global bounds from all frames for consistent scaling
        all_points = np.vstack([frame_data[data_key] for frame_data in synchronized_frames])
        valid_points = all_points[~np.all(all_points == [0, 0, 0], axis=1)]
        
        if len(valid_points) > 0:
            margin = 500  # Add margin around data bounds
            ax.set_xlim(valid_points[:, 0].min() - margin, valid_points[:, 0].max() + margin)
            ax.set_ylim(valid_points[:, 1].min() - margin, valid_points[:, 1].max() + margin)
            ax.set_zlim(valid_points[:, 2].min() - margin, valid_points[:, 2].max() + margin)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        ax.legend()
    
    # Configure both axes with consistent scaling
    setup_3d_axis(triangulation_ax, 'Computer Vision Triangulation', 'triangulation')
    setup_3d_axis(mocap_ax, 'Motion Capture Ground Truth', 'mocap')
    
    def animate_frame(frame_index):
        """Animation update function called for each frame."""
        if frame_index >= len(synchronized_frames):
            return
        
        current_frame_data = synchronized_frames[frame_index]
        
        # Clear previous frame's bone lines
        for line in triangulation_lines:
            line.remove()
        for line in mocap_lines:
            line.remove()
        triangulation_lines.clear()
        mocap_lines.clear()
        
        # Update triangulation visualization
        tri_joints = current_frame_data['triangulation']
        valid_tri_joints = tri_joints[~np.all(tri_joints == [0, 0, 0], axis=1)]
        
        if len(valid_tri_joints) > 0:
            # Update joint positions
            triangulation_points._offsets3d = (valid_tri_joints[:, 0], 
                                             valid_tri_joints[:, 1], 
                                             valid_tri_joints[:, 2])
            
            # Draw skeleton bones
            for start_joint, end_joint in SKELETON_CONNECTIONS:
                if start_joint < len(tri_joints) and end_joint < len(tri_joints):
                    if not (np.allclose(tri_joints[start_joint], [0, 0, 0]) or 
                           np.allclose(tri_joints[end_joint], [0, 0, 0])):
                        line = triangulation_ax.plot([tri_joints[start_joint][0], tri_joints[end_joint][0]],
                                                   [tri_joints[start_joint][1], tri_joints[end_joint][1]],
                                                   [tri_joints[start_joint][2], tri_joints[end_joint][2]],
                                                   'r-', linewidth=2, alpha=0.7)[0]
                        triangulation_lines.append(line)
        
        # Update MoCap visualization
        mocap_joints = current_frame_data['mocap']
        valid_mocap_joints = mocap_joints[~np.all(mocap_joints == [0, 0, 0], axis=1)]
        
        if len(valid_mocap_joints) > 0:
            # Update joint positions
            mocap_points._offsets3d = (valid_mocap_joints[:, 0], 
                                     valid_mocap_joints[:, 1], 
                                     valid_mocap_joints[:, 2])
            
            # Draw skeleton bones
            for start_joint, end_joint in SKELETON_CONNECTIONS:
                if start_joint < len(mocap_joints) and end_joint < len(mocap_joints):
                    if not (np.allclose(mocap_joints[start_joint], [0, 0, 0]) or 
                           np.allclose(mocap_joints[end_joint], [0, 0, 0])):
                        line = mocap_ax.plot([mocap_joints[start_joint][0], mocap_joints[end_joint][0]],
                                           [mocap_joints[start_joint][1], mocap_joints[end_joint][1]],
                                           [mocap_joints[start_joint][2], mocap_joints[end_joint][2]],
                                           'b-', linewidth=2, alpha=0.7)[0]
                        mocap_lines.append(line)
        
        # Update titles with current frame information
        triangulation_ax.set_title(f'Computer Vision Triangulation - Video Frame {current_frame_data["frame"]}')
        mocap_ax.set_title(f'Motion Capture Ground Truth - Frame {current_frame_data["mocap_frame"]} '
                          f'(offset {current_frame_data["offset"]:+d})')
        
        return [triangulation_points, mocap_points] + triangulation_lines + mocap_lines
    
    # Create animation with appropriate timing
    animation_obj = animation.FuncAnimation(fig, animate_frame, frames=len(synchronized_frames), 
                                          interval=500, blit=False, repeat=True)
    
    print("💾 Saving animation as 'skeleton_side_by_side_animation.gif'...")
    animation_obj.save('skeleton_side_by_side_animation.gif', writer='pillow', fps=2, dpi=150)
    
    print("🎬 Displaying animation...")
    plt.show()
    
    return animation_obj


def create_overlay_animation():
    """
    Create overlay animation with both skeletons in the same 3D space.
    
    This visualization allows direct comparison of pose accuracy by showing
    both skeletons simultaneously in the same coordinate system.
    
    Returns:
        matplotlib.animation.FuncAnimation: Animation object
    """
    synchronized_frames = load_and_align_data_with_optimal_mapping()
    
    if not synchronized_frames:
        print("❌ No synchronized frame data available!")
        return
    
    print(f"🎬 Creating overlay animation with {len(synchronized_frames)} synchronized frames...")
    
    # Setup single 3D plot for overlay visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize empty scatter plots for both skeletons
    triangulation_points = ax.scatter([], [], [], c='red', s=50, alpha=0.8, 
                                    label='Computer Vision Triangulation')
    mocap_points = ax.scatter([], [], [], c='blue', s=50, alpha=0.8, 
                            label='Motion Capture Ground Truth')
    triangulation_lines = []
    mocap_lines = []
    
    # Calculate global bounds for consistent visualization scaling
    all_triangulation_data = np.vstack([frame_data['triangulation'] for frame_data in synchronized_frames])
    all_mocap_data = np.vstack([frame_data['mocap'] for frame_data in synchronized_frames])
    all_joint_positions = np.vstack([all_triangulation_data, all_mocap_data])
    valid_positions = all_joint_positions[~np.all(all_joint_positions == [0, 0, 0], axis=1)]
    
    if len(valid_positions) > 0:
        margin = 1000  # Larger margin for overlay visualization
        ax.set_xlim(valid_positions[:, 0].min() - margin, valid_positions[:, 0].max() + margin)
        ax.set_ylim(valid_positions[:, 1].min() - margin, valid_positions[:, 1].max() + margin)
        ax.set_zlim(valid_positions[:, 2].min() - margin, valid_positions[:, 2].max() + margin)
    
    # Configure 3D plot appearance
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend()
    
    def animate_overlay_frame(frame_index):
        """Animation update function for overlay visualization."""
        if frame_index >= len(synchronized_frames):
            return
        
        current_frame_data = synchronized_frames[frame_index]
        
        # Clear previous frame's bone connections
        for line in triangulation_lines + mocap_lines:
            line.remove()
        triangulation_lines.clear()
        mocap_lines.clear()
        
        # Update skeleton data
        tri_joints = current_frame_data['triangulation']
        mocap_joints = current_frame_data['mocap']
        
        # Update triangulation skeleton
        valid_tri_joints = tri_joints[~np.all(tri_joints == [0, 0, 0], axis=1)]
        if len(valid_tri_joints) > 0:
            triangulation_points._offsets3d = (valid_tri_joints[:, 0], 
                                             valid_tri_joints[:, 1], 
                                             valid_tri_joints[:, 2])
            
            # Draw triangulation skeleton bones
            for start_joint, end_joint in SKELETON_CONNECTIONS:
                if start_joint < len(tri_joints) and end_joint < len(tri_joints):
                    if not (np.allclose(tri_joints[start_joint], [0, 0, 0]) or 
                           np.allclose(tri_joints[end_joint], [0, 0, 0])):
                        line = ax.plot([tri_joints[start_joint][0], tri_joints[end_joint][0]],
                                      [tri_joints[start_joint][1], tri_joints[end_joint][1]],
                                      [tri_joints[start_joint][2], tri_joints[end_joint][2]],
                                      'r-', linewidth=2, alpha=0.7)[0]
                        triangulation_lines.append(line)
        
        # Update MoCap skeleton
        valid_mocap_joints = mocap_joints[~np.all(mocap_joints == [0, 0, 0], axis=1)]
        if len(valid_mocap_joints) > 0:
            mocap_points._offsets3d = (valid_mocap_joints[:, 0], 
                                     valid_mocap_joints[:, 1], 
                                     valid_mocap_joints[:, 2])
            
            # Draw MoCap skeleton bones
            for start_joint, end_joint in SKELETON_CONNECTIONS:
                if start_joint < len(mocap_joints) and end_joint < len(mocap_joints):
                    if not (np.allclose(mocap_joints[start_joint], [0, 0, 0]) or 
                           np.allclose(mocap_joints[end_joint], [0, 0, 0])):
                        line = ax.plot([mocap_joints[start_joint][0], mocap_joints[end_joint][0]],
                                      [mocap_joints[start_joint][1], mocap_joints[end_joint][1]],
                                      [mocap_joints[start_joint][2], mocap_joints[end_joint][2]],
                                      'b-', linewidth=2, alpha=0.7)[0]
                        mocap_lines.append(line)
        
        # Update title with synchronization information
        ax.set_title(f'Overlay Comparison - Video Frame {current_frame_data["frame"]} ↔ '
                    f'MoCap Frame {current_frame_data["mocap_frame"]} '
                    f'(similarity: {current_frame_data["similarity"]:.1f})')
        
        return [triangulation_points, mocap_points] + triangulation_lines + mocap_lines
    
    # Create and save overlay animation
    animation_obj = animation.FuncAnimation(fig, animate_overlay_frame, frames=len(synchronized_frames), 
                                          interval=500, blit=False, repeat=True)
    
    print("💾 Saving overlay animation as 'skeleton_overlay_comparison.gif'...")
    animation_obj.save('skeleton_overlay_comparison.gif', writer='pillow', fps=2, dpi=150)
    
    print("🎬 Displaying overlay animation...")
    plt.show()
    
    return animation_obj


if __name__ == "__main__":
    print("🎯 3D SKELETON COMPARISON: Computer Vision vs Motion Capture")
    print("="*70)
    print("This analysis validates computer vision triangulation accuracy")
    print("by comparing against professional motion capture ground truth.")
    print("="*70)
    
    # Create static comparison visualization
    print("\n📊 Creating static pose comparison plots...")
    synchronized_frame_data = create_static_comparison_plot()
    
    # Create animated comparisons
    print("\n🎬 Creating side-by-side animation comparison...")
    side_by_side_animation = create_side_by_side_animation()
    
    print("\n🎯 Creating overlay animation comparison...")
    overlay_animation = create_overlay_animation()
    
    print(f"\n✅ Analysis complete! Generated files:")
    print("📸 skeleton_comparison_best_matches.png - Static comparison of best-matched poses")
    print("🎬 skeleton_side_by_side_animation.gif - Side-by-side animated comparison")
    print("🎯 skeleton_overlay_comparison.gif - Overlay animated comparison")
    print("\nThese visualizations demonstrate the accuracy of the computer vision")
    print("triangulation pipeline compared to professional motion capture data.")