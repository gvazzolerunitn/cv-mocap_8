"""
The script reads triangulated 3D joint positions and renders them as
connected skeletons in 3D space using matplotlib.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3D projection
import matplotlib.animation as animation

# -----------------------------------------------------------------------------
# Usage Examples (for 48 frames dataset)
# -----------------------------------------------------------------------------
""" 
# Basic usage - show both static and animated views
python visualize_3d_poses.py triangulated_positions_v2.json

# Only animated view with all frames
python visualize_3d_poses.py triangulated_positions_v2.json --mode animated

# Static view of first 10 frames
python visualize_3d_poses.py triangulated_positions_v2.json --mode static --frame-range 0 10 --max-frames 10

# View middle portion of sequence (frames 20-35)
python visualize_3d_poses.py triangulated_positions_v2.json --mode animated --frame-range 25 40

# Just statistics without visualization
python visualize_3d_poses.py triangulated_positions_v2.json --stats-only

# Custom animation speed (slower for detailed analysis)
python visualize_3d_poses.py triangulated_positions_v2.json --mode animated --interval 300

# Static overlay without joint labels (cleaner view)
python visualize_3d_poses.py triangulated_positions_v2.json --mode static --no-labels
"""

# -----------------------------------------------------------------------------
# Human Skeleton Definition (18-joint model)
# -----------------------------------------------------------------------------

# Joint names corresponding to the 18-joint human pose model
# Index 0-17 maps to specific body parts in anatomical order
JOINT_NAMES = [
    "Hips",        # 0  - Center of pelvis (root joint)
    "RHip",        # 1  - Right hip
    "RKnee",       # 2  - Right knee
    "RAnkle",      # 3  - Right ankle
    "RFoot",       # 4  - Right foot
    "LHip",        # 5  - Left hip
    "LKnee",       # 6  - Left knee
    "LAnkle",      # 7  - Left ankle
    "LFoot",       # 8  - Left foot
    "Spine",       # 9  - Lower spine
    "Neck",        # 10 - Neck/upper spine
    "Head",        # 11 - Head/skull
    "RShoulder",   # 12 - Right shoulder
    "RElbow",      # 13 - Right elbow
    "RHand",       # 14 - Right hand
    "LShoulder",   # 15 - Left shoulder
    "LElbow",      # 16 - Left elbow
    "LHand"        # 17 - Left hand
]

# Skeleton connections define which joints are connected by bones
# Each tuple (a, b) represents a bone connecting joint a to joint b
SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
    # Right leg chain: Hips -> RHip -> RKnee -> RAnkle -> RFoot
    (0, 1), (1, 2), (2, 3), (3, 4),
    
    # Left leg chain: Hips -> LHip -> LKnee -> LAnkle -> LFoot
    (0, 5), (5, 6), (6, 7), (7, 8),
    
    # Spine chain: Hips -> Spine -> Neck -> Head
    (0, 9), (9, 10), (10, 11),
    
    # Right arm chain: Neck -> RShoulder -> RElbow -> RHand
    (10, 12), (12, 13), (13, 14),
    
    # Left arm chain: Neck -> LShoulder -> LElbow -> LHand
    (10, 15), (15, 16), (16, 17)
]

# -----------------------------------------------------------------------------
# Data Loading and Processing Functions
# -----------------------------------------------------------------------------

def load_3d_poses(path: Path) -> Dict[int, Dict[int, List[float]]]:
    """
    Load 3D pose data from JSON file.
    
    The JSON file contains triangulated 3D joint positions organized as:
    {
        "frame_number": {
            "joint_id": [x, y, z] or [x, y, z, confidence],
            ...
        },
        ...
    }
    
    Args:
        path (Path): Path to the JSON file containing 3D poses
        
    Returns:
        Dict[int, Dict[int, List[float]]]: Dictionary mapping frame numbers to
        joint dictionaries, where each joint maps to [x, y, z] coordinates
    """
    with open(path, "r") as f:
        raw_data = json.load(f)
    
    processed_data: Dict[int, Dict[int, List[float]]] = {}
    
    for frame_str, joints in raw_data.items():
        # Skip metadata entries that might be present
        if frame_str == "summary_3d":
            continue
            
        frame_number = int(frame_str)
        processed_data[frame_number] = {}
        
        for joint_str, position in joints.items():
            # Handle different position formats
            if isinstance(position, list) and len(position) >= 3:
                # Check for confidence value (4th element = -1 means invalid)
                if len(position) == 4 and position[3] == -1:
                    continue  # Skip invalid joints
                
                # Extract only x, y, z coordinates (ignore confidence if present)
                processed_data[frame_number][int(joint_str)] = position[:3]
            else:
                print(f"Warning: Invalid position data for frame {frame_number}, joint {joint_str}: {position}")
    
    # Return sorted by frame number for consistent ordering
    return dict(sorted(processed_data.items()))


def calculate_pose_bounds(poses: Dict[int, Dict[int, List[float]]], margin: float = 100.0) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Calculate the 3D bounding box for all poses.
    
    Determines the minimum and maximum coordinates across all joints in all frames
    to set appropriate axis limits for visualization.
    
    Args:
        poses: Dictionary of poses (frame -> joint -> [x,y,z])
        margin: Additional margin to add around the bounding box
        
    Returns:
        Tuple containing (x_limits, y_limits, z_limits) where each limit
        is a tuple of (min, max) values
    """
    # Collect all 3D points from all frames and joints
    all_points = []
    for frame_joints in poses.values():
        for position in frame_joints.values():
            all_points.append(position)
    
    if not all_points:
        # Return default bounds if no data
        return (-500, 500), (-500, 500), (-500, 500)
    
    # Convert to numpy array for efficient min/max calculation
    points_array = np.asarray(all_points, dtype=float)
    
    # Calculate bounds with margin
    min_coords = points_array.min(axis=0) - margin
    max_coords = points_array.max(axis=0) + margin
    
    return (min_coords[0], max_coords[0]), (min_coords[1], max_coords[1]), (min_coords[2], max_coords[2])

# -----------------------------------------------------------------------------
# 3D Visualization Functions
# -----------------------------------------------------------------------------

def draw_skeleton(ax: Axes3D, joints_3d: np.ndarray, joint_ids: np.ndarray, 
                 *, color: str = "tab:blue", alpha: float = 1.0):
    """
    Draw a 3D skeleton on the given matplotlib 3D axis.
    
    Renders joints as scatter points and bones as lines connecting them
    according to the skeleton structure defined in SKELETON_CONNECTIONS.
    
    Args:
        ax: Matplotlib 3D axis object
        joints_3d: Array of 3D joint positions, shape (n_joints, 3)
        joint_ids: Array of joint indices corresponding to joints_3d
        color: Color for joints and bones
        alpha: Transparency level (0.0 = transparent, 1.0 = opaque)
    """
    # Draw joints as scatter points
    ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], 
              c=color, s=60, alpha=alpha, edgecolors='black', linewidth=0.5)
    
    # Draw bones as lines connecting joints
    for joint_a, joint_b in SKELETON_CONNECTIONS:
        # Check if both joints exist in the current frame
        if joint_a in joint_ids and joint_b in joint_ids:
            # Find positions of the two joints
            idx_a = joint_ids.tolist().index(joint_a)
            idx_b = joint_ids.tolist().index(joint_b)
            
            pos_a = joints_3d[idx_a]
            pos_b = joints_3d[idx_b]
            
            # Draw line connecting the joints
            ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]], [pos_a[2], pos_b[2]], 
                   c=color, linewidth=2, alpha=alpha)

# -----------------------------------------------------------------------------
# Static Visualization (Multiple Frames Overlaid)
# -----------------------------------------------------------------------------

def show_static_poses(poses: Dict[int, Dict[int, List[float]]], 
                     *, frame_range: Tuple[int, int] | None, 
                     max_frames: int, with_labels: bool):
    """
    Display static visualization with multiple frames overlaid.
    
    Shows several frames simultaneously with different colors to visualize
    the motion trajectory and pose variations across time.
    
    Args:
        poses: Dictionary of 3D poses
        frame_range: Optional tuple (start, end) to limit frame range
        max_frames: Maximum number of frames to display
        with_labels: Whether to show joint name labels
    """
    # Filter frames based on range
    if frame_range is None:
        selected_frames = list(poses.keys())
    else:
        selected_frames = [f for f in poses.keys() 
                          if frame_range[0] <= f <= frame_range[1]]
    
    # Limit to maximum number of frames
    selected_frames = selected_frames[:max_frames]
    
    if not selected_frames:
        print("No frames to display in the specified range")
        return
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Set axis limits based on all pose data
    x_lim, y_lim, z_lim = calculate_pose_bounds(poses)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    
    # Set axis labels
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    
    # Generate colors for each frame using a colormap
    colors = plt.cm.plasma(np.linspace(0, 1, len(selected_frames)))
    
    # Draw each frame with a different color
    for color, frame_num in zip(colors, selected_frames):
        frame_joints = poses[frame_num]
        
        # Convert to numpy arrays for easier manipulation
        joint_positions = np.array(list(frame_joints.values()))
        joint_indices = np.array(list(frame_joints.keys()))
        
        # Draw the skeleton for this frame
        draw_skeleton(ax, joint_positions, joint_indices, color=color)
        
        # Add joint labels if requested
        if with_labels:
            for joint_id, position in zip(joint_indices, joint_positions):
                ax.text(position[0], position[1], position[2], 
                       f"{JOINT_NAMES[joint_id]}", fontsize=7)
    
    # Set title showing frame range
    ax.set_title(f"3D Poses - Frames {selected_frames[0]} to {selected_frames[-1]}")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Animated Visualization (Frame-by-Frame)
# -----------------------------------------------------------------------------

def show_animated_poses(poses: Dict[int, Dict[int, List[float]]], 
                       *, frame_range: Tuple[int, int] | None, 
                       interval: int):
    """
    Display animated visualization showing poses frame by frame.
    
    Creates an animation where each frame is displayed sequentially,
    allowing visualization of motion over time.
    
    Args:
        poses: Dictionary of 3D poses
        frame_range: Optional tuple (start, end) to limit frame range
        interval: Time interval between frames in milliseconds
        
    Returns:
        Animation object (needed to keep animation alive)
    """
    # Filter frames based on range
    if frame_range is None:
        animation_frames = list(poses.keys())
    else:
        animation_frames = [f for f in poses.keys() 
                           if frame_range[0] <= f <= frame_range[1]]
    
    if len(animation_frames) < 2:
        print("Need at least 2 frames for animation")
        return None
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    
    # Calculate bounds for consistent axis limits
    x_lim, y_lim, z_lim = calculate_pose_bounds(poses)
    
    def init_animation():
        """Initialize animation with axis settings."""
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        return []
    
    def update_frame(frame_index):
        """Update function called for each animation frame."""
        # Clear previous frame
        ax.cla()
        
        # Reset axis properties
        init_animation()
        
        # Get current frame data
        current_frame = animation_frames[frame_index]
        frame_joints = poses[current_frame]
        
        # Convert to numpy arrays
        joint_positions = np.array(list(frame_joints.values()))
        joint_indices = np.array(list(frame_joints.keys()))
        
        # Draw skeleton for current frame
        draw_skeleton(ax, joint_positions, joint_indices)
        
        # Update title with current frame info
        ax.set_title(f"3D Pose Animation - Frame {current_frame} ({frame_index + 1}/{len(animation_frames)})")
        
        return []
    
    # Create animation object
    animation_obj = animation.FuncAnimation(
        fig, update_frame, 
        frames=len(animation_frames),
        init_func=init_animation,
        interval=interval,
        blit=False,
        repeat=True
    )
    
    plt.tight_layout()
    plt.show()
    
    return animation_obj

# -----------------------------------------------------------------------------
# Statistics and Analysis Functions
# -----------------------------------------------------------------------------

def print_pose_statistics(poses: Dict[int, Dict[int, List[float]]]):
    """
    Print comprehensive statistics about the 3D pose data.
    
    Analyzes the dataset to provide insights about frame count,
    joint coverage, and data completeness.
    
    Args:
        poses: Dictionary of 3D poses
    """
    total_frames = len(poses)
    total_joints = sum(len(frame_joints) for frame_joints in poses.values())
    
    print("\n" + "="*50)
    print("3D POSE DATASET STATISTICS")
    print("="*50)
    print(f"Total frames: {total_frames}")
    print(f"Total joint detections: {total_joints}")
    
    if total_frames > 0:
        avg_joints_per_frame = total_joints / total_frames
        print(f"Average joints per frame: {avg_joints_per_frame:.1f}")
        
        # Calculate joint visibility statistics
        joint_counts = {joint_id: 0 for joint_id in range(len(JOINT_NAMES))}
        for frame_joints in poses.values():
            for joint_id in frame_joints.keys():
                joint_counts[joint_id] += 1
        
        print(f"\nJoint visibility across {total_frames} frames:")
        print("-" * 40)
        for joint_id, count in joint_counts.items():
            visibility_percentage = (count / total_frames) * 100
            print(f"  {joint_id:2d} {JOINT_NAMES[joint_id]:>12}: {count:3d} frames ({visibility_percentage:5.1f}%)")
        
        # Calculate overall completeness
        expected_joints = total_frames * len(JOINT_NAMES)
        completeness = (total_joints / expected_joints) * 100
        print(f"\nDataset completeness: {completeness:.1f}% ({total_joints}/{expected_joints} possible joints)")

# -----------------------------------------------------------------------------
# Command Line Interface
# -----------------------------------------------------------------------------

def create_command_line_parser():
    """
    Create and configure the command line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Visualize 3D human poses from triangulated data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples for 48-frame dataset:
  %(prog)s data.json                                    # Show both static and animated
  %(prog)s data.json --mode animated                    # Animation only
  %(prog)s data.json --mode static --max-frames 8      # Static overlay with 8 frames
  %(prog)s data.json --frame-range 10 30               # Frames 10-30 only
  %(prog)s data.json --stats-only                      # Statistics only
  %(prog)s data.json --interval 250                    # Slower animation
        """
    )
    
    # Required argument
    parser.add_argument("input_file", 
                       help="Path to JSON file containing triangulated 3D poses")
    
    # Visualization mode
    parser.add_argument("--mode", 
                       choices=["static", "animated", "both"], 
                       default="both",
                       help="Visualization mode (default: both)")
    
    # Frame selection
    parser.add_argument("--frame-range", 
                       nargs=2, type=int, 
                       metavar=("START", "END"),
                       help="Frame range to visualize (e.g., --frame-range 0 20)")
    
    parser.add_argument("--max-frames", 
                       type=int, default=5,
                       help="Maximum frames for static overlay (default: 5)")
    
    # Animation settings
    parser.add_argument("--interval", 
                       type=int, default=200,
                       help="Animation interval in milliseconds (default: 200)")
    
    # Display options
    parser.add_argument("--no-labels", 
                       action="store_true",
                       help="Hide joint name labels in static view")
    
    parser.add_argument("--stats-only", 
                       action="store_true",
                       help="Show only statistics, no visualization")
    
    return parser


def main():
    """
    Main function that handles command line arguments and orchestrates visualization.
    """
    # Parse command line arguments
    parser = create_command_line_parser()
    args = parser.parse_args()
    
    # Load 3D pose data
    try:
        poses = load_3d_poses(Path(args.input_file))
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{args.input_file}'")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if not poses:
        print("No valid pose data found in the file")
        return
    
    # Print statistics
    print_pose_statistics(poses)
    
    # If only statistics requested, exit here
    if args.stats_only:
        return
    
    # Prepare frame range
    frame_range = tuple(args.frame_range) if args.frame_range else None
    
    # Show visualizations based on mode
    if args.mode in ("static", "both"):
        print("\nShowing static visualization...")
        show_static_poses(poses, 
                         frame_range=frame_range, 
                         max_frames=args.max_frames, 
                         with_labels=not args.no_labels)
    
    if args.mode in ("animated", "both"):
        print("\nShowing animated visualization...")
        animation_obj = show_animated_poses(poses, 
                                          frame_range=frame_range, 
                                          interval=args.interval)
        
        # Keep reference to animation to prevent garbage collection
        if animation_obj:
            print("Animation created successfully. Close the window to exit.")


if __name__ == "__main__":
    main()