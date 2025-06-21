
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3‑D proj
import matplotlib.animation as animation

""" 
# Basic usage - show both static and animated
python visualize_3d_poses.py triangulated_positions_v2.json

# Only animated view
python visualize_3d_poses.py triangulated_positions_v2.json --mode animated

# Static view of frames 100-200
python visualize_3d_poses.py triangulated_positions_v2.json --mode static --frame-range 0 30

# Just statistics
python visualize_3d_poses.py triangulated_positions_v2.json --stats-only

# Custom animation speed (faster)
python visualize_3d_poses.py triangulated_positions_v2.json --mode animated --interval 100

"""

# -----------------------------------------------------------------------------
# Dataset‑specific skeleton (18 joint)
# -----------------------------------------------------------------------------

JOINT_NAMES = [
    "Hips", "RHip", "RKnee", "RAnkle", "RFoot", "LHip", "LKnee", "LAnkle", "LFoot",
    "Spine", "Neck", "Head", "RShoulder", "RElbow", "RHand", "LShoulder", "LElbow", "LHand"
]

SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
    # Pelvis & legs
    (0, 1), (1, 2), (2, 3), (3, 4),     # right leg
    (0, 5), (5, 6), (6, 7), (7, 8),     # left leg
    # Torso & head
    (0, 9), (9, 10), (10, 11),          # spine → neck → head
    # Right arm
    (10, 12), (12, 13), (13, 14),
    # Left arm
    (10, 15), (15, 16), (16, 17)
]

# -----------------------------------------------------------------------------
# I/O utilities
# -----------------------------------------------------------------------------

def load_3d_poses(path: Path) -> Dict[int, Dict[int, List[float]]]:
    """Load {frame:{joint:[x,y,z]}} JSON where keys are *strings*."""
    with open(path, "r") as f:
        raw = json.load(f)
    data: Dict[int, Dict[int, List[float]]] = {}
    for f_str, joints in raw.items():
        if f_str == "summary_3d":
            continue  # ignore metadata from triangulate_pose3d.py
        frame = int(f_str)
        data[frame] = {}
        for j, pos in joints.items():
            # Handle both 3-element [x,y,z] and 4-element [x,y,z,confidence] formats
            if isinstance(pos, list) and len(pos) >= 3:
                # If it has 4 elements, check confidence (assume -1 means invalid)
                if len(pos) == 4 and pos[3] == -1:
                    continue  # skip invalid joints
                data[frame][int(j)] = pos[:3]  # take only x,y,z
            else:
                print(f"Warning: Invalid position data for frame {frame}, joint {j}: {pos}")
    return dict(sorted(data.items()))


def pose_bounds(poses: Dict[int, Dict[int, List[float]]], margin: float = 100) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    pts = [pos for joints in poses.values() for pos in joints.values()]
    xyz = np.asarray(pts, dtype=float)
    min_xyz = xyz.min(axis=0) - margin
    max_xyz = xyz.max(axis=0) + margin
    return (min_xyz[0], max_xyz[0]), (min_xyz[1], max_xyz[1]), (min_xyz[2], max_xyz[2])

# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def draw_skeleton(ax: Axes3D, joints3d: np.ndarray, joint_ids: np.ndarray, *, colour: str = "tab:blue", alpha: float = 1.0):
    ax.scatter(joints3d[:, 0], joints3d[:, 1], joints3d[:, 2], c=colour, s=60, alpha=alpha)
    for a, b in SKELETON_CONNECTIONS:
        if a in joint_ids and b in joint_ids:
            pa = joints3d[joint_ids.tolist().index(a)]
            pb = joints3d[joint_ids.tolist().index(b)]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], c=colour, linewidth=2, alpha=alpha)

# -----------------------------------------------------------------------------
# Static plot
# -----------------------------------------------------------------------------

def show_static(poses: Dict[int, Dict[int, List[float]]], *, frame_range: Tuple[int, int] | None, max_frames: int, with_labels: bool):
    frames = [f for f in poses if (frame_range is None or frame_range[0] <= f <= frame_range[1])]
    frames = frames[:max_frames]
    if not frames:
        print("No frames to show")
        return
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    (xlim, ylim, zlim) = pose_bounds(poses)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    cmap = plt.cm.plasma(np.linspace(0, 1, len(frames)))
    for clr, f in zip(cmap, frames):
        joints = np.array(list(poses[f].values()))
        ids    = np.array(list(poses[f].keys()))
        draw_skeleton(ax, joints, ids, colour=clr)
        if with_labels:
            for jid, pos in zip(ids, joints):
                ax.text(pos[0], pos[1], pos[2], f"{JOINT_NAMES[jid]}", fontsize=7)

    ax.set_title(f"Frames {frames[0]} – {frames[-1]}")
    plt.tight_layout(); plt.show()

# -----------------------------------------------------------------------------
# Animation
# -----------------------------------------------------------------------------

def show_animation(poses: Dict[int, Dict[int, List[float]]], *, frame_range: Tuple[int, int] | None, interval: int):
    frames = [f for f in poses if (frame_range is None or frame_range[0] <= f <= frame_range[1])]
    if len(frames) < 2:
        print("Need at least 2 frames for animation"); return
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    (xlim, ylim, zlim) = pose_bounds(poses)

    def _init():
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        return []

    def _update(idx):
        ax.cla()
        _init()
        f = frames[idx]
        joints = np.array(list(poses[f].values()))
        ids    = np.array(list(poses[f].keys()))
        draw_skeleton(ax, joints, ids)
        ax.set_title(f"Frame {f} ({idx+1}/{len(frames)})")
        return []

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), init_func=_init, interval=interval, blit=False)
    plt.tight_layout(); plt.show()
    return ani

# -----------------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------------

def print_stats(poses: Dict[int, Dict[int, List[float]]]):
    fnum = len(poses)
    jtotal = sum(len(j) for j in poses.values())
    print("\n=== 3‑D POSE STATS ===")
    print(f"Frames: {fnum}")
    print(f"Total joints: {jtotal}")
    print(f"Avg joints/frame: {jtotal/fnum:.1f}")
    counts = {jid:0 for jid in range(len(JOINT_NAMES))}
    for joints in poses.values():
        for jid in joints: counts[jid]+=1
    print("Joint visibility (frames):")
    for jid, cnt in counts.items():
        print(f"  {jid:2d} {JOINT_NAMES[jid]:>10}: {cnt}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def cli():
    ap = argparse.ArgumentParser("Visualise 3‑D human pose")
    ap.add_argument("input", help="triangulated_positions.json")
    ap.add_argument("--mode", choices=["static", "animated", "both"], default="both")
    ap.add_argument("--frame-range", nargs=2, type=int, metavar=("START", "END"))
    ap.add_argument("--max-frames", type=int, default=5)
    ap.add_argument("--interval", type=int, default=200)
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    poses = load_3d_poses(Path(args.input))
    print_stats(poses)

    if args.stats_only:
        return

    fr = tuple(args.frame_range) if args.frame_range else None
    if args.mode in ("static", "both"):
        show_static(poses, frame_range=fr, max_frames=args.max_frames, with_labels=not args.no_labels)
    if args.mode in ("animated", "both"):
        show_animation(poses, frame_range=fr, interval=args.interval)

if __name__ == "__main__":
    cli()
