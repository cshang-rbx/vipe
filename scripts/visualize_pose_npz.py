#!/usr/bin/env python3

from __future__ import annotations

import argparse

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a ViPE pose NPZ trajectory.")
    parser.add_argument("pose_npz", type=Path, help="Path to a pose .npz file saved by ViPE.")
    parser.add_argument(
        "--convention",
        choices=["c2w", "w2c"],
        default="c2w",
        help="How to interpret the matrices stored in the NPZ. ViPE pose artifacts use c2w.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=20,
        help="Draw one camera orientation marker every N poses.",
    )
    parser.add_argument(
        "--axis-scale",
        type=float,
        default=0.05,
        help="Scale of the drawn camera orientation axes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to <pose_npz stem>_trajectory.png.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title.",
    )
    return parser.parse_args()


def load_pose_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "data" not in data.files or "inds" not in data.files:
        raise ValueError(f"{path} does not look like a ViPE pose artifact. Expected keys: data, inds.")

    poses = data["data"]
    inds = data["inds"]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected pose data with shape (N, 4, 4), got {poses.shape}.")

    return poses, inds


def as_c2w(poses: np.ndarray, convention: str) -> np.ndarray:
    if convention == "c2w":
        return poses
    return np.linalg.inv(poses)


def set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main() -> None:
    args = parse_args()
    poses_raw, inds = load_pose_npz(args.pose_npz)
    poses_c2w = as_c2w(poses_raw, args.convention)

    camera_centers = poses_c2w[:, :3, 3]
    output_path = args.output or args.pose_npz.with_name(f"{args.pose_npz.stem}_trajectory.png")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        camera_centers[:, 0],
        camera_centers[:, 1],
        camera_centers[:, 2],
        color="tab:blue",
        linewidth=2,
        label="camera centers",
    )
    ax.scatter(
        camera_centers[0, 0],
        camera_centers[0, 1],
        camera_centers[0, 2],
        color="tab:green",
        s=50,
        label=f"start (frame {inds[0]})",
    )
    ax.scatter(
        camera_centers[-1, 0],
        camera_centers[-1, 1],
        camera_centers[-1, 2],
        color="tab:red",
        s=50,
        label=f"end (frame {inds[-1]})",
    )

    step = max(args.stride, 1)
    for pose_idx in range(0, len(poses_c2w), step):
        pose = poses_c2w[pose_idx]
        origin = pose[:3, 3]
        rot = pose[:3, :3]

        for axis_idx, color in enumerate(["r", "g", "b"]):
            direction = rot[:, axis_idx] * args.axis_scale
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                direction[0],
                direction[1],
                direction[2],
                color=color,
                linewidth=1.2,
                arrow_length_ratio=0.2,
            )

    set_equal_axes(ax, camera_centers)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    ax.set_title(args.title or f"{args.pose_npz.name} ({args.convention})")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved trajectory plot to {output_path}")


if __name__ == "__main__":
    main()
