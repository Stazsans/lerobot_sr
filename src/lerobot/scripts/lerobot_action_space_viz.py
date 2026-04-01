#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Visualize the Cartesian position distribution of a joint-space LeRobot dataset.

This script reads a local LeRobot dataset, converts the joint-space actions or observations into
Cartesian end-effector positions through forward kinematics, then prints a diversity summary and
saves a visualization figure.

Example:

```
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src /home/sr/miniforge3/envs/lerobot/bin/python \
    -m lerobot.scripts.lerobot_action_space_viz \
    --dataset-root /home/sr/datasets/50step1
```
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from lerobot.model.kinematics import RobotKinematics

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / "matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)


DEFAULT_URDF = Path("/home/sr/SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
DEFAULT_TARGET_FRAME = "gripper_frame_link"
DEFAULT_SOURCE = "action"
DEFAULT_PROJECTION_BINS = 80
DEFAULT_VOXEL_BINS = 20
ROUNDING_RESOLUTIONS_M = (0.001, 0.005, 0.01)
PROJECTION_DIMS = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
AXIS_LABELS = ("x", "y", "z")


def load_dataset_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")

    with info_path.open() as f:
        return json.load(f)


def infer_joint_mapping(
    feature_names: list[str], requested_joint_names: list[str] | None
) -> tuple[list[str], list[int]]:
    available_joint_names = [name.removesuffix(".pos") for name in feature_names if name.endswith(".pos")]
    joint_name_to_index = {name.removesuffix(".pos"): i for i, name in enumerate(feature_names)}

    if requested_joint_names:
        selected_joint_names = [name for name in requested_joint_names if name != "gripper"]
    else:
        selected_joint_names = [name for name in available_joint_names if name != "gripper"]

    missing = [name for name in selected_joint_names if name not in joint_name_to_index]
    if missing:
        raise ValueError(
            f"Requested joints are missing from the dataset feature list: {missing}. "
            f"Available joints: {available_joint_names}"
        )

    selected_indices = [joint_name_to_index[name] for name in selected_joint_names]
    return selected_joint_names, selected_indices


def load_joint_series(
    dataset_root: Path, source: str, selected_indices: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {(dataset_root / 'data')}")

    joint_chunks: list[np.ndarray] = []
    episode_chunks: list[np.ndarray] = []
    frame_chunks: list[np.ndarray] = []

    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=[source, "episode_index", "frame_index"])

        joint_column = table[source].combine_chunks()
        flat_joint_values = joint_column.values.to_numpy(zero_copy_only=False)
        joint_values = flat_joint_values.reshape(len(joint_column), -1)[:, selected_indices].astype(np.float64)

        episode_index = table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
        frame_index = table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)

        joint_chunks.append(joint_values)
        episode_chunks.append(episode_index)
        frame_chunks.append(frame_index)

    return (
        np.concatenate(joint_chunks, axis=0),
        np.concatenate(episode_chunks, axis=0),
        np.concatenate(frame_chunks, axis=0),
    )


def compute_fk_positions(
    joint_values_deg: np.ndarray,
    urdf_path: Path,
    joint_names: list[str],
    target_frame_name: str,
) -> np.ndarray:
    kinematics = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name=target_frame_name,
        joint_names=joint_names,
    )

    positions = np.empty((joint_values_deg.shape[0], 3), dtype=np.float64)
    report_every = max(joint_values_deg.shape[0] // 10, 1000)

    for idx, joints in enumerate(joint_values_deg):
        transform = kinematics.forward_kinematics(joints)
        positions[idx] = transform[:3, 3]

        if (idx + 1) % report_every == 0 or idx + 1 == joint_values_deg.shape[0]:
            print(f"Computed FK for {idx + 1}/{joint_values_deg.shape[0]} frames")

    return positions


def array_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q10": float(np.quantile(values, 0.10)),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q99": float(np.quantile(values, 0.99)),
    }


def rounded_unique_count(points: np.ndarray, resolution_m: float) -> int:
    rounded = np.rint(points / resolution_m).astype(np.int64)
    return int(np.unique(rounded, axis=0).shape[0])


def occupancy_count(points: np.ndarray, bins: int, dims: tuple[int, ...]) -> int:
    selected_points = points[:, dims]
    mins = np.min(selected_points, axis=0)
    maxs = np.max(selected_points, axis=0)
    spans = np.maximum(maxs - mins, 1e-12)

    scaled = (selected_points - mins) / spans
    indices = np.clip(np.floor(scaled * bins).astype(np.int64), 0, bins - 1)
    return int(np.unique(indices, axis=0).shape[0])


def summarize_positions(
    points: np.ndarray,
    episode_index: np.ndarray,
    projection_bins: int,
    voxel_bins: int,
) -> dict[str, Any]:
    position_stats = {axis: array_stats(points[:, axis_idx]) for axis_idx, axis in enumerate(AXIS_LABELS)}
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bbox_size = bbox_max - bbox_min

    rounding_summary = {}
    for resolution in ROUNDING_RESOLUTIONS_M:
        unique_count = rounded_unique_count(points, resolution)
        key = f"{int(round(resolution * 1000))}mm"
        rounding_summary[key] = {
            "resolution_m": resolution,
            "unique_positions": unique_count,
            "unique_ratio": float(unique_count / len(points)),
        }

    projection_summary = {}
    for projection_name, dims in PROJECTION_DIMS.items():
        occupied = occupancy_count(points, projection_bins, dims)
        total_bins = projection_bins ** len(dims)
        projection_summary[projection_name] = {
            "bins_per_axis": projection_bins,
            "occupied_bins": occupied,
            "total_bins": total_bins,
            "occupancy_ratio": float(occupied / total_bins),
        }

    voxel_occupied = occupancy_count(points, voxel_bins, (0, 1, 2))
    voxel_total = voxel_bins**3

    per_episode_boxes = []
    per_episode_unique_5mm = []
    per_episode_lengths = []

    for episode_id in np.unique(episode_index):
        episode_points = points[episode_index == episode_id]
        per_episode_boxes.append(np.max(episode_points, axis=0) - np.min(episode_points, axis=0))
        per_episode_unique_5mm.append(rounded_unique_count(episode_points, 0.005))
        per_episode_lengths.append(len(episode_points))

    per_episode_box_array = np.asarray(per_episode_boxes, dtype=np.float64)
    per_episode_unique_5mm_array = np.asarray(per_episode_unique_5mm, dtype=np.float64)
    per_episode_length_array = np.asarray(per_episode_lengths, dtype=np.float64)

    return {
        "total_frames": int(len(points)),
        "total_episodes": int(len(np.unique(episode_index))),
        "position_stats_m": position_stats,
        "bbox_m": {
            "min": bbox_min.tolist(),
            "max": bbox_max.tolist(),
            "size": bbox_size.tolist(),
            "volume_m3": float(np.prod(bbox_size)),
        },
        "coverage": {
            "voxel": {
                "bins_per_axis": voxel_bins,
                "occupied_bins": voxel_occupied,
                "total_bins": voxel_total,
                "occupancy_ratio": float(voxel_occupied / voxel_total),
            },
            "projection": projection_summary,
        },
        "uniqueness": rounding_summary,
        "per_episode": {
            "frames": {
                "min": float(np.min(per_episode_length_array)),
                "max": float(np.max(per_episode_length_array)),
                "mean": float(np.mean(per_episode_length_array)),
                "median": float(np.median(per_episode_length_array)),
            },
            "bbox_size_m_mean": np.mean(per_episode_box_array, axis=0).tolist(),
            "bbox_size_m_median": np.median(per_episode_box_array, axis=0).tolist(),
            "bbox_size_m_min": np.min(per_episode_box_array, axis=0).tolist(),
            "bbox_size_m_max": np.max(per_episode_box_array, axis=0).tolist(),
            "unique_positions_5mm_mean": float(np.mean(per_episode_unique_5mm_array)),
            "unique_positions_5mm_median": float(np.median(per_episode_unique_5mm_array)),
            "unique_positions_5mm_min": float(np.min(per_episode_unique_5mm_array)),
            "unique_positions_5mm_max": float(np.max(per_episode_unique_5mm_array)),
        },
    }


def extract_episode_min_x_points(
    points: np.ndarray,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
) -> list[dict[str, float | int]]:
    min_x_points: list[dict[str, float | int]] = []

    for episode_id in np.unique(episode_index):
        mask = episode_index == episode_id
        episode_points = points[mask]
        episode_frames = frame_index[mask]
        local_min_index = int(np.argmin(episode_points[:, 0]))
        min_point = episode_points[local_min_index]

        min_x_points.append(
            {
                "episode_index": int(episode_id),
                "frame_index": int(episode_frames[local_min_index]),
                "x": float(min_point[0]),
                "y": float(min_point[1]),
                "z": float(min_point[2]),
            }
        )

    return min_x_points


def save_episode_min_x_points_csv(output_path: Path, min_x_points: list[dict[str, float | int]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_index", "frame_index", "x", "y", "z"])
        writer.writeheader()
        writer.writerows(min_x_points)


def save_episode_min_x_scatter_figure(
    min_x_points: list[dict[str, float | int]],
    output_path: Path,
    dataset_name: str,
    source: str,
    target_frame_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colors
    import matplotlib.pyplot as plt

    episode_ids = np.asarray([int(item["episode_index"]) for item in min_x_points], dtype=np.int64)
    xs = np.asarray([float(item["x"]) for item in min_x_points], dtype=np.float64)
    ys = np.asarray([float(item["y"]) for item in min_x_points], dtype=np.float64)

    color_norm = colors.Normalize(vmin=int(episode_ids.min()), vmax=int(episode_ids.max()))
    color_map = plt.get_cmap("viridis")

    figure, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    scatter = ax.scatter(
        xs,
        ys,
        c=episode_ids,
        cmap=color_map,
        norm=color_norm,
        s=42,
        alpha=0.90,
        edgecolors="black",
        linewidths=0.35,
    )

    for item in min_x_points:
        ax.text(
            float(item["x"]) + 0.0015,
            float(item["y"]) + 0.0015,
            str(int(item["episode_index"])),
            fontsize=6,
            alpha=0.85,
        )

    colorbar = figure.colorbar(scatter, ax=ax, shrink=0.86)
    colorbar.set_label("episode")

    ax.set_title("Per-episode grasp points (minimum x) in XY")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")

    figure.suptitle(
        f"{dataset_name} | {source} -> {target_frame_name} grasp-point XY scatter",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_figure(
    points: np.ndarray,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    output_path: Path,
    dataset_name: str,
    source: str,
    target_frame_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colors
    import matplotlib.pyplot as plt

    unique_episodes = np.unique(episode_index)
    color_norm = colors.Normalize(vmin=int(unique_episodes.min()), vmax=int(unique_episodes.max()))
    color_map = plt.get_cmap("viridis")

    figure, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)

    for episode_id in unique_episodes:
        mask = episode_index == episode_id
        episode_points = points[mask]
        episode_frames = frame_index[mask]
        order = np.argsort(episode_frames, kind="stable")
        ordered_points = episode_points[order]
        color = color_map(color_norm(int(episode_id)))

        ax.plot(
            ordered_points[:, 0],
            ordered_points[:, 1],
            color=color,
            linewidth=1.2,
            alpha=0.55,
        )
        ax.scatter(
            ordered_points[0, 0],
            ordered_points[0, 1],
            color=color,
            s=10,
            alpha=0.8,
            marker="o",
        )
        ax.scatter(
            ordered_points[-1, 0],
            ordered_points[-1, 1],
            color=color,
            s=14,
            alpha=0.9,
            marker="x",
        )

    scalar_mappable = plt.cm.ScalarMappable(norm=color_norm, cmap=color_map)
    colorbar = figure.colorbar(scalar_mappable, ax=ax, shrink=0.85)
    colorbar.set_label("episode")

    ax.set_title("XY trajectories by episode")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")

    figure.suptitle(
        f"{dataset_name} | {source} -> {target_frame_name} XY trajectories",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def format_vector(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def print_summary(
    dataset_root: Path,
    source: str,
    target_frame_name: str,
    joint_names: list[str],
    summary: dict[str, Any],
    plot_path: Path | None,
    json_path: Path,
) -> None:
    print()
    print(f"Dataset: {dataset_root}")
    print(f"Source: {source}")
    print(f"Target frame: {target_frame_name}")
    print(f"Joint names for FK: {', '.join(joint_names)}")
    print(f"Frames: {summary['total_frames']} | Episodes: {summary['total_episodes']}")
    print()
    print("Position stats (meters)")
    for axis in AXIS_LABELS:
        stats = summary["position_stats_m"][axis]
        print(
            f"  {axis}: min={stats['min']:.4f} q01={stats['q01']:.4f} q10={stats['q10']:.4f} "
            f"median={stats['q50']:.4f} q90={stats['q90']:.4f} q99={stats['q99']:.4f} "
            f"max={stats['max']:.4f} std={stats['std']:.4f}"
        )

    bbox = summary["bbox_m"]
    print()
    print("Workspace bounding box")
    print(f"  min: {format_vector(bbox['min'])}")
    print(f"  max: {format_vector(bbox['max'])}")
    print(f"  size: {format_vector(bbox['size'])}")
    print(f"  volume: {bbox['volume_m3']:.6f} m^3")

    voxel = summary["coverage"]["voxel"]
    print()
    print("Coverage")
    print(
        f"  3D voxel occupancy ({voxel['bins_per_axis']}^3): "
        f"{voxel['occupied_bins']} / {voxel['total_bins']} = {100.0 * voxel['occupancy_ratio']:.2f}%"
    )
    for projection_name, projection in summary["coverage"]["projection"].items():
        print(
            f"  {projection_name.upper()} occupancy ({projection['bins_per_axis']}^2): "
            f"{projection['occupied_bins']} / {projection['total_bins']} = "
            f"{100.0 * projection['occupancy_ratio']:.2f}%"
        )

    print()
    print("Unique positions after rounding")
    for label, item in summary["uniqueness"].items():
        print(
            f"  {label}: {item['unique_positions']} / {summary['total_frames']} = "
            f"{100.0 * item['unique_ratio']:.2f}%"
        )

    per_episode = summary["per_episode"]
    print()
    print("Per-episode spread")
    print(
        f"  frames: min={per_episode['frames']['min']:.0f} max={per_episode['frames']['max']:.0f} "
        f"mean={per_episode['frames']['mean']:.1f} median={per_episode['frames']['median']:.1f}"
    )
    print(f"  bbox mean: {format_vector(per_episode['bbox_size_m_mean'])}")
    print(f"  bbox median: {format_vector(per_episode['bbox_size_m_median'])}")
    print(
        f"  unique 5mm positions: min={per_episode['unique_positions_5mm_min']:.0f} "
        f"max={per_episode['unique_positions_5mm_max']:.0f} "
        f"mean={per_episode['unique_positions_5mm_mean']:.1f} "
        f"median={per_episode['unique_positions_5mm_median']:.1f}"
    )

    print()
    if plot_path is not None:
        print(f"Saved plot: {plot_path}")
    print(f"Saved summary: {json_path}")


def print_episode_min_x_points(
    min_x_points: list[dict[str, float | int]],
    csv_path: Path | None = None,
    plot_path: Path | None = None,
) -> None:
    print()
    print("Per-episode grasp points (minimum x)")
    for item in min_x_points:
        print(
            f"  ep={item['episode_index']:02d} frame={item['frame_index']:03d} "
            f"x={item['x']:.4f} y={item['y']:.4f} z={item['z']:.4f}"
        )

    if csv_path is not None:
        print()
        print(f"Saved grasp points CSV: {csv_path}")
    if plot_path is not None:
        print(f"Saved grasp points XY scatter: {plot_path}")


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the local LeRobot dataset root directory.",
    )
    argument_parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE,
        choices=["action", "observation.state"],
        help="Dataset feature to visualize after FK.",
    )
    argument_parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="Path to the SO101 URDF used for forward kinematics.",
    )
    argument_parser.add_argument(
        "--target-frame",
        type=str,
        default=DEFAULT_TARGET_FRAME,
        help="Target frame name inside the URDF.",
    )
    argument_parser.add_argument(
        "--joint-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional FK joint names override. Defaults to dataset joints excluding gripper.",
    )
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the generated PNG and JSON summary.",
    )
    argument_parser.add_argument(
        "--projection-bins",
        type=int,
        default=DEFAULT_PROJECTION_BINS,
        help="Number of bins per axis for the 2D density projections.",
    )
    argument_parser.add_argument(
        "--voxel-bins",
        type=int,
        default=DEFAULT_VOXEL_BINS,
        help="Number of bins per axis used for 3D occupancy statistics.",
    )
    argument_parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the matplotlib figure and only print/save the summary JSON.",
    )
    argument_parser.add_argument(
        "--grasp-point-only",
        action="store_true",
        help="Only print the minimum-x point of each episode and skip the plot.",
    )
    return argument_parser


def main() -> None:
    args = parser().parse_args()

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.dataset_root}")
    if not args.urdf.exists():
        raise FileNotFoundError(f"URDF does not exist: {args.urdf}")

    info = load_dataset_info(args.dataset_root)
    source_feature = info["features"][args.source]
    feature_names = source_feature["names"]
    joint_names, selected_indices = infer_joint_mapping(feature_names, args.joint_names)

    output_dir = args.output_dir or (Path.cwd() / "outputs" / "action_space_viz" / args.dataset_root.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.dataset_root}")
    joint_values, episode_index, frame_index = load_joint_series(args.dataset_root, args.source, selected_indices)
    print(f"Loaded {joint_values.shape[0]} frames from {len(np.unique(episode_index))} episodes")

    print(f"Running FK with target frame '{args.target_frame}'")
    positions = compute_fk_positions(
        joint_values_deg=joint_values,
        urdf_path=args.urdf,
        joint_names=joint_names,
        target_frame_name=args.target_frame,
    )

    min_x_points = extract_episode_min_x_points(
        points=positions,
        episode_index=episode_index,
        frame_index=frame_index,
    )
    min_x_csv_path = output_dir / f"{args.source.replace('.', '_')}_{args.target_frame}_episode_min_x_points.csv"
    save_episode_min_x_points_csv(min_x_csv_path, min_x_points)
    min_x_plot_path = None
    if not args.no_plot:
        min_x_plot_path = (
            output_dir / f"{args.source.replace('.', '_')}_{args.target_frame}_episode_min_x_points_xy_scatter.png"
        )
        save_episode_min_x_scatter_figure(
            min_x_points=min_x_points,
            output_path=min_x_plot_path,
            dataset_name=args.dataset_root.name,
            source=args.source,
            target_frame_name=args.target_frame,
        )

    if args.grasp_point_only:
        print_episode_min_x_points(
            min_x_points=min_x_points,
            csv_path=min_x_csv_path,
            plot_path=min_x_plot_path,
        )
        return

    summary = summarize_positions(
        points=positions,
        episode_index=episode_index,
        projection_bins=args.projection_bins,
        voxel_bins=args.voxel_bins,
    )
    summary.update(
        {
            "dataset_root": str(args.dataset_root),
            "source": args.source,
            "urdf": str(args.urdf),
            "target_frame": args.target_frame,
            "joint_names": joint_names,
            "per_episode_min_x_points": min_x_points,
        }
    )

    summary_path = output_dir / f"{args.source.replace('.', '_')}_{args.target_frame}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    plot_path = None
    if not args.no_plot:
        plot_path = output_dir / f"{args.source.replace('.', '_')}_{args.target_frame}_distribution.png"
        save_figure(
            points=positions,
            episode_index=episode_index,
            frame_index=frame_index,
            output_path=plot_path,
            dataset_name=args.dataset_root.name,
            source=args.source,
            target_frame_name=args.target_frame,
        )

    print_summary(
        dataset_root=args.dataset_root,
        source=args.source,
        target_frame_name=args.target_frame,
        joint_names=joint_names,
        summary=summary,
        plot_path=plot_path,
        json_path=summary_path,
    )
    print_episode_min_x_points(
        min_x_points=min_x_points,
        csv_path=min_x_csv_path,
        plot_path=min_x_plot_path,
    )


if __name__ == "__main__":
    main()
