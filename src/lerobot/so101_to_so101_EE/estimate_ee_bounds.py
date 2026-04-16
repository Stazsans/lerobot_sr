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

"""
Estimate a theoretical SO101 end-effector workspace from calibration ranges and FK sampling.

This script:
1. Loads the local SO101 follower calibration file.
2. Converts calibration motor ranges to approximate joint angle limits.
3. Samples the joint space within those limits.
4. Runs FK on each sample and reports a recommended EE bounds box.

The result is only a theoretical workspace estimate based on the current URDF and calibration file.
It does not guarantee self-collision-free or path-safe motion on the real robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.motors import Motor, MotorNormMode
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower, _degrees_limits_from_calibration

FOLLOWER_ID = "so101_follower"
FOLLOWER_PORT = "/dev/ttyACM0"
REPO_ROOT = Path(__file__).resolve().parents[3]
URDF_PATH = REPO_ROOT / "third_party" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
TARGET_FRAME_NAME = "gripper_frame_link"

# Sampling density per joint. Wrist roll is a full-turn joint, so keep it coarser.
JOINT_SAMPLES = {
    "shoulder_pan": 5,
    "shoulder_lift": 5,
    "elbow_flex": 5,
    "wrist_flex": 5,
    "wrist_roll": 7,
}

# Keep a margin from the measured calibration extremes for a safer recommended box.
SAFETY_MARGIN_M = 0.02


@dataclass
class JointLimit:
    name: str
    min_deg: float
    max_deg: float
    samples: int


def build_robot() -> SO101Follower:
    return SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, use_degrees=True))


def get_joint_limits(robot: SO101Follower) -> list[JointLimit]:
    limits: list[JointLimit] = []
    for motor_name, motor in robot.bus.motors.items():
        if motor_name == "gripper":
            continue

        calibration = robot.calibration[motor_name]
        max_resolution = robot.bus.model_resolution_table[motor.model] - 1
        min_deg, max_deg = _degrees_limits_from_calibration(calibration, max_resolution=max_resolution)
        limits.append(
            JointLimit(
                name=motor_name,
                min_deg=min_deg,
                max_deg=max_deg,
                samples=JOINT_SAMPLES.get(motor_name, 5),
            )
        )
    return limits


def sample_joint_positions(joint_limits: list[JointLimit]) -> np.ndarray:
    per_joint_values = [
        np.linspace(limit.min_deg, limit.max_deg, num=limit.samples, dtype=float) for limit in joint_limits
    ]
    return np.array(list(product(*per_joint_values)), dtype=float)


def estimate_bounds(kinematics: RobotKinematics, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = []
    for joint_sample in samples:
        pose = kinematics.forward_kinematics(joint_sample)
        positions.append(pose[:3, 3].astype(float))

    positions_arr = np.array(positions, dtype=float)
    mins = positions_arr.min(axis=0)
    maxs = positions_arr.max(axis=0)
    return mins, maxs


def print_joint_limits(joint_limits: list[JointLimit]) -> None:
    print("Joint limits inferred from calibration:")
    for limit in joint_limits:
        print(
            f"  {limit.name:<14} min={limit.min_deg:+8.2f}deg max={limit.max_deg:+8.2f}deg "
            f"samples={limit.samples}"
        )


def main() -> None:
    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found at {URDF_PATH}")

    robot = build_robot()
    if not robot.calibration:
        raise FileNotFoundError(
            f"No calibration file loaded for id '{FOLLOWER_ID}'. Expected: {robot.calibration_fpath}"
        )

    joint_limits = get_joint_limits(robot)
    print_joint_limits(joint_limits)

    samples = sample_joint_positions(joint_limits)
    print(f"\nSampling {len(samples)} joint configurations...")

    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=TARGET_FRAME_NAME,
        joint_names=[limit.name for limit in joint_limits],
    )
    mins, maxs = estimate_bounds(kinematics, samples)

    recommended_min = mins + SAFETY_MARGIN_M
    recommended_max = maxs - SAFETY_MARGIN_M

    print("\nTheoretical EE bounds from calibration + FK sampling:")
    print(f"  raw min: [{mins[0]:+.4f}, {mins[1]:+.4f}, {mins[2]:+.4f}]")
    print(f"  raw max: [{maxs[0]:+.4f}, {maxs[1]:+.4f}, {maxs[2]:+.4f}]")
    print("\nRecommended conservative EE bounds:")
    print(
        "  EE_BOUNDS_MIN = "
        f"[{recommended_min[0]:+.4f}, {recommended_min[1]:+.4f}, {recommended_min[2]:+.4f}]"
    )
    print(
        "  EE_BOUNDS_MAX = "
        f"[{recommended_max[0]:+.4f}, {recommended_max[1]:+.4f}, {recommended_max[2]:+.4f}]"
    )
    print("\nNote: This estimate does not account for real-world cable strain, table collisions, or all self-collisions.")


if __name__ == "__main__":
    main()
