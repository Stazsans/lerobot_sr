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
Small real-robot diagnostic for SO101 joint directions and FK consistency.

For each selected joint, this script:
1. Reads the current joint state and FK end-effector pose.
2. Applies a small positive delta to one joint only.
3. Reads the new joint state and FK end-effector pose.
4. Prints the measured joint delta and the EE delta.
5. Returns the arm to the starting configuration.

Use it to identify mismatches between:
- expected joint sign
- real motor motion
- FK end-effector direction
"""

from __future__ import annotations

from pathlib import Path

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import observation_to_transition, transition_to_observation
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import ForwardKinematicsJointsToEE
from lerobot.utils.robot_utils import precise_sleep

FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "so101_follower"
REPO_ROOT = Path(__file__).resolve().parents[3]
URDF_PATH = REPO_ROOT / "third_party" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
TARGET_FRAME_NAME = "gripper_frame_link"

JOINT_TEST_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
JOINT_DELTA_DEG = 5.0
SETTLE_TIME_S = 1.0
RETURN_SETTLE_TIME_S = 1.0


def build_fk_processor(robot: SO101Follower, kinematics: RobotKinematics):
    return RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=kinematics,
                motor_names=list(robot.bus.motors.keys()),
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )


def print_joint_and_ee(prefix: str, observation: dict[str, float], ee_pose: dict[str, float]) -> None:
    print(
        f"{prefix} EE x={ee_pose['ee.x']:+.4f} y={ee_pose['ee.y']:+.4f} z={ee_pose['ee.z']:+.4f} "
        f"wx={ee_pose['ee.wx']:+.4f} wy={ee_pose['ee.wy']:+.4f} wz={ee_pose['ee.wz']:+.4f}"
    )
    print(
        "  joints "
        + " ".join(
            f"{name}={observation[f'{name}.pos']:+.2f}"
            for name in JOINT_TEST_ORDER + ["gripper"]
            if f"{name}.pos" in observation
        )
    )


def main() -> None:
    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found at {URDF_PATH}")

    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, use_degrees=True))
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=TARGET_FRAME_NAME,
        joint_names=list(robot.bus.motors.keys()),
    )
    fk_processor = build_fk_processor(robot, kinematics)

    robot.connect(calibrate=False)
    try:
        base_obs = robot.get_observation()
        base_ee = fk_processor(base_obs.copy())
        print_joint_and_ee("Initial", base_obs, base_ee)

        for joint_name in JOINT_TEST_ORDER:
            print(f"\nTesting joint: {joint_name} (+{JOINT_DELTA_DEG:.1f} deg)")

            start_obs = robot.get_observation()
            start_ee = fk_processor(start_obs.copy())
            action = {key: float(value) for key, value in start_obs.items() if key.endswith(".pos")}
            action[f"{joint_name}.pos"] += JOINT_DELTA_DEG

            robot.send_action(action)
            precise_sleep(SETTLE_TIME_S)

            moved_obs = robot.get_observation()
            moved_ee = fk_processor(moved_obs.copy())

            measured_joint_delta = moved_obs[f"{joint_name}.pos"] - start_obs[f"{joint_name}.pos"]
            dx = moved_ee["ee.x"] - start_ee["ee.x"]
            dy = moved_ee["ee.y"] - start_ee["ee.y"]
            dz = moved_ee["ee.z"] - start_ee["ee.z"]

            print_joint_and_ee("  Before", start_obs, start_ee)
            print_joint_and_ee("  After ", moved_obs, moved_ee)
            print(
                f"  Result joint_delta={measured_joint_delta:+.2f}deg "
                f"EE delta: dx={dx:+.4f} dy={dy:+.4f} dz={dz:+.4f}"
            )

            print("  Returning to start pose")
            return_action = {key: float(value) for key, value in start_obs.items() if key.endswith(".pos")}
            robot.send_action(return_action)
            precise_sleep(RETURN_SETTLE_TIME_S)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
