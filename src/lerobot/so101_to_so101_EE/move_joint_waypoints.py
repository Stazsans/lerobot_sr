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
SO101 real-robot joint-space waypoint script.

This bypasses EE inverse kinematics and sends joint targets directly. Use it to verify that:
- motor calibration is usable
- joint-space commands are stable
- the real arm can execute small controlled motions before debugging EE IK
"""

from __future__ import annotations

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "so101_follower"

FPS = 20
INTERPOLATION_STEPS = 80
HOLD_TIME_S = 1.0

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Set either absolute joint targets ("joint") or relative targets ("delta.joint").
# Values are in degrees for arm joints and 0-100 for the gripper.
JOINT_WAYPOINTS = [
    {
        "name": "small_forward",
        "delta.shoulder_lift": 6.0,
        "delta.elbow_flex": 6.0,
        "delta.wrist_flex": 3.0,
    },
    {
        "name": "small_left",
        "delta.shoulder_pan": 8.0,
    },
    {
        "name": "return_center",
        "delta.shoulder_pan": -8.0,
        "delta.shoulder_lift": -6.0,
        "delta.elbow_flex": -6.0,
        "delta.wrist_flex": -3.0,
    },
]


def observation_to_joint_state(observation: dict[str, float]) -> dict[str, float]:
    return {name: float(observation[f"{name}.pos"]) for name in JOINT_NAMES}


def resolve_joint_target(start: dict[str, float], waypoint: dict[str, float | str]) -> dict[str, float]:
    target = dict(start)
    for name in JOINT_NAMES:
        if name in waypoint:
            target[name] = float(waypoint[name])
        elif f"delta.{name}" in waypoint:
            target[name] = float(start[name]) + float(waypoint[f"delta.{name}"])
    return target


def interpolate_joint_state(
    start: dict[str, float],
    target: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    return {name: (1.0 - alpha) * start[name] + alpha * target[name] for name in JOINT_NAMES}


def to_robot_action(joint_state: dict[str, float]) -> dict[str, float]:
    return {f"{name}.pos": value for name, value in joint_state.items()}


def print_joint_state(prefix: str, joint_state: dict[str, float]) -> None:
    print(
        f"{prefix} "
        + " ".join(
            [
                f"pan={joint_state['shoulder_pan']:+.2f}",
                f"lift={joint_state['shoulder_lift']:+.2f}",
                f"elbow={joint_state['elbow_flex']:+.2f}",
                f"wrist_flex={joint_state['wrist_flex']:+.2f}",
                f"wrist_roll={joint_state['wrist_roll']:+.2f}",
                f"gripper={joint_state['gripper']:+.2f}",
            ]
        )
    )


def print_joint_error(target: dict[str, float], reached: dict[str, float]) -> None:
    print(
        "Joint error "
        + " ".join(f"{name}={reached[name] - target[name]:+.2f}" for name in JOINT_NAMES)
    )


def main() -> None:
    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, use_degrees=True))
    robot.connect(calibrate=False)

    try:
        current = observation_to_joint_state(robot.get_observation())
        print_joint_state("Current:", current)

        for waypoint in JOINT_WAYPOINTS:
            name = str(waypoint.get("name", "waypoint"))
            start = observation_to_joint_state(robot.get_observation())
            target = resolve_joint_target(start, waypoint)

            print(f"\nMoving to joint waypoint: {name}")
            print_joint_state("Start :", start)
            print_joint_state("Target:", target)

            for step in range(1, INTERPOLATION_STEPS + 1):
                alpha = step / INTERPOLATION_STEPS
                command = interpolate_joint_state(start, target, alpha)
                robot.send_action(to_robot_action(command))
                precise_sleep(1.0 / FPS)

            precise_sleep(HOLD_TIME_S)
            reached = observation_to_joint_state(robot.get_observation())
            print_joint_state("Reached:", reached)
            print_joint_error(target, reached)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
