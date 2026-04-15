# !/usr/bin/env python

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
SO101 真机末端位姿直达脚本。

给定一个或多个目标末端位姿 `(x, y, z, wx, wy, wz, gripper_pos)`，
脚本会读取当前关节状态，做 FK 获取当前末端位姿，再通过 IK 逐步插值到目标位姿。
适合单独验证 SO101 的 FK/IK 是否稳定，而不依赖 leader 臂、数据集或策略推理。
"""

from pathlib import Path

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.robot_utils import precise_sleep

# ========== 用户配置 ==========
FOLLOWER_PORT = "/dev/ttyACM0"
REPO_ROOT = Path(__file__).resolve().parents[3]
URDF_PATH = REPO_ROOT / "third_party" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
TARGET_FRAME_NAME = "gripper_frame_link"

FPS = 20
INTERPOLATION_STEPS = 60
HOLD_TIME_S = 1.0

EE_BOUNDS_MIN = [-0.35, -0.35, -0.05]
EE_BOUNDS_MAX = [0.35, 0.35, 0.40]
MAX_EE_STEP_M = 0.015
MAX_JOINT_DELTA_DEG = {
    "shoulder_pan": 10.0,
    "shoulder_lift": 10.0,
    "elbow_flex": 12.0,
    "wrist_flex": 12.0,
    "wrist_roll": 25.0,
}

TARGET_POSES = [
    {
        "name": "ready_front",
        "ee.x": 0.20,
        "ee.y": 0.00,
        "ee.z": 0.18,
        "ee.wx": 0.0,
        "ee.wy": 1.57,
        "ee.wz": 0.0,
        "ee.gripper_pos": 30.0,
    },
]
# ==============================


def build_processors(robot: SO101Follower, kinematics: RobotKinematics):
    fk_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=kinematics,
                motor_names=list(robot.bus.motors.keys()),
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    ik_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEBoundsAndSafety(
                end_effector_bounds={"min": EE_BOUNDS_MIN, "max": EE_BOUNDS_MAX},
                max_ee_step_m=MAX_EE_STEP_M,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=True,
                max_joint_delta_deg=MAX_JOINT_DELTA_DEG,
                position_tolerance_m=0.02,
                orientation_tolerance_rad=0.35,
                fallback_to_current_joints_on_invalid=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    return fk_processor, ik_processor


def interpolate_ee_pose(
    start_pose: dict[str, float],
    target_pose: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    interpolated = {"name": target_pose.get("name", "target")}
    for key in ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"]:
        start_value = float(start_pose[key])
        target_value = float(target_pose[key])
        interpolated[key] = (1.0 - alpha) * start_value + alpha * target_value
    return interpolated


def print_ee_pose(prefix: str, pose: dict[str, float]) -> None:
    print(
        f"{prefix} x={pose['ee.x']:+.4f} y={pose['ee.y']:+.4f} z={pose['ee.z']:+.4f} "
        f"wx={pose['ee.wx']:+.4f} wy={pose['ee.wy']:+.4f} wz={pose['ee.wz']:+.4f} "
        f"gripper={pose['ee.gripper_pos']:+.2f}"
    )


def main():
    if not URDF_PATH.exists():
        raise FileNotFoundError(
            f"URDF not found at {URDF_PATH}. Update URDF_PATH to your local so101_new_calib.urdf."
        )

    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="so101_pose_arm", use_degrees=True))
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=TARGET_FRAME_NAME,
        joint_names=list(robot.bus.motors.keys()),
    )
    fk_processor, ik_processor = build_processors(robot, kinematics)

    robot.connect()

    try:
        robot_obs = robot.get_observation()
        current_ee = fk_processor(robot_obs.copy())
        print_ee_pose("Current EE:", current_ee)

        for target_pose in TARGET_POSES:
            print(f"\nMoving to target: {target_pose.get('name', 'target')}")
            print_ee_pose("Target EE :", target_pose)

            start_pose = {key: float(current_ee[key]) for key in current_ee if key.startswith("ee.")}
            for step in range(1, INTERPOLATION_STEPS + 1):
                alpha = step / INTERPOLATION_STEPS
                ee_action = interpolate_ee_pose(start_pose, target_pose, alpha)
                robot_obs = robot.get_observation()
                joint_action = ik_processor((ee_action, robot_obs))
                robot.send_action(joint_action)
                precise_sleep(1.0 / FPS)

            precise_sleep(HOLD_TIME_S)
            robot_obs = robot.get_observation()
            current_ee = fk_processor(robot_obs.copy())
            print_ee_pose("Reached EE:", current_ee)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
