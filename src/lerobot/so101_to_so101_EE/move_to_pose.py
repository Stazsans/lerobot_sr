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
FOLLOWER_ID = "so101_follower"
REPO_ROOT = Path(__file__).resolve().parents[3]
URDF_PATH = REPO_ROOT / "third_party" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
TARGET_FRAME_NAME = "gripper_frame_link"

FPS = 20
INTERPOLATION_STEPS = 60
HOLD_TIME_S = 1.0
REFINEMENT_PASSES = 1
POSITION_CONVERGENCE_M = 0.03

EE_BOUNDS_MIN = [-0.3186, -0.3973, -0.2272]
EE_BOUNDS_MAX = [0.4700, 0.3975, 0.5210]
MAX_EE_STEP_M = 0.08
MAX_JOINT_DELTA_DEG = {
    "shoulder_pan": 70.0,
    "shoulder_lift": 70.0,
    "elbow_flex": 80.0,
    "wrist_flex": 80.0,
    "wrist_roll": 140.0,
}
IK_POSITION_TOLERANCE_M = 0.18
IK_ORIENTATION_TOLERANCE_RAD = 1.8
USE_TARGET_ORIENTATION = False

# 只执行一个手动设置的目标点。
# 可使用绝对目标 "ee.x/y/z/wx/wy/wz"，也可使用相对当前位姿的 "delta.ee.x/y/z/wx/wy/wz"。
# 当 USE_TARGET_ORIENTATION = False 时，目标中的 ee.wx/wy/wz 会被忽略，末端姿态保持当前值。
TARGET_POSE = {
    "name": "manual_target",
    "delta.ee.x": 0.02,
    "delta.ee.y": 0.02,
    "delta.ee.z": -0.02,
    "ee.wx": 0.1,
    "ee.wy": 0.1,
    "ee.wz": 1.3,
    "ee.gripper_pos": 39.0,
}
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
                max_orientation_step_rad=0.8,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=True,
                max_joint_delta_deg=MAX_JOINT_DELTA_DEG,
                position_tolerance_m=IK_POSITION_TOLERANCE_M,
                orientation_tolerance_rad=IK_ORIENTATION_TOLERANCE_RAD,
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


def resolve_target_pose(current_pose: dict[str, float], target_pose: dict[str, float]) -> dict[str, float]:
    resolved = {"name": target_pose.get("name", "target")}
    for key in ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"]:
        if key in ["ee.wx", "ee.wy", "ee.wz"] and not USE_TARGET_ORIENTATION:
            resolved[key] = float(current_pose[key])
            continue

        if key in target_pose:
            resolved[key] = float(target_pose[key])
        else:
            delta_key = f"delta.{key}"
            resolved[key] = float(current_pose[key]) + float(target_pose.get(delta_key, 0.0))
    return resolved


def print_ee_pose(prefix: str, pose: dict[str, float]) -> None:
    print(
        f"{prefix} x={pose['ee.x']:+.4f} y={pose['ee.y']:+.4f} z={pose['ee.z']:+.4f} "
        f"wx={pose['ee.wx']:+.4f} wy={pose['ee.wy']:+.4f} wz={pose['ee.wz']:+.4f} "
        f"gripper={pose['ee.gripper_pos']:+.2f}"
    )


def print_target_error(target_pose: dict[str, float], reached_pose: dict[str, float]) -> None:
    dx = reached_pose["ee.x"] - target_pose["ee.x"]
    dy = reached_pose["ee.y"] - target_pose["ee.y"]
    dz = reached_pose["ee.z"] - target_pose["ee.z"]
    dpos = (dx**2 + dy**2 + dz**2) ** 0.5
    print(f"Target error: dx={dx:+.4f}m dy={dy:+.4f}m dz={dz:+.4f}m |pos|={dpos:.4f}m")


def position_error_norm(target_pose: dict[str, float], reached_pose: dict[str, float]) -> float:
    dx = reached_pose["ee.x"] - target_pose["ee.x"]
    dy = reached_pose["ee.y"] - target_pose["ee.y"]
    dz = reached_pose["ee.z"] - target_pose["ee.z"]
    return float((dx**2 + dy**2 + dz**2) ** 0.5)


def refresh_target_orientation_from_current(
    target_pose: dict[str, float], current_pose: dict[str, float]
) -> dict[str, float]:
    refreshed = dict(target_pose)
    for key in ["ee.wx", "ee.wy", "ee.wz"]:
        refreshed[key] = float(current_pose[key])
    return refreshed


def main():
    if not URDF_PATH.exists():
        raise FileNotFoundError(
            f"URDF not found at {URDF_PATH}. Update URDF_PATH to your local so101_new_calib.urdf."
        )

    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, use_degrees=True))
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

        start_pose = {key: float(current_ee[key]) for key in current_ee if key.startswith("ee.")}
        resolved_target_pose = resolve_target_pose(start_pose, TARGET_POSE)

        print(f"\nMoving to target: {resolved_target_pose.get('name', 'target')}")
        print_ee_pose("Target EE :", resolved_target_pose)

        best_error = float("inf")
        for pass_idx in range(1, REFINEMENT_PASSES + 1):
            if pass_idx > 1:
                print(f"\nRefinement pass {pass_idx}/{REFINEMENT_PASSES}")
                if not USE_TARGET_ORIENTATION:
                    resolved_target_pose = refresh_target_orientation_from_current(resolved_target_pose, start_pose)
                    print_ee_pose("Refined target EE :", resolved_target_pose)

            for step in range(1, INTERPOLATION_STEPS + 1):
                alpha = step / INTERPOLATION_STEPS
                ee_action = interpolate_ee_pose(start_pose, resolved_target_pose, alpha)
                robot_obs = robot.get_observation()
                joint_action = ik_processor((ee_action, robot_obs))
                robot.send_action(joint_action)
                precise_sleep(1.0 / FPS)

            precise_sleep(HOLD_TIME_S)
            robot_obs = robot.get_observation()
            current_ee = fk_processor(robot_obs.copy())
            err = position_error_norm(resolved_target_pose, current_ee)
            print_ee_pose("Reached EE:", current_ee)
            print_target_error(resolved_target_pose, current_ee)
            if err >= best_error:
                print("Stopping refinement because position error did not improve.")
                break
            best_error = err
            if err <= POSITION_CONVERGENCE_M:
                break
            start_pose = {key: float(current_ee[key]) for key in current_ee if key.startswith("ee.")}

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
