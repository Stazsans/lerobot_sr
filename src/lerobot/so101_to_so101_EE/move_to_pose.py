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

给定一个目标末端位姿 `(x, y, z, wx, wy, wz, gripper_pos)`，
脚本会反复读取当前关节状态，做 FK 获取当前末端位姿，
再通过 IK 朝目标逐步推进。这样真机滞后、限幅、或某步 IK 回退时，
下一步仍会基于当前真实状态继续修正。

用户侧位置量统一使用厘米（cm）配置和打印，内部进入 FK/IK 前再换算成米。
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
MAX_CONTROL_STEPS = 180
HOLD_TIME_S = 0.1
POSITION_CONVERGENCE_CM = 1.0
ORIENTATION_CONVERGENCE_RAD = 0.10
GRIPPER_CONVERGENCE = 0.5
NO_IMPROVEMENT_STEPS = 80
MIN_ERROR_IMPROVEMENT_CM = 0.05

EE_BOUNDS_MIN_CM = [-31.86, -39.73, -22.72]
EE_BOUNDS_MAX_CM = [47.00, 39.75, 52.10]
MAX_COMMAND_STEP_CM = 1.5
MAX_EE_SAFETY_STEP_CM = 2.5
MAX_GRIPPER_STEP = 5.0
MAX_JOINT_DELTA_DEG = {
    "shoulder_pan": 35.0,
    "shoulder_lift": 35.0,
    "elbow_flex": 45.0,
    "wrist_flex": 45.0,
    "wrist_roll": 80.0,
}
IK_POSITION_TOLERANCE_CM = 18.0
IK_ORIENTATION_TOLERANCE_RAD = 1.8
USE_TARGET_ORIENTATION = False
POSE_KEYS = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"]
POSITION_KEYS = ["ee.x", "ee.y", "ee.z"]
ORIENTATION_KEYS = ["ee.wx", "ee.wy", "ee.wz"]
CM_PER_M = 100.0
POSITION_CONVERGENCE_M = POSITION_CONVERGENCE_CM / CM_PER_M
MIN_ERROR_IMPROVEMENT_M = MIN_ERROR_IMPROVEMENT_CM / CM_PER_M
EE_BOUNDS_MIN = [value / CM_PER_M for value in EE_BOUNDS_MIN_CM]
EE_BOUNDS_MAX = [value / CM_PER_M for value in EE_BOUNDS_MAX_CM]
MAX_COMMAND_STEP_M = MAX_COMMAND_STEP_CM / CM_PER_M
MAX_EE_SAFETY_STEP_M = MAX_EE_SAFETY_STEP_CM / CM_PER_M
IK_POSITION_TOLERANCE_M = IK_POSITION_TOLERANCE_CM / CM_PER_M

TARGET_POSE = {
    "name": "manual_target",
    "delta.ee.x": 10,
    "delta.ee.y": 10,
    "delta.ee.z": -10,
    "delta.ee.wx": 0,
    "delta.ee.wy": 0,
    "delta.ee.wz": 0,
    "ee.gripper_pos": 30,
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
                max_ee_step_m=MAX_EE_SAFETY_STEP_M,
                max_orientation_step_rad=0.8 if USE_TARGET_ORIENTATION else None,
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


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def clip_delta(value: float, max_abs_delta: float) -> float:
    return clamp(value, -max_abs_delta, max_abs_delta)


def meters_to_cm(value_m: float) -> float:
    return float(value_m * CM_PER_M)


def cm_to_meters(value_cm: float) -> float:
    return float(value_cm / CM_PER_M)


def pose_meters_to_cm(pose: dict[str, float]) -> dict[str, float]:
    converted = dict(pose)
    for key in POSITION_KEYS:
        converted[key] = meters_to_cm(float(converted[key]))
    return converted


def resolve_target_pose(current_pose: dict[str, float], target_pose: dict[str, float]) -> dict[str, float]:
    resolved = {"name": target_pose.get("name", "target")}
    for key in POSE_KEYS:
        if key in ORIENTATION_KEYS and not USE_TARGET_ORIENTATION:
            resolved[key] = float(current_pose[key])
            continue

        if key in target_pose:
            value = float(target_pose[key])
            resolved[key] = cm_to_meters(value) if key in POSITION_KEYS else value
        else:
            delta_key = f"delta.{key}"
            delta_value = float(target_pose.get(delta_key, 0.0))
            if key in POSITION_KEYS:
                delta_value = cm_to_meters(delta_value)
            resolved[key] = float(current_pose[key]) + delta_value
    return resolved


def clip_target_pose_to_bounds(target_pose: dict[str, float]) -> dict[str, float]:
    clipped = dict(target_pose)
    for idx, key in enumerate(POSITION_KEYS):
        clipped[key] = clamp(float(clipped[key]), EE_BOUNDS_MIN[idx], EE_BOUNDS_MAX[idx])
    clipped["ee.gripper_pos"] = clamp(float(clipped["ee.gripper_pos"]), 0.0, 100.0)
    return clipped


def print_ee_pose(prefix: str, pose: dict[str, float]) -> None:
    pose_cm = pose_meters_to_cm(pose)
    print(
        f"{prefix} x={pose_cm['ee.x']:+.2f}cm y={pose_cm['ee.y']:+.2f}cm z={pose_cm['ee.z']:+.2f}cm "
        f"wx={pose_cm['ee.wx']:+.4f} wy={pose_cm['ee.wy']:+.4f} wz={pose_cm['ee.wz']:+.4f} "
        f"gripper={pose_cm['ee.gripper_pos']:+.2f}"
    )


def print_target_error(target_pose: dict[str, float], reached_pose: dict[str, float]) -> None:
    dx = meters_to_cm(reached_pose["ee.x"] - target_pose["ee.x"])
    dy = meters_to_cm(reached_pose["ee.y"] - target_pose["ee.y"])
    dz = meters_to_cm(reached_pose["ee.z"] - target_pose["ee.z"])
    dpos = (dx**2 + dy**2 + dz**2) ** 0.5
    print(f"Target error: dx={dx:+.2f}cm dy={dy:+.2f}cm dz={dz:+.2f}cm |pos|={dpos:.2f}cm")


def position_error_norm(target_pose: dict[str, float], reached_pose: dict[str, float]) -> float:
    dx = reached_pose["ee.x"] - target_pose["ee.x"]
    dy = reached_pose["ee.y"] - target_pose["ee.y"]
    dz = reached_pose["ee.z"] - target_pose["ee.z"]
    return float((dx**2 + dy**2 + dz**2) ** 0.5)


def orientation_error_norm(target_pose: dict[str, float], reached_pose: dict[str, float]) -> float:
    dwx = float(reached_pose["ee.wx"]) - float(target_pose["ee.wx"])
    dwy = float(reached_pose["ee.wy"]) - float(target_pose["ee.wy"])
    dwz = float(reached_pose["ee.wz"]) - float(target_pose["ee.wz"])
    return float((dwx**2 + dwy**2 + dwz**2) ** 0.5)


def gripper_error_abs(target_pose: dict[str, float], reached_pose: dict[str, float]) -> float:
    return abs(float(reached_pose["ee.gripper_pos"]) - float(target_pose["ee.gripper_pos"]))


def target_reached(target_pose: dict[str, float], reached_pose: dict[str, float]) -> bool:
    if position_error_norm(target_pose, reached_pose) > POSITION_CONVERGENCE_M:
        return False
    if USE_TARGET_ORIENTATION and orientation_error_norm(target_pose, reached_pose) > ORIENTATION_CONVERGENCE_RAD:
        return False
    if gripper_error_abs(target_pose, reached_pose) > GRIPPER_CONVERGENCE:
        return False
    return True


def advance_towards_target(
    current_pose: dict[str, float],
    target_pose: dict[str, float],
) -> dict[str, float]:
    step_pose = {"name": target_pose.get("name", "target")}

    dx = float(target_pose["ee.x"]) - float(current_pose["ee.x"])
    dy = float(target_pose["ee.y"]) - float(current_pose["ee.y"])
    dz = float(target_pose["ee.z"]) - float(current_pose["ee.z"])
    distance = (dx**2 + dy**2 + dz**2) ** 0.5
    alpha = 1.0 if distance <= MAX_COMMAND_STEP_M or distance == 0.0 else MAX_COMMAND_STEP_M / distance

    step_pose["ee.x"] = float(current_pose["ee.x"]) + dx * alpha
    step_pose["ee.y"] = float(current_pose["ee.y"]) + dy * alpha
    step_pose["ee.z"] = float(current_pose["ee.z"]) + dz * alpha

    if USE_TARGET_ORIENTATION:
        for key in ORIENTATION_KEYS:
            step_pose[key] = float(current_pose[key]) + (
                float(target_pose[key]) - float(current_pose[key])
            ) * alpha
    else:
        for key in ORIENTATION_KEYS:
            step_pose[key] = float(current_pose[key])

    gripper_delta = float(target_pose["ee.gripper_pos"]) - float(current_pose["ee.gripper_pos"])
    step_pose["ee.gripper_pos"] = float(current_pose["ee.gripper_pos"]) + clip_delta(
        gripper_delta, MAX_GRIPPER_STEP
    )
    return step_pose


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
        resolved_target_pose = clip_target_pose_to_bounds(resolve_target_pose(start_pose, TARGET_POSE))

        print(f"\nMoving to target: {resolved_target_pose.get('name', 'target')}")
        print_ee_pose("Target EE :", resolved_target_pose)

        best_error = float("inf")
        stale_steps = 0
        for step in range(1, MAX_CONTROL_STEPS + 1):
            robot_obs = robot.get_observation()
            current_ee = fk_processor(robot_obs.copy())
            err = position_error_norm(resolved_target_pose, current_ee)
            if step == 1 or step % FPS == 0:
                if USE_TARGET_ORIENTATION:
                    orientation_err = orientation_error_norm(resolved_target_pose, current_ee)
                    print(
                        f"Step {step:03d}: position error {meters_to_cm(err):.2f}cm "
                        f"orientation error {orientation_err:.3f}rad "
                        f"gripper error {gripper_error_abs(resolved_target_pose, current_ee):.2f}"
                    )
                else:
                    print(f"Step {step:03d}: position error {meters_to_cm(err):.2f}cm")
            if target_reached(resolved_target_pose, current_ee):
                break

            if err < best_error - MIN_ERROR_IMPROVEMENT_M:
                best_error = err
                stale_steps = 0
            else:
                stale_steps += 1
            if stale_steps >= NO_IMPROVEMENT_STEPS:
                print(
                    "Stopping because measured EE position is no longer improving "
                    f"(current={meters_to_cm(err):.2f}cm best={meters_to_cm(best_error):.2f}cm)."
                )
                break

            ee_action = advance_towards_target(current_ee, resolved_target_pose)
            joint_action = ik_processor((ee_action, robot_obs))
            robot.send_action(joint_action)
            precise_sleep(1.0 / FPS)

        precise_sleep(HOLD_TIME_S)
        robot_obs = robot.get_observation()
        current_ee = fk_processor(robot_obs.copy())
        print_ee_pose("Reached EE:", current_ee)
        print_target_error(resolved_target_pose, current_ee)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
