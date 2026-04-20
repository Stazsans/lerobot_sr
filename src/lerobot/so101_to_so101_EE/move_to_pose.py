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

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline, TransitionKey
from lerobot.processor.converters import (
    identity_transition,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsRLStep,
    IKJointPreferences,
    _joint_delta_deg,
    _ordered_joint_positions,
    derive_ik_joint_preferences_from_robot,
)
from lerobot.utils.rotation import Rotation
from lerobot.utils.robot_utils import precise_sleep

# ========== 用户配置 ==========
FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "so101_follower"
REPO_ROOT = Path(__file__).resolve().parents[3]
URDF_PATH = REPO_ROOT / "third_party" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
TARGET_FRAME_NAME = "gripper_frame_link"
IK_DIAGNOSTICS_LOG_DIR = REPO_ROOT / ".tmp" / "move_to_pose_logs"
ENABLE_IK_DIAGNOSTICS_JSONL = True
ENABLE_IK_DIAGNOSTICS_RERUN = False
IGNORE_ORIENTATION_IN_IK = True

FPS = 10
MAX_CONTROL_STEPS = 240
HOLD_TIME_S = 0.05
POSITION_CONVERGENCE_CM = 1.0
ORIENTATION_CONVERGENCE_RAD = 0.10
GRIPPER_CONVERGENCE = 0.5
NO_IMPROVEMENT_STEPS = 80
MIN_ERROR_IMPROVEMENT_CM = 0.05
MIN_PREDICTED_IMPROVEMENT_CM = 0.10
MAX_JOINT_TRACKING_ERROR_WARN_DEG = 10.0
MAX_JOINT_TRACKING_ERROR_HOLD_DEG = 12.0
MAX_JOINT_TRACKING_ERROR_RESUME_DEG = 6.0
MAX_TRACKING_HOLD_STEPS = 3
IK_STEP_RETRY_SHRINK = 0.5
MAX_IK_STEP_RETRIES = 4
MAX_DIRECT_CLAMP_OVER_LIMIT_DEG = 20.0

EE_BOUNDS_MIN_CM = [-31.86, -39.73, -22.72]
EE_BOUNDS_MAX_CM = [47.00, 39.75, 52.10]
MAX_COMMAND_STEP_CM = 4.0
MAX_EE_SAFETY_STEP_CM = 5.0
MAX_GRIPPER_STEP = 5.0
MAX_JOINT_DELTA_DEG = {
    "shoulder_pan": 16.0,
    "shoulder_lift": 12.0,
    "elbow_flex": 14.0,
    "wrist_flex": 12.0,
    "wrist_roll": 18.0,
}
IK_POSITION_TOLERANCE_CM = 18.0
IK_ORIENTATION_TOLERANCE_RAD = 3.2 if IGNORE_ORIENTATION_IN_IK else 1.8
USE_TARGET_ORIENTATION = False
POSE_KEYS = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"]
POSITION_KEYS = ["ee.x", "ee.y", "ee.z"]
ORIENTATION_KEYS = ["ee.wx", "ee.wy", "ee.wz"]
CM_PER_M = 100.0
POSITION_CONVERGENCE_M = POSITION_CONVERGENCE_CM / CM_PER_M
MIN_ERROR_IMPROVEMENT_M = MIN_ERROR_IMPROVEMENT_CM / CM_PER_M
MIN_PREDICTED_IMPROVEMENT_M = MIN_PREDICTED_IMPROVEMENT_CM / CM_PER_M
EE_BOUNDS_MIN = [value / CM_PER_M for value in EE_BOUNDS_MIN_CM]
EE_BOUNDS_MAX = [value / CM_PER_M for value in EE_BOUNDS_MAX_CM]
MAX_COMMAND_STEP_M = MAX_COMMAND_STEP_CM / CM_PER_M
MAX_EE_SAFETY_STEP_M = MAX_EE_SAFETY_STEP_CM / CM_PER_M
IK_POSITION_TOLERANCE_M = IK_POSITION_TOLERANCE_CM / CM_PER_M

TARGET_POSE = {
    "name": "manual_target",
    "ee.x": 35,
    "ee.y": 5,
    "ee.z": 30,
    "delta.ee.wx": 0,
    "delta.ee.wy": 0,
    "delta.ee.wz": 0,
    "ee.gripper_pos": 30,
}
# ==============================


class PositionOnlyRobotKinematics:
    def __init__(self, base_kinematics: RobotKinematics):
        self._base_kinematics = base_kinematics

    def forward_kinematics(self, joint_pos_deg):
        return self._base_kinematics.forward_kinematics(joint_pos_deg)

    def inverse_kinematics(self, current_joint_pos, desired_ee_pose):
        return self._base_kinematics.inverse_kinematics(
            current_joint_pos,
            desired_ee_pose,
            position_weight=1.0,
            orientation_weight=0.0,
        )


def build_processors(
    robot: SO101Follower, kinematics: RobotKinematics
) -> tuple[RobotProcessorPipeline, RobotProcessorPipeline, IKJointPreferences]:
    ik_kinematics = PositionOnlyRobotKinematics(kinematics) if IGNORE_ORIENTATION_IN_IK else kinematics
    ik_preferences = derive_ik_joint_preferences_from_robot(
        robot,
        motor_names=list(robot.bus.motors.keys()),
    )
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

    ik_transition_processor = RobotProcessorPipeline(
        steps=[
            EEBoundsAndSafety(
                end_effector_bounds={"min": EE_BOUNDS_MIN, "max": EE_BOUNDS_MAX},
                max_ee_step_m=MAX_EE_SAFETY_STEP_M,
                max_orientation_step_rad=0.8 if USE_TARGET_ORIENTATION else None,
            ),
            InverseKinematicsRLStep(
                kinematics=ik_kinematics,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=True,
                max_joint_delta_deg=MAX_JOINT_DELTA_DEG,
                continuous_joint_names=ik_preferences.continuous_joint_names,
                joint_position_limits_deg=ik_preferences.joint_position_limits_deg,
                position_tolerance_m=IK_POSITION_TOLERANCE_M,
                orientation_tolerance_rad=IK_ORIENTATION_TOLERANCE_RAD,
                prioritize_orientation=not IGNORE_ORIENTATION_IN_IK,
                fallback_to_current_joints_on_invalid=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=identity_transition,
    )

    return fk_processor, ik_transition_processor, ik_preferences


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
    target = float(target_pose.get("ee.gripper_pos", reached_pose.get("ee.gripper_pos", 0.0)))
    reached = float(reached_pose.get("ee.gripper_pos", target_pose.get("ee.gripper_pos", 0.0)))
    return abs(reached - target)


def target_reached(target_pose: dict[str, float], reached_pose: dict[str, float]) -> bool:
    if position_error_norm(target_pose, reached_pose) > POSITION_CONVERGENCE_M:
        return False
    if USE_TARGET_ORIENTATION and orientation_error_norm(target_pose, reached_pose) > ORIENTATION_CONVERGENCE_RAD:
        return False
    if gripper_error_abs(target_pose, reached_pose) > GRIPPER_CONVERGENCE:
        return False
    return True


def pose_error_score(target_pose: dict[str, float], reached_pose: dict[str, float]) -> float:
    score = position_error_norm(target_pose, reached_pose)
    if USE_TARGET_ORIENTATION:
        score += orientation_error_norm(target_pose, reached_pose)
    score += gripper_error_abs(target_pose, reached_pose) / 100.0
    return float(score)


def predicted_ee_pose_from_joint_action(
    joint_action: RobotAction,
    kinematics: RobotKinematics,
    motor_names: list[str],
) -> dict[str, float]:
    q = [float(joint_action[f"{name}.pos"]) for name in motor_names]
    pose = np.asarray(kinematics.forward_kinematics(q), dtype=float)
    twist = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
    return {
        "ee.x": float(pose[0, 3]),
        "ee.y": float(pose[1, 3]),
        "ee.z": float(pose[2, 3]),
        "ee.wx": float(twist[0]),
        "ee.wy": float(twist[1]),
        "ee.wz": float(twist[2]),
        "ee.gripper_pos": float(joint_action.get("gripper.pos", joint_action.get("ee.gripper_pos", 0.0))),
    }


def predicted_motion_is_progress(
    *,
    current_pose: dict[str, float],
    predicted_pose: dict[str, float],
    target_pose: dict[str, float],
) -> bool:
    current_position_error = position_error_norm(target_pose, current_pose)
    predicted_position_error = position_error_norm(target_pose, predicted_pose)
    if predicted_position_error >= current_position_error - MIN_PREDICTED_IMPROVEMENT_M:
        return False

    current_score = pose_error_score(target_pose, current_pose)
    predicted_score = pose_error_score(target_pose, predicted_pose)
    return predicted_score < current_score


def ik_best_candidate_is_progress(
    *,
    complementary_data: dict[str, Any],
    current_pose: dict[str, float],
    target_pose: dict[str, float],
) -> bool | None:
    diagnostics = complementary_data.get("IK_diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        return None

    best = diagnostics[0]
    if bool(best.get("is_unsafe", False)) or bool(best.get("pose_invalid", False)):
        return False

    current_position_error = position_error_norm(target_pose, current_pose)
    predicted_position_error = float(best.get("position_error_m", current_position_error))
    return predicted_position_error < current_position_error - MIN_PREDICTED_IMPROVEMENT_M


def format_ik_candidate_summary(candidate: dict[str, float | bool]) -> str:
    return (
        f"pos={meters_to_cm(float(candidate['position_error_m'])):.2f}cm "
        f"ori={float(candidate['orientation_error_rad']):.3f}rad "
        f"max_dq={float(candidate['max_joint_delta_deg']):.1f}deg "
        f"over_dq={float(candidate['max_over_limit_deg']):.1f}deg "
        f"limit_margin={float(candidate['min_joint_limit_margin_deg']):.1f}deg "
        f"limit_violation={float(candidate['max_joint_limit_violation_deg']):.1f}deg "
        f"unsafe={bool(candidate['is_unsafe'])} "
        f"pose_invalid={bool(candidate['pose_invalid'])}"
    )


def print_ik_diagnostics(complementary_data: dict[str, object], *, step: int) -> None:
    diagnostics = complementary_data.get("IK_diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        return

    best = diagnostics[0]
    print(f"IK step {step:03d} best: {format_ik_candidate_summary(best)}")
    if len(diagnostics) > 1:
        print(f"IK step {step:03d} alt : {format_ik_candidate_summary(diagnostics[1])}")


def create_ik_diagnostics_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return IK_DIAGNOSTICS_LOG_DIR / f"move_to_pose_ik_diagnostics_{timestamp}.jsonl"


def build_ik_diagnostics_record(
    *,
    step: int,
    current_pose: dict[str, float],
    target_pose: dict[str, float],
    predicted_pose: dict[str, float],
    complementary_data: dict[str, Any],
    sent_joint_action: dict[str, float] | None = None,
    observed_joint_positions: dict[str, float] | None = None,
    joint_tracking_error_deg: dict[str, float] | None = None,
) -> dict[str, Any]:
    diagnostics = complementary_data.get("IK_diagnostics", [])
    return {
        "step": int(step),
        "current_pose": {k: float(v) for k, v in current_pose.items() if k.startswith("ee.")},
        "target_pose": {k: float(v) for k, v in target_pose.items() if k.startswith("ee.")},
        "predicted_pose": {k: float(v) for k, v in predicted_pose.items() if k.startswith("ee.")},
        "position_error_m": position_error_norm(target_pose, current_pose),
        "orientation_error_rad": orientation_error_norm(target_pose, current_pose)
        if USE_TARGET_ORIENTATION
        else 0.0,
        "gripper_error_abs": gripper_error_abs(target_pose, current_pose),
        "pose_score": pose_error_score(target_pose, current_pose),
        "ik_solution": [float(v) for v in complementary_data.get("IK_solution", [])],
        "ik_diagnostics": diagnostics if isinstance(diagnostics, list) else [],
        "sent_joint_action": sent_joint_action or {},
        "observed_joint_positions": observed_joint_positions or {},
        "joint_tracking_error_deg": joint_tracking_error_deg or {},
    }


def append_ik_diagnostics_jsonl(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def log_ik_diagnostics_rerun(record: dict[str, Any]) -> None:
    try:
        import rerun as rr
    except ImportError:
        return

    rr.set_time_sequence("step", record["step"])
    rr.log("ik/current/position_error_m", rr.Scalars(float(record["position_error_m"])))
    rr.log("ik/current/orientation_error_rad", rr.Scalars(float(record["orientation_error_rad"])))
    rr.log("ik/current/gripper_error_abs", rr.Scalars(float(record["gripper_error_abs"])))
    rr.log("ik/current/pose_score", rr.Scalars(float(record["pose_score"])))

    diagnostics = record.get("ik_diagnostics", [])
    if isinstance(diagnostics, list) and diagnostics:
        best = diagnostics[0]
        for key in [
            "position_error_m",
            "orientation_error_rad",
            "max_joint_delta_deg",
            "max_over_limit_deg",
            "min_joint_limit_margin_deg",
            "max_joint_limit_violation_deg",
        ]:
            if key in best:
                rr.log(f"ik/best/{key}", rr.Scalars(float(best[key])))


def joint_hold_action_from_observation(
    observation: RobotObservation,
    motor_names: list[str],
) -> RobotAction:
    return {f"{name}.pos": float(observation[f"{name}.pos"]) for name in motor_names}


def send_hold_action(robot: SO101Follower, observation: RobotObservation) -> None:
    robot.send_action(joint_hold_action_from_observation(observation, list(robot.bus.motors.keys())))


def compute_joint_tracking_error(
    *,
    sent_joint_action: dict[str, float] | None,
    observation: RobotObservation,
    motor_names: list[str],
    continuous_joint_names: tuple[str, ...] = (),
) -> dict[str, float]:
    if not sent_joint_action:
        return {}

    target = _ordered_joint_positions(sent_joint_action, motor_names)
    observed = _ordered_joint_positions(observation, motor_names)
    delta = np.abs(_joint_delta_deg(target, observed, motor_names, continuous_joint_names))
    return {name: float(delta[idx]) for idx, name in enumerate(motor_names)}


def clamp_joint_action_to_delta_limits(
    *,
    joint_action: RobotAction,
    observation: RobotObservation,
    motor_names: list[str],
    max_joint_delta_deg: float | dict[str, float],
    continuous_joint_names: tuple[str, ...] = (),
    extra_scale: float = 1.0,
) -> RobotAction:
    clamped_action = dict(joint_action)
    current_joint_pos = _ordered_joint_positions(observation, motor_names)
    target_joint_pos = _ordered_joint_positions(joint_action, motor_names)
    delta_limits = {
        name: float(max_joint_delta_deg.get(name, np.inf))
        for name in motor_names
    } if isinstance(max_joint_delta_deg, dict) else {name: float(max_joint_delta_deg) for name in motor_names}

    joint_delta = _joint_delta_deg(target_joint_pos, current_joint_pos, motor_names, continuous_joint_names)
    required_scales = []
    for idx, motor_name in enumerate(motor_names):
        if motor_name == "gripper":
            continue
        limit = delta_limits.get(motor_name, float("inf"))
        delta_abs = abs(float(joint_delta[idx]))
        if not np.isfinite(limit) or limit <= 0.0 or delta_abs <= 0.0:
            continue
        required_scales.append(float(limit / delta_abs))

    uniform_scale = 1.0
    if required_scales:
        uniform_scale = min(1.0, min(required_scales))
    uniform_scale *= float(extra_scale)
    uniform_scale = float(np.clip(uniform_scale, 0.0, 1.0))

    for idx, motor_name in enumerate(motor_names):
        if motor_name == "gripper":
            continue
        clamped_delta = float(joint_delta[idx] * uniform_scale)
        clamped_action[f"{motor_name}.pos"] = float(current_joint_pos[idx] + clamped_delta)

    return clamped_action


def joint_action_from_joint_positions(
    *,
    joint_positions_deg: np.ndarray | list[float],
    observation: RobotObservation,
    motor_names: list[str],
) -> RobotAction:
    joint_positions = np.asarray(joint_positions_deg, dtype=float)
    if joint_positions.shape != (len(motor_names),):
        raise ValueError(
            f"Expected {len(motor_names)} joint positions, got shape {joint_positions.shape}."
        )

    action: RobotAction = {}
    for idx, motor_name in enumerate(motor_names):
        if motor_name == "gripper":
            action["gripper.pos"] = float(observation.get("gripper.pos", joint_positions[idx]))
            continue
        action[f"{motor_name}.pos"] = float(joint_positions[idx])

    if "gripper.pos" not in action and "gripper.pos" in observation:
        action["gripper.pos"] = float(observation["gripper.pos"])
    return action


def find_progressive_clamped_joint_action(
    *,
    joint_action: RobotAction,
    raw_solution: np.ndarray | list[float] | None,
    observation: RobotObservation,
    motor_names: list[str],
    continuous_joint_names: tuple[str, ...],
    max_joint_delta_deg: float | dict[str, float],
    current_pose: dict[str, float],
    target_pose: dict[str, float],
    kinematics: RobotKinematics,
) -> tuple[RobotAction, dict[str, float], float] | None:
    source_joint_action = joint_action
    if raw_solution is not None:
        source_joint_action = joint_action_from_joint_positions(
            joint_positions_deg=raw_solution,
            observation=observation,
            motor_names=motor_names,
        )

    for clamp_scale in (1.0, 0.5, 0.25, 0.125):
        candidate_joint_action = clamp_joint_action_to_delta_limits(
            joint_action=source_joint_action,
            observation=observation,
            motor_names=motor_names,
            max_joint_delta_deg=max_joint_delta_deg,
            continuous_joint_names=continuous_joint_names,
            extra_scale=clamp_scale,
        )
        candidate_predicted_ee = predicted_ee_pose_from_joint_action(candidate_joint_action, kinematics, motor_names)
        if predicted_motion_is_progress(
            current_pose=current_pose,
            predicted_pose=candidate_predicted_ee,
            target_pose=target_pose,
        ):
            return candidate_joint_action, candidate_predicted_ee, clamp_scale

    return None


def print_joint_tracking_error(joint_tracking_error_deg: dict[str, float]) -> None:
    if not joint_tracking_error_deg:
        return

    worst_joint, worst_error = max(joint_tracking_error_deg.items(), key=lambda item: item[1])
    if worst_error < MAX_JOINT_TRACKING_ERROR_WARN_DEG:
        return

    summary = ", ".join(
        f"{name}={error:.1f}deg"
        for name, error in sorted(joint_tracking_error_deg.items(), key=lambda item: item[1], reverse=True)[:3]
    )
    print(f"Joint tracking warning: max={worst_joint}:{worst_error:.1f}deg | {summary}")


def max_joint_tracking_error(joint_tracking_error_deg: dict[str, float]) -> float:
    if not joint_tracking_error_deg:
        return 0.0
    return max(joint_tracking_error_deg.values())


def advance_towards_target(
    current_pose: dict[str, float],
    target_pose: dict[str, float],
    *,
    position_scale: float = 1.0,
    gripper_scale: float = 1.0,
) -> dict[str, float]:
    step_pose = {"name": target_pose.get("name", "target")}

    dx = float(target_pose["ee.x"]) - float(current_pose["ee.x"])
    dy = float(target_pose["ee.y"]) - float(current_pose["ee.y"])
    dz = float(target_pose["ee.z"]) - float(current_pose["ee.z"])
    distance = (dx**2 + dy**2 + dz**2) ** 0.5
    alpha = 1.0 if distance <= MAX_COMMAND_STEP_M or distance == 0.0 else MAX_COMMAND_STEP_M / distance

    alpha *= float(position_scale)
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
        gripper_delta * float(gripper_scale), MAX_GRIPPER_STEP
    )
    return step_pose


def best_ik_diagnostic(complementary_data: dict[str, Any]) -> dict[str, Any] | None:
    diagnostics = complementary_data.get("IK_diagnostics")
    if isinstance(diagnostics, list) and diagnostics:
        return diagnostics[0]
    return None


def ik_candidate_is_clampable(best_diag: dict[str, Any] | None) -> bool:
    if best_diag is None:
        return False
    if not bool(best_diag.get("is_unsafe", False)):
        return False
    if bool(best_diag.get("pose_invalid", False)):
        return False
    if float(best_diag.get("max_joint_limit_violation_deg", 0.0)) > 0.0:
        return False
    return float(best_diag.get("max_over_limit_deg", 0.0)) > 0.0


def ik_candidate_can_use_direct_clamp(best_diag: dict[str, Any] | None) -> bool:
    return ik_candidate_is_clampable(best_diag) and float(best_diag.get("max_over_limit_deg", 0.0)) <= float(
        MAX_DIRECT_CLAMP_OVER_LIMIT_DEG
    )


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
    fk_processor, ik_transition_processor, ik_preferences = build_processors(robot, kinematics)
    motor_names = list(robot.bus.motors.keys())
    diagnostics_log_path = create_ik_diagnostics_log_path()
    last_sent_joint_action: dict[str, float] | None = None
    tracking_hold_steps = 0

    robot.connect()

    try:
        robot_obs = robot.get_observation()
        current_ee = fk_processor(robot_obs.copy())
        print_ee_pose("Current EE:", current_ee)

        start_pose = {key: float(current_ee[key]) for key in current_ee if key.startswith("ee.")}
        resolved_target_pose = clip_target_pose_to_bounds(resolve_target_pose(start_pose, TARGET_POSE))

        print(f"\nMoving to target: {resolved_target_pose.get('name', 'target')}")
        print_ee_pose("Target EE :", resolved_target_pose)
        if ENABLE_IK_DIAGNOSTICS_JSONL:
            diagnostics_log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"IK diagnostics log: {diagnostics_log_path}")

        best_score = float("inf")
        best_obs = robot_obs.copy()
        stale_steps = 0
        for step in range(1, MAX_CONTROL_STEPS + 1):
            robot_obs = robot.get_observation()
            joint_tracking_error_deg = compute_joint_tracking_error(
                sent_joint_action=last_sent_joint_action,
                observation=robot_obs,
                motor_names=motor_names,
                continuous_joint_names=ik_preferences.continuous_joint_names,
            )
            print_joint_tracking_error(joint_tracking_error_deg)
            tracking_error_max = max_joint_tracking_error(joint_tracking_error_deg)
            current_ee = fk_processor(robot_obs.copy())
            err = position_error_norm(resolved_target_pose, current_ee)
            orientation_err = (
                orientation_error_norm(resolved_target_pose, current_ee) if USE_TARGET_ORIENTATION else 0.0
            )
            gripper_err = gripper_error_abs(resolved_target_pose, current_ee)
            pose_score = pose_error_score(resolved_target_pose, current_ee)
            if step == 1 or step % FPS == 0:
                if USE_TARGET_ORIENTATION:
                    print(
                        f"Step {step:03d}: position error {meters_to_cm(err):.2f}cm "
                        f"orientation error {orientation_err:.3f}rad "
                        f"gripper error {gripper_err:.2f}"
                    )
                else:
                    print(f"Step {step:03d}: position error {meters_to_cm(err):.2f}cm")
            if target_reached(resolved_target_pose, current_ee):
                break

            if last_sent_joint_action is not None and tracking_error_max > MAX_JOINT_TRACKING_ERROR_HOLD_DEG:
                tracking_hold_steps += 1
                print(
                    f"Holding for joint settling: max tracking error {tracking_error_max:.1f}deg "
                    f"(hold {tracking_hold_steps}/{MAX_TRACKING_HOLD_STEPS})"
                )
                if tracking_hold_steps <= MAX_TRACKING_HOLD_STEPS:
                    precise_sleep(1.0 / FPS)
                    continue
            elif tracking_error_max <= MAX_JOINT_TRACKING_ERROR_RESUME_DEG:
                tracking_hold_steps = 0

            if pose_score < best_score - MIN_ERROR_IMPROVEMENT_M:
                best_score = pose_score
                best_obs = robot_obs.copy()
                stale_steps = 0
            else:
                stale_steps += 1
            if stale_steps >= NO_IMPROVEMENT_STEPS:
                print(
                    "Stopping because measured EE pose is no longer improving "
                    f"(pos={meters_to_cm(err):.2f}cm ori={orientation_err:.3f}rad grip={gripper_err:.2f})."
                )
                print("Returning to the best measured joint pose.")
                send_hold_action(robot, best_obs)
                break

            shrink_scale = 1.0
            ik_transition = None
            ee_action = None
            clamped_candidate: tuple[RobotAction, dict[str, float], float] | None = None
            for retry_idx in range(MAX_IK_STEP_RETRIES):
                ee_action = advance_towards_target(
                    current_ee,
                    resolved_target_pose,
                    position_scale=shrink_scale,
                    gripper_scale=shrink_scale,
                )
                ik_transition = ik_transition_processor((ee_action, robot_obs))
                best_diag = best_ik_diagnostic(ik_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
                if best_diag is None or not bool(best_diag.get("is_unsafe", False)):
                    break
                if ik_candidate_can_use_direct_clamp(best_diag):
                    complementary_data = ik_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
                    clamped_candidate = find_progressive_clamped_joint_action(
                        joint_action=ik_transition[TransitionKey.ACTION],
                        raw_solution=complementary_data.get("IK_raw_solution"),
                        observation=robot_obs,
                        motor_names=motor_names,
                        continuous_joint_names=ik_preferences.continuous_joint_names,
                        max_joint_delta_deg=MAX_JOINT_DELTA_DEG,
                        current_pose=current_ee,
                        target_pose=resolved_target_pose,
                        kinematics=kinematics,
                    )
                    if clamped_candidate is not None:
                        break
                print(
                    f"Retrying IK with smaller EE step: scale={shrink_scale * IK_STEP_RETRY_SHRINK:.3f} "
                    f"(retry {retry_idx + 1}/{MAX_IK_STEP_RETRIES - 1})"
                )
                shrink_scale *= IK_STEP_RETRY_SHRINK

            if ik_transition is None or ee_action is None:
                raise RuntimeError("Failed to build IK transition")
            complementary_data = ik_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
            print_ik_diagnostics(complementary_data, step=step)
            joint_action = ik_transition[TransitionKey.ACTION]
            best_diag = best_ik_diagnostic(complementary_data)
            used_clamped_joint_action = False
            predicted_ee = None
            if ik_candidate_can_use_direct_clamp(best_diag):
                print("Clamping joint action to delta limits instead of rejecting IK candidate.")
                selected_joint_action = None
                selected_predicted_ee = None
                selected_scale = None
                if clamped_candidate is not None:
                    selected_joint_action, selected_predicted_ee, selected_scale = clamped_candidate
                else:
                    fallback_clamped_candidate = find_progressive_clamped_joint_action(
                        joint_action=joint_action,
                        raw_solution=complementary_data.get("IK_raw_solution"),
                        observation=robot_obs,
                        motor_names=motor_names,
                        continuous_joint_names=ik_preferences.continuous_joint_names,
                        max_joint_delta_deg=MAX_JOINT_DELTA_DEG,
                        current_pose=current_ee,
                        target_pose=resolved_target_pose,
                        kinematics=kinematics,
                    )
                    if fallback_clamped_candidate is not None:
                        selected_joint_action, selected_predicted_ee, selected_scale = fallback_clamped_candidate

                if selected_joint_action is None or selected_predicted_ee is None:
                    print("Skipping this IK step because no clamped joint action improves the target pose.")
                    precise_sleep(1.0 / FPS)
                    continue

                if selected_scale is not None and selected_scale < 1.0:
                    print(f"Using smaller clamped joint step scale={selected_scale:.3f} to preserve progress.")
                joint_action = selected_joint_action
                predicted_ee = selected_predicted_ee
                used_clamped_joint_action = True
            if predicted_ee is None:
                predicted_ee = predicted_ee_pose_from_joint_action(joint_action, kinematics, motor_names)
            diagnostics_record = build_ik_diagnostics_record(
                step=step,
                current_pose=current_ee,
                target_pose=resolved_target_pose,
                predicted_pose=predicted_ee,
                complementary_data=complementary_data,
                sent_joint_action=last_sent_joint_action,
                observed_joint_positions={
                    f"{name}.pos": float(robot_obs[f"{name}.pos"]) for name in motor_names if f"{name}.pos" in robot_obs
                },
                joint_tracking_error_deg=joint_tracking_error_deg,
            )
            if ENABLE_IK_DIAGNOSTICS_JSONL:
                append_ik_diagnostics_jsonl(diagnostics_log_path, diagnostics_record)
            if ENABLE_IK_DIAGNOSTICS_RERUN:
                log_ik_diagnostics_rerun(diagnostics_record)
            ik_progress = ik_best_candidate_is_progress(
                complementary_data=complementary_data,
                current_pose=current_ee,
                target_pose=resolved_target_pose,
            )
            if used_clamped_joint_action:
                ik_progress = None
            predicted_progress = predicted_motion_is_progress(
                current_pose=current_ee,
                predicted_pose=predicted_ee,
                target_pose=resolved_target_pose,
            )
            if ik_progress is False or (ik_progress is None and not predicted_progress):
                predicted_error = position_error_norm(resolved_target_pose, predicted_ee)
                predicted_orientation_error = (
                    orientation_error_norm(resolved_target_pose, predicted_ee) if USE_TARGET_ORIENTATION else 0.0
                )
                predicted_gripper_error = gripper_error_abs(resolved_target_pose, predicted_ee)
                print(
                    "Stopping because IK prediction does not improve the target pose "
                    f"(predicted={meters_to_cm(predicted_error):.2f}cm "
                    f"current={meters_to_cm(err):.2f}cm "
                    f"pred_ori={predicted_orientation_error:.3f}rad current_ori={orientation_err:.3f}rad "
                    f"pred_grip={predicted_gripper_error:.2f} current_grip={gripper_err:.2f})."
                )
                print_ik_diagnostics(complementary_data, step=step)
                print("Returning to the best measured joint pose.")
                send_hold_action(robot, best_obs)
                break
            last_sent_joint_action = robot.send_action(joint_action)
            tracking_hold_steps = 0
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
