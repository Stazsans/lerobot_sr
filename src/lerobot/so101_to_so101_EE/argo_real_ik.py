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
Run Argo-Robot/controls URDF IK on the real SO101 follower arm.

This script intentionally uses the code under `third_party/argo-controls/scripts`
instead of the LeRobot `placo` IK pipeline. By default it is a dry run: it reads
the real robot state, computes Argo FK/IK, prints diagnostics, and does not send
motor commands. Pass `--execute` to stream the interpolated joint target.
"""

from __future__ import annotations

import argparse
import collections
import collections.abc
import fractions
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.rotation import Rotation
from lerobot.utils.robot_utils import precise_sleep

FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "so101_follower"

REPO_ROOT = Path(__file__).resolve().parents[3]
ARGO_ROOT = REPO_ROOT / "third_party" / "argo-controls"
ARGO_URDF_PATH = REPO_ROOT / "third_party" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
ARGO_TARGET_LINK_NAME = "gripper_frame_link"

ROBOT_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DEFAULT_TARGET_DELTA_CM = [-3.0, 0.0, 3.0]
DEFAULT_GRIPPER_POS = 30.0
DEFAULT_USE_ORIENTATION = False
DEFAULT_FPS = 30
DEFAULT_INTERPOLATION_STEPS = 24
MAX_JOINT_STEP_DEG = 8.0
IK_CANDIDATE_LIMIT = 5
POSITION_CONVERGENCE_CM = 1.0
ORIENTATION_CONVERGENCE_RAD = 0.10
MIN_PROGRESS_CM = 0.2
MAX_STALE_STEPS = 8
TRACKING_SETTLE_DEG = 3.0
TRACKING_HEAVY_ERROR_DEG = 6.0
NEAR_TARGET_ERROR_CM = 5.0
VERY_NEAR_TARGET_ERROR_CM = 3.5
NEAR_TARGET_MIN_PROGRESS_CM = 0.02
NEAR_TARGET_MAX_STALE_STEPS = 20
NEAR_TARGET_REPLAN_INTERVAL = 5
NEAR_TARGET_MAX_JOINT_STEP_DEG = 4.0
NEAR_TARGET_TRACKING_MAX_JOINT_STEP_DEG = 2.5
NEAR_TARGET_HEAVY_TRACKING_MAX_JOINT_STEP_DEG = 1.5
VERY_NEAR_TARGET_MAX_JOINT_STEP_DEG = 2.0
VERY_NEAR_TARGET_TRACKING_MAX_JOINT_STEP_DEG = 1.0
DEFAULT_MAX_SERVO_CYCLES = 240
DEFAULT_REPLAN_INTERVAL = 40
DEFAULT_LOG_INTERVAL = 30
TARGET_HOLD_TIME_S = 3.0
DEFAULT_FIXED_GOAL_DIAGNOSTIC_SECONDS = 0.0
DEFAULT_FINAL_TRIM_DIAGNOSTIC = False
DEFAULT_APPLY_FINAL_TRIM = True
FINAL_TRIM_SETTLE_SECONDS = 1.5
FAR_TARGET_ERROR_CM = 12.0
FAR_TARGET_MAX_JOINT_STEP_DEG = 12.0
MID_TARGET_ERROR_CM = 6.0
MID_TARGET_MAX_JOINT_STEP_DEG = 10.0
MID_TARGET_TRACKING_MAX_JOINT_STEP_DEG = 6.0
REPLAN_TRACKING_GUARD_ERROR_CM = 8.0
NEAR_TARGET_GOAL_BLEND = 0.25
NEAR_TARGET_TRACKING_GOAL_BLEND = 0.15
MID_TARGET_GOAL_BLEND = 0.60
SERVO_HOLD_TRACKING_ERROR_DEG = 2.5
SERVO_HOLD_ERROR_CM = 6.0
VERY_NEAR_SERVO_HOLD_TRACKING_ERROR_DEG = 1.5
VERY_NEAR_REPLAN_TRACKING_ERROR_DEG = 1.5
MAX_TRACKING_WAIT_CYCLES = 12
MAX_SERVO_HOLD_CYCLES = 6
GRIPPER_LOWER_RAD = -0.174533
GRIPPER_UPPER_RAD = 1.74533


def import_argo_controls() -> tuple[Any, Any, Any, Any]:
    if not ARGO_ROOT.exists():
        raise FileNotFoundError(
            f"Argo controls repo not found at {ARGO_ROOT}. Clone https://github.com/Argo-Robot/controls "
            "into third_party/argo-controls first."
        )

    # urdfpy==0.0.22 pins networkx==2.2. That old networkx still imports names
    # removed from Python 3.10+, so patch them before Argo imports urdfpy.
    for name in ("Mapping", "MutableMapping", "Set", "Iterable"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))
    if not hasattr(fractions, "gcd"):
        fractions.gcd = math.gcd  # type: ignore[attr-defined]
    numpy_legacy_aliases = {
        "bool": np.bool_,
        "int": np.int64,
        "float": np.float64,
        "complex": np.complex128,
        "float_": np.float64,
        "complex_": np.complex128,
    }
    for name, value in numpy_legacy_aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)

    sys.path.insert(0, str(ARGO_ROOT))
    try:
        from scripts.kinematics import URDF_Kinematics
        from scripts.model import RobotModel, URDF_loader
        from scripts.utils import RobotUtils
    except ImportError as exc:
        raise ImportError(
            "Failed to import Argo controls. Install its runtime dependencies in your WSL env, for example:\n"
            "  python -m pip install scipy urdfpy==0.0.22 networkx==2.2\n\n"
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc
    return URDF_loader, RobotModel, URDF_Kinematics, RobotUtils


def gripper_percent_to_argo_rad(value: float) -> float:
    clamped = float(np.clip(value, 0.0, 100.0))
    return GRIPPER_LOWER_RAD + (GRIPPER_UPPER_RAD - GRIPPER_LOWER_RAD) * (clamped / 100.0)


def argo_rad_to_gripper_percent(value: float) -> float:
    percent = (float(value) - GRIPPER_LOWER_RAD) / (GRIPPER_UPPER_RAD - GRIPPER_LOWER_RAD) * 100.0
    return float(np.clip(percent, 0.0, 100.0))


def observation_to_argo_q_rad(
    observation: dict[str, float],
    argo_joint_names: list[str],
) -> np.ndarray:
    q = []
    for joint_name in argo_joint_names:
        if joint_name == "gripper":
            q.append(gripper_percent_to_argo_rad(float(observation["gripper.pos"])))
        else:
            q.append(np.deg2rad(float(observation[f"{joint_name}.pos"])))
    return np.asarray(q, dtype=float)


def argo_q_rad_to_robot_action(
    q_rad: np.ndarray,
    argo_joint_names: list[str],
    *,
    gripper_pos: float,
) -> dict[str, float]:
    action: dict[str, float] = {}
    for idx, joint_name in enumerate(argo_joint_names):
        if joint_name == "gripper":
            action["gripper.pos"] = float(gripper_pos)
        else:
            action[f"{joint_name}.pos"] = float(np.rad2deg(q_rad[idx]))
    return action


def robot_action_to_argo_q_rad(action: dict[str, float], argo_joint_names: list[str]) -> np.ndarray:
    q = []
    for joint_name in argo_joint_names:
        if joint_name == "gripper":
            q.append(gripper_percent_to_argo_rad(float(action["gripper.pos"])))
        else:
            q.append(np.deg2rad(float(action[f"{joint_name}.pos"])))
    return np.asarray(q, dtype=float)


def format_q_deg(q_rad: np.ndarray, argo_joint_names: list[str]) -> str:
    parts = []
    for idx, joint_name in enumerate(argo_joint_names):
        if joint_name == "gripper":
            parts.append(f"{joint_name}={argo_rad_to_gripper_percent(q_rad[idx]):+.2f}%")
        else:
            parts.append(f"{joint_name}={np.rad2deg(q_rad[idx]):+.2f}deg")
    return " ".join(parts)


def format_joint_tracking_error(
    *,
    commanded_q_rad: np.ndarray,
    measured_q_rad: np.ndarray,
    argo_joint_names: list[str],
) -> str:
    parts = []
    for idx, joint_name in enumerate(argo_joint_names):
        if joint_name == "gripper":
            error = argo_rad_to_gripper_percent(measured_q_rad[idx]) - argo_rad_to_gripper_percent(
                commanded_q_rad[idx]
            )
            parts.append(f"{joint_name}={error:+.2f}%")
        else:
            error = np.rad2deg(measured_q_rad[idx] - commanded_q_rad[idx])
            parts.append(f"{joint_name}={error:+.2f}deg")
    return " ".join(parts)


def max_arm_tracking_error_deg(
    *,
    commanded_q_rad: np.ndarray,
    measured_q_rad: np.ndarray,
    argo_joint_names: list[str],
) -> float:
    errors = []
    for idx, joint_name in enumerate(argo_joint_names):
        if joint_name == "gripper":
            continue
        errors.append(abs(float(np.rad2deg(measured_q_rad[idx] - commanded_q_rad[idx]))))
    return max(errors, default=0.0)


def arm_joint_delta_deg(
    *,
    q_from_rad: np.ndarray,
    q_to_rad: np.ndarray,
    argo_joint_names: list[str],
) -> float:
    deltas = []
    for idx, joint_name in enumerate(argo_joint_names):
        if joint_name == "gripper":
            continue
        deltas.append(abs(float(np.rad2deg(q_to_rad[idx] - q_from_rad[idx]))))
    return max(deltas, default=0.0)


def print_pose(prefix: str, pose: np.ndarray) -> None:
    x, y, z = pose[:3, 3] * 100.0
    wx, wy, wz = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
    print(f"{prefix} x={x:+.2f}cm y={y:+.2f}cm z={z:+.2f}cm wx={wx:+.4f} wy={wy:+.4f} wz={wz:+.4f}")


def position_error_cm(current_pose: np.ndarray, target_pose: np.ndarray) -> float:
    return float(np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3]) * 100.0)


def orientation_error_rad(current_pose: np.ndarray, target_pose: np.ndarray) -> float:
    rotation_error = target_pose[:3, :3] @ current_pose[:3, :3].T
    return float(np.linalg.norm(Rotation.from_matrix(rotation_error).as_rotvec()))


def pose_error_score_cm(current_pose: np.ndarray, target_pose: np.ndarray, *, use_orientation: bool) -> float:
    score = position_error_cm(current_pose, target_pose)
    if use_orientation:
        score += orientation_error_rad(current_pose, target_pose) * (
            POSITION_CONVERGENCE_CM / ORIENTATION_CONVERGENCE_RAD
        )
    return float(score)


def candidate_pose_error_score_cm(*, pos_err_m: float, ori_err_rad: float, use_orientation: bool) -> float:
    score = float(pos_err_m) * 100.0
    if use_orientation:
        score += float(ori_err_rad) * (POSITION_CONVERGENCE_CM / ORIENTATION_CONVERGENCE_RAD)
    return float(score)


def target_reached(current_pose: np.ndarray, target_pose: np.ndarray, *, use_orientation: bool) -> bool:
    if position_error_cm(current_pose, target_pose) > POSITION_CONVERGENCE_CM:
        return False
    if use_orientation and orientation_error_rad(current_pose, target_pose) > ORIENTATION_CONVERGENCE_RAD:
        return False
    return True


def target_requests_orientation(args: argparse.Namespace) -> bool:
    return bool(
        args.use_orientation or args.target_rotvec is not None or args.target_delta_rotvec is not None
    )


def build_target_pose(start_pose: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    target_pose = np.array(start_pose, dtype=float, copy=True)
    if args.target_x_cm is not None:
        target_pose[0, 3] = args.target_x_cm / 100.0
    if args.target_y_cm is not None:
        target_pose[1, 3] = args.target_y_cm / 100.0
    if args.target_z_cm is not None:
        target_pose[2, 3] = args.target_z_cm / 100.0
    if args.target_x_cm is None and args.target_y_cm is None and args.target_z_cm is None:
        target_pose[:3, 3] += np.asarray(args.target_delta_cm, dtype=float) / 100.0

    if args.target_rotvec is not None:
        target_pose[:3, :3] = Rotation.from_rotvec(np.asarray(args.target_rotvec, dtype=float)).as_matrix()
    if args.target_delta_rotvec is not None:
        target_pose[:3, :3] = target_pose[:3, :3] @ Rotation.from_rotvec(
            np.asarray(args.target_delta_rotvec, dtype=float)
        ).as_matrix()
    return target_pose


def choose_max_joint_step_deg(
    current_error_cm: float,
    default_max_joint_step_deg: float,
    *,
    use_orientation: bool,
    orientation_error_rad: float,
    tracking_error_deg: float,
) -> float:
    near_target = current_error_cm <= NEAR_TARGET_ERROR_CM and (
        not use_orientation or orientation_error_rad <= ORIENTATION_CONVERGENCE_RAD
    )
    very_near_target = current_error_cm <= VERY_NEAR_TARGET_ERROR_CM and (
        not use_orientation or orientation_error_rad <= ORIENTATION_CONVERGENCE_RAD
    )
    if near_target:
        step_deg = min(default_max_joint_step_deg, NEAR_TARGET_MAX_JOINT_STEP_DEG)
        if very_near_target:
            step_deg = min(step_deg, VERY_NEAR_TARGET_MAX_JOINT_STEP_DEG)
            if tracking_error_deg >= TRACKING_SETTLE_DEG:
                return min(step_deg, VERY_NEAR_TARGET_TRACKING_MAX_JOINT_STEP_DEG)
        if tracking_error_deg >= TRACKING_HEAVY_ERROR_DEG:
            return min(step_deg, NEAR_TARGET_HEAVY_TRACKING_MAX_JOINT_STEP_DEG)
        if tracking_error_deg >= TRACKING_SETTLE_DEG:
            return min(step_deg, NEAR_TARGET_TRACKING_MAX_JOINT_STEP_DEG)
        return step_deg
    if current_error_cm >= FAR_TARGET_ERROR_CM:
        return max(default_max_joint_step_deg, FAR_TARGET_MAX_JOINT_STEP_DEG)
    if current_error_cm >= MID_TARGET_ERROR_CM:
        step_deg = max(default_max_joint_step_deg, MID_TARGET_MAX_JOINT_STEP_DEG)
        if tracking_error_deg >= TRACKING_HEAVY_ERROR_DEG:
            return min(step_deg, MID_TARGET_TRACKING_MAX_JOINT_STEP_DEG)
        return step_deg
    return float(default_max_joint_step_deg)


def choose_replan_interval(current_error_cm: float, default_replan_interval: int) -> int:
    if current_error_cm <= NEAR_TARGET_ERROR_CM:
        return min(default_replan_interval, NEAR_TARGET_REPLAN_INTERVAL)
    return int(default_replan_interval)


def choose_progress_threshold_cm(current_error_cm: float) -> float:
    if current_error_cm <= 2.0:
        return 0.005
    if current_error_cm <= 4.0:
        return 0.01
    if current_error_cm <= NEAR_TARGET_ERROR_CM:
        return NEAR_TARGET_MIN_PROGRESS_CM
    return MIN_PROGRESS_CM


def choose_max_stale_steps(current_error_cm: float) -> int:
    if current_error_cm <= NEAR_TARGET_ERROR_CM:
        return NEAR_TARGET_MAX_STALE_STEPS
    return MAX_STALE_STEPS


def should_pause_replan(
    *,
    current_error_cm: float,
    tracking_error_deg: float,
    has_existing_goal: bool,
    tracking_wait_cycles: int,
) -> bool:
    tracking_threshold_deg = TRACKING_SETTLE_DEG
    if current_error_cm <= VERY_NEAR_TARGET_ERROR_CM:
        tracking_threshold_deg = VERY_NEAR_REPLAN_TRACKING_ERROR_DEG
    return (
        has_existing_goal
        and current_error_cm <= REPLAN_TRACKING_GUARD_ERROR_CM
        and tracking_error_deg >= tracking_threshold_deg
        and tracking_wait_cycles < MAX_TRACKING_WAIT_CYCLES
    )


def should_hold_servo_command(
    *,
    current_error_cm: float,
    tracking_error_deg: float,
    has_previous_command: bool,
    servo_hold_cycles: int,
) -> bool:
    tracking_threshold_deg = SERVO_HOLD_TRACKING_ERROR_DEG
    if current_error_cm <= VERY_NEAR_TARGET_ERROR_CM:
        tracking_threshold_deg = VERY_NEAR_SERVO_HOLD_TRACKING_ERROR_DEG
    return (
        has_previous_command
        and current_error_cm <= SERVO_HOLD_ERROR_CM
        and tracking_error_deg >= tracking_threshold_deg
        and servo_hold_cycles < MAX_SERVO_HOLD_CYCLES
    )


def choose_goal_blend_alpha(current_error_cm: float, tracking_error_deg: float) -> float:
    if current_error_cm <= NEAR_TARGET_ERROR_CM:
        if tracking_error_deg >= TRACKING_SETTLE_DEG:
            return NEAR_TARGET_TRACKING_GOAL_BLEND
        return NEAR_TARGET_GOAL_BLEND
    if current_error_cm <= REPLAN_TRACKING_GUARD_ERROR_CM:
        return MID_TARGET_GOAL_BLEND
    return 1.0


def joint_limit_violation_rad(robot_model: Any, q_rad: np.ndarray) -> float:
    lower = np.asarray(robot_model.mech_joint_limits_low, dtype=float)
    upper = np.asarray(robot_model.mech_joint_limits_up, dtype=float)
    low_violation = np.clip(lower - q_rad, 0.0, None)
    high_violation = np.clip(q_rad - upper, 0.0, None)
    return float(max(float(low_violation.max(initial=0.0)), float(high_violation.max(initial=0.0))))


def clip_q_to_argo_limits(robot_model: Any, q_rad: np.ndarray) -> np.ndarray:
    lower = np.asarray(robot_model.mech_joint_limits_low, dtype=float)
    upper = np.asarray(robot_model.mech_joint_limits_up, dtype=float)
    return np.clip(q_rad, lower, upper)


def argo_ik_error(
    *,
    kin: Any,
    robot_model: Any,
    q_rad: np.ndarray,
    target_pose: np.ndarray,
    target_link_name: str,
    robot_utils: Any,
    use_orientation: bool,
) -> tuple[float, float, np.ndarray]:
    pose = kin.forward_kinematics(robot_model, q_rad, target_link_name=target_link_name)
    err_lin = robot_utils.calc_lin_err(pose, target_pose)
    err_ang = robot_utils.calc_ang_err(pose, target_pose) if use_orientation else np.zeros(3)
    return float(np.linalg.norm(err_lin)), float(np.linalg.norm(err_ang)), pose


def build_argo_seed_candidates(
    q_start: np.ndarray,
    argo_joint_names: list[str],
    *,
    target_pose: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    seeds: list[tuple[str, np.ndarray]] = [("measured", q_start.copy())]
    seed_offsets_deg = [
        {},
        {"shoulder_pan": -45.0},
        {"shoulder_pan": 45.0},
        {"shoulder_pan": -90.0},
        {"shoulder_pan": 90.0},
        {"shoulder_lift": 25.0, "elbow_flex": -25.0},
        {"shoulder_lift": -25.0, "elbow_flex": 25.0},
        {"shoulder_lift": 40.0, "elbow_flex": -40.0},
        {"shoulder_lift": -40.0, "elbow_flex": 40.0},
        {"wrist_flex": -25.0},
        {"wrist_flex": 25.0},
        {"shoulder_lift": 30.0},
        {"shoulder_lift": -30.0},
        {"elbow_flex": -30.0},
        {"elbow_flex": 30.0},
        {"shoulder_pan": -60.0, "shoulder_lift": 30.0},
        {"shoulder_pan": 60.0, "shoulder_lift": 30.0},
        {"shoulder_pan": -60.0, "elbow_flex": -30.0},
        {"shoulder_pan": 60.0, "elbow_flex": -30.0},
    ]
    if target_pose is not None:
        target_xyz = target_pose[:3, 3] * 100.0
        if target_xyz[2] >= 35.0:
            seed_offsets_deg.extend(
                [
                    {"shoulder_lift": 50.0, "elbow_flex": -50.0},
                    {"shoulder_lift": 35.0, "wrist_flex": -20.0},
                    {"shoulder_lift": 20.0, "elbow_flex": -45.0, "wrist_flex": -15.0},
                ]
            )
        if abs(target_xyz[1]) >= 12.0:
            pan_sign = -1.0 if target_xyz[1] > 0 else 1.0
            seed_offsets_deg.extend(
                [
                    {"shoulder_pan": 75.0 * pan_sign},
                    {"shoulder_pan": 90.0 * pan_sign, "shoulder_lift": 20.0},
                    {"shoulder_pan": 75.0 * pan_sign, "elbow_flex": -25.0},
                ]
            )
    seen: set[tuple[float, ...]] = set()
    unique: list[tuple[str, np.ndarray]] = []
    for idx, offsets in enumerate(seed_offsets_deg):
        seed = q_start.copy()
        for joint_name, offset_deg in offsets.items():
            if joint_name in argo_joint_names:
                seed[argo_joint_names.index(joint_name)] += np.deg2rad(offset_deg)
        key = tuple(np.round(seed, 6))
        if key in seen:
            continue
        seen.add(key)
        label = "measured" if idx == 0 else "+".join(f"{k}{v:+.0f}" for k, v in offsets.items())
        unique.append((label, seed))
    return unique


def solve_argo_ik_candidates(
    *,
    kin: Any,
    robot_model: Any,
    robot_utils: Any,
    q_start: np.ndarray,
    target_pose: np.ndarray,
    target_link_name: str,
    argo_joint_names: list[str],
    gripper_pos: float,
    use_orientation: bool,
    gain: float,
    iterations: int,
) -> list[dict[str, Any]]:
    desired_base_t_n = (
        robot_utils.inv_homog_mat(robot_model.worldTbase)
        @ target_pose
        @ robot_utils.inv_homog_mat(robot_model.nTtool)
    )
    gripper_idx = argo_joint_names.index("gripper")
    candidates: list[dict[str, Any]] = []
    target_xyz = target_pose[:3, 3] * 100.0
    interpolation_plan: list[tuple[str, int, int]] = [
        ("direct_step", 1, iterations),
        ("interpolated_10", 10, max(1, iterations // 5)),
        ("interpolated_30", 30, max(1, iterations // 10)),
    ]
    if target_xyz[2] >= 35.0 or abs(target_xyz[1]) >= 12.0:
        interpolation_plan.extend(
            [
                ("interpolated_50", 50, max(1, iterations // 12)),
                ("interpolated_80", 80, max(1, iterations // 16)),
            ]
        )
    for seed_label, seed in build_argo_seed_candidates(q_start, argo_joint_names, target_pose=target_pose):
        seed = clip_q_to_argo_limits(robot_model, seed)
        for method, n_steps, step_iterations in interpolation_plan:
            q = seed.copy()
            try:
                if method == "direct_step":
                    q = kin._inverse_kinematics_step_baseTn(
                        robot_model,
                        q,
                        desired_base_t_n,
                        target_link_name,
                        use_orientation,
                        gain,
                        step_iterations,
                    )
                else:
                    current_base_t_n = kin._forward_kinematics_baseTn(
                        robot_model,
                        q,
                        target_link_name,
                    )
                    kin._interp_init(
                        current_base_t_n,
                        desired_base_t_n,
                        freq=n_steps,
                        trans_speed=1.0,
                        rot_speed=1.0,
                    )
                    for interp_idx in range(n_steps + 1):
                        q = kin._inverse_kinematics_step_baseTn(
                            robot_model,
                            q,
                            kin._interp_execute(interp_idx),
                            target_link_name,
                            use_orientation,
                            gain,
                            step_iterations,
                        )
                        q = clip_q_to_argo_limits(robot_model, q)
                q[gripper_idx] = gripper_percent_to_argo_rad(gripper_pos)
                q = clip_q_to_argo_limits(robot_model, q)
                pos_err, ori_err, pose = argo_ik_error(
                    kin=kin,
                    robot_model=robot_model,
                    q_rad=q,
                    target_pose=target_pose,
                    target_link_name=target_link_name,
                    robot_utils=robot_utils,
                    use_orientation=use_orientation,
                )
                candidates.append(
                    {
                        "q": q,
                        "pose": pose,
                        "pos_err": pos_err,
                        "ori_err": ori_err,
                        "limit_violation": joint_limit_violation_rad(robot_model, q),
                        "joint_delta_deg": arm_joint_delta_deg(
                            q_from_rad=q_start,
                            q_to_rad=q,
                            argo_joint_names=argo_joint_names,
                        ),
                        "method": method,
                        "seed": seed_label,
                    }
                )
            except Exception as exc:
                candidates.append(
                    {
                        "q": seed.copy(),
                        "pose": None,
                        "pos_err": float("inf"),
                        "ori_err": float("inf"),
                        "limit_violation": float("inf"),
                        "joint_delta_deg": float("inf"),
                        "method": method,
                        "seed": seed_label,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    if use_orientation:
        candidates.sort(
            key=lambda item: (
                candidate_pose_error_score_cm(
                    pos_err_m=float(item["pos_err"]),
                    ori_err_rad=float(item["ori_err"]),
                    use_orientation=True,
                ),
                float(item["ori_err"]),
                float(item["joint_delta_deg"]),
                float(item["limit_violation"]),
            )
        )
    else:
        candidates.sort(
            key=lambda item: (
                float(item["pos_err"]),
                float(item["joint_delta_deg"]),
                float(item["limit_violation"]),
                float(item["ori_err"]),
            )
        )
    return candidates


def next_velocity_limited_action(
    *,
    current_action: dict[str, float],
    target_action: dict[str, float],
    max_joint_step_deg: float,
) -> dict[str, float]:
    action = dict(current_action)
    for motor_name in ROBOT_MOTOR_NAMES:
        key = f"{motor_name}.pos"
        if motor_name == "gripper":
            action[key] = float(target_action[key])
            continue
        delta = float(target_action[key]) - float(current_action[key])
        action[key] = float(current_action[key] + np.clip(delta, -max_joint_step_deg, max_joint_step_deg))
    return action


def blend_q_goal(
    *,
    previous_q_goal: np.ndarray | None,
    new_q_goal: np.ndarray,
    argo_joint_names: list[str],
    blend_alpha: float,
) -> np.ndarray:
    if previous_q_goal is None or blend_alpha >= 1.0:
        return new_q_goal.copy()

    blended_q_goal = previous_q_goal.copy()
    for idx, joint_name in enumerate(argo_joint_names):
        if joint_name == "gripper":
            blended_q_goal[idx] = new_q_goal[idx]
            continue
        blended_q_goal[idx] = (1.0 - blend_alpha) * previous_q_goal[idx] + blend_alpha * new_q_goal[idx]
    return blended_q_goal


def hold_current_pose(
    *,
    robot: SO101Follower,
    hold_action: dict[str, float],
    fps: float,
    hold_time_s: float,
) -> None:
    hold_cycles = max(1, int(round(float(fps) * float(hold_time_s))))
    print(f"Holding target pose for {hold_time_s:.1f}s ({hold_cycles} cycles).")
    for _ in range(hold_cycles):
        robot.send_action(hold_action)
        precise_sleep(1.0 / float(fps))


def run_fixed_goal_tracking_diagnostic(
    *,
    robot: SO101Follower,
    robot_model: Any,
    kin: Any,
    argo_joint_names: list[str],
    target_link: str,
    target_pose: np.ndarray,
    target_action: dict[str, float],
    fps: float,
    duration_s: float,
    log_interval: int,
) -> None:
    cycles = max(1, int(round(float(fps) * float(duration_s))))
    print(
        f"Fixed-goal diagnostic: sending the initial final goal for {duration_s:.1f}s "
        f"({cycles} cycles) with no replan or blend."
    )
    for cycle in range(1, cycles + 1):
        commanded_action = robot.send_action(target_action)
        precise_sleep(1.0 / float(fps))

        reached_obs = robot.get_observation()
        reached_q = observation_to_argo_q_rad(reached_obs, argo_joint_names)
        reached_pose = kin.forward_kinematics(robot_model, reached_q, target_link_name=target_link)
        reached_error_cm = position_error_cm(reached_pose, target_pose)
        commanded_q = robot_action_to_argo_q_rad(commanded_action, argo_joint_names)
        tracking_error_deg = max_arm_tracking_error_deg(
            commanded_q_rad=commanded_q,
            measured_q_rad=reached_q,
            argo_joint_names=argo_joint_names,
        )

        if cycle == 1 or cycle % int(log_interval) == 0 or cycle == cycles:
            print(
                f"Fixed-goal cycle {cycle:03d}: error={reached_error_cm:.2f}cm "
                f"track={tracking_error_deg:.1f}deg"
            )
            print(
                "Joint tracking error:",
                format_joint_tracking_error(
                    commanded_q_rad=commanded_q,
                    measured_q_rad=reached_q,
                    argo_joint_names=argo_joint_names,
                ),
            )
            print_pose("Reached FK   :", reached_pose)

    print("Fixed-goal diagnostic completed.")


def run_final_joint_trim_diagnostic(
    *,
    robot: SO101Follower,
    robot_model: Any,
    kin: Any,
    argo_joint_names: list[str],
    target_link: str,
    target_pose: np.ndarray,
    q_goal: np.ndarray,
    gripper_pos: float,
    fps: float,
    log_interval: int,
) -> None:
    target_xyz_cm = target_pose[:3, 3] * 100.0
    target_y_cm = float(target_xyz_cm[1])
    target_z_cm = float(target_xyz_cm[2])

    trim_sets_deg = [
        ("base_goal", {}),
        ("elbow-4", {"elbow_flex": -4.0}),
        ("elbow+4", {"elbow_flex": 4.0}),
        ("wrist-3", {"wrist_flex": -3.0}),
        ("wrist+3", {"wrist_flex": 3.0}),
        ("lift-3", {"shoulder_lift": -3.0}),
        ("lift+3", {"shoulder_lift": 3.0}),
        ("elbow-4_wrist-3", {"elbow_flex": -4.0, "wrist_flex": -3.0}),
        ("elbow+4_wrist+3", {"elbow_flex": 4.0, "wrist_flex": 3.0}),
        ("elbow-4_lift-3", {"elbow_flex": -4.0, "shoulder_lift": -3.0}),
        ("elbow+4_lift+3", {"elbow_flex": 4.0, "shoulder_lift": 3.0}),
    ]
    if target_y_cm >= 10.0 and target_z_cm >= 35.0:
        trim_sets_deg.extend(
            [
                ("wrist+3_lift-3", {"wrist_flex": 3.0, "shoulder_lift": -3.0}),
                ("wrist+3_pan+2", {"wrist_flex": 3.0, "shoulder_pan": 2.0}),
                ("wrist+3_pan-2", {"wrist_flex": 3.0, "shoulder_pan": -2.0}),
                ("wrist+3_lift-3_pan+2", {"wrist_flex": 3.0, "shoulder_lift": -3.0, "shoulder_pan": 2.0}),
                ("wrist+3_lift-3_pan-2", {"wrist_flex": 3.0, "shoulder_lift": -3.0, "shoulder_pan": -2.0}),
            ]
        )
    if target_y_cm <= -15.0 and target_z_cm >= 35.0:
        trim_sets_deg.extend(
            [
                ("lift-3_pan+2", {"shoulder_lift": -3.0, "shoulder_pan": 2.0}),
                ("lift-3_pan-2", {"shoulder_lift": -3.0, "shoulder_pan": -2.0}),
                ("lift-3_wrist+3", {"shoulder_lift": -3.0, "wrist_flex": 3.0}),
                ("lift-3_wrist-3", {"shoulder_lift": -3.0, "wrist_flex": -3.0}),
            ]
        )
    seen_trim_labels: set[str] = set()
    unique_trim_sets: list[tuple[str, dict[str, float]]] = []
    for label, offsets_deg in trim_sets_deg:
        if label in seen_trim_labels:
            continue
        seen_trim_labels.add(label)
        unique_trim_sets.append((label, offsets_deg))
    trim_sets_deg = unique_trim_sets
    settle_cycles = max(1, int(round(float(fps) * FINAL_TRIM_SETTLE_SECONDS)))
    lower = np.asarray(robot_model.mech_joint_limits_low, dtype=float)
    upper = np.asarray(robot_model.mech_joint_limits_up, dtype=float)

    print(
        "Final joint trim diagnostic: testing small offsets around the IK goal "
        f"({len(trim_sets_deg)} candidates, {FINAL_TRIM_SETTLE_SECONDS:.1f}s each)."
    )

    best_result: dict[str, Any] | None = None
    for label, offsets_deg in trim_sets_deg:
        q_candidate = q_goal.copy()
        for joint_name, offset_deg in offsets_deg.items():
            if joint_name in argo_joint_names:
                q_candidate[argo_joint_names.index(joint_name)] += np.deg2rad(offset_deg)
        q_candidate = np.clip(q_candidate, lower, upper)
        target_action = argo_q_rad_to_robot_action(
            q_candidate,
            argo_joint_names,
            gripper_pos=float(gripper_pos),
        )

        for cycle in range(1, settle_cycles + 1):
            commanded_action = robot.send_action(target_action)
            precise_sleep(1.0 / float(fps))
            if cycle == settle_cycles:
                reached_obs = robot.get_observation()
                reached_q = observation_to_argo_q_rad(reached_obs, argo_joint_names)
                reached_pose = kin.forward_kinematics(robot_model, reached_q, target_link_name=target_link)
                reached_error_cm = position_error_cm(reached_pose, target_pose)
                commanded_q = robot_action_to_argo_q_rad(commanded_action, argo_joint_names)
                tracking_error_deg = max_arm_tracking_error_deg(
                    commanded_q_rad=commanded_q,
                    measured_q_rad=reached_q,
                    argo_joint_names=argo_joint_names,
                )
                print(
                    f"Trim {label:>18}: error={reached_error_cm:.2f}cm "
                    f"track={tracking_error_deg:.1f}deg offsets={offsets_deg or '{}'}"
                )
                print_pose("Reached FK   :", reached_pose)
                result = {
                    "label": label,
                    "offsets_deg": offsets_deg,
                    "error_cm": reached_error_cm,
                    "tracking_error_deg": tracking_error_deg,
                    "pose": reached_pose,
                    "q": reached_q,
                }
                if best_result is None or reached_error_cm < best_result["error_cm"]:
                    best_result = result

    if best_result is not None:
        print(
            f"Best trim: {best_result['label']} error={best_result['error_cm']:.2f}cm "
            f"track={best_result['tracking_error_deg']:.1f}deg offsets={best_result['offsets_deg'] or '{}'}"
        )
        print_pose("Best trim FK :", best_result["pose"])
    print("Final joint trim diagnostic completed.")


def apply_joint_offsets_deg(
    *,
    q_rad: np.ndarray,
    argo_joint_names: list[str],
    offsets_deg: dict[str, float],
) -> np.ndarray:
    q_adjusted = q_rad.copy()
    for joint_name, offset_deg in offsets_deg.items():
        if joint_name in argo_joint_names:
            q_adjusted[argo_joint_names.index(joint_name)] += np.deg2rad(offset_deg)
    return q_adjusted


def choose_final_trim_offsets_deg(target_pose: np.ndarray) -> dict[str, float]:
    target_xyz_cm = target_pose[:3, 3] * 100.0
    target_y_cm = float(target_xyz_cm[1])
    target_z_cm = float(target_xyz_cm[2])

    if target_y_cm <= -15.0 and target_z_cm >= 35.0:
        return {"shoulder_lift": -3.0}
    return {
        "elbow_flex": -4.0,
        "wrist_flex": -3.0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=FOLLOWER_PORT)
    parser.add_argument("--id", default=FOLLOWER_ID)
    parser.add_argument("--target-delta-cm", nargs=3, type=float, default=DEFAULT_TARGET_DELTA_CM)
    parser.add_argument("--target-x-cm", type=float, default=None)
    parser.add_argument("--target-y-cm", type=float, default=None)
    parser.add_argument("--target-z-cm", type=float, default=None)
    parser.add_argument(
        "--target-rotvec",
        nargs=3,
        type=float,
        default=None,
        metavar=("WX", "WY", "WZ"),
        help="Absolute target orientation as a rotation vector in radians.",
    )
    parser.add_argument(
        "--target-delta-rotvec",
        nargs=3,
        type=float,
        default=None,
        metavar=("DWX", "DWY", "DWZ"),
        help="Orientation delta applied on top of the current or absolute target orientation, in radians.",
    )
    parser.add_argument("--target-link", default=ARGO_TARGET_LINK_NAME)
    parser.add_argument("--gripper", type=float, default=DEFAULT_GRIPPER_POS)
    parser.add_argument("--use-orientation", action="store_true", default=DEFAULT_USE_ORIENTATION)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--steps", type=int, default=DEFAULT_INTERPOLATION_STEPS)
    parser.add_argument("--ik-gain", type=float, default=0.8)
    parser.add_argument("--ik-iterations", type=int, default=50)
    parser.add_argument("--max-joint-step-deg", type=float, default=MAX_JOINT_STEP_DEG)
    parser.add_argument("--max-servo-cycles", type=int, default=DEFAULT_MAX_SERVO_CYCLES)
    parser.add_argument("--replan-interval", type=int, default=DEFAULT_REPLAN_INTERVAL)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument(
        "--fixed-goal-diagnostic-seconds",
        type=float,
        default=DEFAULT_FIXED_GOAL_DIAGNOSTIC_SECONDS,
        help="If > 0, send the initial final goal for this many seconds with no replan or blend.",
    )
    parser.add_argument(
        "--final-joint-trim-diagnostic",
        action="store_true",
        default=DEFAULT_FINAL_TRIM_DIAGNOSTIC,
        help="Test a few small joint offsets around the IK goal directly on the real robot.",
    )
    parser.add_argument(
        "--apply-final-trim",
        action="store_true",
        default=DEFAULT_APPLY_FINAL_TRIM,
        help="Apply the validated final joint trim compensation to the IK goal before execution.",
    )
    parser.add_argument(
        "--no-final-trim",
        action="store_false",
        dest="apply_final_trim",
        help="Disable the default final joint trim compensation.",
    )
    parser.add_argument("--verbose-candidates", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    URDFLoader, RobotModel, URDFKinematics, RobotUtils = import_argo_controls()

    if not ARGO_URDF_PATH.exists():
        raise FileNotFoundError(f"Argo SO101 URDF not found at {ARGO_URDF_PATH}")

    urdf_loader = URDFLoader()
    urdf_loader.load(str(ARGO_URDF_PATH))
    robot_model = RobotModel(urdf_loader)
    kin = URDFKinematics()
    argo_joint_names = [
        joint.name for joint in robot_model.loader.robot.joints if joint.joint_type != "fixed"
    ]
    unknown_joints = sorted(set(argo_joint_names) - set(ROBOT_MOTOR_NAMES))
    if unknown_joints:
        raise ValueError(f"Argo URDF contains unsupported movable joints: {unknown_joints}")

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    robot.connect(calibrate=False)

    try:
        observation = robot.get_observation()
        q_start = observation_to_argo_q_rad(observation, argo_joint_names)
        start_pose = kin.forward_kinematics(robot_model, q_start, target_link_name=args.target_link)
        target_pose = build_target_pose(start_pose, args)
        use_orientation = target_requests_orientation(args)

        print("Argo movable joint order:", ", ".join(argo_joint_names))
        print("Current joints:", format_q_deg(q_start, argo_joint_names))
        print_pose("Current Argo FK:", start_pose)
        print_pose("Target       :", target_pose)
        print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
        print(f"Orientation IK: {'enabled' if use_orientation else 'disabled'}")

        current_obs = observation
        current_q = q_start
        current_pose = start_pose
        current_error_cm = position_error_cm(current_pose, target_pose)
        current_orientation_error = orientation_error_rad(current_pose, target_pose) if use_orientation else 0.0
        best_pose_score = pose_error_score_cm(current_pose, target_pose, use_orientation=use_orientation)
        stale_steps = 0
        q_goal = None
        target_action = None
        tracking_error_deg = 0.0
        last_commanded_action = None
        tracking_wait_cycles = 0
        servo_hold_cycles = 0

        def plan_from_current_state(*, print_diagnostics: bool) -> tuple[np.ndarray, dict[str, float]]:
            candidates = solve_argo_ik_candidates(
                kin=kin,
                robot_model=robot_model,
                robot_utils=RobotUtils,
                q_start=current_q,
                target_pose=target_pose,
                target_link_name=args.target_link,
                argo_joint_names=argo_joint_names,
                gripper_pos=float(args.gripper),
                use_orientation=use_orientation,
                gain=float(args.ik_gain),
                iterations=int(args.ik_iterations),
            )
            if print_diagnostics:
                print("Argo IK candidates:")
                for idx, candidate in enumerate(candidates[:IK_CANDIDATE_LIMIT], start=1):
                    if "error" in candidate:
                        print(
                            f"  {idx}. method={candidate['method']} seed={candidate['seed']} "
                            f"error={candidate['error']}"
                        )
                        continue
                    print(
                        f"  {idx}. method={candidate['method']} seed={candidate['seed']} "
                        f"pos={candidate['pos_err'] * 100.0:.2f}cm "
                        f"dq={candidate['joint_delta_deg']:.1f}deg "
                        f"ori={candidate['ori_err']:.4f} "
                        f"limit_violation={np.rad2deg(candidate['limit_violation']):.2f}deg"
                    )

            if not candidates or not np.isfinite(candidates[0]["pos_err"]):
                raise RuntimeError("Argo IK did not produce any finite candidate.")

            best_candidate = candidates[0]
            planned_q_goal = best_candidate["q"]
            final_trim_offsets_deg = choose_final_trim_offsets_deg(target_pose)
            if bool(args.apply_final_trim):
                planned_q_goal = clip_q_to_argo_limits(
                    robot_model,
                    apply_joint_offsets_deg(
                        q_rad=planned_q_goal,
                        argo_joint_names=argo_joint_names,
                        offsets_deg=final_trim_offsets_deg,
                    ),
                )
            final_pose = best_candidate["pose"]
            if bool(args.apply_final_trim):
                final_pose = kin.forward_kinematics(
                    robot_model,
                    planned_q_goal,
                    target_link_name=args.target_link,
                )
            err_lin = RobotUtils.calc_lin_err(final_pose, target_pose)
            err_ang = RobotUtils.calc_ang_err(final_pose, target_pose)
            if print_diagnostics:
                if bool(args.apply_final_trim):
                    print(f"Applying final trim offsets (deg): {final_trim_offsets_deg}")
                print("Goal joints   :", format_q_deg(planned_q_goal, argo_joint_names))
                print_pose("Predicted FK :", final_pose)
                print(f"Predicted linear error: {np.linalg.norm(err_lin) * 100.0:.2f}cm vector={err_lin}")
                print(f"Predicted angular error norm: {np.linalg.norm(err_ang):.4f}")
            planned_target_action = argo_q_rad_to_robot_action(
                planned_q_goal,
                argo_joint_names,
                gripper_pos=float(args.gripper),
            )
            return planned_q_goal, planned_target_action

        if use_orientation:
            print(
                f"Initial target error: pos={current_error_cm:.2f}cm "
                f"ori={current_orientation_error:.4f}rad score={best_pose_score:.2f}"
            )
        else:
            print(f"Initial target error: {current_error_cm:.2f}cm")
        q_goal, target_action = plan_from_current_state(print_diagnostics=True)

        if not args.execute:
            start_action = argo_q_rad_to_robot_action(
                current_q,
                argo_joint_names,
                gripper_pos=float(current_obs["gripper.pos"]),
            )
            max_joint_step_deg = choose_max_joint_step_deg(
                current_error_cm,
                float(args.max_joint_step_deg),
                use_orientation=use_orientation,
                orientation_error_rad=current_orientation_error,
                tracking_error_deg=tracking_error_deg,
            )
            preview_action = next_velocity_limited_action(
                current_action=start_action,
                target_action=target_action,
                max_joint_step_deg=max_joint_step_deg,
            )
            preview_q = robot_action_to_argo_q_rad(preview_action, argo_joint_names)
            preview_pose = kin.forward_kinematics(robot_model, preview_q, target_link_name=args.target_link)
            print(f"Preview max joint step: {max_joint_step_deg:.2f}deg")
            print("Preview cmd          :", format_q_deg(preview_q, argo_joint_names))
            print_pose("Preview FK           :", preview_pose)
            preview_error_cm = position_error_cm(preview_pose, target_pose)
            if use_orientation:
                preview_orientation_error = orientation_error_rad(preview_pose, target_pose)
                print(
                    f"Predicted preview error: pos={preview_error_cm:.2f}cm "
                    f"ori={preview_orientation_error:.4f}rad"
                )
            else:
                print(f"Predicted preview error: {preview_error_cm:.2f}cm")
            print("Dry-run only. Re-run with --execute to stream continuous servo commands.")
            return

        if float(args.fixed_goal_diagnostic_seconds) > 0.0:
            run_fixed_goal_tracking_diagnostic(
                robot=robot,
                robot_model=robot_model,
                kin=kin,
                argo_joint_names=argo_joint_names,
                target_link=args.target_link,
                target_pose=target_pose,
                target_action=target_action,
                fps=float(args.fps),
                duration_s=float(args.fixed_goal_diagnostic_seconds),
                log_interval=int(args.log_interval),
            )
            return

        if bool(args.final_joint_trim_diagnostic):
            run_final_joint_trim_diagnostic(
                robot=robot,
                robot_model=robot_model,
                kin=kin,
                argo_joint_names=argo_joint_names,
                target_link=args.target_link,
                target_pose=target_pose,
                q_goal=q_goal,
                gripper_pos=float(args.gripper),
                fps=float(args.fps),
                log_interval=int(args.log_interval),
            )
            return

        print(
            f"Streaming servo: max_cycles={int(args.max_servo_cycles)} "
            f"replan_interval={int(args.replan_interval)} log_interval={int(args.log_interval)} "
            f"fps={float(args.fps):.1f}"
        )
        for servo_cycle in range(1, int(args.max_servo_cycles) + 1):
            current_replan_interval = choose_replan_interval(current_error_cm, int(args.replan_interval))
            current_progress_threshold_cm = choose_progress_threshold_cm(current_error_cm)
            current_max_stale_steps = choose_max_stale_steps(current_error_cm)

            if servo_cycle == 1 or servo_cycle % current_replan_interval == 0:
                if should_pause_replan(
                    current_error_cm=current_error_cm,
                    tracking_error_deg=tracking_error_deg,
                    has_existing_goal=target_action is not None,
                    tracking_wait_cycles=tracking_wait_cycles,
                ):
                    if servo_cycle == 1 or servo_cycle % int(args.log_interval) == 0:
                        print(
                            "Holding previous IK goal until joint tracking settles "
                            f"(track={tracking_error_deg:.1f}deg, wait={tracking_wait_cycles})."
                        )
                else:
                    if (
                        tracking_wait_cycles >= MAX_TRACKING_WAIT_CYCLES
                        and (servo_cycle == 1 or servo_cycle % int(args.log_interval) == 0)
                    ):
                        print(
                            "Replanning despite residual tracking lag "
                            f"(track={tracking_error_deg:.1f}deg, wait={tracking_wait_cycles})."
                        )
                    planned_q_goal, planned_target_action = plan_from_current_state(
                        print_diagnostics=bool(args.verbose_candidates),
                    )
                    blend_alpha = choose_goal_blend_alpha(current_error_cm, tracking_error_deg)
                    q_goal = clip_q_to_argo_limits(
                        robot_model,
                        blend_q_goal(
                            previous_q_goal=q_goal,
                            new_q_goal=planned_q_goal,
                            argo_joint_names=argo_joint_names,
                            blend_alpha=blend_alpha,
                        ),
                    )
                    target_action = argo_q_rad_to_robot_action(
                        q_goal,
                        argo_joint_names,
                        gripper_pos=float(args.gripper),
                    )
                    if (
                        blend_alpha < 1.0
                        and (servo_cycle == 1 or servo_cycle % int(args.log_interval) == 0)
                    ):
                        print(
                            f"Target blend   : alpha={blend_alpha:.2f} "
                            f"track={tracking_error_deg:.1f}deg"
                        )

            current_max_joint_step_deg = choose_max_joint_step_deg(
                current_error_cm,
                float(args.max_joint_step_deg),
                use_orientation=use_orientation,
                orientation_error_rad=current_orientation_error,
                tracking_error_deg=tracking_error_deg,
            )
            if should_hold_servo_command(
                current_error_cm=current_error_cm,
                tracking_error_deg=tracking_error_deg,
                has_previous_command=last_commanded_action is not None,
                servo_hold_cycles=servo_hold_cycles,
            ):
                if current_error_cm <= VERY_NEAR_TARGET_ERROR_CM and target_action is not None:
                    servo_action_target = dict(target_action)
                else:
                    servo_action_target = dict(last_commanded_action)
                servo_hold_cycles += 1
                if servo_cycle == 1 or servo_cycle % int(args.log_interval) == 0:
                    print(
                        "Holding previous servo cmd to let joints catch up "
                        f"(track={tracking_error_deg:.1f}deg, hold={servo_hold_cycles})."
                    )
            else:
                if (
                    servo_hold_cycles >= MAX_SERVO_HOLD_CYCLES
                    and (servo_cycle == 1 or servo_cycle % int(args.log_interval) == 0)
                ):
                    print(
                        "Resuming micro-step servo despite residual tracking lag "
                        f"(track={tracking_error_deg:.1f}deg, hold={servo_hold_cycles})."
                    )
                servo_hold_cycles = 0
                start_action = argo_q_rad_to_robot_action(
                    current_q,
                    argo_joint_names,
                    gripper_pos=float(current_obs["gripper.pos"]),
                )
                servo_action_target = next_velocity_limited_action(
                    current_action=start_action,
                    target_action=target_action,
                    max_joint_step_deg=current_max_joint_step_deg,
                )
            commanded_action = robot.send_action(servo_action_target)
            last_commanded_action = dict(commanded_action)
            precise_sleep(1.0 / float(args.fps))

            reached_obs = robot.get_observation()
            reached_q = observation_to_argo_q_rad(reached_obs, argo_joint_names)
            reached_pose = kin.forward_kinematics(robot_model, reached_q, target_link_name=args.target_link)
            reached_error_cm = position_error_cm(reached_pose, target_pose)
            reached_orientation_error = orientation_error_rad(reached_pose, target_pose) if use_orientation else 0.0
            reached_pose_score = pose_error_score_cm(reached_pose, target_pose, use_orientation=use_orientation)
            commanded_q = robot_action_to_argo_q_rad(commanded_action, argo_joint_names)
            tracking_error_deg = max_arm_tracking_error_deg(
                commanded_q_rad=commanded_q,
                measured_q_rad=reached_q,
                argo_joint_names=argo_joint_names,
            )
            if servo_cycle == 1 or servo_cycle % int(args.log_interval) == 0:
                if use_orientation:
                    print(
                        f"Servo cycle {servo_cycle:03d}: pos={reached_error_cm:.2f}cm "
                        f"ori={reached_orientation_error:.4f}rad "
                        f"max_step={current_max_joint_step_deg:.1f}deg "
                        f"track={tracking_error_deg:.1f}deg"
                    )
                else:
                    print(
                        f"Servo cycle {servo_cycle:03d}: error={reached_error_cm:.2f}cm "
                        f"max_step={current_max_joint_step_deg:.1f}deg "
                        f"track={tracking_error_deg:.1f}deg"
                    )
                print(
                    "Joint tracking error:",
                    format_joint_tracking_error(
                        commanded_q_rad=commanded_q,
                        measured_q_rad=reached_q,
                        argo_joint_names=argo_joint_names,
                    ),
                )

            if target_reached(reached_pose, target_pose, use_orientation=use_orientation):
                if use_orientation:
                    print("Target reached within position and orientation tolerance.")
                else:
                    print("Target reached within position tolerance.")
                hold_current_pose(
                    robot=robot,
                    hold_action=commanded_action,
                    fps=float(args.fps),
                    hold_time_s=TARGET_HOLD_TIME_S,
                )
                break

            progress_cm = best_pose_score - reached_pose_score
            if progress_cm >= current_progress_threshold_cm:
                best_pose_score = reached_pose_score
                stale_steps = 0
            elif tracking_error_deg > TRACKING_SETTLE_DEG:
                tracking_wait_cycles += 1
                if servo_cycle == 1 or servo_cycle % int(args.log_interval) == 0:
                    print(
                        f"Waiting for joint tracking to settle before counting stale progress "
                        f"(track={tracking_error_deg:.1f}deg > {TRACKING_SETTLE_DEG:.1f}deg, "
                        f"wait={tracking_wait_cycles})."
                    )
            else:
                tracking_wait_cycles = 0
                stale_steps += 1
                print(
                    f"Measured progress {progress_cm:.2f}cm is below threshold "
                    f"{current_progress_threshold_cm:.2f}cm "
                    f"(stale {stale_steps}/{current_max_stale_steps})."
                )
                if stale_steps >= current_max_stale_steps:
                    print("Stopping because progress stayed below threshold for too many control steps.")
                    break

            if tracking_error_deg <= TRACKING_SETTLE_DEG:
                tracking_wait_cycles = 0

            current_obs = reached_obs
            current_q = reached_q
            current_pose = reached_pose
            current_error_cm = reached_error_cm
            current_orientation_error = reached_orientation_error

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
