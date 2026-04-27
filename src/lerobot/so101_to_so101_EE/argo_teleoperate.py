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
SO101 leader-to-follower teleoperation using Argo-Robot/controls IK.

This is the online counterpart to `argo_real_ik.py`: leader joints are converted
to a target end-effector pose with Argo FK, then follower joints are solved with
a lightweight single-step Argo IK pass.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.so101_to_so101_EE.argo_real_ik import (
    ARGO_TARGET_LINK_NAME,
    ARGO_URDF_PATH,
    DEFAULT_FPS,
    DEFAULT_IK_GAIN,
    FOLLOWER_ID,
    FOLLOWER_PORT,
    ROBOT_MOTOR_NAMES,
    argo_ik_error,
    argo_q_rad_to_robot_action,
    candidate_pose_error_score_cm,
    clip_q_to_argo_limits,
    gripper_percent_to_argo_rad,
    import_argo_controls,
    observation_to_argo_q_rad,
)
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep

LEADER_PORT = "/dev/ttyACM1"
LEADER_ID = "so101_leader"

DEFAULT_IK_ITERATIONS = 8
USE_ORIENTATION = True


def argo_joint_names_from_model(robot_model: Any) -> list[str]:
    return [joint.name for joint in robot_model.loader.robot.joints if joint.joint_type != "fixed"]


def validate_argo_joint_names(argo_joint_names: list[str]) -> None:
    unknown_joints = sorted(set(argo_joint_names) - set(ROBOT_MOTOR_NAMES))
    if unknown_joints:
        raise ValueError(f"Argo URDF contains unsupported movable joints: {unknown_joints}")


def fast_seed_candidates(
    *,
    measured_q_rad: np.ndarray,
    previous_q_goal_rad: np.ndarray | None,
) -> list[tuple[str, np.ndarray]]:
    seeds: list[tuple[str, np.ndarray]] = []
    if previous_q_goal_rad is not None:
        seeds.append(("previous_goal", previous_q_goal_rad.copy()))
    if previous_q_goal_rad is None or not np.allclose(previous_q_goal_rad, measured_q_rad, atol=1e-6):
        seeds.append(("measured", measured_q_rad.copy()))
    return seeds


def solve_fast_argo_ik(
    *,
    kin: Any,
    robot_model: Any,
    robot_utils: Any,
    measured_q_rad: np.ndarray,
    previous_q_goal_rad: np.ndarray | None,
    target_pose: np.ndarray,
    target_link_name: str,
    argo_joint_names: list[str],
    gripper_pos: float,
    gain: float,
    iterations: int,
) -> dict[str, Any]:
    desired_base_t_n = (
        robot_utils.inv_homog_mat(robot_model.worldTbase)
        @ target_pose
        @ robot_utils.inv_homog_mat(robot_model.nTtool)
    )
    gripper_idx = argo_joint_names.index("gripper")
    candidates: list[dict[str, Any]] = []

    for seed_label, seed in fast_seed_candidates(
        measured_q_rad=measured_q_rad,
        previous_q_goal_rad=previous_q_goal_rad,
    ):
        seed = clip_q_to_argo_limits(robot_model, seed)
        try:
            q = kin._inverse_kinematics_step_baseTn(
                robot_model,
                seed.copy(),
                desired_base_t_n,
                target_link_name,
                USE_ORIENTATION,
                gain,
                iterations,
            )
            q[gripper_idx] = gripper_percent_to_argo_rad(gripper_pos)
            q = clip_q_to_argo_limits(robot_model, q)
            pos_err, ori_err, pose = argo_ik_error(
                kin=kin,
                robot_model=robot_model,
                q_rad=q,
                target_pose=target_pose,
                target_link_name=target_link_name,
                robot_utils=robot_utils,
                use_orientation=USE_ORIENTATION,
            )
            candidates.append(
                {
                    "q": q,
                    "pose": pose,
                    "pos_err": pos_err,
                    "ori_err": ori_err,
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
                    "seed": seed_label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    candidates.sort(
        key=lambda item: (
            candidate_pose_error_score_cm(
                pos_err_m=float(item["pos_err"]),
                ori_err_rad=float(item["ori_err"]),
                use_orientation=USE_ORIENTATION,
            ),
            float(item["ori_err"]),
        )
    )
    best = candidates[0]
    if not np.isfinite(best["pos_err"]):
        raise RuntimeError(f"Argo teleop IK failed for all fast seeds: {candidates}")
    return best


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", default=LEADER_PORT)
    parser.add_argument("--follower-port", default=FOLLOWER_PORT)
    parser.add_argument("--leader-id", default=LEADER_ID)
    parser.add_argument("--follower-id", default=FOLLOWER_ID)
    parser.add_argument("--target-link", default=ARGO_TARGET_LINK_NAME)
    parser.add_argument("--calibrate", action="store_true", help="Allow interactive calibration on connect.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--ik-gain", type=float, default=DEFAULT_IK_GAIN)
    parser.add_argument("--ik-iterations", type=int, default=DEFAULT_IK_ITERATIONS)
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
    argo_joint_names = argo_joint_names_from_model(robot_model)
    validate_argo_joint_names(argo_joint_names)

    follower = SO101Follower(
        SO101FollowerConfig(port=args.follower_port, id=args.follower_id, use_degrees=True)
    )
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id, use_degrees=True))

    follower.connect(calibrate=bool(args.calibrate))
    leader.connect(calibrate=bool(args.calibrate))

    previous_q_goal: np.ndarray | None = None

    try:
        print("Argo movable joint order:", ", ".join(argo_joint_names))
        print("Orientation IK: enabled")
        print(f"Starting Argo teleop loop at {float(args.fps):.1f} FPS...")

        while True:
            t0 = time.perf_counter()

            leader_action = leader.get_action()
            follower_obs = follower.get_observation()

            leader_q = observation_to_argo_q_rad(leader_action, argo_joint_names)
            follower_q = observation_to_argo_q_rad(follower_obs, argo_joint_names)
            target_pose = kin.forward_kinematics(
                robot_model,
                leader_q,
                target_link_name=args.target_link,
            )
            ik_result = solve_fast_argo_ik(
                kin=kin,
                robot_model=robot_model,
                robot_utils=RobotUtils,
                measured_q_rad=follower_q,
                previous_q_goal_rad=previous_q_goal,
                target_pose=target_pose,
                target_link_name=args.target_link,
                argo_joint_names=argo_joint_names,
                gripper_pos=float(leader_action["gripper.pos"]),
                gain=float(args.ik_gain),
                iterations=int(args.ik_iterations),
            )
            q_goal = ik_result["q"]
            previous_q_goal = q_goal.copy()

            target_action = argo_q_rad_to_robot_action(
                q_goal,
                argo_joint_names,
                gripper_pos=float(leader_action["gripper.pos"]),
            )
            _ = follower.send_action(target_action)

            precise_sleep(max(1.0 / float(args.fps) - (time.perf_counter() - t0), 0.0))
    finally:
        leader.disconnect()
        follower.disconnect()


if __name__ == "__main__":
    main()
