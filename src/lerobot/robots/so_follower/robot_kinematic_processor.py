#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    TransitionKey,
)
from lerobot.utils.rotation import Rotation

logger = logging.getLogger(__name__)

DEFAULT_SO_ARM_IK_SEED_OFFSETS_DEG = {
    "shoulder_pan": 12.0,
    "shoulder_lift": 12.0,
    "elbow_flex": 18.0,
    "wrist_flex": 15.0,
    "wrist_roll": 45.0,
    "gripper": 0.0,
}


def _ordered_joint_positions(joints: dict[str, Any], motor_names: list[str]) -> np.ndarray:
    missing = [name for name in motor_names if f"{name}.pos" not in joints]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing joint observations/actions for motors: {missing_str}")
    return np.array([float(joints[f"{name}.pos"]) for name in motor_names], dtype=float)


def _active_joint_mask(motor_names: list[str]) -> np.ndarray:
    return np.array([name != "gripper" for name in motor_names], dtype=bool)


def _angle_delta_deg(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (target - reference + 180.0) % 360.0 - 180.0


def _rotation_error_rad(desired_pose: np.ndarray, actual_pose: np.ndarray) -> float:
    delta = desired_pose[:3, :3].T @ actual_pose[:3, :3]
    cos_theta = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def _per_joint_values(
    values: float | dict[str, float] | None, motor_names: list[str], default: float
) -> np.ndarray:
    if values is None:
        return np.full(len(motor_names), default, dtype=float)
    if isinstance(values, dict):
        return np.array([float(values.get(name, default)) for name in motor_names], dtype=float)
    return np.full(len(motor_names), float(values), dtype=float)


def _unique_joint_vectors(candidates: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for candidate in candidates:
        key = tuple(np.round(candidate.astype(float), 6))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate.astype(float, copy=True))
    return unique


def _build_ik_seed_candidates(
    current_joint_pos: np.ndarray,
    preferred_seed: np.ndarray,
    previous_solution: np.ndarray | None,
    motor_names: list[str],
    seed_joint_offsets_deg: float | dict[str, float] | None,
) -> list[np.ndarray]:
    candidates = [preferred_seed, current_joint_pos]

    if previous_solution is not None and len(previous_solution) == len(current_joint_pos):
        candidates.extend(
            [
                previous_solution,
                0.5 * (current_joint_pos + previous_solution),
            ]
        )

    per_joint_offsets = _per_joint_values(
        seed_joint_offsets_deg if seed_joint_offsets_deg is not None else DEFAULT_SO_ARM_IK_SEED_OFFSETS_DEG,
        motor_names,
        0.0,
    )
    for idx, (motor_name, offset) in enumerate(zip(motor_names, per_joint_offsets, strict=True)):
        if motor_name == "gripper" or offset <= 0.0:
            continue
        plus = preferred_seed.copy()
        plus[idx] += offset
        candidates.append(plus)

        minus = preferred_seed.copy()
        minus[idx] -= offset
        candidates.append(minus)

    return _unique_joint_vectors(candidates)


def _score_ik_candidate(
    candidate: np.ndarray,
    *,
    kinematics: RobotKinematics,
    desired_pose: np.ndarray,
    current_joint_pos: np.ndarray,
    preferred_seed: np.ndarray,
    motor_names: list[str],
    max_joint_delta_deg: float | dict[str, float] | None,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> dict[str, Any]:
    achieved_pose = kinematics.forward_kinematics(candidate)
    position_error = float(np.linalg.norm(desired_pose[:3, 3] - achieved_pose[:3, 3]))
    orientation_error = _rotation_error_rad(desired_pose, achieved_pose)

    active_mask = _active_joint_mask(motor_names)
    current_delta = np.abs(_angle_delta_deg(candidate, current_joint_pos))[active_mask]
    preferred_delta = np.abs(_angle_delta_deg(candidate, preferred_seed))[active_mask]
    delta_limits = _per_joint_values(max_joint_delta_deg, motor_names, np.inf)[active_mask]
    over_limit = np.clip(current_delta - delta_limits, 0.0, None)

    max_over_limit = float(over_limit.max()) if over_limit.size else 0.0
    max_joint_delta = float(current_delta.max()) if current_delta.size else 0.0
    mean_joint_delta = float(current_delta.mean()) if current_delta.size else 0.0
    mean_preferred_delta = float(preferred_delta.mean()) if preferred_delta.size else 0.0

    is_unsafe = bool(np.any(over_limit > 0.0))
    pose_invalid = bool(
        position_error > position_tolerance_m or orientation_error > orientation_tolerance_rad
    )

    return {
        "candidate": candidate,
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
        "max_joint_delta_deg": max_joint_delta,
        "mean_joint_delta_deg": mean_joint_delta,
        "mean_preferred_delta_deg": mean_preferred_delta,
        "max_over_limit_deg": max_over_limit,
        "is_unsafe": is_unsafe,
        "pose_invalid": pose_invalid,
        "sort_key": (
            int(is_unsafe),
            int(pose_invalid),
            max_over_limit,
            position_error,
            orientation_error,
            max_joint_delta,
            mean_joint_delta,
            mean_preferred_delta,
        ),
    }


def _solve_best_ik_solution(
    *,
    kinematics: RobotKinematics,
    motor_names: list[str],
    current_joint_pos: np.ndarray,
    desired_pose: np.ndarray,
    previous_solution: np.ndarray | None,
    prefer_current_joints: bool,
    max_joint_delta_deg: float | dict[str, float] | None,
    seed_joint_offsets_deg: float | dict[str, float] | None,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    fallback_to_current_joints_on_invalid: bool,
) -> np.ndarray:
    preferred_seed = (
        current_joint_pos
        if prefer_current_joints or previous_solution is None
        else previous_solution.astype(float, copy=True)
    )

    scored_candidates = []
    for seed in _build_ik_seed_candidates(
        current_joint_pos=current_joint_pos,
        preferred_seed=preferred_seed,
        previous_solution=previous_solution,
        motor_names=motor_names,
        seed_joint_offsets_deg=seed_joint_offsets_deg,
    ):
        try:
            candidate = kinematics.inverse_kinematics(seed, desired_pose)
            scored_candidates.append(
                _score_ik_candidate(
                    candidate,
                    kinematics=kinematics,
                    desired_pose=desired_pose,
                    current_joint_pos=current_joint_pos,
                    preferred_seed=preferred_seed,
                    motor_names=motor_names,
                    max_joint_delta_deg=max_joint_delta_deg,
                    position_tolerance_m=position_tolerance_m,
                    orientation_tolerance_rad=orientation_tolerance_rad,
                )
            )
        except Exception:
            logger.debug("IK solve failed for seed %s", np.round(seed, 3), exc_info=True)

    if not scored_candidates:
        logger.warning("IK failed for all seeds, holding current joints.")
        return current_joint_pos.copy()

    best = min(scored_candidates, key=lambda item: item["sort_key"])
    if fallback_to_current_joints_on_invalid and (best["is_unsafe"] or best["pose_invalid"]):
        logger.warning(
            "Rejecting IK candidate: unsafe=%s pose_invalid=%s pos_err=%.4fm ori_err=%.4frad max_dq=%.2fdeg. Holding current joints.",
            best["is_unsafe"],
            best["pose_invalid"],
            best["position_error_m"],
            best["orientation_error_rad"],
            best["max_joint_delta_deg"],
        )
        return current_joint_pos.copy()

    return best["candidate"].copy()


@ProcessorStepRegistry.register("ee_reference_and_delta")
@dataclass
class EEReferenceAndDelta(RobotActionProcessorStep):
    """
    Computes a target end-effector pose from a relative delta command.

    This step takes a desired change in position and orientation (`target_*`) and applies it to a
    reference end-effector pose to calculate an absolute target pose. The reference pose is derived
    from the current robot joint positions using forward kinematics.

    The processor can operate in two modes:
    1.  `use_latched_reference=True`: The reference pose is "latched" or saved at the moment the action
        is first enabled. Subsequent commands are relative to this fixed reference.
    2.  `use_latched_reference=False`: The reference pose is updated to the robot's current pose at
        every step.

    Attributes:
        kinematics: The robot's kinematic model for forward kinematics.
        end_effector_step_sizes: A dictionary scaling the input delta commands.
        motor_names: A list of motor names required for forward kinematics.
        use_latched_reference: If True, latch the reference pose on enable; otherwise, always use the
            current pose as the reference.
        reference_ee_pose: Internal state storing the latched reference pose.
        _prev_enabled: Internal state to detect the rising edge of the enable signal.
        _command_when_disabled: Internal state to hold the last command while disabled.
    """

    kinematics: RobotKinematics
    end_effector_step_sizes: dict
    motor_names: list[str]
    use_latched_reference: bool = (
        True  # If True, latch reference on enable; if False, always use current pose
    )
    use_ik_solution: bool = False

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _command_when_disabled: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.use_ik_solution and "IK_solution" in self.transition.get(TransitionKey.COMPLEMENTARY_DATA):
            q_raw = np.array(
                self.transition.get(TransitionKey.COMPLEMENTARY_DATA)["IK_solution"],
                dtype=float,
            )
        else:
            q_raw = _ordered_joint_positions(observation, self.motor_names)

        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        # Current pose from FK on measured joints
        t_curr = self.kinematics.forward_kinematics(q_raw)

        enabled = bool(action.pop("enabled"))
        tx = float(action.pop("target_x"))
        ty = float(action.pop("target_y"))
        tz = float(action.pop("target_z"))
        wx = float(action.pop("target_wx"))
        wy = float(action.pop("target_wy"))
        wz = float(action.pop("target_wz"))
        gripper_vel = float(action.pop("gripper_vel"))

        desired = None

        if enabled:
            ref = t_curr
            if self.use_latched_reference:
                # Latched reference mode: latch reference at the rising edge
                if not self._prev_enabled or self.reference_ee_pose is None:
                    self.reference_ee_pose = t_curr.copy()
                ref = self.reference_ee_pose if self.reference_ee_pose is not None else t_curr

            delta_p = np.array(
                [
                    tx * self.end_effector_step_sizes["x"],
                    ty * self.end_effector_step_sizes["y"],
                    tz * self.end_effector_step_sizes["z"],
                ],
                dtype=float,
            )
            r_abs = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
            desired = np.eye(4, dtype=float)
            desired[:3, :3] = ref[:3, :3] @ r_abs
            desired[:3, 3] = ref[:3, 3] + delta_p

            self._command_when_disabled = desired.copy()
        else:
            # While disabled, keep sending the same command to avoid drift.
            if self._command_when_disabled is None:
                # If we've never had an enabled command yet, freeze current FK pose once.
                self._command_when_disabled = t_curr.copy()
            desired = self._command_when_disabled.copy()

        # Write action fields
        pos = desired[:3, 3]
        tw = Rotation.from_matrix(desired[:3, :3]).as_rotvec()
        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(tw[0])
        action["ee.wy"] = float(tw[1])
        action["ee.wz"] = float(tw[2])
        action["ee.gripper_vel"] = gripper_vel

        self._prev_enabled = enabled
        return action

    def reset(self):
        """Resets the internal state of the processor."""
        self._prev_enabled = False
        self.reference_ee_pose = None
        self._command_when_disabled = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION].pop(f"{feat}", None)

        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_vel"]:
            features[PipelineFeatureType.ACTION][f"ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


@ProcessorStepRegistry.register("ee_bounds_and_safety")
@dataclass
class EEBoundsAndSafety(RobotActionProcessorStep):
    """
    Clips the end-effector pose to predefined bounds and checks for unsafe jumps.

    This step ensures that the target end-effector pose remains within a safe operational workspace.
    It also moderates the command to prevent large, sudden movements between consecutive steps.

    Attributes:
        end_effector_bounds: A dictionary with "min" and "max" keys for position clipping.
        max_ee_step_m: The maximum allowed change in position (in meters) between steps.
        _last_pos: Internal state storing the last commanded position.
    """

    end_effector_bounds: dict
    max_ee_step_m: float = 0.05
    max_orientation_step_rad: float | None = 0.35
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_twist: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        x = action["ee.x"]
        y = action["ee.y"]
        z = action["ee.z"]
        wx = action["ee.wx"]
        wy = action["ee.wy"]
        wz = action["ee.wz"]
        # TODO(Steven): ee.gripper_vel does not need to be bounded

        if None in (x, y, z, wx, wy, wz):
            raise ValueError(
                "Missing required end-effector pose components: x, y, z, wx, wy, wz must all be present in action"
            )

        pos = np.array([x, y, z], dtype=float)
        twist = np.array([wx, wy, wz], dtype=float)

        # Clip position
        pos = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        # Check for jumps in position
        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_m and n > 0:
                print(f"WARNING: EE jump {n:.3f}m > {self.max_ee_step_m}m, clamping.")
                pos = self._last_pos + dpos * (self.max_ee_step_m / n)

        # Check for jumps in orientation
        if self.max_orientation_step_rad is not None and self._last_twist is not None:
            dtwist = twist - self._last_twist
            n = float(np.linalg.norm(dtwist))
            if n > self.max_orientation_step_rad and n > 0:
                print(
                    f"WARNING: EE orientation jump {n:.3f}rad > {self.max_orientation_step_rad}rad, clamping."
                )
                twist = self._last_twist + dtwist * (self.max_orientation_step_rad / n)

        self._last_pos = pos
        self._last_twist = twist

        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(twist[0])
        action["ee.wy"] = float(twist[1])
        action["ee.wz"] = float(twist[2])
        return action

    def reset(self):
        """Resets the last known position and orientation."""
        self._last_pos = None
        self._last_twist = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("inverse_kinematics_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints(RobotActionProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using inverse kinematics (IK).

    This step translates a Cartesian command (position and orientation of the end-effector) into
    the corresponding joint-space commands for each motor.

    Attributes:
        kinematics: The robot's kinematic model for inverse kinematics.
        motor_names: A list of motor names for which to compute joint positions.
        q_curr: Internal state storing the last accepted IK solution.
        initial_guess_current_joints: If True, prefer the robot's measured joint state as the main IK seed.
            If False, prefer the last accepted IK solution while still considering the current state as a backup.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True
    max_joint_delta_deg: float | dict[str, float] | None = 35.0
    seed_joint_offsets_deg: float | dict[str, float] | None = field(
        default_factory=lambda: DEFAULT_SO_ARM_IK_SEED_OFFSETS_DEG.copy()
    )
    position_tolerance_m: float = 0.02
    orientation_tolerance_rad: float = 0.35
    fallback_to_current_joints_on_invalid: bool = True

    def action(self, action: RobotAction) -> RobotAction:
        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = _ordered_joint_positions(observation, self.motor_names)

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        q_target = _solve_best_ik_solution(
            kinematics=self.kinematics,
            motor_names=self.motor_names,
            current_joint_pos=q_raw,
            desired_pose=t_des,
            previous_solution=self.q_curr,
            prefer_current_joints=self.initial_guess_current_joints,
            max_joint_delta_deg=self.max_joint_delta_deg,
            seed_joint_offsets_deg=self.seed_joint_offsets_deg,
            position_tolerance_m=self.position_tolerance_m,
            orientation_tolerance_rad=self.orientation_tolerance_rad,
            fallback_to_current_joints_on_invalid=self.fallback_to_current_joints_on_invalid,
        )
        self.q_curr = q_target

        # TODO: This is sentitive to order of motor_names = q_target mapping
        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """Resets the initial guess for the IK solver."""
        self.q_curr = None


@ProcessorStepRegistry.register("gripper_velocity_to_joint")
@dataclass
class GripperVelocityToJoint(RobotActionProcessorStep):
    """
    Converts a gripper velocity command into a target gripper joint position.

    This step integrates a normalized velocity command over time to produce a position command,
    taking the current gripper position as a starting point. It also supports a discrete mode
    where integer actions map to open, close, or no-op.

    Attributes:
        motor_names: A list of motor names, which must include 'gripper'.
        speed_factor: A scaling factor to convert the normalized velocity command to a position change.
        clip_min: The minimum allowed gripper joint position.
        clip_max: The maximum allowed gripper joint position.
        discrete_gripper: If True, treat the input action as discrete (0: open, 1: close, 2: stay).
    """

    speed_factor: float = 20.0
    clip_min: float = 0.0
    clip_max: float = 100.0
    discrete_gripper: bool = False

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        gripper_vel = action.pop("ee.gripper_vel")

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if "gripper.pos" not in observation:
            raise ValueError("Joints observation must include 'gripper.pos' for gripper velocity integration")

        gripper_curr = float(observation["gripper.pos"])

        if self.discrete_gripper:
            # Discrete gripper actions are in [0, 1, 2]
            # 0: open, 1: close, 2: stay
            # We need to shift them to [-1, 0, 1] and then scale them to clip_max
            gripper_vel = (gripper_vel - 1) * self.clip_max

        # Compute desired gripper position
        delta = gripper_vel * float(self.speed_factor)
        gripper_pos = float(np.clip(gripper_curr + delta, self.clip_min, self.clip_max))
        action["ee.gripper_pos"] = gripper_pos

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION].pop("ee.gripper_vel", None)
        features[PipelineFeatureType.ACTION]["ee.gripper_pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )

        return features


def compute_forward_kinematics_joints_to_ee(
    joints: dict[str, Any], kinematics: RobotKinematics, motor_names: list[str]
) -> dict[str, Any]:
    motor_joint_values = [joints[f"{n}.pos"] for n in motor_names]

    q = np.array(motor_joint_values, dtype=float)
    t = kinematics.forward_kinematics(q)
    pos = t[:3, 3]
    tw = Rotation.from_matrix(t[:3, :3]).as_rotvec()
    gripper_pos = joints["gripper.pos"]
    for n in motor_names:
        joints.pop(f"{n}.pos")
    joints["ee.x"] = float(pos[0])
    joints["ee.y"] = float(pos[1])
    joints["ee.z"] = float(pos[2])
    joints["ee.wx"] = float(tw[0])
    joints["ee.wy"] = float(tw[1])
    joints["ee.wz"] = float(tw[2])
    joints["ee.gripper_pos"] = float(gripper_pos)
    return joints


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_observation")
@dataclass
class ForwardKinematicsJointsToEEObservation(ObservationProcessorStep):
    """
    Computes the end-effector pose from joint positions using forward kinematics (FK).

    This step is typically used to add the robot's Cartesian pose to the observation space,
    which can be useful for visualization or as an input to a policy.

    Attributes:
        kinematics: The robot's kinematic model.
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def observation(self, observation: RobotObservation) -> RobotObservation:
        return compute_forward_kinematics_joints_to_ee(observation, self.kinematics, self.motor_names)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We only use the ee pose in the dataset, so we don't need the joint positions
        for n in self.motor_names:
            features[PipelineFeatureType.OBSERVATION].pop(f"{n}.pos", None)
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.OBSERVATION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_action")
@dataclass
class ForwardKinematicsJointsToEEAction(RobotActionProcessorStep):
    """
    Computes the end-effector pose from joint positions using forward kinematics (FK).

    This step is typically used to add the robot's Cartesian pose to the observation space,
    which can be useful for visualization or as an input to a policy.

    Attributes:
        kinematics: The robot's kinematic model.
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def action(self, action: RobotAction) -> RobotAction:
        return compute_forward_kinematics_joints_to_ee(action, self.kinematics, self.motor_names)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We only use the ee pose in the dataset, so we don't need the joint positions
        for n in self.motor_names:
            features[PipelineFeatureType.ACTION].pop(f"{n}.pos", None)
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register(name="forward_kinematics_joints_to_ee")
@dataclass
class ForwardKinematicsJointsToEE(ProcessorStep):
    kinematics: RobotKinematics
    motor_names: list[str]

    def __post_init__(self):
        self.joints_to_ee_action_processor = ForwardKinematicsJointsToEEAction(
            kinematics=self.kinematics, motor_names=self.motor_names
        )
        self.joints_to_ee_observation_processor = ForwardKinematicsJointsToEEObservation(
            kinematics=self.kinematics, motor_names=self.motor_names
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if transition.get(TransitionKey.ACTION) is not None:
            transition = self.joints_to_ee_action_processor(transition)
        if transition.get(TransitionKey.OBSERVATION) is not None:
            transition = self.joints_to_ee_observation_processor(transition)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if features[PipelineFeatureType.ACTION] is not None:
            features = self.joints_to_ee_action_processor.transform_features(features)
        if features[PipelineFeatureType.OBSERVATION] is not None:
            features = self.joints_to_ee_observation_processor.transform_features(features)
        return features


@ProcessorStepRegistry.register("inverse_kinematics_rl_step")
@dataclass
class InverseKinematicsRLStep(ProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using inverse kinematics (IK).

    This is modified from the InverseKinematicsEEToJoints step to be used in the RL pipeline.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True
    max_joint_delta_deg: float | dict[str, float] | None = 35.0
    seed_joint_offsets_deg: float | dict[str, float] | None = field(
        default_factory=lambda: DEFAULT_SO_ARM_IK_SEED_OFFSETS_DEG.copy()
    )
    position_tolerance_m: float = 0.02
    orientation_tolerance_rad: float = 0.35
    fallback_to_current_joints_on_invalid: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = dict(transition)
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("Action is required for InverseKinematicsEEToJoints")
        action = dict(action)

        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = new_transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = _ordered_joint_positions(observation, self.motor_names)

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        q_target = _solve_best_ik_solution(
            kinematics=self.kinematics,
            motor_names=self.motor_names,
            current_joint_pos=q_raw,
            desired_pose=t_des,
            previous_solution=self.q_curr,
            prefer_current_joints=self.initial_guess_current_joints,
            max_joint_delta_deg=self.max_joint_delta_deg,
            seed_joint_offsets_deg=self.seed_joint_offsets_deg,
            position_tolerance_m=self.position_tolerance_m,
            orientation_tolerance_rad=self.orientation_tolerance_rad,
            fallback_to_current_joints_on_invalid=self.fallback_to_current_joints_on_invalid,
        )
        self.q_curr = q_target

        # TODO: This is sentitive to order of motor_names = q_target mapping
        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        new_transition[TransitionKey.ACTION] = action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data["IK_solution"] = q_target
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """Resets the initial guess for the IK solver."""
        self.q_curr = None
