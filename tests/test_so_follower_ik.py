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

import numpy as np

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.processor import TransitionKey, create_transition
from lerobot.robots.so_follower.so_follower import clip_goal_positions_to_calibration
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEReferenceAndDelta,
    EEBoundsAndSafety,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
    InverseKinematicsRLStep,
)

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class BranchingFakeKinematics:
    def __init__(self):
        self.desired_pose = np.eye(4, dtype=float)

    def inverse_kinematics(self, current_joint_pos, desired_ee_pose, position_weight=1.0, orientation_weight=0.01):
        self.desired_pose = desired_ee_pose.copy()
        if current_joint_pos[0] > 50.0:
            return np.array([95.0, 20.0, 30.0, 40.0, 50.0, current_joint_pos[-1]], dtype=float)
        return np.array([11.0, 21.0, 31.0, 41.0, 51.0, current_joint_pos[-1]], dtype=float)

    def forward_kinematics(self, joint_pos_deg):
        return self.desired_pose.copy()


class InvalidPoseFakeKinematics:
    def __init__(self):
        self.desired_pose = np.eye(4, dtype=float)

    def inverse_kinematics(self, current_joint_pos, desired_ee_pose, position_weight=1.0, orientation_weight=0.01):
        self.desired_pose = desired_ee_pose.copy()
        return np.array([15.0, 25.0, 35.0, 45.0, 55.0, current_joint_pos[-1]], dtype=float)

    def forward_kinematics(self, joint_pos_deg):
        pose = self.desired_pose.copy()
        pose[0, 3] += 0.10
        return pose


class RecordingForwardKinematics:
    def __init__(self):
        self.last_joint_pos = None

    def forward_kinematics(self, joint_pos_deg):
        self.last_joint_pos = np.array(joint_pos_deg, dtype=float)
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = self.last_joint_pos[:3]
        return pose


def make_observation():
    return {
        "elbow_flex.pos": 30.0,
        "gripper.pos": 60.0,
        "wrist_roll.pos": 50.0,
        "shoulder_pan.pos": 10.0,
        "wrist_flex.pos": 40.0,
        "shoulder_lift.pos": 20.0,
    }


def make_ee_action(gripper_pos: float = 35.0):
    return {
        "ee.x": 0.20,
        "ee.y": 0.01,
        "ee.z": 0.18,
        "ee.wx": 0.0,
        "ee.wy": 1.57,
        "ee.wz": 0.0,
        "ee.gripper_pos": gripper_pos,
    }


def test_inverse_kinematics_prefers_safe_solution_and_motor_name_order():
    step = InverseKinematicsEEToJoints(
        kinematics=BranchingFakeKinematics(),
        motor_names=MOTOR_NAMES,
        initial_guess_current_joints=False,
        max_joint_delta_deg=15.0,
        seed_joint_offsets_deg=0.0,
    )
    step.q_curr = np.array([80.0, 80.0, 80.0, 80.0, 80.0, 60.0], dtype=float)

    transition = create_transition(action=make_ee_action(), observation=make_observation())
    result = step(transition)
    action = result[TransitionKey.ACTION]

    assert action["shoulder_pan.pos"] == 11.0
    assert action["shoulder_lift.pos"] == 21.0
    assert action["elbow_flex.pos"] == 31.0
    assert action["wrist_flex.pos"] == 41.0
    assert action["wrist_roll.pos"] == 51.0
    assert action["gripper.pos"] == 35.0


def test_inverse_kinematics_falls_back_to_current_joints_on_invalid_pose():
    step = InverseKinematicsEEToJoints(
        kinematics=InvalidPoseFakeKinematics(),
        motor_names=MOTOR_NAMES,
        initial_guess_current_joints=True,
        max_joint_delta_deg=15.0,
        seed_joint_offsets_deg=0.0,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.1,
        fallback_to_current_joints_on_invalid=True,
    )

    transition = create_transition(action=make_ee_action(gripper_pos=42.0), observation=make_observation())
    result = step(transition)
    action = result[TransitionKey.ACTION]

    assert action["shoulder_pan.pos"] == 10.0
    assert action["shoulder_lift.pos"] == 20.0
    assert action["elbow_flex.pos"] == 30.0
    assert action["wrist_flex.pos"] == 40.0
    assert action["wrist_roll.pos"] == 50.0
    assert action["gripper.pos"] == 42.0


def test_inverse_kinematics_rl_step_keeps_selected_solution_in_complementary_data():
    step = InverseKinematicsRLStep(
        kinematics=BranchingFakeKinematics(),
        motor_names=MOTOR_NAMES,
        initial_guess_current_joints=True,
        max_joint_delta_deg=15.0,
        seed_joint_offsets_deg=0.0,
    )

    transition = create_transition(action=make_ee_action(), observation=make_observation())
    result = step(transition)
    action = result[TransitionKey.ACTION]
    ik_solution = result[TransitionKey.COMPLEMENTARY_DATA]["IK_solution"]

    np.testing.assert_allclose(ik_solution, np.array([11.0, 21.0, 31.0, 41.0, 51.0, 60.0]))
    assert action["shoulder_pan.pos"] == 11.0
    assert action["wrist_roll.pos"] == 51.0
    assert action["gripper.pos"] == 35.0


def test_clip_goal_positions_to_calibration_clips_degree_joints_to_absolute_limits():
    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    calibration = {
        "shoulder_pan": MotorCalibration(id=1, drive_mode=0, homing_offset=0, range_min=1000, range_max=3000),
        "gripper": MotorCalibration(id=6, drive_mode=0, homing_offset=0, range_min=500, range_max=3500),
    }
    model_resolution_table = {"sts3215": 4096}

    clipped = clip_goal_positions_to_calibration(
        {"shoulder_pan": 120.0, "gripper": 120.0},
        motors=motors,
        calibration=calibration,
        model_resolution_table=model_resolution_table,
    )

    np.testing.assert_allclose(clipped["shoulder_pan"], 87.91208791208791)
    assert clipped["gripper"] == 100.0


def test_clip_goal_positions_to_calibration_preserves_in_range_targets():
    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    calibration = {
        "shoulder_pan": MotorCalibration(id=1, drive_mode=0, homing_offset=0, range_min=1000, range_max=3000),
        "gripper": MotorCalibration(id=6, drive_mode=0, homing_offset=0, range_min=500, range_max=3500),
    }
    model_resolution_table = {"sts3215": 4096}

    clipped = clip_goal_positions_to_calibration(
        {"shoulder_pan": 20.0, "gripper": 42.0},
        motors=motors,
        calibration=calibration,
        model_resolution_table=model_resolution_table,
    )

    assert clipped["shoulder_pan"] == 20.0
    assert clipped["gripper"] == 42.0


def test_ee_reference_and_delta_uses_motor_names_order_for_observation_fk():
    kinematics = RecordingForwardKinematics()
    step = EEReferenceAndDelta(
        kinematics=kinematics,
        end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
        motor_names=MOTOR_NAMES,
        use_latched_reference=False,
        use_ik_solution=False,
    )

    transition = create_transition(
        action={
            "enabled": True,
            "target_x": 0.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper_vel": 0.0,
        },
        observation=make_observation(),
    )

    result = step(transition)

    np.testing.assert_allclose(kinematics.last_joint_pos, np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
    action = result[TransitionKey.ACTION]
    assert action["ee.x"] == 10.0
    assert action["ee.y"] == 20.0
    assert action["ee.z"] == 30.0


def test_ee_reference_and_delta_prefers_ik_solution_when_enabled():
    kinematics = RecordingForwardKinematics()
    step = EEReferenceAndDelta(
        kinematics=kinematics,
        end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
        motor_names=MOTOR_NAMES,
        use_latched_reference=False,
        use_ik_solution=True,
    )

    transition = create_transition(
        action={
            "enabled": True,
            "target_x": 0.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper_vel": 0.0,
        },
        observation=make_observation(),
        complementary_data={"IK_solution": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])},
    )

    result = step(transition)

    np.testing.assert_allclose(kinematics.last_joint_pos, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    action = result[TransitionKey.ACTION]
    assert action["ee.x"] == 1.0
    assert action["ee.y"] == 2.0
    assert action["ee.z"] == 3.0


def test_ee_bounds_and_safety_clamps_orientation_jump():
    step = EEBoundsAndSafety(
        end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        max_ee_step_m=1.0,
        max_orientation_step_rad=0.2,
    )

    first = step(
        create_transition(
            action={
                "ee.x": 0.0,
                "ee.y": 0.0,
                "ee.z": 0.0,
                "ee.wx": 0.0,
                "ee.wy": 0.0,
                "ee.wz": 0.0,
            }
        )
    )
    second = step(
        create_transition(
            action={
                "ee.x": 0.0,
                "ee.y": 0.0,
                "ee.z": 0.0,
                "ee.wx": 1.0,
                "ee.wy": 0.0,
                "ee.wz": 0.0,
            }
        )
    )

    assert first[TransitionKey.ACTION]["ee.wx"] == 0.0
    np.testing.assert_allclose(second[TransitionKey.ACTION]["ee.wx"], 0.2)
    assert second[TransitionKey.ACTION]["ee.wy"] == 0.0
    assert second[TransitionKey.ACTION]["ee.wz"] == 0.0


def test_ee_bounds_and_safety_reset_clears_orientation_history():
    step = EEBoundsAndSafety(
        end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        max_ee_step_m=1.0,
        max_orientation_step_rad=0.2,
    )

    _ = step(
        create_transition(
            action={
                "ee.x": 0.0,
                "ee.y": 0.0,
                "ee.z": 0.0,
                "ee.wx": 0.0,
                "ee.wy": 0.0,
                "ee.wz": 0.0,
            }
        )
    )
    step.reset()
    result = step(
        create_transition(
            action={
                "ee.x": 0.0,
                "ee.y": 0.0,
                "ee.z": 0.0,
                "ee.wx": 1.0,
                "ee.wy": 0.0,
                "ee.wz": 0.0,
            }
        )
    )

    assert result[TransitionKey.ACTION]["ee.wx"] == 1.0


def test_gripper_velocity_to_joint_uses_named_gripper_observation_not_last_joint():
    step = GripperVelocityToJoint(speed_factor=2.0, clip_min=0.0, clip_max=100.0, discrete_gripper=False)

    transition = create_transition(
        action={"ee.gripper_vel": 3.0},
        observation={
            "gripper.pos": 60.0,
            "shoulder_pan.pos": 10.0,
            "wrist_roll.pos": 50.0,
            "elbow_flex.pos": 30.0,
        },
    )

    result = step(transition)

    assert result[TransitionKey.ACTION]["ee.gripper_pos"] == 66.0


def test_gripper_velocity_to_joint_discrete_mode_clips_using_gripper_position():
    step = GripperVelocityToJoint(speed_factor=1.0, clip_min=0.0, clip_max=100.0, discrete_gripper=True)

    transition = create_transition(
        action={"ee.gripper_vel": 0},
        observation={
            "shoulder_pan.pos": 10.0,
            "gripper.pos": 20.0,
            "wrist_roll.pos": 50.0,
        },
    )

    result = step(transition)

    assert result[TransitionKey.ACTION]["ee.gripper_pos"] == 0.0
