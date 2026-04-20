# !/usr/bin/env python

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

"""
SO101 EE 位姿空间遥操作脚本。

Leader 关节角度经 FK 转为 EE 位姿指令，再经 IK 转为 follower 关节角度。
用于验证 FK/IK 管线和硬件连接。

使用前修改以下常量：
- FOLLOWER_PORT / LEADER_PORT: 串口路径
- URDF_PATH: 本地 so101_new_calib.urdf 路径
"""

import time
from pathlib import Path

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
    derive_ik_joint_preferences_from_robot,
)
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ========== 用户配置 ==========
FOLLOWER_PORT = "/dev/ttyACM0"
LEADER_PORT = "/dev/ttyACM1"
FPS = 30
REPO_ROOT = Path(__file__).resolve().parents[3]
URDF_PATH = REPO_ROOT / "third_party" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
# ==============================


def main():
    follower_config = SO101FollowerConfig(
        port=FOLLOWER_PORT, id="so_follower_arm", use_degrees=True
    )
    leader_config = SO101LeaderConfig(port=LEADER_PORT, id="so_leader_arm")

    follower = SO101Follower(follower_config)
    leader = SO101Leader(leader_config)
    ik_preferences = derive_ik_joint_preferences_from_robot(
        follower,
        motor_names=list(follower.bus.motors.keys()),
    )

    follower_kinematics_solver = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )

    leader_kinematics_solver = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=list(leader.bus.motors.keys()),
    )

    # FK: leader 关节 → EE 动作
    leader_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kinematics_solver, motor_names=list(leader.bus.motors.keys())
            ),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    # IK: EE 动作 → follower 关节
    ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=False,
                continuous_joint_names=ik_preferences.continuous_joint_names,
                joint_position_limits_deg=ik_preferences.joint_position_limits_deg,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    follower.connect()
    leader.connect()

    init_rerun(session_name="so101_ee_teleop")

    print("Starting teleop loop...")
    while True:
        t0 = time.perf_counter()

        robot_obs = follower.get_observation()
        leader_joints_obs = leader.get_action()

        leader_ee_act = leader_to_ee(leader_joints_obs)
        follower_joints_act = ee_to_follower_joints((leader_ee_act, robot_obs))

        _ = follower.send_action(follower_joints_act)

        log_rerun_data(observation=leader_ee_act, action=follower_joints_act)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
