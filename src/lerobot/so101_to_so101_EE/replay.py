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
SO101 EE 位姿空间回放脚本。

读取本地数据集中的 EE 动作，通过 IK 转为关节动作并执行。
用于验证采集的数据集质量。

使用前修改以下常量：
- FOLLOWER_PORT: 串口路径
- DATASET_NAME / DATASET_ROOT: 数据集名称和路径
- EPISODE_IDX: 要回放的 episode 索引
"""

import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    InverseKinematicsEEToJoints,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

# ========== 用户配置 ==========
FOLLOWER_PORT = "/dev/ttyACM0"

DATASET_NAME = "hanoi_ee"
DATASET_ROOT = Path("/home/sr/datasets") / DATASET_NAME
EPISODE_IDX = 0
# ==============================


def main():
    robot_config = SO101FollowerConfig(
        port=FOLLOWER_PORT, id="so_follower_arm", use_degrees=True
    )
    robot = SO101Follower(robot_config)

    kinematics_solver = RobotKinematics(
        urdf_path="/home/sr/SO-ARM100/Simulation/SO101",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # IK: EE 动作 → 关节动作
    robot_ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=False,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # 从本地路径加载数据集
    dataset = LeRobotDataset(DATASET_NAME, root=DATASET_ROOT, episodes=[EPISODE_IDX])
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == EPISODE_IDX)
    actions = episode_frames.select_columns(ACTION)

    robot.connect()

    try:
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        print("Starting replay loop...")
        log_say(f"Replaying episode {EPISODE_IDX}")
        for idx in range(len(episode_frames)):
            t0 = time.perf_counter()

            ee_action = {
                name: float(actions[idx][ACTION][i])
                for i, name in enumerate(dataset.features[ACTION]["names"])
            }

            robot_obs = robot.get_observation()
            joint_action = robot_ee_to_joints_processor((ee_action, robot_obs))
            _ = robot.send_action(joint_action)

            precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
