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
SO101 EE 位姿空间数据采集脚本，参照 lerobot-record 流程编写。

使用 leader-follower 遥操作录制末端执行器空间的数据集。
观测和动作通过 FK/IK 在 EE 空间表示。

键盘控制：
- 右箭头 →：结束当前 episode，进入下一集
- 左箭头 ←：结束当前 episode 并重新录制
- Esc：停止整个录制流程
"""

from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
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
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun

# ========== 用户配置 ==========
FOLLOWER_PORT = "/dev/ttyACM0"
LEADER_PORT = "/dev/ttyACM1"
CAMERA_INDEXES = {"camera1": 3, "camera2": 5}

DATASET_NAME = "hanoi_ee"
DATASET_ROOT = Path("/home/sr/datasets") / DATASET_NAME

NUM_EPISODES = 50
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 60
TASK_DESCRIPTION = "Transfer the top disk from the left pillar to the right pillar."

# 数据集选项（与 lerobot-record 的 DatasetRecordConfig 对齐）
USE_VIDEOS = True
PUSH_TO_HUB = False
NUM_IMAGE_WRITER_PROCESSES = 0
NUM_IMAGE_WRITER_THREADS_PER_CAMERA = 4
VIDEO_ENCODING_BATCH_SIZE = 1
VCODEC = "libsvtav1"

# 运动学配置
URDF_PATH = "/home/sr/SO-ARM100/Simulation/SO101"
TARGET_FRAME_NAME = "gripper_frame_link"
EE_BOUNDS_MIN = [-1.0, -1.0, -1.0]
EE_BOUNDS_MAX = [1.0, 1.0, 1.0]
MAX_EE_STEP_M = 0.10

# 显示与音效
DISPLAY_DATA = True
PLAY_SOUNDS = True
# ==============================


def build_ee_processors(follower, leader):
    """构建 EE 位姿空间的 FK/IK 处理管线。"""
    follower_kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=TARGET_FRAME_NAME,
        joint_names=list(follower.bus.motors.keys()),
    )
    leader_kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=TARGET_FRAME_NAME,
        joint_names=list(leader.bus.motors.keys()),
    )

    # FK: follower 关节 → EE 观测
    follower_joints_to_ee = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=follower_kin, motor_names=list(follower.bus.motors.keys())
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # FK: leader 关节 → EE 动作
    leader_joints_to_ee = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kin, motor_names=list(leader.bus.motors.keys())
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # IK: EE 动作 → follower 关节
    ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": EE_BOUNDS_MIN, "max": EE_BOUNDS_MAX},
                max_ee_step_m=MAX_EE_STEP_M,
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kin,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    return leader_joints_to_ee, ee_to_follower_joints, follower_joints_to_ee


def main():
    init_logging()

    camera_config = {
        name: OpenCVCameraConfig(index_or_path=idx, width=640, height=480, fps=FPS)
        for name, idx in CAMERA_INDEXES.items()
    }
    follower_config = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id="so_follower_arm",
        cameras=camera_config,
        use_degrees=True,
    )
    leader_config = SO101LeaderConfig(port=LEADER_PORT, id="so_leader_arm")

    follower = SO101Follower(follower_config)
    leader = SO101Leader(leader_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = build_ee_processors(
        follower, leader
    )

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=leader.action_features),
            use_videos=USE_VIDEOS,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=follower.observation_features),
            use_videos=USE_VIDEOS,
        ),
    )

    dataset = None
    listener = None

    try:
        dataset = LeRobotDataset.create(
            DATASET_NAME,
            FPS,
            root=DATASET_ROOT,
            robot_type=follower.name,
            features=dataset_features,
            use_videos=USE_VIDEOS,
            image_writer_processes=NUM_IMAGE_WRITER_PROCESSES,
            image_writer_threads=NUM_IMAGE_WRITER_THREADS_PER_CAMERA * len(follower.cameras),
            batch_encoding_size=VIDEO_ENCODING_BATCH_SIZE,
            vcodec=VCODEC,
        )

        follower.connect()
        leader.connect()

        listener, events = init_keyboard_listener()

        if DISPLAY_DATA:
            init_rerun(session_name="so101_ee_record")

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", PLAY_SOUNDS)

                record_loop(
                    robot=follower,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=leader,
                    dataset=dataset,
                    control_time_s=EPISODE_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=DISPLAY_DATA,
                )

                if not events["stop_recording"] and (
                    (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", PLAY_SOUNDS)
                    record_loop(
                        robot=follower,
                        events=events,
                        fps=FPS,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=leader,
                        control_time_s=RESET_TIME_SEC,
                        single_task=TASK_DESCRIPTION,
                        display_data=DISPLAY_DATA,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", PLAY_SOUNDS)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

    finally:
        log_say("Stop recording", PLAY_SOUNDS, blocking=True)

        if dataset:
            dataset.finalize()

        if follower.is_connected:
            follower.disconnect()
        if leader.is_connected:
            leader.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if PUSH_TO_HUB and dataset:
            dataset.push_to_hub()

        log_say("Exiting", PLAY_SOUNDS)


if __name__ == "__main__":
    main()
