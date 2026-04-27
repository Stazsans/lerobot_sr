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
SO101 EE-space recording with Argo-Robot/controls FK/IK.

This mirrors `record.py`, but leader FK and follower IK are computed with the
Argo URDF implementation used by `argo_teleoperate.py`. Dataset actions and
observations stay in EE pose space: `ee.x/y/z/wx/wy/wz/gripper_pos`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import (
    EnvTransition,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    TransitionKey,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.processor.pipeline import ProcessorStep
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.so101_to_so101_EE.argo_real_ik import (
    ARGO_TARGET_LINK_NAME,
    ARGO_URDF_PATH,
    DEFAULT_FPS,
    DEFAULT_IK_GAIN,
    FOLLOWER_ID,
    FOLLOWER_PORT,
    argo_q_rad_to_robot_action,
    import_argo_controls,
    observation_to_argo_q_rad,
)
from lerobot.so101_to_so101_EE.argo_teleoperate import (
    DEFAULT_IK_ITERATIONS,
    LEADER_ID,
    LEADER_PORT,
    argo_joint_names_from_model,
    solve_fast_argo_ik,
    validate_argo_joint_names,
)
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.rotation import Rotation
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun

# ========== User configuration ==========
CAMERA_INDEXES = {}

DATASET_NAME = "argo_ee"
DATASET_ROOT = Path("/home/seer/datasets") / DATASET_NAME

NUM_EPISODES = 50
FPS = DEFAULT_FPS
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 60
TASK_DESCRIPTION = "Transfer the top disk from the left pillar to the right pillar."

USE_VIDEOS = False
PUSH_TO_HUB = False
NUM_IMAGE_WRITER_PROCESSES = 0
NUM_IMAGE_WRITER_THREADS_PER_CAMERA = 4
VIDEO_ENCODING_BATCH_SIZE = 1
VCODEC = "libsvtav1"

DISPLAY_DATA = True
PLAY_SOUNDS = False
# ========================================

EE_FEATURE_NAMES = ("x", "y", "z", "wx", "wy", "wz", "gripper_pos")


def pose_to_ee_dict(pose: np.ndarray, *, gripper_pos: float) -> dict[str, float]:
    wx, wy, wz = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
    return {
        "ee.x": float(pose[0, 3]),
        "ee.y": float(pose[1, 3]),
        "ee.z": float(pose[2, 3]),
        "ee.wx": float(wx),
        "ee.wy": float(wy),
        "ee.wz": float(wz),
        "ee.gripper_pos": float(gripper_pos),
    }


def ee_dict_to_pose(ee_action: RobotAction) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [float(ee_action[f"ee.{axis}"]) for axis in ("x", "y", "z")]
    pose[:3, :3] = Rotation.from_rotvec(
        [float(ee_action["ee.wx"]), float(ee_action["ee.wy"]), float(ee_action["ee.wz"])]
    ).as_matrix()
    return pose


def add_ee_features(
    features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    *,
    key: PipelineFeatureType,
    feature_type: FeatureType,
) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
    for name in EE_FEATURE_NAMES:
        features[key][f"ee.{name}"] = PolicyFeature(type=feature_type, shape=(1,))
    return features


@dataclass
class ArgoFKJointsToEE(ProcessorStep):
    kin: Any
    robot_model: Any
    argo_joint_names: list[str]
    target_link_name: str

    def _convert(self, values: dict[str, float]) -> dict[str, float]:
        q_rad = observation_to_argo_q_rad(values, self.argo_joint_names)
        pose = self.kin.forward_kinematics(
            self.robot_model,
            q_rad,
            target_link_name=self.target_link_name,
        )
        return pose_to_ee_dict(pose, gripper_pos=float(values["gripper.pos"]))

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            new_transition[TransitionKey.ACTION] = self._convert(action)
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            processed_observation = dict(observation)
            motor_only_observation = {
                f"{joint_name}.pos": observation[f"{joint_name}.pos"]
                for joint_name in self.argo_joint_names
            }
            processed_observation.update(self._convert(motor_only_observation))
            new_transition[TransitionKey.OBSERVATION] = processed_observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if features[PipelineFeatureType.ACTION] is not None:
            for joint_name in self.argo_joint_names:
                features[PipelineFeatureType.ACTION].pop(f"{joint_name}.pos", None)
            features = add_ee_features(
                features,
                key=PipelineFeatureType.ACTION,
                feature_type=FeatureType.ACTION,
            )
        if features[PipelineFeatureType.OBSERVATION] is not None:
            for joint_name in self.argo_joint_names:
                features[PipelineFeatureType.OBSERVATION].pop(f"{joint_name}.pos", None)
            features = add_ee_features(
                features,
                key=PipelineFeatureType.OBSERVATION,
                feature_type=FeatureType.STATE,
            )
        return features


@dataclass
class ArgoIKEEToJoints(ProcessorStep):
    kin: Any
    robot_model: Any
    robot_utils: Any
    argo_joint_names: list[str]
    target_link_name: str
    gain: float = DEFAULT_IK_GAIN
    iterations: int = DEFAULT_IK_ITERATIONS
    previous_q_goal: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if action is None or not isinstance(action, dict):
            raise ValueError("ArgoIKEEToJoints requires an EE action dictionary.")
        if observation is None or not isinstance(observation, dict):
            raise ValueError("ArgoIKEEToJoints requires a follower observation dictionary.")

        measured_q = observation_to_argo_q_rad(observation, self.argo_joint_names)
        ik_result = solve_fast_argo_ik(
            kin=self.kin,
            robot_model=self.robot_model,
            robot_utils=self.robot_utils,
            measured_q_rad=measured_q,
            previous_q_goal_rad=self.previous_q_goal,
            target_pose=ee_dict_to_pose(action),
            target_link_name=self.target_link_name,
            argo_joint_names=self.argo_joint_names,
            gripper_pos=float(action["ee.gripper_pos"]),
            gain=float(self.gain),
            iterations=int(self.iterations),
        )
        q_goal = ik_result["q"]
        self.previous_q_goal = q_goal.copy()
        new_transition[TransitionKey.ACTION] = argo_q_rad_to_robot_action(
            q_goal,
            self.argo_joint_names,
            gripper_pos=float(action["ee.gripper_pos"]),
        )
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for name in EE_FEATURE_NAMES:
            features[PipelineFeatureType.ACTION].pop(f"ee.{name}", None)
        for joint_name in self.argo_joint_names:
            features[PipelineFeatureType.ACTION][f"{joint_name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(1,),
            )
        return features

    def reset(self) -> None:
        self.previous_q_goal = None


def build_argo_kinematics() -> tuple[Any, Any, Any, list[str]]:
    URDFLoader, RobotModel, URDFKinematics, RobotUtils = import_argo_controls()
    if not ARGO_URDF_PATH.exists():
        raise FileNotFoundError(f"Argo SO101 URDF not found at {ARGO_URDF_PATH}")
    urdf_loader = URDFLoader()
    urdf_loader.load(str(ARGO_URDF_PATH))
    robot_model = RobotModel(urdf_loader)
    kin = URDFKinematics()
    argo_joint_names = argo_joint_names_from_model(robot_model)
    validate_argo_joint_names(argo_joint_names)
    return kin, robot_model, RobotUtils, argo_joint_names


def build_argo_ee_processors(
    *,
    kin: Any,
    robot_model: Any,
    robot_utils: Any,
    argo_joint_names: list[str],
) -> tuple[
    RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    RobotProcessorPipeline[RobotObservation, RobotObservation],
]:
    fk_step = ArgoFKJointsToEE(
        kin=kin,
        robot_model=robot_model,
        argo_joint_names=argo_joint_names,
        target_link_name=ARGO_TARGET_LINK_NAME,
    )
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[fk_step],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            ArgoIKEEToJoints(
                kin=kin,
                robot_model=robot_model,
                robot_utils=robot_utils,
                argo_joint_names=argo_joint_names,
                target_link_name=ARGO_TARGET_LINK_NAME,
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[fk_step],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return teleop_action_processor, robot_action_processor, robot_observation_processor


def main() -> None:
    init_logging()

    camera_config = {
        name: OpenCVCameraConfig(index_or_path=idx, width=640, height=480, fps=FPS)
        for name, idx in CAMERA_INDEXES.items()
    }
    follower = SO101Follower(
        SO101FollowerConfig(
            port=FOLLOWER_PORT,
            id=FOLLOWER_ID,
            cameras=camera_config,
            use_degrees=True,
        )
    )
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID, use_degrees=True))

    kin, robot_model, robot_utils, argo_joint_names = build_argo_kinematics()
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        build_argo_ee_processors(
            kin=kin,
            robot_model=robot_model,
            robot_utils=robot_utils,
            argo_joint_names=argo_joint_names,
        )
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
            init_rerun(session_name="so101_argo_ee_record")

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
