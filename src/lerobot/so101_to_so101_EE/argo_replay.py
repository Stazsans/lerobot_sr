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
Replay an Argo EE-space SO101 dataset on the real follower arm.

This is the Argo FK/IK counterpart to `replay.py`. It expects dataset actions in
the EE feature order recorded by `argo_record.py` and converts each EE action to
follower joints with Argo IK.
"""

from __future__ import annotations

import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.so101_to_so101_EE.argo_record import build_argo_ee_processors, build_argo_kinematics
from lerobot.so101_to_so101_EE.argo_real_ik import FOLLOWER_ID, FOLLOWER_PORT
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

# ========== User configuration ==========
DATASET_NAME = "argo_ee"
DATASET_ROOT = Path("/home/seer/datasets") / DATASET_NAME
EPISODE_IDX = 0
# ========================================


def dataset_action_at(dataset: LeRobotDataset, actions, idx: int) -> dict[str, float]:
    return {
        name: float(actions[idx][ACTION][i])
        for i, name in enumerate(dataset.features[ACTION]["names"])
    }


def main() -> None:
    robot = SO101Follower(
        SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, use_degrees=True)
    )
    kin, robot_model, robot_utils, argo_joint_names = build_argo_kinematics()
    _, robot_action_processor, _ = build_argo_ee_processors(
        kin=kin,
        robot_model=robot_model,
        robot_utils=robot_utils,
        argo_joint_names=argo_joint_names,
    )

    dataset = LeRobotDataset(DATASET_NAME, root=DATASET_ROOT, episodes=[EPISODE_IDX])
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == EPISODE_IDX)
    actions = episode_frames.select_columns(ACTION)

    robot.connect()

    try:
        if not robot.is_connected:
            raise ValueError("Robot is not connected.")

        print("Starting Argo replay loop...")
        log_say(f"Replaying episode {EPISODE_IDX}")
        for idx in range(len(episode_frames)):
            t0 = time.perf_counter()

            ee_action = dataset_action_at(dataset, actions, idx)
            robot_obs = robot.get_observation()
            joint_action = robot_action_processor((ee_action, robot_obs))
            _ = robot.send_action(joint_action)

            precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
