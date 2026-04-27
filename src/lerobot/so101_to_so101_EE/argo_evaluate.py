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
SO101 Argo EE-space async policy evaluation.

This mirrors `evaluate.py`, but uses the Argo FK/IK processors from
`argo_record.py` so policies trained on Argo EE datasets are evaluated with the
same kinematic semantics.
"""

from __future__ import annotations

import pickle  # nosec
import threading
import time
from queue import Empty, Queue

import grpc

from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.so101_to_so101_EE.argo_record import (
    EE_FEATURE_NAMES,
    build_argo_ee_processors,
    build_argo_kinematics,
)
from lerobot.so101_to_so101_EE.argo_real_ik import FOLLOWER_ID, FOLLOWER_PORT
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

# ========== User configuration ==========
CAMERA_INDEXES = {"camera1": 3, "camera2": 5}

MODEL_PATH = "/home/sr/outputs/train/hanoi_argo_ee_step1/checkpoints/016000/pretrained_model"
POLICY_TYPE = "act"
SERVER_ADDRESS = "192.168.10.73:8080"

FPS = 15
ACTIONS_PER_CHUNK = 150
CHUNK_SIZE_THRESHOLD = 0.8
TASK_DESCRIPTION = "Transfer the top disk from the left pillar to the right pillar."

POLICY_DEVICE = "cuda"

EE_OFFSET_X = 0.0
EE_OFFSET_Y = 0.0
EE_OFFSET_Z = 0.0
EE_OFFSET_WX = 0.0
EE_OFFSET_WY = 0.0
EE_OFFSET_WZ = 0.0
# ========================================

EE_KEYS = [f"ee.{name}" for name in EE_FEATURE_NAMES]


class ArgoEEAsyncEvaluator:
    def __init__(self, robot, fk_processor, ik_processor, server_address, policy_config):
        self.robot = robot
        self.fk_processor = fk_processor
        self.ik_processor = ik_processor

        self.channel = grpc.insecure_channel(
            server_address,
            grpc_channel_options(initial_backoff=f"{1 / FPS:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.policy_config = policy_config

        self.action_queue = Queue()
        self.action_chunk_size = ACTIONS_PER_CHUNK
        self.latest_action = 1
        self.latest_action_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.must_go = threading.Event()
        self.must_go.set()
        self.start_barrier = threading.Barrier(2)

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def start(self) -> bool:
        try:
            self.stub.Ready(services_pb2.Empty())
            policy_config_bytes = pickle.dumps(self.policy_config)
            self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=policy_config_bytes))
            self.shutdown_event.clear()
            print(f"Connected to policy server at {SERVER_ADDRESS}")
            return True
        except grpc.RpcError as exc:
            print(f"Failed to connect to policy server: {exc}")
            return False

    def stop(self) -> None:
        self.shutdown_event.set()
        self.robot.disconnect()
        self.channel.close()

    def receive_actions(self) -> None:
        self.start_barrier.wait()
        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                timed_actions: list[TimedAction] = pickle.loads(actions_chunk.data)  # nosec

                for action in timed_actions:
                    with self.latest_action_lock:
                        if action.get_timestep() <= self.latest_action:
                            continue
                    self.action_queue.put(action)

                self.must_go.set()

            except grpc.RpcError as exc:
                print(f"Error receiving actions: {exc}")

    def _ready_to_send_observation(self) -> bool:
        return self.action_queue.qsize() / max(self.action_chunk_size, 1) <= CHUNK_SIZE_THRESHOLD

    def control_loop(self, task: str) -> None:
        self.start_barrier.wait()
        print("Argo control loop starting...")

        while self.running:
            t0 = time.perf_counter()

            try:
                timed_action = self.action_queue.get_nowait()
                ee_action = self._timed_action_to_ee_action(timed_action)
                robot_obs = self.robot.get_observation()
                current_ee = self.fk_processor(robot_obs.copy())
                joint_action = self.ik_processor((ee_action, robot_obs))
                self.robot.send_action(joint_action)
                self._log_tracking_snapshot(ee_action, current_ee)

                with self.latest_action_lock:
                    self.latest_action = timed_action.get_timestep()

            except Empty:
                pass

            if self._ready_to_send_observation():
                self._send_observation(task)

            elapsed = time.perf_counter() - t0
            time.sleep(max(0, 1 / FPS - elapsed))

    def _timed_action_to_ee_action(self, timed_action: TimedAction) -> dict[str, float]:
        action_tensor = timed_action.get_action()
        ee_action = {key: action_tensor[i].item() for i, key in enumerate(EE_KEYS)}
        ee_action["ee.x"] += EE_OFFSET_X
        ee_action["ee.y"] += EE_OFFSET_Y
        ee_action["ee.z"] += EE_OFFSET_Z
        ee_action["ee.wx"] += EE_OFFSET_WX
        ee_action["ee.wy"] += EE_OFFSET_WY
        ee_action["ee.wz"] += EE_OFFSET_WZ
        return ee_action

    def _log_tracking_snapshot(self, ee_action: dict[str, float], current_ee: dict[str, float]) -> None:
        print(
            f"target x={ee_action['ee.x']:+.4f} y={ee_action['ee.y']:+.4f} z={ee_action['ee.z']:+.4f} | "
            f"actual x={current_ee['ee.x']:+.4f} y={current_ee['ee.y']:+.4f} z={current_ee['ee.z']:+.4f} | "
            f"diff x={ee_action['ee.x'] - current_ee['ee.x']:+.4f} "
            f"y={ee_action['ee.y'] - current_ee['ee.y']:+.4f} "
            f"z={ee_action['ee.z'] - current_ee['ee.z']:+.4f}"
        )

    def _send_observation(self, task: str) -> None:
        robot_obs = self.robot.get_observation()
        ee_obs = self.fk_processor(robot_obs)
        ee_obs["task"] = task

        with self.latest_action_lock:
            current_timestep = max(self.latest_action, 0)

        observation = TimedObservation(
            timestamp=time.time(),
            observation=ee_obs,
            timestep=current_timestep,
        )

        with_must_go = self.must_go.is_set() and self.action_queue.empty()
        observation.must_go = with_must_go
        if with_must_go:
            self.must_go.clear()

        try:
            obs_bytes = pickle.dumps(observation)
            obs_iterator = send_bytes_in_chunks(
                obs_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            self.stub.SendObservations(obs_iterator)
        except grpc.RpcError as exc:
            print(f"Error sending observation: {exc}")


def build_policy_config(robot: SO101Follower) -> RemotePolicyConfig:
    ee_hw_features = {key: float for key in EE_KEYS}
    cam_features = {
        key: value for key, value in robot.observation_features.items() if isinstance(value, tuple)
    }
    ee_hw_features.update(cam_features)
    ee_lerobot_features = hw_to_dataset_features(ee_hw_features, "observation", use_video=False)

    return RemotePolicyConfig(
        policy_type=POLICY_TYPE,
        pretrained_name_or_path=MODEL_PATH,
        lerobot_features=ee_lerobot_features,
        actions_per_chunk=ACTIONS_PER_CHUNK,
        device=POLICY_DEVICE,
    )


def main() -> None:
    camera_config = {
        name: OpenCVCameraConfig(index_or_path=idx, width=640, height=480, fps=30)
        for name, idx in CAMERA_INDEXES.items()
    }
    robot = SO101Follower(
        SO101FollowerConfig(
            port=FOLLOWER_PORT,
            id=FOLLOWER_ID,
            cameras=camera_config,
            use_degrees=True,
        )
    )

    kin, robot_model, robot_utils, argo_joint_names = build_argo_kinematics()
    _, ik_processor, fk_processor = build_argo_ee_processors(
        kin=kin,
        robot_model=robot_model,
        robot_utils=robot_utils,
        argo_joint_names=argo_joint_names,
    )
    policy_config = build_policy_config(robot)

    robot.connect()

    evaluator = ArgoEEAsyncEvaluator(
        robot=robot,
        fk_processor=fk_processor,
        ik_processor=ik_processor,
        server_address=SERVER_ADDRESS,
        policy_config=policy_config,
    )

    if evaluator.start():
        action_thread = threading.Thread(target=evaluator.receive_actions, daemon=True)
        action_thread.start()

        try:
            evaluator.control_loop(task=TASK_DESCRIPTION)
        except KeyboardInterrupt:
            print("\nStopping Argo evaluation...")
        finally:
            evaluator.stop()
            action_thread.join(timeout=3)
            print("Argo evaluation finished.")


if __name__ == "__main__":
    main()
