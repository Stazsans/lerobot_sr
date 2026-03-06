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
SO101 EE 位姿空间异步推理评估脚本。

使用 gRPC 异步推理架构进行策略评估：
- 策略推理在远程 PolicyServer 上执行（可部署在 GPU 机器）
- 本脚本作为客户端，处理机器人控制和 FK/IK 转换
- 推理与控制异步运行，机器人保持高帧率运动

使用前：
1. 在另一个终端启动 PolicyServer：
   python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080

2. 修改以下常量：
   - FOLLOWER_PORT: follower 串口路径
   - CAMERA_INDEXES: 摄像头索引
   - MODEL_PATH: 训练好的模型路径
   - SERVER_ADDRESS: 策略服务器地址
"""

import pickle  # nosec
import threading
import time
from queue import Empty, Queue

import grpc
import torch

from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    map_robot_keys_to_lerobot_features,
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
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
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

# ========== 用户配置 ==========
FOLLOWER_PORT = "/dev/ttyACM0"
CAMERA_INDEXES = {"camera1": 2, "camera2": 4}

MODEL_PATH = "./outputs/act_so101_ee/checkpoints/last/pretrained_model"
POLICY_TYPE = "act"
SERVER_ADDRESS = "127.0.0.1:8080"

FPS = 30
ACTIONS_PER_CHUNK = 50
CHUNK_SIZE_THRESHOLD = 0.5
TASK_DESCRIPTION = "Transfer the top disk from the left pillar to the right pillar."

POLICY_DEVICE = "cuda"
# ==============================

EE_KEYS = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"]


class EEAsyncEvaluator:
    """异步推理评估器，集成 FK/IK 转换。

    在 RobotClient 的基础上增加：
    - 观测发送前自动应用 FK（关节 → EE）
    - 动作执行前自动应用 IK（EE → 关节）
    """

    def __init__(self, robot, fk_processor, ik_processor, server_address, policy_config):
        self.robot = robot
        self.fk_processor = fk_processor
        self.ik_processor = ik_processor

        self.channel = grpc.insecure_channel(
            server_address, grpc_channel_options(initial_backoff=f"{1/FPS:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.policy_config = policy_config

        self.action_queue = Queue()
        self.action_chunk_size = ACTIONS_PER_CHUNK
        self.latest_action = -1
        self.latest_action_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.must_go = threading.Event()
        self.must_go.set()
        self.start_barrier = threading.Barrier(2)

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        try:
            self.stub.Ready(services_pb2.Empty())
            policy_config_bytes = pickle.dumps(self.policy_config)
            self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=policy_config_bytes))
            self.shutdown_event.clear()
            print(f"Connected to policy server at {SERVER_ADDRESS}")
            return True
        except grpc.RpcError as e:
            print(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        self.shutdown_event.set()
        self.robot.disconnect()
        self.channel.close()

    def receive_actions(self):
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

            except grpc.RpcError as e:
                print(f"Error receiving actions: {e}")

    def _ready_to_send_observation(self):
        return self.action_queue.qsize() / max(self.action_chunk_size, 1) <= CHUNK_SIZE_THRESHOLD

    def control_loop(self, task):
        self.start_barrier.wait()
        print("Control loop starting...")

        timestep = 0
        while self.running:
            t0 = time.perf_counter()

            # 1. 执行动作（如果有）
            try:
                timed_action = self.action_queue.get_nowait()

                # EE 动作张量 → EE 动作字典
                action_tensor = timed_action.get_action()
                ee_action = {key: action_tensor[i].item() for i, key in enumerate(EE_KEYS)}

                # 获取当前关节观测（IK 需要）
                robot_obs = self.robot.get_observation()

                # IK: EE → 关节
                joint_action = self.ik_processor((ee_action, robot_obs))

                self.robot.send_action(joint_action)

                with self.latest_action_lock:
                    self.latest_action = timed_action.get_timestep()

            except Empty:
                pass

            # 2. 发送观测（按需）
            if self._ready_to_send_observation():
                robot_obs = self.robot.get_observation()

                # FK: 关节 → EE
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
                except grpc.RpcError as e:
                    print(f"Error sending observation: {e}")

            timestep += 1
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, 1 / FPS - elapsed))


def main():
    # 配置摄像头和机器人
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
    robot = SO101Follower(follower_config)

    kinematics_solver = RobotKinematics(
        urdf_path="so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # FK: 关节观测 → EE 观测
    fk_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=kinematics_solver, motor_names=list(robot.bus.motors.keys())
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # IK: EE 动作 → 关节动作
    ik_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # 构建 EE 特征映射（用于告知服务器观测格式）
    ee_hw_features = {key: float for key in EE_KEYS}
    # 加入摄像头特征
    cam_features = {k: v for k, v in robot.observation_features.items() if isinstance(v, tuple)}
    ee_hw_features.update(cam_features)
    ee_lerobot_features = hw_to_dataset_features(ee_hw_features, "observation", use_video=False)

    policy_config = RemotePolicyConfig(
        policy_type=POLICY_TYPE,
        pretrained_name_or_path=MODEL_PATH,
        lerobot_features=ee_lerobot_features,
        actions_per_chunk=ACTIONS_PER_CHUNK,
        device=POLICY_DEVICE,
    )

    robot.connect()

    evaluator = EEAsyncEvaluator(
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
            print("\nStopping evaluation...")
        finally:
            evaluator.stop()
            action_thread.join(timeout=3)
            print("Evaluation finished.")


if __name__ == "__main__":
    main()
