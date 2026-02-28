import logging
from functools import cached_property

from lerobot.cameras import Camera, make_cameras_from_configs
from lerobot.processor import RobotObservation, RobotAction
from lerobot.robots import Robot
from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower
from lerobot.robots.sr0.config_sr0 import SR0Config


class SR0(Robot):
    """ 仙工开源机器人 """
    config_class = SR0Config
    name = "SR0"

    def __init__(self, config: SR0Config):
        super().__init__(config)
        self.config = config

        # TODO 机械臂配置
        left_arm_config = SOFollowerRobotConfig()
        right_arm_config = SOFollowerRobotConfig()

        self.left_arm = SOFollower(left_arm_config)
        self.right_arm = SOFollower(right_arm_config)

        # 基座摄像头
        self.base_cameras = make_cameras_from_configs(config.base_cameras_config)

        self.camera = {**self.left_arm.cameras, **self.right_arm.cameras, **self.base_cameras}

    # TODO 继承 Robot 父类方法

    @cached_property
    def observation_features(self) -> dict:
        return {}

    @cached_property
    def action_features(self) -> dict:
        return {}

    @property
    def is_conected(self) -> bool:
        return True

    @property
    def is_disconnected(self) -> bool:
        return False

    def get_observation(self) -> RobotObservation:
        return RobotObservation()

    def send_action(self, action: RobotAction) -> RobotAction:
        return RobotAction(action)

    def connect(self, calibrate=True):
        self.left_arm.connect()
        self.right_arm.connect()
        self.top_camera.connect()

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        self.top_camera.disconnect()
