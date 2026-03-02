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

    # TODO 使用配置文件，减少终端命令长度
    config_path = "./sr0.yaml"

    def __init__(self, config: SR0Config):
        super().__init__(config)
        self.config = config

        # 机械臂配置
        left_arm_config = SOFollowerRobotConfig(
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            cameras=config.left_arm_config.cameras,
        )
        right_arm_config = SOFollowerRobotConfig(
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            cameras=config.right_arm_config.cameras,
        )

        self.left_arm = SOFollower(left_arm_config)
        self.right_arm = SOFollower(right_arm_config)

        # 基座摄像头
        self.base_cameras = make_cameras_from_configs(config.base_cameras_config)
        self.camera = {**self.left_arm.cameras, **self.right_arm.cameras, **self.base_cameras}

    # TODO 继承 Robot 父类方法

    @cached_property
    def observation_features(self) -> dict:
        """
        观测数据描述
        TODO 先用 joint 描述机械臂状态，后续补充 ee_pose
        TODO 增加底盘等状态描述
        """
        return {}

    @cached_property
    def action_features(self) -> dict:
        """
        动作指令描述
        TODO 增加 ee_pose
        """
        return {}

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def is_disconnected(self) -> bool:
        return False

    def get_observation(self) -> RobotObservation:
        """
        获取观测数据
        TODO 三个相机两个机械臂
        """
        return RobotObservation()

    def send_action(self, action: RobotAction) -> RobotAction:
        """
        发送动作指令
        TODO 增加 ee_pose，注意还需逆运动学解算
        """


        return RobotAction(action)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def connect(self, calibrate=True):
        self.left_arm.connect()
        self.right_arm.connect()
        for cam in self.base_cameras.values():
            cam.connect()

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        for cam in self.base_cameras.values():
            cam.disconnect()
