from lerobot.cameras import Camera
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

        # TODO 三视角摄像头
        self.top_camera = Camera()

        self.camera = {**self.left_arm.cameras, **self.right_arm.cameras, **self.top_camera}