#!/usr/bin/env python

from dataclasses import dataclass
from lerobot.cameras import CameraConfig
from ..config import RobotConfig
from ..so_follower import SOFollowerConfig


@RobotConfig.register_subclass("sr0")
@dataclass
class SR0Config(RobotConfig):
    """ seer 开源机器人基础配置类"""

    # tips: SO101 已经包含了摄像头，每个臂一个，所以只需要额外配置一个三视角摄像头即可
    # TODO 是否需要增加不启用 左/右 臂的配置项，视为单臂机器人
    left_arm_config: SOFollowerConfig
    right_arm_config: SOFollowerConfig

    # TODO 底盘配置
    # classics_config = None

    # TODO 三视角摄像头配置
    top_camera_config: CameraConfig

    # TODO 机械臂基坐标系相对于底盘基坐标系的位置，可能会用到
    # left_arm_position = None
    # right_arm_position = None

    # TODO 雷达配置，可能会用到
    # lidar_config = None

    # TODO 机械臂表示方式 ee_pose or joint，可能会用到
    # arm_control_mode: str = "joint"

