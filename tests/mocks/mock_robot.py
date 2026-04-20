#!/usr/bin/env python

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot, RobotConfig

TEST_TMP_ROOT = Path(__file__).resolve().parents[2] / ".tmp" / "tests"


@RobotConfig.register_subclass("mock_robot")
@dataclass(kw_only=True)
class MockRobotConfig(RobotConfig):
    id: str | None = "mock_robot"
    calibration_dir: Path | None = TEST_TMP_ROOT / "calibration" / "robots" / "mock_robot"
    cameras: dict = field(default_factory=dict)


class MockRobot(Robot):
    config_class = MockRobotConfig
    name = "mock_robot"

    def __init__(self, config: MockRobotConfig):
        super().__init__(config)
        self._is_connected = False
        self._last_action = {
            "joint_1.pos": 0.0,
            "joint_2.pos": 0.0,
            "gripper.pos": 0.0,
        }
        self.cameras = {}

    @property
    def observation_features(self) -> dict:
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "gripper.pos": float,
        }

    @property
    def action_features(self) -> dict:
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "gripper.pos": float,
        }

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        self._is_connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> RobotObservation:
        return dict(self._last_action)

    def send_action(self, action: RobotAction) -> RobotAction:
        self._last_action.update({key: float(value) for key, value in action.items() if key.endswith(".pos")})
        return dict(self._last_action)

    def disconnect(self) -> None:
        self._is_connected = False
