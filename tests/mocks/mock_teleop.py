#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

TEST_TMP_ROOT = Path(__file__).resolve().parents[2] / ".tmp" / "tests"


@TeleoperatorConfig.register_subclass("mock_teleop")
@dataclass(kw_only=True)
class MockTeleopConfig(TeleoperatorConfig):
    id: str | None = "mock_teleop"
    calibration_dir: Path | None = TEST_TMP_ROOT / "calibration" / "teleoperators" / "mock_teleop"


class MockTeleop(Teleoperator):
    config_class = MockTeleopConfig
    name = "mock_teleop"

    def __init__(self, config: MockTeleopConfig):
        super().__init__(config)
        self._is_connected = False
        self._action = {
            "joint_1.pos": 1.0,
            "joint_2.pos": -1.0,
            "gripper.pos": 50.0,
        }

    @property
    def action_features(self) -> dict:
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

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

    def get_action(self) -> RobotAction:
        return dict(self._action)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    def disconnect(self) -> None:
        self._is_connected = False
