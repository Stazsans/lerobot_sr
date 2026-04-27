import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("grpc")

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lerobot"
    / "so101_to_so101_EE"
    / "argo_evaluate.py"
)
SPEC = importlib.util.spec_from_file_location("argo_evaluate", MODULE_PATH)
argo_evaluate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(argo_evaluate)


class FakeTimedAction:
    def __init__(self, action):
        self._action = torch.tensor(action)

    def get_action(self):
        return self._action


def test_timed_action_to_ee_action_maps_tensor_to_ee_keys():
    evaluator = argo_evaluate.ArgoEEAsyncEvaluator.__new__(argo_evaluate.ArgoEEAsyncEvaluator)

    result = evaluator._timed_action_to_ee_action(FakeTimedAction([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 40.0]))

    assert result == {
        "ee.x": 1.0,
        "ee.y": 2.0,
        "ee.z": 3.0,
        "ee.wx": 0.10000000149011612,
        "ee.wy": 0.20000000298023224,
        "ee.wz": 0.30000001192092896,
        "ee.gripper_pos": 40.0,
    }
