import importlib.util
from pathlib import Path

from lerobot.utils.constants import ACTION

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lerobot"
    / "so101_to_so101_EE"
    / "argo_replay.py"
)
SPEC = importlib.util.spec_from_file_location("argo_replay", MODULE_PATH)
argo_replay = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(argo_replay)


class FakeDataset:
    features = {ACTION: {"names": ["ee.x", "ee.y", "ee.z"]}}


def test_dataset_action_at_maps_dataset_action_vector_to_names():
    actions = [{ACTION: [0.1, -0.2, 0.3]}]

    result = argo_replay.dataset_action_at(FakeDataset(), actions, 0)

    assert result == {"ee.x": 0.1, "ee.y": -0.2, "ee.z": 0.3}
