from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "lerobot" / "so101_to_so101_EE" / "move_to_pose.py"
SPEC = importlib.util.spec_from_file_location("move_to_pose", MODULE_PATH)
move_to_pose = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(move_to_pose)


def test_clip_target_pose_to_bounds_clamps_position_and_gripper():
    target_pose = {
        "name": "target",
        "ee.x": 99.0,
        "ee.y": -99.0,
        "ee.z": 99.0,
        "ee.wx": 0.1,
        "ee.wy": 0.2,
        "ee.wz": 0.3,
        "ee.gripper_pos": 150.0,
    }

    clipped = move_to_pose.clip_target_pose_to_bounds(target_pose)

    assert clipped["ee.x"] == move_to_pose.EE_BOUNDS_MAX[0]
    assert clipped["ee.y"] == move_to_pose.EE_BOUNDS_MIN[1]
    assert clipped["ee.z"] == move_to_pose.EE_BOUNDS_MAX[2]
    assert clipped["ee.gripper_pos"] == 100.0


def test_advance_towards_target_limits_position_step_and_gripper_step():
    current_pose = {
        "name": "current",
        "ee.x": 0.0,
        "ee.y": 0.0,
        "ee.z": 0.0,
        "ee.wx": 0.4,
        "ee.wy": 0.5,
        "ee.wz": 0.6,
        "ee.gripper_pos": 10.0,
    }
    target_pose = {
        "name": "target",
        "ee.x": 0.0,
        "ee.y": 0.03,
        "ee.z": 0.04,
        "ee.wx": 1.4,
        "ee.wy": 1.5,
        "ee.wz": 1.6,
        "ee.gripper_pos": 30.0,
    }

    step_pose = move_to_pose.advance_towards_target(current_pose, target_pose)

    assert abs(step_pose["ee.x"] - 0.0) < 1e-9
    assert abs(step_pose["ee.y"] - 0.009) < 1e-9
    assert abs(step_pose["ee.z"] - 0.012) < 1e-9
    assert abs(step_pose["ee.gripper_pos"] - 15.0) < 1e-9


def test_advance_towards_target_keeps_current_orientation_when_disabled():
    current_pose = {
        "name": "current",
        "ee.x": 0.1,
        "ee.y": 0.2,
        "ee.z": 0.3,
        "ee.wx": 0.4,
        "ee.wy": 0.5,
        "ee.wz": 0.6,
        "ee.gripper_pos": 7.0,
    }
    target_pose = {
        "name": "target",
        "ee.x": 0.2,
        "ee.y": 0.3,
        "ee.z": 0.4,
        "ee.wx": 1.4,
        "ee.wy": 1.5,
        "ee.wz": 1.6,
        "ee.gripper_pos": 9.0,
    }

    step_pose = move_to_pose.advance_towards_target(current_pose, target_pose)

    assert step_pose["ee.wx"] == current_pose["ee.wx"]
    assert step_pose["ee.wy"] == current_pose["ee.wy"]
    assert step_pose["ee.wz"] == current_pose["ee.wz"]
