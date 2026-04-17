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

    assert clipped["ee.x"] == move_to_pose.EE_BOUNDS_MAX_CM[0] / move_to_pose.CM_PER_M
    assert clipped["ee.y"] == move_to_pose.EE_BOUNDS_MIN_CM[1] / move_to_pose.CM_PER_M
    assert clipped["ee.z"] == move_to_pose.EE_BOUNDS_MAX_CM[2] / move_to_pose.CM_PER_M
    assert clipped["ee.gripper_pos"] == 100.0


def test_resolve_target_pose_uses_cm_for_position_targets_and_deltas():
    current_pose = {
        "name": "current",
        "ee.x": 0.10,
        "ee.y": 0.20,
        "ee.z": 0.30,
        "ee.wx": 0.4,
        "ee.wy": 0.5,
        "ee.wz": 0.6,
        "ee.gripper_pos": 7.0,
    }
    target_pose = {
        "name": "target",
        "delta.ee.x": 5.0,
        "ee.y": 25.0,
        "delta.ee.z": -2.0,
        "ee.wx": 1.4,
        "ee.wy": 1.5,
        "ee.wz": 1.6,
        "ee.gripper_pos": 9.0,
    }

    resolved = move_to_pose.resolve_target_pose(current_pose, target_pose)

    assert abs(resolved["ee.x"] - 0.15) < 1e-9
    assert abs(resolved["ee.y"] - 0.25) < 1e-9
    assert abs(resolved["ee.z"] - 0.28) < 1e-9
    assert resolved["ee.wx"] == current_pose["ee.wx"]
    assert resolved["ee.wy"] == current_pose["ee.wy"]
    assert resolved["ee.wz"] == current_pose["ee.wz"]


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


def test_target_reached_requires_orientation_when_enabled():
    original_flag = move_to_pose.USE_TARGET_ORIENTATION
    try:
        move_to_pose.USE_TARGET_ORIENTATION = True
        target_pose = {
            "ee.x": 0.1,
            "ee.y": 0.2,
            "ee.z": 0.3,
            "ee.wx": 0.0,
            "ee.wy": 0.0,
            "ee.wz": 1.0,
            "ee.gripper_pos": 20.0,
        }
        reached_pose = {
            "ee.x": 0.1,
            "ee.y": 0.2,
            "ee.z": 0.3,
            "ee.wx": 0.0,
            "ee.wy": 0.0,
            "ee.wz": 0.0,
            "ee.gripper_pos": 20.0,
        }

        assert move_to_pose.target_reached(target_pose, reached_pose) is False
    finally:
        move_to_pose.USE_TARGET_ORIENTATION = original_flag


def test_target_reached_requires_gripper_convergence():
    target_pose = {
        "ee.x": 0.1,
        "ee.y": 0.2,
        "ee.z": 0.3,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": 20.0,
    }
    reached_pose = {
        "ee.x": 0.1,
        "ee.y": 0.2,
        "ee.z": 0.3,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": 25.0,
    }

    assert move_to_pose.target_reached(target_pose, reached_pose) is False
