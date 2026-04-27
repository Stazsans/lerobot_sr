import importlib.util
import sys
from pathlib import Path

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lerobot"
    / "so101_to_so101_EE"
    / "argo_record.py"
)
SPEC = importlib.util.spec_from_file_location("argo_record", MODULE_PATH)
argo_record = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = argo_record
SPEC.loader.exec_module(argo_record)


def test_pose_to_ee_dict_round_trips_pose_orientation():
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.1, -0.2, 0.3]
    pose[:3, :3] = argo_record.Rotation.from_rotvec([0.1, 0.2, -0.3]).as_matrix()

    ee_action = argo_record.pose_to_ee_dict(pose, gripper_pos=42.0)
    round_trip_pose = argo_record.ee_dict_to_pose(ee_action)

    np.testing.assert_allclose(round_trip_pose, pose, atol=1e-9)
    assert ee_action["ee.gripper_pos"] == 42.0


def test_fk_transform_features_replaces_joint_action_with_ee_action():
    step = argo_record.ArgoFKJointsToEE(
        kin=None,
        robot_model=None,
        argo_joint_names=["shoulder_pan", "gripper"],
        target_link_name="gripper_frame_link",
    )
    features = {
        PipelineFeatureType.ACTION: {
            "shoulder_pan.pos": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
            "gripper.pos": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
        },
        PipelineFeatureType.OBSERVATION: {},
    }

    transformed = step.transform_features(features)

    assert "shoulder_pan.pos" not in transformed[PipelineFeatureType.ACTION]
    assert "gripper.pos" not in transformed[PipelineFeatureType.ACTION]
    assert set(transformed[PipelineFeatureType.ACTION]) == {
        "ee.x",
        "ee.y",
        "ee.z",
        "ee.wx",
        "ee.wy",
        "ee.wz",
        "ee.gripper_pos",
    }


def test_ik_processor_reset_clears_previous_goal():
    step = argo_record.ArgoIKEEToJoints(
        kin=None,
        robot_model=None,
        robot_utils=None,
        argo_joint_names=["shoulder_pan", "gripper"],
        target_link_name="gripper_frame_link",
    )
    step.previous_q_goal = np.array([1.0, 2.0])

    step.reset()

    assert step.previous_q_goal is None
