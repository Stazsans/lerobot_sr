import argparse
import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "lerobot" / "so101_to_so101_EE" / "argo_real_ik.py"
SPEC = importlib.util.spec_from_file_location("argo_real_ik", MODULE_PATH)
argo_real_ik = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(argo_real_ik)


def make_args(**overrides):
    defaults = {
        "use_orientation": False,
        "orientation_only": False,
        "target_rotvec": None,
        "target_delta_rotvec": None,
        "target_x_cm": None,
        "target_y_cm": None,
        "target_z_cm": None,
        "target_delta_cm": [1.0, 2.0, 3.0],
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_target_pose_preserves_position_defaults_and_applies_delta_translation():
    start_pose = np.eye(4, dtype=float)
    target_pose = argo_real_ik.build_target_pose(start_pose, make_args())

    np.testing.assert_allclose(target_pose[:3, 3], np.array([0.01, 0.02, 0.03]))
    np.testing.assert_allclose(target_pose[:3, :3], np.eye(3))


def test_build_target_pose_supports_absolute_target_rotvec():
    start_pose = np.eye(4, dtype=float)
    target_rotvec = [0.0, 0.0, np.pi / 2.0]

    target_pose = argo_real_ik.build_target_pose(start_pose, make_args(target_rotvec=target_rotvec))

    reached_rotvec = argo_real_ik.Rotation.from_matrix(target_pose[:3, :3]).as_rotvec()
    np.testing.assert_allclose(reached_rotvec, np.array(target_rotvec), atol=1e-6)


def test_build_target_pose_supports_delta_target_rotvec_from_current_orientation():
    start_pose = np.eye(4, dtype=float)
    start_pose[:3, :3] = argo_real_ik.Rotation.from_rotvec([0.2, -0.1, 0.05]).as_matrix()
    delta_rotvec = [0.0, 0.3, 0.0]

    target_pose = argo_real_ik.build_target_pose(start_pose, make_args(target_delta_rotvec=delta_rotvec))

    expected = start_pose[:3, :3] @ argo_real_ik.Rotation.from_rotvec(delta_rotvec).as_matrix()
    np.testing.assert_allclose(target_pose[:3, :3], expected, atol=1e-6)


def test_target_requests_orientation_when_pose_target_includes_rotvec():
    assert argo_real_ik.target_requests_orientation(make_args(target_rotvec=[0.0, 0.0, 0.1])) is True
    assert argo_real_ik.target_requests_orientation(make_args(target_delta_rotvec=[0.0, 0.0, 0.0])) is True
    assert argo_real_ik.target_requests_orientation(make_args(orientation_only=True)) is True
    assert argo_real_ik.target_requests_orientation(make_args()) is False


def test_target_reached_requires_orientation_when_enabled():
    current_pose = np.eye(4, dtype=float)
    target_pose = np.eye(4, dtype=float)
    target_pose[:3, :3] = argo_real_ik.Rotation.from_rotvec([0.0, 0.0, 0.2]).as_matrix()

    assert argo_real_ik.target_reached(current_pose, target_pose, use_orientation=False) is True
    assert argo_real_ik.target_reached(current_pose, target_pose, use_orientation=True) is False


def test_target_reached_can_ignore_position_when_orientation_only():
    current_pose = np.eye(4, dtype=float)
    current_pose[:3, 3] = np.array([0.2, 0.0, 0.0])
    target_pose = np.eye(4, dtype=float)
    target_pose[:3, :3] = argo_real_ik.Rotation.from_rotvec([0.0, 0.0, 0.02]).as_matrix()

    assert argo_real_ik.target_reached(
        current_pose,
        target_pose,
        use_orientation=True,
        ignore_position=True,
    ) is True


def test_parse_args_accepts_rotvec_targets():
    args = argo_real_ik.parse_args(
        ["--target-rotvec", "0.1", "0.2", "0.3", "--target-delta-rotvec", "0.0", "0.0", "0.1"]
    )

    assert args.target_rotvec == [0.1, 0.2, 0.3]
    assert args.target_delta_rotvec == [0.0, 0.0, 0.1]


def test_parse_args_uses_updated_servo_defaults():
    args = argo_real_ik.parse_args([])

    assert args.max_servo_cycles == 150
    assert args.replan_interval == 30
    assert args.ik_gain == 0.5
    assert args.apply_final_trim is False


def test_parse_args_can_enable_final_trim_explicitly():
    args = argo_real_ik.parse_args(["--apply-final-trim"])

    assert args.apply_final_trim is True


def test_parse_args_can_enable_orientation_only():
    args = argo_real_ik.parse_args(["--orientation-only"])

    assert args.orientation_only is True


def test_choose_max_joint_step_deg_accepts_explicit_tracking_error_kwarg():
    result = argo_real_ik.choose_max_joint_step_deg(
        4.0,
        8.0,
        use_orientation=False,
        orientation_error_rad=0.0,
        tracking_error_deg=0.0,
    )

    assert result == 4.0


def test_choose_max_joint_step_deg_keeps_very_near_limit_for_position_only():
    result = argo_real_ik.choose_max_joint_step_deg(
        0.0,
        8.0,
        use_orientation=False,
        orientation_error_rad=0.0,
        tracking_error_deg=0.0,
    )

    assert result == 2.0


def test_choose_max_joint_step_deg_does_not_use_very_near_limit_when_orientation_error_is_large():
    result = argo_real_ik.choose_max_joint_step_deg(
        0.0,
        8.0,
        use_orientation=True,
        orientation_error_rad=0.3,
        tracking_error_deg=0.0,
    )

    assert result == 8.0


def test_candidate_pose_error_score_cm_penalizes_orientation_when_enabled():
    worse_orientation = argo_real_ik.candidate_pose_error_score_cm(
        pos_err_m=0.0112,
        ori_err_rad=0.1830,
        use_orientation=True,
    )
    better_orientation = argo_real_ik.candidate_pose_error_score_cm(
        pos_err_m=0.0177,
        ori_err_rad=0.0034,
        use_orientation=True,
    )

    assert better_orientation < worse_orientation


def test_candidate_pose_error_score_cm_can_ignore_position():
    score = argo_real_ik.candidate_pose_error_score_cm(
        pos_err_m=0.25,
        ori_err_rad=0.01,
        use_orientation=True,
        ignore_position=True,
    )

    assert abs(score - 0.1) < 1e-9


def test_orientation_candidate_sort_prefers_better_pose_score_over_better_position_only():
    candidates = [
        {
            "pos_err": 0.0112,
            "ori_err": 0.1830,
            "joint_delta_deg": 13.3,
            "limit_violation": 0.0,
        },
        {
            "pos_err": 0.0177,
            "ori_err": 0.0034,
            "joint_delta_deg": 27.8,
            "limit_violation": 0.0,
        },
    ]

    candidates.sort(
        key=lambda item: (
            argo_real_ik.candidate_pose_error_score_cm(
                pos_err_m=float(item["pos_err"]),
                ori_err_rad=float(item["ori_err"]),
                use_orientation=True,
            ),
            float(item["ori_err"]),
            float(item["joint_delta_deg"]),
            float(item["limit_violation"]),
        )
    )

    assert candidates[0]["ori_err"] == 0.0034
