import json
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
    assert abs(step_pose["ee.y"] - 0.024) < 1e-9
    assert abs(step_pose["ee.z"] - 0.032) < 1e-9
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


def test_pose_error_score_includes_orientation_and_gripper_when_enabled():
    original_flag = move_to_pose.USE_TARGET_ORIENTATION
    try:
        move_to_pose.USE_TARGET_ORIENTATION = True
        target_pose = {
            "ee.x": 0.1,
            "ee.y": 0.2,
            "ee.z": 0.3,
            "ee.wx": 0.0,
            "ee.wy": 0.0,
            "ee.wz": 0.0,
            "ee.gripper_pos": 10.0,
        }
        reached_pose = {
            "ee.x": 0.1,
            "ee.y": 0.2,
            "ee.z": 0.31,
            "ee.wx": 0.3,
            "ee.wy": 0.4,
            "ee.wz": 0.0,
            "ee.gripper_pos": 20.0,
        }

        score = move_to_pose.pose_error_score(target_pose, reached_pose)

        assert score > 0.01
        assert score > move_to_pose.position_error_norm(target_pose, reached_pose)
    finally:
        move_to_pose.USE_TARGET_ORIENTATION = original_flag


def test_predicted_motion_is_progress_requires_error_improvement():
    current_pose = {"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.0}
    target_pose = {"ee.x": 0.1, "ee.y": 0.0, "ee.z": 0.0}
    better_pose = {"ee.x": 0.02, "ee.y": 0.0, "ee.z": 0.0}
    same_pose = {"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.0}

    assert move_to_pose.predicted_motion_is_progress(
        current_pose=current_pose,
        predicted_pose=better_pose,
        target_pose=target_pose,
    ) is True
    assert move_to_pose.predicted_motion_is_progress(
        current_pose=current_pose,
        predicted_pose=same_pose,
        target_pose=target_pose,
    ) is False


def test_predicted_motion_is_progress_rejects_worse_pose_score_when_orientation_enabled():
    original_flag = move_to_pose.USE_TARGET_ORIENTATION
    try:
        move_to_pose.USE_TARGET_ORIENTATION = True
        current_pose = {
            "ee.x": 0.0,
            "ee.y": 0.0,
            "ee.z": 0.0,
            "ee.wx": 0.0,
            "ee.wy": 0.0,
            "ee.wz": 0.0,
            "ee.gripper_pos": 0.0,
        }
        target_pose = {
            "ee.x": 0.1,
            "ee.y": 0.0,
            "ee.z": 0.0,
            "ee.wx": 0.0,
            "ee.wy": 0.0,
            "ee.wz": 0.0,
            "ee.gripper_pos": 0.0,
        }
        predicted_pose = {
            "ee.x": 0.02,
            "ee.y": 0.0,
            "ee.z": 0.0,
            "ee.wx": 1.0,
            "ee.wy": 0.0,
            "ee.wz": 0.0,
            "ee.gripper_pos": 0.0,
        }

        assert move_to_pose.predicted_motion_is_progress(
            current_pose=current_pose,
            predicted_pose=predicted_pose,
            target_pose=target_pose,
        ) is False
    finally:
        move_to_pose.USE_TARGET_ORIENTATION = original_flag


def test_predicted_ee_pose_from_joint_action_includes_orientation_and_gripper():
    class FakeKinematics:
        def forward_kinematics(self, q):
            return [
                [0.0, -1.0, 0.0, 0.11],
                [1.0, 0.0, 0.0, 0.22],
                [0.0, 0.0, 1.0, 0.33],
                [0.0, 0.0, 0.0, 1.0],
            ]

    pose = move_to_pose.predicted_ee_pose_from_joint_action(
        {
            "joint_a.pos": 1.0,
            "joint_b.pos": 2.0,
            "gripper.pos": 30.0,
            "ee.wx": 0.4,
            "ee.wy": 0.5,
            "ee.wz": 0.6,
        },
        FakeKinematics(),
        ["joint_a", "joint_b"],
    )

    assert pose == {
        "ee.x": 0.11,
        "ee.y": 0.22,
        "ee.z": 0.33,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 1.5707963267948966,
        "ee.gripper_pos": 30.0,
    }


def test_format_ik_candidate_summary_and_print_diagnostics(capsys):
    candidate = {
        "position_error_m": 0.012,
        "orientation_error_rad": 0.25,
        "max_joint_delta_deg": 18.0,
        "max_over_limit_deg": 0.0,
        "min_joint_limit_margin_deg": 7.5,
        "max_joint_limit_violation_deg": 0.0,
        "is_unsafe": False,
        "pose_invalid": False,
    }

    summary = move_to_pose.format_ik_candidate_summary(candidate)
    assert "pos=1.20cm" in summary
    assert "ori=0.250rad" in summary

    move_to_pose.print_ik_diagnostics({"IK_diagnostics": [candidate, candidate]}, step=3)
    captured = capsys.readouterr().out
    assert "IK step 003 best:" in captured
    assert "IK step 003 alt :" in captured


def test_build_ik_diagnostics_record_and_append_jsonl():
    original_flag = move_to_pose.USE_TARGET_ORIENTATION
    try:
        move_to_pose.USE_TARGET_ORIENTATION = True
        current_pose = {
            "ee.x": 0.1,
            "ee.y": 0.2,
            "ee.z": 0.3,
            "ee.wx": 0.1,
            "ee.wy": 0.2,
            "ee.wz": 0.3,
            "ee.gripper_pos": 15.0,
        }
        target_pose = {
            "ee.x": 0.2,
            "ee.y": 0.2,
            "ee.z": 0.3,
            "ee.wx": 0.0,
            "ee.wy": 0.0,
            "ee.wz": 0.0,
            "ee.gripper_pos": 10.0,
        }
        predicted_pose = {
            "ee.x": 0.15,
            "ee.y": 0.2,
            "ee.z": 0.3,
            "ee.wx": 0.05,
            "ee.wy": 0.0,
            "ee.wz": 0.0,
            "ee.gripper_pos": 12.0,
        }
        complementary_data = {
            "IK_solution": [1.0, 2.0, 3.0],
            "IK_diagnostics": [{"position_error_m": 0.1, "orientation_error_rad": 0.2}],
        }

        record = move_to_pose.build_ik_diagnostics_record(
            step=7,
            current_pose=current_pose,
            target_pose=target_pose,
            predicted_pose=predicted_pose,
            complementary_data=complementary_data,
        )

        assert record["step"] == 7
        assert record["ik_solution"] == [1.0, 2.0, 3.0]
        assert record["ik_diagnostics"][0]["position_error_m"] == 0.1

        log_path = Path(__file__).resolve().parents[1] / ".tmp" / "test_logs" / "move_to_pose_ik.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.unlink()
        move_to_pose.append_ik_diagnostics_jsonl(log_path, record)
        written = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert written[0]["step"] == 7
        assert written[0]["predicted_pose"]["ee.x"] == 0.15
        log_path.unlink()
    finally:
        move_to_pose.USE_TARGET_ORIENTATION = original_flag


def test_ik_best_candidate_is_progress_prefers_internal_ik_diagnostics():
    current_pose = {
        "ee.x": 0.0,
        "ee.y": 0.0,
        "ee.z": 0.0,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": 0.0,
    }
    target_pose = {
        "ee.x": 0.1,
        "ee.y": 0.0,
        "ee.z": 0.0,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": 0.0,
    }
    complementary_data = {
        "IK_diagnostics": [
            {
                "position_error_m": 0.02,
                "orientation_error_rad": 0.4,
                "is_unsafe": False,
                "pose_invalid": False,
            }
        ]
    }

    assert move_to_pose.ik_best_candidate_is_progress(
        complementary_data=complementary_data,
        current_pose=current_pose,
        target_pose=target_pose,
    ) is True


def test_compute_joint_tracking_error_respects_continuous_joint_wraparound():
    sent_joint_action = {
        "shoulder_pan.pos": 179.0,
        "wrist_roll.pos": 10.0,
    }
    observation = {
        "shoulder_pan.pos": -179.0,
        "wrist_roll.pos": 25.0,
    }

    tracking_error = move_to_pose.compute_joint_tracking_error(
        sent_joint_action=sent_joint_action,
        observation=observation,
        motor_names=["shoulder_pan", "wrist_roll"],
        continuous_joint_names=("shoulder_pan",),
    )

    assert tracking_error["shoulder_pan"] == 2.0
    assert tracking_error["wrist_roll"] == 15.0


def test_max_joint_tracking_error_handles_empty_and_non_empty_inputs():
    assert move_to_pose.max_joint_tracking_error({}) == 0.0
    assert move_to_pose.max_joint_tracking_error({"a": 1.5, "b": 3.0}) == 3.0


def test_best_ik_diagnostic_handles_missing_and_present_diagnostics():
    assert move_to_pose.best_ik_diagnostic({}) is None
    diag = {"position_error_m": 0.1}
    assert move_to_pose.best_ik_diagnostic({"IK_diagnostics": [diag]}) == diag


def test_ik_candidate_is_clampable_only_for_delta_limit_overflow():
    assert (
        move_to_pose.ik_candidate_is_clampable(
            {
                "is_unsafe": True,
                "pose_invalid": False,
                "max_joint_limit_violation_deg": 0.0,
                "max_over_limit_deg": 2.0,
            }
        )
        is True
    )
    assert (
        move_to_pose.ik_candidate_is_clampable(
            {
                "is_unsafe": True,
                "pose_invalid": True,
                "max_joint_limit_violation_deg": 0.0,
                "max_over_limit_deg": 2.0,
            }
        )
        is False
    )
    assert (
        move_to_pose.ik_candidate_is_clampable(
            {
                "is_unsafe": True,
                "pose_invalid": False,
                "max_joint_limit_violation_deg": 1.0,
                "max_over_limit_deg": 2.0,
            }
        )
        is False
    )


def test_ik_candidate_can_use_direct_clamp_rejects_extreme_over_limit():
    assert (
        move_to_pose.ik_candidate_can_use_direct_clamp(
            {
                "is_unsafe": True,
                "pose_invalid": False,
                "max_joint_limit_violation_deg": 0.0,
                "max_over_limit_deg": 5.0,
            }
        )
        is True
    )
    assert (
        move_to_pose.ik_candidate_can_use_direct_clamp(
            {
                "is_unsafe": True,
                "pose_invalid": False,
                "max_joint_limit_violation_deg": 0.0,
                "max_over_limit_deg": 50.0,
            }
        )
        is False
    )


def test_clamp_joint_action_to_delta_limits_caps_each_joint_delta():
    joint_action = {
        "shoulder_pan.pos": 10.0,
        "shoulder_lift.pos": -10.0,
        "gripper.pos": 33.0,
    }
    observation = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "gripper.pos": 30.0,
    }

    clamped = move_to_pose.clamp_joint_action_to_delta_limits(
        joint_action=joint_action,
        observation=observation,
        motor_names=["shoulder_pan", "shoulder_lift", "gripper"],
        max_joint_delta_deg={"shoulder_pan": 4.0, "shoulder_lift": 3.0, "gripper": 99.0},
    )

    assert clamped["shoulder_pan.pos"] == 3.0
    assert clamped["shoulder_lift.pos"] == -3.0
    assert clamped["gripper.pos"] == 33.0


def test_clamp_joint_action_to_delta_limits_uses_uniform_scale():
    joint_action = {
        "shoulder_pan.pos": 10.0,
        "shoulder_lift.pos": -10.0,
        "gripper.pos": 33.0,
    }
    observation = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "gripper.pos": 30.0,
    }

    clamped = move_to_pose.clamp_joint_action_to_delta_limits(
        joint_action=joint_action,
        observation=observation,
        motor_names=["shoulder_pan", "shoulder_lift", "gripper"],
        max_joint_delta_deg={"shoulder_pan": 4.0, "shoulder_lift": 8.0, "gripper": 99.0},
    )

    assert clamped["shoulder_pan.pos"] == 4.0
    assert clamped["shoulder_lift.pos"] == -4.0
    assert clamped["gripper.pos"] == 33.0


def test_joint_action_from_joint_positions_preserves_gripper_observation():
    action = move_to_pose.joint_action_from_joint_positions(
        joint_positions_deg=[4.0, -3.0, 99.0],
        observation={"gripper.pos": 30.0},
        motor_names=["shoulder_pan", "shoulder_lift", "gripper"],
    )

    assert action["shoulder_pan.pos"] == 4.0
    assert action["shoulder_lift.pos"] == -3.0
    assert action["gripper.pos"] == 30.0


def test_find_progressive_clamped_joint_action_finds_smaller_progressive_step():
    class FakeKinematics:
        def forward_kinematics(self, q):
            x = float(q[0]) / 100.0
            return [
                [1.0, 0.0, 0.0, x],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]

    result = move_to_pose.find_progressive_clamped_joint_action(
        joint_action={"joint_a.pos": 10.0, "gripper.pos": 30.0},
        raw_solution=None,
        observation={"joint_a.pos": 0.0, "gripper.pos": 30.0},
        motor_names=["joint_a"],
        continuous_joint_names=(),
        max_joint_delta_deg={"joint_a": 4.0},
        current_pose={"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.0, "ee.gripper_pos": 30.0},
        target_pose={"ee.x": 0.015, "ee.y": 0.0, "ee.z": 0.0, "ee.gripper_pos": 30.0},
        kinematics=FakeKinematics(),
    )

    assert result is not None
    joint_action, predicted_pose, scale = result
    assert joint_action["joint_a.pos"] == 2.0
    assert predicted_pose["ee.x"] == 0.02
    assert scale == 0.5


def test_position_only_robot_kinematics_zeroes_orientation_weight():
    class FakeBaseKinematics:
        def forward_kinematics(self, joint_pos_deg):
            return joint_pos_deg

        def inverse_kinematics(self, current_joint_pos, desired_ee_pose, position_weight=1.0, orientation_weight=0.01):
            return {
                "current_joint_pos": current_joint_pos,
                "desired_ee_pose": desired_ee_pose,
                "position_weight": position_weight,
                "orientation_weight": orientation_weight,
            }

    wrapper = move_to_pose.PositionOnlyRobotKinematics(FakeBaseKinematics())

    result = wrapper.inverse_kinematics([1, 2, 3], "pose")
    assert result["position_weight"] == 1.0
    assert result["orientation_weight"] == 0.0
    assert wrapper.forward_kinematics([1, 2, 3]) == [1, 2, 3]


def test_joint_hold_action_from_observation_copies_joint_positions():
    observation = {
        "joint_a.pos": 1.2,
        "joint_b.pos": -3.4,
    }

    hold_action = move_to_pose.joint_hold_action_from_observation(observation, ["joint_a", "joint_b"])

    assert hold_action == {
        "joint_a.pos": 1.2,
        "joint_b.pos": -3.4,
    }
