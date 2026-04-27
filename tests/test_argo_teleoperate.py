import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lerobot"
    / "so101_to_so101_EE"
    / "argo_teleoperate.py"
)
SPEC = importlib.util.spec_from_file_location("argo_teleoperate", MODULE_PATH)
argo_teleoperate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(argo_teleoperate)


def test_parse_args_uses_direct_teleop_defaults():
    args = argo_teleoperate.parse_args([])

    assert not hasattr(args, "execute")
    assert not hasattr(args, "use_orientation")
    assert args.ik_iterations == 8
    assert args.fps == 30


def test_parse_args_rejects_orientation_flag_because_it_is_always_enabled():
    try:
        argo_teleoperate.parse_args(["--use-orientation"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected --use-orientation to be rejected.")


def test_fast_seed_candidates_prefers_previous_goal():
    measured_q = np.array([0.0, 1.0, 2.0])
    previous_q = np.array([3.0, 4.0, 5.0])

    seeds = argo_teleoperate.fast_seed_candidates(
        measured_q_rad=measured_q,
        previous_q_goal_rad=previous_q,
    )

    assert [label for label, _ in seeds] == ["previous_goal", "measured"]
    np.testing.assert_allclose(seeds[0][1], previous_q)
    np.testing.assert_allclose(seeds[1][1], measured_q)


def test_fast_seed_candidates_deduplicates_matching_previous_goal():
    measured_q = np.array([0.0, 1.0, 2.0])

    seeds = argo_teleoperate.fast_seed_candidates(
        measured_q_rad=measured_q,
        previous_q_goal_rad=measured_q.copy(),
    )

    assert [label for label, _ in seeds] == ["previous_goal"]


def test_validate_argo_joint_names_rejects_unknown_joint():
    try:
        argo_teleoperate.validate_argo_joint_names(["shoulder_pan", "unknown_joint"])
    except ValueError as exc:
        assert "unknown_joint" in str(exc)
    else:
        raise AssertionError("Expected unsupported Argo joint to raise ValueError.")
