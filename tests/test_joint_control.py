"""Unit tests for the pure joint-command reconciliation (joint_control)."""

import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")

_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                     "utils", "joint_control.py")


def _load():
    spec = importlib.util.spec_from_file_location("joint_control", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def jc():
    return _load()


ORDER = ["arm_base", "arm_elbow", "gripper"]


def test_reorders_full_command(jc):
    # command lists joints out of order -> mapped onto the articulation order.
    targets, ignored = jc.map_named_command(
        ["gripper", "arm_base", "arm_elbow"], [0.5, 1.0, 2.0], ORDER)
    assert np.allclose(targets, [1.0, 2.0, 0.5])
    assert ignored == []


def test_partial_command_holds_current(jc):
    # only arm_elbow named -> others keep their current value.
    targets, ignored = jc.map_named_command(
        ["arm_elbow"], [9.0], ORDER, current=[0.1, 0.2, 0.3])
    assert np.allclose(targets, [0.1, 9.0, 0.3])
    assert ignored == []


def test_partial_command_without_current_is_nan(jc):
    targets, ignored = jc.map_named_command(["gripper"], [0.7], ORDER)
    assert np.isnan(targets[0]) and np.isnan(targets[1])
    assert targets[2] == 0.7


def test_unknown_joints_ignored(jc):
    targets, ignored = jc.map_named_command(
        ["arm_base", "thruster_1", "nonexistent"], [1.0, 5.0, 6.0], ORDER,
        current=[0.0, 0.0, 0.0])
    assert targets[0] == 1.0 and targets[1] == 0.0 and targets[2] == 0.0
    assert ignored == ["thruster_1", "nonexistent"]


def test_current_wrong_length_raises(jc):
    with pytest.raises(ValueError):
        jc.map_named_command(["arm_base"], [1.0], ORDER, current=[0.0, 0.0])


def test_clamp_to_limits(jc):
    t = jc.clamp_to_limits([5.0, -5.0, 0.5], lower=[-1.0, -1.0, -1.0],
                           upper=[1.0, 1.0, 1.0])
    assert np.allclose(t, [1.0, -1.0, 0.5])


def test_clamp_leaves_nan_and_infinite_bounds(jc):
    # NaN ("no change") stays NaN; +/-inf bounds disable clamping on that side.
    t = jc.clamp_to_limits([np.nan, 100.0], lower=[-1.0, -np.inf],
                           upper=[1.0, np.inf])
    assert np.isnan(t[0])
    assert t[1] == 100.0          # +inf upper -> unclamped


def test_clamp_one_sided_infinite_lower(jc):
    t = jc.clamp_to_limits([-100.0, 100.0], lower=[-np.inf, -np.inf],
                           upper=[5.0, 5.0])
    assert t[0] == -100.0 and t[1] == 5.0
