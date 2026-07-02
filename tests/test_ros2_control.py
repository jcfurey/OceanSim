"""Unit tests for isaacsim/oceansim/utils/ros2_control_math.py (pure numpy, no
ROS/Isaac Sim -- ros2_control.py itself imports rclpy/pxr/isaacsim.core at
module level, so it can't be exec'd outside that environment; the watchdog and
clamping logic it added are covered here via the pure math it delegates to,
following the same test-seam split as ros2_math.py / ros2_sensors.py."""

import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")

_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                     "utils", "ros2_control_math.py")


def _load():
    spec = importlib.util.spec_from_file_location("ros2_control_math", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load()


# --------------------------------------------------------------- clamp_magnitude

def test_clamp_none_is_noop(m):
    # Default (unset) limit -- matches ROS2ControlReceiver's default behavior
    # exactly, a regression guard for every existing caller.
    out = m.clamp_magnitude([1.0, -2.0, 3.0], None)
    assert out == [1.0, -2.0, 3.0]


def test_clamp_under_limit_is_unchanged(m):
    out = m.clamp_magnitude([0.1, 0.0, 0.0], max_mag=5.0)
    assert np.allclose(out, [0.1, 0.0, 0.0])


def test_clamp_over_limit_preserves_direction(m):
    # magnitude 5 along +x, clamp to 2 -> direction unchanged, magnitude == 2.
    out = m.clamp_magnitude([5.0, 0.0, 0.0], max_mag=2.0)
    assert np.allclose(out, [2.0, 0.0, 0.0])


def test_clamp_over_limit_preserves_direction_offaxis(m):
    vec = [3.0, 4.0, 0.0]  # magnitude 5
    out = m.clamp_magnitude(vec, max_mag=1.0)
    assert np.isclose(np.linalg.norm(out), 1.0)
    # direction preserved: out is vec scaled by a positive factor.
    assert np.allclose(np.array(out) / np.linalg.norm(out),
                       np.array(vec) / np.linalg.norm(vec))


def test_clamp_zero_vector_stays_zero(m):
    out = m.clamp_magnitude([0.0, 0.0, 0.0], max_mag=2.0)
    assert np.allclose(out, [0.0, 0.0, 0.0])


# --------------------------------------------------------------- is_command_stale

def test_not_stale_within_timeout(m):
    assert m.is_command_stale(now=10.0, last_command_time=9.0, timeout=2.0) is False


def test_stale_after_timeout_elapsed(m):
    assert m.is_command_stale(now=12.1, last_command_time=10.0, timeout=2.0) is True


def test_exactly_at_timeout_is_not_yet_stale(m):
    # Strictly greater-than, matching ros2_control.py's dead-man's-switch check
    # (>, not >=) -- a command that just barely lands at the timeout boundary
    # is not treated as a dropped link.
    assert m.is_command_stale(now=12.0, last_command_time=10.0, timeout=2.0) is False


# ------------------------------------------------- non-finite command rejection

def test_clamp_rejects_infinite_command(m):
    # inf command used to become NaN via arr * (max_mag / inf): 0*inf = NaN.
    out = m.clamp_magnitude([float("inf"), 1.0, 0.0], max_mag=5.0)
    assert out == [0.0, 0.0, 0.0]


def test_clamp_rejects_nan_command_even_unclamped(m):
    # NaN norm made norm > max_mag False, so NaN passed through "unclamped";
    # and the None (unbounded) path must reject it too -- garbage must never
    # reach the thrusters.
    assert m.clamp_magnitude([float("nan"), 0.0, 0.0], max_mag=5.0) == [0.0, 0.0, 0.0]
    assert m.clamp_magnitude([0.0, float("nan"), 0.0], None) == [0.0, 0.0, 0.0]
    assert m.clamp_magnitude([0.0, 0.0, float("-inf")], None) == [0.0, 0.0, 0.0]


def test_clamp_finite_paths_unchanged(m):
    # regression: finite behavior identical after the non-finite guard.
    assert m.clamp_magnitude([3.0, 4.0, 0.0], max_mag=5.0) == pytest.approx([3.0, 4.0, 0.0])
    out = m.clamp_magnitude([6.0, 8.0, 0.0], max_mag=5.0)
    assert out == pytest.approx([3.0, 4.0, 0.0])   # scaled to mag 5, direction kept
