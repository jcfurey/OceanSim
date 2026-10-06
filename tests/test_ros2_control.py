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


# --------------------------------------------------------- dynamic velocity mode

def _simulate(m, ctrl, v_cmd, mass=26.0, k_damp=0.0, disturbance=0.0, dt=1.0 / 60.0,
              steps=900):
    """1-DOF surge plant stepped like PhysX: the force is integrated, then
    linear damping k_damp (1/s) applied as v /= (1 + k_damp dt); disturbance is
    a constant push (N), e.g. net buoyancy on the heave axis."""
    v = 0.0
    hist = []
    for _ in range(steps):
        acc, _ = ctrl.update([v_cmd, 0, 0], [0, 0, 0], [v, 0, 0], [0, 0, 0], dt)
        force, _ = m.body_wrench(acc, [0, 0, 0], mass, [1, 1, 1])
        v = (v + dt * (force[0] + disturbance) / mass) / (1.0 + k_damp * dt)
        hist.append(v)
    return np.array(hist)


@pytest.mark.parametrize("k_damp,t90_max", [(0.0, 7.0), (10.0, 0.5), (15.0, 0.5)])
def test_velocity_pi_tracks_platform_damping_and_buoyancy(m, k_damp, t90_max):
    """The platforms' PhysX damping (10-15 1/s) plus a steady 15 N push: with the
    damping feedforward and the auto ki the command is reached quickly, with
    no steady-state error and little overshoot."""
    ctrl = m.BodyVelocityPI(damping_lin=k_damp)
    v = _simulate(m, ctrl, 0.5, k_damp=k_damp, disturbance=-15.0, steps=1800)
    assert v[-1] == pytest.approx(0.5, abs=2e-3)
    assert np.argmax(v >= 0.45) / 60.0 < t90_max
    assert np.max(v) < 0.5 * 1.12


@pytest.mark.parametrize("k_damp", [10.0, 15.0])
def test_velocity_pi_step_settles_within_a_second(m, k_damp):
    v = _simulate(m, m.BodyVelocityPI(damping_lin=k_damp), 0.5, k_damp=k_damp, steps=120)
    settled = np.nonzero(np.abs(v - 0.5) > 0.01)[0].max() + 1
    assert settled / 60.0 < 1.5 and np.max(v) < 0.5 * 1.12


def test_velocity_pi_auto_ki_scales_with_damping(m):
    ctrl = m.BodyVelocityPI(kp_lin=2.0, damping_lin=10.0, kp_ang=3.0, damping_ang=15.0)
    assert ctrl.ki.tolist() == pytest.approx([18.0] * 3 + [40.5] * 3)
    assert m.BodyVelocityPI(ki_lin=0.7).ki[0] == 0.7


def test_velocity_pi_without_feedforward_lags_heavy_damping(m):
    """Why the feedforward exists: P-only on k_damp = 10 settles at
    kp / (kp + k_damp) of the command."""
    ctrl = m.BodyVelocityPI(ki_lin=0.0)
    v = _simulate(m, ctrl, 0.5, k_damp=10.0)
    assert v[-1] == pytest.approx(0.5 * 2.0 / 12.0, abs=2e-3)


def test_velocity_pi_integral_is_clamped(m):
    ctrl = m.BodyVelocityPI(kp_lin=0.0, ki_lin=10.0, i_limit_lin=0.3)
    for _ in range(100):
        acc, _ = ctrl.update([1, -1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 0.1)
    assert acc.tolist() == pytest.approx([0.3, -0.3, 0.0])
    ctrl.reset()
    acc, _ = ctrl.update([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 0.1)
    assert acc.tolist() == [0.0, 0.0, 0.0]


def test_velocity_pi_rejects_non_finite_and_zero_dt(m):
    ctrl = m.BodyVelocityPI()
    acc_l, acc_a = ctrl.update([np.nan, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 0.1)
    assert acc_l.tolist() == [0, 0, 0] and acc_a.tolist() == [0, 0, 0]
    acc_l, _ = ctrl.update([1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 0.0)
    assert acc_l.tolist() == pytest.approx([2.0, 0, 0])   # P term only, no integration
    acc_l, _ = ctrl.update([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 0.1)
    assert acc_l.tolist() == [0, 0, 0]                    # integral never moved


def test_velocity_pi_angular_axis_uses_inertia(m):
    ctrl = m.BodyVelocityPI(damping_ang=10.0)
    acc_l, acc_a = ctrl.update([0, 0, 0], [0, 0, 0.4], [0, 0, 0], [0, 0, 0.1], 0.0)
    force, torque = m.body_wrench(acc_l, acc_a, 26.0, [1.2, 1.5, 0.8])
    assert force.tolist() == [0, 0, 0]
    assert torque.tolist() == pytest.approx([0, 0, 0.8 * (10.0 * 0.4 + 3.0 * 0.3)])


def test_world_to_body_rotation(m):
    # yaw +90 deg: world +x is the body's -y (the body faces world +y)
    s = np.sqrt(0.5)
    v = m.world_to_body([s, 0, 0, s], [1.0, 0.0, 0.0])
    assert v.tolist() == pytest.approx([0.0, -1.0, 0.0], abs=1e-12)
    assert m.world_to_body([1, 0, 0, 0], [0.1, 0.2, 0.3]).tolist() == pytest.approx([0.1, 0.2, 0.3])
