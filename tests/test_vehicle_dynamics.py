"""Tests for utils/vehicle_dynamics.py (pure numpy, no Isaac Sim)."""

import importlib.util
import math
import os

import pytest

np = pytest.importorskip("numpy")

_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                     "utils", "vehicle_dynamics.py")


def _load():
    spec = importlib.util.spec_from_file_location("vehicle_dynamics", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vd():
    return _load()


S = math.sqrt(0.5)
# BlueROV2-like layout (vectored X + two verticals), body x fwd / y port / z up
POS = [(0.14, -0.092, 0.0), (0.14, 0.092, 0.0), (-0.14, -0.092, 0.0), (-0.14, 0.092, 0.0),
       (0.0, -0.109, 0.077), (0.0, 0.109, 0.077)]
DIR = [(S, S, 0.0), (S, -S, 0.0), (S, -S, 0.0), (S, S, 0.0), (0, 0, 1.0), (0, 0, 1.0)]


def _hydro(vd, **kw):
    p = dict(mass=11.5, inertia=(0.114, 0.205, 0.309),
             added_mass=(6.36, 7.12, 18.68, 0.189, 0.135, 0.222),
             linear_damping=(13.7, 0.0, 33.0, 0.0, 0.8, 0.0),
             quadratic_damping=(141.0, 217.0, 190.0, 1.19, 0.47, 1.5),
             volume=0.0116, cob=(0.0, 0.0, 0.01), height=0.254)
    p.update(kw)
    return vd.HydroModel(**p)


def _thrusters(vd, fwd=47.0, rev=36.8, tau=0.06):
    return vd.ThrusterArray(POS, DIR, fwd, rev, tau)


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# --- thrusters -----------------------------------------------------------------

def test_t200_datasheet_points(vd):
    assert vd.t200_max_thrust(16.0) == pytest.approx((5.25 * 9.81, 4.1 * 9.81))
    assert vd.t200_max_thrust(25.0) == pytest.approx((6.7 * 9.81, 5.05 * 9.81))
    fwd, rev = vd.t200_max_thrust(14.8)
    assert 5.25 * 9.81 > fwd > 3.71 * 9.81 and rev < fwd


def test_thrust_curve_monotonic_and_asymmetric(vd):
    u = np.linspace(-1, 1, 81)
    f = vd.thrust_from_command(u, 50.0, 40.0)
    assert f[0] == pytest.approx(-40.0) and f[-1] == pytest.approx(50.0)
    assert f[40] == 0.0
    assert np.all(np.diff(f) >= 0.0)
    assert vd.thrust_from_command(0.3, 50.0, 40.0) < 0.3 * 50.0   # cubic-ish low end


def test_allocation_reproduces_feasible_wrench(vd):
    t = _thrusters(vd)
    w = np.array([20.0, -10.0, 15.0, 1.0, 0.0, 2.0])    # no pitch (uncontrollable)
    f = t.allocate(w)
    assert np.allclose(t.B @ f, w, atol=1e-9)


def test_pitch_is_uncontrollable_with_six_thrusters(vd):
    t = _thrusters(vd)
    assert np.linalg.matrix_rank(t.B) == 5
    f = t.allocate([0, 0, 0, 0, 5.0, 0])
    assert np.allclose(t.B @ f, 0.0, atol=1e-9)


def test_allocation_saturates_preserving_direction(vd):
    t = _thrusters(vd)
    w = np.array([500.0, 0, 0, 0, 0, 30.0])
    f = t.allocate(w)
    assert np.all(f <= t.max_forward + 1e-9) and np.all(f >= -t.max_reverse - 1e-9)
    out = t.B @ f
    assert np.allclose(out / np.linalg.norm(out), w / np.linalg.norm(w), atol=1e-9)
    assert np.isclose(np.max(np.where(f >= 0, f / t.max_forward, -f / t.max_reverse)), 1.0)


def test_max_surge_uses_all_four_horizontal_thrusters(vd):
    t = _thrusters(vd)
    f = t.allocate([1e6, 0, 0, 0, 0, 0])
    assert (t.B @ f)[0] == pytest.approx(4 * 47.0 * S)


def test_thrust_lag(vd):
    t = _thrusters(vd, tau=0.1)
    cmd = np.full(6, 10.0)
    for _ in range(50):                        # 0.5 s = 5 tau at 100 Hz
        t.step(cmd, 0.01)
    assert t.thrust == pytest.approx(cmd * (1 - math.exp(-5)), rel=1e-6)
    t.step(np.full(6, 1e3), 0.0)               # dt 0 -> jumps, clipped to limits
    assert t.thrust.tolist() == pytest.approx([47.0] * 6)


# --- hydrodynamics -------------------------------------------------------------

def test_damping_opposes_motion(vd):
    h = _hydro(vd)
    nu = np.array([0.5, -0.3, 0.2, 0.4, -0.1, 0.3])
    d = h.damping(nu)
    assert np.all(d * nu <= 0.0)
    assert d[0] == pytest.approx(-(13.7 * 0.5 + 141.0 * 0.25))


def test_restoring_net_buoyancy_and_righting_moment(vd):
    h = _hydro(vd)
    up = h.restoring(np.eye(3), rho=1000.0)
    assert up[2] == pytest.approx(1000 * 9.81 * 0.0116 - 11.5 * 9.81)
    assert np.allclose(up[[0, 1, 3, 4, 5]], 0.0)
    rolled = h.restoring(_rot("x", 10.0), rho=1000.0)
    assert rolled[3] < 0.0                    # positive roll -> negative (righting) moment
    pitched = h.restoring(_rot("y", -15.0), rho=1000.0)
    assert pitched[4] > 0.0


def test_munk_moment_sign(vd):
    """A body with m_11 < m_22 moving at a sideslip gets the destabilising
    yaw moment (m_11 - m_22) u v."""
    h = _hydro(vd)
    ca = h.added_mass_coriolis([1.0, 0.2, 0, 0, 0, 0])
    assert ca[5] == pytest.approx((6.36 - 7.12) * 1.0 * 0.2)


def _skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def _rot_step(r, w, dt):
    th = np.linalg.norm(w) * dt
    if th < 1e-12:
        return r
    k = _skew(w / np.linalg.norm(w))
    return r @ (np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * k @ k)


def test_physx_wrench_reproduces_full_fossen_model(vd):
    """Integrate (a) the full body-frame model with added mass and C_A, and (b)
    a PhysX-like rigid body (world-frame momentum, body Euler equation with the
    RIGID mass / inertia) driven by physx_wrench. Trajectories must agree."""
    h = _hydro(vd)
    m = h.mass
    inertia = h.inertia
    eff = h.effective_mass()
    thrust = np.array([60.0, 25.0, -10.0, 0.8, 0.0, 1.5])

    def tau_ext(nu, r):
        return thrust + h.damping(nu) + h.restoring(r)

    dt, steps = 2e-4, 10000                    # 2 s
    # (a) reference
    nu_a = np.array([0.1, 0.0, 0.0, 0.5, -0.3, 0.8])
    r_a = np.eye(3)
    # (b) PhysX-like
    v_w = r_a @ nu_a[:3]
    w_b = nu_a[3:].copy()
    r_b = np.eye(3)
    for _ in range(steps):
        v, w = nu_a[:3], nu_a[3:]
        ca = h.added_mass_coriolis(nu_a)
        rhs = tau_ext(nu_a, r_a) + ca - np.concatenate([m * np.cross(w, v),
                                                        np.cross(w, inertia * w)])
        nu_a = nu_a + dt * rhs / eff
        r_a = _rot_step(r_a, w, dt)

        nu_b = np.concatenate([r_b.T @ v_w, w_b])
        force, torque = h.physx_wrench(tau_ext(nu_b, r_b), nu_b)
        v_w = v_w + dt * (r_b @ force) / m
        w_old = w_b
        w_b = w_b + dt * (torque - np.cross(w_b, inertia * w_b)) / inertia
        r_b = _rot_step(r_b, w_old, dt)
    nu_b = np.concatenate([r_b.T @ v_w, w_b])
    assert np.abs(nu_a).max() > 0.2            # it actually moved
    assert np.allclose(nu_a, nu_b, atol=2e-3), (nu_a, nu_b)
    assert np.allclose(r_a, r_b, atol=2e-3)


def test_added_mass_slows_the_initial_acceleration(vd):
    h = _hydro(vd)
    force, torque = h.physx_wrench([10.0, 0, 0, 0, 0, 1.0], np.zeros(6))
    assert force[0] / h.mass == pytest.approx(10.0 / (11.5 + 6.36))
    assert torque[2] / h.inertia[2] == pytest.approx(1.0 / (0.309 + 0.222))


# --- surface, vehicle model -----------------------------------------------------

def test_submerged_fraction(vd):
    assert vd.submerged_fraction(-1.0, 0.0, 0.25) == 1.0
    assert vd.submerged_fraction(1.0, 0.0, 0.25) == 0.0
    assert vd.submerged_fraction(0.0, 0.0, 0.25) == pytest.approx(0.5)
    assert vd.submerged_fraction(5.0, None, 0.25) == 1.0


def test_positively_buoyant_vehicle_floats_at_the_surface(vd):
    """Released below the surface with no thrust, a positively buoyant ROV
    rises and settles with weight / buoyancy of its height submerged."""
    h = _hydro(vd)
    model = vd.VehicleModel(h, _thrusters(vd), rho=1000.0, surface_z=0.0)
    z, vz, dt = -1.0, 0.0, 0.005
    for _ in range(int(60 / dt)):
        force, _ = model.step([0, 0, vz, 0, 0, 0], np.eye(3), [0, 0, z], dt)
        vz += dt * force[2] / h.mass
        z += dt * vz
    frac = (11.5 * 9.81) / (1000 * 9.81 * 0.0116)
    assert model.last_submerged == pytest.approx(frac, abs=2e-3)
    assert abs(vz) < 1e-3


def test_top_speed_from_thrust_and_drag(vd):
    model = vd.VehicleModel(_hydro(vd), _thrusters(vd))
    model.command_wrench([1e6, 0, 0], [0, 0, 0])
    u, dt = 0.0, 0.01
    for _ in range(3000):
        force, _ = model.step([u, 0, 0, 0, 0, 0], np.eye(3), [0, 0, -5], dt)
        u += dt * force[0] / 11.5
    f_max = 4 * 47.0 * S
    expect = (-13.7 + math.sqrt(13.7 ** 2 + 4 * 141.0 * f_max)) / (2 * 141.0)
    assert u == pytest.approx(expect, rel=1e-3)


def test_thruster_commands_through_curve(vd):
    model = vd.VehicleModel(_hydro(vd), _thrusters(vd, tau=0.0))
    model.command_thrusters([1, 1, 1, 1, 0, 0])
    model.step(np.zeros(6), np.eye(3), [0, 0, -5], 0.01)
    assert model.thrusters.thrust[:4].tolist() == pytest.approx([47.0] * 4)
    with pytest.raises(ValueError):
        model.command_thrusters([1, 1])
    model.command_thrusters([np.nan, 0, 0, 0, 0, -1])
    model.step(np.zeros(6), np.eye(3), [0, 0, -5], 0.01)
    assert model.thrusters.thrust[0] == 0.0 and model.thrusters.thrust[5] == pytest.approx(-36.8)


def test_feedforward_cancels_drag(vd):
    model = vd.VehicleModel(_hydro(vd), _thrusters(vd))
    nu = np.array([0.4, 0.1, 0, 0, 0, 0.2])
    assert np.allclose(model.feedforward(nu) + model.hydro.damping(nu), 0.0)


# --- added-mass estimate ---------------------------------------------------------

def test_ellipsoid_added_mass_sphere(vd):
    r = 0.2
    vol = 4 / 3 * math.pi * r ** 3
    ma = vd.ellipsoid_added_mass(r, r, r, 1000.0, vol)
    assert ma[:3] == pytest.approx([0.5 * 1000 * vol] * 3, rel=1e-3)
    assert ma[3:] == pytest.approx([0, 0, 0], abs=1e-9)


def test_ellipsoid_added_mass_prolate_spheroid_matches_lamb(vd):
    """Closed-form Lamb coefficients for a prolate spheroid a > b = c."""
    a, b = 2.0, 1.0
    e = math.sqrt(1 - (b / a) ** 2)
    alpha0 = 2 * (1 - e * e) / e ** 3 * (0.5 * math.log((1 + e) / (1 - e)) - e)
    beta0 = 1 / e ** 2 - (1 - e * e) / (2 * e ** 3) * math.log((1 + e) / (1 - e))
    vol = 4 / 3 * math.pi * a * b * b
    k1 = alpha0 / (2 - alpha0)
    k2 = beta0 / (2 - beta0)
    k_rot = e ** 4 * (beta0 - alpha0) / ((2 - e * e) * (2 * e * e - (2 - e * e) * (beta0 - alpha0)))
    i_fluid = 0.2 * 1000 * vol * (a * a + b * b)
    ma = vd.ellipsoid_added_mass(a, b, b, 1000.0, vol)
    assert ma[0] == pytest.approx(k1 * 1000 * vol, rel=2e-3)
    assert ma[1] == pytest.approx(k2 * 1000 * vol, rel=2e-3)
    assert ma[3] == pytest.approx(0.0, abs=1e-9)
    assert ma[4] == pytest.approx(k_rot * i_fluid, rel=5e-3)
    assert ma[5] == pytest.approx(k_rot * i_fluid, rel=5e-3)
