"""Underwater vehicle dynamics for OceanSim platforms (pure numpy).

A 6-DOF Fossen model, applied on top of PhysX's rigid-body integration:

    (M_RB + M_A) nu_dot + C_RB(nu) nu + C_A(nu) nu + D(nu) nu + g(eta) = tau

nu = [v, w] is the body velocity in the body frame (x forward, y port, z up --
REP-103, the frame the platforms' sensor mounts use), about the centre of
gravity (the body origin). PhysX integrates M_RB and C_RB itself; everything
else is applied as a body-frame force / torque each physics step:

* D(nu) nu   -- diagonal linear + quadratic damping (drag).
* g(eta)     -- weight at the CoG and buoyancy at the CoB, so a vehicle with
                its CoB above its CoG rights itself. The scene's gravity stays
                zero (other objects are unchanged); the vehicle's weight is
                applied here. Buoyancy scales with the submerged fraction near
                the water surface, so a positively buoyant ROV floats there.
* M_A, C_A   -- diagonal added mass. PhysX only knows the rigid mass, so the
                applied force is scaled per axis by m / (m + m_A) with the
                Coriolis corrections that make PhysX's rigid-body integration
                reproduce the full equation exactly (physx_wrench).
* tau        -- thrusters: allocation of a commanded wrench (or per-thruster
                commands), saturation, first-order thrust lag.

Contact forces from PhysX see the rigid mass only (no added mass), which is
the usual simulator approximation. Parameters and their sources live with
each platform in ``utils.platforms``.
"""

import math

import numpy as np

GRAVITY = 9.81

# Blue Robotics T200 bollard thrust at full throttle vs supply voltage
# (T200 datasheet: 12 V 3.71 / 2.92 kgf, 16 V 5.25 / 4.1 kgf, 20 V 6.7 / 5.05
# kgf, forward / reverse).
_T200_VOLTS = (12.0, 16.0, 20.0)
_T200_FWD_KGF = (3.71, 5.25, 6.7)
_T200_REV_KGF = (2.92, 4.1, 5.05)

# Shape of thrust vs normalised command, from the T200 regression in von Benzon
# et al. 2022 (J. Mar. Sci. Eng. 10, 1898), eq. 18:
#   F(V) = -140.3 V^9 + 389.9 V^7 - 404.1 V^5 + 176.0 V^3 + 8.9 V.
# Used only as a normalised shape (it peaks just below full command; made
# monotonic here), scaled to the forward / reverse bollard thrust.
_VB_COEFFS = (-140.3, 389.9, -404.1, 176.0, 8.9)
_SHAPE_U = np.linspace(0.0, 1.0, 201)
_SHAPE = np.maximum.accumulate(
    sum(c * _SHAPE_U ** p for c, p in zip(_VB_COEFFS, (9, 7, 5, 3, 1))))
_SHAPE = _SHAPE / _SHAPE[-1]


def t200_max_thrust(voltage):
    """(forward, reverse) full-throttle T200 thrust in N at ``voltage``
    (linear between the datasheet points, clamped to 12-20 V)."""
    v = float(voltage)
    fwd = float(np.interp(v, _T200_VOLTS, _T200_FWD_KGF)) * GRAVITY
    rev = float(np.interp(v, _T200_VOLTS, _T200_REV_KGF)) * GRAVITY
    return fwd, rev


def thrust_from_command(u, max_forward, max_reverse):
    """Thrust (N) for normalised commands ``u`` in [-1, 1] (array-like):
    the T200 shape scaled to the forward / reverse maximum."""
    u = np.clip(np.asarray(u, dtype=float), -1.0, 1.0)
    shape = np.interp(np.abs(u), _SHAPE_U, _SHAPE)
    return np.where(u >= 0.0, shape * np.asarray(max_forward, float),
                    -shape * np.asarray(max_reverse, float))


class ThrusterArray:
    """Fixed thrusters: position r_i (m) and unit direction e_i in the body
    frame, the direction of positive ("forward") thrust. Thrust f_i is limited
    to [-max_reverse_i, max_forward_i] and follows its command with a
    first-order lag (time constant tau, s). The body wrench is B f with
    columns [e_i; r_i x e_i]."""

    def __init__(self, positions, directions, max_forward, max_reverse, time_constant=0.06):
        self.positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        d = np.asarray(directions, dtype=float).reshape(-1, 3)
        self.directions = d / np.linalg.norm(d, axis=1, keepdims=True)
        n = self.positions.shape[0]
        self.max_forward = np.broadcast_to(np.asarray(max_forward, float), (n,)).copy()
        self.max_reverse = np.broadcast_to(np.asarray(max_reverse, float), (n,)).copy()
        self.time_constant = float(time_constant)
        self.B = np.vstack([self.directions.T,
                            np.cross(self.positions, self.directions).T])
        self._pinv = np.linalg.pinv(self.B)
        self.thrust = np.zeros(n)

    @property
    def count(self):
        return self.positions.shape[0]

    def allocate(self, wrench):
        """Least-squares thrusts for a body wrench [F; T] (min-norm where the
        layout is redundant; uncontrollable components, e.g. pitch on a
        6-thruster BlueROV2, are dropped). If any thruster would exceed its
        limit the whole vector is scaled down, keeping the wrench direction."""
        f = self._pinv @ np.asarray(wrench, dtype=float).reshape(6)
        ratio = np.where(f >= 0.0, f / self.max_forward, -f / self.max_reverse)
        worst = float(np.max(ratio)) if ratio.size else 0.0
        if worst > 1.0:
            f = f / worst
        return f

    def saturate(self, f):
        return np.clip(np.asarray(f, dtype=float), -self.max_reverse, self.max_forward)

    def step(self, f_cmd, dt):
        """Advance the thrust lag one step toward ``f_cmd`` (clipped to the
        limits); returns the thrust now produced."""
        target = self.saturate(f_cmd)
        if dt is None or dt <= 0.0 or self.time_constant <= 0.0:
            self.thrust = target
        else:
            self.thrust = self.thrust + (target - self.thrust) * (
                1.0 - math.exp(-float(dt) / self.time_constant))
        return self.thrust

    def wrench(self, f=None):
        return self.B @ (self.thrust if f is None else np.asarray(f, dtype=float))

    def reset(self):
        self.thrust[:] = 0.0


class HydroModel:
    """Rigid-body and hydrodynamic parameters (body frame, about the CoG).

    mass (kg); inertia: principal (Ixx, Iyy, Izz) kg m^2; added_mass,
    linear_damping, quadratic_damping: 6-vectors (surge, sway, heave, roll,
    pitch, yaw) as positive magnitudes (SI: kg / kg m^2; N s/m, N m s;
    N s^2/m^2, N m s^2); volume: displaced volume (m^3); cob: centre of
    buoyancy relative to the CoG (m); height: vertical extent (m) used for
    partial submergence at the surface."""

    def __init__(self, mass, inertia, added_mass, linear_damping, quadratic_damping,
                 volume, cob=(0.0, 0.0, 0.0), height=0.25):
        self.mass = float(mass)
        self.inertia = np.asarray(inertia, dtype=float).reshape(3)
        self.added_mass = np.asarray(added_mass, dtype=float).reshape(6)
        self.linear_damping = np.asarray(linear_damping, dtype=float).reshape(6)
        self.quadratic_damping = np.asarray(quadratic_damping, dtype=float).reshape(6)
        self.volume = float(volume)
        self.cob = np.asarray(cob, dtype=float).reshape(3)
        self.height = float(height)

    def effective_mass(self):
        """Diagonal of M_RB + M_A (6,): what a body-frame acceleration costs."""
        return np.concatenate([np.full(3, self.mass), self.inertia]) + self.added_mass

    def damping(self, nu):
        """-D(nu) nu: the drag wrench on the body at velocity nu (6,)."""
        nu = np.asarray(nu, dtype=float).reshape(6)
        return -(self.linear_damping + self.quadratic_damping * np.abs(nu)) * nu

    def restoring(self, rot_wb, rho=1000.0, g=GRAVITY, submerged=1.0):
        """-g(eta): weight at the CoG plus buoyancy at the CoB, as a body
        wrench. rot_wb is the body -> world rotation (world z up)."""
        rot_wb = np.asarray(rot_wb, dtype=float).reshape(3, 3)
        weight = self.mass * g
        buoyancy = float(submerged) * rho * g * self.volume
        up_body = rot_wb.T @ np.array([0.0, 0.0, 1.0])
        force = (buoyancy - weight) * up_body
        torque = np.cross(self.cob, buoyancy * up_body)
        return np.concatenate([force, torque])

    def added_mass_coriolis(self, nu):
        """-C_A(nu) nu for diagonal added mass: [a x w; a x v + b x w] with
        a = M_A,lin v and b = M_A,rot w (gives the Munk moment)."""
        nu = np.asarray(nu, dtype=float).reshape(6)
        v, w = nu[:3], nu[3:]
        a = self.added_mass[:3] * v
        b = self.added_mass[3:] * w
        return np.concatenate([np.cross(a, w), np.cross(a, v) + np.cross(b, w)])

    def physx_wrench(self, tau, nu):
        """Body wrench to hand PhysX so its rigid-body step (mass m, inertia I,
        its own C_RB) produces the full model's acceleration for the external
        wrench ``tau`` (thrust + damping + restoring, without C_A):

            F = s_v * (tau_v + tau_CA_v + m_A * (w x v)),     s_v = m / (m + m_A)
            T = s_w * (tau_w + tau_CA_w + (I_A / I) * (w x I w)),  s_w = I / (I + I_A)

        Exact for diagonal added mass with the CoG at the body origin."""
        tau = np.asarray(tau, dtype=float).reshape(6)
        nu = np.asarray(nu, dtype=float).reshape(6)
        v, w = nu[:3], nu[3:]
        ca = self.added_mass_coriolis(nu)
        m_a = self.added_mass[:3]
        i_a = self.added_mass[3:]
        s_v = self.mass / (self.mass + m_a)
        s_w = self.inertia / (self.inertia + i_a)
        force = s_v * (tau[:3] + ca[:3] + m_a * np.cross(w, v))
        torque = s_w * (tau[3:] + ca[3:] + (i_a / self.inertia) * np.cross(w, self.inertia * w))
        return force, torque


def submerged_fraction(z_cob_world, surface_z, height):
    """Fraction of the hull below the surface, treating the hull as a slab of
    ``height`` centred on the CoB (1 when surface_z is None)."""
    if surface_z is None or height <= 0.0:
        return 1.0
    return float(np.clip((float(surface_z) - (float(z_cob_world) - 0.5 * height)) / height,
                         0.0, 1.0))


def rot_from_quat_wxyz(q):
    """Body -> world rotation matrix from a scalar-first quaternion."""
    w, x, y, z = (float(c) for c in q)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0.0 else 0.0
    return np.array([
        [1.0 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1.0 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1.0 - s * (x * x + y * y)],
    ])


class VehicleModel:
    """Hydrodynamics + thrusters for one vehicle. Each physics step, call
    ``step`` with the body velocity and pose; it returns the body-frame force
    and torque to apply to the PhysX rigid body (at its centre of mass).

    Commands: ``command_wrench`` (allocated to the thrusters) or
    ``command_thrusters`` (normalised per-thruster commands in [-1, 1], through
    the thrust curve). The latest command holds until replaced."""

    def __init__(self, hydro, thrusters, rho=1000.0, g=GRAVITY, surface_z=None):
        self.hydro = hydro
        self.thrusters = thrusters
        self.rho = float(rho)
        self.g = float(g)
        self.surface_z = surface_z
        self._wrench_cmd = np.zeros(6)
        self._thruster_cmd = None
        self.last_submerged = 1.0

    def command_wrench(self, force, torque):
        self._wrench_cmd = np.concatenate([np.asarray(force, float).reshape(3),
                                           np.asarray(torque, float).reshape(3)])
        self._thruster_cmd = None

    def command_thrusters(self, u):
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.size != self.thrusters.count:
            raise ValueError(f"expected {self.thrusters.count} thruster commands, got {u.size}")
        self._thruster_cmd = np.where(np.isfinite(u), u, 0.0)

    def stop(self):
        self.command_wrench(np.zeros(3), np.zeros(3))

    def max_wrench(self, direction):
        """Largest thrust wrench achievable along a 6-D ``direction``."""
        d = np.asarray(direction, dtype=float).reshape(6)
        return self.thrusters.wrench(self.thrusters.allocate(d * 1e6))

    def feedforward(self, nu_cmd):
        """Thrust wrench that holds the steady velocity ``nu_cmd`` against drag."""
        return -self.hydro.damping(nu_cmd)

    def step(self, nu, rot_wb, position, dt):
        """(force, torque) body-frame wrench for PhysX this step."""
        nu = np.asarray(nu, dtype=float).reshape(6)
        rot_wb = np.asarray(rot_wb, dtype=float).reshape(3, 3)
        if self._thruster_cmd is not None:
            f_cmd = thrust_from_command(self._thruster_cmd, self.thrusters.max_forward,
                                        self.thrusters.max_reverse)
        else:
            f_cmd = self.thrusters.allocate(self._wrench_cmd)
        self.thrusters.step(f_cmd, dt)
        z_cob = float(np.asarray(position, float).reshape(3)[2] + (rot_wb @ self.hydro.cob)[2])
        self.last_submerged = submerged_fraction(z_cob, self.surface_z, self.hydro.height)
        tau = (self.thrusters.wrench() + self.hydro.damping(nu)
               + self.hydro.restoring(rot_wb, self.rho, self.g, self.last_submerged))
        return self.hydro.physx_wrench(tau, nu)


def ellipsoid_added_mass(a, b, c, rho, volume, samples=20000):
    """Added mass (6,) of a solid ellipsoid with semi-axes a, b, c along x, y,
    z (Lamb's coefficients), scaled to a body of displaced ``volume``:
    translational k_i * rho * volume with k = alpha0 / (2 - alpha0) etc., and
    rotational from Lamb's ellipsoid formulas. An estimate for vehicles with no
    measured added mass."""
    t = (np.arange(samples) + 0.5) / samples
    s = t / (1.0 - t)
    ds = 1.0 / (1.0 - t) ** 2
    a2, b2, c2 = a * a, b * b, c * c
    delta = np.sqrt((a2 + s) * (b2 + s) * (c2 + s))
    abc = a * b * c
    alpha0 = abc * np.sum(ds / ((a2 + s) * delta)) / samples
    beta0 = abc * np.sum(ds / ((b2 + s) * delta)) / samples
    gamma0 = abc * np.sum(ds / ((c2 + s) * delta)) / samples
    m = rho * volume
    k_lin = [alpha0 / (2.0 - alpha0), beta0 / (2.0 - beta0), gamma0 / (2.0 - gamma0)]

    def _rot(p2, q2, p0, q0):
        # rotation about the third axis, with (p, q) the other two
        if abs(p2 - q2) < 1e-12:
            return 0.0
        return 0.2 * m * (p2 - q2) ** 2 * (q0 - p0) / (
            2.0 * (p2 - q2) + (p2 + q2) * (p0 - q0))

    i_x = _rot(b2, c2, beta0, gamma0)
    i_y = _rot(c2, a2, gamma0, alpha0)
    i_z = _rot(a2, b2, alpha0, beta0)
    return np.array([k_lin[0] * m, k_lin[1] * m, k_lin[2] * m, i_x, i_y, i_z])


def from_platform(spec, rho=1000.0, g=GRAVITY, surface_z=None, voltage=None,
                  drag_scale=1.0, mass=None):
    """VehicleModel for a ``platforms.PlatformSpec`` with a ``hydro`` entry.
    voltage overrides the T200 supply voltage; drag_scale multiplies both
    damping terms (for calibrating against your own vehicle); mass overrides
    the in-air mass (the displaced volume, and so the buoyancy, is kept)."""
    h = spec.hydro
    if h is None:
        raise ValueError(f"platform {spec.name!r} has no hydrodynamic parameters")
    hydro = HydroModel(spec.mass if mass is None else float(mass), h.inertia, h.added_mass,
                       np.asarray(h.linear_damping) * drag_scale,
                       np.asarray(h.quadratic_damping) * drag_scale,
                       h.displaced_volume, h.cob, h.height)
    pos = [t.position for t in h.thrusters]
    dirs = [t.direction for t in h.thrusters]
    if h.thruster_model == "T200":
        fwd, rev = t200_max_thrust(voltage if voltage is not None else h.battery_voltage)
        max_fwd = [fwd] * len(pos)
        max_rev = [rev] * len(pos)
    else:
        max_fwd = [t.max_forward for t in h.thrusters]
        max_rev = [t.max_reverse for t in h.thrusters]
    thrusters = ThrusterArray(pos, dirs, max_fwd, max_rev, h.thruster_time_constant)
    return VehicleModel(hydro, thrusters, rho=rho, g=g, surface_z=surface_z)
