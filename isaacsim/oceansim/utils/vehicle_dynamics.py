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

    def __init__(self, hydro, thrusters, rho=1000.0, g=GRAVITY, surface_z=None,
                 cog=(0.0, 0.0, 0.0)):
        self.hydro = hydro
        self.thrusters = thrusters
        self.rho = float(rho)
        self.g = float(g)
        self.surface_z = surface_z
        # CoG relative to the prim origin: positions are reported for the
        # origin, everything else in this model is about the CoG.
        self.cog = np.asarray(cog, dtype=float).reshape(3)
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
        z_cob = float(np.asarray(position, float).reshape(3)[2]
                      + (rot_wb @ (self.cog + self.hydro.cob))[2])
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


def _thruster_limits(thrusters, thruster_model, voltage):
    if thruster_model == "T200":
        fwd, rev = t200_max_thrust(voltage)
        return [fwd] * len(thrusters), [rev] * len(thrusters)
    fwd = [t.max_forward for t in thrusters]
    rev = [t.max_reverse if t.max_reverse is not None else t.max_forward for t in thrusters]
    if any(v is None for v in fwd):
        raise ValueError("thrusters need max_forward unless thruster_model is 'T200'")
    return fwd, rev


def estimate_hydro(spec, rho=1000.0):
    """Hydrodynamic parameters estimated from a platform's ``HydroEstimate``
    (see platforms.HydroEstimate for the method). Returns a HydroSpec-like
    dict (keys as HydroSpec fields)."""
    est = spec.hydro
    length, width, height = spec.dimensions
    mass = float(spec.mass)
    volume = mass * (1.0 + est.net_buoyancy) / rho
    voltage = est.battery_voltage
    areas = (width * height, length * height, length * width)
    speeds = list(est.top_speeds or (None, None, None))
    thruster_specs = est.thrusters
    if est.thruster_model is None:
        vertical = [abs(t.direction[2]) > 0.7 for t in est.thrusters]
        h_unknown = any(t.max_forward is None for t, v in zip(est.thrusters, vertical) if not v)
        v_unknown = any(t.max_forward is None for t, v in zip(est.thrusters, vertical) if v)
        if h_unknown or v_unknown:
            thruster_specs = _size_thrusters(est, areas, speeds, rho, h_unknown, v_unknown)
            # axes whose thrust was sized from the drag model keep that drag
            if h_unknown:
                speeds[0] = speeds[1] = None
            if v_unknown:
                speeds[2] = None
    fwd, rev = _thruster_limits(thruster_specs, est.thruster_model, voltage)
    thrusters = ThrusterArray([t.position for t in thruster_specs],
                              [t.direction for t in thruster_specs], fwd, rev)
    quad = []
    cds = []
    for axis in range(3):
        speed = speeds[axis]
        if speed:
            d = np.zeros(6)
            d[axis] = 1.0
            force = float(thrusters.wrench(thrusters.allocate(d * 1e6))[axis])
            q = force / (speed * speed + est.linear_fraction * speed)
            cd = 2.0 * q / (rho * areas[axis])
        else:
            cd = float(est.drag_coefficients[axis])
            q = 0.5 * rho * cd * areas[axis]
        quad.append(q)
        cds.append(cd)
    quad += [rho * cds[2] * length * width ** 4 / 64.0,      # roll: heave strips across W
             rho * cds[2] * width * length ** 4 / 64.0,      # pitch: heave strips along L
             rho * cds[1] * height * length ** 4 / 64.0]     # yaw: sway strips along L
    box = mass / 12.0
    inertia = tuple(est.inertia_factor * box * v for v in
                    (width ** 2 + height ** 2, length ** 2 + height ** 2, length ** 2 + width ** 2))
    added = ellipsoid_added_mass(length / 2, width / 2, height / 2, rho, volume)
    return dict(inertia=inertia, displaced_volume=volume, cob=(0.0, 0.0, est.cob_height),
                height=height, added_mass=tuple(float(v) for v in added),
                linear_damping=tuple(est.linear_fraction * q for q in quad),
                quadratic_damping=tuple(float(q) for q in quad),
                thrusters=thruster_specs, thruster_model=est.thruster_model,
                battery_voltage=est.battery_voltage,
                thruster_time_constant=est.thruster_time_constant,
                sources=est.sources, cog=(0.0, 0.0, 0.0),
                drag_coefficients=tuple(cds))


FOAM_DENSITY = 350.0      # kg/m^3, syntactic trim foam (shallow-rated)
LEAD_DENSITY = 11340.0


def _size_thrusters(est, areas, speeds, rho, h_unknown, v_unknown):
    """Limits for thrusters whose thrust is unpublished: the force the drag
    model (drag_coefficients) needs at the listed surge (horizontal) / heave
    (vertical) speed, shared by that group (symmetric limits). Vertical
    thrusters without a listed heave speed get the horizontal per-thruster
    limit (same motors assumed)."""
    vertical = [abs(t.direction[2]) > 0.7 for t in est.thrusters]

    def _need(axis, speed):
        q = 0.5 * rho * float(est.drag_coefficients[axis]) * areas[axis]
        return q * speed * speed + est.linear_fraction * q * speed

    def _per_unit(axis, group):
        # wrench along axis per newton of limit, using only the group's thrusters
        idx = [i for i, g in enumerate(vertical) if g == group]
        if not idx:
            return 0.0
        sub = ThrusterArray([est.thrusters[i].position for i in idx],
                            [est.thrusters[i].direction for i in idx], 1.0, 1.0)
        d = np.zeros(6)
        d[axis] = 1.0
        return float(sub.wrench(sub.allocate(d * 1e6))[axis])

    h_limit = v_limit = None
    if h_unknown:
        if not speeds[0]:
            raise ValueError("unpublished horizontal thrust needs a listed surge speed")
        h_limit = _need(0, speeds[0]) / _per_unit(0, False)
    if v_unknown:
        if speeds[2]:
            v_limit = _need(2, speeds[2]) / _per_unit(2, True)
        else:
            known = [t.max_forward for t, v in zip(est.thrusters, vertical)
                     if not v and t.max_forward is not None]
            v_limit = h_limit if h_limit is not None else (sum(known) / len(known) if known else None)
            if v_limit is None:
                raise ValueError("unpublished vertical thrust needs a heave speed or horizontal thrust")
    out = []
    for t, v in zip(est.thrusters, vertical):
        if t.max_forward is not None:
            out.append(t)
        else:
            lim = v_limit if v else h_limit
            out.append(type(t)(t.position, t.direction, lim, lim))
    return tuple(out)


def resolve_hydro(spec, payloads=(), rho=1000.0, mass=None, trim=True):
    """The vehicle's hydrodynamic parameters with ``payloads`` fitted, as a
    dict of HydroSpec fields plus "mass" and "trim".

    Measured HydroSpecs are used as given, HydroEstimates estimated. Each
    payload (utils.payloads.PayloadSpec, mounted at the platform's mount for
    its kind) adds its mass and displaced volume at its mount point -- moving
    the CoG and CoB, with parallel-axis inertia -- and drag over its projected
    area (Cd 1). Positions in the result (thrusters, cob) are relative to the
    new CoG; ``cog`` is the CoG's offset from the prim origin.

    ``trim`` re-ballasts like an operator does after fitting payloads: it
    restores the bare vehicle's net buoyancy fraction (in water of density
    ``rho``) with syntactic foam at the top of the frame, or lead at the
    bottom if the payloads made it lighter. The result's "trim" entry says
    what was added (kind, mass, volume)."""
    h = spec.hydro
    if h is None:
        raise ValueError(f"platform {spec.name!r} has no hydrodynamic parameters")
    if hasattr(h, "top_speeds"):
        base = estimate_hydro(spec, rho)
    else:
        base = {f: getattr(h, f) for f in (
            "inertia", "displaced_volume", "cob", "height", "added_mass", "linear_damping",
            "quadratic_damping", "thrusters", "thruster_model", "battery_voltage",
            "thruster_time_constant", "sources", "cog")}
    m0 = float(spec.mass if mass is None else mass)
    r0 = np.asarray(base["cog"], dtype=float)
    v0 = float(base["displaced_volume"])
    cob0 = r0 + np.asarray(base["cob"], dtype=float)          # CoB relative to the origin
    masses = [(m0, r0)]
    volumes = [(v0, cob0)]
    quad = np.asarray(base["quadratic_damping"], dtype=float).copy()
    for p in payloads:
        mount = spec.mount(p.kind)
        r = np.asarray(mount.translation, dtype=float) + np.asarray(p.offset, dtype=float)
        masses.append((float(p.mass), r))
        volumes.append((float(p.volume), r))
        length, width, height = p.size
        quad[:3] += 0.5 * rho * 1.0 * np.array([width * height, length * height, length * width])
    trim_info = None
    if trim and payloads:
        net0 = (rho * v0 - m0) / m0                      # bare vehicle's trim
        lift = (1.0 + net0) * sum(m for m, _ in masses) - rho * sum(v for v, _ in volumes)
        height = float(base["height"])
        # (rho (V + v) - (M + m)) / (M + m) = net0 with the trim's own (m, v)
        if lift > 1e-9:          # heavier: foam high on the frame
            vol = lift / (rho - FOAM_DENSITY * (1.0 + net0))
            at = r0 + np.array([0.0, 0.0, 0.4 * height])
            masses.append((FOAM_DENSITY * vol, at))
            volumes.append((vol, at))
            trim_info = dict(kind="foam", mass=FOAM_DENSITY * vol, volume=vol, position=tuple(at))
        elif lift < -1e-9:       # lighter: lead low on the frame
            m_b = -lift / (1.0 + net0 - rho / LEAD_DENSITY)
            at = r0 + np.array([0.0, 0.0, -0.4 * height])
            masses.append((m_b, at))
            volumes.append((m_b / LEAD_DENSITY, at))
            trim_info = dict(kind="lead", mass=m_b, volume=m_b / LEAD_DENSITY, position=tuple(at))
    m_tot = sum(m for m, _ in masses)
    cog = sum(m * r for m, r in masses) / m_tot
    v_tot = sum(v for v, _ in volumes)
    cob = sum(v * r for v, r in volumes) / v_tot
    inertia = np.asarray(base["inertia"], dtype=float) + m0 * _parallel_axis(r0 - cog)
    for m, r in masses[1:]:
        inertia = inertia + m * _parallel_axis(r - cog)
    shift = cog - r0
    thrusters = tuple(type(t)(tuple(np.asarray(t.position, float) - shift), t.direction,
                              t.max_forward, t.max_reverse) for t in base["thrusters"])
    lin = np.asarray(base["linear_damping"], dtype=float)
    # payload drag keeps the vehicle's linear / quadratic ratio
    ratio = np.divide(lin[:3], np.asarray(base["quadratic_damping"], float)[:3],
                      out=np.zeros(3), where=np.asarray(base["quadratic_damping"], float)[:3] > 0)
    lin = lin.copy()
    lin[:3] = ratio * quad[:3]
    out = dict(base)
    out.update(mass=m_tot, inertia=tuple(float(v) for v in inertia), displaced_volume=v_tot,
               cob=tuple(float(v) for v in cob - cog), cog=tuple(float(v) for v in cog),
               thrusters=thrusters, quadratic_damping=tuple(float(v) for v in quad),
               linear_damping=tuple(float(v) for v in lin), trim=trim_info)
    return out


def _parallel_axis(d):
    d = np.asarray(d, dtype=float)
    return np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])


def from_platform(spec, rho=1000.0, g=GRAVITY, surface_z=None, voltage=None,
                  drag_scale=1.0, mass=None, payloads=(), trim=True):
    """VehicleModel for a ``platforms.PlatformSpec`` with hydrodynamics, with
    ``payloads`` fitted. voltage overrides the T200 supply voltage; drag_scale
    multiplies both damping terms (for calibrating against your own vehicle);
    mass overrides the bare vehicle's in-air mass (its displaced volume, and so
    its buoyancy, is kept)."""
    h = resolve_hydro(spec, payloads, rho=rho, mass=mass, trim=trim)
    hydro = HydroModel(h["mass"], h["inertia"], h["added_mass"],
                       np.asarray(h["linear_damping"]) * drag_scale,
                       np.asarray(h["quadratic_damping"]) * drag_scale,
                       h["displaced_volume"], h["cob"], h["height"])
    thr = h["thrusters"]
    v = voltage if voltage is not None else h["battery_voltage"]
    fwd, rev = _thruster_limits(thr, h["thruster_model"], v)
    thrusters = ThrusterArray([t.position for t in thr], [t.direction for t in thr],
                              fwd, rev, h["thruster_time_constant"])
    return VehicleModel(hydro, thrusters, rho=rho, g=g, surface_z=surface_z, cog=h["cog"])
