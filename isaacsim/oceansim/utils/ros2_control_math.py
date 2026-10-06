"""Pure (numpy-only) math for the OceanSim ROS2 control receiver.

Deliberately free of ``rclpy`` / Isaac Sim / ``pxr`` imports so it is unit
testable without ROS or Isaac Sim, mirroring ``ros2_math.py``'s split for the
sensor publishers. ``ros2_control.py`` calls these helpers rather than
re-implementing the logic inline.
"""

import numpy as np


def clamp_magnitude(vec, max_mag):
    """Clamp a 3-vector's magnitude to max_mag, preserving direction.

    max_mag=None disables clamping -- a finite vec is returned unchanged (as a
    plain list of floats, matching the clamped-path return shape).

    A non-finite command (any NaN/inf component) is rejected to zeros in BOTH
    modes: with clamping on, ``norm=inf`` made ``max_mag/norm`` collapse the
    finite components and turn the infinite one into ``0*inf=NaN``, and a NaN
    component made ``norm > max_mag`` False so NaN passed through "unclamped"
    -- either way garbage reached the thrusters. Zero (the dead-man behavior)
    is the only safe output for an invalid command.
    """
    arr = np.asarray(vec, dtype=float)
    if not np.all(np.isfinite(arr)):
        return [0.0, 0.0, 0.0]
    if max_mag is None:
        return [float(arr[0]), float(arr[1]), float(arr[2])]
    norm = np.linalg.norm(arr)
    if norm > max_mag and norm > 0.0:
        arr = arr * (max_mag / norm)
    return arr.tolist()


def is_command_stale(now, last_command_time, timeout):
    """True if more than timeout seconds have elapsed since last_command_time.

    Dead-man's-switch check for the vel/force command receiver: a lost ROS2
    link (crashed teleop, dropped session, network partition) must not leave
    the vehicle actuating its last received command forever.
    """
    return (now - last_command_time) > timeout


def world_to_body(quat_wxyz, vec_world):
    """Rotate a world-frame 3-vector into the body frame of orientation
    ``quat_wxyz`` (scalar-first, as Isaac returns it): v_body = R^T v_world."""
    w, x, y, z = (float(c) for c in quat_wxyz)
    n = w * w + x * x + y * y + z * z
    if n <= 0.0:
        return np.asarray(vec_world, dtype=float)
    s = 2.0 / n
    rot = np.array([
        [1.0 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1.0 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1.0 - s * (x * x + y * y)],
    ])
    return rot.T @ np.asarray(vec_world, dtype=float)


class BodyVelocityPI:
    """Closed-loop body-frame velocity controller ("dynamic velocity control").

    Turns a cmd_vel (body-frame linear m/s, angular rad/s) into a body-frame
    acceleration demand:
        a = k_d * v_cmd + kp * e + clamp(sum(ki * e * dt), +-i_limit),
        e = v_cmd - v_meas.
    The caller scales it by mass / inertia into a wrench and applies it as a
    force, so buoyancy, drag and collisions keep acting -- unlike the kinematic
    "velocity control" mode, which overwrites the body velocity every step.

    k_d (1/s) is a damping feedforward: the platforms model water drag with
    PhysX linear / angular damping (dv/dt = -k_d v, k_d ~10-15 1/s), which a PI
    loop alone could only overcome with a huge integral (P-only settles at
    kp / (kp + k_d) of the command). With it the PI only corrects residual
    loads (net buoyancy, collisions); i_limit bounds the integral (anti-windup).
    Gains are mass-normalised (1/s, 1/s^2). With the feedforward the error
    dynamics are s^2 + (kp + k_d) s + ki, so ki=None (default) scales with
    them: (kp + k_d)^2 / 8, which keeps load rejection fast however heavy the
    damping is (k_d = 10-15: ~0.15 s rise, ~1 s to 2%, <= ~10% overshoot on a
    step; /4 rejects loads faster but overshoots ~16%).
    """

    def __init__(self, kp_lin=2.0, ki_lin=None, kp_ang=3.0, ki_ang=None,
                 i_limit_lin=2.0, i_limit_ang=2.0, damping_lin=0.0, damping_ang=0.0):
        if ki_lin is None:
            ki_lin = (kp_lin + damping_lin) ** 2 / 8.0
        if ki_ang is None:
            ki_ang = (kp_ang + damping_ang) ** 2 / 8.0
        self.kp = np.array([kp_lin] * 3 + [kp_ang] * 3, dtype=float)
        self.ki = np.array([ki_lin] * 3 + [ki_ang] * 3, dtype=float)
        self.i_limit = np.array([i_limit_lin] * 3 + [i_limit_ang] * 3, dtype=float)
        self.damping = np.array([damping_lin] * 3 + [damping_ang] * 3, dtype=float)
        self._integral = np.zeros(6)

    def reset(self):
        self._integral[:] = 0.0

    def update(self, cmd_lin, cmd_ang, meas_lin, meas_ang, dt):
        """Return (acc_lin, acc_ang) body-frame demands for one control step.

        A non-finite command or measurement yields zero demand and leaves the
        integral untouched; dt <= 0 skips integration."""
        cmd = np.concatenate([np.asarray(cmd_lin, float), np.asarray(cmd_ang, float)])
        meas = np.concatenate([np.asarray(meas_lin, float), np.asarray(meas_ang, float)])
        if not (np.all(np.isfinite(cmd)) and np.all(np.isfinite(meas))):
            return np.zeros(3), np.zeros(3)
        err = cmd - meas
        if dt is not None and dt > 0.0:
            self._integral = np.clip(self._integral + self.ki * err * float(dt),
                                     -self.i_limit, self.i_limit)
        acc = self.damping * cmd + self.kp * err + self._integral
        return acc[:3], acc[3:]


def body_wrench(acc_lin, acc_ang, mass, inertia_diag):
    """Body-frame (force N, torque N*m) for an acceleration demand: F = m a,
    tau = I alpha with the principal-axis inertia diagonal (gyroscopic terms
    neglected -- small at ROV angular rates)."""
    force = float(mass) * np.asarray(acc_lin, dtype=float)
    torque = np.asarray(inertia_diag, dtype=float) * np.asarray(acc_ang, dtype=float)
    return force, torque
