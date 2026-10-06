"""Pure (numpy-only) DVL math, extracted from ``DVLsensor`` so the Janus
beam->velocity transform and the frequency-adaptive update-rate ramp can be
unit tested without Isaac Sim.

``DVLsensor`` imports Isaac (BaseSensor, the light-beam physics interface), so
these closed-form pieces used to be untestable. They live here now and the
sensor delegates to them -- the tested code IS the production code.
"""

import numpy as np


def beam_velocity_transform(elevation_deg):
    """Return the 3x4 matrix that maps the four Janus beam range-rate
    measurements to a body-frame velocity ``(vx, vy, vz)``.

    Depends only on the beam elevation angle (the four beams share it, in a
    symmetric Janus configuration). Raises ``ValueError`` for an elevation that
    is a multiple of 90 degrees, where ``sin``/``cos(elevation)`` is zero and the
    transform would divide by (near-)zero.

    Note the tolerance: ``np.cos(np.deg2rad(90))`` is ~6e-17, not exactly 0, so
    an ``== 0`` check would miss 90/180 deg and emit an absurd ~1e16 transform.
    """
    sin_elev = np.sin(np.deg2rad(elevation_deg))
    cos_elev = np.cos(np.deg2rad(elevation_deg))
    if abs(sin_elev) < 1e-9 or abs(cos_elev) < 1e-9:
        raise ValueError(
            f"DVL beam elevation must not be a multiple of 90 degrees (got "
            f"{elevation_deg}); the velocity transform divides by sin/cos(elevation).")
    return np.array([[1 / (2 * sin_elev), 0, -1 / (2 * sin_elev), 0],
                     [0, 1 / (2 * sin_elev), 0, -1 / (2 * sin_elev)],
                     [1 / (4 * cos_elev), 1 / (4 * cos_elev), 1 / (4 * cos_elev), 1 / (4 * cos_elev)]])


def adaptive_sensor_dt(min_range, freq_bound, range_bound, sound_speed):
    """Frequency-adaptive DVL update period (seconds) as a function of the
    closest beam range ``min_range``.

    - At/below the near range bound: the fixed maximum frequency.
    - At/above the far range bound: the fixed minimum frequency.
    - Between: a linear ramp from the max frequency down toward the
      sound-speed-limited frequency ``sound_speed / (2 * min_range)``.

    ``freq_bound`` is ``(min_freq, max_freq)`` Hz and ``range_bound`` is
    ``(near, far)`` metres. A non-finite ``min_range`` (e.g. every beam missed,
    so the depth list is all NaN) falls back to the minimum frequency (slowest
    safe rate) instead of producing a NaN period.
    """
    lo_f, hi_f = freq_bound
    near, far = range_bound
    if not np.isfinite(min_range):
        freq = lo_f
    elif min_range <= near:
        freq = hi_f
    elif near < min_range < far:
        # Linear ramp from the max frequency (at near) down to the sound-speed-
        # limited frequency at the FAR bound. The old formula interpolated toward
        # sound_speed/(2*min_range) -- a moving target at the current range --
        # which makes the frequency RISE with range (physically backwards: a
        # longer round trip must lower, not raise, the max ping rate) whenever
        # sound_speed/(2*near) > hi_f. Use a fixed far endpoint and clamp.
        far_freq = min(hi_f, sound_speed / (2.0 * far))
        freq = hi_f + (far_freq - hi_f) * (min_range - near) / (far - near)
        freq = min(max(freq, lo_f), hi_f)
    else:
        freq = lo_f
    return 1.0 / freq


def mount_point_velocity_body(v_com_world, omega_world, rot_world_from_body, lever_arm_body):
    """Velocity of the DVL mount point, expressed in the vehicle body frame.

    A DVL measures the velocity of the point it is mounted at, not of the
    vehicle's centre of mass:  v_mount = v_com + omega x r, with r the vector
    from the centre of mass to the mount. In the body frame that is

        R^T v_com + (R^T omega) x r_body

    ``v_com_world`` / ``omega_world`` are the rigid body's world-frame linear
    (centre-of-mass) and angular (rad/s) velocities, ``rot_world_from_body``
    the 3x3 body->world rotation, and ``lever_arm_body`` = mount position minus
    centre-of-mass position, both in the body frame.
    """
    rot = np.asarray(rot_world_from_body, dtype=float).reshape(3, 3)
    v_body = rot.T @ np.asarray(v_com_world, dtype=float).reshape(3)
    w_body = rot.T @ np.asarray(omega_world, dtype=float).reshape(3)
    return v_body + np.cross(w_body, np.asarray(lever_arm_body, dtype=float).reshape(3))


def velocity_covariance(transform, beam_sqrt_cov):
    """Body-frame velocity covariance T Sigma T^T for beam-space noise with
    covariance Sigma = L L^T (L = ``beam_sqrt_cov``, 4x4) pushed through the
    3x4 Janus transform T. With 22.5 deg beams an isotropic per-beam variance
    becomes ~3.4x larger in x/y and ~0.29x in z."""
    t = np.asarray(transform, dtype=float).reshape(3, 4)
    l = np.asarray(beam_sqrt_cov, dtype=float).reshape(4, 4)
    sigma = l @ l.T
    return t @ sigma @ t.T


def altitude_from_beam_ranges(ranges, elevation_deg):
    """Altitude above the bottom from the four slant beam ranges: the mean
    vertical component r_i * cos(beam angle from vertical) of the beams that
    hit. The minimum SLANT range (the old estimate) overstates altitude by
    1/cos(elevation) -- +8.2% for 22.5 deg beams over a flat bottom. Returns NaN
    when no beam has a finite range."""
    r = np.asarray(ranges, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float(np.mean(r) * np.cos(np.deg2rad(elevation_deg)))
