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
        # Linear ramp between the bounds (continuous with the near branch at
        # min_range == near, where the second term is zero -> freq == hi_f).
        freq = hi_f - (hi_f - sound_speed / (2 * min_range)) / (far - near) * (min_range - near)
    else:
        freq = lo_f
    return 1.0 / freq
