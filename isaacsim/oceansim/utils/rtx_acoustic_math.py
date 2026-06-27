"""Pure (numpy-only) folding of acoustic GenericModelOutput samples into the
OceanSim ``(n_range, n_beams)`` intensity grid.

Extracted from ``RtxAcousticSensor.make_sonar_data`` so the time->range and
receiver->beam mapping (an experimental scaffold that still needs calibrating
against real sonar geometry) is unit testable without Isaac Sim, and so the
accumulation can be optimised against a characterisation test.
"""

import numpy as np


def fold_gmo_to_grid(rx, amp, t_ns, sound_speed, min_range, range_res,
                     n_range, n_beams, n_elements):
    """Fold per-sample acoustic returns into a normalised intensity grid.

    For each sample: range = sound_speed * t / 2 (t in ns) -> range bin; receiver
    mount id -> beam by a linear spread across the receiver fan. ``|amplitude|`` is
    accumulated per (range, beam) cell (bincount over flattened indices -- many
    samples share a cell), then the grid is normalised by its peak.

    Returns a float32 ``(n_range, n_beams)`` array in [0, 1] (all zeros if there
    are no samples in range).
    """
    rx = np.asarray(rx)
    amp = np.asarray(amp)
    t_ns = np.asarray(t_ns, dtype=np.float64)
    grid = np.zeros((n_range, n_beams), dtype=np.float32)
    if t_ns.size == 0:
        return grid

    rng = sound_speed * (t_ns * 1e-9) / 2.0
    # Map non-finite ranges (e.g. a missing time sample) to a finite, out-of-range
    # sentinel (rbin == -1) so the int cast below never sees inf/nan (an undefined
    # cast). These are dropped by the rbin >= 0 check, same as before.
    rng = np.where(np.isfinite(rng), rng, min_range - range_res)
    rbin = np.round((rng - min_range) / range_res).astype(np.int64)
    n_mounts = max(n_elements, int(rx.max()) + 1 if rx.size else n_elements)
    beam = np.round(rx.astype(np.float64) / max(n_mounts - 1, 1) * (n_beams - 1)).astype(np.int64)

    valid = ((rbin >= 0) & (rbin < n_range)
             & (beam >= 0) & (beam < n_beams))
    if np.any(valid):
        flat = rbin[valid] * n_beams + beam[valid]
        acc = np.bincount(flat, weights=np.abs(amp[valid]),
                          minlength=n_range * n_beams)
        grid[:] = acc.reshape(n_range, n_beams)
        peak = float(grid.max())
        if peak > 0.0:
            grid /= peak
    return grid
