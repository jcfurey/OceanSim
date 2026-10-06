"""Pure (numpy-only) folding of acoustic GenericModelOutput samples into the
OceanSim ``(n_range, n_beams)`` intensity grid.

Extracted from ``RtxAcousticSensor.make_sonar_data`` so the sample->range and
signal-way->beam mapping is unit testable without Isaac Sim.

GMO layout (measured against Isaac 6.0.1, ``aux_output_level="BASIC"``): the
acoustic GMO is organised as **signal ways**, NOT a per-sample point cloud --
``numElements = numSgws * numSamplesPerSgw``, laid out as ``numSgws`` *contiguous
row-major blocks*. Each block is one signal way's amplitude envelope vs SAMPLE
INDEX (an A-scan). Range lives in the sample index; the per-element
``timeOffsetNs`` field is ALWAYS 0 for acoustic (the original ``range =
sound_speed * timeOffsetNs / 2`` model collapsed every sample to range 0). Sample
``k`` maps to range ``range_offset + k * meters_per_sample`` where
``meters_per_sample = c_sensor * sampleDuration / 2`` (the sensor models air,
c~=343 m/s; ``sampleDuration`` is a readable prim attribute).
"""

from functools import lru_cache

import numpy as np


def _bin_mean_linear(x, y, edges):
    """Mean of the piecewise-linear interpolant of ``y`` over each bin.

    ``x`` (K,) increasing knot positions shared by every row of ``y`` (M, K);
    ``edges`` (B+1,) increasing bin edges. Each bin's value is the integral of
    the interpolant over the part of the bin inside ``[x[0], x[-1]]``, divided
    by that covered width (0 where a bin has no coverage). Sparse knots (wider
    than a bin) are therefore linearly interpolated, dense ones averaged -- no
    empty bins between samples, and no banding from bins catching an unequal
    number of samples. Knots with no extent (one knot, or all at the same
    position) are point-binned: their mean goes to the bin holding them.
    """
    x = np.asarray(x, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    n_bins = edges.size - 1
    out = np.zeros((y.shape[0], n_bins), dtype=np.float64)
    if x[-1] <= x[0]:
        b = int(np.searchsorted(edges, x[0], side="right")) - 1
        if 0 <= b < n_bins:
            out[:, b] = y.mean(axis=1)
        return out

    # Cumulative integral at the knots (trapezoids are exact for a linear
    # interpolant), then evaluated at the clipped edges.
    c = np.zeros_like(y)
    np.cumsum(0.5 * (y[:, 1:] + y[:, :-1]) * np.diff(x), axis=1, out=c[:, 1:])
    e = np.clip(edges, x[0], x[-1])
    k = np.clip(np.searchsorted(x, e, side="right") - 1, 0, x.size - 2)
    t = e - x[k]
    y0, y1 = y[:, k], y[:, k + 1]
    ye = y0 + (y1 - y0) * (t / (x[k + 1] - x[k]))
    integral = c[:, k] + t * 0.5 * (y0 + ye)

    width = np.diff(e)
    # Skip slivers (float noise at a clipped edge): the area / width
    # cancellation would amplify rounding error there.
    covered = width > 1e-9 * np.max(np.diff(edges))
    out[:, covered] = np.diff(integral, axis=1)[:, covered] / width[covered]
    return out


@lru_cache(maxsize=8)
def _linear_bin_weights(x0, dx, n_knots, e0, de, n_bins):
    """(n_knots, n_bins) float32 matrix W with ``y @ W == _bin_mean_linear(x, y,
    edges)`` for the uniform knots ``x0 + k*dx`` and edges ``e0 + i*de`` (the
    resampling is linear in y). Cached: the sensor geometry is fixed, so each
    frame is two matmuls instead of rebuilding the interpolation."""
    x = x0 + np.arange(n_knots) * dx
    edges = e0 + np.arange(n_bins + 1) * de
    w = _bin_mean_linear(x, np.eye(n_knots), edges).astype(np.float32)
    w.setflags(write=False)
    return w


def fold_gmo_to_grid(amp, num_samples_per_sgw, meters_per_sample, range_offset,
                     min_range, range_res, n_range, n_beams):
    """Fold acoustic signal-way A-scans into a normalised ``(n_range, n_beams)`` grid.

    ``amp`` is the flat GMO ``scalar`` buffer (``numSgws * num_samples_per_sgw``
    amplitude samples). It is reshaped to ``(numSgws, num_samples_per_sgw)``; each
    row is one signal way's A-scan. Sample ``k`` is at range
    ``range_offset + k * meters_per_sample``; range bin ``i`` spans
    ``[min_range + i*range_res, min_range + (i+1)*range_res)`` (centres at
    ``(i + 0.5)*range_res``, the Oculus convention of ros2_math.sonar_ranges).
    Signal way ``s`` sits at beam position ``s / (numSgws-1) * (n_beams-1)`` (the
    GMO carries no per-sample azimuth; true delay-and-sum beamforming across the
    receiver array is a separate task), beam ``j`` spanning ``[j-0.5, j+0.5]``.

    ``|amplitude|`` is resampled onto the grid along both axes as the bin mean of
    its linear interpolant (``_bin_mean_linear``), then peak-normalised. The old
    point-scatter dropped each sample into one bin: at the runner defaults
    (17.6 mm samples, 5 mm bins, 63 signal ways over 520 beams) that lit ~28% of
    the range rows and ~12% of the beam columns.

    Returns a float32 ``(n_range, n_beams)`` array in [0, 1] (all zeros if there
    are no samples in range).
    """
    amp = np.abs(np.asarray(amp, dtype=np.float32))
    n_range = int(n_range)
    n_beams = int(n_beams)
    grid = np.zeros((max(n_range, 0), max(n_beams, 0)), dtype=np.float32)
    nspg = int(num_samples_per_sgw)
    # grid.size == 0 also guards the peak-normalise below: grid.max() on a
    # zero-size array raises ValueError instead of returning the empty grid.
    if amp.size == 0 or nspg <= 0 or amp.size < nspg or grid.size == 0:
        return grid

    n_sgw = amp.size // nspg
    a2 = amp[:n_sgw * nspg].reshape(n_sgw, nspg)

    # Range: (n_sgw, nspg) samples -> (n_sgw, n_range) bins.
    w_range = _linear_bin_weights(float(range_offset), float(meters_per_sample), nspg,
                                  float(min_range), float(range_res), n_range)
    # Azimuth: (n_range, n_sgw) -> (n_range, n_beams). Signal ways keep the
    # linear index -> beam spread (first / last on the edge beams' centres).
    w_beam = _linear_bin_weights(0.0, (n_beams - 1) / max(n_sgw - 1, 1), n_sgw,
                                 -0.5, 1.0, n_beams)
    np.matmul((a2 @ w_range).T, w_beam, out=grid)
    # Resampling only averages non-negative values; clip float32 rounding.
    np.maximum(grid, 0.0, out=grid)

    peak = float(grid.max())
    if peak > 0.0:
        grid /= peak
    return grid
