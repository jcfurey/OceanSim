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

import numpy as np


def fold_gmo_to_grid(amp, num_samples_per_sgw, meters_per_sample, range_offset,
                     min_range, range_res, n_range, n_beams):
    """Fold acoustic signal-way A-scans into a normalised ``(n_range, n_beams)`` grid.

    ``amp`` is the flat GMO ``scalar`` buffer (``numSgws * num_samples_per_sgw``
    amplitude samples). It is reshaped to ``(numSgws, num_samples_per_sgw)``; each
    row is one signal way's A-scan. Sample ``k`` maps to range
    ``range_offset + k * meters_per_sample`` -> a range bin. Each signal way's
    INDEX maps linearly to an azimuth beam (the GMO carries no per-sample azimuth;
    true delay-and-sum beamforming across the receiver array is a separate task).
    ``|amplitude|`` is accumulated per (range_bin, beam) then peak-normalised.

    Returns a float32 ``(n_range, n_beams)`` array in [0, 1] (all zeros if there
    are no samples in range).
    """
    amp = np.abs(np.asarray(amp, dtype=np.float64))
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

    # sample index -> range bin (shared across all signal ways). FLOOR binning
    # (bin i covers [min + i*res, min + (i+1)*res)) so the published bin CENTRES
    # (i + 0.5)*res (ros2_math.sonar_ranges, the Oculus convention) are zero-mean
    # about the true range -- round() binning put every centre a systematic
    # +res/2 above the return. The epsilon (1e-6 of a bin, physically
    # nanometres) absorbs float error at exact bin boundaries (0.5/0.01 ->
    # 49.999... must land in bin 50, not 49).
    k = np.arange(nspg)
    rng = range_offset + k * meters_per_sample
    rbin = np.floor((rng - min_range) / range_res + 1e-6).astype(np.int64)
    kv = (rbin >= 0) & (rbin < n_range)
    if not np.any(kv):
        return grid
    rbin_v = rbin[kv]

    # signal-way index -> azimuth beam (linear spread across the fan)
    beams = np.round(np.arange(n_sgw) / max(n_sgw - 1, 1) * (n_beams - 1)).astype(np.int64)
    bv = (beams >= 0) & (beams < n_beams)
    if not np.any(bv):
        return grid

    # Single vectorised scatter-add over flattened (range_bin, beam) indices --
    # replaces the per-signal-way Python loop of np.add.at (an unbuffered ufunc,
    # ~10x slower) on this per-frame path. Duplicate (bin, beam) cells (several
    # samples per bin, several signal ways per beam) accumulate, same as before.
    flat = (rbin_v[None, :] * n_beams + beams[bv][:, None]).ravel()
    acc = np.bincount(flat, weights=a2[bv][:, kv].ravel(),
                      minlength=n_range * n_beams)
    grid[:] = acc.reshape(n_range, n_beams)

    peak = float(grid.max())
    if peak > 0.0:
        grid /= peak
    return grid
