"""Characterisation tests for the RTX acoustic GMO -> grid folding (pure numpy).

The acoustic GMO is signal-way A-scans: ``amp`` is ``numSgws * num_samples_per_sgw``
amplitude samples laid out as numSgws contiguous A-scan blocks. Sample ``k`` maps
to range ``range_offset + k * meters_per_sample``; each signal way's index maps to
an azimuth beam.
"""

import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")

_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                     "utils", "rtx_acoustic_math.py")

# Clean constants: range_offset 0, min_range 0, mps == range_res == 0.01 so a
# sample at index k maps exactly to range bin k.
NSPG = 50
MPS, R0 = 0.01, 0.0
MN, RES = 0.0, 0.01
NR, NB = 100, 8


def _load():
    spec = importlib.util.spec_from_file_location("rtx_acoustic_math", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load()


def _fold(m, amp, nspg=NSPG, mps=MPS, r0=R0):
    return m.fold_gmo_to_grid(np.asarray(amp, float), nspg, mps, r0, MN, RES, NR, NB)


def _ascan(peak_k, val=1.0, nspg=NSPG):
    a = np.zeros(nspg)
    a[peak_k] = val
    return a


def test_empty(m):
    g = _fold(m, [])
    assert g.shape == (NR, NB) and g.dtype == np.float32 and not g.any()


def test_zero_nspg_returns_empty(m):
    g = _fold(m, np.ones(NSPG), nspg=0)
    assert not g.any()


def test_single_sgw_peak_maps_to_sample_bin(m):
    # one signal way, peak at sample 30 -> range bin 30; sole sgw -> beam 0.
    g = _fold(m, _ascan(30, 2.0))
    assert g[30, 0] == pytest.approx(1.0)      # single sample normalises to 1
    assert g.sum() == pytest.approx(1.0)       # nothing else set


def test_two_sgw_map_to_extreme_beams(m):
    # 2 signal ways -> beams 0 and NB-1; peaks at samples 10 and 20.
    amp = np.concatenate([_ascan(10, 1.0), _ascan(20, 1.0)])
    g = _fold(m, amp)
    assert g[10, 0] == pytest.approx(1.0)
    assert g[20, NB - 1] == pytest.approx(1.0)
    assert g.sum() == pytest.approx(2.0)


def test_range_offset_shifts_bins(m):
    # range_offset 0.5 with res 0.01 -> sample 0 lands at bin 50.
    g = _fold(m, _ascan(0, 1.0), r0=0.5)
    assert g[50, 0] == pytest.approx(1.0)
    assert g.sum() == pytest.approx(1.0)


def test_out_of_range_samples_dropped(m):
    # nspg 150 > n_range 100: samples k>=100 fall outside the grid and drop.
    a = np.zeros(150)
    a[120] = 9.0          # range bin 120 >= NR(100) -> dropped
    a[40] = 3.0           # in range
    g = _fold(m, a, nspg=150)
    assert g[40, 0] == pytest.approx(1.0)
    assert g.sum() == pytest.approx(1.0)   # the out-of-range peak contributed nothing


def test_relative_normalisation(m):
    # within one A-scan, two samples |amp| 2 and 4 -> 0.5 and 1.0 after peak-norm.
    a = np.zeros(NSPG)
    a[10], a[20] = 2.0, -4.0
    g = _fold(m, a)
    assert g[10, 0] == pytest.approx(0.5)
    assert g[20, 0] == pytest.approx(1.0)


def test_abs_amplitude(m):
    g = _fold(m, _ascan(30, -3.0))
    assert g[30, 0] == pytest.approx(1.0)


def test_degenerate_grid_returns_empty_not_crash(m):
    # n_beams == 0 used to raise ValueError from grid.max() on a zero-size array.
    g = m.fold_gmo_to_grid(np.ones(NSPG), NSPG, MPS, R0, MN, RES, NR, 0)
    assert g.shape == (NR, 0) and not g.size
    g = m.fold_gmo_to_grid(np.ones(NSPG), NSPG, MPS, R0, MN, RES, 0, NB)
    assert g.shape == (0, NB) and not g.size


def test_floor_binning_not_round(m):
    # A return inside bin i's span [i*res, (i+1)*res) must stay in bin i even in
    # its upper half: sample at range 0.306 (res 0.01) -> bin 30 with floor
    # binning (round() put it in 31, a systematic +res/2 bias vs the published
    # (i+0.5)*res bin centres).
    g = _fold(m, _ascan(1, 1.0), mps=0.306, r0=0.0)   # sample 1 -> range 0.306
    assert g[30, 0] == pytest.approx(1.0)
    assert g.sum() == pytest.approx(1.0)


def test_fractional_sample_pitch(m):
    # Production-like: mps (~0.0176) is NOT an integer multiple of range_res
    # (0.005) -- the real rtx_acoustic geometry. Every sample must land in
    # floor((r0 + k*mps - min)/res) and nothing may drop or shift.
    mps, res, r0 = 0.0176, 0.005, 0.42875
    nspg = 40
    a = np.ones(nspg)
    g = m.fold_gmo_to_grid(a, nspg, mps, r0, 0.0, res, 400, 4)
    ks = np.arange(nspg)
    expected_bins = np.floor((r0 + ks * mps) / res + 1e-6).astype(int)
    lit = np.nonzero(g[:, 0])[0]
    assert set(lit.tolist()) == set(expected_bins.tolist())
    # all 40 unit samples survive (some bins take 2 samples at this pitch), and
    # the per-bin accumulation matches an independent histogram of the mapping.
    counts = np.bincount(expected_bins, minlength=400).astype(np.float64)
    assert np.allclose(g[:, 0], counts / counts.max())


def test_boundary_sample_lands_in_upper_bin(m):
    # A range EXACTLY at a bin boundary belongs to the upper bin: 0.5/0.01
    # evaluates to 49.999... in floats; the epsilon must land it in bin 50.
    g = _fold(m, _ascan(0, 1.0), r0=0.5)
    assert g[50, 0] == pytest.approx(1.0)


def test_many_sgw_share_beams_accumulate(m):
    # More signal ways than beams: sgws mapping to the same beam ACCUMULATE
    # (the vectorised scatter must keep duplicate-cell accumulation). 16 sgws
    # spread over 4 beams; beams 0 and 3 each take 5 endpoints-inclusive sgws?
    # Compute the expected per-beam counts independently instead of guessing.
    n_sgw, nspg, nb = 16, 10, 4
    amp = np.tile(_ascan(3, 1.0, nspg=nspg), n_sgw)
    g = m.fold_gmo_to_grid(amp, nspg, MPS, R0, MN, RES, NR, nb)
    beams = np.round(np.arange(n_sgw) / (n_sgw - 1) * (nb - 1)).astype(int)
    expected = np.bincount(beams, minlength=nb).astype(np.float64)
    assert np.allclose(g[3, :], expected / expected.max())
    # only range bin 3 is lit
    assert g.sum() == pytest.approx(g[3, :].sum())
