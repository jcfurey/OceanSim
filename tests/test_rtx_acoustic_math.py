"""Characterisation tests for the RTX acoustic GMO -> grid folding (pure numpy)."""

import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")

_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                     "utils", "rtx_acoustic_math.py")

SS = 1500.0      # sound speed
MN, RES = 0.1, 0.01
NR, NB, NE = 100, 8, 8


def _load():
    spec = importlib.util.spec_from_file_location("rtx_acoustic_math", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load()


def _t_ns(rng_m):
    """ns time-of-flight for a one-way range (range = ss * t/2)."""
    return 2.0 * rng_m / SS * 1e9


def _fold(m, rx, amp, t_ns):
    return m.fold_gmo_to_grid(np.asarray(rx), np.asarray(amp, float),
                              np.asarray(t_ns, float), SS, MN, RES, NR, NB, NE)


def test_empty(m):
    g = _fold(m, [], [], [])
    assert g.shape == (NR, NB) and g.dtype == np.float32 and not g.any()


def test_single_sample_maps_to_expected_cell(m):
    # range 0.6 m -> rbin = round((0.6-0.1)/0.01) = 50; rx 0 -> beam 0.
    g = _fold(m, [0], [2.0], [_t_ns(0.6)])
    assert g[50, 0] == pytest.approx(1.0)      # single sample normalises to 1
    assert g.sum() == pytest.approx(1.0)       # nothing else set
    # extreme receiver maps to the last beam
    g2 = _fold(m, [NE - 1], [1.0], [_t_ns(0.6)])
    assert g2[50, NB - 1] == pytest.approx(1.0)


def test_duplicate_cell_accumulates(m):
    g = _fold(m, [0, 0], [1.0, 3.0], [_t_ns(0.6), _t_ns(0.6)])
    assert g[50, 0] == pytest.approx(1.0)      # (1+3)=4 then /peak(4) -> 1
    assert g.sum() == pytest.approx(1.0)


def test_relative_normalisation(m):
    # cell A gets |amp| 2, cell B gets 4 -> A=0.5, B=1.0 after peak-normalise.
    g = _fold(m, [0, 1], [2.0, -4.0], [_t_ns(0.6), _t_ns(1.0)])
    b_a = int(round(0 / (NE - 1) * (NB - 1)))   # 0
    b_b = int(round(1 / (NE - 1) * (NB - 1)))   # round(1/7*7)=1
    assert g[50, b_a] == pytest.approx(0.5)
    assert g[int(round((1.0 - MN) / RES)), b_b] == pytest.approx(1.0)


def test_out_of_range_dropped(m):
    # range 5 m -> rbin 490 >= n_range(100); inf time; both dropped -> empty grid.
    g = _fold(m, [0, 0], [9.0, 9.0], [_t_ns(5.0), np.inf])
    assert not g.any()


def test_abs_amplitude(m):
    # negative amplitudes contribute their magnitude.
    g = _fold(m, [0], [-3.0], [_t_ns(0.6)])
    assert g[50, 0] == pytest.approx(1.0)
