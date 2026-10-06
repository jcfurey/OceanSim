"""Unit tests for the vehicle platform registry + robot-description resolver.

Pure stdlib -- no Isaac Sim. Loaded by file path to avoid the isaacsim.oceansim
namespace package.
"""

import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                     "utils", "platforms.py")


def _load():
    spec = importlib.util.spec_from_file_location("platforms", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def p():
    return _load()


# --- registry --------------------------------------------------------------

ALL_PLATFORMS = ["bluerov2", "bluerov2_heavy", "deeptrekker_revolution", "deeptrekker_dtg3",
                 "deeptrekker_pivot", "videoray_pro5", "videoray_defender", "chasing_m2_pro_max",
                 "qysea_fifish_v6_expert", "saab_seaeye_falcon", "seabotix_vlbv300"]


def test_platforms_registered(p):
    assert set(p.available_platforms()) == set(ALL_PLATFORMS)
    assert p.get_platform("bluerov2heavy").name == "bluerov2_heavy"
    assert p.get_platform("Falcon").name == "saab_seaeye_falcon"
    assert p.get_platform("fifish").name == "qysea_fifish_v6_expert"


def test_lookup_is_case_and_alias_insensitive(p):
    assert p.get_platform("bluerov2").name == "bluerov2"
    assert p.get_platform("BlueROV2").name == "bluerov2"
    assert p.get_platform("bluerov").name == "bluerov2"          # alias
    assert p.get_platform("Revolution").name == "deeptrekker_revolution"
    assert p.get_platform("deep-trekker-revolution").name == "deeptrekker_revolution"


def test_unknown_platform_raises_with_options(p):
    with pytest.raises(KeyError) as e:
        p.get_platform("submarine")
    assert "deeptrekker_revolution" in str(e.value)  # lists what IS available


def test_bluerov2_values_locked(p):
    """Regression lock: selecting bluerov2 reproduces the values that were
    hardcoded before the registry, except the in-air mass, which is now the
    real 11.5 kg (it was 5.0) for the hydrodynamic model."""
    b = p.get_platform("bluerov2")
    assert b.usd_subpath == os.path.join("Bluerov", "BROV_low.usd")
    assert b.mass == 11.5
    assert b.linear_damping == 10.0 and b.angular_damping == 10.0
    assert b.collision_approximation == "boundingCube"
    assert b.spawn_translation == (-2.0, 0.0, -0.8)
    assert b.sonar_mount.translation == (0.3, 0.0, 0.3)
    assert b.sonar_mount.rpy_deg == (0.0, 45.0, 0.0)
    assert b.camera_mount.translation == (0.3, 0.0, 0.1)
    assert b.dvl_mount.translation == (0.0, 0.0, -0.1)


def test_deeptrekker_real_specs(p):
    d = p.get_platform("deeptrekker_revolution")
    assert d.mass == 26.0                       # real in-air mass (spec sheet)
    assert d.collision_approximation == "boundingCube"
    assert d.usd_subpath == os.path.join("DeepTrekker", "revolution.usd")


def test_usd_and_urdf_paths_join_under_root(p):
    b = p.get_platform("bluerov2")
    assert b.usd_path("/assets") == os.path.join("/assets", "Bluerov", "BROV_low.usd")
    assert b.urdf_path("/assets") == os.path.join("/assets", "Bluerov", "bluerov2.urdf")


def test_urdf_path_none_when_unset(p):
    # A spec with no urdf_subpath returns None rather than joining a bad path.
    spec = p.PlatformSpec(
        name="x", usd_subpath="x.usd", mass=1.0, linear_damping=1.0,
        angular_damping=1.0, collision_approximation="boundingCube",
        spawn_translation=(0, 0, 0),
        sonar_mount=p.SensorMount((0, 0, 0)),
        camera_mount=p.SensorMount((0, 0, 0)),
        dvl_mount=p.SensorMount((0, 0, 0)))
    assert spec.urdf_path("/assets") is None


# --- resolve_robot_description ---------------------------------------------

def test_resolve_inline_wins(p):
    text, src = p.resolve_robot_description(inline="<robot/>", path="/nope.urdf",
                                            platform="bluerov2", asset_root="/assets")
    assert text == "<robot/>" and src == "inline"


def test_resolve_explicit_path(p, tmp_path):
    f = tmp_path / "my.urdf"
    f.write_text("<robot name='custom'/>")
    text, src = p.resolve_robot_description(path=str(f))
    assert "custom" in text and src == str(f)


def test_resolve_from_platform_asset(p, tmp_path):
    # Lay out <root>/Bluerov/bluerov2.urdf and resolve via the platform default.
    urdf = tmp_path / "Bluerov" / "bluerov2.urdf"
    urdf.parent.mkdir(parents=True)
    urdf.write_text("<robot name='bluerov2'/>")
    text, src = p.resolve_robot_description(asset_root=str(tmp_path), platform="bluerov2")
    assert "bluerov2" in text and src == str(urdf)


def test_resolve_missing_file_reports_reason(p, tmp_path):
    text, src = p.resolve_robot_description(asset_root=str(tmp_path), platform="bluerov2")
    assert text is None and src.startswith("missing:")


def test_resolve_nothing_available(p):
    text, src = p.resolve_robot_description()
    assert text is None and src == "none"


# --- resolve_robot_source (USD vs URDF import) -----------------------------

def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return str(path)


def test_source_prefers_existing_usd(p, tmp_path):
    usd = _touch(tmp_path / "Bluerov" / "BROV_low.usd")
    src, why = p.resolve_robot_source(asset_root=str(tmp_path), platform="bluerov2")
    assert why == "ok" and src.kind == "usd" and src.path == usd


def test_source_falls_back_to_urdf_when_no_usd(p, tmp_path):
    # only a URDF on disk -> "I only have a URDF" resolves to the URDF import.
    urdf = _touch(tmp_path / "Bluerov" / "bluerov2.urdf")
    src, why = p.resolve_robot_source(asset_root=str(tmp_path), platform="bluerov2")
    assert why == "ok" and src.kind == "urdf" and src.path == urdf


def test_source_prefer_urdf_when_both_exist(p, tmp_path):
    _touch(tmp_path / "Bluerov" / "BROV_low.usd")
    urdf = _touch(tmp_path / "Bluerov" / "bluerov2.urdf")
    src, why = p.resolve_robot_source(asset_root=str(tmp_path), platform="bluerov2",
                                      prefer="urdf")
    assert src.kind == "urdf" and src.path == urdf


def test_source_explicit_urdf_override_wins(p, tmp_path):
    _touch(tmp_path / "Bluerov" / "BROV_low.usd")          # platform usd exists
    custom = _touch(tmp_path / "custom.urdf")
    src, why = p.resolve_robot_source(asset_root=str(tmp_path), platform="bluerov2",
                                      urdf_path=custom)
    assert src.kind == "urdf" and src.path == custom


def test_source_missing_reports_paths(p, tmp_path):
    src, why = p.resolve_robot_source(asset_root=str(tmp_path), platform="bluerov2")
    assert src is None and why.startswith("missing:")
    assert "BROV_low.usd" in why and "bluerov2.urdf" in why


def test_source_nothing_configured(p):
    src, why = p.resolve_robot_source()
    assert src is None and why == "none"


# --- hydrodynamic models ----------------------------------------------------------

_VD_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                        "utils", "vehicle_dynamics.py")


@pytest.fixture(scope="module")
def vd():
    pytest.importorskip("numpy")
    spec = importlib.util.spec_from_file_location("vehicle_dynamics", _VD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


KNOT = 0.514444


def _top_speed(vd, plat, axis, sign=1.0):
    """Steady speed along body ``axis`` (0-5) at full thrust in that direction."""
    import numpy as np
    model = vd.from_platform(plat)
    d = np.zeros(6)
    d[axis] = sign
    f = float(model.max_wrench(d)[axis]) * sign
    q = model.hydro.quadratic_damping[axis]
    lin = model.hydro.linear_damping[axis]
    return (-lin + (lin * lin + 4 * q * f) ** 0.5) / (2 * q)


@pytest.mark.parametrize("name", ALL_PLATFORMS)
def test_hydro_models_are_physical(p, vd, name):
    import numpy as np
    plat = p.get_platform(name)
    h = vd.resolve_hydro(plat)
    assert h["sources"]
    model = vd.from_platform(plat)
    # near neutral: |net buoyancy| within 2% of weight in fresh water
    net = 1000.0 * h["displaced_volume"] - plat.mass
    assert abs(net) < 0.02 * plat.mass
    assert h["cob"][2] > 0.0                               # statically stable
    assert all(v > 0 for v in h["inertia"])
    assert all(v >= 0 for v in h["added_mass"] + h["linear_damping"] + h["quadratic_damping"])
    assert np.all(model.thrusters.max_forward > 0) and np.all(model.thrusters.max_reverse > 0)
    assert plat.dimensions and plat.manufacturer
    assert _top_speed(vd, plat, 0) > 0.3                   # every vehicle can surge
    # controllable axes = what the thruster layout can produce on its own
    B = model.thrusters.B
    controllable = {i for i in range(6)
                    if np.allclose(B @ np.linalg.pinv(B) @ np.eye(6)[i], np.eye(6)[i], atol=1e-6)}
    assert controllable == CONTROLLABLE[name]


def test_bluerov2_speeds_bracketed_by_measurement_and_claim(p, vd):
    """0.72 m/s measured by von Benzon et al. at lower thrust (tethered),
    1.5 m/s quoted by Blue Robotics: the model sits between."""
    u = _top_speed(vd, p.get_platform("bluerov2"), 0)
    assert 0.72 < u < 1.5
    assert _top_speed(vd, p.get_platform("bluerov2_heavy"), 2) > _top_speed(
        vd, p.get_platform("bluerov2"), 2)          # 4 vertical thrusters vs 2


def test_revolution_matches_listed_speeds(p, vd):
    rev = p.get_platform("deeptrekker_revolution")
    assert _top_speed(vd, rev, 0) == pytest.approx(3 * KNOT, rel=0.01)
    assert _top_speed(vd, rev, 0, -1.0) == pytest.approx(3 * KNOT, rel=0.01)
    assert _top_speed(vd, rev, 1) == pytest.approx(2 * KNOT, rel=0.01)
    assert _top_speed(vd, rev, 2) == pytest.approx(3 * KNOT, rel=0.01)
    import numpy as np
    assert np.abs(vd.from_platform(rev).max_wrench(np.eye(6)[0])[0]) == pytest.approx(12 * 9.81, rel=1e-3)


def test_revolution_estimate_reproduces_documented_values(p, vd):
    """The HydroEstimate path must give the values derived by hand when the
    Revolution model was first written (regression lock on the method)."""
    h = vd.resolve_hydro(p.get_platform("deeptrekker_revolution"))
    assert h["displaced_volume"] == pytest.approx(0.02613)
    assert list(h["inertia"]) == pytest.approx([0.3235, 0.7401, 0.92], rel=1e-3)
    assert list(h["added_mass"]) == pytest.approx(
        [4.575, 10.354, 34.246, 0.1045, 0.5211, 0.0804], rel=2e-3)
    assert list(h["quadratic_damping"]) == pytest.approx(
        [46.42, 101.35, 46.42, 0.1236, 0.5347, 1.1675], rel=2e-3)
    assert list(h["linear_damping"]) == pytest.approx(
        [4.642, 10.135, 4.642, 0.01236, 0.05347, 0.11675], rel=2e-3)


def test_thruster_voltage_and_drag_overrides(p, vd):
    plat = p.get_platform("bluerov2")
    hi = vd.from_platform(plat, voltage=20.0)
    assert hi.thrusters.max_forward[0] == pytest.approx(6.7 * 9.81)
    slow = vd.from_platform(plat, drag_scale=2.0)
    assert slow.hydro.quadratic_damping[0] == pytest.approx(282.0)


def test_strip_theory_rotational_drag_checked_on_bluerov2(p):
    """The Revolution's rotational drag comes from strip theory. Applied to the
    BlueROV2 Heavy (0.46 x 0.58 x 0.38 m, Table 5 of von Benzon et al.) with its
    measured translational drag coefficients, the same formulas land within a
    factor of two of its measured rotational coefficients."""
    rho, length, width, height = 1000.0, 0.46, 0.58, 0.38
    q = p.get_platform("bluerov2_heavy").hydro.quadratic_damping   # measured HydroSpec
    cd_y = 2 * q[1] / (rho * 0.1131)          # A_v
    cd_z = 2 * q[2] / (rho * 0.2049)          # A_w
    strip = (rho * cd_z * length * width ** 4 / 64,
             rho * cd_z * width * length ** 4 / 64,
             rho * cd_y * height * length ** 4 / 64)
    for est, measured in zip(strip, q[3:]):
        assert 0.5 < est / measured < 2.0


ALL6 = {0, 1, 2, 3, 4, 5}
CONTROLLABLE = {                              # surge sway heave roll pitch yaw
    "bluerov2": {0, 1, 2, 3, 5},              # no pitch (6-thruster frame)
    "bluerov2_heavy": ALL6,
    "deeptrekker_revolution": {0, 1, 2, 3, 5},
    "deeptrekker_dtg3": {0, 2, 5},            # 2 forward + 1 vertical
    "deeptrekker_pivot": {0, 1, 2, 3, 5},
    "videoray_pro5": {0, 2, 5},
    "videoray_defender": ALL6,                # 4 vectored + 3 vertical
    "chasing_m2_pro_max": ALL6,               # 4 vectored + 4 vertical
    "qysea_fifish_v6_expert": {0, 1, 2, 3, 5},
    "saab_seaeye_falcon": {0, 1, 2, 5},       # single vertical thruster
    "seabotix_vlbv300": {0, 1, 2, 3, 5},
}

# Published figures each estimated model must reproduce (kgf thrust per axis,
# top speeds in kn or m/s), from the sources cited in platforms.py.
PUBLISHED = {
    "deeptrekker_dtg3": dict(surge_kgf=2.5, heave_kgf=2.5, surge_kn=2.5, heave_kn=2.5),
    "deeptrekker_pivot": dict(surge_kn=2.0, heave_kn=1.0),
    "videoray_pro5": dict(surge_kgf=20.3, reverse_kgf=13.0, surge_kn=4.4, heave_ms=0.8),
    "videoray_defender": dict(surge_kgf=23.6, reverse_kgf=15.0, sway_kgf=8.6, heave_kgf=23.1,
                              down_kgf=12.9, surge_kn=3.8, sway_kn=0.9, heave_ms=0.8),
    "chasing_m2_pro_max": dict(surge_kgf=5.7, sway_kgf=3.6, heave_kgf=4.0, surge_ms=1.5),
    "qysea_fifish_v6_expert": dict(surge_ms=1.5),
    "saab_seaeye_falcon": dict(surge_kgf=42.0, sway_kgf=25.0, heave_kgf=13.0, surge_kn=3.0),
    "seabotix_vlbv300": dict(surge_kgf=18.1, sway_kgf=15.2, heave_kgf=9.0, surge_kn=3.0),
}
_AXIS = {"surge": (0, 1.0), "reverse": (0, -1.0), "sway": (1, 1.0), "heave": (2, 1.0),
         "down": (2, -1.0)}


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_estimated_vehicles_reproduce_published_figures(p, vd, name):
    import numpy as np
    plat = p.get_platform(name)
    model = vd.from_platform(plat)
    for key, value in PUBLISHED[name].items():
        what, unit = key.rsplit("_", 1)
        axis, sign = _AXIS[what]
        if unit == "kgf":
            d = np.zeros(6)
            d[axis] = sign
            assert abs(model.max_wrench(d)[axis]) == pytest.approx(value * 9.80665, rel=2e-3), key
        else:
            want = value * KNOT if unit == "kn" else value
            assert _top_speed(vd, plat, axis, sign) == pytest.approx(want, rel=0.01), key


def test_vector_angle_from_thrust(p):
    assert p.vector_angle_from_thrust(10.0, 10.0) == pytest.approx(45.0)
    # sway is bounded by the weaker (reverse) direction
    assert p.vector_angle_from_thrust(23.6, 8.6, 15.0) == pytest.approx(29.83, abs=0.01)
