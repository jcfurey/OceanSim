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

def test_both_platforms_registered(p):
    assert set(p.available_platforms()) == {"bluerov2", "deeptrekker_revolution"}


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
    """Regression lock: selecting bluerov2 must reproduce the values that were
    hardcoded before the registry (behaviour-preserving refactor)."""
    b = p.get_platform("bluerov2")
    assert b.usd_subpath == os.path.join("Bluerov", "BROV_low.usd")
    assert b.mass == 5.0
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
