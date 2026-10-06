"""Payload catalogue, payload effects on the vehicle model, and URDF
export / import (pure Python + numpy; no Isaac Sim)."""

import importlib.util
import math
import os
import sys
import xml.etree.ElementTree as ET

import pytest

np = pytest.importorskip("numpy")

_UTILS = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim", "utils")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_UTILS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mods():
    # urdf_export / urdf_platform load their siblings by file path; load the
    # same module objects here so isinstance / dataclass types agree.
    platforms = _load("platforms")
    payloads = _load("payloads")
    vd = _load("vehicle_dynamics")
    exp = _load("urdf_export")
    imp = _load("urdf_platform")
    imp.platforms = platforms
    return platforms, payloads, vd, exp, imp


SONAR_KEYS = {"hori_fov_deg", "vert_fov_deg", "n_beams", "frequency_hz", "min_range",
              "max_range", "working_range", "range_res"}
DVL_KEYS = {"beam_angle_deg", "min_range", "max_range", "rate_hz", "velocity_noise"}


# --- catalogue ----------------------------------------------------------------

def test_catalogue_entries_are_complete(mods):
    _, payloads, *_ = mods
    assert len(payloads.PAYLOADS) >= 10
    for name, p in payloads.PAYLOADS.items():
        assert p.sources and p.manufacturer and p.description
        assert p.mass > 0 and p.volume >= 0 and len(p.size) == 3
        assert p.weight_in_water >= -1e-12        # none of these float on their own
        if p.kind == "sonar" and p.simulated:
            assert SONAR_KEYS <= set(p.params), name
            assert p.params["min_range"] < p.params["working_range"] <= p.params["max_range"]
        if p.kind == "dvl":
            assert DVL_KEYS <= set(p.params), name


def test_select_payloads_rules(mods):
    platforms, payloads, *_ = mods
    rov = platforms.get_platform("bluerov2")
    assert payloads.select_payloads(rov, []) == []
    chosen = payloads.select_payloads(rov, ["oculus_m750d", "waterlinked_a50", "Newton-Gripper"])
    assert [p.kind for p in chosen] == ["sonar", "dvl", "gripper"]
    with pytest.raises(ValueError):
        payloads.select_payloads(rov, ["oculus_m750d", "tritech_gemini_720is"])
    with pytest.raises(KeyError):
        payloads.select_payloads(rov, ["sidescan_9000"])
    assert payloads.sensor_payload(payloads.select_payloads(rov, ["ping360"]), "sonar") is None


def test_platform_payload_lists_reference_the_catalogue(mods):
    platforms, payloads, *_ = mods
    for name in platforms.available_platforms():
        plat = platforms.get_platform(name)
        for p in tuple(plat.payload_options) + tuple(plat.default_payloads):
            payloads.get_payload(p)
        payloads.select_payloads(plat)        # defaults are a valid combination


# --- payload effects --------------------------------------------------------------

def test_payloads_shift_mass_cog_inertia_and_drag(mods):
    platforms, payloads, vd, *_ = mods
    rov = platforms.get_platform("bluerov2")
    bare = vd.resolve_hydro(rov)
    fitted = payloads.select_payloads(rov, ["tritech_gemini_720is", "nortek_dvl500"])
    h = vd.resolve_hydro(rov, fitted, trim=False)
    added_mass = 3.40 + 3.5
    assert h["mass"] == pytest.approx(rov.mass + added_mass)
    assert h["displaced_volume"] == pytest.approx(bare["displaced_volume"] + 0.0021 + 0.003)
    # CoG moves toward the payload mass centre (sonar fwd-high, DVL low)
    r = (3.40 * np.asarray(rov.sonar_mount.translation) + 3.5 * np.asarray(rov.dvl_mount.translation))
    assert np.allclose(h["cog"], r / (rov.mass + added_mass))
    assert all(a > b for a, b in zip(h["inertia"], bare["inertia"]))
    assert all(a > b for a, b in zip(h["quadratic_damping"][:3], bare["quadratic_damping"][:3]))
    assert h["quadratic_damping"][3:] == bare["quadratic_damping"][3:]
    # thrusters keep their place on the hull: positions re-expressed about the new CoG
    t0 = np.asarray(bare["thrusters"][0].position)
    t1 = np.asarray(h["thrusters"][0].position)
    assert np.allclose(t1 + np.asarray(h["cog"]), t0)
    # heavy payloads make the untrimmed vehicle sink
    assert 1000 * h["displaced_volume"] < h["mass"] and h["trim"] is None


@pytest.mark.parametrize("rho", [1000.0, 1025.0])
def test_trim_restores_the_bare_vehicles_buoyancy(mods, rho):
    platforms, payloads, vd, *_ = mods
    rov = platforms.get_platform("bluerov2")
    bare = vd.resolve_hydro(rov, rho=rho)
    fitted = payloads.select_payloads(rov, ["oculus_m750d", "waterlinked_a50", "newton_gripper"])
    h = vd.resolve_hydro(rov, fitted, rho=rho)
    frac = lambda d: (rho * d["displaced_volume"] - d["mass"]) / d["mass"]
    assert frac(h) == pytest.approx(frac(bare), abs=1e-9)
    assert h["trim"]["kind"] == "foam" and h["trim"]["volume"] > 0
    untrimmed = vd.resolve_hydro(rov, fitted, rho=rho, trim=False)
    assert h["cob"][2] > untrimmed["cob"][2]          # foam high on the frame: more stable


def test_buoyant_payload_is_trimmed_with_lead(mods):
    platforms, payloads, vd, *_ = mods
    rov = platforms.get_platform("bluerov2")
    float_pod = payloads.PayloadSpec("float", "skid", "test float", "x", mass=0.5, volume=0.003,
                                     size=(0.1, 0.1, 0.1))
    h = vd.resolve_hydro(rov, [float_pod])
    assert h["trim"]["kind"] == "lead"
    bare = vd.resolve_hydro(rov)
    assert ((1000 * h["displaced_volume"] - h["mass"]) / h["mass"]
            == pytest.approx((1000 * bare["displaced_volume"] - bare["mass"]) / bare["mass"]))


def test_vehicle_model_uses_the_cog_for_surface_buoyancy(mods):
    platforms, payloads, vd, *_ = mods
    rov = platforms.get_platform("bluerov2")
    fitted = payloads.select_payloads(rov, ["oculus_m750d"])
    model = vd.from_platform(rov, payloads=fitted, surface_z=0.0)
    assert np.allclose(model.cog, vd.resolve_hydro(rov, fitted)["cog"])
    assert model.hydro.mass == pytest.approx(vd.resolve_hydro(rov, fitted)["mass"])
    assert vd.from_platform(rov, payloads=fitted, trim=False).hydro.mass == pytest.approx(rov.mass + 0.98)


# --- URDF export / import -----------------------------------------------------------

def _hydro_equal(a, b):
    for k in ("mass", "displaced_volume", "height", "thruster_time_constant"):
        assert a[k] == pytest.approx(b[k], rel=1e-6), k
    for k in ("cog", "inertia", "cob", "added_mass", "linear_damping", "quadratic_damping"):
        assert list(a[k]) == pytest.approx(list(b[k]), rel=1e-5, abs=1e-9), k
    assert len(a["thrusters"]) == len(b["thrusters"])
    for ta, tb in zip(a["thrusters"], b["thrusters"]):
        assert list(ta.position) == pytest.approx(list(tb.position), abs=1e-6)
        assert list(ta.direction) == pytest.approx(list(tb.direction), abs=1e-6)


ALL = ["bluerov2", "bluerov2_heavy", "deeptrekker_revolution", "deeptrekker_dtg3",
       "deeptrekker_pivot", "videoray_pro5", "videoray_defender", "chasing_m2_pro_max",
       "qysea_fifish_v6_expert", "saab_seaeye_falcon", "seabotix_vlbv300"]


@pytest.mark.parametrize("name", ALL)
def test_export_import_round_trip_is_lossless(mods, name):
    platforms, payloads, vd, exp, imp = mods
    plat = platforms.get_platform(name)
    fitted = payloads.select_payloads(plat, ["oculus_m750d", "waterlinked_a50", "newton_gripper"])
    urdf = exp.platform_to_urdf(plat, fitted)
    ET.fromstring(urdf)                                         # well-formed
    assert imp.hydro_source(urdf) == "oceansim"
    spec, notes = imp.platform_from_urdf(urdf)
    assert spec.name == name and spec.usd_subpath is None
    _hydro_equal(vd.resolve_hydro(spec), vd.resolve_hydro(plat, fitted))
    a = vd.from_platform(spec)
    b = vd.from_platform(plat, payloads=fitted)
    assert np.allclose(a.thrusters.max_forward, b.thrusters.max_forward)
    assert np.allclose(a.thrusters.B, b.thrusters.B)
    assert any("payloads" in n for n in notes)


def test_exported_sensor_frames_match_the_platform_mounts(mods):
    platforms, payloads, _, exp, _ = mods
    urdf_parse = _load("urdf_parse")
    plat = platforms.get_platform("bluerov2")
    urdf = exp.platform_to_urdf(plat, payloads.select_payloads(plat, ["ping2"]))
    for kind in ("sonar", "camera", "dvl"):
        tr, rpy = urdf_parse.sensor_mount(urdf, kind)
        m = plat.mount(kind)
        assert list(tr) == pytest.approx(list(m.translation), abs=1e-6)
        assert list(rpy) == pytest.approx(list(m.rpy_deg), abs=1e-4)
    assert urdf_parse.link_pose_in_base(urdf, "altimeter_link") is not None


def test_gripper_jaws_are_revolute_joints(mods):
    platforms, payloads, _, exp, _ = mods
    plat = platforms.get_platform("bluerov2")
    root = ET.fromstring(exp.platform_to_urdf(plat, payloads.select_payloads(plat, ["newton_gripper"])))
    jaws = [j for j in root.findall("joint") if j.get("name").startswith("gripper_jaw")]
    assert len(jaws) == 2 and all(j.get("type") == "revolute" for j in jaws)


def test_gazebo_plugins_import_without_oceansim_block(mods):
    """A URDF carrying only the Gazebo Sim plugins (as bluerov2_gz-style
    models do) imports damping, added mass and thrusters; the displaced volume
    is an announced assumption."""
    platforms, payloads, vd, exp, imp = mods
    plat = platforms.get_platform("bluerov2")
    root = ET.fromstring(exp.platform_to_urdf(plat, (), gazebo=True))
    root.remove(root.find("oceansim"))
    urdf = ET.tostring(root, encoding="unicode")
    assert imp.hydro_source(urdf) == "gazebo"
    spec, notes = imp.platform_from_urdf(urdf)
    got, want = vd.resolve_hydro(spec), vd.resolve_hydro(plat)
    for k in ("added_mass", "linear_damping", "quadratic_damping", "inertia", "cog"):
        assert list(got[k]) == pytest.approx(list(want[k]), rel=1e-5, abs=1e-9), k
    model = vd.from_platform(spec)
    fwd, rev = vd.t200_max_thrust(14.8)
    assert model.thrusters.max_forward.tolist() == pytest.approx([fwd] * 6, rel=1e-5)
    assert model.thrusters.max_reverse.tolist() == pytest.approx([rev] * 6, rel=1e-5)
    assert np.allclose(model.thrusters.B, vd.from_platform(plat).thrusters.B, atol=1e-6)
    assert any("volume" in n for n in notes)


UUV_URDF = """<?xml version="1.0"?>
<robot name="uuv_rov">
  <link name="base_link">
    <inertial><origin xyz="0 0 0"/><mass value="10"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.4"/></inertial>
    <collision><geometry><box size="0.5 0.4 0.3"/></geometry></collision>
  </link>
  <link name="thruster_0"/>
  <joint name="thruster_0_joint" type="continuous">
    <parent link="base_link"/><child link="thruster_0"/>
    <origin xyz="-0.2 0.1 0" rpy="0 0 0"/><axis xyz="1 0 0"/>
  </joint>
  <gazebo>
    <plugin name="uuv_plugin" filename="libuuv_underwater_object_ros_plugin.so">
      <link name="base_link">
        <volume>0.0101</volume>
        <center_of_buoyancy>0 0 0.02</center_of_buoyancy>
        <hydrodynamic_model>
          <type>fossen</type>
          <added_mass>5 0 0 0 0 0  0 6 0 0 0 0  0 0 7 0 0 0  0 0 0 0.1 0 0  0 0 0 0 0.2 0  0 0 0 0 0 0.3</added_mass>
          <linear_damping>-1 -2 -3 -0.1 -0.2 -0.3</linear_damping>
          <quadratic_damping>-10 -20 -30 -1 -2 -3</quadratic_damping>
        </hydrodynamic_model>
      </link>
    </plugin>
    <plugin name="thruster_0_plugin" filename="libuuv_thruster_ros_plugin.so">
      <linkName>thruster_0</linkName><jointName>thruster_0_joint</jointName>
      <clampMax>200</clampMax><clampMin>-200</clampMin>
      <conversion><type>Basic</type><rotorConstant>0.001</rotorConstant></conversion>
    </plugin>
  </gazebo>
</robot>
"""


def test_uuv_simulator_urdf_import(mods):
    _, _, vd, _, imp = mods
    assert imp.hydro_source(UUV_URDF) == "uuv"
    spec, notes = imp.platform_from_urdf(UUV_URDF)
    h = vd.resolve_hydro(spec)
    assert spec.mass == 10.0 and h["displaced_volume"] == pytest.approx(0.0101)
    assert list(h["added_mass"]) == [5, 6, 7, 0.1, 0.2, 0.3]
    assert list(h["linear_damping"]) == [1, 2, 3, 0.1, 0.2, 0.3]
    assert list(h["quadratic_damping"]) == [10, 20, 30, 1, 2, 3]
    assert list(h["cob"]) == pytest.approx([0, 0, 0.02])
    t = h["thrusters"][0]
    assert list(t.position) == pytest.approx([-0.2, 0.1, 0.0])
    assert list(t.direction) == pytest.approx([1.0, 0.0, 0.0])
    assert t.max_forward == pytest.approx(0.001 * 200 ** 2)
    assert h["height"] == pytest.approx(0.3)


def test_urdf_without_hydrodynamics_is_rejected(mods):
    *_, imp = mods
    plain = '<robot name="r"><link name="base_link"/></robot>'
    assert imp.hydro_source(plain) is None
    with pytest.raises(ValueError):
        imp.platform_from_urdf(plain)


def test_platforms_without_assets_resolve_to_nothing(mods, tmp_path):
    platforms, *_ = mods
    spec = platforms.PlatformSpec(
        name="x", usd_subpath=None, mass=1.0, linear_damping=1.0, angular_damping=1.0,
        collision_approximation="boundingCube", spawn_translation=(0, 0, 0),
        sonar_mount=platforms.SensorMount((0, 0, 0)), camera_mount=platforms.SensorMount((0, 0, 0)),
        dvl_mount=platforms.SensorMount((0, 0, 0)))
    assert spec.usd_path(str(tmp_path)) is None
    src, why = platforms.resolve_robot_source(asset_root=str(tmp_path), platform=spec)
    assert src is None and why == "none"


# --- sensor presets -------------------------------------------------------------

def test_sonar_preset_maps_datasheet_to_sensor_args(mods):
    _, payloads, *_ = mods
    sp = _load("sensor_presets")
    m750 = payloads.get_payload("oculus_m750d")
    kw = sp.sonar_kwargs(m750)
    assert kw["hori_fov"] == 130.0 and kw["vert_fov"] == 20.0
    assert kw["angular_res"] == pytest.approx(130.0 / 512)
    assert kw["max_range"] == 20.0 and kw["min_range"] == 0.1
    assert kw["range_res"] == pytest.approx(20.0 / sp.MAX_RANGE_BINS)   # coarser than 4 mm
    short = sp.sonar_kwargs(m750, {"max_range": 2.0})
    assert short["range_res"] == pytest.approx(0.004)        # device limit at short range
    assert sp.sonar_kwargs(m750, {"max_range": 2.0, "range_res": 0.004})["range_res"] == 0.004
    with pytest.raises(ValueError):
        sp.sonar_kwargs(m750, {"max_range": 500.0})
    assert sp.sonar_frequency(m750) == 750e3


@pytest.mark.parametrize("name,freq,fov,limit,res,beamwidth", [
    ("oculus_m3000d", 1.2e6, 130.0, 30.0, 0.0025, 0.6),
    ("oculus_m3000d_hf", 3.0e6, 40.0, 5.0, 0.002, 0.25),
])
def test_m3000d_modes_and_resolution_are_distinct(mods, name, freq, fov, limit, res, beamwidth):
    _, payloads, *_ = mods
    presets = _load("sensor_presets")
    payload = payloads.get_payload(name)
    assert payload.params["max_range"] == limit
    assert payload.params["range_res"] == res
    assert payload.params["rate_hz"] == 40.0
    kw = presets.sonar_kwargs(payload)
    assert kw["acoustic_frequency"] == freq
    assert kw["hori_fov"] == fov and kw["vert_fov"] == 20.0
    assert kw["min_range"] == 0.1
    assert kw["hori_fov"] / kw["angular_res"] == pytest.approx(512)
    assert kw["beam_fwhm_deg"] == beamwidth
    assert kw["beam_fwhm_deg"] > kw["angular_res"]
    assert presets.sonar_kwargs(payload, {"max_range": 1.0})["range_res"] == res
    with pytest.raises(ValueError, match="exceeds"):
        presets.sonar_kwargs(payload, {"max_range": limit + 0.1})
    assert presets.sonar_model_params(payload) == {"normalizing_method": "all", "gamma": 0.5,
                                                   "beam_fwhm_deg": beamwidth}
    override = presets.sonar_model_params(
        payload, {"beam_fwhm_deg": 0, "normalizing_method": "range", "gamma": 1.0})
    assert override == {"beam_fwhm_deg": 0, "normalizing_method": "range", "gamma": 1.0}


def test_every_imaging_sonar_preset_normalises_per_ping(mods):
    """Normalisation is not tied to whether a payload records a beamwidth:
    swapping the M3000d for another sonar must not switch it to per-range."""
    _, payloads, *_ = mods
    sp = _load("sensor_presets")
    sonars = [p for p in payloads.PAYLOADS.values() if p.kind == "sonar" and p.simulated]
    assert len(sonars) > 5
    for payload in sonars:
        params = sp.sonar_model_params(payload)
        assert params["normalizing_method"] == "all", payload.name
        assert params["gamma"] == 0.5, payload.name
        beamwidth = payload.params.get("beamwidth_h_deg")
        assert params.get("beam_fwhm_deg") == beamwidth, payload.name
        assert sp.sonar_kwargs(payload)["beam_fwhm_deg"] == (beamwidth or 0.0), payload.name


@pytest.mark.parametrize("name,beamwidth", [
    ("oculus_m370s", 2.0), ("oculus_m750d_hf", 0.6), ("oculus_m1200d", 0.6),
    ("oculus_m1200d_hf", 0.4), ("oculus_m3000d", 0.6), ("oculus_m3000d_hf", 0.25),
])
def test_oculus_payload_beamwidths_match_band_constants(mods, name, beamwidth):
    """Payload beamwidths agree with the Oculus band constants ros2_math
    publishes for the RTX backend, so both backends describe a device alike."""
    _, payloads, *_ = mods
    payload = payloads.get_payload(name)
    assert payload.params["beamwidth_h_deg"] == beamwidth
    az, _ = _load("ros2_math").oculus_beamwidths(payload.params["frequency_hz"])
    assert math.degrees(az) == pytest.approx(beamwidth)


def _init_defaults(path, cls):
    """Constant default arguments of cls.__init__ in a source file (parsed,
    not imported: the sensor modules need Isaac Sim)."""
    import ast
    tree = ast.parse(open(path).read())
    init = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    args = init.args.args[-len(init.args.defaults):]
    return {a.arg: eval(compile(ast.Expression(d), path, "eval"), {"__builtins__": {}})
            for a, d in zip(args, init.args.defaults)}


@pytest.mark.parametrize("module,cls", [("ImagingSonarSensor", "ImagingSonarSensor"),
                                        ("ImagingSonarSensor_ROS", "ImagingSonarSensor_ROS")])
def test_sonar_constructor_defaults_match_default_payload(mods, module, cls):
    """A sensor built with no arguments matches what the GUI and the ROS
    runner build from DEFAULT_SONAR_PAYLOAD (and stays within the bin cap)."""
    _, payloads, *_ = mods
    sp = _load("sensor_presets")
    expected = sp.sonar_kwargs(payloads.get_payload(sp.DEFAULT_SONAR_PAYLOAD))
    path = os.path.join(_UTILS, "..", "sensors", module + ".py")
    defaults = _init_defaults(path, cls)
    shared = expected.keys() & defaults.keys()
    assert {"min_range", "max_range", "range_res", "hori_fov", "angular_res"} <= shared
    for key in shared:
        assert defaults[key] == pytest.approx(expected[key]), key
    span = defaults["max_range"] - defaults["min_range"]
    assert math.ceil(span / defaults["range_res"]) <= sp.MAX_RANGE_BINS


def test_dvl_altimeter_and_camera_presets(mods):
    _, payloads, *_ = mods
    sp = _load("sensor_presets")
    a50 = sp.dvl_kwargs(payloads.get_payload("waterlinked_a50"))
    assert a50 == dict(elevation=22.5, min_range=0.05, max_range=50.0, vel_cov=0.0)
    nortek = sp.dvl_kwargs(payloads.get_payload("nortek_dvl500"))
    assert nortek["vel_cov"] == pytest.approx(0.008 ** 2) and nortek["elevation"] == 25.0
    ping2 = sp.altimeter_kwargs(payloads.get_payload("ping2"))
    assert ping2["beamwidth_deg"] == 25.0 and ping2["max_range"] == 100.0
    f = sp.focal_length_for_hfov(90.0, 20.955)
    assert 2 * math.degrees(math.atan(20.955 / 2 / f)) == pytest.approx(90.0)
    with pytest.raises(ValueError):
        sp.focal_length_for_hfov(200.0, 20.955)
