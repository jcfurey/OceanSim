"""Build an OceanSim platform (PlatformSpec with hydrodynamics + thrusters)
from a URDF, so any vehicle with a URDF can be simulated with its dynamics.

Understood, in order of preference:

1. ``<oceansim>`` -- written by urdf_export; lossless.
2. Gazebo Sim plugins -- ``gz-sim-hydrodynamics-system`` (xDotU, xU, xUabsU,
   ...) and ``gz-sim-thruster-system`` per thruster joint (position and axis
   from the joint, limits from max/min_thrust_cmd), as in bluerov2_gz / orca.
3. UUV Simulator plugins -- ``uuv_underwater_object`` (volume, centre of
   buoyancy, Fossen added mass / damping) and ``uuv_thruster`` (Basic
   conversion: thrust = rotorConstant * w|w| at the clamp).

Mass, CoG and inertia come from base_link's <inertial>; sensor mounts from
the usual link names (urdf_parse). Whatever the URDF lacks (a gz URDF has no
displaced volume, for instance) is filled with a stated assumption -- each one
is listed in the returned notes. Xacro must be expanded first.
"""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np

try:
    from isaacsim.oceansim.utils import platforms, urdf_parse
except ImportError:  # loaded by file path (tests, scripts/oceansim_urdf.py)
    import importlib.util
    _here = os.path.dirname(os.path.abspath(__file__))

    def _load(name):
        spec = importlib.util.spec_from_file_location(name, os.path.join(_here, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    platforms = _load("platforms")
    urdf_parse = _load("urdf_parse")


def _floats(text, n=None):
    vals = [float(v) for v in str(text).replace(",", " ").split()]
    if n is not None and len(vals) != n:
        raise ValueError(f"expected {n} numbers, got {text!r}")
    return vals


def _diag6(text):
    """6 values, or the diagonal of 36 (a 6x6 matrix)."""
    vals = _floats(text)
    if len(vals) == 36:
        return [vals[i * 7] for i in range(6)]
    if len(vals) == 6:
        return vals
    raise ValueError(f"expected 6 or 36 numbers, got {len(vals)}")


def _rot(rpy_deg):
    r, p, y = (math.radians(a) for a in rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def hydro_source(urdf_text):
    """'oceansim', 'gazebo', 'uuv' or None: which hydrodynamic data the URDF has."""
    root = ET.fromstring(urdf_text)
    if root.find("oceansim") is not None:
        return "oceansim"
    for plugin in root.iter("plugin"):
        fname = (plugin.get("filename") or "").lower()
        if "hydrodynamics" in fname:
            return "gazebo"
        if "underwater_object" in fname:
            return "uuv"
    return None


def _base_inertial(root, base):
    for link in root.findall("link"):
        if link.get("name") == base:
            inertial = link.find("inertial")
            if inertial is None:
                return None
            origin = inertial.find("origin")
            xyz = _floats(origin.get("xyz", "0 0 0"), 3) if origin is not None else [0, 0, 0]
            mass = float(inertial.find("mass").get("value"))
            i = inertial.find("inertia")
            inertia = [float(i.get(k, 0.0)) for k in ("ixx", "iyy", "izz")]
            return mass, xyz, inertia
    return None


def _base_box(root, base):
    for link in root.findall("link"):
        if link.get("name") == base:
            for tag in ("collision", "visual"):
                box = link.find(f"{tag}/geometry/box")
                if box is not None:
                    return _floats(box.get("size"), 3)
    return None


def _joint_child(root, joint_name):
    for j in root.findall("joint"):
        if j.get("name") == joint_name:
            axis = j.find("axis")
            return j.find("child").get("link"), (
                _floats(axis.get("xyz"), 3) if axis is not None else [1.0, 0.0, 0.0])
    return None, None


def _thruster_from_joint(urdf_text, root, joint_name, max_forward, max_reverse, cog):
    child, axis = _joint_child(root, joint_name)
    if child is None:
        raise ValueError(f"thruster joint {joint_name!r} not found")
    pose = urdf_parse.link_pose_in_base(urdf_text, child)
    if pose is None:
        raise ValueError(f"thruster link {child!r} is not connected to the base")
    xyz, rpy = pose
    d = _rot(rpy) @ np.asarray(axis, dtype=float)
    d = d / np.linalg.norm(d)
    return platforms.ThrusterSpec(tuple(float(v) for v in np.asarray(xyz) - cog),
                                  tuple(float(v) for v in d), max_forward, max_reverse)


def platform_from_urdf(urdf_text, urdf_path=None, fallback=None, rho=1000.0):
    """``(PlatformSpec, notes)`` for the vehicle in ``urdf_text``. ``fallback``
    (a PlatformSpec) supplies sensor mounts / spawn when the URDF has none.
    Raises ValueError if the URDF has no hydrodynamic data."""
    root = ET.fromstring(urdf_text)
    source = hydro_source(urdf_text)
    if source is None:
        raise ValueError("URDF has no hydrodynamic data (<oceansim>, Gazebo Hydrodynamics "
                         "or UUV Simulator plugin)")
    base = urdf_parse.root_link(urdf_text)
    notes = [f"hydrodynamics from {source}"]
    name = root.get("name") or "urdf_vehicle"
    manufacturer = ""
    dims = _base_box(root, base)

    if source == "oceansim":
        block = root.find("oceansim")
        hyd = block.find("hydro")
        name = block.get("platform", name)
        manufacturer = block.get("manufacturer", "")
        if block.get("dimensions"):
            dims = _floats(block.get("dimensions"), 3)
        cog = np.asarray(_floats(hyd.get("cog"), 3))
        thrusters = []
        for t in block.findall("thruster"):
            mf = t.get("max_forward")
            mr = t.get("max_reverse")
            thrusters.append(platforms.ThrusterSpec(
                tuple(np.asarray(_floats(t.get("xyz"), 3)) - cog), tuple(_floats(t.get("direction"), 3)),
                float(mf) if mf else None, float(mr) if mr else None))
        mass = float(hyd.get("mass"))
        hydro = platforms.HydroSpec(
            inertia=tuple(_floats(hyd.get("inertia"), 3)),
            displaced_volume=float(hyd.get("volume")),
            cob=tuple(_floats(hyd.get("cob"), 3)),
            height=float(hyd.get("height")),
            added_mass=tuple(_floats(hyd.get("added_mass"), 6)),
            linear_damping=tuple(_floats(hyd.get("linear_damping"), 6)),
            quadratic_damping=tuple(_floats(hyd.get("quadratic_damping"), 6)),
            thrusters=tuple(thrusters),
            thruster_model=hyd.get("thruster_model"),
            battery_voltage=float(hyd.get("battery_voltage")) if hyd.get("battery_voltage") else None,
            thruster_time_constant=float(hyd.get("thruster_time_constant", 0.06)),
            sources=hyd.get("sources", ""),
            cog=tuple(float(v) for v in cog))
        payload_names = [p.get("name") for p in block.findall("payload")]
        if payload_names:
            notes.append(f"payloads included in the model: {payload_names}")
    else:
        inertial = _base_inertial(root, base)
        if inertial is None:
            raise ValueError(f"base link {base!r} has no <inertial> (mass / inertia)")
        mass, cog, inertia = inertial
        cog = np.asarray(cog, dtype=float)
        if source == "gazebo":
            added, lin, quad, thrusters = _gazebo_model(urdf_text, root, cog, notes)
            volume, cob = None, None
        else:
            added, lin, quad, thrusters, volume, cob = _uuv_model(urdf_text, root, cog, notes)
        if volume is None:
            volume = mass * 1.005 / rho
            notes.append("displaced volume not given: assumed 0.5% positive buoyancy")
        if cob is None:
            cob = (0.0, 0.0, 0.01)
            notes.append("centre of buoyancy not given: assumed 1 cm above the CoG")
        height = dims[2] if dims else 0.25
        if dims is None:
            notes.append("no base box geometry: height for surface buoyancy assumed 0.25 m")
        hydro = platforms.HydroSpec(
            inertia=tuple(inertia), displaced_volume=float(volume), cob=tuple(cob), height=height,
            added_mass=tuple(abs(v) for v in added), linear_damping=tuple(abs(v) for v in lin),
            quadratic_damping=tuple(abs(v) for v in quad), thrusters=tuple(thrusters),
            sources=f"URDF {os.path.basename(urdf_path) if urdf_path else name} ({source})",
            cog=tuple(float(v) for v in cog))

    def _mount(kind, fb):
        m = urdf_parse.sensor_mount(urdf_text, kind)
        if m is not None:
            return platforms.SensorMount(tuple(m[0]), tuple(m[1]))
        return fb
    fb = fallback
    origin = platforms.SensorMount((0.0, 0.0, 0.0))
    spec = platforms.PlatformSpec(
        name=name,
        usd_subpath=None,
        mass=mass,
        linear_damping=fb.linear_damping if fb else 10.0,
        angular_damping=fb.angular_damping if fb else 10.0,
        collision_approximation="boundingCube",
        spawn_translation=fb.spawn_translation if fb else (-2.0, 0.0, -0.8),
        sonar_mount=_mount("sonar", fb.sonar_mount if fb else origin),
        camera_mount=_mount("camera", fb.camera_mount if fb else origin),
        dvl_mount=_mount("dvl", fb.dvl_mount if fb else origin),
        description=f"{name} (imported from URDF)",
        urdf_subpath=os.path.abspath(urdf_path) if urdf_path else None,
        hydro=hydro,
        dimensions=tuple(dims) if dims else None,
        manufacturer=manufacturer,
    )
    return spec, notes


def _gazebo_model(urdf_text, root, cog, notes):
    hyd = next(p for p in root.iter("plugin") if "hydrodynamics" in (p.get("filename") or "").lower())

    def _get(tag):
        el = hyd.find(tag)
        return float(el.text) if el is not None and el.text else 0.0
    names, vel = ("x", "y", "z", "k", "m", "n"), ("U", "V", "W", "P", "Q", "R")
    added = [_get(f"{names[i]}Dot{vel[i]}") for i in range(6)]
    lin = [_get(f"{names[i]}{vel[i]}") for i in range(6)]
    quad = [_get(f"{names[i]}{vel[i]}abs{vel[i]}") for i in range(6)]
    thrusters = []
    for p in root.iter("plugin"):
        if "thruster" not in (p.get("filename") or "").lower():
            continue
        joint = p.findtext("joint_name")
        mx = p.findtext("max_thrust_cmd")
        mn = p.findtext("min_thrust_cmd")
        if mx is None or mn is None:
            notes.append(f"{joint}: no max/min_thrust_cmd, thrust limit assumed 50 N")
        f_max = float(mx) if mx is not None else 50.0
        r_max = abs(float(mn)) if mn is not None else 50.0
        thrusters.append(_thruster_from_joint(urdf_text, root, joint, f_max, r_max, cog))
    if not thrusters:
        notes.append("no Gazebo Thruster plugins: the vehicle has no thrusters")
    return added, lin, quad, thrusters


def _uuv_model(urdf_text, root, cog, notes):
    obj = next(p for p in root.iter("plugin")
               if "underwater_object" in (p.get("filename") or "").lower())
    link = obj.find("link")
    volume = float(link.findtext("volume")) if link.findtext("volume") else None
    cob = None
    if link.findtext("center_of_buoyancy"):
        cob = tuple(np.asarray(_floats(link.findtext("center_of_buoyancy"), 3)) - cog)
    model = link.find("hydrodynamic_model")
    added = _diag6(model.findtext("added_mass")) if model.findtext("added_mass") else [0.0] * 6
    lin = _diag6(model.findtext("linear_damping")) if model.findtext("linear_damping") else [0.0] * 6
    quad = (_diag6(model.findtext("quadratic_damping"))
            if model.findtext("quadratic_damping") else [0.0] * 6)
    thrusters = []
    for p in root.iter("plugin"):
        if "uuv_thruster" not in (p.get("filename") or "").lower():
            continue
        joint = p.findtext("jointName")
        k = p.findtext("conversion/rotorConstant")
        clamp = p.findtext("clampMax")
        if k is not None and clamp is not None:
            f = float(k) * float(clamp) ** 2
        else:
            f = 50.0
            notes.append(f"{joint}: thrust limit not derivable, assumed 50 N")
        thrusters.append(_thruster_from_joint(urdf_text, root, joint, f, f, cog))
    return added, lin, quad, thrusters, volume, cob
