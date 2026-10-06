"""Export an OceanSim platform (with its payloads) as a URDF.

The URDF carries:

* base_link with the vehicle's total mass, CoG and principal inertia (payloads
  included), and a box hull of the platform's dimensions (visual + collision);
* one link per thruster (a cylinder along its thrust direction), per payload
  (a box of its housing size; a gripper gets two revolute jaws) and the sensor
  frames OceanSim and ROS look for (sonar_link, camera_link, dvl_link, ...);
* an ``<oceansim>`` element with the complete model -- hydrodynamics, thruster
  limits, payload list -- so ``urdf_platform.platform_from_urdf`` re-imports
  it losslessly (urdfdom / robot_state_publisher ignore unknown elements);
* with ``gazebo=True``: continuous thruster joints plus Gazebo Sim's
  Hydrodynamics and Thruster system plugins, for use in gz-sim.

Primitive shapes stand in for a mesh: the vehicles without a 3D asset in the
OceanSim asset pack are imported into Isaac from this URDF.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np

try:
    from isaacsim.oceansim.utils import vehicle_dynamics
except ImportError:  # loaded by file path (tests, scripts/oceansim_urdf.py)
    import importlib.util
    import os
    _spec = importlib.util.spec_from_file_location(
        "vehicle_dynamics", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "vehicle_dynamics.py"))
    vehicle_dynamics = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(vehicle_dynamics)

FORMAT_VERSION = "1"
_TINY_MASS = 0.001          # kg on frame links: valid URDF inertials, negligible

_COLOURS = {
    "Blue Robotics": "0.10 0.35 0.75 1",
    "Deep Trekker": "0.95 0.75 0.10 1",
    "VideoRay": "0.90 0.45 0.10 1",
    "Chasing": "0.85 0.85 0.85 1",
    "QYSEA": "0.90 0.90 0.92 1",
    "Saab Seaeye": "0.95 0.80 0.15 1",
    "Teledyne SeaBotix": "0.95 0.70 0.10 1",
}
_SENSOR_LINKS = {"sonar": "sonar_link", "camera": "camera_link", "dvl": "dvl_link",
                 "altimeter": "altimeter_link", "usbl": "usbl_link", "gripper": "gripper_base",
                 "light": "light_link", "laser": "laser_link", "skid": "skid_link"}


def _fmt(values):
    return " ".join(f"{float(v):.9g}" for v in values)


def _dir_rpy(d):
    """URDF rpy (rad) that turns the link's +z axis onto unit vector d."""
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    return (0.0, math.atan2(math.hypot(d[0], d[1]), d[2]), math.atan2(d[1], d[0]))


def _inertial(link, mass, xyz=(0, 0, 0), inertia=None):
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=_fmt(xyz), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{float(mass):.6g}")
    ixx, iyy, izz = inertia if inertia is not None else (1e-6, 1e-6, 1e-6)
    ET.SubElement(inertial, "inertia", ixx=f"{ixx:.6g}", ixy="0", ixz="0",
                  iyy=f"{iyy:.6g}", iyz="0", izz=f"{izz:.6g}")


def _visual(link, geom, size, material, xyz=(0, 0, 0), rpy=(0, 0, 0), collision=False):
    for tag in (("visual", "collision") if collision else ("visual",)):
        el = ET.SubElement(link, tag)
        ET.SubElement(el, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
        g = ET.SubElement(ET.SubElement(el, "geometry"), geom[0])
        for k, v in geom[1].items():
            g.set(k, v)
        if tag == "visual":
            ET.SubElement(el, "material", name=material)
    return link


def _joint(robot, name, parent, child, xyz, rpy, jtype="fixed", axis=None, limit=None):
    j = ET.SubElement(robot, "joint", name=name, type=jtype)
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)
    ET.SubElement(j, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
    if axis is not None:
        ET.SubElement(j, "axis", xyz=_fmt(axis))
    if limit is not None:
        ET.SubElement(j, "limit", **{k: f"{v:.6g}" for k, v in limit.items()})
    return j


def platform_to_urdf(spec, payloads=(), gazebo=False, rho=1000.0):
    """URDF XML string for ``spec`` with ``payloads`` (PayloadSpec list) fitted."""
    h = vehicle_dynamics.resolve_hydro(spec, payloads, rho=rho)
    cog = np.asarray(h["cog"], dtype=float)
    length, width, height = spec.dimensions or (0.4, 0.3, 0.25)

    robot = ET.Element("robot", name=spec.name)
    ET.SubElement(ET.SubElement(robot, "material", name="hull"), "color",
                  rgba=_COLOURS.get(spec.manufacturer, "0.7 0.7 0.7 1"))
    ET.SubElement(ET.SubElement(robot, "material", name="thruster"), "color", rgba="0.1 0.1 0.1 1")
    ET.SubElement(ET.SubElement(robot, "material", name="payload"), "color", rgba="0.25 0.25 0.3 1")

    base = ET.SubElement(robot, "link", name="base_link")
    _inertial(base, h["mass"], cog, h["inertia"])
    _visual(base, ("box", {"size": _fmt((length, width, height))}), (length, width, height),
            "hull", collision=True)

    thruster_els = []
    for i, t in enumerate(h["thrusters"], start=1):
        name = f"thruster{i}"
        link = ET.SubElement(robot, "link", name=name)
        _inertial(link, _TINY_MASS)
        _visual(link, ("cylinder", {"radius": "0.04", "length": "0.09"}), None, "thruster")
        pos = cog + np.asarray(t.position, dtype=float)
        if gazebo:
            thruster_els.append(_joint(robot, f"{name}_joint", "base_link", name, pos,
                                       _dir_rpy(t.direction), "continuous", axis=(0, 0, 1)))
        else:
            _joint(robot, f"{name}_joint", "base_link", name, pos, _dir_rpy(t.direction))

    # payload bodies + the sensor frames (sonar / camera / dvl always present)
    fitted = {p.kind: p for p in payloads}
    for kind in sorted(set(("sonar", "camera", "dvl")) | set(fitted)):
        p = fitted.get(kind)
        mount = spec.mount(kind)
        xyz = np.asarray(mount.translation, dtype=float) + (
            np.asarray(p.offset, dtype=float) if p is not None else 0.0)
        rpy = tuple(math.radians(a) for a in mount.rpy_deg)
        link_name = _SENSOR_LINKS.get(kind, f"{kind}_link")
        link = ET.SubElement(robot, "link", name=link_name)
        _inertial(link, _TINY_MASS)
        if p is not None:
            _visual(link, ("box", {"size": _fmt(p.size)}), p.size, "payload")
        _joint(robot, f"{link_name}_joint", "base_link", link_name, xyz, rpy)
        if kind == "gripper" and p is not None:
            _add_gripper_jaws(robot, link_name, p)

    _add_oceansim_block(robot, spec, h, payloads)
    if gazebo:
        _add_gazebo(robot, h, thruster_els, rho)
    ET.indent(robot, space="  ")
    return '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode") + "\n"


def _add_gripper_jaws(robot, parent, p):
    """Two revolute jaws (driven through /oceansim/robot/joint_command)."""
    jaw = p.params.get("jaw_length", 0.08)
    opening = math.radians(p.params.get("opening_deg", 60.0))
    for side, sign in (("upper", 1.0), ("lower", -1.0)):
        name = f"gripper_jaw_{side}"
        link = ET.SubElement(robot, "link", name=name)
        _inertial(link, 0.02, (jaw / 2, 0, 0), (1e-5, 1e-5, 1e-5))
        _visual(link, ("box", {"size": _fmt((jaw, 0.03, 0.01))}), None, "payload",
                xyz=(jaw / 2, 0, 0), collision=True)
        _joint(robot, f"{name}_joint", parent, name, (p.size[0] / 2, 0, sign * 0.015), (0, 0, 0),
               "revolute", axis=(0, -sign, 0),
               limit={"lower": 0.0, "upper": opening / 2, "effort": 5.0, "velocity": 1.0})


def _add_oceansim_block(robot, spec, h, payloads):
    block = ET.SubElement(robot, "oceansim", version=FORMAT_VERSION, platform=spec.name)
    if spec.manufacturer:
        block.set("manufacturer", spec.manufacturer)
    if spec.dimensions:
        block.set("dimensions", _fmt(spec.dimensions))
    ET.SubElement(
        block, "hydro", mass=f"{h['mass']:.8g}", cog=_fmt(h["cog"]), inertia=_fmt(h["inertia"]),
        volume=f"{h['displaced_volume']:.8g}", cob=_fmt(h["cob"]), height=f"{h['height']:.6g}",
        added_mass=_fmt(h["added_mass"]), linear_damping=_fmt(h["linear_damping"]),
        quadratic_damping=_fmt(h["quadratic_damping"]),
        thruster_time_constant=f"{h['thruster_time_constant']:.6g}",
        **({"thruster_model": h["thruster_model"]} if h["thruster_model"] else {}),
        **({"battery_voltage": f"{h['battery_voltage']:.6g}"} if h["battery_voltage"] else {}),
    ).set("sources", h["sources"])
    cog = np.asarray(h["cog"], dtype=float)
    for i, t in enumerate(h["thrusters"], start=1):
        el = ET.SubElement(block, "thruster", name=f"thruster{i}",
                           xyz=_fmt(cog + np.asarray(t.position, float)),
                           direction=_fmt(t.direction))
        if t.max_forward is not None:
            el.set("max_forward", f"{t.max_forward:.6g}")
        if t.max_reverse is not None:
            el.set("max_reverse", f"{t.max_reverse:.6g}")
    for p in payloads:
        ET.SubElement(block, "payload", name=p.name, kind=p.kind,
                      link=_SENSOR_LINKS.get(p.kind, f"{p.kind}_link"))


def _add_gazebo(robot, h, thruster_joints, rho):
    """Gazebo Sim (gz-sim) Hydrodynamics + Thruster system plugins. gz's
    buoyancy comes from its world-level Buoyancy system and the collision
    volume, so the net buoyancy differs from OceanSim's unless you tune it."""
    gz = ET.SubElement(robot, "gazebo")
    hyd = ET.SubElement(gz, "plugin", filename="gz-sim-hydrodynamics-system",
                        name="gz::sim::systems::Hydrodynamics")
    ET.SubElement(hyd, "link_name").text = "base_link"
    ET.SubElement(hyd, "water_density").text = f"{rho:g}"
    names = ("x", "y", "z", "k", "m", "n")
    vel = ("U", "V", "W", "P", "Q", "R")
    for i in range(6):
        ET.SubElement(hyd, f"{names[i]}Dot{vel[i]}").text = f"{-h['added_mass'][i]:.6g}"
        ET.SubElement(hyd, f"{names[i]}{vel[i]}").text = f"{-h['linear_damping'][i]:.6g}"
        ET.SubElement(hyd, f"{names[i]}{vel[i]}abs{vel[i]}").text = f"{-h['quadratic_damping'][i]:.6g}"
    fwd, rev = vehicle_dynamics._thruster_limits(h["thrusters"], h["thruster_model"],
                                                 h["battery_voltage"])
    for joint, f_max, r_max in zip(thruster_joints, fwd, rev):
        th = ET.SubElement(gz, "plugin", filename="gz-sim-thruster-system",
                           name="gz::sim::systems::Thruster")
        ET.SubElement(th, "namespace").text = robot.get("name")
        ET.SubElement(th, "joint_name").text = joint.get("name")
        ET.SubElement(th, "thrust_coefficient").text = "0.02"
        ET.SubElement(th, "fluid_density").text = f"{rho:g}"
        ET.SubElement(th, "propeller_diameter").text = "0.1"
        ET.SubElement(th, "max_thrust_cmd").text = f"{f_max:.6g}"
        ET.SubElement(th, "min_thrust_cmd").text = f"{-r_max:.6g}"


def write_generated_urdf(spec, payloads=(), out_dir=None, rho=1000.0):
    """Write ``spec`` + ``payloads`` as a URDF for the simulator to import (the
    vehicles without a 3D asset). Returns (path, urdf_text). The file name
    encodes the payloads, so configurations don't overwrite each other."""
    import os
    import tempfile
    out_dir = out_dir or os.path.join(tempfile.gettempdir(), "oceansim_urdf")
    os.makedirs(out_dir, exist_ok=True)
    tag = "__".join(p.name for p in payloads) or "bare"
    path = os.path.join(out_dir, f"{spec.name}__{tag}.urdf")
    text = platform_to_urdf(spec, payloads, rho=rho)
    with open(path, "w") as f:
        f.write(text)
    return path, text
