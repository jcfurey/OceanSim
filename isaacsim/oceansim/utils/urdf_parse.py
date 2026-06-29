"""Pure URDF parsing: resolve sensor mount poses from a robot's URDF.

When a vehicle is imported from a URDF, the URDF usually already defines where
the sensors go -- a ``sonar`` / ``camera`` / ``dvl`` link attached to the base by
a fixed joint whose ``<origin>`` is the mount pose. This module reads that pose
(composing the joint chain from the base link) so OceanSim can spawn its sensors
at the URDF-defined frames instead of the platform's hardcoded fallback mounts.

Pure: ``xml.etree`` + ``numpy`` only, so it is unit tested without Isaac. Returns
``(translation, rpy_deg)`` tuples in the same (roll, pitch, yaw) degrees
convention the sensor mounts / ``euler_angles_to_quat(..., degrees=True)`` use.
"""

import xml.etree.ElementTree as ET

import numpy as np

# Conventional URDF link names per sensor kind (matched case-insensitively).
DEFAULT_SENSOR_LINKS = {
    "sonar": ["sonar", "sonar_link", "imaging_sonar", "oculus", "sonar0", "forward_sonar"],
    "camera": ["camera", "camera_link", "cam", "uw_camera", "camera0", "front_camera"],
    "dvl": ["dvl", "dvl_link", "dvl0", "doppler"],
    "baro": ["baro", "barometer", "pressure", "pressure_sensor", "depth"],
}


def _floats(text, n):
    vals = [float(v) for v in text.split()] if text else [0.0] * n
    if len(vals) != n:
        raise ValueError(f"expected {n} numbers, got {text!r}")
    return np.array(vals, dtype=float)


def _rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    # URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (fixed-axis XYZ).
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _matrix_to_rpy(R):
    # Inverse of _rpy_to_matrix (no gimbal-lock special-casing; sensor mounts are
    # not at +/-90 deg pitch in practice).
    sp = -R[2, 0]
    sp = max(-1.0, min(1.0, sp))
    pitch = np.arcsin(sp)
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def _homog(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def _parse(urdf_text):
    """Return (joints_by_child, link_names). joints_by_child[child] =
    (parent, xyz, rpy) for the fixed/articulated joint whose child is `child`."""
    root = ET.fromstring(urdf_text)
    joints_by_child = {}
    links = set()
    for link in root.findall("link"):
        if link.get("name"):
            links.add(link.get("name"))
    for joint in root.findall("joint"):
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link")
        child = child_el.get("link")
        origin = joint.find("origin")
        xyz = _floats(origin.get("xyz") if origin is not None else None, 3)
        rpy = _floats(origin.get("rpy") if origin is not None else None, 3)
        joints_by_child[child] = (parent, xyz, rpy)
        links.add(parent)
        links.add(child)
    return joints_by_child, links


def _root_link(joints_by_child, links):
    """The link that is never a joint child (the kinematic root / base)."""
    children = set(joints_by_child)
    roots = [l for l in links if l not in children]
    return roots[0] if len(roots) == 1 else (roots[0] if roots else None)


def link_pose_in_base(urdf_text, link_name, base_link=None):
    """Pose of ``link_name`` expressed in ``base_link`` (the kinematic root if
    None), as ``(translation, rpy_deg)``. Composes the joint origins along the
    chain. Returns None if the link doesn't exist or isn't connected to base.
    """
    joints_by_child, links = _parse(urdf_text)
    if link_name not in links:
        return None
    if base_link is None:
        base_link = _root_link(joints_by_child, links)
    if link_name == base_link:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    # Walk parent pointers from the target up to base, collecting joint origins.
    chain = []
    node = link_name
    seen = set()
    while node != base_link:
        if node not in joints_by_child or node in seen:
            return None  # reached a different root, or a cycle
        seen.add(node)
        parent, xyz, rpy = joints_by_child[node]
        chain.append((xyz, rpy))
        node = parent

    # chain is target..base; compose base->target = product of origins reversed.
    T = np.eye(4)
    for xyz, rpy in reversed(chain):
        T = T @ _homog(xyz, rpy)
    translation = tuple(float(v) for v in T[:3, 3])
    rpy_deg = tuple(float(np.degrees(a)) for a in _matrix_to_rpy(T[:3, :3]))
    return translation, rpy_deg


def root_link(urdf_text):
    """Name of the kinematic root (base) link -- the link that is never a joint
    child. None if the URDF can't be parsed. Used as the ROS base frame so the
    odom / sensor frames line up with robot_state_publisher's TF tree."""
    try:
        joints_by_child, links = _parse(urdf_text)
    except (ET.ParseError, ValueError):
        return None
    return _root_link(joints_by_child, links)


def sensor_link(urdf_text, kind, candidates=None):
    """The URDF link name OceanSim would mount sensor ``kind`` to (the matched
    conventional link), or None if the URDF defines none / can't be parsed."""
    try:
        cands = candidates if candidates is not None else DEFAULT_SENSOR_LINKS.get(kind, [])
        return find_link(urdf_text, cands)
    except (ET.ParseError, ValueError):
        return None


def find_link(urdf_text, candidates):
    """First link in ``candidates`` that exists in the URDF (case-insensitive),
    returned with its actual casing; None if none match."""
    _, links = _parse(urdf_text)
    lower = {l.lower(): l for l in links}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def sensor_mount(urdf_text, kind, candidates=None):
    """``(translation, rpy_deg)`` for sensor ``kind`` from the URDF, or None if
    the URDF has no matching sensor link (or can't be parsed)."""
    try:
        cands = candidates if candidates is not None else DEFAULT_SENSOR_LINKS.get(kind, [])
        link = find_link(urdf_text, cands)
        if link is None:
            return None
        return link_pose_in_base(urdf_text, link)
    except (ET.ParseError, ValueError):
        return None


def sensor_mount_or(urdf_text, kind, fallback_translation, fallback_rpy_deg, candidates=None):
    """URDF-derived ``(translation, rpy_deg)`` for ``kind`` if the URDF defines a
    matching sensor link, else the supplied fallbacks. ``urdf_text`` may be None
    (e.g. a USD-sourced robot) -> always the fallback."""
    m = sensor_mount(urdf_text, kind, candidates) if urdf_text else None
    if m is None:
        return tuple(fallback_translation), tuple(fallback_rpy_deg)
    return m
