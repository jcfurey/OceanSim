"""Unit tests for the pure URDF sensor-mount parser (urdf_parse)."""

import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")

_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                     "utils", "urdf_parse.py")


def _load():
    spec = importlib.util.spec_from_file_location("urdf_parse", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def u():
    return _load()


# A small URDF: base_link with a sonar (offset + 45 deg pitch), a camera, and a
# DVL on a two-hop chain (base -> mount_plate -> dvl) to exercise composition.
URDF = """<?xml version="1.0"?>
<robot name="testbot">
  <link name="base_link"/>
  <link name="sonar_link"/>
  <link name="camera_link"/>
  <link name="mount_plate"/>
  <link name="dvl_link"/>
  <joint name="j_sonar" type="fixed">
    <parent link="base_link"/>
    <child link="sonar_link"/>
    <origin xyz="0.3 0.0 0.3" rpy="0 0.7853981633974483 0"/>
  </joint>
  <joint name="j_camera" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.35 0.0 0.1" rpy="0 0 0"/>
  </joint>
  <joint name="j_plate" type="fixed">
    <parent link="base_link"/>
    <child link="mount_plate"/>
    <origin xyz="0.0 0.0 -0.1" rpy="0 0 0"/>
  </joint>
  <joint name="j_dvl" type="fixed">
    <parent link="mount_plate"/>
    <child link="dvl_link"/>
    <origin xyz="0.0 0.0 -0.05" rpy="0 0 0"/>
  </joint>
</robot>
"""


def test_single_joint_pose_and_rpy(u):
    tr, rpy = u.link_pose_in_base(URDF, "sonar_link")
    assert np.allclose(tr, [0.3, 0.0, 0.3])
    assert np.allclose(rpy, [0.0, 45.0, 0.0], atol=1e-6)   # 0.785398 rad -> 45 deg pitch


def test_root_is_inferred(u):
    # base_link is the only link that is never a child -> the root.
    tr, rpy = u.link_pose_in_base(URDF, "camera_link")
    assert np.allclose(tr, [0.35, 0.0, 0.1])
    assert np.allclose(rpy, [0.0, 0.0, 0.0])


def test_two_hop_translation_composes(u):
    # base -> mount_plate (-0.1 z) -> dvl (-0.05 z) == -0.15 z.
    tr, rpy = u.link_pose_in_base(URDF, "dvl_link")
    assert np.allclose(tr, [0.0, 0.0, -0.15])


def test_rotation_composes_into_translation(u):
    # A yaw at the base rotates a child's x-offset into y. base(yaw 90) -> tip(x=1).
    urdf = """<robot name="r">
      <link name="base_link"/><link name="a"/><link name="tip"/>
      <joint name="j1" type="fixed"><parent link="base_link"/><child link="a"/>
        <origin xyz="0 0 0" rpy="0 0 1.5707963267948966"/></joint>
      <joint name="j2" type="fixed"><parent link="a"/><child link="tip"/>
        <origin xyz="1 0 0" rpy="0 0 0"/></joint>
    </robot>"""
    tr, rpy = u.link_pose_in_base(urdf, "tip")
    assert np.allclose(tr, [0.0, 1.0, 0.0], atol=1e-9)     # x rotated into +y
    assert np.allclose(rpy, [0.0, 0.0, 90.0], atol=1e-6)


def test_find_link_case_insensitive_candidates(u):
    assert u.find_link(URDF, ["nope", "SONAR_LINK"]) == "sonar_link"
    assert u.find_link(URDF, ["missing"]) is None


def test_sensor_mount_uses_default_candidates(u):
    m = u.sensor_mount(URDF, "sonar")
    assert m is not None and np.allclose(m[0], [0.3, 0.0, 0.3])


def test_sensor_mount_missing_returns_none(u):
    # no 'baro'/'pressure' link in this URDF
    assert u.sensor_mount(URDF, "baro") is None


def test_sensor_mount_or_falls_back(u):
    # camera present -> URDF wins
    tr, rpy = u.sensor_mount_or(URDF, "camera", (9, 9, 9), (1, 2, 3))
    assert np.allclose(tr, [0.35, 0.0, 0.1])
    # baro absent -> fallback returned verbatim
    tr, rpy = u.sensor_mount_or(URDF, "baro", (0.1, 0.2, 0.3), (0, 0, 0))
    assert tr == (0.1, 0.2, 0.3)


def test_sensor_mount_or_none_urdf_is_fallback(u):
    tr, rpy = u.sensor_mount_or(None, "sonar", (0.3, 0.0, 0.3), (0.0, 45.0, 0.0))
    assert tr == (0.3, 0.0, 0.3) and rpy == (0.0, 45.0, 0.0)


def test_malformed_urdf_returns_none(u):
    assert u.sensor_mount("<robot><broken", "sonar") is None
