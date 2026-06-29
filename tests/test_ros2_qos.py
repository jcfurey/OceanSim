"""Verification of OceanSim's two-way ROS pub/sub via QoS compatibility.

A publisher and subscriber only exchange messages if their QoS is compatible;
a mismatch fails silently. These tests check the ROS2 compatibility rules and
assert every OceanSim topic contract (our endpoint vs its peer) is satisfiable.
Pure -- no rclpy needed.
"""

import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                     "utils", "ros2_qos.py")


def _load():
    spec = importlib.util.spec_from_file_location("ros2_qos", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def q():
    return _load()


# --- the compatibility rule (offered vs requested) -------------------------

def test_reliability_rule(q):
    rel = q.QoS(q.RELIABLE, q.VOLATILE, 1)
    be = q.QoS(q.BEST_EFFORT, q.VOLATILE, 1)
    assert q.is_compatible(rel, rel)        # reliable -> reliable
    assert q.is_compatible(rel, be)         # reliable offer satisfies best-effort req
    assert q.is_compatible(be, be)          # best-effort -> best-effort
    assert not q.is_compatible(be, rel)     # best-effort offer CANNOT satisfy reliable req


def test_durability_rule(q):
    vol = q.QoS(q.RELIABLE, q.VOLATILE, 1)
    tl = q.QoS(q.RELIABLE, q.TRANSIENT_LOCAL, 1)
    assert q.is_compatible(tl, tl)          # latched -> latched
    assert q.is_compatible(tl, vol)         # latched offer satisfies volatile req
    assert q.is_compatible(vol, vol)        # volatile -> volatile
    assert not q.is_compatible(vol, tl)     # volatile offer CANNOT satisfy latched req


def test_both_axes_must_pass(q):
    # incompatible on durability even though reliability is fine.
    offered = q.QoS(q.RELIABLE, q.VOLATILE, 1)
    requested = q.QoS(q.RELIABLE, q.TRANSIENT_LOCAL, 1)
    assert not q.is_compatible(offered, requested)


# --- the actual OceanSim topic graph ---------------------------------------

def test_all_topic_contracts_compatible(q):
    bad = q.verify_all()
    assert bad == [], ("QoS-incompatible topic(s): "
                       + ", ".join(f"{c.topic} (ours={c.ours}, peer={c.peer_qos})" for c in bad))


def test_every_contract_is_individually_verified(q):
    for c in q.TOPIC_CONTRACTS:
        assert q.verify_contract(c), f"{c.topic} incompatible with {c.peer}"


def test_robot_description_is_latched(q):
    # RViz / robot_state_publisher need TRANSIENT_LOCAL to receive a description
    # published once before they start.
    c = next(c for c in q.TOPIC_CONTRACTS if c.topic == "/robot_description")
    assert c.ours.durability == q.TRANSIENT_LOCAL


def test_joint_states_reliable_for_robot_state_publisher(q):
    # Regression guard for the real bug: a BEST_EFFORT joint_states publisher is
    # NOT received by robot_state_publisher (which requests RELIABLE).
    c = next(c for c in q.TOPIC_CONTRACTS if c.topic == "/joint_states")
    assert c.ours.reliability == q.RELIABLE
    # prove the BEST_EFFORT alternative would have been incompatible
    assert not q.is_compatible(q.SENSOR_DATA, c.peer_qos)


def test_no_topic_contract_drift(q):
    """Guard: every OceanSim topic literal in the ROS source must have a verified
    QoS contract, and every contract must still correspond to a real topic. So a
    topic added (or removed) without updating ros2_qos.TOPIC_CONTRACTS fails CI."""
    import re
    base = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim")
    pat = re.compile(r'"(/[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*)"')
    sources = [os.path.join(base, "utils", "ros2_sensors.py"),
               os.path.join(base, "utils", "ros2_control.py"),
               os.path.join(base, "sensors", "UW_Camera.py")]
    src_topics = set()
    for path in sources:
        with open(path) as f:
            src_topics |= set(pat.findall(f.read()))
    contract_topics = {c.topic for c in q.TOPIC_CONTRACTS}
    assert src_topics - contract_topics == set(), \
        f"topics used in source but missing a QoS contract: {src_topics - contract_topics}"
    assert contract_topics - src_topics == set(), \
        f"QoS contracts for topics not found in source (stale?): {contract_topics - src_topics}"


def test_sensor_topics_require_best_effort_consumers(q):
    # Document the contract: sensor streams are BEST_EFFORT, so a RELIABLE-only
    # consumer would drop them. (Verified compatible against BEST_EFFORT peers.)
    sonar = next(c for c in q.TOPIC_CONTRACTS if c.topic == "/oceansim/robot/sonar")
    assert sonar.ours.reliability == q.BEST_EFFORT
    assert not q.is_compatible(sonar.ours, q.QoS(q.RELIABLE, q.VOLATILE, 10))
