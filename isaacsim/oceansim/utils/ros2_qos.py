"""Pure QoS profiles + ROS2 compatibility rules for OceanSim's ROS topics.

Two endpoints connect only if the publisher's OFFERED QoS satisfies the
subscriber's REQUESTED QoS. A mismatch is a *silent* failure -- no error, just no
messages -- which is exactly the kind of two-way pub/sub bug that is easy to ship
and hard to debug.

So the QoS lives here as plain data: ``ros2_sensors`` builds its rclpy
``QoSProfile`` objects from these named profiles (single source of truth), and
``TOPIC_CONTRACTS`` records, for every OceanSim topic, the QoS on our side and on
the peer it must talk to (RViz, robot_state_publisher, robot_localization,
teleop, ...). ``verify_contract`` then checks each pairing -- so the whole
publisher/subscriber graph is verified in CI without a ROS runtime.

No rclpy import here, so it runs in plain pytest.
"""

from collections import namedtuple

# reliability
RELIABLE = "reliable"
BEST_EFFORT = "best_effort"
# durability
VOLATILE = "volatile"
TRANSIENT_LOCAL = "transient_local"

QoS = namedtuple("QoS", ["reliability", "durability", "depth"])

# Named profiles (mirrored by ros2_sensors' QoS helpers).
SENSOR_DATA = QoS(BEST_EFFORT, VOLATILE, 10)    # odom/imu/dvl/baro/sonar, joint_command
CLOCK = QoS(RELIABLE, VOLATILE, 1)              # /clock (reliable offer satisfies any sub)
LATCHED = QoS(RELIABLE, TRANSIENT_LOCAL, 1)     # /robot_description (RViz / rsp expect latched)
RELIABLE_STATE = QoS(RELIABLE, VOLATILE, 10)    # /joint_states (robot_state_publisher is RELIABLE)
DEFAULT_CMD = QoS(RELIABLE, VOLATILE, 10)       # rclpy-default command pubs/subs (cmd_vel, etc.)


def is_compatible(offered, requested):
    """ROS2 QoS compatibility: can a publisher offering ``offered`` deliver to a
    subscriber requesting ``requested``?

    Reliability is ordered RELIABLE > BEST_EFFORT, durability TRANSIENT_LOCAL >
    VOLATILE; the offer must be at least as strong as the request on BOTH axes.
    (History/depth do not affect whether a connection forms.)
    """
    rel_ok = offered.reliability == RELIABLE or requested.reliability == BEST_EFFORT
    dur_ok = offered.durability == TRANSIENT_LOCAL or requested.durability == VOLATILE
    return rel_ok and dur_ok


# role: "pub" -> OceanSim publishes (we offer, peer requests);
#       "sub" -> OceanSim subscribes (peer offers, we request).
Contract = namedtuple("Contract", ["topic", "role", "ours", "peer", "peer_qos"])

TOPIC_CONTRACTS = [
    Contract("/clock", "pub", CLOCK, "use_sim_time consumers", SENSOR_DATA),
    Contract("/oceansim/robot/odom", "pub", SENSOR_DATA, "robot_localization", SENSOR_DATA),
    Contract("/oceansim/robot/imu", "pub", SENSOR_DATA, "robot_localization", SENSOR_DATA),
    Contract("/oceansim/robot/dvl/twist", "pub", SENSOR_DATA, "robot_localization", SENSOR_DATA),
    Contract("/oceansim/robot/pressure", "pub", SENSOR_DATA, "depth consumer", SENSOR_DATA),
    Contract("/oceansim/robot/sonar", "pub", SENSOR_DATA, "sonar_image_proc", SENSOR_DATA),
    Contract("/robot_description", "pub", LATCHED, "robot_state_publisher / RViz", LATCHED),
    # robot_state_publisher subscribes to joint_states with the rclcpp default
    # (RELIABLE) -- a BEST_EFFORT joint_states publisher would silently never be
    # received, so OceanSim publishes it RELIABLE.
    Contract("/joint_states", "pub", RELIABLE_STATE, "robot_state_publisher", DEFAULT_CMD),
    Contract("/oceansim/robot/joint_command", "sub", SENSOR_DATA, "controller / teleop", DEFAULT_CMD),
    Contract("/oceansim/robot/vel_cmd", "sub", DEFAULT_CMD, "teleop_twist_*", DEFAULT_CMD),
    Contract("/oceansim/robot/force_cmd", "sub", DEFAULT_CMD, "controller", DEFAULT_CMD),
    # UW_Camera streams (multi-MB images): BEST_EFFORT like every other sensor, so
    # they don't apply RELIABLE publisher-side backpressure on the render thread.
    Contract("/oceansim/robot/uw_img", "pub", SENSOR_DATA, "image consumer / RViz", SENSOR_DATA),
    Contract("/oceansim/robot/image_raw", "pub", SENSOR_DATA, "image_proc / RViz", SENSOR_DATA),
    Contract("/oceansim/robot/depth", "pub", SENSOR_DATA, "depth_image_proc / RViz", SENSOR_DATA),
    Contract("/oceansim/robot/camera_info", "pub", SENSOR_DATA, "image_proc / RViz", SENSOR_DATA),
]


def verify_contract(c):
    """True if OceanSim's endpoint and its peer can actually communicate."""
    if c.role == "pub":
        return is_compatible(c.ours, c.peer_qos)        # we offer, peer requests
    return is_compatible(c.peer_qos, c.ours)            # peer offers, we request


def verify_all():
    """Return the list of contracts whose two endpoints are QoS-incompatible
    (empty == the whole pub/sub graph is wired up correctly)."""
    return [c for c in TOPIC_CONTRACTS if not verify_contract(c)]
