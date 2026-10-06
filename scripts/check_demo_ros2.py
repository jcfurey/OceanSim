#!/usr/bin/env python3
"""Check live demo ROS topics and save camera/sonar frames (no robot commands)."""
import argparse
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--output", type=Path, default=Path(".local/previews"))
    parser.add_argument("--clock-only", action="store_true", help="Readiness check for container health.")
    args = parser.parse_args()
    import cv2
    import numpy as np
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from geometry_msgs.msg import TwistWithCovarianceStamped
    from marine_acoustic_msgs.msg import ProjectedSonarImage
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CompressedImage, FluidPressure, Imu, JointState
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("oceansim_demo_check")
    messages = {}
    counts = {}
    topics = {
        "/clock": Clock,
        "/robot_description": String,
        "/joint_states": JointState,
        "/oceansim/robot/odom": Odometry,
        "/oceansim/robot/imu": Imu,
        "/oceansim/robot/dvl/twist": TwistWithCovarianceStamped,
        "/oceansim/robot/pressure": FluidPressure,
        "/oceansim/robot/sonar": ProjectedSonarImage,
        "/oceansim/robot/uw_img": CompressedImage,
    }
    if args.clock_only:
        topics = {"/clock": Clock}

    def receive(topic, message):
        messages[topic] = message
        counts[topic] = counts.get(topic, 0) + 1

    try:
        for topic, kind in topics.items():
            qos = (QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                   if kind is String else qos_profile_sensor_data)
            node.create_subscription(kind, topic, lambda msg, topic=topic: receive(topic, msg), qos)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and len(messages) < len(topics):
            rclpy.spin_once(node, timeout_sec=0.2)
        missing = set(topics) - set(messages)
        for topic in topics:
            print(f"{topic}: {counts.get(topic, 0)} messages")
        if missing:
            raise RuntimeError(f"No messages received on {sorted(missing)}; match ROS_DOMAIN_ID and RMW_IMPLEMENTATION")
        if args.clock_only:
            return
        odom = messages["/oceansim/robot/odom"].pose.pose.position
        assert all(np.isfinite([odom.x, odom.y, odom.z])), "Vehicle pose is invalid"
        assert odom.z > -2, "Vehicle sank or spawned inside the seabed"
        dvl = messages["/oceansim/robot/dvl/twist"].twist.twist.linear
        assert all(np.isfinite([dvl.x, dvl.y, dvl.z])), "DVL output is invalid"
        assert messages["/oceansim/robot/pressure"].fluid_pressure > 101325, "Pressure does not indicate submersion"
        camera = cv2.imdecode(np.frombuffer(bytes(messages["/oceansim/robot/uw_img"].data), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert camera is not None and camera.std() > 1, "Camera returned an empty or uniform image"
        sonar = messages["/oceansim/robot/sonar"]
        bins = np.frombuffer(bytes(sonar.image.data), dtype=np.uint8).reshape(len(sonar.ranges), len(sonar.beam_directions))
        assert np.count_nonzero(bins) > 0, "Sonar returned no intensity"
        args.output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output / "underwater_camera.jpg"), camera)
        cv2.imwrite(str(args.output / "sonar.png"), bins)
        joints = messages["/joint_states"]
        print(f"Pose: ({odom.x:.3f}, {odom.y:.3f}, {odom.z:.3f}); joints: {list(joints.name)}")
        print(f"Camera: {camera.shape[1]}x{camera.shape[0]}; sonar: {bins.shape}, {np.count_nonzero(bins)} nonzero bins")
        print(f"ROS demo check passed; captures saved to {args.output}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
