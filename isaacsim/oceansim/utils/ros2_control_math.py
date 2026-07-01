"""Pure (numpy-only) math for the OceanSim ROS2 control receiver.

Deliberately free of ``rclpy`` / Isaac Sim / ``pxr`` imports so it is unit
testable without ROS or Isaac Sim, mirroring ``ros2_math.py``'s split for the
sensor publishers. ``ros2_control.py`` calls these helpers rather than
re-implementing the logic inline.
"""

import numpy as np


def clamp_magnitude(vec, max_mag):
    """Clamp a 3-vector's magnitude to max_mag, preserving direction.

    max_mag=None disables clamping -- vec is returned unchanged (as a plain
    list of floats, matching the clamped-path return shape).
    """
    if max_mag is None:
        return [float(vec[0]), float(vec[1]), float(vec[2])]
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm > max_mag and norm > 0.0:
        arr = arr * (max_mag / norm)
    return arr.tolist()


def is_command_stale(now, last_command_time, timeout):
    """True if more than timeout seconds have elapsed since last_command_time.

    Dead-man's-switch check for the vel/force command receiver: a lost ROS2
    link (crashed teleop, dropped session, network partition) must not leave
    the vehicle actuating its last received command forever.
    """
    return (now - last_command_time) > timeout
