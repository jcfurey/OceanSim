"""Pure (numpy-only) math for the OceanSim ROS2 control receiver.

Deliberately free of ``rclpy`` / Isaac Sim / ``pxr`` imports so it is unit
testable without ROS or Isaac Sim, mirroring ``ros2_math.py``'s split for the
sensor publishers. ``ros2_control.py`` calls these helpers rather than
re-implementing the logic inline.
"""

import numpy as np


def clamp_magnitude(vec, max_mag):
    """Clamp a 3-vector's magnitude to max_mag, preserving direction.

    max_mag=None disables clamping -- a finite vec is returned unchanged (as a
    plain list of floats, matching the clamped-path return shape).

    A non-finite command (any NaN/inf component) is rejected to zeros in BOTH
    modes: with clamping on, ``norm=inf`` made ``max_mag/norm`` collapse the
    finite components and turn the infinite one into ``0*inf=NaN``, and a NaN
    component made ``norm > max_mag`` False so NaN passed through "unclamped"
    -- either way garbage reached the thrusters. Zero (the dead-man behavior)
    is the only safe output for an invalid command.
    """
    arr = np.asarray(vec, dtype=float)
    if not np.all(np.isfinite(arr)):
        return [0.0, 0.0, 0.0]
    if max_mag is None:
        return [float(arr[0]), float(arr[1]), float(arr[2])]
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
