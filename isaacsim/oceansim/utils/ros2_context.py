"""Shared, reference-counted rclpy lifecycle for OceanSim ROS2 components.

Several OceanSim components (e.g. the ``UW_Camera`` image publisher and the
``ROS2ControlReceiver``) live in the same Isaac Sim process and share a single
process-wide rclpy context. If each component called ``rclpy.init()`` /
``rclpy.shutdown()`` independently, one component closing would tear down the
context still in use by another.

Components should call :func:`acquire` when they set up their ROS2 resources and
:func:`release` when they close. rclpy is initialized on the first acquire and
shut down only once the last holder releases.

Note: ``rclpy`` is provided by the ``isaacsim.ros2.bridge`` extension, which must
be activated before the OceanSim extension (see the ROS2 Bridge docs).
"""

import threading

import rclpy

_refcount = 0
_lock = threading.Lock()


def acquire():
    """Ensure the shared rclpy context is initialized and register a holder."""
    global _refcount
    with _lock:
        if not rclpy.ok():
            rclpy.init()
        _refcount += 1


def release():
    """Release a holder, shutting down rclpy once the last holder is gone."""
    global _refcount
    with _lock:
        if _refcount > 0:
            _refcount -= 1
        if _refcount == 0 and rclpy.ok():
            rclpy.shutdown()
