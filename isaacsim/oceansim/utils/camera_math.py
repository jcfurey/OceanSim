"""Pure (numpy-only) camera math for UW_Camera.

No Isaac Sim / ROS imports, so it is unit testable in isolation. UW_Camera builds
the sensor_msgs/CameraInfo and the depth Image from the values computed here.
"""

import numpy as np


def pinhole_intrinsics(width, height, focal, h_aper, v_aper):
    """Pinhole intrinsics + ROS CameraInfo K/R/P/D from the camera aperture.

    fx = width * focal / horizontal_aperture (fy analogously); principal point at
    the image centre; no distortion. Falls back to fx = width if an aperture is 0.
    """
    width = int(width)
    height = int(height)
    fx = (width * focal / h_aper) if h_aper else float(width)
    fy = (height * focal / v_aper) if v_aper else fx
    cx, cy = width / 2.0, height / 2.0
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "distortion_model": "plumb_bob",
        "d": [0.0, 0.0, 0.0, 0.0, 0.0],
        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    }


def radial_to_planar_factor(height, width, fx, fy, cx, cy):
    """Per-pixel factor converting Isaac ``distance_to_camera`` (radial / ray
    length) to the planar z-depth ROS depth consumers expect:

        planar_z(u,v) = radial * 1/sqrt(((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1)

    The factor is static for fixed intrinsics, so callers cache it.
    """
    u = (np.arange(width, dtype=np.float32) - cx) / fx
    v = (np.arange(height, dtype=np.float32) - cy) / fy
    uu, vv = np.meshgrid(u, v)
    return (1.0 / np.sqrt(uu * uu + vv * vv + 1.0)).astype(np.float32)
