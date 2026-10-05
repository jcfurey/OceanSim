# Convenience re-exports. Guarded per sensor: UW_Camera imports rclpy/cv2 at
# module load (needs isaacsim.ros2.bridge active), and an eager failure here
# would otherwise break importing ANY isaacsim.oceansim.sensors.* submodule.
def _warn_optional_import(name, exc):
    msg = f"[isaacsim.oceansim.sensors] optional import '{name}' unavailable: {exc!r}"
    try:
        import carb
        carb.log_warn(msg)
    except ImportError:
        print(msg)

try:
    from .BarometerSensor import BarometerSensor
except Exception as _exc:  # noqa: BLE001
    _warn_optional_import("BarometerSensor", _exc)
try:
    from .DVLsensor import DVLsensor
except Exception as _exc:  # noqa: BLE001
    _warn_optional_import("DVLsensor", _exc)
try:
    from .ImagingSonarSensor import ImagingSonarSensor
except Exception as _exc:  # noqa: BLE001
    _warn_optional_import("ImagingSonarSensor", _exc)
try:
    from .UW_Camera import UW_Camera
except Exception as _exc:  # noqa: BLE001
    _warn_optional_import("UW_Camera", _exc)
