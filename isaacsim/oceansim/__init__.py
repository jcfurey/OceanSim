# Modules are UI extensions and have to be imported separately for UI to work properly
# import omni.kit.app

# app = omni.kit.app.get_app()
# ext_manager = app.get_extension_manager()
# ext_id = "isaacsim.oceansim"

# # Toggle extension state
# if ext_manager.is_extension_enabled(ext_id):
#     ext_manager.set_extension_enabled(ext_id, False)
#     print(f"EXTENSION {ext_id} DEACTIVATED")
# else:
#     ext_manager.set_extension_enabled(ext_id, True)
#     print(f"EXTENSION {ext_id} ACTIVATED")
#
# Convenience re-exports. Each is optional: the sensors pull in rclpy/cv2 (needs
# isaacsim.ros2.bridge loaded) and the writers need isaacsim.replicator.writers,
# and this package is loaded as an extension python.module -- so a missing
# optional dependency must log a warning, not fail the whole extension's
# startup. Code that needs a submodule imports it directly anyway.
def _warn_optional_import(name, exc):
    msg = f"[isaacsim.oceansim] optional import '{name}' unavailable: {exc!r}"
    try:
        import carb
        carb.log_warn(msg)
    except ImportError:
        print(msg)

try:
    from .sensors import *
except Exception as _exc:  # noqa: BLE001
    _warn_optional_import("sensors", _exc)
try:
    from .utils import *
except Exception as _exc:  # noqa: BLE001
    _warn_optional_import("utils", _exc)
try:
    from .watersurface import *
except Exception as _exc:  # noqa: BLE001
    _warn_optional_import("watersurface", _exc)
try:
    from .writers import UWCam_KittiWriter
except Exception as _exc:  # noqa: BLE001
    _warn_optional_import("writers", _exc)

__all__ = [
    "sensors",
    "utils",
    "watersurface",
    "writers",
]

