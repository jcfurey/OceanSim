"""Runtime URDF import for OceanSim (Isaac Sim only).

When a platform has only a URDF (no prebuilt USD), bring it onto the stage with
Isaac's URDF importer. A URDF gives more than a bare hull USD: it defines the
articulation (links + joints), so joint manipulation works out of the box and
the same file can feed ``/robot_description``.

This is a thin, defensive wrapper over the importer extension. The command /
config API has shifted across Isaac releases (``isaacsim.asset.importer.urdf``
vs the older ``omni.importer.urdf``), so the calls are guarded and a clear error
is raised if the importer is unavailable -- in which case the reliable fallback
is to convert the URDF to USD once (the importer UI or CLI) and point the
platform's ``usd_subpath`` at the result.

EXPERIMENTAL: the import path can only be fully validated on an Isaac Sim
install; the runner keeps the well-trodden USD path as the default.
"""


def _enable_importer():
    from isaacsim.core.utils.extensions import enable_extension
    for ext in ("isaacsim.asset.importer.urdf", "omni.importer.urdf"):
        try:
            if enable_extension(ext):
                return ext
        except Exception:  # noqa: BLE001 - try the next known name
            continue
    return None


def import_urdf_to_stage(urdf_path, fix_base=False, merge_fixed_joints=True,
                         self_collision=False, distance_scale=1.0):
    """Import ``urdf_path`` onto the current stage and return the prim path of
    the created robot (an articulation).

    Args:
        fix_base: pin the base link to the world. False for a free-floating ROV.
        merge_fixed_joints: collapse fixed joints (fewer links, same kinematics).
        self_collision: enable self-collision between the robot's own links.
        distance_scale: URDF length unit -> metres (1.0 if the URDF is in metres).

    Raises RuntimeError if the importer extension or command is unavailable, so
    the caller can fall back to (or instruct the user toward) offline conversion.
    """
    import omni.kit.commands

    ext = _enable_importer()
    if ext is None:
        raise RuntimeError(
            "URDF importer extension not available (tried isaacsim.asset.importer.urdf "
            "and omni.importer.urdf). Convert the URDF to USD offline and point the "
            "platform's usd_subpath at it instead.")

    try:
        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"URDFCreateImportConfig failed ({e}); see offline conversion.")

    # Best-effort config; attribute names have been stable but guard each so a
    # renamed field doesn't abort the whole import.
    for attr, val in (("merge_fixed_joints", merge_fixed_joints),
                      ("fix_base", fix_base),
                      ("self_collision", self_collision),
                      ("make_default_prim", False),
                      ("distance_scale", distance_scale),
                      ("import_inertia_tensor", True)):
        try:
            setattr(import_config, attr, val)
        except Exception:  # noqa: BLE001
            pass

    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=urdf_path, import_config=import_config)
    if not prim_path:
        raise RuntimeError(f"URDF import produced no prim for {urdf_path}")
    return prim_path
