"""Runtime URDF import for OceanSim (Isaac Sim only).

When a platform has only a URDF (no prebuilt USD), bring it onto the stage with
Isaac's URDF importer. A URDF gives more than a bare hull USD: it defines the
articulation (links + joints), so joint manipulation works out of the box and
the same file can feed ``/robot_description``.

Isaac's URDF importer API has shifted across releases, so this wrapper supports
both and is defensive about each:

  * Isaac 6.x (``isaacsim.asset.importer.urdf`` ~3.x) exposes a class-based
    URDF->USD *converter*: ``URDFImporter(URDFImporterConfig).import_urdf()``
    writes a USD on disk and returns its path. There is NO omni.kit.commands
    ``URDFParseAndImportFile`` / ``URDFCreateImportConfig`` -- calling them errors
    with "command ... wasn't registered". So we convert to a USD, then reference
    it onto the current stage.
  * Older builds (``omni.importer.urdf`` / early ``isaacsim.asset.importer.urdf``)
    used the omni.kit.commands API.

We try the new class converter first and fall back to the legacy command API; a
clear error is raised only if neither is available (then convert to USD offline
and point the platform's ``usd_subpath`` at the result).
"""


def _import_via_converter(urdf_path, fix_base, merge_fixed_joints, self_collision,
                          dest_prim_path, link_density):
    """Isaac 6.x class API: convert the URDF to a USD on disk, then reference it
    onto the current stage at ``dest_prim_path``. Returns the prim path, or None
    if this API isn't present (caller falls back to the legacy command API)."""
    try:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.asset.importer.urdf")
    except Exception:  # noqa: BLE001
        pass
    try:
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    except Exception:  # noqa: BLE001 - not this Isaac build
        return None

    import os
    import tempfile

    out_dir = tempfile.mkdtemp(prefix="oceansim_urdf_usd_")
    cfg = URDFImporterConfig(
        urdf_path=urdf_path,
        usd_path=out_dir,
        merge_fixed_joints=merge_fixed_joints,
        # Tri-state in this API: True pins the base, False makes it floating.
        fix_base=bool(fix_base),
        # Our platform URDFs carry visual meshes but no <collision>; generate
        # convex-hull collision from the visuals so the hull has collision geometry.
        collision_from_visuals=True,
        # Links with no <inertial> (our staged URDFs) otherwise get an invalid /
        # negative mass; a density lets PhysX compute valid mass+inertia from the
        # convex-hull colliders above. The caller can override the base body's mass
        # to the exact platform figure afterwards.
        link_density=link_density,
        allow_self_collision=self_collision,
        # Deterministic single-file output (skip the asset-transformer packaging).
        run_asset_transformer=False,
    )
    usd_file = URDFImporter(cfg).import_urdf()
    if not usd_file or not os.path.isfile(usd_file):
        raise RuntimeError(
            f"URDF->USD conversion produced no file for {urdf_path} (got {usd_file!r}).")

    from isaacsim.core.utils.stage import add_reference_to_stage
    add_reference_to_stage(usd_path=usd_file, prim_path=dest_prim_path)
    return dest_prim_path


def _import_via_commands(urdf_path, fix_base, merge_fixed_joints, self_collision,
                         distance_scale):
    """Legacy omni.kit.commands API (older Isaac). Returns the prim path, or None
    if the command API isn't available / the import didn't succeed."""
    import omni.kit.commands
    from isaacsim.core.utils.extensions import enable_extension

    ext = None
    for name in ("omni.importer.urdf", "isaacsim.asset.importer.urdf"):
        try:
            if enable_extension(name):
                ext = name
                break
        except Exception:  # noqa: BLE001 - try the next known name
            continue
    if ext is None:
        return None

    try:
        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    except Exception:  # noqa: BLE001
        return None
    if import_config is None:
        return None

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
    if not status or not prim_path:
        return None
    return prim_path


def import_urdf_to_stage(urdf_path, fix_base=False, merge_fixed_joints=True,
                         self_collision=False, distance_scale=1.0,
                         dest_prim_path="/World/rob", link_density=500.0):
    """Import ``urdf_path`` onto the current stage and return the prim path of the
    created robot (an articulation, unless all joints merge to a single body).

    Args:
        fix_base: pin the base link to the world. False for a free-floating ROV.
        merge_fixed_joints: collapse fixed joints (fewer links, same kinematics).
        self_collision: enable self-collision between the robot's own links.
        distance_scale: URDF length unit -> metres (legacy API only; 1.0 if metres).
        dest_prim_path: where to place the robot on the current stage.
        link_density: kg/m^3 used to compute mass+inertia for links that carry no
            <inertial> (new-API converter only); avoids the invalid/negative mass
            PhysX otherwise assigns. ~500 gives a sealed ROV hull a plausible mass;
            override the base body's mass to the exact platform figure separately.

    Tries the Isaac 6.x class converter first, then the legacy command API. Raises
    RuntimeError if neither works, so the caller can fall back to (or instruct the
    user toward) offline conversion.
    """
    prim = _import_via_converter(urdf_path, fix_base, merge_fixed_joints,
                                 self_collision, dest_prim_path, link_density)
    if prim is not None:
        return prim

    prim = _import_via_commands(urdf_path, fix_base, merge_fixed_joints,
                                self_collision, distance_scale)
    if prim is not None:
        return prim

    raise RuntimeError(
        f"No working URDF importer for {urdf_path} (tried the isaacsim.asset."
        f"importer.urdf class API and the legacy URDF commands). Convert the URDF "
        f"to USD offline and set the platform's usd_subpath instead.")
