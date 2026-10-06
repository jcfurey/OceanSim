"""Validate the bundled scene geometry and asset portability using OpenUSD."""
from pathlib import Path

import pytest

pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

ASSETS = Path(__file__).resolve().parents[1] / "demo/assets"


@pytest.mark.parametrize("name", ["inspection_site", "rocky_reef"])
def test_bundled_scene_is_portable_and_physical(name):
    path = ASSETS / "environments" / f"{name}.usdc"
    stage = Usd.Stage.Open(str(path))
    assert stage.GetDefaultPrim()
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetDefaultPrim().GetCustomDataByKey("oceansim:waterSurface") == 0.0
    _, dependencies, missing = UsdUtils.ComputeAllDependencies(str(path))
    assert not missing
    assert dependencies  # the PBR materials actually reference bundled textures
    meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert len(meshes) > 50
    for mesh in meshes:
        points = mesh.GetPointsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        assert min(indices) >= 0
        assert max(indices) < len(points)
        assert sum(mesh.GetFaceVertexCountsAttr().Get()) == len(indices)
        assert mesh.GetPrim().GetAttribute("oceansim:reflectivity").Get() > 0
    floor = stage.GetPrimAtPath(str(stage.GetDefaultPrim().GetPath()) + "/Seabed")
    assert floor.HasAPI(UsdPhysics.CollisionAPI)


def test_spawn_corridor_has_clearance_above_seabed():
    stage = Usd.Stage.Open(str(ASSETS / "environments/inspection_site.usdc"))
    floor = UsdGeom.Mesh(stage.GetPrimAtPath("/InspectionSite/Seabed"))
    # Both vehicle and down-looking DVL have over 1 m clearance at the default pose.
    assert max(p[2] for p in floor.GetPointsAttr().Get()) < -2.4
    station = stage.GetPrimAtPath("/InspectionSite/Station/Panel")
    assert station.HasAPI(UsdPhysics.CollisionAPI)
