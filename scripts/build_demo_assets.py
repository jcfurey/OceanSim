#!/usr/bin/env python3
"""Build portable, seeded OpenUSD demo scenes and optionally stage REVOLUTION CAD.

Requires numpy, Pillow and usd-core (or Isaac Sim's OpenUSD environment).
The generated environments need no network access or proprietary asset pack.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parents[1]
SEED = 24


def load_util(name):
    path = ROOT / "isaacsim/oceansim/utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_textures(directory):
    """Tileable silt ripples and mottled corrosion; all textures are original."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    n = 512
    y, x = np.mgrid[:n, :n] / n * (2 * math.pi)
    broad = np.sin(3*x + np.sin(2*y)) + 0.5*np.cos(5*y - 2*x)
    grain = rng.normal(0, 0.025, (n, n))
    ripple = np.sin(24*y + 2*np.sin(2*x) + 0.4*np.cos(3*x))
    sand = 0.055*broad + 0.04*ripple + grain
    rust = np.clip(0.5 + 0.23*broad + 0.08*np.sin(17*x + 11*y) + grain, 0, 1)
    colours = {
        "silt": np.array([0.43, 0.40, 0.31]) + sand[..., None],
        "rock": np.array([0.28, 0.32, 0.29]) + (0.06*broad + grain)[..., None],
        "rust": (1-rust[..., None])*np.array([0.16, 0.23, 0.22])
                + rust[..., None]*np.array([0.48, 0.23, 0.095]),
    }
    for name, rgb in colours.items():
        Image.fromarray((np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(directory / f"{name}.png")
    # Tangent-space normals derived from the same periodic height function.
    dx = (np.roll(sand, -1, axis=1) - np.roll(sand, 1, axis=1))*2
    dy = (np.roll(sand, -1, axis=0) - np.roll(sand, 1, axis=0))*2
    normal = np.stack([-dx, -dy, np.ones_like(dx)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    Image.fromarray(((normal*0.5 + 0.5)*255).astype(np.uint8)).save(directory / "silt_normal.png")


class Scene:
    def __init__(self, path, name):
        self.stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.root = f"/{name}"
        root = UsdGeom.Xform.Define(self.stage, self.root).GetPrim()
        self.stage.SetDefaultPrim(root)
        root.SetCustomDataByKey("oceansim:seed", SEED)
        root.SetCustomDataByKey("oceansim:waterSurface", 0.0)
        self.materials = {}
        for key, colour, metal, rough, texture in (
            ("Silt", (0.43, 0.40, 0.31), 0.0, 0.94, "silt"),
            ("Rock", (0.28, 0.32, 0.29), 0.0, 0.87, "rock"),
            ("CorrodedSteel", (0.34, 0.24, 0.13), 0.45, 0.72, "rust"),
            ("Graphite", (0.075, 0.095, 0.11), 0.35, 0.29, None),
            ("Titanium", (0.43, 0.48, 0.51), 0.85, 0.28, None),
            ("SafetyYellow", (0.86, 0.56, 0.065), 0.25, 0.43, None),
            ("EnamelBlue", (0.035, 0.19, 0.25), 0.35, 0.32, None),
            ("White", (0.83, 0.84, 0.74), 0.0, 0.5, None),
            ("Algae", (0.12, 0.21, 0.10), 0.0, 0.96, None),
        ):
            self.materials[key] = self.material(key, colour, metal, rough, texture)

    def material(self, name, colour, metallic, roughness, texture=None):
        path = f"{self.root}/Looks/{name}"
        mat = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, path + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        if texture:
            reader = UsdShade.Shader.Define(self.stage, path + "/UV")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            for suffix, file, target, colourspace in (
                ("Albedo", texture + ".png", "diffuseColor", "sRGB"),
                *(([("Normal", "silt_normal.png", "normal", "raw")]) if texture == "silt" else []),
            ):
                tex = UsdShade.Shader.Define(self.stage, path + "/" + suffix)
                tex.CreateIdAttr("UsdUVTexture")
                tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("../textures/" + file))
                tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colourspace)
                tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
                if target == "normal":
                    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                    tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
                shader.CreateInput(target, Sdf.ValueTypeNames.Normal3f if target == "normal"
                                   else Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        return mat

    def finish_prim(self, prim, material, collide, reflectivity):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.materials[material])
        prim.CreateAttribute("oceansim:reflectivity", Sdf.ValueTypeNames.Float).Set(reflectivity)
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim)
        return prim

    def cube(self, name, pos, size, material="CorrodedSteel", rotate=(0, 0, 0), collide=True):
        cube = UsdGeom.Cube.Define(self.stage, self.root + "/" + name)
        cube.CreateSizeAttr(1)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*rotate))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        self.finish_prim(cube.GetPrim(), material, collide, 1.6)
        return cube

    def mesh(self, name, points, faces, material, uv=None, collide=True, reflectivity=1.0):
        mesh = UsdGeom.Mesh.Define(self.stage, self.root + "/" + name)
        mesh.CreatePointsAttr([tuple(map(float, p)) for p in points])
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([int(i) for f in faces for i in f])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
        if uv is not None:
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                   UsdGeom.Tokens.vertex).Set(uv)
        self.finish_prim(mesh.GetPrim(), material, collide, reflectivity)
        return mesh

    def tube(self, name, start, end, radius, material="CorrodedSteel", segments=32, collide=True):
        start, end = np.asarray(start, float), np.asarray(end, float)
        axis = end-start
        axis /= np.linalg.norm(axis)
        u = np.cross(axis, (0, 0, 1) if abs(axis[2]) < 0.9 else (0, 1, 0))
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        points, uv = [], []
        for i, centre in enumerate((start, end)):
            for j in range(segments+1):
                a = j*2*math.pi/segments
                points.append(centre + radius*(u*math.cos(a)+v*math.sin(a)))
                uv.append((j/segments, i*np.linalg.norm(end-start)))
        faces = [(j, j+1, j+segments+2, j+segments+1) for j in range(segments)]
        faces += [tuple(reversed(range(segments))), tuple(range(segments+1, 2*segments+1))]
        return self.mesh(name, points, faces, material, uv, collide, 2.0)

    def ring(self, name, centre, major, minor, material="CorrodedSteel", axis="X"):
        points, faces, uv = [], [], []
        for i in range(49):
            a = i*2*math.pi/48
            for j in range(9):
                b = j*2*math.pi/8
                p = ((major+minor*math.cos(b))*math.cos(a),
                     (major+minor*math.cos(b))*math.sin(a), minor*math.sin(b))
                p = (p[2], p[0], p[1]) if axis == "X" else p
                points.append(np.asarray(centre)+p)
                uv.append((i/48, j/8))
        for i in range(48):
            for j in range(8):
                k = i*9+j
                faces.append((k, k+9, k+10, k+1))
        return self.mesh(name, points, faces, material, uv, False, 2.0)

    def rock(self, name, pos, size, rng):
        points, uv, faces = [], [], []
        # Smooth enough for highlights; irregular silhouette instead of ellipsoids.
        for i in range(13):
            phi = 0.025 + i*(math.pi-0.05)/12
            for j in range(25):
                theta = j*2*math.pi/24
                r = 1 + 0.12*math.sin(theta*5 + phi*3) + 0.06*math.cos(theta*3-phi*7)
                p = np.array([r*math.sin(phi)*math.cos(theta),
                              r*math.sin(phi)*math.sin(theta), r*math.cos(phi)])
                points.append(np.array(pos)+p*np.array(size))
                uv.append((j/24*2, i/12*2))
        for i in range(12):
            for j in range(24):
                k = i*25+j
                faces.append((k, k+25, k+26, k+1))
        mesh = self.mesh(name, points, faces, "Rock", uv, True, 0.65)
        # Static convex hull is cheaper and more robust than tiny rock triangles.
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")

    def terrain(self):
        n = 97
        points, uv, faces = [], [], []
        for j in range(n):
            y = -12 + j*24/(n-1)
            for i in range(n):
                x = -12 + i*30/(n-1)
                z = seabed_height(x, y)
                points.append((x, y, z))
                uv.append((x/2, y/2))
        for j in range(n-1):
            for i in range(n-1):
                k = j*n+i
                faces.append((k, k+1, k+n+1, k+n))
        self.mesh("Seabed", points, faces, "Silt", uv, True, 0.35)

    def lighting(self):
        dome = UsdLux.DomeLight.Define(self.stage, self.root + "/Lighting/Ambient")
        dome.CreateColorAttr(Gf.Vec3f(0.44, 0.68, 0.73))
        dome.CreateIntensityAttr(320)
        key = UsdLux.DistantLight.Define(self.stage, self.root + "/Lighting/SurfaceLight")
        key.CreateColorAttr(Gf.Vec3f(0.68, 0.84, 0.86))
        key.CreateIntensityAttr(1700)
        key.CreateAngleAttr(8)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(22, -35, -25))
        self.camera("Overview", (-7.5, -8, 3.5), (2, 0, -1.6))
        self.camera("Inspection", (-2, 0, -0.75), (3, 0, -1.4))

    def camera(self, name, eye, target):
        cam = UsdGeom.Camera.Define(self.stage, self.root + "/Cameras/" + name)
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())
        cam.CreateFocalLengthAttr(23)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 150))

    def save(self):
        self.stage.GetRootLayer().Save()


def seabed_height(x, y):
    # Keep the launch/inspection corridor clear while the outskirts undulate.
    hills = 0.12*math.sin(0.6*x)*math.cos(0.7*y)
    return -2.75 + hills + 0.018*math.sin(7*y + math.sin(x))


def inspection_site(path):
    scene = Scene(path, "InspectionSite")
    scene.terrain()
    scene.lighting()
    scene.tube("Pipeline/Main", (-3, 1.6, -2.2), (11, 1.6, -2.2), 0.19)
    for i, x in enumerate((-1, 2.5, 6, 9)):
        scene.tube(f"Pipeline/Flange{i}", (x-0.055, 1.6, -2.2), (x+0.055, 1.6, -2.2), 0.28, "Titanium")
        scene.ring(f"Pipeline/Weld{i}", (x+0.12, 1.6, -2.2), 0.19, 0.012)
        for j in range(8):
            a = 2*math.pi*j/8
            y, z = 1.6+0.235*math.cos(a), -2.2+0.235*math.sin(a)
            scene.tube(f"Pipeline/Bolt{i}_{j}", (x-0.085, y, z), (x+0.085, y, z), 0.019, "Graphite", 8, False)
        scene.cube(f"Pipeline/Support{i}", (x, 1.6, -2.56), (0.5, 0.7, 0.28), "Rock")
    scene.tube("Pipeline/Riser", (6, 1.6, -2.2), (6, 1.6, -0.75), 0.14)
    scene.ring("Pipeline/ValveWheel", (6, 1.6, -0.65), 0.28, 0.022, "SafetyYellow", "Z")
    for i in range(4):
        a = math.pi*i/2
        scene.tube(f"Pipeline/ValveSpoke{i}", (6, 1.6, -0.65),
                   (6+0.26*math.cos(a), 1.6+0.26*math.sin(a), -0.65), 0.013, "SafetyYellow", 12, False)
    # Inspection frame: two full-height posts, a calibration panel and grab handles.
    for y in (-1.0, 1.0):
        side = "Port" if y > 0 else "Starboard"
        scene.cube(f"Station/{side}Post", (2.4, y, -1.8), (0.12, 0.12, 1.7), "SafetyYellow")
        scene.cube(f"Station/{side}Foot", (2.4, y, -2.62), (0.6, 0.5, 0.12), "CorrodedSteel")
    scene.cube("Station/TopBeam", (2.4, 0, -0.95), (0.12, 2.12, 0.12), "SafetyYellow")
    scene.cube("Station/Panel", (2.38, 0, -1.65), (0.08, 1.35, 0.8), "EnamelBlue")
    # A high-contrast checker target helps see camera focus and scale (10 cm cells).
    for i in range(5):
        for j in range(5):
            scene.cube(f"Station/Checker{i}_{j}", (2.331, -0.24+i*0.12, -1.65-0.24+j*0.12),
                       (0.012, 0.12, 0.12), "White" if (i+j)%2 else "Graphite", collide=False)
    for i, y in enumerate((-0.52, 0.52)):
        scene.tube(f"Station/Handle{i}", (2.27, y, -1.83), (2.27, y, -1.47), 0.028, "SafetyYellow")
    # Piles with collars, braces and an open grated landing deck.
    for i, (x, y) in enumerate(((7, -3.3), (7, 3.3), (10, -3.3), (10, 3.3))):
        scene.tube(f"Pier/Pile{i}", (x, y, -2.8), (x, y, 1.8), 0.24)
        for j, z in enumerate((-1.8, 0.5)):
            scene.ring(f"Pier/Collar{i}_{j}", (x, y, z), 0.25, 0.026, "Titanium", "Z")
    scene.tube("Pier/BracePort", (7, 3.3, -2.3), (10, 3.3, 0.3), 0.07)
    scene.tube("Pier/BraceStarboard", (7, -3.3, 0.3), (10, -3.3, -2.3), 0.07)
    for i in range(23):
        scene.cube(f"Pier/Grating{i}", (7+i*3/22, 0, 0.4), (0.045, 6.8, 0.055), "Titanium", collide=False)
    for i, y in enumerate((-3.3, 0, 3.3)):
        scene.cube(f"Pier/Stringer{i}", (8.5, y, 0.35), (3.4, 0.075, 0.1))
    scene.cube("Equipment/Case", (4.4, -2, -2.43), (0.8, 0.5, 0.45), "SafetyYellow", (0, 0, 18))
    for i, z in enumerate((-2.55, -2.3)):
        scene.cube(f"Equipment/Band{i}", (4.4, -2.255, z), (0.69, 0.025, 0.025), "Graphite", collide=False)
    scatter_rocks(scene, 48)
    scene.save()


def scatter_rocks(scene, count, reef=False):
    rng = np.random.default_rng(SEED)
    for i in range(count):
        x, y = rng.uniform(-9, 14), rng.uniform(-9, 9)
        if abs(y) < (2.5 if not reef else 1.2) and -4 < x < 11:
            continue
        scale = rng.uniform(0.15, 0.75) * (1.8 if reef else 1)
        size = (scale, scale*rng.uniform(0.7, 1.3), scale*rng.uniform(0.4, 0.8))
        scene.rock(f"Rocks/Boulder{i:03d}", (x, y, seabed_height(x, y)+size[2]*0.35), size, rng)
        # Tufts of kelp as bent, double-sided ribbons on the reef's outer rocks.
        if reef and i%3 == 0:
            for k in range(4):
                points, uv, faces = [], [], []
                for j in range(9):
                    z = j/8*0.7
                    bend = 0.12*math.sin(z*5 + k)
                    centre = (x+0.08*k+bend, y+0.05*k, seabed_height(x, y)+z)
                    points.extend([(centre[0]-0.035, centre[1], centre[2]),
                                   (centre[0]+0.035, centre[1], centre[2])])
                    uv.extend([(0, j/8), (1, j/8)])
                    if j:
                        a = (j-1)*2
                        faces.append((a, a+1, a+3, a+2))
                mesh = scene.mesh(f"Kelp/Tuft{i}_{k}", points, faces, "Algae", uv, False, 0.1)
                mesh.CreateDoubleSidedAttr(True)


def rocky_reef(path):
    scene = Scene(path, "RockyReef")
    scene.terrain()
    scene.lighting()
    scatter_rocks(scene, 115, reef=True)
    # A survey transect and salvage target give the natural scene a mission.
    scene.tube("Survey/Transect", (-3, 0.8, -2.5), (9, 0.8, -2.5), 0.012, "SafetyYellow", 8, False)
    for i, x in enumerate((-2, 0, 2, 4, 6, 8)):
        scene.tube(f"Survey/Stake{i}", (x, 0.8, -2.7), (x, 0.8, -2.15), 0.022, "Titanium", 12)
        scene.cube(f"Survey/Marker{i}", (x, 0.8, -2.14), (0.08, 0.2, 0.05), "SafetyYellow")
    scene.cube("Salvage/Case", (3.2, -0.9, -2.48), (0.7, 0.5, 0.4), "EnamelBlue", (0, 0, -20))
    scene.ring("Salvage/LiftingEye", (3.2, -0.9, -2.24), 0.1, 0.016, "Titanium", "X")
    scene.save()


def read_stl(path):
    """Read the local binary CAD STLs without an additional mesh dependency."""
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + count*50:
        raise ValueError(f"Expected a binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    triangles = np.frombuffer(data, dtype=dtype, offset=84)["vertices"]
    return triangles.reshape(-1, 3).copy()


def stage_revolution(mesh_dir, output):
    """Copy local CAD, retain the model's hydrodynamics, and add an actuated head."""
    required = ("chassis.stl", "pivot_head.stl", "claw_clone.stl")
    for name in required:
        if not (mesh_dir / name).is_file():
            raise FileNotFoundError(mesh_dir / name)
    vehicle_dir = output / "DeepTrekker"
    dest = vehicle_dir / "meshes"
    dest.mkdir(parents=True, exist_ok=True)
    for name in required:
        if (mesh_dir / name).resolve() != (dest / name).resolve():
            shutil.copy2(mesh_dir / name, dest / name)
    platforms = load_util("platforms")
    exporter = load_util("urdf_export")
    spec = platforms.get_platform("deeptrekker_revolution")
    robot = ET.fromstring(exporter.platform_to_urdf(spec))
    hull = robot.find("link[@name='base_link']")
    hull.remove(hull.find("visual"))
    # The chassis STL already includes the six thruster housings.
    for link in list(robot.findall("link")):
        if link.get("name", "").startswith("thruster"):
            visual = link.find("visual")
            if visual is not None:
                link.remove(visual)
    robot.find("material[@name='hull']/color").set("rgba", "0.16 0.18 0.20 1")
    ET.SubElement(ET.SubElement(robot, "material", name="head"), "color", rgba="0.26 0.29 0.32 1")
    ET.SubElement(ET.SubElement(robot, "material", name="claw"), "color", rgba="0.4 0.43 0.46 1")

    def visual(link, filename, material, xyz="0 0 0", rpy="0 0 0"):
        v = ET.SubElement(link, "visual")
        ET.SubElement(v, "origin", xyz=xyz, rpy=rpy)
        ET.SubElement(ET.SubElement(v, "geometry"), "mesh", filename="meshes/" + filename)
        ET.SubElement(v, "material", name=material)

    visual(hull, "chassis.stl", "hull")
    head = ET.SubElement(robot, "link", name="pivot_head")
    exporter._inertial(head, 0.7, inertia=(0.003, 0.003, 0.003))
    visual(head, "pivot_head.stl", "head")
    exporter._joint(robot, "pivot_head_joint", "base_link", "pivot_head", (0.215, 0, 0), (0, 0, 0),
                    "revolute", axis=(0, -1, 0), limit={"lower": -1.8849556, "upper": 2.6179939,
                                                      "effort": 20, "velocity": 1.0})
    claw = ET.SubElement(robot, "link", name="claw_arm")
    exporter._inertial(claw, 0.001)
    visual(claw, "claw_clone.stl", "claw", xyz="0 0 0.0635", rpy="0 0.1 0")
    exporter._joint(robot, "pivot_head_to_claw", "pivot_head", "claw_arm", (0.0625, 0, -0.098), (0, 0, 0))
    # Head-mounted sensor links stay attached to the articulated head.
    for kind in ("sonar", "camera"):
        joint = robot.find(f"joint[@name='{kind}_link_joint']")
        if joint is not None:
            joint.find("parent").set("link", "pivot_head")
            mount = getattr(spec, f"{kind}_mount")
            joint.find("origin").set("xyz", exporter._fmt(np.array(mount.translation)-(0.215, 0, 0)))
    # Keep the full vehicle at its 26 kg model mass when importing the links.
    hull.find("inertial/mass").set("value", "25.3")
    ET.indent(robot)
    urdf = vehicle_dir / "revolution.urdf"
    urdf.write_text(ET.tostring(robot, encoding="unicode") + "\n")
    # A static, material-bound CAD USD is useful for scene previews and layout.
    preview = Scene(vehicle_dir / "revolution_preview.usdc", "Revolution")
    for name, pos, material, filename in (
        ("Chassis", (0, 0, 0), "Graphite", "chassis.stl"),
        ("PivotHead", (0.215, 0, 0), "Graphite", "pivot_head.stl"),
        ("Claw", (0.2775, 0, -0.0345), "Titanium", "claw_clone.stl"),
    ):
        points = read_stl(dest / filename)
        if name == "Claw":
            c, s = math.cos(0.1), math.sin(0.1)
            points = points @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        points += np.array(pos)
        preview.mesh(name, points, np.arange(len(points)).reshape(-1, 3), material, collide=False)
    preview.save()
    (vehicle_dir / "source.json").write_text(json.dumps({
        "source": str(mesh_dir.resolve()), "meshes": list(required),
        "description": "Local Nautilus REVOLUTION CAD; frames from its platform description.",
        "note": "CAD visuals are approximate; OceanSim registry supplies estimated hydrodynamics.",
    }, indent=2) + "\n")
    print(f"Staged articulated REVOLUTION: {urdf}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "demo/assets")
    parser.add_argument("--revolution-meshes", type=Path, help="Local Nautilus REVOLUTION meshes directory.")
    args = parser.parse_args()
    scenes = args.output / "environments"
    scenes.mkdir(parents=True, exist_ok=True)
    make_textures(args.output / "textures")
    inspection_site(scenes / "inspection_site.usdc")
    rocky_reef(scenes / "rocky_reef.usdc")
    if args.revolution_meshes:
        stage_revolution(args.revolution_meshes, args.output)
    (args.output / "manifest.json").write_text(json.dumps({
        "seed": SEED, "meters_per_unit": 1, "up_axis": "Z", "water_surface_z": 0,
        "environments": ["environments/inspection_site.usdc", "environments/rocky_reef.usdc"],
        "generator": "scripts/build_demo_assets.py",
    }, indent=2) + "\n")
    print(f"Demo assets built at {args.output.resolve()}")


if __name__ == "__main__":
    main()
