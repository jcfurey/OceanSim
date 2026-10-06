"""Tests for the depth -> world point reconstruction that replaced the
'pointcloud' composite annotator in FLS_KittiWriter (it crashed at
world.play() on Isaac Sim 6.x): the compact_depth_points Warp kernel plus
sonar_scan_math.depth_unprojection_from_camera_params.

The reference is an independent numpy copy of Isaac Sim's
Camera.get_world_points_from_image_coords() math (isaacsim.sensors.camera):
K from focal length / aperture, pixel centres at +0.5, ROS optical frame
flipped to the USD camera frame by diag(1, -1, -1), then the camera's
local-to-world transform. Runs on the Warp CPU device; no Isaac Sim needed.
"""

import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")
wp = pytest.importorskip("warp")

_UTILS = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim", "utils")
DEV = "cpu"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_UTILS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def kern():
    wp.init()
    return _load("ImagingSonar_kernels")


@pytest.fixture(scope="module")
def ssm():
    return _load("sonar_scan_math")


W, H = 16, 10
FOCAL, H_AP = 2.0, 4.0
V_AP = H_AP * H / W            # square pixels, as Isaac enforces


def _rot(yaw_deg, pitch_deg):
    y, p = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return rz @ rx


def _camera():
    """world_from_camU (column form) + the CameraParams dict a render would emit."""
    world_from_cam = np.eye(4)
    world_from_cam[:3, :3] = _rot(30.0, 10.0)
    world_from_cam[:3, 3] = [1.5, -2.0, 0.7]
    view_col = np.linalg.inv(world_from_cam)          # world -> camU, column form
    proj = np.zeros((4, 4))
    proj[0, 0] = 2.0 * FOCAL / H_AP
    proj[1, 1] = 2.0 * FOCAL / V_AP
    proj[2, 2], proj[2, 3], proj[3, 2] = 0.0, -1.0, 0.01   # (only the diagonal is read)
    params = {
        # USD row-vector layout: the transpose of the column-form matrix.
        "cameraViewTransform": view_col.T.reshape(-1),
        "cameraProjection": proj.T.reshape(-1),
    }
    return world_from_cam, params


def _isaac_reference(depth, world_from_cam):
    """Camera.get_world_points_from_image_coords, transcribed to numpy."""
    fx, fy = W * FOCAL / H_AP, H * FOCAL / V_AP
    k = np.array([[fx, 0, W * 0.5], [0, fy, H * 0.5], [0, 0, 1.0]])
    xs, ys = np.meshgrid(np.linspace(0.5, W - 0.5, W), np.linspace(0.5, H - 0.5, H), indexing="xy")
    pts2d = np.column_stack((xs.ravel(), ys.ravel(), np.ones(W * H)))
    cam_ros = (np.linalg.inv(k) @ (pts2d.T * depth.ravel()[None, :])).T
    r_u = np.diag([1.0, -1.0, -1.0, 1.0])
    view_ros = r_u @ np.linalg.inv(world_from_cam)
    homo = np.column_stack((cam_ros, np.ones(W * H)))
    return (np.linalg.inv(view_ros) @ homo.T).T[:, :3]


def _run(kern, ssm, depth, sem=None, exclude=None, min_r=0.1, max_r=50.0):
    world_from_cam, params = _camera()
    cam_to_world, fx, fy, cx, cy = ssm.depth_unprojection_from_camera_params(params, W, H)
    n = W * H
    normals = np.zeros((H, W, 4), dtype=np.float32)
    normals[..., 0] = np.arange(n, dtype=np.float32).reshape(H, W)   # tag = pixel index
    sem = np.full((H, W), 3, dtype=np.uint32) if sem is None else sem
    inst = np.arange(n, dtype=np.uint32).reshape(H, W)                # pixel index
    exclude = np.zeros(1, dtype=np.uint8) if exclude is None else exclude
    counter = wp.zeros(1, dtype=wp.int32, device=DEV)
    out_pcl = wp.zeros((n, 3), dtype=wp.float32, device=DEV)
    out_nrm = wp.zeros((n, 3), dtype=wp.float32, device=DEV)
    out_sem = wp.zeros(n, dtype=wp.uint32, device=DEV)
    out_inst = wp.zeros(n, dtype=wp.uint32, device=DEV)
    wp.launch(kern.compact_depth_points, dim=(H, W),
              inputs=[wp.array(depth.astype(np.float32), dtype=wp.float32, device=DEV),
                      wp.array(normals, dtype=wp.float32, device=DEV),
                      wp.array(sem, dtype=wp.uint32, device=DEV),
                      wp.array(inst, dtype=wp.uint32, device=DEV),
                      wp.array(exclude, dtype=wp.uint8, device=DEV),
                      wp.mat44(cam_to_world.astype(np.float32)),
                      float(fx), float(fy), float(cx), float(cy),
                      float(min_r), float(max_r),
                      counter, out_pcl, out_nrm, out_sem, out_inst],
              device=DEV)
    wp.synchronize()
    k = int(counter.numpy()[0])
    idx = out_inst.numpy()[:k]
    order = np.argsort(idx)
    return (idx[order], out_pcl.numpy()[:k][order], out_nrm.numpy()[:k][order],
            out_sem.numpy()[:k][order], world_from_cam)


def test_intrinsics_from_projection_match_isaac(ssm):
    _, params = _camera()
    cam_to_world, fx, fy, cx, cy = ssm.depth_unprojection_from_camera_params(params, W, H)
    assert fx == pytest.approx(W * FOCAL / H_AP)
    assert fy == pytest.approx(H * FOCAL / V_AP)
    assert (cx, cy) == (W * 0.5, H * 0.5)
    world_from_cam, _ = _camera()
    assert np.allclose(cam_to_world, world_from_cam)


def test_world_points_match_isaac_get_pointcloud(kern, ssm):
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 8.0, (H, W))
    idx, pcl, nrm, _, world_from_cam = _run(kern, ssm, depth)
    assert idx.tolist() == list(range(W * H))          # every pixel kept, once
    ref = _isaac_reference(depth, world_from_cam)
    assert np.allclose(pcl, ref, atol=1e-4)
    assert np.allclose(nrm[:, 0], idx)                  # normals travel with their pixel


def test_points_reproject_to_their_pixel_and_depth(kern, ssm):
    """Independent forward check: world -> camU -> pixel must land on the pixel
    centre the point came from, at the input distance-to-image-plane."""
    rng = np.random.default_rng(1)
    depth = rng.uniform(0.5, 8.0, (H, W))
    idx, pcl, _, _, world_from_cam = _run(kern, ssm, depth)
    cam = (np.linalg.inv(world_from_cam) @ np.column_stack((pcl, np.ones(len(pcl)))).T).T
    d = -cam[:, 2]                                      # USD camera looks down -Z
    u = cam[:, 0] / d * (W * FOCAL / H_AP) + W * 0.5
    v = -cam[:, 1] / d * (H * FOCAL / V_AP) + H * 0.5
    assert np.allclose(d, depth.ravel()[idx], atol=1e-4)
    assert np.allclose(u, idx % W + 0.5, atol=1e-3)
    assert np.allclose(v, idx // W + 0.5, atol=1e-3)


def test_range_window_and_non_finite_depth_are_dropped(kern, ssm):
    depth = np.full((H, W), 2.0)
    depth[0, 0] = np.inf          # no hit (background)
    depth[0, 1] = np.nan
    depth[0, 2] = 0.05            # inside min range
    depth[0, 3] = 60.0            # beyond max range
    idx, *_ = _run(kern, ssm, depth, min_r=0.1, max_r=50.0)
    assert set(idx.tolist()) == set(range(W * H)) - {0, 1, 2, 3}


def test_excluded_semantic_ids_are_dropped(kern, ssm):
    depth = np.full((H, W), 2.0)
    sem = np.full((H, W), 2, dtype=np.uint32)
    sem[1, :] = 1                 # e.g. UNLABELLED
    sem[2, :] = 4000000000        # id past the lookup table: kept, no OOB read
    exclude = np.array([0, 1], dtype=np.uint8)
    idx, _, _, out_sem, _ = _run(kern, ssm, depth, sem=sem, exclude=exclude)
    assert set(idx.tolist()) == set(range(W * H)) - set(range(W, 2 * W))
    assert 1 not in out_sem.tolist() and 4000000000 in out_sem.tolist()
