"""Pure (numpy-only) point selection for the imaging-sonar scan pipeline.

``ImagingSonarSensor.scan()`` fetches per-pixel depth / pointcloud / normals /
semantics from the Replicator annotators (Isaac Sim), then keeps only the pixels
that are a finite hit inside the sonar range window. That selection is pure numpy
and lives here so it can be unit tested without Isaac Sim -- and so it can be
optimised against a characterisation test rather than by eyeballing sonar images.
"""

import numpy as np


def valid_point_mask(depth_flat, pcl, min_range, max_range):
    """Reference one-pass validity mask: finite depth in (min, max) AND finite 3D point.

    This is the readable definition that ``select_in_range_points`` is tested to
    match; production uses the optimised path below.
    """
    depth_flat = np.asarray(depth_flat).reshape(-1)
    return (np.isfinite(depth_flat)
            & (depth_flat > min_range)
            & (depth_flat < max_range)
            & np.isfinite(pcl).all(axis=1))


def select_in_range_points(depth_flat, pcl, normals, semantics, min_range, max_range):
    """Select the in-range, finite points; return contiguous
    (pcl float32 (M,3), normals float32 (M,3), semantics uint32 (M,)).

    Equivalent to indexing the inputs by :func:`valid_point_mask`, but two-stage:
    the cheap depth-window mask runs over every pixel, while the more expensive
    per-point finiteness check and the gathers touch only the depth-passing
    subset. That is a large saving when most pixels fall outside the (typically
    narrow) sonar range window. ``M`` may be 0.
    """
    depth_flat = np.asarray(depth_flat).reshape(-1)
    dmask = (np.isfinite(depth_flat)
             & (depth_flat > min_range)
             & (depth_flat < max_range))
    idx = np.flatnonzero(dmask)
    if idx.size:
        idx = idx[np.isfinite(pcl[idx]).all(axis=1)]
    pcl_v = np.ascontiguousarray(pcl[idx], dtype=np.float32)
    normals_v = np.ascontiguousarray(normals[idx], dtype=np.float32)
    semantics_v = np.ascontiguousarray(semantics[idx]).astype(np.uint32)
    return pcl_v, normals_v, semantics_v


def make_indexToProp_array(idToLabels, query_property):
    """Build the ``indexToProp`` lookup the intensity kernel indexes by semantic
    id: a 1-D array where entry ``i`` is the queried property (e.g. reflectivity)
    of semantic id ``i``.

    ``idToLabels`` maps stringified ids ('0', '1', ...) to a dict of label
    properties (e.g. ``{'reflectivity': '2.0'}``). Keys are compared numerically
    so an id >= 10 does not lexicographically undersize the array. Ids that lack
    ``query_property`` -- or carry a non-numeric value (a 'BACKGROUND' /
    'UNLABELLED' fallback) -- keep the default reflectivity 1.0 rather than
    raising mid-scan. Returns a float64 array of length ``max_id + 1`` (empty if
    there are no labels).
    """
    max_id = max((int(k) for k in idToLabels.keys()), default=-1)
    indexToProp_array = np.ones((max_id + 1,))
    for id in idToLabels.keys():
        for property in idToLabels.get(id):
            if property == query_property:
                try:
                    indexToProp_array[int(id)] = float(idToLabels.get(id).get(property))
                except (TypeError, ValueError):
                    pass
    return indexToProp_array


def depth_unprojection_from_camera_params(camera_params, width, height):
    """Unprojection inputs for compact_depth_points from a CameraParams
    annotator dict: ``(cam_to_world, fx, fy, cx, cy)``.

    ``cameraViewTransform`` is the world->camera (USD camera frame) transform in
    USD's row-vector layout, so its transpose is the column-vector form and
    ``cam_to_world`` is that matrix's inverse. ``cameraProjection`` is the
    render's perspective matrix; its diagonal gives P00 = 2f / h_aperture and
    P11 = 2f / v_aperture, so fx = W * P00 / 2 and fy = H * P11 / 2 -- the
    effective intrinsics the renderer used. The principal point is the image
    centre, as Isaac's Camera.get_intrinsics_matrix() assumes.
    """
    view = np.asarray(camera_params["cameraViewTransform"], dtype=np.float64).reshape(4, 4).T
    proj = np.asarray(camera_params["cameraProjection"], dtype=np.float64).reshape(4, 4)
    cam_to_world = np.linalg.inv(view)
    fx = 0.5 * float(width) * proj[0, 0]
    fy = 0.5 * float(height) * proj[1, 1]
    return cam_to_world, fx, fy, 0.5 * float(width), 0.5 * float(height)


# --- FLS camera geometry ------------------------------------------------------
# The sonar is rendered with a pinhole camera (square pixels) whose horizontal
# FOV is the fan width. In the sensor's local frame (x right, y forward along
# the optical axis, z up) a pixel ray has azimuth atan2(forward, x) (pi/2 =
# boresight) and elevation atan2(up, sqrt(x^2 + forward^2)).

def sonar_render_height(width, hori_fov_deg, vert_fov_deg):
    """Render height (px) so the camera covers elevation +-vert_fov/2 across the
    WHOLE fan. A pinhole's vertical extent for a fixed elevation grows as
    1/cos(azimuth), so the fan edge needs H = W tan(vfov/2) / sin(hfov/2); the
    old W * vfov / hfov (a ratio of angles) gave ~36.5 deg at boresight but only
    ~16 deg at the 65 deg edge for a 130 x 20 deg sonar. Points above/below
    +-vfov/2 are then dropped by elevation in the binning kernels."""
    h = np.deg2rad(float(hori_fov_deg)) / 2.0
    v = np.deg2rad(float(vert_fov_deg)) / 2.0
    return int(np.ceil(float(width) * np.tan(v) / np.sin(h)))


def slant_range_near_depth(min_range, width, height, fx, fy):
    """Near clip / depth-filter threshold that keeps every point with SLANT
    range >= min_range. Depth is measured along the optical axis, so the
    closest slant range a depth cut d admits is d / cos(alpha); cutting at
    min_range * cos(alpha_max) (alpha_max = the image-corner ray's angle off the
    axis) keeps the whole fan down to min_range, and binning rejects anything
    closer than min_range. Cutting at min_range itself lost targets nearer than
    min_range / cos(65 deg) = 0.47 m at the fan edge for a 0.2 m sonar."""
    tx = (float(width) / 2.0) / float(fx)
    ty = (float(height) / 2.0) / float(fy)
    return float(min_range) / np.sqrt(1.0 + tx * tx + ty * ty)


def _pixel_rays(width, height, fx, fy, cx=None, cy=None):
    cx = float(width) / 2.0 if cx is None else float(cx)
    cy = float(height) / 2.0 if cy is None else float(cy)
    x = (np.arange(width, dtype=np.float64) + 0.5 - cx) / float(fx)    # right, per unit forward
    up = -(np.arange(height, dtype=np.float64) + 0.5 - cy) / float(fy)  # image v grows downward
    return np.meshgrid(x, up)


def beam_solid_angle_gain(width, height, fx, fy, min_azi, azi_res, n_beams,
                          hori_fov, half_vfov):
    """Per-beam gain g[j] so that sum over the beam's pixels of
    cos^3(alpha) * g[j] equals the beam's true solid angle.

    A pixel subtends dOmega ~ cos^3(alpha) (alpha = angle off the optical
    axis), so summing raw pixel returns ("sum" binning) weighted beams by how
    many pixels happen to land in them: more toward the fan edges (sec^2) and
    alternately 2 or 3 pixel columns per 0.25 deg beam (1.5x stripes). With
    each point weighted by cos^3(alpha) * g[beam] (see bin_intensity_sa), every
    beam integrates over its real solid angle  overlap_width * 2 sin(half_vfov)
    instead. All angles in radians; min_azi / hori_fov describe the fan the
    grid bins (azimuth pi/2 = boresight). Beams no pixel reaches get 0."""
    xg, ug = _pixel_rays(width, height, fx, fy)
    azi = np.arctan2(1.0, xg)
    elev = np.arctan2(ug, np.sqrt(xg * xg + 1.0))
    cos3 = (1.0 / np.sqrt(xg * xg + ug * ug + 1.0)) ** 3
    j = np.floor((azi - float(min_azi)) / float(azi_res)).astype(np.int64)
    keep = (np.abs(elev) <= float(half_vfov)) & (j >= 0) & (j < int(n_beams))
    sums = np.bincount(j[keep], weights=cos3[keep], minlength=int(n_beams))
    lo = float(min_azi) + np.arange(int(n_beams)) * float(azi_res)
    width_in_fan = np.clip(np.minimum(lo + float(azi_res), float(min_azi) + float(hori_fov)) - lo,
                           0.0, float(azi_res))
    ideal = width_in_fan * 2.0 * np.sin(float(half_vfov))
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(sums > 0.0, ideal / sums, 0.0)
    return gain.astype(np.float32)


def depth_to_world_points(depth, cam_to_world, fx, fy, cx=None, cy=None):
    """(H*W, 3) world points from a distance_to_image_plane image -- the numpy
    twin of the compact_depth_points kernel (same convention as Isaac's
    Camera.get_pointcloud() depth fallback, row-major over (H, W))."""
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    cx = float(w) / 2.0 if cx is None else float(cx)
    cy = float(h) / 2.0 if cy is None else float(cy)
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x_ros = (uu - cx) * depth / float(fx)
    y_ros = (vv - cy) * depth / float(fy)
    cam = np.stack([x_ros, -y_ros, -depth, np.ones_like(depth)], axis=-1).reshape(-1, 4)
    return (cam @ np.asarray(cam_to_world, dtype=np.float64).T)[:, :3]
