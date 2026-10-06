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
