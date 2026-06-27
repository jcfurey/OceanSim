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
