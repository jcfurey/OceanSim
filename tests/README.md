# OceanSim unit tests

gtest-style (pytest) tests for the parts of OceanSim that are pure computation
and so can be validated **without launching Isaac Sim**.

## What's covered today (no Isaac Sim needed)
- `test_imaging_sonar_kernels.py` — the imaging-sonar **Warp kernels** run on the
  CPU device. Locks down the bin-index bounds check, the global-normalization
  ÷0 guard, the `make_sonar_image` column flip / no-out-of-bounds write, the
  `compact_in_range` GPU stream-compaction kernel (proven equal to the numpy
  reference selection across random seeds, including the `(H,W)` reshape /
  4-channel-normals slice / reusable-buffer glue that `scan()` uses), and the
  `make_sonar_data` work-buffer reuse (bit-identical to fresh allocation across
  grow→shrink frames, so no stale residue leaks).
- `test_sonar_scan_math.py` — pure point selection: `select_in_range_points`
  exactly equals the readable `valid_point_mask` reference, and
  `make_indexToProp_array` (reflectivity lookup: numeric key ordering / array
  sizing, missing-property default, non-numeric fallback).
- `test_rtx_acoustic_math.py` — `fold_gmo_to_grid` time→range / receiver→beam
  folding and peak normalisation, including the non-finite-range sentinel.
- `test_ros2_math.py` — the ROS2 publishing math (quaternion conversion, world→
  body transforms, IMU specific force, covariance diagonals, sim-time split,
  rate gate, sonar beam directions / ranges / uint8 intensity layout).
- `test_camera_math.py` — pinhole `CameraInfo` intrinsics (K/R/P/D) and the
  radial→planar depth factor.
- `test_noise_samplers.py` — `MultivariateNormal` (in-place Cholesky, covariance
  reconstruction, std-dev conventions) and `MultivariateUniform` (bounds,
  exponential mean).

These import the modules **by file path**, so they don't need the
`isaacsim.oceansim` namespace package on `sys.path`.

CI runs them on every push / PR (`.github/workflows/tests.yml`, Python 3.10 &
3.11) — Warp on the CPU device, no GPU or Isaac Sim required.

## Running
```bash
# standalone (CI / dev box)
pip install warp-lang numpy pytest
pytest tests/

# inside the OceanSim / Isaac Sim container (warp + numpy already present)
/isaac-sim/python.sh -m pytest tests/
```
Each test `importorskip`s `numpy` / `warp`, so a missing dependency skips rather
than errors.

The math that used to be embedded in `UW_Camera` / `ros2_sensors` (IMU specific
force, frame transforms, `ProjectedSonarImage` layout, `CameraInfo` intrinsics,
radial→planar depth) has since been extracted into dependency-free modules
(`ros2_math`, `camera_math`, `sonar_scan_math`, `rtx_acoustic_math`) and is now
covered above.

## Not covered here (needs Isaac Sim / ROS hardware)
- The Isaac-runtime seams themselves: the Replicator annotator schema,
  `get_pointcloud`, on-device residency of the optional `gpu_point_filter` fast
  path, the RTX acoustic GMO delivery, and the actual ROS publishing. The pure
  math each of these feeds is unit tested above, but the integration still needs
  an end-to-end / image-A-B smoke test on real Isaac Sim.
