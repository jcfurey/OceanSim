# OceanSim unit tests

gtest-style (pytest) tests for the parts of OceanSim that are pure computation
and so can be validated **without launching Isaac Sim**.

## What's covered today (no Isaac Sim needed)
- `test_imaging_sonar_kernels.py` — the imaging-sonar **Warp kernels** run on the
  CPU device. Locks down the bin-index bounds check, the global-normalization
  ÷0 guard, and the `make_sonar_image` column flip / no-out-of-bounds write.
- `test_noise_samplers.py` — `MultivariateNormal` (in-place Cholesky, covariance
  reconstruction, std-dev conventions) and `MultivariateUniform` (bounds,
  exponential mean).

These import the modules **by file path**, so they don't need the
`isaacsim.oceansim` namespace package on `sys.path`.

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

## Not covered here (needs Isaac Sim / ROS, or a small refactor first)
- Helpers embedded in modules that import `omni` / `rclpy` at top
  (`UW_Camera`, `ros2_sensors`) — the IMU specific force, frame transforms,
  `ProjectedSonarImage` layout, `CameraInfo` intrinsics, radial→planar depth.
  These become unit-testable once extracted into dependency-free math modules
  (see the "seam" refactor) — at which point they get tests here too.
- The Isaac-runtime seams (Replicator annotator schema, `get_pointcloud`, the
  RTX acoustic GMO delivery, ROS publishing). Those still need an end-to-end /
  image-A-B smoke test on real Isaac Sim.
