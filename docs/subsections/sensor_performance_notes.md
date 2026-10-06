# Sensor Performance Notes

Working notes on where the OceanSim sensor pipeline spends time and which
optimizations are done vs. still on the table. Focused on the sonars (the
dominant cost) but the camera / DVL paths are noted too. Targets Isaac Sim
6.1.0.

Each item lists the **lever**, the expected **payoff**, the **risk**, and its
**status**. "Hardware-gated" means it can't be validated in CI (needs a GPU +
Isaac session) and has historically been where wrong assumptions hide.

---

## Done

### Imaging sonar: single on-device semantic fetch per scan
`ImagingSonarSensor.scan()` used to call the semantic-segmentation annotator's
`get_data()` **to host** every frame purely to read `idToLabels` (the
CUDA-warmup / empty-FOV gate), then — on the runner default
(`gpu_point_filter=True`) — `_scan_gpu_compact` re-fetched the *same* annotator
**on-device** for the compaction kernel. Every default-config scan therefore
DMA'd the whole `H×W` semantic image host-ward and discarded it.

Now the annotator is fetched once, on-device, and the dict is reused for both
the gate and the compaction kernel. Falls back to a host fetch for the labels
only if a build omits `info` from the device dict, so it is strictly
non-regressive. **Payoff:** removes one full-image host readback + sim-thread
stall per scan (helps odom/imu jitter; does *not* raise GPU-bound framerate).

### Pass 2 (Isaac Sim 6.1.0): output-identical host-side wins
- **Scan / render only on publish ticks (headless runner).** With no sensor
  viewport and the sync sonar, the scan only feeds the ROS publisher, so the
  runner now scans exactly when the sonar publish is due
  (`publisher.sonar_due()` via the non-mutating `RateGate.due()`), and
  `UW_Camera.render()` skips the annotator fetch + `UW_render` for frames its
  own publish gate won't send. At the 15 Hz compute / 5 Hz publish defaults
  that is 2/3 fewer scans and camera renders, and published frames are
  captured on their publish tick. (Not with render gating, async sonar, a
  sonar_rate of 0, or viewports.)
- **ROS `uint8[]` payloads as `array('B')`** (`ros2_math.uint8_payload`):
  rosidl takes it as-is, whereas `bytes` is checked element by element in
  Python on Humble (~0.4 s per 1080p rgb8 frame); one copy instead of two.
- **GUI sonar uses `gpu_point_filter=True`** like the runner (the numpy path
  copied ~89 MB to the host and spent ~80 ms per scan at `hori_res=4000`).
- **KITTI writers:** single-pass instance segmentation export (782 -> 165 ms
  per 1080p frame at 100 ids, bit-exact), debug-only annotators, reused
  buffers, cached reflectivity upload, constant JSON written once; cached
  sonar segmentation palette.

---

## Outstanding — sonar

### 1. Drop the RTX raytrace on non-scan steps (biggest lever, hardware-gated)
The imaging-sonar camera raytraces **every** rendered step even when the sonar
only scans every *N*th step; the loop is GPU-render-bound (`nvidia-smi` ~100%),
so this — not any host-side work — caps framerate.

- **Current state:** `oceansim_ros2.py` wires a `render_rate` gate, but the only
  per-step lever available (`render_product.hydra_texture.set_updates_enabled()`)
  stops the AOV readback **without** removing the raytrace from the GPU pipeline.
  It breaks the sonar (scan gets no data) and frees no GPU, so it is left dormant
  (see the `KNOWN-LIMITED` comment near the render-gate block).
- **What's needed:** a lever that actually drops the sonar render product from
  the per-step SDG/GPU pipeline, plus a re-enable warmup to cover render latency
  when a scan is due. Candidates to investigate: toggling the render product's
  activation, detaching/attaching the RTX sensor per cadence, or a second render
  product driven only at scan cadence.
- **Payoff:** potentially large (raytrace is the dominant cost). **Risk:** high,
  hardware-gated — render-latency and warmup behavior must be measured live.

### 2. Reuse the depth AOV inside `get_pointcloud` (medium, hardware-gated)
`_scan_gpu_compact` fetches the depth AOV on-device (used by `compact_in_range`),
then calls `Camera.get_pointcloud(...)`, which internally re-reads the same
`distance_to_image_plane` annotator to reconstruct points — so depth is fetched
twice per scan (both device-side).

- **Lever:** reconstruct the point cloud from the depth array we already hold,
  instead of calling the base `get_pointcloud()` (or feed it our depth if a
  supported hook exists).
- **Payoff:** one fewer annotator `get_data()` per scan (device→device, so
  smaller than item 1). **Risk:** medium — reimplementing the base
  `get_pointcloud()` projection blind is easy to get subtly wrong (intrinsics,
  world-frame transform); needs a characterization test against the current
  output before switching.

### 3. Skip the noise kernels when their params are zero (small)
`make_sonar_data` launches `normal_2d` (Gaussian) and
`range_dependent_rayleigh_2d` over the full grid every frame. When
`gau_noise_param == 0` (and `ray_noise_param == 0` **and** `central_peak == 0`)
the results are all-zero and contribute nothing downstream.

- **Lever:** skip the launch and instead point the map kernel at a
  zeroed-once buffer when the corresponding param is zero.
- **Payoff:** small — two grid-sized kernels are cheap next to the raytrace, and
  the default config uses non-zero noise, so this rarely triggers. **Risk:** low
  but must zero the buffers once (they're currently fully overwritten each
  frame, so a naive skip would leak stale values into the image). Low priority.

### 4. `fold_gmo_to_grid` — done
The fold now resamples the A-scans with two cached float32 weight matrices
(range, then azimuth; `rtx_acoustic_math._linear_bin_weights`): ~1.0 ms per
frame at the runner defaults vs ~2.6 ms for the old `float64` scatter, while
filling every bin (see CHANGELOG).

---

### Concrete design for item 1: render only on sensor steps (hardware-gated)
`world.step(render=True)` raytraces every render product (sonar + camera) on
every physics step. Step physics with `render=False` and render only on steps
where a sensor is due (Isaac Lab's `render_interval` is the precedent). Unlike
`set_updates_enabled()` this actually removes the raytrace: up to ~4x less RTX
work at the defaults (~12x if aligned to the 5 Hz publish). Check on hardware:
whether `get_data()` after one render returns that frame or the previous one
(render two consecutive steps if it lags), whether `world.current_time`
advances on `render=False` steps (stamps / rate gates depend on it), and
TAA/DLSS ghosting on sparse camera frames. Headless only.

### Item 2 update
`compact_depth_points` + `sonar_scan_math.depth_unprojection_from_camera_params`
(now used by `FLS_KittiWriter`, unit-tested against Isaac's own unprojection)
can replace `get_pointcloud()` + `compact_in_range` in `_scan_gpu_compact`.
Besides the second depth fetch and four `.contiguous()` copies, it makes the
points use the render-time camera pose like intensity / binning already do
(`get_pointcloud()` uses the current USD pose; a 1 deg / 1 cm lag shifts
targets 3-4 beams / 2 range bins). Changes the default sonar path.

## Outstanding — camera / ROS plumbing

Audited in pass 2; what remains:

- **`rclpy.spin_once(node)`** (`ros2_sensors`, `ros2_control`, `UW_Camera`)
  adds/removes the node from the global executor on every call; one persistent
  executor with the nodes added once would save ~0.1-0.3 ms per call (estimate).
- **JPEG encode on the sim thread** (previously flagged) — image compression on
  the render/sim thread applies backpressure; a worker-thread encode (like the
  sonar's `async_compute`) would decouple it.
- **DVL** — the beam→velocity math is already pure/tested; confirm the sensor
  read path adds no per-tick host sync.

---

## Notes

- The sonar **compute** paths (`fold_gmo_to_grid`, `sonar_scan_math`,
  `make_sonar_data`) are already hardened: vectorized bincount scatter,
  two-stage point selection, reused device buffers, cached reflectivity upload,
  channel-2-only writes, and frame-skip dedup on the RtxAcoustic writer. The
  headroom is in the **render pipeline** (item 1), not the math.
- Host-side wins (like the shipped semantic-fetch change) reduce sim-thread
  stalls and help sensor/odometry timing jitter, but they do **not** move a
  GPU-render-bound framerate. Item 1 is the only lever that does.
