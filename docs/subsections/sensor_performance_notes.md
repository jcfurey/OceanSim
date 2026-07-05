# Sensor Performance Notes

Working notes on where the OceanSim sensor pipeline spends time and which
optimizations are done vs. still on the table. Focused on the sonars (the
dominant cost) but the camera / DVL paths are noted too. Targets Isaac Sim
6.0.1.

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

### 4. `fold_gmo_to_grid` dtype (micro, low value)
`rtx_acoustic_math.fold_gmo_to_grid` upcasts the already-`float32` GMO amplitude
buffer to `float64` for `np.abs` + the scatter. Keeping it `float32` halves the
intermediate footprint, but `np.bincount` weights return `float64` regardless
and the wider accumulation is slightly more accurate. **Verdict:** not worth the
precision trade; noted only for completeness.

---

## Outstanding — camera / DVL (unaudited)

Not yet reviewed for performance this pass; listed so they aren't forgotten:

- **`UW_Camera`** — check for redundant annotator `get_data()` host readbacks
  and per-frame allocations on the publish path, mirroring the sonar audit.
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
