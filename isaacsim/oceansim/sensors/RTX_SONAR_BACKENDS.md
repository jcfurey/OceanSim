# OceanSim sonar backends: `oceansim` / `rtx_acoustic` / `rtx_lidar`

Three interchangeable imaging-sonar implementations, selectable at runtime. This
doc records *why* (a measured perf breakthrough), the *shared interface*, the
*target geometry*, and the *remaining work* per backend.

## TL;DR — the perf case

The original `oceansim` sonar subclasses Isaac's `Camera` and renders a
**2500×384 RTX camera + AOVs every scan**, then reconstructs a point cloud and
bins it. That render is ~95% of the GPU.

Measured 2026-06-29 (DeepTrekker, full Isaac GUI, RTX 3090):

| backend | GPU util | odom | imu | sonar |
|---|---|---|---|---|
| `oceansim` (Camera+AOV) | **100%** | ~16 Hz | ~15 Hz | ~1.3 Hz |
| `rtx_acoustic` (native) | **1–8%** | **84 Hz** | **76 Hz** | **6.4 Hz** |

`rtx_acoustic` raytraces the beam fan on the RT cores for almost nothing — **with
the full GUI on**. So the odom ceiling was the sonar camera render, not the GUI.
The RTX backends are the performance path; their remaining work is **calibration /
fidelity, not speed**. (`OCEANSIM_SONAR_BACKEND=rtx_acoustic` reproduces this.)

## Selection + shared interface

Pick via `sonar_backend: oceansim | rtx_acoustic | rtx_lidar` in scenario.json,
`--sonar-backend`, or `OCEANSIM_SONAR_BACKEND`. Dispatch is the `if sonar_backend
== ...` block in `standalone/oceansim_ros2.py`.

Every backend must implement (consumed by `scenario.py` + `OceanSimSensorPublisher`):

- `__init__(prim_path, translation, orientation, range_res, angular_res, ...)`
- `sonar_initialize(output_dir=None, viewport=True, include_unlabelled=False)`
- `make_sonar_data(...)` — refresh the result; called each enabled sim step
- `sonar_map` (numpy or warp `(n_range, n_beams, 3)`, channel 2 = intensity) and/or
  `get_sonar_map_np()` — read by the publisher
- `get_range() -> [min, max]`, `get_fov() -> [hori_deg, vert_deg]`
- `close()`

## Target geometry — Blueprint Subsea Oculus M-series

Reference: `src/packages/utilities/liboculus` (per-ping `BearingData`,
`ping_to_sonar_image.h`) + `src/packages/drivers/oculus_sonar_driver`. The sim
currently approximates:

- **Azimuth FOV** ~130° (HF). Current sim: `angular_res 0.25° × 520 beams = 130°`.
- **Beams** ~512 (HF) / 256 (LF). liboculus reads `ping()->nBeams` + a real bearing
  table (non-uniform) — the authoritative azimuths for validation.
- **Elevation beamwidth** 12–20° (liboculus `ElevationBeamwidthDeg`) — thin slab.
- **Range** configurable: `range_res` (e.g. 0.005 m) × `n_range = (max-min)/res`.
- **Frequency** 1.2 MHz (LF) / 3.0 MHz (HF), `freq_mode` in the oculus params.

## Backend 1 — `oceansim` (Camera + AOV)  [DONE, slow, validated]

Geometric. Renders depth/normals/semantic AOVs of a 2500×384 camera, reconstructs
the cloud via `Camera.get_pointcloud()` (depth fallback — the source of the benign
`pointcloud annotator not attached` log spam), bins to `(n_range, n_beams)`.
Calibrated and correct; GPU-heavy. **Keep as the reference** to cross-check the RTX
backends. (hori_res + gpu_point_filter + the opt-in async worker live here.)

## Backend 2 — `rtx_acoustic` (native `AcousticSensor`)  [FAST, needs calibration]

`isaacsim.sensors.experimental.rtx` `Acoustic`/`AcousticSensor` over an
`OmniAcoustic` prim; a `GenericModelOutput` writer returns per-sample
`(rx, amp, t_ns)`. Range = `sound_speed · t / 2` — already **physically correct**.

Calibration gaps (all in `RtxAcousticSensor._build_acoustic_attributes` +
`utils/rtx_acoustic_math.fold_gmo_to_grid`):

0. **CAPTURE IS BROKEN (do this first).** Measured 2026-06-29: the
   `OceanSimAcousticGmoSink` writer never fires — `writer frames=0, latest=none`
   for the whole run, no `FIRST VALID acoustic frame` log line ever. So the backend
   publishes **all-zero** frames at the publish rate, and the 5% GPU / 84 Hz was
   measured with the acoustic sensor effectively *idle*. Root cause is the
   Replicator capture not being driven for the acoustic render product:
   `rep.orchestrator.set_capture_on_play(True)` + the runner's
   `world.step(render=True)` is not delivering GMO frames to the attached writer,
   and the code deliberately avoids `rep.orchestrator.step()` ("renderer failed to
   advance").

   ROOT CAUSE (traced 2026-06-29): Isaac's own
   `standalone_examples/api/isaacsim.sensors.experimental.rtx/create_acoustic_basic.py`
   drives its loop with **`simulation_app.update()`**, which pumps the Replicator
   SDG pipeline that fires the writer. OceanSim's runner uses **`world.step(render=
   True)`** instead, which steps physics + renders the main views but does NOT tick
   the acoustic sensor's own RTX render product / SDG graph -> the writer never
   captures. Confirmed by the live warning `UsdContext::createRunloopThreadForHydra
   Engine ... type: rtx tickRateInHz: 0` (the acoustic hydra engine is never ticked)
   and `get_data("generic-model-output")` returning an invalid GMO magic under this
   loop. So BOTH delivery paths are broken specifically because the acoustic render
   product isn't driven by `world.step`.

   Fix direction: in the runner loop, for the rtx_acoustic backend, additionally
   drive the SDG/render for the acoustic render product (e.g. `simulation_app.
   update()` after `world.step`, or a reconciled `rep.orchestrator.step(...)` that
   doesn't fight `world.step`'s render control) so the writer captures -- then
   verify the `FIRST VALID acoustic frame` log appears + sonar_map is non-zero.
   Watch perf after: the 5% GPU / 84 Hz was with the sensor IDLE; the working cost
   (still RT-core, so far cheaper than the Camera backend) is unknown until capture
   runs. Only then are the geometry/beamforming items below testable.

   UPDATE (headless test, 2026-06-29): running rtx_acoustic with OCEANSIM_GUI=false
   did NOT fix capture -- still no `FIRST VALID` / `writer frames`, and the
   `tickRateInHz: 0` + `failed to create Hydra Engine thread` warnings persist
   headless. So it is NOT a GUI/viewport conflict; the acoustic sensor's hydra
   engine ticks at 0 regardless (created with tick_rate=30 -- the rate isn't
   reaching the engine), so the render product never renders. This is a deeper
   experimental-API integration issue (likely needs rep.orchestrator driving the
   SDG, an explicit render-product/tick wiring, or an Isaac fix) -- uncertain,
   multi-cycle. PRAGMATIC ALTERNATIVE: `rtx_lidar` via the STANDARD, battle-tested
   `isaacsim.sensors.rtx` RTX-Lidar API (not the experimental one) is far more
   likely to capture cleanly under `world.step` + reuses the validated geometric
   binning -- it is probably the faster route to a working fast sonar, with
   rtx_acoustic as a later physical-fidelity track once its capture is solved.

   UPDATE (isolated-example test, 2026-06-29): Isaac's own create_acoustic_basic.py
   run in THIS image (oceansim:6.0.1) produces data immediately -- "First data
   received at frame 0, numElements=640" -- with NO tickRate/Hydra-Engine warnings.
   So the sensor WORKS in 6.0.1; OceanSim's integration is the bug. The hydra-engine
   warnings appear ONLY in OceanSim, where the acoustic render product is created
   alongside the UW-camera + main-viewport render products -> a render-product /
   hydra-engine CONFLICT (the acoustic render product can't get its engine thread,
   so it never renders, so the writer never fires). NOT primarily the world.step
   loop driver. Likely fixes: create the acoustic render product so it doesn't
   fight the existing ones (creation order, a compatible engine config, matching
   tickRate/deviceMask), or run rtx_acoustic without the UW camera. NB Isaac's test
   needs ~180 frames before first data; the example drives via simulation_app.
   update(), so the loop may still need an explicit SDG pump once the conflict is
   resolved. The `isaac-sim-remote` skill (port 8226) can inspect the live
   render-product/engine state interactively.

   CONCLUSION (2026-06-29, debugged to ground): the render-product/engine conflict
   is FATAL and is with `World`/`SimulationContext` itself, not the camera or the
   loop driver. Tested and ruled out: GUI vs headless (same), camera ON vs OFF
   (same -- 2131 warnings, no data either way), and adding `simulation_app.update()`
   after `world.step` to drive the full SDG (same -- 2226 warnings, no data). The
   acoustic sensor's render product cannot create its hydra engine thread while
   `World` owns the rendering (deviceMask/tickRate mismatch in the warning). The
   isolated example works ONLY because it never creates a `World` (raw
   `SimulationApp` + `omni.timeline` + `simulation_app.update()`). So fixing
   rtx_acoustic needs a SUBSTANTIAL effort: either restructure the runner to drive
   physics + the acoustic sensor without `World` (breaks the rest of OceanSim, which
   is built on SimulationContext tensor views for control/odom/IMU), or an
   Isaac-side fix so the experimental acoustic render product coexists with `World`.
   NOT a quick integration fix. => `rtx_lidar` via the STANDARD `isaacsim.sensors.
   rtx` API (designed to run under `World` with annotators + `world.step`) is the
   recommended path to a working fast sonar; revisit rtx_acoustic if/when the
   World-coexistence issue is resolved upstream.

1. **Receiver array** — currently 8 placeholder elements at 2 cm spacing across the
   FOV. Replace with the real Oculus receiver geometry (element count + spacing);
   this sets the achievable azimuth resolution.
2. **Beamforming (the core gap)** — `fold_gmo_to_grid` maps ~8 receivers onto 520
   beams, so azimuth is quantized to ~8 values. Need delay-and-sum beamforming
   across the receiver elements to synthesize the 520 beams — or configure the
   `AcousticSensor` to emit beam-resolved GMO directly (check `aux_output_level` /
   `rxGroup` semantics).
3. **Intensity** — map GMO amplitude → sonar-intensity scale (vs the reflectivity
   model the `oceansim` backend uses).
4. **Validation** — compare range/azimuth/intensity vs `oceansim` on the same scene,
   and vs real Oculus pings (rosbag via `oculus_sonar_driver`) if available.

Effort: **high** (acoustic beamforming). Highest physical fidelity (multipath, TOF).

## Backend 3 — `rtx_lidar` (RTX Lidar → grid)  [FAST, simplest to make correct, NEW]

Isaac `isaacsim.sensors.rtx` RTX Lidar with a **custom scan = sonar FOV** (azimuth
−65°..+65° at `angular_res`, a thin/single elevation row, `max_range`); read
per-ray range + intensity from the RTX-lidar annotator (RT cores, **no camera**) and
bin into `(n_range, n_beams)` exactly like `oceansim` does from depth.

Why it's attractive: it reuses the **already-validated geometric model** of the
`oceansim` backend but at RTX-lidar speed — the quickest path to a "fast +
correct-enough" sonar while `rtx_acoustic` beamforming is calibrated.

Tasks:

1. New `RtxLidarSensor` class (interface parity above).
2. RTX-lidar scan config (Isaac lidar JSON / programmatic): az span = FOV,
   `horizontalResolution = angular_res`, single elevation, `maxRange`.
3. `make_sonar_data`: read the lidar `range`/`intensity` annotator, bin to
   `(n_range, n_beams)` (reuse the `oceansim` binning kernels where possible).
4. Wire `sonar_backend == "rtx_lidar"` in `oceansim_ros2.py`.
5. Validate vs `oceansim`.

Effort: **medium**. Geometric (no acoustic speckle/multipath) but fast + direct.

## Recommended sequencing

1. **`rtx_lidar` first** — medium effort, fast + geometrically correct, an immediate
   usable ~5×-class win that reuses the validated binning.
2. **`rtx_acoustic` calibration** — high effort, the physical-fidelity track.
3. **Keep `oceansim`** as the validated reference for cross-checking both.
