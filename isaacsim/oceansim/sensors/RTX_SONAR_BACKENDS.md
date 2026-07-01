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

> **CAVEAT: the 84 Hz / 6.4 Hz `rtx_acoustic` row above was measured with the
> acoustic sensor effectively IDLE** (capture was broken at the time — see the
> SOLVED block below). With capture + calibration actually working, measured
> throughput (headless, full DeepTrekker + camera + sonar pipeline, 2026-07-01)
> is **odom ~22 Hz, sonar/drawn_sonar ~1.9 Hz** — still well above `oceansim`'s
> ~16 Hz / ~1.3 Hz, but nowhere near the idle-sensor numbers. Re-measure GPU
> util once a proper before/after comparison matters.

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

   CONCLUSION 2 (fork-patch attempt, 2026-06-29): also ruled out `/app/asyncRendering
   = False` (force synchronous rendering so the render product doesn't spawn its own
   engine thread) -- still 2034 warnings, no data. FOUR Python/config levers now
   exhausted (GUI/headless, camera on/off, simulation_app.update() SDG pump,
   asyncRendering off). CRUCIALLY: the failing call `UsdContext::createRunloopThread
   ForHydraEngineImpl` is in PRECOMPILED omni.usd / Kit CORE -- it is NOT in the
   IsaacSim fork source (grep confirms). So "patch the Isaac fork" CANNOT fix this:
   the broken code isn't in the fork; it ships as a compiled Kit plugin. Real
   options: (1) `rtx_lidar` via the standard/deprecated `isaacsim.sensors.rtx`
   `LidarRtx` (source/deprecated/isaacsim.sensors.rtx/python/impl/lidar_rtx.py),
   which uses a different creation path and runs under `World` -- RECOMMENDED;
   (2) file an upstream NVIDIA bug (experimental `AcousticSensor` render product
   can't create its hydra engine thread under `World`/`SimulationContext`; min repro
   = the example + `World()`); (3) the big runner restructure (drive without
   `World`). Do NOT spend more cycles on Python/carb levers for rtx_acoustic.

   ===========================================================================
   *** SOLVED (2026-06-29) -- both CONCLUSIONs above were WRONG. ***
   It was NOT World, NOT a Kit-core conflict, NOT a fork patch. ROOT CAUSE: one
   line in `RtxAcousticSensor.sonar_initialize` flipped a RENDER SETTING AT
   RUNTIME -- `carb.settings.set("/renderer/raytracingMotion/enabled", True)`.
   Changing /renderer/* after the renderer + Kit viewport already exist forces a
   hydra-engine RECONFIGURATION; with the viewport + sonar render products both
   live it fails to spawn new engine threads (deviceMask 0) -> the viewport AND
   the sonar silently stop rendering (the thousands of "failed to create Hydra
   Engine thread for viewport" warnings + empty GMO frames). The whole
   "World-incompatible" story was a red herring -- the isolated example worked
   only because it never sets raytracingMotion at runtime.

   How it was found: built a minimal standalone repro from create_acoustic_basic.py
   and added OceanSim's factors ONE AT A TIME (World, world.step drive, capture-on-
   play, RayTracedLighting, ros2.bridge, a 2nd camera render product, tick_rate=30,
   aux_output_level=BASIC, in-process URDF import, the 49MB MHL scene, robot physics,
   sonar-under-articulation, SingleArticulation.initialize(), an rclpy/rmw_zenoh
   node, real 8-mount attrs). ALL passed; the boot kit-args were byte-identical to
   OceanSim. The ONLY thing left was the mid-run raytracingMotion set -> adding it
   reproduced the exact failure (33k createViewport failures); moving it to a BOOT
   kit-arg fixed it (0 warnings, FIRST DATA frame 0).

   THE FIX (committed): the standalone runner (`oceansim_ros2.py`) appends
   `--/renderer/raytracingMotion/enabled=True` to sys.argv BEFORE `SimulationApp`
   (only for `sonar_backend == rtx_acoustic`), so the renderer boots with Motion
   BVH already on -- no mid-run reconfiguration. `RtxAcousticSensor.sonar_initialize`
   no longer touches that setting. Verified on the real full-scenario sim:
   `FIRST VALID acoustic frame 1: numElements=2560`, sonar publishing on ROS ~2 Hz,
   odom ~22 Hz, 0 hydra warnings.

   ===========================================================================
   *** RANGE/TIMING CALIBRATED (2026-06-30) -- the timeOffsetNs=0 blocker is FIXED. ***
   The acoustic GMO is NOT a per-sample point cloud: it is SIGNAL WAYS (A-scans).
   `numElements = numSgws * numSamplesPerSgw`, laid out as numSgws contiguous
   row-major blocks; each block is one signal way's amplitude envelope vs SAMPLE
   INDEX. Range lives in the sample index, NOT in `timeOffsetNs` (always 0 for
   acoustic -- Isaac's own examples never read it). The fix (committed):
   `fold_gmo_to_grid` reshapes scalar -> (numSgws, numSamplesPerSgw) and maps
   sample k -> range = range_offset + k * meters_per_sample, with
   `meters_per_sample = c_sensor * sampleDuration / 2`. Measured constants (isolated
   GMO dump `dump_acoustic_gmo.py`, 4 point targets @ known range/azimuth; fit R vs
   peak-sample-index):
     - `numSamplesPerSgw = 320` -- ONLY populated at aux_output_level="BASIC"; the
       GMO header maxRangeM/minAzRad/numSgws... are 0 unless BASIC is set.
     - `sampleDuration = 1.024e-4 s` -- a READABLE prim attr
       (`omni:sensor:WpmAcoustic:sampleDuration`), read at sonar_initialize.
     - The sensor models AIR, **c_sensor ~= 343 m/s** (NO sound-speed attr exists) ->
       meters_per_sample ~= 0.01756 (empirical fit 0.01754). So the range window is
       only **~6 m** (320*0.01756) -- a HARD CAP for an underwater sonar wanting
       10 m+ (numSamplesPerSgw is not obviously settable). OceanSim sound_speed=1500
       is IRRELEVANT here; `sensor_sound_speed=343` is a new RtxAcousticSensor ctor arg.
     - `range_offset ~= 0.40 m` (~= c_sensor * pulseDuration/2, pulseDuration=2.5e-3 s
       attr -- the pulse-length echo delay).
   Verify on the full sim: the `FOLDED grid` log (lit cells + range-bin extent, proves
   the image is no longer collapsed) and the `FIRST VALID` line (numSgws,
   meters_per_sample).
   ===========================================================================

   ===========================================================================
   *** M300D CALIBRATION + PROJECTEDSONARIMAGE FORMAT ALIGNED (2026-06-30). ***
   Items 1 and 3 below are DONE, and item 2 (beamforming) is DIAGNOSED AS NOT
   ACHIEVABLE with this sensor's data — not a remaining task, an accepted physical
   limit. Details:
   - **Receiver array capped, FOV corrected.** `azSpanDeg`/`elSpanDeg` are real
     settable acoustic attrs (schema default 90) — set to the Oculus M300d LF FOV
     130x20 deg. Receiver-mount count has a **hard cap <256**: 256/512 mounts fail
     with RTXMemUtil "buffer with size 0" (no data); 128 is the usable max. So
     azimuth resolution is fixed at **128 signal ways -> 128 of the 520 output
     beams lit**, one per receiver (not per-element rx ids, which stay unreliable
     at BASIC).
   - **True delay-and-sum beamforming is NOT possible**: the GMO `scalar` is an
     all-positive ENVELOPE (min = noiseMin, no phase information), and the
     receivers are directional elements on a ~2cm array. Without phase you cannot
     synthesize additional beams from the 128 physical receivers — more beams
     would require more receiver mounts, which are capped at 128. **For a full
     520-beam image, use the `oceansim` backend.** This closes item 2 as a known
     limitation rather than an open task.
   - **ProjectedSonarImage format aligned** to `oculus_sonar_driver/
     ping_to_sonar_image.h`: `ranges` are bin CENTRES `(i+0.5)*range_res`;
     `rx_beamwidths`/`tx_beamwidths` are the real per-frequency Oculus values
     (`ros2_math.oculus_beamwidths()`) — 0.6 deg az / 20 deg el @ 1.2 MHz — instead
     of beam spacing. `beam_directions` and range-major uint8 layout already
     matched. This closes item 3.
   - **KNOWN GAP (not yet touched):** `ping_info.sound_speed` is always published
     from the scenario's general `publisher.sound_speed` (1500.0, underwater), but
     `rtx_acoustic`'s actual range mapping uses `sensor_sound_speed=343` (air —
     the sensor has no underwater acoustic model). No current consumer
     (`sonar_image_proc`, `erdc_sensor_fusion`) reads `ping_info.sound_speed` yet
     (even `oculus_sonar_driver`'s own field is an unset TODO), so this is latent,
     not a live bug — but fix it before anything downstream starts trusting that
     field.
   ===========================================================================
   *** LIVE END-TO-END VERIFICATION (2026-07-01), full `oceansim.launch.py` stack,
   DeepTrekker Revolution platform. *** Confirmed exactly per the calibration
   above: `RtxAcousticSonar` log = `128 receiver mounts ... grid 520 beams x 1980
   range`, `FOLDED grid: 40960 lit cells, range bins [66,1186] = [0.43,6.03] m,
   beams hit=128/520`. `/oceansim/robot/sonar` (`ProjectedSonarImage`): `ranges`
   len 1980, `beam_directions`/`tx_beamwidths`/`rx_beamwidths` len 520,
   `rx_beamwidths`=0.010472 rad (0.6 deg), `tx_beamwidths`=0.349066 rad (20 deg) —
   matches the M300d LF beamwidths exactly. `drawn_sonar` renders a real (sparse)
   fan image, 1980x3590 rgb8, ~21% nonzero pixels (consistent with 128/520 beams
   lit). Perf with the full DeepTrekker + camera + sonar pipeline: odom ~22 Hz,
   sonar/drawn_sonar ~1.9 Hz — matches the 2026-06-30 M300d-calibration figures,
   no regression from the later capture/watchdog/safety commits. Not yet done:
   item 4 (formal side-by-side numeric comparison vs the `oceansim` backend on an
   identical scene, and vs a real Oculus rosbag).

   ===========================================================================
   *** TWO MORE REAL BUGS FOUND + FIXED, PLUS A HARD GPU-MEMORY CEILING
   DISCOVERED (2026-07-01, live interactive debugging — rotating the vehicle in
   the GUI and cross-checking the raw ProjectedSonarImage grid, not just the
   rendered image). The "beams hit=128/520" log line above is misleading: it
   counts any nonzero cell, including noise-floor residue (amplitude ~1), so it
   looked fine even when only 2 of 128 elements had a real echo (amplitude
   200+). ***

   **Bug A — mount pose was a one-time WORLD-frame teleport with no orientation,
   in `RtxAcousticSensor.sonar_initialize`.** `set_world_poses(positions=
   self._translation...)` fed a small LOCAL mount-offset vector (e.g. (0.06, 0,
   0.04)) into a WORLD-frame position setter, once, and never set orientation at
   all. Symptom: rotating the vehicle 90 deg in the GUI did not change which
   beams saw a return. Fix: `set_local_poses(translations=..., orientations=...)`
   — LOCAL-frame xformOps relative to the parent, exactly like `UW_Camera`'s
   constructor args — so the sensor now rotates/translates with the robot body
   via ordinary USD parenting. Verified: after a confirmed 90 deg body yaw
   (odom-tracked), the sonar's active beams follow.

   **Bug B — no `firingSeq` was ever authored, so only the schema's own default
   single event (`txSensorId=[0]`, `rxGroupId=[0]`) ever fired.** `sensorMount`
   and `rxGroup` only DEFINE the array geometry; a separate `firingSeq`
   multi-apply schema (`omni:sensor:WpmAcoustic:firingSeq:*`, arrays
   `eventTimeNs`/`txSensorId`/`rxGroupId`/`channel`) enumerates which (tx
   element, rx group) pairs actually fire each cycle. `Acoustic._create_prim()`'s
   auto-apply table (in the installed `isaacsim.sensors.experimental.rtx`
   package) only recognises the `sensorMount:`/`rxGroup:` prefixes, not
   `firingSeq:`, so it has to be applied and populated by hand after
   construction (`prim.ApplyAPI("OmniSensorWpmAcousticFiringSeqAPI", "seq1")` +
   setting the four array attrs). Without it, 126 of 128 configured elements
   were never actually pinged — confirmed by amplitude, not just cell count:
   with 128 elements, only columns 0 and 4 (mount indices 0 and 1) ever showed
   amplitude >1; all other "hit" columns were noise floor. After authoring one
   event per rxGroup pair (element g -> rxGroup [g,g+1]), ALL configured
   elements produce real amplitude data.

   **New hard constraint found while fixing B: each concurrent firing event
   costs roughly ~1.1-1.2 GB of GPU VRAM** (measured on a 24 GB RTX 3090,
   headless, isolating the variable by re-running with only `sonar_params.
   n_elements` changed): 8 elements (7 events) -> ~10.3 GB total, clean; 12
   elements (11 events) -> ~17.4 GB, clean but close to the ceiling; 16 elements
   (15 events) and 32 elements (31 events) both blow past available VRAM --
   `ERROR_OUT_OF_DEVICE_MEMORY`/`vkAllocateMemory failed`, and/or a
   `cudaErrorIllegalAddress` crash that can occur even before VRAM is fully
   exhausted. GPU memory fully releases when the container stops (confirmed, not
   a leak) -- this is real per-run cost, apparently because each firing event
   allocates something render-product-sized rather than a lightweight scalar
   buffer. **This revises the earlier "128 receiver mounts is the usable max"
   finding down substantially**: 128 was the ceiling for a DIFFERENT failure
   ("buffer with size 0" at 256/512 mounts); the GPU-memory ceiling found today
   is far more restrictive in practice (~12 elements on this GPU) and was masked
   before today because bug B meant only 1 event ever actually fired regardless
   of how many elements were configured. Not yet investigated: whether a
   different `aux_output_level`, resolution, or other per-event setting reduces
   the per-event memory cost; whether newer/bigger GPUs raise the practical
   ceiling proportionally with more VRAM.
   ===========================================================================

1. ~~Receiver array / azimuth source~~ — DONE (see M300D CALIBRATION above).
2. ~~Beamforming (the core gap)~~ — DIAGNOSED AS NOT ACHIEVABLE with the envelope-
   only GMO. Additionally (2026-07-01): even setting aside beamforming, the
   practical element/beam ceiling on a 24 GB GPU is now ~12 (GPU-memory bound
   per firing event), not the previously-assumed 128 -- see the 2026-07-01 block
   above.
3. ~~Intensity~~ — DONE (ProjectedSonarImage format alignment above).
4. **Validation (remaining)** — formal side-by-side comparison of range/azimuth/
   intensity vs `oceansim` on the same scene/target placement, and vs real Oculus
   pings (rosbag via `oculus_sonar_driver`) if available. Live functional
   verification (2026-07-01 above) confirms the pipeline produces sane, correctly-
   shaped, correctly-calibrated output — but no numeric ground-truth comparison
   has been run yet.

Effort: **medium** remaining (just validation — the beamforming ceiling means this
backend caps out at 128/520 beams; if 520-beam resolution matters, use `oceansim`
instead). Highest physical fidelity of the three backends (multipath, TOF) within
that beam-count ceiling.

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

===========================================================================
*** BLOCKED (2026-07-01) -- isolated smoke test never got a valid GMO frame.
Parked; not started on the real `RtxLidarSensor` class. ***

Schema research (via the `omniSensorGenericLidarCoreAPI.h` /
`...EmitterStateAPI.h` C++ headers in the image, the same way `firingSeq` was
found for the acoustic sensor) found a promising-looking custom-scan path:
`omni:sensor:Core:scanType="SOLID_STATE"` (no mechanical rotation),
`omni:sensor:Core:instantLidar=true` ("produce one full scan at frame end
timepoint disregarding actual timing and scan rates"),
`omni:sensor:Core:elementsCoordsType="SPHERICAL"` (native range/az/el, no XYZ
reconstruction), and a `MultipleApplyAPI` `EmitterState` schema taking direct
parallel arrays (`azimuthDeg[]`/`elevationDeg[]`/`fireTimeNs[]`/`channelId[]`)
for a fully custom ray pattern -- authored via
`isaacsim.sensors.experimental.rtx.Lidar(path, attributes={...})`, same
pattern as `Acoustic`.

Two real API gotchas found and fixed along the way (kept here in case anyone
retries this):
- Array-valued attributes in the `attributes=` dict must be `pxr.Vt.FloatArray`/
  `Vt.UIntArray` objects, not plain Python lists -- the Replicator helper
  (`omni.replicator.core...functional.create_batch.omni_lidar` ->
  `modify.set_attributes`) does `python_class(*v)` for a `list` value, which
  *unpacks* it as individual positional args and throws a `Boost.Python.
  ArgumentError` for any array longer than a few elements.
- `channelId` is 1-indexed (`channelId 0 at index 0 is greater than the
  numberOfChannels 128 or less than 1` if you use `range(n)`).
- The same Motion-BVH boot-arg gotcha as the acoustic sensor applies:
  `--/renderer/raytracingMotion/enabled=True` must be a `sys.argv` entry set
  BEFORE `SimulationApp()` is constructed, or you get the "Multi-tick is
  enabled but motion BVH is not active" warning and an empty frame.
- `get_data("generic-model-output")` is unreliable here too (same lesson as
  acoustic) -- use a Replicator `Writer` (`attach_writer`) instead.

None of that was enough to get real data, though. With all of the above
correctly applied (verified via `rep.orchestrator.set_capture_on_play(True)`
too), the writer never fired a single frame -- not even an empty one -- under
EITHER driving method tried:
- `World.step(render=True)` (matches OceanSim's actual runner): 0 writer
  frames after 55 steps.
- `simulation_app.update()` (matches the official reference test
  `isaacsim.sensors.experimental.rtx.tests.test_lidar_sensor.
  TestLidarSensor.test_gmo_writer`'s driving pattern): still 0 writer frames,
  AND a severe unexplained per-call slowdown appeared around iteration 85
  (2.9s -> 26.5s -> 4s per `update()` call, CPU pegged at 99%, GPU ~idle at
  2%) with no data ever appearing before the run was killed.

The reference test itself (only found AFTER the above attempts) uses the
BARE DEFAULT config (no custom `scanType`/`instantLidar`/`EmitterState` at
all -- just `outputFrameOfReference: "WORLD"` + `aux_output_level: "FULL"`),
driven by `omni.timeline.play()` + `await next_update_async()` for a full
3 SECONDS (180 updates at 60fps) before asserting `valid_frame_count > 0` --
i.e. it tolerates many empty frames before a valid one appears, which our
smoke test's much shorter windows (55-200 iterations, no real timeline) may
not have given long enough a runway even ignoring the custom-scan question.

**Not yet tried, and the logical next step if this is picked back up:** run
the reference test's EXACT recipe verbatim (default config, `timeline.play()`
+ `next_update_async()`, 3s window) first to confirm the baseline pipeline
captures at all in this image, then reintroduce the custom
`SOLID_STATE`/`instantLidar`/`EmitterState` attributes one at a time to find
which one (if any) breaks capture, before writing `RtxLidarSensor` for real.
===========================================================================

## Recommended sequencing

1. **`rtx_lidar` first** — medium effort, fast + geometrically correct, an immediate
   usable ~5×-class win that reuses the validated binning.
2. **`rtx_acoustic` calibration** — high effort, the physical-fidelity track.
3. **Keep `oceansim`** as the validated reference for cross-checking both.
