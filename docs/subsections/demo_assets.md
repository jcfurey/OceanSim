# Ready-to-run inspection and reef demos

The repository includes two portable OpenUSD environments with seeded terrain,
PBR materials and local textures. No external asset pack is needed. Units are
metres, Z is up, the water surface is at Z=0, and the vehicle spawns at
(-2, 0, -0.8), above the seabed at approximately Z=-2.75.

| Environment | Contents | Launch selection |
|---|---|---|
| Inspection site | Corroded pipeline, bolted flanges, riser and valve wheel, calibration panel, pier piles and grated deck, rock clutter | `builtin` |
| Rocky reef | Rippled silt, irregular rocks, kelp ribbons, marked survey transect and salvage case | `reef` |

Solid terrain, rocks and inspection structures have static collision geometry.
Fine decoration is visual only. Acoustic response is stored in each geometry's
`oceansim:reflectivity` attribute and registered as Isaac segmentation labels by
the standalone runner. The coefficients are illustrative material contrasts,
not measured acoustic calibration. The authored `Overview` and `Inspection`
cameras can also be selected when viewing the USD directly in Isaac Sim.

## Run with Docker

Build the supplied image once if it is not already installed:

```bash
docker build -t oceansim:6.1.0 .
docker compose up -d
docker compose logs -f oceansim
```

Wait for `simulation running; publishing ROS2 sensor data`. The first launch can
take several minutes while Isaac initializes its renderer and compiles shaders.
The container uses the current checkout, with caches and logs in `.local/`.
Compose defaults to Cyclone DDS and ROS domain **71** for standalone work. Match
`RMW_IMPLEMENTATION` and `ROS_DOMAIN_ID` in your subscribers, or set them before
launching to join your existing robot graph. With Zenoh, start your router first.

The Docker launchers disable the optional OmniHub cache by default, avoiding
daemon launch retries when loading local scenes. To use a working Hub service,
set `OMNICLIENT_HUB_MODE=shared` before launching. This switch follows
[NVIDIA's client-library configuration](https://docs.omniverse.nvidia.com/kit/docs/client_library/latest/index.html#hub).

```bash
# Switch to the reef scene.
OCEANSIM_ENVIRONMENT=reef docker compose up -d --force-recreate

# Stop the simulation.
docker compose down

# A finite smoke run (do this after stopping the running service).
docker compose run --rm oceansim \
  './scripts/run_deeptrekker_revolution.sh --environment builtin --max-steps 120'

# Interactive Isaac viewport and sensor windows, with X11 passthrough.
docker compose down
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=71 ./docker/run.sh -lc \
  'cd /isaac-sim/extsUser/OceanSim && ./scripts/run_deeptrekker_revolution.sh --environment builtin --no-headless'
```

The GUI starts on the scene's `Overview` camera with underwater camera and sonar
windows. To launch it in the background, prefix the GUI command with
`OCEANSIM_DETACH=1 OCEANSIM_CONTAINER_NAME=oceansim-gui`. This requires a valid
`XAUTHORITY` file for display authentication after the launcher returns.
Use `docker logs -f oceansim-gui` for logs and `docker stop oceansim-gui` to close it.

The demo preset in `demo/revolution.json` uses CPU PhysX, GPU RTX rendering, a
960x540 underwater camera, and a 960-pixel sonar render product. Sonar output
uses a 12 m working range, 2.5 cm range bins and 0.5 degree beam spacing. The
vehicle still uses the registry's estimated hydrodynamics and six thrusters.
Use `--camera-resolution 1920 1080` or your own `--config` for different fidelity.
The preset reads water level from the bundled scene metadata; MHL scenes
without that metadata keep the original Z=1.43389 surface height.

Check live ROS output and save camera/sonar captures from inside the service:

```bash
docker compose exec oceansim bash -lc \
  'source /opt/ros/jazzy/setup.bash && python3 scripts/check_demo_ros2.py'
```

This checks message delivery, a finite submerged vehicle pose, finite DVL
velocity, nonuniform camera pixels and nonzero sonar intensity. It sends no
vehicle or joint commands.

Useful ROS topics include `/clock`, `/robot_description`, `/joint_states`,
`/oceansim/robot/odom`, `/oceansim/robot/imu`, `/oceansim/robot/pressure`,
`/oceansim/robot/dvl/twist`, `/oceansim/robot/sonar` and the underwater camera
topics. See [platform bringup](ros2_platform_bringup.md) for QoS and TF setup.
For CAD sensor frames, run `robot_state_publisher` as described there; the
standalone runner avoids duplicating sensor frames owned by the URDF.

## Stage a detailed REVOLUTION vehicle

Without vehicle CAD, the existing generated URDF provides a runnable fallback.
To use the local Nautilus chassis, pivot head and claw STLs, build/stage assets
with a Python environment containing `usd-core`, `numpy` and `Pillow`:

```bash
python3 -m venv .venv-assets
.venv-assets/bin/pip install usd-core numpy Pillow
.venv-assets/bin/python scripts/build_demo_assets.py \
  --revolution-meshes /path/to/description/meshes
```

The builder copies the CAD into `demo/assets/DeepTrekker/meshes`, generates
`revolution.urdf` with a driven `pivot_head_joint`, and creates a static
`revolution_preview.usdc` for scene layout. Camera and sonar frames attach to
the head, so they follow its motion. Copied CAD and provenance are local and
gitignored; each checkout can use its own source meshes. Meshes are approximate
visuals, and staging them does not replace the registry's physical model. The
claw is a fixed visual assembly in this demo.

The launcher finds this URDF automatically. An external asset pack can still
be selected with `OCEANSIM_ASSETS`; the `auto` environment selector prefers its
MHL scan when available. Compose uses the bundled assets by default; to mount
an external pack use the existing `docker/run.sh` helper.

## Rebuild and preview the environments

```bash
.venv-assets/bin/python scripts/build_demo_assets.py
OCEANSIM_HEADLESS=1 ./docker/run.sh -lc \
  'cd /isaac-sim/extsUser/OceanSim && /isaac-sim/python.sh scripts/render_demo_preview.py --scene demo/assets/environments/inspection_site.usdc --vehicle demo/assets/DeepTrekker/revolution_preview.usdc --output .local/previews/inspection_site.png'
```

Omit `--vehicle` if no CAD is staged. Use `--camera Inspection` for the vehicle's
forward inspection view, or change `--scene` to the reef USD. The preview uses
Isaac's RTX renderer and does not require ROS. Generated environment assets and
textures are original procedural content, under this repository's BSD license.
