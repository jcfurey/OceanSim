# OceanSim Installation Documentation
We design OceanSim as an extension package for NVIDIA Isaac Sim. This design allows better integration with Isaac Sim and users can pair OceanSim with other Isaac Sim extensions. This document provides a step-by-step guide to install OceanSim.

## Prerequisites
OceanSim does not enforce any additional prerequisites beyond those required by Isaac Sim. Please refer to the [official Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html#system-requirements) for the prerequisites.

OceanSim is now compatible with Isaac Sim 6.1.0 (6.0.1 also works). Due to the changes in recent Isaac Sim releases compared to previous versions, the OceanSim main branch release may not work with older versions of Isaac Sim.

We have tested OceanSim on Ubuntu 20.04, 22.04, and 24.04. We have also tested OceanSim using various GPUs, including NVIDIA RTX 3090, RTX A6000, and RTX 4080 Super, TX 5070Ti. 

## Installation
For Isaac Sim 6.1.0, we build from their [source code](https://github.com/isaac-sim/IsaacSim). If you plan to use the [ROS2 Bridge](../README.md#ros2-bridge), set up your ROS2 workspace by following the official [Isaac Sim ROS 2 installation tutorial](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_ros.html) (Ubuntu 24.04 + ROS 2 Jazzy by default; ROS 2 Humble on Ubuntu 22.04 is also supported).



Clone this repository to your local machine. We recommend cloning the repository to the Isaac Sim workspace directory.
```bash
cd /path/to/isaacsim/extsUser
git clone https://github.com/umfieldrobotics/OceanSim.git
```
`/extsUser` folder is guaranteed that the extension is discoverable in the extension browser of Isaac Sim.

Download `OceanSim_assets` from [Google Drive](https://drive.google.com/drive/folders/1qg4-Y_GMiybnLc1BFjx0DsWfR0AgeZzA?usp=sharing) which contains USD assets of robot and environment.

Then, run the following to configure OceanSim to point to your asset path:

```bash
cd /path/to/OceanSim
python3 config/register_asset_path.py /path/to/OceanSim_assets
```
For older releases (e.g. Isaac Sim 4.5), follow the official [workstation installation guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html) and check out the matching OceanSim release tag.

**NOTE**: The main branch is always the latest release and does not have backward compatibility due to Omniverse being a fast evolving ecosystem. 
Please download previous release and the installation is exactly the same as above.

### Isaac Sim 6.1.0 notes
Moving from 6.0.1 to 6.1.0 needs no OceanSim code changes beyond this release, but a few Isaac Sim behaviours changed:
- **TF aggregation**: the ROS 2 TF publisher nodes now merge their output into one `TFMessage` per topic (`isaacsim.ros2.nodes` setting `tfAggregation.enabled`, on by default). Launch with `--/exts/isaacsim.ros2.nodes/tfAggregation/enabled=false` for the old per-node messages.
- **`python.sh` sets up ROS itself**: when `ROS_DISTRO` / `RMW_IMPLEMENTATION` are unset it picks the bundled distro and `rmw_fastrtps_cpp`. The Docker image defaults to Zenoh and also installs Cyclone DDS; on bare metal export the `RMW_IMPLEMENTATION` used by your ROS 2 graph before running `scripts/run_oceansim_ros2.sh`.
- **CPU threads**: `SimulationApp`'s default `limit_cpu_threads` dropped from 32 to 16.

## Running in Docker (Isaac Sim 6.1.0 + ROS 2 Jazzy)
A [`Dockerfile`](../../Dockerfile) is provided that builds on NVIDIA's official Isaac Sim 6.1.0 container (Ubuntu 24.04) and layers ROS 2 Jazzy and OceanSim on top.

Prerequisites: an NVIDIA GPU with a recent driver, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), and an [NGC](https://catalog.ngc.nvidia.com/) login to pull the base image (`docker login nvcr.io`).

```bash
# Build the image
docker build -t oceansim:6.1.0 .

# Launch with GPU access and X11 display passthrough
./docker/run.sh
```

To start a headless Deep Trekker REVOLUTION simulation immediately after the
build, without downloading the optional asset pack:

```bash
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ./docker/run.sh -lc \
  'cd /isaac-sim/extsUser/OceanSim && ./scripts/run_deeptrekker_revolution.sh'
```

This uses the repository's textured subsea pipeline-inspection environment and
selects locally staged REVOLUTION CAD when available, otherwise generating a
URDF from the built-in vehicle dimensions, hydrodynamics and six-thruster layout.
See [demo assets](demo_assets.md) for CAD staging, Docker Compose, rendered
previews and the rocky reef environment. The command selects Cyclone DDS so a standalone run does not
need a Zenoh router. Omit the environment override when connecting to a
Zenoh-based robot stack, and ensure the stack's router is running. Set
`OCEANSIM_ASSETS=/path/to/OceanSim_assets` when launching the container to use
the scanned MHL environment and detailed `DeepTrekker/revolution.usd` model
instead. The launcher supports
`--environment auto|builtin|reef|mhl|/path/to/scene.usd`; `auto` is the default.

The [`docker/run.sh`](../../docker/run.sh) helper mounts the current checkout and handles display passthrough: when `DISPLAY` is set it runs `xhost +local:root`, forwards `DISPLAY`, mounts `/tmp/.X11-unix` and your existing `.Xauthority`, and requests the GPU. Set `OCEANSIM_HEADLESS=1` to skip X11 authorization. Caches live in `.local/isaac-sim`; set `OCEANSIM_CACHE_ROOT=$HOME/docker/isaac-sim` to reuse an older cache directory. Inside the container, start the GUI with `./isaac-sim.sh`. To pass downloaded USD assets, set `OCEANSIM_ASSETS=/path/to/OceanSim_assets` before running. The helper mounts that directory at `/isaac-sim/OceanSim_assets` and exports its container path for the REVOLUTION launcher. For the GUI extension, register it once with `python3 config/register_asset_path.py /isaac-sim/OceanSim_assets` from `/isaac-sim/extsUser/OceanSim`.

When you are done, you can revoke the X server grant with `xhost -local:root`.

### ROS 2 middleware (Zenoh, Cyclone DDS, or Fast DDS)
The container defaults to `RMW_IMPLEMENTATION=rmw_zenoh_cpp` so the sim joins the same Zenoh graph as the rest of the robot stack. The Docker image also includes Cyclone DDS, and Isaac Sim includes Fast DDS. `docker/run.sh` forwards an explicitly set `RMW_IMPLEMENTATION`, plus `ROS_DOMAIN_ID` and `CYCLONEDDS_URI`, into the container. When `CYCLONEDDS_URI` is a `file://` URI, the helper mounts that host file read-only at the same path inside the container:

```bash
# Cyclone DDS (no router required)
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ./docker/run.sh

# Fast DDS
RMW_IMPLEMENTATION=rmw_fastrtps_cpp ./docker/run.sh
```

With Zenoh, multicast discovery is **off** by default — nodes discover each other via a **Zenoh router**, which must be running before any ROS 2 node (the OceanSim publishers, RViz, `robot_localization`, `sonar_image_proc`, …) can see each other.

- In a real deployment the router belongs to the robot stack; its endpoint config (`ROS_DOMAIN_ID`, peer/router endpoints) is sourced at runtime from the workspace's `bashrc.d/99-zenoh_configs.bashrc`, not baked into the image.
- To run OceanSim **standalone** (a dev box / smoke test) where no stack router exists, start one in a separate terminal:
  ```bash
  ./scripts/start_zenoh_router.sh   # ros2 run rmw_zenoh_cpp rmw_zenohd
  ```

Sensor topics publish with `BEST_EFFORT` reliability (Zenoh-friendly for high-rate data); `/clock` is `RELIABLE`. The underwater camera's raw (`rgb8`) and depth (`32FC1`) image streams are large — disable them with `publish_image_raw=False` / `publish_depth=False` if only the compressed stream is needed over the wire.

**TF for standalone runs.** In a robot-stack deployment the URDF / `robot_state_publisher` owns the sensor frames, so the sim publishes no TF by default. To run OceanSim by itself (so RViz / `sonar_image_proc` can place the sonar and camera data), pass `--publish-static-tf` to the runner (or `"publish_static_tf": true` in the config) — it broadcasts latched `base_link → {sonar0/optical_frame, camera}` from the sensor mount poses. (DVL/barometer/IMU already report in `base_link`.)

## Launching OceanSim
There is no separate building process needed for OceanSim, as it is an extension. To load OceanSim: 
- IsaacSim, follow `Window -> Extensions`
- On the window that shows up, remove the `@feature` filter that comes by default
- Activate `OCEANSIM`
- You can now exit the `Extensions` window, and OceanSim should be an option on the IsaacSim panel. You can freely import OceanSim sensors and modules into your own Isaac Sim workflow.
