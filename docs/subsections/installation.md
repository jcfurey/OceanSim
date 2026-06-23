# OceanSim Installation Documentation
We design OceanSim as an extension package for NVIDIA Isaac Sim. This design allows better integration with Isaac Sim and users can pair OceanSim with other Isaac Sim extensions. This document provides a step-by-step guide to install OceanSim.

## Prerequisites
OceanSim does not enforce any additional prerequisites beyond those required by Isaac Sim. Please refer to the [official Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/requirements.html#system-requirements) for the prerequisites.

OceanSim is now compatible with Isaac Sim 6.0.1. Due to the changes in recent Isaac Sim releases compared to previous versions, the OceanSim main branch release may not work with older versions of Isaac Sim.

We have tested OceanSim on Ubuntu 20.04, 22.04, and 24.04. We have also tested OceanSim using various GPUs, including NVIDIA RTX 3090, RTX A6000, and RTX 4080 Super, TX 5070Ti. 

## Installation
For Isaac Sim 6.0.1, we build from their [source code](https://github.com/isaac-sim/IsaacSim). If you plan to use the [ROS2 Bridge](../README.md#ros2-bridge), set up your ROS2 workspace by following the official [Isaac Sim ROS 2 installation tutorial](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_ros.html) (Ubuntu 24.04 + ROS 2 Jazzy by default; ROS 2 Humble on Ubuntu 22.04 is also supported).



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
For Isaac Sim 4.5, we follow the official [workstation installation guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html).

**NOTE**: The main branch is always the latest release and does not have backward compatibility due to Omniverse being a fast evolving ecosystem. 
Please download previous release and the installation is exactly the same as above.

## Running in Docker (Isaac Sim 6.0.1 + ROS 2 Jazzy)
A [`Dockerfile`](../../Dockerfile) is provided that builds on NVIDIA's official Isaac Sim 6.0.1 container (Ubuntu 24.04) and layers ROS 2 Jazzy and OceanSim on top.

Prerequisites: an NVIDIA GPU with a recent driver, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), and an [NGC](https://catalog.ngc.nvidia.com/) login to pull the base image (`docker login nvcr.io`).

```bash
# Build the image
docker build -t oceansim:6.0.1 .

# Launch with GPU access and X11 display passthrough
./docker/run.sh
```

The [`docker/run.sh`](../../docker/run.sh) helper handles display passthrough for you: it runs `xhost +local:root` to authorize the container against your host X server, forwards `DISPLAY`, mounts `/tmp/.X11-unix` and your `.Xauthority`, and requests the GPU via `--runtime=nvidia --gpus all`. Inside the container, start the GUI with `./isaac-sim.sh`. To pass your downloaded USD assets, set `OCEANSIM_ASSETS=/path/to/OceanSim_assets` before running, then inside the container run `python3 config/register_asset_path.py /isaac-sim/OceanSim_assets` from the OceanSim directory (`/isaac-sim/extsUser/OceanSim`).

When you are done, you can revoke the X server grant with `xhost -local:root`.

## Launching OceanSim
There is no separate building process needed for OceanSim, as it is an extension. To load OceanSim: 
- IsaacSim, follow `Window -> Extensions`
- On the window that shows up, remove the `@feature` filter that comes by default
- Activate `OCEANSIM`
- You can now exit the `Extensions` window, and OceanSim should be an option on the IsaacSim panel. You can freely import OceanSim sensors and modules into your own Isaac Sim workflow.
