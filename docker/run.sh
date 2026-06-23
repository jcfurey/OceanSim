#!/usr/bin/env bash
#
# Launch the OceanSim Isaac Sim 6.0.1 container with GPU access and X11 display
# passthrough (Ubuntu 24.04 / ROS 2 Jazzy).
#
# Usage:
#   ./docker/run.sh                 # interactive bash inside the container
#   ./docker/run.sh ./isaac-sim.sh  # launch the Isaac Sim GUI directly
#
# Override the image tag with OCEANSIM_IMAGE (default: oceansim:6.0.1).
# Mount your downloaded assets with OCEANSIM_ASSETS=/path/to/OceanSim_assets.
set -euo pipefail

IMAGE="${OCEANSIM_IMAGE:-oceansim:6.0.1}"

# --- X11 display passthrough -------------------------------------------------
# Allow the container's user to talk to the host X server. Revoke afterwards
# with:  xhost -local:root
if command -v xhost >/dev/null 2>&1; then
    xhost +local:root >/dev/null
else
    echo "warning: 'xhost' not found - the GUI may fail to display. Install x11-xserver-utils." >&2
fi

# Persisted Isaac Sim caches (first run is slow while shaders compile).
mkdir -p \
    ~/docker/isaac-sim/cache/main \
    ~/docker/isaac-sim/cache/computecache \
    ~/docker/isaac-sim/logs \
    ~/docker/isaac-sim/config \
    ~/docker/isaac-sim/data \
    ~/docker/isaac-sim/pkg \
    ~/.cache/ov/hub

# Optional: mount downloaded USD assets and point OceanSim at them.
ASSET_ARGS=()
if [[ -n "${OCEANSIM_ASSETS:-}" ]]; then
    ASSET_ARGS=(-v "${OCEANSIM_ASSETS}:/isaac-sim/OceanSim_assets:rw")
fi

docker run --name oceansim --rm -it \
    --runtime=nvidia --gpus all \
    --network=host \
    --entrypoint bash \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "DISPLAY=${DISPLAY:-:0}" \
    -e "QT_X11_NO_MITSHM=1" \
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \
    -e "NVIDIA_VISIBLE_DEVICES=all" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${XAUTHORITY:-$HOME/.Xauthority}:/root/.Xauthority:rw" \
    -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
    -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
    -v ~/.cache/ov/hub:/var/cache/hub:rw \
    "${ASSET_ARGS[@]}" \
    "${IMAGE}" "${@:--i}"
