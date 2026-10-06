#!/usr/bin/env bash
#
# Launch the OceanSim Isaac Sim 6.1.0 container with GPU access and X11 display
# passthrough (Ubuntu 24.04 / ROS 2 Jazzy).
#
# Usage:
#   ./docker/run.sh                 # interactive bash inside the container
#   ./docker/run.sh ./isaac-sim.sh  # launch the Isaac Sim GUI directly
#
# Override the image tag with OCEANSIM_IMAGE (default: oceansim:6.1.0).
# Mount your downloaded assets with OCEANSIM_ASSETS=/path/to/OceanSim_assets.
# Set OCEANSIM_DETACH=1 to leave an explicit GUI/runner command in the background.
# Override the image's Zenoh default with, for example,
# RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ./docker/run.sh
# Local demos disable the optional Hub daemon. Set OMNICLIENT_HUB_MODE=shared
# to enable Hub caching when a working Hub service is available.
set -euo pipefail

IMAGE="${OCEANSIM_IMAGE:-oceansim:6.1.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCEANSIM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_ROOT="${OCEANSIM_CACHE_ROOT:-$OCEANSIM_ROOT/.local/isaac-sim}"
CONTAINER_NAME="${OCEANSIM_CONTAINER_NAME:-oceansim}"

# --- X11 display passthrough -------------------------------------------------
# Allow the container's user to talk to the host X server, and revoke the grant
# on ANY exit (normal, error, or Ctrl-C) so it doesn't stay open for the rest of
# the login session.
if [[ -n "${DISPLAY:-}" && "${OCEANSIM_HEADLESS:-0}" != 1 ]] && command -v xhost >/dev/null 2>&1; then
    xhost +local:root >/dev/null
    trap 'xhost -local:root >/dev/null 2>&1 || true' EXIT
fi

# Persisted Isaac Sim caches (first run is slow while shaders compile). The bulk
# of the RTX/MDL shader cache on 6.x is /isaac-sim/kit/cache (~570 MB); without
# it the shader compile is paid on every --rm run.
mkdir -p \
    "$CACHE_ROOT/cache/kit" \
    "$CACHE_ROOT/cache/main" \
    "$CACHE_ROOT/cache/computecache" \
    "$CACHE_ROOT/logs" \
    "$CACHE_ROOT/config" \
    "$CACHE_ROOT/data" \
    "$CACHE_ROOT/pkg" \
    "$CACHE_ROOT/hub"

# Optional: mount downloaded USD assets and point OceanSim at them.
ASSET_ARGS=()
if [[ -n "${OCEANSIM_ASSETS:-}" ]]; then
    ASSET_ARGS=(
        -v "${OCEANSIM_ASSETS}:/isaac-sim/OceanSim_assets:rw"
        -e "OCEANSIM_ASSETS=/isaac-sim/OceanSim_assets"
    )
fi

# Pass through an explicitly selected ROS middleware and its common discovery
# settings. With no host override, the image's rmw_zenoh_cpp default remains in
# effect.
ROS_ENV_ARGS=()
for ENV_NAME in RMW_IMPLEMENTATION ROS_DOMAIN_ID; do
    if [[ -n "${!ENV_NAME:-}" ]]; then
        ROS_ENV_ARGS+=(-e "${ENV_NAME}=${!ENV_NAME}")
    fi
done

# Cyclone DDS commonly points CYCLONEDDS_URI at a host config file. Bind-mount
# a file:// URI at the same absolute path so it remains valid in the container;
# inline XML and other URI forms can be forwarded directly.
CYCLONEDDS_ARGS=()
if [[ -n "${CYCLONEDDS_URI:-}" ]]; then
    if [[ "$CYCLONEDDS_URI" == file://* ]]; then
        CYCLONEDDS_CONFIG="${CYCLONEDDS_URI#file://}"
        if [[ ! -f "$CYCLONEDDS_CONFIG" ]]; then
            echo "ERROR: CYCLONEDDS_URI config file not found: $CYCLONEDDS_CONFIG" >&2
            exit 1
        fi
        CYCLONEDDS_CONFIG="$(realpath "$CYCLONEDDS_CONFIG")"
        CYCLONEDDS_ARGS=(
            -v "${CYCLONEDDS_CONFIG}:${CYCLONEDDS_CONFIG}:ro"
            -e "CYCLONEDDS_URI=file://${CYCLONEDDS_CONFIG}"
        )
    else
        CYCLONEDDS_ARGS=(-e "CYCLONEDDS_URI=${CYCLONEDDS_URI}")
    fi
fi

# Only bind-mount the X cookie when it is an existing file. On Wayland/GDM the
# $HOME/.Xauthority fallback often does not exist, and the old unconditional -v
# created a stray empty host DIRECTORY where a cookie file is expected. Local
# socket auth (xhost +local:root above) still works without it.
XAUTH_ARGS=()
XAUTH_FILE="${XAUTHORITY:-$HOME/.Xauthority}"
if [[ -f "${XAUTH_FILE}" ]]; then
    XAUTH_ARGS=(-v "${XAUTH_FILE}:/root/.Xauthority:rw" -e "XAUTHORITY=/root/.Xauthority")
fi

# Batch/headless invocations also work without a terminal (CI, redirected logs).
TERMINAL_ARGS=()
if [[ "${OCEANSIM_DETACH:-0}" == 1 ]]; then
    TERMINAL_ARGS=(-d)
elif [[ -t 0 && -t 1 ]]; then
    TERMINAL_ARGS=(-it)
fi

docker run --name "$CONTAINER_NAME" --rm "${TERMINAL_ARGS[@]}" \
    --runtime=nvidia --gpus all \
    --network=host \
    --entrypoint bash \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "OMNICLIENT_HUB_MODE=${OMNICLIENT_HUB_MODE:-disabled}" \
    -e "DISPLAY=${DISPLAY:-:0}" \
    -e "QT_X11_NO_MITSHM=1" \
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \
    -e "NVIDIA_VISIBLE_DEVICES=all" \
    -e "OCEANSIM_ROOT=/isaac-sim/extsUser/OceanSim" \
    -v "$OCEANSIM_ROOT:/isaac-sim/extsUser/OceanSim:rw" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    "${XAUTH_ARGS[@]}" \
    -v "$CACHE_ROOT/cache/kit:/isaac-sim/kit/cache:rw" \
    -v "$CACHE_ROOT/cache/main:/isaac-sim/.cache:rw" \
    -v "$CACHE_ROOT/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
    -v "$CACHE_ROOT/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
    -v "$CACHE_ROOT/config:/isaac-sim/.nvidia-omniverse/config:rw" \
    -v "$CACHE_ROOT/data:/isaac-sim/.local/share/ov/data:rw" \
    -v "$CACHE_ROOT/pkg:/isaac-sim/.local/share/ov/pkg:rw" \
    -v "$CACHE_ROOT/hub:/var/cache/hub:rw" \
    "${ASSET_ARGS[@]}" \
    "${ROS_ENV_ARGS[@]}" \
    "${CYCLONEDDS_ARGS[@]}" \
    "${IMAGE}" "${@:--i}"
