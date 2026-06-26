#!/usr/bin/env bash
#
# Start the ROS 2 Zenoh router (rmw_zenohd) for standalone OceanSim testing.
#
# With RMW_IMPLEMENTATION=rmw_zenoh_cpp (the OceanSim container default), nodes
# discover each other via the Zenoh router's gossip -- multicast discovery is
# OFF by default -- so a router MUST be running before any ROS 2 node (the
# OceanSim sim publishers, RViz, robot_localization, sonar_image_proc, ...) can
# see each other. https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_ros.html
#
# In a real deployment the router is part of the robot stack and its endpoint
# config is sourced from the workspace bashrc.d/99-zenoh_configs.bashrc (see the
# Dockerfile). This helper is for running OceanSim BY ITSELF (e.g. a dev box or
# a smoke test) where no stack router exists.
#
# Usage:
#   ./scripts/start_zenoh_router.sh          # foreground; Ctrl-C to stop
#   ZENOH_ROUTER_CONFIG_URI=/path/router.json5 ./scripts/start_zenoh_router.sh
set -euo pipefail

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

if ! command -v ros2 >/dev/null 2>&1; then
    echo "ERROR: 'ros2' not found. Source your ROS 2 install first " \
         "(e.g. 'source /opt/ros/\$ROS_DISTRO/setup.bash')." >&2
    exit 1
fi

echo "[start_zenoh_router] launching rmw_zenohd (RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION)"
echo "[start_zenoh_router] keep this running; start the OceanSim runner in another terminal."
exec ros2 run rmw_zenoh_cpp rmw_zenohd
