# syntax=docker/dockerfile:1
#
# OceanSim on NVIDIA Isaac Sim 6.0.1 (Ubuntu 24.04 / ROS 2 Jazzy)
# ---------------------------------------------------------------------------
# Builds on NVIDIA's official Isaac Sim 6.0.1 container (Ubuntu 24.04 base, which
# is why ROS 2 Jazzy is the matching distro), layers ROS 2 Jazzy and the OceanSim
# Python dependencies on top, and installs OceanSim as an Isaac Sim user
# extension under /isaac-sim/extsUser.
#
# Build:
#   docker build -t oceansim:6.0.1 .
# Run (GPU + X11 display passthrough):
#   ./docker/run.sh          # see that script for xhost / display flags
#
# Pulling the base image requires an NGC login:
#   docker login nvcr.io
# ---------------------------------------------------------------------------
ARG ISAACSIM_VERSION=6.0.1
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}

ARG ROS_DISTRO=jazzy
ENV ROS_DISTRO=${ROS_DISTRO} \
    DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all

USER root

# ROS 2 Jazzy (ros-base) plus the message/helper packages used by OceanSim's
# ROS2 bridge utilities, and the X11/GL client libraries needed for GUI display
# passthrough (the Isaac Sim Qt UI and the cv2.imshow viewer in
# ros2_image_subscriber.py).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg2 lsb-release software-properties-common locales \
        libgl1 libglu1-mesa libxext6 libsm6 libxrender1 \
        libxkbcommon-x11-0 libxcb-xinerama0 libxcb-cursor0 x11-apps \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && add-apt-repository -y universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
         -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
         > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-ros-base \
        ros-${ROS_DISTRO}-sensor-msgs \
        ros-${ROS_DISTRO}-geometry-msgs \
        ros-${ROS_DISTRO}-std-msgs \
        ros-${ROS_DISTRO}-vision-msgs \
        ros-${ROS_DISTRO}-cv-bridge \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install OceanSim into the Isaac Sim user-extensions directory so it is
# discoverable in the extension browser.
ENV OCEANSIM_PATH=/isaac-sim/extsUser/OceanSim
COPY . ${OCEANSIM_PATH}

# cv2 for the bundled Isaac Sim interpreter (UW_Camera / ROS2 image publisher).
# numpy, pyyaml and warp already ship with Isaac Sim.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir opencv-python-headless

# Source ROS 2 automatically in interactive shells.
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc

WORKDIR /isaac-sim
