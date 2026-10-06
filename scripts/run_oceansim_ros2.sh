#!/usr/bin/env bash
# Launch the headless OceanSim ROS2 node inside an Isaac Sim python environment.
#
# OceanSim runs *inside* Isaac Sim, so this wrapper finds an Isaac Sim python
# launcher and execs the standalone runner with OceanSim on PYTHONPATH.  It is
# deliberately agnostic about whether Isaac Sim is a native install, a
# source build, or (as on the ERDC sim host) an Isaac Sim Docker container --
# in the container case the script is invoked with `isaacsim` already importable.
#
# Resolution order for the Isaac Sim python launcher:
#   $ISAAC_SIM_PYTHON                         (explicit python.sh path)
#   $OCEANSIM_ISAAC_SIM_ROOT/python.sh
#   $ISAAC_SIM_ROOT/python.sh
#   /isaac-sim/python.sh                      (official Isaac Sim container)
#   $HOME/isaacsim/python.sh                  (pip / local install)
# Fallback: if `python3 -c "import isaacsim"` works (e.g. already inside the
#   container's kit python), run the runner with plain python3.
#
# All arguments are forwarded to oceansim_ros2.py (e.g. --config scenario.json).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCEANSIM_ROOT="${OCEANSIM_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RUNNER="$OCEANSIM_ROOT/isaacsim/oceansim/standalone/oceansim_ros2.py"

# Make `isaacsim.oceansim` importable alongside Isaac Sim's own `isaacsim`
# namespace package (PEP 420).  Inside Isaac Sim, OceanSim is normally loaded as
# a registered extension; for the standalone runner we expose it on PYTHONPATH.
export PYTHONPATH="$OCEANSIM_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [ ! -f "$RUNNER" ]; then
  echo "ERROR: runner not found at $RUNNER" >&2
  exit 1
fi

# Non-interactive shells (including `docker/run.sh -lc ...`) do not read the
# /etc/bash.bashrc hook installed by the Dockerfile. Source an external ROS 2
# installation explicitly so its shared libraries and Python packages are
# visible before Isaac starts the ROS bridge. Keep Isaac's bundled ROS fallback
# untouched when no matching external installation exists.
if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  set +u
  # shellcheck disable=SC1090 -- ROS_DISTRO selects the installed distribution.
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  set -u
fi

find_isaac_python() {
  local c
  for c in \
      "${ISAAC_SIM_PYTHON:-}" \
      "${OCEANSIM_ISAAC_SIM_ROOT:-}/python.sh" \
      "${ISAAC_SIM_ROOT:-}/python.sh" \
      "/isaac-sim/python.sh" \
      "$HOME/isaacsim/python.sh"; do
    if [ -n "$c" ] && [ -x "$c" ]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

if ISAAC_PY="$(find_isaac_python)"; then
  echo "[run_oceansim_ros2] using Isaac Sim python: $ISAAC_PY"
  exec "$ISAAC_PY" "$RUNNER" "$@"
elif python3 -c 'import isaacsim' >/dev/null 2>&1; then
  echo "[run_oceansim_ros2] 'isaacsim' importable in current python; using python3"
  exec python3 "$RUNNER" "$@"
else
  cat >&2 <<'EOF'
ERROR: could not find an Isaac Sim python environment.
  - Set OCEANSIM_ISAAC_SIM_ROOT to the Isaac Sim install dir containing python.sh, or
  - set ISAAC_SIM_PYTHON to the python.sh path directly, or
  - run this inside an Isaac Sim container / pip env where `import isaacsim` works.
EOF
  exit 1
fi
