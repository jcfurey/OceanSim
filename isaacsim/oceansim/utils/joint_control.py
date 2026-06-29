"""Pure (numpy-only) joint-command reconciliation for OceanSim.

A ROS ``sensor_msgs/JointState`` command may list joints in a different order
than the Isaac articulation, name only a subset, or include joints that do not
exist on this robot. Mapping that safely onto the articulation's fixed DOF order
is the part that is easy to get subtly wrong, so it lives here -- dependency-free
and unit tested -- while the articulation read/write stays in the ROS layer.
"""

import numpy as np


def map_named_command(cmd_names, cmd_values, joint_order, current=None):
    """Reorder a named joint command onto the articulation's DOF order.

    Args:
        cmd_names: joint names from the incoming command.
        cmd_values: values aligned with ``cmd_names``.
        joint_order: the articulation's DOF names, in index order.
        current: optional current targets (len == joint_order). Joints not named
            in the command keep their current value. If omitted, unnamed joints
            are returned as NaN -- a "no change" sentinel the caller can filter.

    Returns:
        (targets, ignored): ``targets`` is a float array aligned with
        ``joint_order``; ``ignored`` is the list of command names that are not
        joints of this robot (silently dropped from ``targets``).
    """
    index = {n: i for i, n in enumerate(joint_order)}
    n = len(joint_order)
    if current is not None:
        targets = np.asarray(current, dtype=float).copy()
        if targets.shape[0] != n:
            raise ValueError(
                f"current has {targets.shape[0]} entries, expected {n} (one per joint).")
    else:
        targets = np.full(n, np.nan)

    ignored = []
    for name, val in zip(cmd_names, cmd_values):
        j = index.get(name)
        if j is None:
            ignored.append(name)
        else:
            targets[j] = float(val)
    return targets, ignored


def clamp_to_limits(targets, lower, upper):
    """Clamp ``targets`` to ``[lower, upper]`` elementwise, leaving NaN ("no
    change") entries untouched. Bounds may contain non-finite entries (e.g. a
    continuous joint with +/-inf limits), which disable clamping on that side.
    """
    targets = np.asarray(targets, dtype=float).copy()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    finite = ~np.isnan(targets)
    lo_ok = finite & np.isfinite(lower)
    hi_ok = finite & np.isfinite(upper)
    targets[lo_ok] = np.maximum(targets[lo_ok], lower[lo_ok])
    targets[hi_ok] = np.minimum(targets[hi_ok], upper[hi_ok])
    return targets
