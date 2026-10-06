"""Apply a platform's hydrodynamics + thrusters (utils.vehicle_dynamics) to the
vehicle's PhysX rigid body each physics step.

The maths lives in ``vehicle_dynamics`` (pure numpy, unit tested); this module
is the thin Isaac Sim side: it sets the body's mass properties to the model's,
turns off the PhysX damping proxy, reads the pose / velocity through a
RigidPrim view and applies the resulting body-frame force and torque.

Controllers command it through ``command_wrench`` (allocated to the thrusters,
so thrust limits apply) or ``command_thrusters`` (normalised per-thruster
commands). Without a command the thrusters are idle and the vehicle just
drifts, floats or sinks with its buoyancy.
"""

import numpy as np

from isaacsim.oceansim.utils import vehicle_dynamics


def configure_prim(robot_prim, spec, mass=None):
    """Give the robot body the platform's mass properties (mass, principal
    inertia, CoG at the prim origin) and zero PhysX damping -- drag comes from
    the hydrodynamic model. ``mass`` overrides the spec's. Call before
    world.play()."""
    from pxr import Gf, PhysxSchema, UsdPhysics
    h = spec.hydro
    mass_api = UsdPhysics.MassAPI.Apply(robot_prim)
    mass_api.CreateMassAttr().Set(float(spec.mass if mass is None else mass))
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(v) for v in h.inertia]))
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(robot_prim)
    rb.CreateLinearDampingAttr().Set(0.0)
    rb.CreateAngularDampingAttr().Set(0.0)
    rb.CreateDisableGravityAttr(True)


class IsaacVehicle:
    """A VehicleModel bound to the robot's rigid body."""

    def __init__(self, robot_prim, model, name="vehicle"):
        self._robot_prim = robot_prim
        self.model = model
        self._name = name
        self._view = None
        self._warned = False
        self.reset()                # the GUI reuses one model across RESETs

    # ---------------------------------------------------------------- commands
    def command_wrench(self, force, torque):
        self.model.command_wrench(force, torque)

    def command_thrusters(self, u):
        self.model.command_thrusters(u)

    def stop(self):
        self.model.stop()

    @property
    def thruster_count(self):
        return self.model.thrusters.count

    def effective_mass(self):
        """Diagonal of M_RB + M_A (6,)."""
        return self.model.hydro.effective_mass()

    def feedforward(self, nu_cmd):
        """Thrust wrench (6,) that holds the body velocity nu_cmd against drag."""
        return self.model.feedforward(nu_cmd)

    # -------------------------------------------------------------------- step
    def _ensure_view(self):
        if self._view is not None:
            return self._view
        try:
            from isaacsim.core.prims import RigidPrim
            from isaacsim.core.utils.prims import get_prim_path
            view = RigidPrim(prim_paths_expr=get_prim_path(self._robot_prim))
            view.initialize()
        except Exception as e:  # noqa: BLE001 - physics not live yet: retry next step
            if not self._warned:
                self._warned = True
                print(f"[{self._name}] rigid-body view not ready yet ({e}); retrying")
            return None
        self._view = view
        return view

    def state(self):
        """(nu body (6,), body->world rotation, world position) or None."""
        view = self._ensure_view()
        if view is None:
            return None

        def _np(a):
            return a.numpy() if hasattr(a, "numpy") else np.asarray(a)
        pos, quat = view.get_world_poses()
        pos = _np(pos).reshape(-1, 3)[0].astype(float)
        rot = vehicle_dynamics.rot_from_quat_wxyz(_np(quat).reshape(-1, 4)[0])
        lin_w = _np(view.get_linear_velocities()).reshape(-1, 3)[0].astype(float)
        ang_w = _np(view.get_angular_velocities()).reshape(-1, 3)[0].astype(float)
        nu = np.concatenate([rot.T @ lin_w, rot.T @ ang_w])
        return nu, rot, pos

    def step(self, dt):
        """Compute and apply this physics step's hydrodynamic + thrust wrench."""
        st = self.state()
        if st is None:
            return
        nu, rot, pos = st
        force, torque = self.model.step(nu, rot, pos, dt)
        self._view.apply_forces_and_torques_at_pos(
            forces=np.asarray(force, dtype=np.float32).reshape(1, 3),
            torques=np.asarray(torque, dtype=np.float32).reshape(1, 3),
            is_global=False)

    def reset(self):
        self.model.thrusters.reset()
        self.model.stop()

    def close(self):
        self._view = None
