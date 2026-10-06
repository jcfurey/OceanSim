# Omniverse import
import numpy as np
import os
import omni.timeline
import omni.ui as ui
from omni.usd import StageEventType
from pxr import PhysxSchema
import carb

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim, SingleGeometryPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage, create_new_stage, open_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.semantics import add_labels  # Isaac 6.0.1 renamed add_update_semantics -> add_labels
from isaacsim.gui.components import CollapsableFrame, StateButton, get_style, setup_ui_headers, CheckBox, combo_cb_xyz_plot_builder, combo_cb_plot_builder, dropdown_builder, str_builder
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.core.utils.extensions import get_extension_path

# Custom import
from .scenario import MHL_Sensor_Example_Scenario
from .global_variables import EXTENSION_DESCRIPTION, EXTENSION_TITLE, EXTENSION_LINK
from isaacsim.oceansim.utils.assets_utils import get_oceansim_assets_path
from isaacsim.oceansim.utils import platforms
from isaacsim.oceansim.utils import payloads as payload_catalogue

class UIBuilder():
    def __init__(self):

        self._ext_id = omni.kit.app.get_app().get_extension_manager().get_extension_id_by_module(__name__)
        self._file_path = os.path.abspath(__file__)
        self._title = EXTENSION_TITLE
        self._doc_link =  EXTENSION_LINK
        self._overview = EXTENSION_DESCRIPTION
        self._extension_path = get_extension_path(self._ext_id)
        
        self._ctrl_mode = 'Manual control'
        self._waypoints_path = self._extension_path + '/demo/demo_waypoints.txt'
        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        # UI frames created
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Run initialization for the provided example
        self._on_init()

    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        pass

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # When the user hits the stop button through the UI, they will inevitably discover edge cases where things break
            # For complete robustness, the user should resolve those edge cases here
            # In general, for extensions based off this template, there is no value to having the user click the play/stop
            # button instead of using the Load/Reset/Run buttons provided.
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        pass

    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(StageEventType.OPENED):
            # If the user opens a new stage, the extension should completely reset
            self._reset_extension()

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from omni.isaac.ui.element_wrappers implement a cleanup function that should be called
        """
        self._DVL_event_sub = None
        self._baro_event_sub = None
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()
        for frame in self.frames:
            frame.cleanup()

    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """

        setup_ui_headers(
            ext_id=self._ext_id, 
            file_path=self._file_path, 
            title=self._title, 
            doc_link=self._doc_link, 
            overview=self._overview, 
            info_collapsed=False
        )

        sensor_choosing_frame = CollapsableFrame('Sensors', collapsed=False)
        self.frames.append(sensor_choosing_frame)
        with sensor_choosing_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # Upstream OceanSim's OmniGraph ROS2 publishers (the *_ROS sensor
                # classes). OFF by default: this fork's rclpy bridge (UW_Camera
                # publisher, ROS control mode, headless runner) is the primary
                # ROS path. When ON, the camera's rclpy publisher is turned off
                # so topics aren't published twice.
                omnigraph_ros_check_box = CheckBox(
                    "OmniGraph ROS",
                    default_value=False,
                    tooltip="Publish the selected sensors through upstream OceanSim's OmniGraph "
                            "ROS 2 publishers (isaacsim.ros2.bridge) and subscribe /cmd_vel",
                    on_click_fn=self._on_omnigraph_ros_checkbox_click_fn,
                )
                self._use_omnigraph_ros = False
                self.wrapped_ui_elements.append(omnigraph_ros_check_box)

                imu_check_box = CheckBox(
                    "Imu",
                    default_value=False,
                    tooltip=" Click this checkbox to activate Imu",
                    on_click_fn=self._on_imu_checkbox_click_fn,
                )
                self._use_imu = False
                self.wrapped_ui_elements.append(imu_check_box)

                sonar_check_box = CheckBox(
                    "Imaging Sonar",
                    default_value=False,
                    tooltip=" Click this checkbox to activate imaging sonar",
                    on_click_fn=self._on_sonar_checkbox_click_fn,
                )
                self._use_sonar = False
                self.wrapped_ui_elements.append(sonar_check_box)
                camera_check_box = CheckBox(
                    "Underwater Camera",
                    default_value=False,
                    tooltip=" Click this checkbox to activate underwater camera",
                    on_click_fn=self._on_camera_checkbox_click_fn,
                )
                self._use_camera = False
                self.wrapped_ui_elements.append(camera_check_box)

                DVL_check_box = CheckBox(
                    'DVL',
                    default_value=False,
                    tooltip=" Click this checkbox to activate DVL",
                    on_click_fn=self._on_DVL_checkbox_click_fn
                )
                self._use_DVL = False
                self.wrapped_ui_elements.append(DVL_check_box)

                baro_check_box = CheckBox(
                    "Barometer",
                    default_value=False,
                    tooltip='Click this checkbox to activate barometer',
                    on_click_fn=self._on_baro_checkbox_click_fn
                ) 
                self._use_baro = False
                self.wrapped_ui_elements.append(baro_check_box)

                # Payloads (utils.payloads): one picker per kind. "standard" =
                # the platform's own standard set for that kind. Sensor payloads
                # configure the matching sensor above with the device's
                # datasheet values; every payload adds mass / buoyancy / drag
                # (the vehicle is re-trimmed) and appears in generated models.
                self._payload_choice = {}
                for kind, label in (("sonar", "Sonar payload"), ("dvl", "DVL payload"),
                                    ("gripper", "Gripper"), ("light", "Lights")):
                    items = ["standard", "none"] + payload_catalogue.available_payloads(kind)
                    self._payload_choice[kind] = "standard"
                    dropdown_builder(
                        label=label, default_val=0, items=items,
                        tooltip=f"{label} fitted on LOAD ('standard' = the platform's default)",
                        on_clicked_fn=lambda item, k=kind: self._payload_choice.__setitem__(k, item))

                
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)
        self.frames.append(world_controls_frame)
        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                
                # self._build_USD_filepicker()
                self._USD_path_field = str_builder(
                    label='Path to USD',
                    default_val="",
                    tooltip='Select the USD file for the scene',
                    use_folder_picker=True,
                    folder_button_title="Select USD",
                    folder_dialog_title='Select the USD scene to test')
                
                self._ctrl_mode_model = dropdown_builder(
                    label='Control Mode',
                    default_val=3,
                    items=['No control', 'Straight line', 'Waypoints', 'Manual control', 'ROS control'],
                    tooltip='Select preferred control mode',
                    on_clicked_fn=self._on_ctrl_mode_dropdown_clicked
                )

                # Vehicle platform picker (utils.platforms). Items are the
                # registry's platforms, so adding a vehicle there adds it here.
                self._platform_keys = platforms.available_platforms()
                self._platform_labels = [
                    platforms.get_platform(k).description.split('(')[0].strip() or k
                    for k in self._platform_keys
                ]
                _default_platform_idx = (self._platform_keys.index(self._platform)
                                         if self._platform in self._platform_keys else 0)
                self._platform_dropdown_model = dropdown_builder(
                    label='Vehicle Platform',
                    default_val=_default_platform_idx,
                    items=self._platform_labels,
                    tooltip='Vehicle spawned on Load (e.g. BlueROV2 or DeepTrekker Revolution).',
                    on_clicked_fn=self._on_platform_dropdown_clicked
                )

                # Optional URDF override: if set, the robot is imported from this
                # URDF (creating the articulation) instead of the platform's USD.
                # Leave empty to use the platform's own asset.
                self._urdf_path_field = str_builder(
                    label='URDF (optional)',
                    default_val="",
                    tooltip='Import the robot from this URDF instead of the platform USD',
                    use_folder_picker=True,
                    folder_button_title="Select URDF",
                    folder_dialog_title='Select a robot URDF to import')

                self._load_btn = LoadButton(
                    "Load Button", "LOAD", setup_scene_fn=self._setup_scene, setup_post_load_fn=self._setup_scenario
                )
                # self._load_btn.set_world_settings(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button", "RESET", pre_reset_fn=None, post_reset_fn=self._on_post_reset_btn
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

        run_scenario_frame = CollapsableFrame("Run Scenario", collapsed=False)
        self.frames.append(run_scenario_frame)
        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

        self.sensor_reading_frame = CollapsableFrame('Sensor Reading', collapsed=False, visible=False)
        self.frames.append(self.sensor_reading_frame)
        self.waypoints_frame = CollapsableFrame('Waypoints',collapsed=False, visible=False)
        self.frames.append(self.waypoints_frame)
        self.ros2_control_frame = CollapsableFrame('ROS2 Control Mode Setting', collapsed=False, visible=False)
        self.frames.append(self.ros2_control_frame)




    ######################################################################################
    # Functions Below This Point Related to Scene Setup (USD\PhysX..)
    ######################################################################################

    def _on_init(self):

        # Vehicle platform (utils.platforms): which vehicle the Load button
        # spawns, chosen via the "Vehicle Platform" dropdown. The selected
        # platform's spec supplies the USD, mass/damping, collision, spawn pose
        # and the sensor mount poses below.
        # PRESERVE an existing selection across _on_init(): _reset_extension()
        # re-runs this on every stage OPENED event, but the dropdown widget is
        # NOT rebuilt -- resetting to the default here desynced the state from
        # the UI (dropdown still shows "Deep Trekker REVOLUTION", Load silently
        # spawns a BlueROV2). Same reason _ctrl_mode lives in __init__ only.
        if not hasattr(self, "_platform"):
            self._platform = platforms.DEFAULT_PLATFORM

        # Sensor
        self._imu = None
        self._sonar = None
        self._cam = None
        self._cam_focal_length = 21
        self._DVL = None
        self._baro = None
        self._water_surface = 1.43389 # Arbitrary
        
        # Scenario
        self._scenario = MHL_Sensor_Example_Scenario()


    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        create_new_stage()
        if self._USD_path_field.get_value_as_string() != "":
            scene_prim_path = '/World/scene'
            add_reference_to_stage(usd_path=self._USD_path_field.get_value_as_string(), prim_path=scene_prim_path)
            print('User USD scene is loaded.')
        else:
            print('USD path is empty. Default to example scene')

            # add MHL scene as reference
            MHL_prim_path = '/World/mhl'
            MHL_usd_path = get_oceansim_assets_path() + "/collected_MHL/mhl_scaled.usd"
            add_reference_to_stage(usd_path=MHL_usd_path, prim_path=MHL_prim_path)
            # Toggle MHL mesh's collider
            SingleGeometryPrim(prim_path=MHL_prim_path, collision=True)
            # apply a reflectivity of 1.0 to mesh of the scene for sonar simulation
            add_labels(get_prim_at_path(MHL_prim_path + "/Mesh/mesh"),
                       labels=['1.0'], instance_name='reflectivity')
            # Load the rock
            rock_prim_path = '/World/rock'
            rock_usd_path = get_oceansim_assets_path() + "/collected_rock/rock.usd"
            rock_prim = add_reference_to_stage(usd_path=rock_usd_path, prim_path=rock_prim_path)
            # apply a reflectivity of 2.0 for sonar simulation
            add_labels(get_prim_at_path(rock_prim_path+ '/Mesh/mesh'),
                       labels=['2.0'], instance_name='reflectivity')
            # Toggle collider for the rock
            rock_collider_prim = SingleGeometryPrim(prim_path=rock_prim_path,
                            collision=True)
            # Set collision approximation using convexDecomposition to automatically compute inertia matrix
            rock_collider_prim.set_collision_approximation('convexDecomposition')
            # Toggle rigid body for the rock
            rock_rigid_prim = SingleRigidPrim(prim_path=rock_prim_path,                          
                                            translation=np.array([1.0, 0.1, -1.5]),
                                            orientation=euler_angles_to_quat(np.array([0.0,0.0,90]), degrees=True), 
                                            )
            
        # Spawn the selected vehicle platform (utils.platforms). Normally the
        # platform USD is referenced; if the "URDF (optional)" field is set (or
        # only a URDF exists), import that instead -- it creates the articulation,
        # so joint manipulation works. The spec supplies dynamics, collision,
        # spawn pose and sensor mounts.
        spec = platforms.get_platform(self._platform)
        robot_prim_path = "/World/rob"
        _urdf_override = self._urdf_path_field.get_value_as_string().strip()
        # A URDF with hydrodynamics (OceanSim export / Gazebo / UUV Simulator)
        # defines the vehicle itself.
        if _urdf_override and os.path.isfile(_urdf_override):
            from isaacsim.oceansim.utils import urdf_platform
            with open(_urdf_override) as _f:
                _utext = _f.read()
            if urdf_platform.hydro_source(_utext):
                spec, _notes = urdf_platform.platform_from_urdf(_utext, _urdf_override, fallback=spec)
                for _n in _notes:
                    print(f"[OceanSim] URDF vehicle: {_n}")
        print(f"[OceanSim] platform: {spec.name} -- {spec.description}")
        self._fitted_payloads = self._selected_payloads(spec)
        print(f"[OceanSim] payloads: {[p.name for p in self._fitted_payloads] or 'none'}")
        src, why = platforms.resolve_robot_source(
            asset_root=get_oceansim_assets_path(), platform=spec,
            urdf_path=_urdf_override or None,
            prefer="urdf" if _urdf_override else "usd")
        if src is None and spec.dimensions and spec.hydro is not None:
            # No 3D asset for this vehicle: import a URDF generated from the
            # platform data (primitive shapes + the payloads).
            from isaacsim.oceansim.utils import urdf_export
            if why.startswith("missing:"):
                carb.log_warn(f"[OceanSim] {spec.name} asset not found ({why[8:]}); "
                              f"using the generated primitive-shape model")
            _gpath, _ = urdf_export.write_generated_urdf(spec, self._fitted_payloads)
            src = platforms.RobotSource("urdf", _gpath)
            print(f"[OceanSim] generated URDF for {spec.name} -> {_gpath}")
        if src is None:
            carb.log_error(f"[OceanSim] no robot asset for platform '{spec.name}' ({why}).")
            return

        if src.kind == "urdf":
            # URDF defines inertials/joints/collisions; only apply the underwater
            # setup (no gravity, damping, spawn) -- don't override mass/collision.
            from isaacsim.oceansim.utils import urdf_import
            print(f"[OceanSim] importing URDF -> {src.path}")
            robot_prim_path = urdf_import.import_urdf_to_stage(src.path, fix_base=False)
            rob_rigidBody_API = PhysxSchema.PhysxRigidBodyAPI.Apply(get_prim_at_path(robot_prim_path))
            rob_rigidBody_API.CreateDisableGravityAttr(True)
            try:
                rob_rigidBody_API.GetLinearDampingAttr().Set(spec.linear_damping)
                rob_rigidBody_API.GetAngularDampingAttr().Set(spec.angular_damping)
            except Exception:  # noqa: BLE001 - articulation root may differ
                pass
            SingleRigidPrim(prim_path=robot_prim_path,
                            translation=np.array(spec.spawn_translation, dtype=float))
        else:
            print(f"[OceanSim] referencing USD -> {src.path}")
            self._rob = add_reference_to_stage(usd_path=src.path, prim_path=robot_prim_path)
            # Toggle rigid body and collider preset for robot, and set zero gravity to mimic underwater environment
            rob_rigidBody_API = PhysxSchema.PhysxRigidBodyAPI.Apply(get_prim_at_path(robot_prim_path))
            rob_rigidBody_API.CreateDisableGravityAttr(True)
            rob_rigidBody_API.GetLinearDampingAttr().Set(spec.linear_damping)
            rob_rigidBody_API.GetAngularDampingAttr().Set(spec.angular_damping)
            rob_collider_prim = SingleGeometryPrim(prim_path=robot_prim_path,
                                                   collision=True)
            rob_collider_prim.set_collision_approximation(spec.collision_approximation)
            SingleRigidPrim(prim_path=robot_prim_path,
                            mass=spec.mass,
                            translation=np.array(spec.spawn_translation, dtype=float))
        self._rob = get_prim_at_path(robot_prim_path)

        # Hydrodynamics + thrusters for platforms that have a model (drag,
        # buoyancy, added mass, thrust limits) in place of the PhysX damping
        # proxy set above.
        self._vehicle_model = None
        if spec.hydro is not None:
            from isaacsim.oceansim.utils import vehicle_dynamics, vehicle_physics
            vehicle_physics.configure_prim(
                self._rob, vehicle_dynamics.resolve_hydro(spec, self._fitted_payloads))
            self._vehicle_model = vehicle_dynamics.from_platform(
                spec, rho=1000.0, surface_z=self._water_surface, payloads=self._fitted_payloads)
            print(f"[OceanSim] hydrodynamics: {spec.name}, "
                  f"{self._vehicle_model.thrusters.count} thrusters")

        set_camera_view(eye=np.array([5, 0.6, 0.4]),
                        target=np.array(spec.spawn_translation, dtype=float))

        # When imported from a URDF, place the sensors at the URDF's sensor-link
        # frames (fixed-joint origins), else at the platform spec mounts.
        from isaacsim.oceansim.utils import urdf_parse
        _urdf_text = None
        if src.kind == "urdf":
            try:
                with open(src.path, 'r') as _f:
                    _urdf_text = _f.read()
            except Exception as e:  # noqa: BLE001
                carb.log_warn(f"[OceanSim] could not read URDF for sensor mounts: {e}")

        def _mount(kind, fallback_mount):
            tr, rpy = urdf_parse.sensor_mount_or(
                _urdf_text, kind, fallback_mount.translation, fallback_mount.rpy_deg)
            return np.array(tr, dtype=float), np.array(rpy, dtype=float)

        use_og_ros = getattr(self, "_use_omnigraph_ros", False)
        # The sensor classes are fixed here, at LOAD. Remember the choice so
        # RESET wires the scenario for the classes actually built, not for the
        # checkbox's current state (toggling it then pressing RESET would hand
        # og_node kwargs to the plain sensors).
        self._scene_use_omnigraph_ros = use_og_ros

        if getattr(self, "_use_imu", False):
            if use_og_ros:
                from isaacsim.oceansim.sensors.ImuSensor_ROS import ImuSensor_ROS as imu_cls
            else:
                from isaacsim.sensors.physics import IMUSensor as imu_cls
            self._imu = imu_cls(prim_path=robot_prim_path + "/imu",
                                name="Imu",
                                frequency=60,
                                translation=np.array([0, 0, 0]))

        if self._use_sonar:
            if use_og_ros:
                from isaacsim.oceansim.sensors.ImagingSonarSensor_ROS import (
                    ImagingSonarSensor_ROS as ImagingSonarSensor,
                )
            else:
                from isaacsim.oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor
            from isaacsim.oceansim.utils import sensor_presets
            _sonar_tr, _sonar_rpy = _mount("sonar", spec.sonar_mount)
            _sonar_pl = payload_catalogue.sensor_payload(self._fitted_payloads, "sonar")
            if _sonar_pl is None:
                _sonar_pl = payload_catalogue.get_payload(sensor_presets.DEFAULT_SONAR_PAYLOAD)
            _sk = sensor_presets.sonar_kwargs(_sonar_pl)
            self._sonar = ImagingSonarSensor(prim_path=robot_prim_path + '/sonar',
                                            translation=_sonar_tr,
                                            orientation=euler_angles_to_quat(_sonar_rpy, degrees=True),
                                            **_sk,
                                            hori_res=4000,
                                            # On-device point selection, as the headless runner
                                            # uses: the numpy path copies ~89 MB to the host and
                                            # spends ~80 ms per scan at hori_res=4000. Falls back
                                            # to numpy by itself if the AOVs aren't on-device.
                                            gpu_point_filter=True,
                                            )
            self._sonar.make_sonar_data_params = sensor_presets.sonar_model_params(_sonar_pl)

        if self._use_camera:
            if use_og_ros:
                from isaacsim.oceansim.sensors.UW_Camera_ROS import UW_Camera_ROS as UW_Camera
            else:
                from isaacsim.oceansim.sensors.UW_Camera import UW_Camera

            _cam_tr, _ = _mount("camera", spec.camera_mount)
            self._cam = UW_Camera(prim_path=robot_prim_path + '/UW_camera',
                                    resolution=[1920,1080],
                                    translation=_cam_tr)
            if spec.camera_hfov_deg:
                from isaacsim.oceansim.utils import sensor_presets
                self._cam.set_focal_length(sensor_presets.focal_length_for_hfov(
                    spec.camera_hfov_deg, self._cam.get_horizontal_aperture()))
            else:
                self._cam.set_focal_length(0.1 * self._cam_focal_length)
            self._cam.set_clipping_range(0.1, 100)

        if self._use_DVL:
            if use_og_ros:
                from isaacsim.oceansim.sensors.DVLSensor_ROS import DVLSensor_ROS as DVLsensor
            else:
                from isaacsim.oceansim.sensors.DVLsensor import DVLsensor

            from isaacsim.oceansim.utils import sensor_presets
            _dvl_tr, _ = _mount("dvl", spec.dvl_mount)
            _dvl_pl = payload_catalogue.sensor_payload(self._fitted_payloads, "dvl")
            self._DVL = (DVLsensor(**sensor_presets.dvl_kwargs(_dvl_pl)) if _dvl_pl is not None
                         else DVLsensor(max_range=10))
            self._DVL.attachDVL(rigid_body_path=robot_prim_path,
                                translation=_dvl_tr)
            self._DVL.add_debug_lines()
            
        if self._use_baro:
            if use_og_ros:
                from isaacsim.oceansim.sensors.BarometerSensor_ROS import (
                    BarometerSensor_ROS as BarometerSensor,
                )
            else:
                from isaacsim.oceansim.sensors.BarometerSensor import BarometerSensor

            self._baro = BarometerSensor(prim_path=robot_prim_path + '/Baro',
                                        water_surface_z=self._water_surface)
            


    def _setup_scenario(self):
        """
        This function is attached to the Load Button as the setup_post_load_fn callback.
        The user may assume that their assets have been loaded by t setup_scene_fn callback, that
        their objects are properly initialized, and that the timeline is paused on timestep 0.
        """
        self._reset_scenario()
        self._add_extra_ui()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    def _reset_scenario(self):
        self._scenario.teardown_scenario()
        self._scenario.setup_scenario(self._rob, self._sonar, self._cam, self._DVL, self._baro, self._ctrl_mode,
                                      imu=self._imu,
                                      use_omnigraph_ros=getattr(self, "_scene_use_omnigraph_ros", False),
                                      vehicle_model=getattr(self, "_vehicle_model", None))
    def _on_post_reset_btn(self):
        """
        This function is attached to the Reset Button as the post_reset_fn callback.
        The user may assume that their objects are properly initialized, and that the timeline is paused on timestep 0.

        They may also assume that objects that were added to the World.Scene have been moved to their default positions.
        I.e. the cube prim will move back to the position it was in when it was created in self._setup_scene().
        """
        self._reset_scenario()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float):
        """This function is attached to the Run Scenario StateButton.
        This function was passed in as the physics_callback_fn argument.
        This means that when the a_text "RUN" is pressed, a subscription is made to call this function on every physics step.
        When the b_text "STOP" is pressed, the physics callback is removed.

        Args:
            step (float): The dt of the current physics step
        """
        self._scenario.update_scenario(step)

    def _on_run_scenario_a_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_a_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "RUN".

        This function simply plays the timeline, which means that physics steps will start happening.  After the world is loaded or reset,
        the timeline is paused, which means that no physics steps will occur until the user makes it play either programmatically or
        through the left-hand UI toolbar.
        """
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_b_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "STOP"

        Pausing the timeline on b_text is not strictly necessary for this example to run.
        Clicking "STOP" will cancel the physics subscription that updates the scenario, which means that
        the robot will stop getting new commands and the cube will stop updating without needing to
        pause at all.  The reason that the timeline is paused here is to prevent the robot being carried
        forward by momentum for a few frames after the physics subscription is canceled.  Pausing here makes
        this example prettier, but if curious, the user should observe what happens when this line is removed.
        """
        self._timeline.pause()

    def _reset_extension(self):
        """This is called when the user opens a new stage from self.on_stage_event().
        All state should be reset.
        """
        # Tear down the previous scenario before _on_init() discards it. Otherwise
        # opening a new stage orphans the old scenario's render-product annotators
        # (GPU caches), the carb keyboard subscription (which keeps a strong ref to
        # the scenario, preventing GC), and any rclpy nodes/context it acquired.
        # teardown_scenario() is null-safe on a never-set-up scenario.
        if getattr(self, "_scenario", None) is not None:
            try:
                self._scenario.teardown_scenario()
            except Exception as exc:  # noqa: BLE001
                print(f"[OceanSim] scenario teardown on stage-open warning: {exc}")
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._reset_btn.enabled = False


    def _on_omnigraph_ros_checkbox_click_fn(self, model):
        self._use_omnigraph_ros = model
        print("Reload the scene for changes to take effect.")

    def _on_imu_checkbox_click_fn(self, model):
        self._use_imu = model
        print("Reload the scene for changes to take effect.")

    def _on_sonar_checkbox_click_fn(self, model):
        self._use_sonar = model
        print('Reload the scene for changes to take effect.')

    def _on_camera_checkbox_click_fn(self, model):
        self._use_camera = model
        print('Reload the scene for changes to take effect.')

    def _on_DVL_checkbox_click_fn(self, model):
        self._use_DVL = model
        print('Reload the scene for changes to take effect.')

    def _on_baro_checkbox_click_fn(self, model):
        self._use_baro = model
        print('Reload the scene for changes to take effect.')
    
    def _on_manual_ctrl_cb_click_fn(self, model):
        self._manual_ctrl = model
        print('Reload the scene for changes to take effect.')

    def _on_ctrl_mode_dropdown_clicked(self, model):
        self._ctrl_mode = model
        print(f'Ctrl mode: {model}. Reload the scene for changes to take effect.')

    def _selected_payloads(self, spec):
        """Payloads for this LOAD from the pickers: per kind, the platform's
        standard choice, none, or the picked catalogue entry."""
        choice = getattr(self, "_payload_choice", {})
        names = []
        standard = [payload_catalogue.get_payload(n) for n in spec.default_payloads]
        for kind in ("sonar", "dvl", "gripper", "light"):
            pick = choice.get(kind, "standard")
            if pick == "standard":
                names += [p.name for p in standard if p.kind == kind]
            elif pick != "none":
                names.append(pick)
        names += [p.name for p in standard if p.kind not in ("sonar", "dvl", "gripper", "light")]
        return payload_catalogue.select_payloads(spec, names)

    def _on_platform_dropdown_clicked(self, label):
        """Map the selected dropdown label back to its canonical platform key
        (utils.platforms). The vehicle is spawned on the next Load."""
        try:
            self._platform = self._platform_keys[self._platform_labels.index(label)]
        except (ValueError, AttributeError):
            self._platform = label  # fall back to treating the label as a key
        print(f'Vehicle platform: {self._platform}. Reload the scene for changes to take effect.')

   
    def _add_extra_ui(self):
        with self.sensor_reading_frame:
            with ui.VStack(spacing=5, height=0):                
                if self._use_DVL is True:
                    self._build_DVL_plot()
                    self.sensor_reading_frame.visible = True
                if self._use_baro is True:
                    self._build_baro_plot()
                    self.sensor_reading_frame.visible = True
                if not self._use_baro and not self._use_DVL:
                    self.sensor_reading_frame.visible = False 
        with self.waypoints_frame:
            if self._ctrl_mode == 'Waypoints':
                self._build_waypoints_filepicker()
                self.waypoints_frame.visible = True
            else:
                self.waypoints_frame.visible = False
        with self.ros2_control_frame:
            if self._ctrl_mode == 'ROS control':
                # Build the ROS2 control UI
                self._build_ros2_control_ui()
                self.ros2_control_frame.visible = True
            else:
                self.ros2_control_frame.visible = False

    def _build_ros2_control_ui(self):
        """Build the ROS2 control UI elements"""
        with self.ros2_control_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # ROS2 control mode dropdown
                self._ros2_control_mode_model = dropdown_builder(
                    label='ROS2 Control Mode',
                    default_val=0,
                    items=['velocity control', 'force control', 'dynamic velocity control',
                           'thruster control'],
                    tooltip=('velocity control: sets the body velocity (kinematic); '
                             'force control: Wrench in N / N*m; dynamic velocity control: '
                             'cmd_vel tracked by a PI loop through forces'),
                    on_clicked_fn=self._on_ros2_control_mode_dropdown_clicked
                )

    def _on_ros2_control_mode_dropdown_clicked(self, mode):
        self._scenario._ros2_control_mode = mode
        if self._scenario._ros2_control_receiver is None:
            print('ROS2 control receiver is not initialized; ignoring control mode change. '
                  'Make sure the isaacsim.ros2.bridge extension is enabled and the scenario is loaded.')
            return
        self._scenario._ros2_control_receiver._setup_ros2_control_mode(
                self._scenario._ros2_control_mode
            )
        print(f'ROS control mode switch to: {self._scenario._ros2_control_mode}.')

    def _build_waypoints_filepicker(self):
        self._waypoints_path_field = str_builder(
            label='Path to waypoints',
            default_val=self._waypoints_path,
            tooltip='Select the txt files containing the waypoint data',
            use_folder_picker=True,
            folder_button_title='Select txt',
            folder_dialog_title='Select the txt file containing the waypoint'
        )
        self._scenario.setup_waypoints(
            waypoint_path=self._waypoints_path, 
            default_waypoint_path=self._extension_path + '/demo/demo_waypoints.txt'
            )
        self._waypoints_path_field.add_value_changed_fn(self._on_waypoints_path_changed_fn)

    def _on_waypoints_path_changed_fn(self, model):
        self._waypoints_path = model.get_value_as_string()
        self._scenario.setup_waypoints(
            waypoint_path=model.get_value_as_string(), 
            default_waypoint_path=self._extension_path + '/demo/demo_waypoints.txt'
            )

    def _build_DVL_plot(self):
        self._DVL_event_sub = None
        self._DVL_x_vel = []
        self._DVL_y_vel = []
        self._DVL_z_vel = []

        kwargs = {
            "label": "DVL reading xyz vel (m/s)",
            "on_clicked_fn": self.toggle_DVL_step,
            "data": [self._DVL_x_vel, self._DVL_y_vel, self._DVL_z_vel],
        }
        (
            self._DVL_plot,
            self._DVL_plot_value,
        ) = combo_cb_xyz_plot_builder(**kwargs)
    def toggle_DVL_step(self, val=None):
        print("DVL DAQ: ", val)
        if val:
            if not self._DVL_event_sub:
                self._DVL_event_sub = (
                    omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_DVL_step)
                )
            else:
                self._DVL_event_sub = None
        else:
            self._DVL_event_sub = None

    def _on_DVL_step(self, e: carb.events.IEvent):
        # Casting np.float32 to float32 is necessary for the ui.Plot expects a consistent data type flow
        x_vel = float(self._scenario._DVL_reading[0])
        y_vel = float(self._scenario._DVL_reading[1])
        z_vel = float(self._scenario._DVL_reading[2])

        self._DVL_plot_value[0].set_value(x_vel)
        self._DVL_plot_value[1].set_value(y_vel)
        self._DVL_plot_value[2].set_value(z_vel)

        self._DVL_x_vel.append(x_vel)
        self._DVL_y_vel.append(y_vel)
        self._DVL_z_vel.append(z_vel)
        if len(self._DVL_x_vel) > 50:
            self._DVL_x_vel.pop(0)
            self._DVL_y_vel.pop(0)
            self._DVL_z_vel.pop(0)

        self._DVL_plot[0].set_data(*self._DVL_x_vel)
        self._DVL_plot[1].set_data(*self._DVL_y_vel)
        self._DVL_plot[2].set_data(*self._DVL_z_vel)

    def _build_baro_plot(self):
        self._baro_event_sub = None
        self._baro_data = []

        kwargs = {
                "label": "Barometer reading (Pa)", 
                "on_clicked_fn": self.toggle_baro_step, 
                "data": self._baro_data,
                "min": 101325.0,
                'max': 101325.0 + 50000,
                  }
        self._baro_plot, self._baro_plot_value = combo_cb_plot_builder(**kwargs)


    def toggle_baro_step(self, val=None):
        print('Barometer DAQ: ', val)
        if val:
            if not self._baro_event_sub:
                self._baro_event_sub= (
                    omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_baro_step)
                )
            else:
                self._baro_event_sub = None
        else:
            self._baro_event_sub = None

    def _on_baro_step(self, e: carb.events.IEvent):
        baro = float(self._scenario._baro_reading)
        self._baro_plot_value.set_value(baro)
        self._baro_data.append(baro)
        if len(self._baro_data) > 50:
            self._baro_data.pop(0)
        self._baro_plot.set_data(*self._baro_data)

        
