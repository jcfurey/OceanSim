from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep
import omni.ui as ui
import numpy as np
from omni.replicator.core.scripts.functional import write_np
import warp as wp
import threading
from isaacsim.oceansim.utils.ImagingSonar_kernels import *
from isaacsim.oceansim.utils import sonar_scan_math


# Future TODO
# In future release, wrap this class around RTX lidar

class ImagingSonarSensor(Camera):
    def __init__(self, 
                 prim_path, 
                 name = "ImagingSonar", 
                 frequency = None, 
                 dt = None, 
                 position = None, 
                 orientation = None, 
                 translation = None, 
                 render_product_path = None,
                 physics_sim_view = None,
                 min_range: float = 0.1, # m (Oculus M3000d)
                 max_range: float = 10.0, # m (M3000d 1.2 MHz working range)
                 range_res: float = 10.0 / 1024, # m (working range / 1024-bin budget)
                 hori_fov: float = 130.0, # deg
                 vert_fov: float = 20.0, # deg
                 angular_res: float = 130.0 / 512, # deg (output beam spacing)
                 hori_res: int = 3000, # query render width; height covers the fan's elevation
                 gpu_point_filter: bool = False, # on-device point compaction; skips the
                                      # device->host->device round-trip. Self-heals to the
                                      # numpy path if the AOV outputs aren't on-device Warp arrays.
                 async_compute: bool = False, # run the post-scan kernels + sonar_map readback on
                                      # a worker thread so the sonar doesn't block the sim loop / odom.
                 acoustic_frequency: float = 1.2e6, # Hz, distinct from camera frame rate
                 beam_fwhm_deg: float = 0.6, # M3000d LF angular resolving power
                 ):
        
    
        """Initialize an imaging sonar sensor with physical parameters.
    
        Args:
            prim_path (str): prim path of the Camera Prim to encapsulate or create.
            name (str, optional): shortname to be used as a key by Scene class.
                                    Note: needs to be unique if the object is added to the Scene.
                                    Defaults to "ImagingSonar".
            frequency (Optional[int], optional): Frequency of the sensor (i.e: how often is the data frame updated).
                                                Defaults to None.
            dt (Optional[str], optional): dt of the sensor (i.e: period at which a the data frame updated). Defaults to None.
            resolution (Optional[Tuple[int, int]], optional): resolution of the camera (width, height). Defaults to None.
            position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
            translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                            (with respect to its parent prim). shape is (3, ).
                                                            Defaults to None, which means left unchanged.
            orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                            (depends if translation or position is specified).
                                                            quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                            Defaults to None, which means left unchanged.
            render_product_path (str): path to an existing render product, will be used instead of creating a new render product
                                    the resolution and camera attached to this render product will be set based on the input arguments.
                                    Note: Using same render product path on two Camera objects with different camera prims, resolutions is not supported
                                    Defaults to None

            physics_sim_view (_type_, optional): _description_. Defaults to None.            
            min_range (float, optional): Minimum detection range in meters. Defaults to 0.1.
            max_range (float, optional): Maximum detection range in meters. Defaults to 10.0.
            range_res (float, optional): Range resolution in meters. Defaults to 10 / 1024.
            hori_fov (float, optional): Horizontal field of view in degrees. Defaults to 130.0.
            vert_fov (float, optional): Vertical field of view in degrees. Defaults to 20.0.
            angular_res (float, optional): Output beam spacing in degrees. Defaults to 130 / 512.
            hori_res (int, optional): Horizontal pixel resolution. Defaults to 3000.
            acoustic_frequency (float): Acoustic carrier in Hz, default 1.2 MHz.
            beam_fwhm_deg (float): Beam response FWHM in degrees, default 0.6.
    
        Note:
            - Vertical resolution is automatically calculated to maintain aspect ratio
            - Uses Warp for GPU-accelerated sonar image generation
            - Creates polar coordinate meshgrid for sonar returns processing
        """


        self._name = name
        # Defaults equal sensor_presets.sonar_kwargs for the default payload
        # (Oculus M3000d, 1.2 MHz), which the GUI and ROS runner build from.
        self.max_range = max_range
        self.min_range = min_range
        self.range_res = range_res
        self.hori_fov = hori_fov
        self.vert_fov = vert_fov
        self.angular_res = angular_res
        self.hori_res= hori_res
        # Requested gpu_point_filter; applied in sonar_initialize (which resets the
        # live flag) so it survives (re)initialization.
        self._gpu_point_filter_init = gpu_point_filter
        self._async_compute_init = async_compute
        self.acoustic_frequency = float(acoustic_frequency)
        self.beam_fwhm_deg = float(beam_fwhm_deg)
        # Beam FWHM make_sonar_data last applied (model params can override
        # the device's); azimuth_beamwidth_deg reports it to ROS.
        self._applied_beam_fwhm_deg = self.beam_fwhm_deg

        # Generate sonar map's r and z meshgrid
        self.min_azi = np.deg2rad(90-self.hori_fov/2)
        # Cell centres agree with ROS ranges/bearings. Binning offsets below
        # remain at the cell edges.
        n_range = int(np.ceil((self.max_range - self.min_range) / self.range_res))
        n_beams = int(np.ceil(self.hori_fov / self.angular_res))
        r, azi = np.meshgrid(self.min_range + (np.arange(n_range) + 0.5) * self.range_res,
                            self.min_azi + (np.arange(n_beams) + 0.5) * np.deg2rad(self.angular_res),
                            indexing='ij')
        self.r = wp.array(r, shape=r.shape, dtype=wp.float32)
        self.azi = wp.array(azi, shape=r.shape, dtype=wp.float32)

        # Load array that doesn't change shapes to cuda for reusage memory
        # Users can also automatically see if they have set a reasonable parameter 
        # for sonar map bin size\resolution once load the sensor
        self.bin_sum = wp.zeros(shape=self.r.shape, dtype=wp.float32)
        self.bin_count = wp.zeros(shape=self.r.shape, dtype=wp.int32)
        self.binned_intensity = wp.zeros(shape=self.r.shape, dtype=wp.float32)
        self.sonar_map = wp.zeros(shape=self.r.shape, dtype=wp.vec3)
        self.sonar_image = wp.zeros(shape=(self.r.shape[0], self.r.shape[1], 4), dtype=wp.uint8)
        self.gau_noise = wp.zeros(shape=self.r.shape, dtype=wp.float32)
        self.range_dependent_ray_noise = wp.zeros(shape=self.r.shape, dtype=wp.float32)
        # Per-bin segmentation (upstream OceanSim 0.2). Only filled when
        # sonar_initialize(segmentation=True); see make_sonar_data.
        self.bin_min_zenith = wp.full(shape=self.r.shape, value=wp.PI, dtype=wp.float32)
        self.bin_semantics = wp.zeros(shape=self.r.shape, dtype=wp.uint32)
        self.sonar_semantics_image = wp.zeros(shape=(self.r.shape[0], self.r.shape[1], 4), dtype=wp.uint8)

        # Binning grid as the sonarGrid struct the upstream kernels (bin_process,
        # FLS_KittiWriter) take. angular_res stays in DEGREES on the sensor (the
        # ROS publisher reads it); the grid holds radians.
        self.sonar_grid = sonarGrid()
        self.sonar_grid.x_offset = self.min_range
        self.sonar_grid.y_offset = self.min_azi
        self.sonar_grid.x_res = self.range_res
        self.sonar_grid.y_res = float(np.deg2rad(self.angular_res))
        self.sonar_grid.x_num = self.r.shape[0]
        self.sonar_grid.y_num = self.r.shape[1]

        # Render tall enough to cover elevation +-vert_fov/2 across the whole
        # fan (pixels are square, and a pinhole's vertical extent for a fixed
        # elevation grows toward the fan edges); the binning kernels then drop
        # everything outside +-vert_fov/2. The old hori_res / AR (a ratio of
        # angles) gave ~36.5 deg at boresight but only ~16 deg at the 65 deg
        # edge for a 130 x 20 deg sonar. See sonar_scan_math.sonar_render_height.
        self.vert_res = sonar_scan_math.sonar_render_height(
            self.hori_res, self.hori_fov, self.vert_fov)
        self._half_vfov = float(np.deg2rad(self.vert_fov) / 2.0)
        # Pinhole focal length in pixels for the horizontal FOV (square pixels).
        self._fx_px = self.hori_res / (2.0 * np.tan(np.deg2rad(self.hori_fov) / 2.0))
        # Depth (distance to the image plane) admitting every point at slant
        # range >= min_range; binning rejects what's closer than min_range.
        self._depth_near = sonar_scan_math.slant_range_near_depth(
            self.min_range, self.hori_res, self.vert_res, self._fx_px, self._fx_px)
        self._fan_image = None
        self._fan_display_image = None
        self._fan_semantics_image = None
        self._fan_stale = True
        
        super().__init__(prim_path=prim_path, 
                         name=name, 
                         frequency=frequency,
                         dt=dt, 
                         resolution=[self.hori_res, self.vert_res],
                         position=position, 
                         orientation=orientation, 
                         translation=translation, 
                         render_product_path=render_product_path)

        # Near clip at the depth of min_range SLANT range along the corner ray
        # (clipping depth at min_range cut targets nearer than
        # min_range / cos(65 deg) = 0.47 m at the fan edge for a 0.2 m sonar).
        self.set_clipping_range(
            near_distance=self._depth_near,
            far_distance=self.max_range
        )
        # Isaac Sim 6.0.1 port: do NOT call self.initialize() here. The runner
        # builds the sonar BEFORE world.reset() (oceansim_ros2.py), and reset
        # reopens the stage -- which invalidates a render product created in
        # __init__ (hydra texture gets released -> native SIGSEGV in
        # librtx.syntheticdata when the annotators later attach). Mirror UW_Camera:
        # the scenario calls sonar_initialize() AFTER world.reset(), so we defer
        # initialize() + the aperture setup into sonar_initialize() below.
        # Notice if you would like to observe sonar view from linked viewport.
        # Only horizontal fov is displayed correctly while the vertical fov is
        # followed by your viewport aspect ratio settings.
        

    # Initialize the sensor so that annotator is 
    # loaded on cuda and ready to acquire data
    # Data is generated per simulation tick

    # do_array_copy: If True, retrieve a copy of the data array. 
    # This is recommended for workflows using asynchronous
    # backends to manage the data lifetime. 
    # Can be set to False to gain performance if the data is 
    # expected to be used immediately within the writer. Defaults to True.

    def sonar_initialize(self,
                         output_dir : str = None,
                         viewport: bool = True,
                         include_unlabelled = False,
                         if_array_copy: bool = True,
                         normalizing_method: str = "range",
                         privileged_bbox: bool = False,
                         segmentation: bool = False):
        """Initialize sonar data processing pipeline and annotators.
    
        Args:
            output_dir (str, optional): Directory to save sonar data. Defaults to None.
                                        If set to None, sonar will not write data.
            viewport (bool, optional): Enable viewport visualization. Defaults to True.
                                        Set to False for Sonar running without visualization.
            normalizing_method (str, optional): Default normalization for make_sonar_data:
                                        "range" (per-range max) or "all" (global max).
                                        A normalizing_method passed to make_sonar_data overrides it.
            privileged_bbox (bool, optional): Attach the bounding_box_3d_fast annotator so
                                        get_priviledged_bbox() can project scene bboxes onto the
                                        sonar grid. Defaults to False.
            segmentation (bool, optional): Also bin per-point semantic labels (bin_semantics)
                                        and show the segmentation panel next to the sonar image in
                                        the viewport. The labels are the sensor's semantic ids
                                        (the 'reflectivity' semantic type). Defaults to False:
                                        it adds two kernels and a second viewport upload per scan.
            include_unlabelled (bool, optional): Include unlabelled objects to be scanned into sonar view. Defaults to False.
            if_array_copy (bool, optional): If True, retrieve a copy of the data array. 
                                            This is recommended for workflows using asynchronous backends to manage the data lifetime. 
                                            Can be set to False to gain performance if the data is expected to be used immediately within the writer. 
                                            Defaults to True.
                                            
        Note:
            - Attaches pointcloud, camera params, and semantic segmentation annotators
            - Sets up Warp arrays for sonar image processing
            - Can optionally write data to disk if output_dir specified
        """
        self.writing = False
        self._viewport = viewport
        self._privileged_bbox = privileged_bbox
        self._segmentation = segmentation
        if normalizing_method not in ("range", "all"):
            raise ValueError(f"[{self._name}] normalizing_method must be 'range' or 'all', "
                             f"got {normalizing_method!r}")
        self._normalizing_method = normalizing_method
        self._device = str(wp.get_preferred_device())
        self.scan_data = {}
        self.id = 0
        self._scan_logged = False  # one-shot shape diagnostic in scan()

        # Optional on-device point compaction (compact_depth_points kernel).
        # Default OFF: the kernel is unit tested against the numpy reference
        # (tests/test_imaging_sonar_kernels.py), but whether the AOV annotators
        # actually return Warp arrays resident on
        # self._device can only be confirmed on hardware. So it self-heals --
        # if the outputs are not on-device Warp arrays (or anything throws) it
        # disables itself and falls back to the proven numpy path. Enable with
        # `sensor.gpu_point_filter = True` after sonar_initialize(), or pass
        # gpu_point_filter=True to the constructor (honored here).
        self.gpu_point_filter = self._gpu_point_filter_init

        # Async compute worker (opt-in). scan() stays on the caller/main thread
        # (it reads the render annotators, which is not thread-safe); the post-scan
        # kernels + the sonar_map host readback run on this worker so they don't
        # block the sim loop (and thus odom/imu). The worker only touches device
        # buffers the main thread isn't using -- scan() is gated on _async_busy so
        # it never overwrites scan_data mid-process.
        # Re-init guard: if sonar_initialize() is called again without close()
        # (which joins the worker via stop_async), the old worker thread would
        # survive, pick up the NEW event object on its next self-lookup, and run
        # make_sonar_data concurrently with the new worker on the same device
        # buffers. Stop any previous worker before rebuilding the async state.
        if getattr(self, "_async_thread", None) is not None:
            self.stop_async()
        # Same re-init concern for the render annotators: a second
        # sonar_initialize() without an intervening close() would recreate
        # cameraParams_annot and re-add the depth/normals/semantics AOVs below,
        # leaking the previous annotator (still attached to the old render
        # product and held in AnnotatorCache) and double-adding the AOVs to the
        # SDG graph (growing per-frame render cost each re-init). Tear the
        # previous ones down first -- mirroring close() -- while the OLD render
        # product is still current (self.initialize() below rebuilds it). Guard
        # the hydra-texture updates the same way close() does (detaching mutates
        # the SDG graph). Defensive: a cleanup hiccup must never abort re-init.
        if getattr(self, "cameraParams_annot", None) is not None:
            _old_rp = getattr(self, "_render_product", None)
            if _old_rp is not None:
                try:
                    _old_rp.hydra_texture.set_updates_enabled(False)
                except Exception:  # noqa: BLE001
                    pass
            try:
                self.remove_distance_to_image_plane_from_frame()
                self.remove_normals_from_frame()
                self.remove_semantic_segmentation_from_frame()
                self.cameraParams_annot.detach(self._render_product_path)
                rep.AnnotatorCache.clear(self.cameraParams_annot)
            except Exception as exc:  # noqa: BLE001
                print(f"[{self._name}] re-init annotator cleanup warning: {exc}", flush=True)
            # Separate guard so a failure above can't orphan the bbox annotator
            # (still attached to the old render product) when it's reset below.
            if getattr(self, "bbox_annot", None) is not None:
                try:
                    self.bbox_annot.detach(self._render_product_path)
                    rep.AnnotatorCache.clear(self.bbox_annot)
                except Exception as exc:  # noqa: BLE001
                    print(f"[{self._name}] re-init bbox annotator cleanup warning: {exc}", flush=True)
                self.bbox_annot = None
            if _old_rp is not None:
                try:
                    _old_rp.hydra_texture.set_updates_enabled(True)
                except Exception:  # noqa: BLE001
                    pass
        self.async_compute = self._async_compute_init
        self._async_busy = False
        # (capture_sim_time, grid) -- capture_sim_time is the sim time scan()
        # actually ran at (recorded on the main thread in _submit_scan_async),
        # NOT the time the worker finished processing it or the time the
        # publisher later reads it. Lets the publisher stamp messages with
        # when the data was captured instead of "now", and lets a stalled/dead
        # worker's frozen frame be detected (the capture time stops advancing).
        self._async_result = None
        self._async_capture_time = None
        self._async_lock = threading.Lock()
        self._async_scan_evt = threading.Event()
        self._async_stop = False
        self._async_params = {}
        self._async_thread = None
        # Sim time of the most recent successful scan in SYNC (non-async) mode,
        # for the same reason -- kept separate from _async_capture_time so
        # get_sonar_map_np() has a uniform (capture_time, grid) contract
        # regardless of backend mode.
        self._last_sync_capture_time = None
        if self.async_compute:
            self._async_thread = threading.Thread(
                target=self._async_worker, name=f"{self._name}_sonar_worker", daemon=True)
            self._async_thread.start()
            print(f"[{self._name}] async sonar compute enabled (worker thread started)", flush=True)

        self._gpu_out_pcl = None
        self._gpu_out_normals = None
        self._gpu_out_sem = None
        self._gpu_counter = None

        # Reusable per-point work buffers for make_sonar_data. The valid-point
        # count varies frame to frame, so these grow on demand to the running
        # high-water mark (kernels launch over [:num_points] views) instead of
        # allocating three fresh device arrays every frame.
        self._wp_intensity = None
        self._wp_pcl_spher = None
        self._wp_pcl_bin_idx = None
        # Cached reflectivity lookup (semantic id -> reflectivity) keyed on the
        # (idToLabels, query_prop) it was built from, so the GPU upload is only
        # redone when the labelled-mesh set in the FOV changes.
        self._refl_cache_key = None
        self._refl_cache_arr = None
        # Fixed-shape normalization-max buffers, preallocated and re-zeroed each
        # frame (global max -> shape (1,), per-range max -> shape (n_range,)).
        self._max_all = wp.zeros(shape=(1,), dtype=wp.float32, device=self._device)
        self._max_range = wp.zeros(shape=(self.r.shape[0],), dtype=wp.float32, device=self._device)
        # Per-beam solid-angle gains for the render camera (static): binning
        # weights each point by its pixel's solid angle so a beam integrates
        # over its real solid angle instead of counting render pixels.
        self._beam_gain = wp.array(sonar_scan_math.beam_solid_angle_gain(
            self.hori_res, self.vert_res, self._fx_px, self._fx_px,
            self.min_azi, np.deg2rad(self.angular_res), self.r.shape[1],
            np.deg2rad(self.hori_fov), self._half_vfov), dtype=wp.float32, device=self._device)
        self._dummy_instances = None   # compact_depth_points' unused instance output
        self._point_check_done = False  # one-time depth-vs-get_pointcloud diagnostic

        # Initialize the camera (creates the render product) HERE -- post
        # world.reset() (deferred from __init__ for the Isaac Sim 6.0.1 port; see
        # __init__). UW_Camera does the same via its scenario-driven initialize().
        # Then set the horizontal aperture (needs initialize() first, per the
        # upstream aperture-ordering bug) so the sonar FOV geometry is correct.
        self.initialize()
        self.focal_length = self.get_focal_length()
        horizontal_aper = 2 * self.focal_length * np.tan(np.deg2rad(self.hori_fov) / 2)
        self.set_horizontal_aperture(horizontal_aper)
        # Explicitly set both apertures to the query render's aspect ratio;
        # a viewport's aspect must never change the sonar elevation coverage.
        self.set_vertical_aperture(horizontal_aper * self.vert_res / self.hori_res)

        # Isaac Sim 6.0.1 port (FIX for the world.play() SIGSEGV): the old
        # `pointcloud` COMPOSITE annotator crashes natively at play() on 6.0.1
        # (dangling SdfPath in the RTX SDG pipeline; bisected to this one
        # annotator). Replace it with PRIMITIVE AOVs and reconstruct the point
        # cloud from depth, exactly as Isaac's own Camera.get_pointcloud() does:
        #   - distance_to_image_plane (depth) -> Camera.get_pointcloud() world pts
        #   - normals                          -> per-pixel surface normals
        #   - semantic_segmentation            -> per-pixel reflectivity labels
        # distance_to_camera is proven safe on 6.0.1 (UW_Camera uses it); these
        # primitive per-pixel annotators avoid the composite that broke. We attach
        # them through the base Camera's add_*_to_frame() helpers so get_depth() /
        # get_pointcloud() can consume them. CameraParams stays (lightweight
        # metadata) for the cameraViewTransform the warp kernels need.
        self.cameraParams_annot = rep.AnnotatorRegistry.get_annotator(
            name="CameraParams",
            do_array_copy=if_array_copy,
            device=self._device
            )

        print(f'[{self._name}] Using {self._device}' )
        print(f'[{self._name}] Render query res: {self.hori_res} x {self.vert_res}. Binning res: {self.r.shape[0]} x {self.r.shape[1]}')

        # Attach with hydra-texture updates disabled (NVIDIA MobilityGen pattern,
        # IsaacSim .../mobility_gen/.../camera.py enable_rendering/finalize_rendering)
        # so the SDG graph is never evaluated half-built during attach.
        _rp = getattr(self, "_render_product", None)
        if _rp is not None:
            _rp.hydra_texture.set_updates_enabled(False)
        self.add_distance_to_image_plane_to_frame()
        self.add_normals_to_frame()
        # Segment by the custom 'reflectivity' semantic type (the OceanSim runner
        # labels meshes via add_labels(..., instance_name='reflectivity', e.g. tank
        # '1.0', rock '2.0')). The default 'class' type yields only BACKGROUND/
        # UNLABELLED, so make_indexToProp would fall back to uniform reflectivity=1.
        self.add_semantic_segmentation_to_frame(
            init_params={"semanticTypes": ["reflectivity"], "colorize": False})
        self.cameraParams_annot.attach(self._render_product_path)
        self.bbox_annot = None
        if self._privileged_bbox:
            self.bbox_annot = rep.AnnotatorRegistry.get_annotator(
                name='bounding_box_3d_fast',
                do_array_copy=if_array_copy,
            )
            self.bbox_annot.attach(self._render_product_path)
        if _rp is not None:
            _rp.hydra_texture.set_updates_enabled(True)

        if output_dir is not None:
            self.writing = True
            self.backend = rep.BackendDispatch({"paths": {"out_dir": output_dir}})
        if self._viewport:
            self.make_sonar_viewport()
        
        print(f'[{self._name}] Initialized successfully. Data writing: {self.writing}')

        self.bin_sum.zero_()
        self.bin_count.zero_()
        self.binned_intensity.zero_()
        self.sonar_map.zero_()
        self.sonar_image.zero_()
        self.range_dependent_ray_noise.zero_()
        self.gau_noise.zero_()
        self.bin_semantics.zero_()
        self.sonar_semantics_image.zero_()

        

    def scan(self):

        """Capture a single sonar scan frame and store the raw data.

        Returns:
            bool: True if scan was successful (valid data received), False otherwise

        Note:
            - Stores pointcloud, normals, semantics, and camera transform in scan_data dict
            - First few frames may be empty due to CUDA initialization
            - Automatically skips frames with no detected objects
        """
        # Due to the time to load annotators to cuda, the first few simulation ticks give no annotation in memory.
        # This is also the case when no mesh is within the sonar fov.
        # Semantic labels double as the warmup gate.
        sem_annot = self._custom_annotators["semantic_segmentation"]

        # On the GPU fast path (the runner default) fetch semantics straight to
        # the device and reuse that same dict both for the idToLabels gate here
        # and for the compaction kernel in _scan_gpu_compact. The old code always
        # did a full-image HOST get_data() here purely to read idToLabels, then
        # _scan_gpu_compact re-fetched the array on-device -- so every default
        # scan paid a wasted HxW device->host readback that was immediately
        # discarded. idToLabels is host-computed annotator metadata that get_data()
        # returns regardless of device; if a build ever omits `info` from the
        # device dict, fall back to a host fetch for the labels (the proven path)
        # so the warmup/empty-FOV gate is never wrongly tripped.
        if self.gpu_point_filter:
            sem_data = sem_annot.get_data(device=self._device)
            try:
                id_to_labels = self._annot_get(sem_data, ('info', 'idToLabels'),
                                               'semantic_segmentation')
            except KeyError:
                id_to_labels = self._annot_get(sem_annot.get_data(),
                                               ('info', 'idToLabels'), 'semantic_segmentation')
        else:
            sem_data = sem_annot.get_data()
            id_to_labels = self._annot_get(sem_data, ('info', 'idToLabels'),
                                           'semantic_segmentation')

        # No labels yet (CUDA warmup) or nothing in the FOV -> skip this frame.
        if len(id_to_labels) == 0:
            return False

        # Optional on-device fast path: unproject + compact the in-range points
        # with the compact_depth_points kernel so the per-pixel depth/normals/semantics
        # never round-trip device->host->device. Returns True/False on success,
        # or None if it cannot run on-device (in which case it disables itself
        # and we drop through to the numpy path). See sonar_initialize().
        if self.gpu_point_filter:
            result = self._scan_gpu_compact(id_to_labels, sem_data)
            if result is not None:
                return result
            self.gpu_point_filter = False
            print(f"[{self._name}] gpu_point_filter unavailable (annotator outputs "
                  f"not Warp arrays on '{self._device}'); using numpy scan path.",
                  flush=True)
            # The numpy path needs HOST-resident semantics; the device dict above
            # isn't it, so fetch the host copy for the fallback.
            sem_data = sem_annot.get_data()

        return self._scan_numpy(sem_data, id_to_labels)

    def _scan_gpu_compact(self, id_to_labels, sem_dict):
        """On-device point selection via the compact_depth_points kernel.

        ``sem_dict`` is the semantic-segmentation annotator's already-fetched
        on-device get_data() dict (fetched once in scan(), reused here) so this
        method doesn't re-issue the device get_data() for it.

        Keeps the depth / normals / semantics AOVs on ``self._device``,
        unprojects depth to world points with this frame's CameraParams and
        appends the in-range, finite points into reusable output buffers with an
        atomic counter -- the GPU equivalent of _scan_numpy's
        ``sonar_scan_math.depth_to_world_points`` + ``select_in_range_points``
        (proven equal in tests/test_imaging_sonar_kernels.py).

        Returns:
            True  - in-range points stored in scan_data.
            False - valid frame but no in-range points (skip, same as numpy path).
            None  - the AOV outputs are not on-device Warp arrays (or something
                    threw); caller should fall back to the numpy path.
        """
        try:
            depth = self._custom_annotators["distance_to_image_plane"].get_data(device=self._device)
            normals = self._custom_annotators["normals"].get_data(device=self._device)
            sem = self._annot_get(sem_dict, ('data',), 'semantic_segmentation')

            # The fast path only engages when every AOV is genuinely a Warp array
            # resident on self._device with the dtype the kernel expects. Any
            # mismatch -> None -> numpy fallback (no silent host round-trip).
            depth = self._require_warp(depth, wp.float32)
            normals = self._require_warp(normals, wp.float32)
            sem = self._require_warp(sem, wp.uint32)
            if depth is None or normals is None or sem is None:
                return None

            # Image-shaped views for the 2-D kernel. Warp's reshape needs
            # C-contiguity and the AOVs can come back strided, so contiguous()
            # first (a no-op when already contiguous; otherwise a GPU->GPU copy).
            width, height = (int(v) for v in self.get_resolution())
            n_px = width * height
            if depth.size != n_px or sem.size != n_px or normals.size % n_px:
                return None
            depth_2d = depth.contiguous().reshape((height, width))
            sem_2d = sem.contiguous().reshape((height, width))
            n_ch = normals.size // n_px
            if n_ch < 3:
                return None
            nm_3d = normals.contiguous().reshape((height, width, n_ch))

            # World points from depth with THIS frame's CameraParams -- the same
            # render-time pose world2local / compute_intensity use below.
            # (Camera.get_pointcloud() unprojected with the CURRENT USD pose, so
            # any render lag misregistered the image: 1 deg -> 3-4 beams.)
            cam_data = self.cameraParams_annot.get_data()
            view_tf = self._annot_get(cam_data, ('cameraViewTransform',), 'CameraParams')
            cam_to_world, fx, fy, cx, cy = sonar_scan_math.depth_unprojection_from_camera_params(
                cam_data, width, height)

            # Reusable, device-resident output buffers sized to the full pixel
            # count (the worst case all-in-range). Allocated once, kept across
            # frames; only the counter is re-zeroed each scan.
            if self._gpu_out_pcl is None or self._gpu_out_pcl.shape[0] != n_px:
                self._gpu_out_pcl = wp.zeros((n_px, 3), dtype=wp.float32, device=self._device)
                self._gpu_out_normals = wp.zeros((n_px, 3), dtype=wp.float32, device=self._device)
                self._gpu_out_sem = wp.zeros(n_px, dtype=wp.uint32, device=self._device)
                self._dummy_instances = wp.zeros(n_px, dtype=wp.uint32, device=self._device)
                self._gpu_counter = wp.zeros(1, dtype=wp.int32, device=self._device)
                self._gpu_keep_all = wp.zeros(1, dtype=wp.uint8, device=self._device)
            self._gpu_counter.zero_()

            wp.launch(kernel=compact_depth_points,
                      dim=(height, width),
                      inputs=[depth_2d, nm_3d, sem_2d,
                              sem_2d,                 # instance ids: unused by the sensor
                              self._gpu_keep_all,     # exclude no semantic id
                              wp.mat44(cam_to_world.astype(np.float32)),
                              float(fx), float(fy), float(cx), float(cy),
                              float(self._depth_near), float(self.max_range),
                              self._gpu_counter],
                      outputs=[self._gpu_out_pcl, self._gpu_out_normals,
                               self._gpu_out_sem, self._dummy_instances],
                      device=self._device)
            # counter.numpy() already does a blocking default-stream device->host
            # copy that orders this readback, so a global wp.synchronize() here
            # only adds an unnecessary all-device stall on the sim thread.
            n_valid = min(int(self._gpu_counter.numpy()[0]), n_px)

            if not self._scan_logged:
                self._scan_logged = True
                print(f"[{self._name}] scan(gpu): N={n_px} valid={n_valid} "
                      f"idToLabels={id_to_labels}", flush=True)
                self._log_point_source_check(depth_2d.numpy(), cam_data)
            if n_valid == 0:
                return False

            # Contiguous prefix views into the reusable buffers. They stay valid
            # through make_sonar_data's kernels (all run before the next scan()).
            self.scan_data['pcl'] = self._gpu_out_pcl[:n_valid]          # (N,3)
            self.scan_data['normals'] = self._gpu_out_normals[:n_valid]  # (N,3)
            self.scan_data['semantics'] = self._gpu_out_sem[:n_valid]    # (N,)
            self.scan_data['viewTransform'] = np.asarray(view_tf).reshape(4, 4).T
            self.scan_data['idToLabels'] = id_to_labels
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[{self._name}] gpu_point_filter error ({exc!r}); "
                  f"falling back to numpy scan path.", flush=True)
            return None

    def _log_point_source_check(self, depth_np, cam_data):
        """One-time hardware check of the depth + CameraParams unprojection:
        log how far its world points are from Camera.get_pointcloud()'s (which
        uses the current USD pose). On a static first frame they should agree
        to ~mm; a large gap means the CameraParams projection assumptions
        don't hold on this build. Diagnostic only -- never raises."""
        if getattr(self, "_point_check_done", True):
            return
        self._point_check_done = True
        try:
            depth_np = np.asarray(depth_np, dtype=np.float64)
            h, w = depth_np.shape
            c2w, fx, fy, cx, cy = sonar_scan_math.depth_unprojection_from_camera_params(cam_data, w, h)
            ours = sonar_scan_math.depth_to_world_points(depth_np, c2w, fx, fy, cx, cy)
            ref = self._to_numpy(self.get_pointcloud(device=self._device, world_frame=True))
            ref = np.asarray(ref, dtype=np.float64).reshape(-1, 3)
            ok = (np.isfinite(depth_np.reshape(-1)) & (depth_np.reshape(-1) > 0)
                  & np.all(np.isfinite(ref), axis=1))
            if ref.shape[0] != ours.shape[0] or not np.any(ok):
                print(f"[{self._name}] point-source check skipped (get_pointcloud {ref.shape}, "
                      f"depth {depth_np.shape})", flush=True)
                return
            d = np.linalg.norm(ours[ok] - ref[ok], axis=1)
            print(f"[{self._name}] point-source check vs get_pointcloud: median "
                  f"{np.median(d) * 1e3:.2f} mm, p99 {np.percentile(d, 99) * 1e3:.2f} mm, "
                  f"max {d.max() * 1e3:.2f} mm over {int(ok.sum())} px", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{self._name}] point-source check failed: {exc!r}", flush=True)

    def _require_warp(self, arr, dtype):
        """Return ``arr`` only if it is a Warp array of ``dtype`` already
        resident on ``self._device``; otherwise None (so the caller falls back
        rather than silently copying through the host)."""
        if not isinstance(arr, wp.array):
            return None
        if str(arr.device) != self._device:
            return None
        if arr.dtype != dtype:
            return None
        return arr

    def _ensure_point_buffers(self, num_points):
        """Grow the reusable per-point work buffers so they hold at least
        ``num_points`` entries. Only reallocates when a frame needs more points
        than any previous frame; steady state reuses the same device arrays."""
        cap = 0 if self._wp_intensity is None else self._wp_intensity.shape[0]
        if cap < num_points:
            self._wp_intensity = wp.empty(shape=(num_points,), dtype=wp.float32, device=self._device)
            self._wp_pcl_spher = wp.empty(shape=(num_points,), dtype=wp.vec3, device=self._device)
            self._wp_pcl_bin_idx = None
        # Only the segmentation path needs the per-point bin index.
        if self._segmentation and (self._wp_pcl_bin_idx is None
                                   or self._wp_pcl_bin_idx.shape[0] < num_points):
            self._wp_pcl_bin_idx = wp.empty(shape=(self._wp_intensity.shape[0],),
                                            dtype=wp.vec2ui, device=self._device)

    @staticmethod
    def make_indexToProp_array(idToLabels: dict, query_property: str) -> np.ndarray:
        """Convert idToLabels into an indexToProp array (semantic id -> property
        value), the layout the Warp kernels index into. Upstream API; delegates to
        the unit-tested sonar_scan_math implementation."""
        return sonar_scan_math.make_indexToProp_array(idToLabels, query_property)

    def _get_indexToRefl(self, id_to_labels, query_prop):
        """Return the device reflectivity-lookup array for ``id_to_labels`` /
        ``query_prop``, rebuilding + re-uploading only when those inputs change
        (the common case is identical labels frame to frame)."""
        key = (query_prop, tuple(sorted(
            (k, tuple(sorted(v.items())) if isinstance(v, dict) else v)
            for k, v in id_to_labels.items())))
        if key != self._refl_cache_key:
            arr = sonar_scan_math.make_indexToProp_array(id_to_labels, query_prop)
            self._refl_cache_arr = wp.array(arr, dtype=wp.float32, device=self._device)
            self._refl_cache_key = key
        return self._refl_cache_arr

    def _scan_numpy(self, sem_data, id_to_labels):
        """Reference scan path: pull the AOVs to the host and select in-range
        points with the (unit-tested) pure-numpy sonar_scan_math. This is the
        default and the fallback for the optional GPU path."""
        # Reconstruct world points from the depth AOV (the 'pointcloud' annotator
        # crashed on Isaac 6.x) with THIS frame's CameraParams -- the same
        # unprojection as Camera.get_pointcloud()'s depth fallback, but with the
        # render-time pose that world2local / compute_intensity also use (the
        # current USD pose misregistered the image under motion). Row-major over
        # (H, W), so the per-pixel normals / semantics flatten the same way.
        depth = self._custom_annotators["distance_to_image_plane"].get_data(device=self._device)
        depth_np = np.squeeze(self._to_numpy(depth))
        if depth_np.ndim != 2 or depth_np.size == 0:
            return False
        cam_data = self.cameraParams_annot.get_data()
        view_tf = self._annot_get(cam_data, ('cameraViewTransform',), 'CameraParams')
        cam_to_world, fx, fy, cx, cy = sonar_scan_math.depth_unprojection_from_camera_params(
            cam_data, depth_np.shape[1], depth_np.shape[0])
        pcl_np = sonar_scan_math.depth_to_world_points(
            depth_np, cam_to_world, fx, fy, cx, cy).astype(np.float32)

        normals_img = self._to_numpy(self._custom_annotators["normals"].get_data(device=self._device))
        sem_img = np.squeeze(self._to_numpy(self._annot_get(sem_data, ('data',), 'semantic_segmentation')))

        n_px = depth_np.size
        normals_flat = normals_img.reshape(-1, normals_img.shape[-1])[:, :3]   # (H*W, 3) world normals
        sem_flat = sem_img.reshape(-1).astype(np.uint32)                       # (H*W,)
        if pcl_np.shape[0] != n_px or normals_flat.shape[0] != n_px or sem_flat.shape[0] != n_px:
            # Layout/size mismatch -> can't align points with AOVs; skip safely.
            return False

        # Keep only pixels with a finite hit inside the sonar range window.
        # (sonar_scan_math.select_in_range_points is pure numpy + unit tested; it
        # masks the depth window cheaply over all pixels, then does the per-point
        # finiteness check and the gathers only on the depth-passing subset.)
        depth_flat = depth_np.reshape(-1)
        pcl_v, normals_v, sem_v = sonar_scan_math.select_in_range_points(
            depth_flat, pcl_np, normals_flat, sem_flat, self._depth_near, self.max_range)
        n_valid = pcl_v.shape[0]
        if not getattr(self, "_scan_logged", False):
            self._scan_logged = True
            self._log_point_source_check(depth_np, cam_data)
            uniq_sem = np.unique(sem_v) if n_valid else np.array([])
            print(f"[{self._name}] scan: depth{tuple(depth_np.shape)} pcl{tuple(pcl_np.shape)} "
                  f"normals{tuple(normals_img.shape)} sem{tuple(sem_img.shape)} "
                  f"valid={n_valid}/{n_px}", flush=True)
            # Material-reflectivity check: idToLabels should carry the 'reflectivity'
            # values, and the in-FOV pixels should span >1 semantic id (contrast).
            print(f"[{self._name}] reflectivity: idToLabels={id_to_labels} "
                  f"unique_sem_in_fov={uniq_sem.tolist()[:12]}", flush=True)
        if n_valid == 0:
            return False

        self.scan_data['pcl'] = wp.array(pcl_v, dtype=wp.float32)              # (N,3)
        self.scan_data['normals'] = wp.array(normals_v, dtype=wp.float32)      # (N,3)
        self.scan_data['semantics'] = wp.array(sem_v, dtype=wp.uint32)         # (N,)
        self.scan_data['viewTransform'] = np.asarray(view_tf).reshape(4, 4).T  # 4x4 extrinsic
        self.scan_data['idToLabels'] = id_to_labels                           # dict
        return True

    @staticmethod
    def _to_numpy(arr):
        """Coerce an annotator return (warp array, numpy, or None) to numpy."""
        if arr is None:
            return np.array([])
        if hasattr(arr, "numpy"):
            return arr.numpy()
        return np.asarray(arr)

    @staticmethod
    def _annot_get(data, key_path, annot_name):
        """Fetch a nested key from an annotator's get_data() dict.

        Raises a precise error (naming the missing key and listing what *is*
        present) if the schema differs from what OceanSim expects -- e.g. after
        an Isaac Sim / Replicator upgrade renames or moves an output field --
        instead of an opaque KeyError deep in the scan pipeline.
        """
        node = data
        for i, key in enumerate(key_path):
            if not isinstance(node, dict) or key not in node:
                available = list(node.keys()) if isinstance(node, dict) else type(node).__name__
                raise KeyError(
                    f"[ImagingSonarSensor] '{annot_name}' annotator output is missing "
                    f"'{'->'.join(key_path[:i + 1])}'. The Isaac Sim Replicator annotator "
                    f"schema likely changed. Present at this level: {available}."
                )
            node = node[key]
        return node

    @staticmethod
    def _squeeze_leading(arr, expected_ndim, name):
        """Drop a leading singleton batch dim if present so ``arr`` has
        ``expected_ndim`` dims.

        Handles the (1,N,..) <-> (N,..) pointcloud-annotator shape differences
        between Isaac Sim releases. Raises a clear error if the shape is neither
        the expected layout nor a leading-singleton of it.
        """
        shape = tuple(arr.shape)
        ndim = len(shape)
        if ndim == expected_ndim:
            return arr
        if ndim == expected_ndim + 1 and shape[0] == 1:
            return arr[0]
        raise ValueError(
            f"[ImagingSonarSensor] unexpected '{name}' shape {shape}; expected "
            f"{expected_ndim} dim(s), optionally with a leading singleton. The "
            f"Replicator pointcloud annotator layout may have changed."
        )


    def make_sonar_data(self, 
                        binning_method: str = "sum", 
                        normalizing_method: str = None, # None -> sonar_initialize's choice
                        query_prop: str ='reflectivity', # Do not modify this if not developing the sensor.
                        attenuation: float = 0.1, # Control the attentuation along distance when computing attenuation
                        gau_noise_param: float = 0.2, # multiplicative noise coefficient 
                        ray_noise_param: float = 0.05, # additive noise parameter
                        intensity_offset: float = 0.0, # offset intensity after normalization
                        intensity_gain: float = 1.0, # scale intensity after normalization
                        gamma: float = 1.0, # echo display gamma after normalisation; <1 lifts weak echoes
                        central_peak: float = 0.0, # boresight "streak" strength; 0 = OFF.
                                      # A real Oculus has no persistent full-range centre
                                      # stripe, so this is off by default. Set >0 (e.g. 2)
                                      # only to deliberately model a sensor that shows one.
                        central_std: float = 0.001, # spread of the streak (unused when central_peak=0)
                        # --- optional physical model terms, all OFF by default ---
                        spreading_exponent: float = 0.0, # two-way spreading loss r^-n (4: point, ~3: area targets)
                        absorption: float = 0.0, # two-way absorption exp(-2*a*r), a in Np/m
                        tvg_exponent: float = 0.0, # time-varied gain r^+m applied after the losses
                        speckle_looks: int = 0, # >0: Gamma(L) speckle (L=1 single-look) instead of 0.5+N(0, gau_noise_param)
                        speckle_cell: tuple = (1, 1), # speckle correlation cell in (range, azimuth) bins
                        beam_fwhm_deg: float = None, # None -> device response; 0 disables blur
                        sim_time: float = None, # sim time this scan is captured at (for
                                      # publisher header stamps); None if unknown/unused.
                        _skip_scan: bool = False, # internal: worker has already run scan() on
                                      # the main thread; run only the kernels on the live scan_data.
                        ):
        """Process raw scan data into a sonar image with configurable parameters.

        Args:
            binning_method (str): "sum" or "mean" for intensity accumulation
                                Remember to adjust your noise scale accordingly after changing this.
            normalizing_method (str): "all" (global max) or "range" (per-range max).
                                None (default) uses the normalizing_method given to sonar_initialize.
                                Remember to adjust your noise scale accordingly after changing this.
            query_prop (str): Material property to query (default 'reflectivity')
                            Don't modify this if not for development.
            attenuation (float): Distance attenuation coefficient (0-1)
            gau_noise_param (float): Gaussian noise multiplier
            ray_noise_param (float): Rayleigh noise scale factor
            intensity_offset (float): Post-normalization intensity offset
            intensity_gain (float): Post-normalization intensity multiplier
            gamma (float): Display gamma (> 0) on the normalised echo, as an Oculus
                                applies gamma correction: values < 1 compress the echo
                                dynamic range so weak returns stay visible beside a strong
                                broadside one. Applied before the speckle and noise floor.
            central_peak (float): Central beam streak intensity
            central_std (float): Central beam streak width
            spreading_exponent / absorption / tvg_exponent (float): range-dependent
                                echo gain exp(-2 a r) * r^(tvg - spreading) applied per
                                bin before normalisation. "range" normalisation divides
                                each range row by its max and so cancels it -- use
                                normalizing_method="all" to see range effects.
            speckle_looks (int): >0 replaces the 0.5 + N(0, gau_noise_param) multiplier with
                                fully developed speckle, Gamma(L, 1/L) (mean 1, contrast
                                1/sqrt(L)), constant over speckle_cell (range, azimuth) bins.
            beam_fwhm_deg (float): None uses the device beamwidth; >0 blurs each range row along azimuth with a Gaussian of
                                this FWHM -- the beam pattern (e.g. the published
                                azimuth beamwidth), applied before noise.
    
        """



        if normalizing_method is None:
            normalizing_method = getattr(self, "_normalizing_method", "range")
        if not gamma > 0.0:
            raise ValueError(f"[{self._name}] gamma must be positive, got {gamma!r}")
        if beam_fwhm_deg is None:
            beam_fwhm_deg = self.beam_fwhm_deg
        self._applied_beam_fwhm_deg = float(beam_fwhm_deg or 0.0)

        if self.async_compute and not _skip_scan:
            # Main thread: scan (reads annotators) + hand off to the worker; the
            # heavy kernels + sonar_map readback run there. Returns immediately.
            self._submit_scan_async(dict(
                binning_method=binning_method, normalizing_method=normalizing_method,
                query_prop=query_prop, attenuation=attenuation,
                gau_noise_param=gau_noise_param, ray_noise_param=ray_noise_param,
                intensity_offset=intensity_offset, intensity_gain=intensity_gain,
                gamma=gamma, central_peak=central_peak, central_std=central_std,
                spreading_exponent=spreading_exponent, absorption=absorption,
                tvg_exponent=tvg_exponent, speckle_looks=speckle_looks,
                speckle_cell=speckle_cell, beam_fwhm_deg=beam_fwhm_deg),
                sim_time=sim_time)
            return

        if _skip_scan or self.scan():
            if not _skip_scan:
                # True sync-mode scan (not the async worker re-entering with the
                # main thread's already-captured scan_data) -- record when it
                # happened so get_sonar_map_np()/the publisher can stamp with the
                # actual capture time instead of "now".
                self._last_sync_capture_time = sim_time
            num_points = self.scan_data['pcl'].shape[0]
            # Reflectivity lookup (semantic id -> reflectivity). The mapping is a
            # pure function of (idToLabels, query_prop), which only changes when
            # the set of labelled meshes in the FOV changes -- so cache the GPU
            # upload and rebuild only when the inputs differ, instead of building
            # a numpy array and uploading it every frame.
            indexToRefl = self._get_indexToRefl(self.scan_data['idToLabels'], query_prop)
            viewTransform=wp.mat44(self.scan_data['viewTransform'])
            # directly use warp array loaded on cuda
            pcl = self.scan_data['pcl']
            normals = self.scan_data['normals']
            semantics = self.scan_data['semantics']
        else:
            return

        # Compute intensity for each ray query. Reuse the grow-on-demand work
        # buffers (views over [:num_points]) instead of allocating three fresh
        # device arrays every frame.
        self._ensure_point_buffers(num_points)
        intensity = self._wp_intensity[:num_points]
        wp.launch(kernel=compute_intensity,
                  dim=num_points,
                  inputs=[
                      pcl,
                      normals,
                      viewTransform,
                      semantics,
                      indexToRefl,
                      attenuation,
                  ],
                  outputs=[
                      intensity
                  ]
                )
                
        # Transform pointcloud from world cooridates to sonar local and convert to spherical coord
        pcl_spher = self._wp_pcl_spher[:num_points]
        wp.launch(kernel=world2local,
                  dim=num_points,
                  inputs=[
                      viewTransform,
                      pcl
                  ],
                    outputs=[
                      pcl_spher
                    ]
                )
        
        # Collapse three dimensional intensity data to 2D
        # Simply sum intensity return and compute number of return that falls into the same bin
        self.bin_sum.zero_()
        self.bin_count.zero_()
        self.binned_intensity.zero_()

        if self._segmentation:
            # Same binning as bin_intensity, plus each point's bin index and the
            # per-bin minimum zenith, so bin_semantics_process can label each bin
            # with its top-most return (upstream OceanSim 0.2 segmentation).
            pcl_bin_idx = self._wp_pcl_bin_idx[:num_points]
            self.bin_min_zenith.fill_(wp.PI)
            self.bin_semantics.zero_()
            wp.launch(kernel=bin_process,
                      dim=num_points,
                      inputs=[
                          pcl_spher,
                          intensity,
                          semantics,
                          self.sonar_grid,
                          self._half_vfov,
                          self._beam_gain
                      ],
                      outputs=[
                          self.bin_sum,
                          self.bin_count,
                          pcl_bin_idx,
                          self.bin_min_zenith
                      ]
                      )
            wp.launch(kernel=bin_semantics_process,
                      dim=num_points,
                      inputs=[
                          pcl_spher,
                          semantics,
                          pcl_bin_idx,
                          self.bin_min_zenith
                      ],
                      outputs=[
                          self.bin_semantics
                      ]
                      )
        else:
            wp.launch(kernel=bin_intensity_sa,
                      dim=num_points,
                      inputs=[
                          pcl_spher,
                          intensity,
                          self.min_range,
                          self.min_azi,
                          self.range_res,
                          wp.radians(self.angular_res),
                          self._half_vfov,
                          self._beam_gain,
                      ],
                      outputs=[
                          self.bin_sum,
                          self.bin_count
                      ]
                      )
        
        # Process intensity data by either sum as it is or averaging. Use a
        # LOCAL binding for the downstream kernels: the old code REBOUND
        # self.binned_intensity to self.bin_sum in "sum" mode, permanently
        # aliasing the two attributes -- every later zero_() then cleared the
        # same buffer twice, the dedicated binned_intensity allocation was
        # orphaned, and a subsequent "mean"-mode call wrote the average INTO
        # bin_sum through the alias.
        if binning_method == "mean":
            wp.launch(
                kernel=average,
                dim=self.bin_sum.shape,
                inputs=[
                    self.bin_sum,
                    self.bin_count
                ],
                outputs=[
                    self.binned_intensity,
                ]
                )
            binned = self.binned_intensity
        else:  # "sum" (default)
            binned = self.bin_sum

        # Optional physical model terms (all off by default).
        if spreading_exponent or absorption or tvg_exponent:
            wp.launch(kernel=apply_range_gain, dim=binned.shape,
                      inputs=[binned, self.r, float(spreading_exponent), float(absorption),
                              float(tvg_exponent)])
        if beam_fwhm_deg and beam_fwhm_deg > 0.0:
            sigma_bins = float(beam_fwhm_deg) / 2.3548 / float(self.angular_res)
            if getattr(self, "_blur_buf", None) is None:
                self._blur_buf = wp.zeros(shape=self.r.shape, dtype=wp.float32, device=self._device)
            wp.launch(kernel=azimuth_gaussian_blur, dim=binned.shape,
                      inputs=[binned, sigma_bins, max(1, int(np.ceil(3.0 * sigma_bins)))],
                      outputs=[self._blur_buf])
            binned = self._blur_buf


        # gau_noise / range_dependent_ray_noise are fully overwritten every frame
        # by normal_2d / range_dependent_rayleigh_2d (which write every cell), so
        # zeroing them first is redundant. sonar_map.zero_() is kept: an
        # unrecognized normalizing_method runs no map kernel, so it must start clean.
        self.sonar_map.zero_()

        # Calculate multiplicative gaussian noise
        #
        # Seed note: both noise kernels derive per-cell RNG state as
        # rand_init(seed, i*W+j). Passing the SAME seed to both (the old code
        # passed self.id twice) made the Gaussian field bit-identical to the
        # Rayleigh kernel's first draw -- the "independent" multiplicative and
        # additive noise rose and fell together every frame. Splitting the seed
        # space even/odd (2*id vs 2*id+1) keeps the streams disjoint across both
        # kernels AND frames.
        if speckle_looks and int(speckle_looks) > 0:
            # Fully developed speckle (same even seed half), written as
            # (multiplier - 0.5) so the map kernels' (0.5 + noise) applies it.
            cell_r, cell_a = (int(c) for c in speckle_cell)
            wp.launch(kernel=gamma_speckle_2d, dim=self.bin_sum.shape,
                      inputs=[2 * self.id, int(speckle_looks), cell_r, cell_a],
                      outputs=[self.gau_noise])
        else:
            wp.launch(
                kernel=normal_2d,
                dim=self.bin_sum.shape,
                inputs=[
                    2 * self.id,       # frame-incremented seed, even half
                    0.0,
                    gau_noise_param
                ],
                outputs=[
                    self.gau_noise
                ]
            )

        # Calculate additive rayleigh noise (range dependent and mimic central beam)

        wp.launch(
            kernel=range_dependent_rayleigh_2d,
            dim=self.bin_sum.shape,
            inputs=[
                2 * self.id + 1,   # frame-incremented seed, odd half (see above)
                self.r,
                self.azi,
                self.max_range,
                ray_noise_param,
                central_peak,
                central_std,
            ],
            outputs=[
                self.range_dependent_ray_noise

            ]
        )

        
        
        # Normalizing intensity at each bin either by global maximum or rangewise maximum
        # Compute global maximum
        if normalizing_method == "all":
            maximum = self._max_all       # reused (1,) buffer, re-zeroed each frame
            maximum.zero_()
            wp.launch(
                dim=self.bin_sum.shape,
                kernel=compute_max_intensity_all,
                inputs=[
                    binned,
                ],
                outputs=[
                    maximum # wp.array of shape (1,), max value is stored at maximum[0]
                ]
            )
            
            # Apply noise, normalize by global maximum, and convert (r, azi) to (x,y) for plotting
            wp.launch(
                  kernel=make_sonar_map_all,
                  dim=self.sonar_map.shape,
                  inputs=[
                      self.r,
                      self.azi,
                      binned,
                      maximum,
                      self.gau_noise,
                      self.range_dependent_ray_noise,
                      intensity_offset,
                      intensity_gain,
                      gamma
                  ],
                  outputs=[
                      self.sonar_map
                  ]
                  )
            
        if normalizing_method == "range":
            # Compute rangewise maximum
            maximum = self._max_range     # reused (n_range,) buffer, re-zeroed each frame
            maximum.zero_()
            wp.launch(
                dim=self.bin_sum.shape,
                kernel=compute_max_intensity_range,
                inputs=[
                    binned,
                ],
                outputs=[
                    maximum      # wp.array of shape (number of range bins, )
                ]
            )
            # Apply noise, normalize by range maximum, and convert (r, azi) to (x,y) for plotting
            wp.launch(
                  kernel=make_sonar_map_range,
                  dim=self.sonar_map.shape,
                  inputs=[
                      self.r,
                      self.azi, 
                      binned,
                      maximum,
                      self.gau_noise,
                      self.range_dependent_ray_noise,
                      intensity_offset,
                      intensity_gain,
                      gamma
                  ],
                  outputs=[
                      self.sonar_map
                  ]
                  )
        # New sonar_map: the next get_sonar_fan_image reprojects it once,
        # shared by the viewport and the ROS publisher.
        self._fan_stale = True
        
        
        # Write data to the dir
        if self.writing:
            # self.backend.schedule(write_np, f"intensity_{self.id}.npy", data=intensity)
            # self.backend.schedule(write_np, f'pcl_local_{self.id}.npy', data=pcl_local)
            # Snapshot to host BEFORE scheduling: the backend writes
            # asynchronously, and handing it the live reused device sonar_map --
            # which the next frame zeroes and overwrites -- raced the disk write
            # against the next frame's kernels (torn/corrupted .npy contents).
            self.backend.schedule(write_np, f'sonar_data_{self.id}.npy',
                                  data=self.sonar_map.numpy())
            print(f"[{self._name}] [{self.id}] Writing sonar data to {self.backend.output_dir}")
        
        if self._viewport and not self.async_compute:
            # Skip in async mode: this pushes to the Isaac UI byte provider, which
            # must not be touched from the worker thread. ROS consumers read the
            # published sonar_map, not this in-Isaac viewport texture.
            fan = self.get_sonar_fan_image(show_grid=True)
            self._sonar_provider.set_bytes_data_from_gpu(fan.ptr,
                                                       [fan.shape[1], fan.shape[0]])
            if self._segmentation:
                fan_semantics = self.get_semantics_fan_image()
                self._sonar_segmentation_provider.set_bytes_data_from_gpu(
                    fan_semantics.ptr, [fan_semantics.shape[1], fan_semantics.shape[0]])
            # self.backend.schedule(write_image, f'sonar_{self.id}.png', data = self.make_sonar_image())        
            
        self.id += 1
    

    def _submit_scan_async(self, params, sim_time=None):
        """Main-thread half of async sonar: scan() (reads the annotators) then hand
        the device-resident scan_data to the worker. Skips entirely while the worker
        is still processing the previous scan, so scan_data is never overwritten
        mid-process (which also self-throttles scans to the worker's throughput).

        sim_time is the sim time THIS scan actually ran at -- recorded here (main
        thread, at the moment scan() succeeds), not when the worker later finishes
        processing it. _async_capture_time is only written here and only read by
        the worker after this same call sets _async_busy=True, so it can't be
        overwritten mid-cycle without going through the busy-gate above."""
        if self._async_busy:
            return
        if not self.scan():
            return
        self._async_params = params
        self._async_capture_time = sim_time
        self._async_busy = True
        self._async_scan_evt.set()

    def _async_worker(self):
        """Worker half: run the post-scan kernels + sonar_map readback off the sim
        loop. Re-enters make_sonar_data(_skip_scan=True) so the kernel path stays in
        one place. Only touches device buffers the main thread isn't using (gated by
        _async_busy) and the worker-owned result; never reads the annotators."""
        while not self._async_stop:
            if not self._async_scan_evt.wait(timeout=0.5):
                continue
            self._async_scan_evt.clear()
            if self._async_stop:
                break
            try:
                capture_time = self._async_capture_time
                with wp.ScopedDevice(self._device):
                    self.make_sonar_data(_skip_scan=True, **self._async_params)
                    grid = self.sonar_map.numpy()
                with self._async_lock:
                    self._async_result = (capture_time, grid)
            except Exception as exc:  # noqa: BLE001
                print(f"[{self._name}] async sonar worker error ({exc!r})", flush=True)
            finally:
                self._async_busy = False

    def get_sonar_map_np(self):
        """(capture_sim_time, grid) for the publisher, where grid is the host-side
        (n_range, n_azimuth, 3) sonar_map. Async mode returns the worker's latest
        readback (no GPU sync on the caller) paired with the sim time THAT scan
        was captured at -- which can lag the caller's current sim time if the
        worker is still catching up. Sync mode reads the device sonar_map
        directly, paired with the sim time of the scan that produced it. Returns
        None if nothing has been produced yet."""
        if getattr(self, 'async_compute', False):
            with self._async_lock:
                return self._async_result
        sm = getattr(self, 'sonar_map', None)
        if sm is None:
            return None
        grid = sm.numpy() if hasattr(sm, 'numpy') else np.asarray(sm)
        return (self._last_sync_capture_time, grid)

    def set_render_enabled(self, enabled: bool):
        """Enable/disable this sonar camera's render-product updates. The sonar
        raytrace is the dominant per-step render cost, but it only needs to render
        at the scan cadence -- the runner disables it on non-scan steps so the sim
        loop (physics + odom/imu + the GUI viewport) runs unblocked on those steps."""
        rp = getattr(self, "_render_product", None)
        if rp is None:
            return
        try:
            rp.hydra_texture.set_updates_enabled(bool(enabled))
        except Exception:  # noqa: BLE001
            pass

    def ready_for_scan(self) -> bool:
        """True when a new scan can be started. In async mode that means the worker
        isn't still processing the previous scan (else rendering the sonar this step
        would just be wasted -- the scan would be skipped)."""
        if getattr(self, 'async_compute', False):
            return not self._async_busy
        return True

    def stop_async(self):
        """Stop the async worker thread (idempotent)."""
        if getattr(self, '_async_thread', None) is None:
            return
        self._async_stop = True
        self._async_scan_evt.set()
        self._async_thread.join(timeout=2.0)
        if self._async_thread.is_alive():
            # Still mid-scan after the grace period (e.g. a very large grid on a
            # busy GPU taking >2 s). Don't null the handle: a daemon that keeps
            # running would still write sonar_map/_async_result after teardown
            # began, and nulling would silently orphan it. Keeping it set lets a
            # later stop_async()/close() join it again, and surfaces the stall.
            print(f"[{self._name}] async sonar worker did not stop within 2s; "
                  f"leaving the handle for a later join", flush=True)
            return
        self._async_thread = None

    def make_sonar_image(self):
        """Convert processed sonar data to a polar grayscale image.
    
        Returns:
            wp.array: GPU array containing the sonar image (RGBA format)
    
        Note:
            - Polar rows/columns are range/bearing; use get_sonar_fan_image for viewing
            - Image dimensions match the sonar's polar binning resolution
        """
        # make_sonar_image writes all four channels (RGB + A=255) for every pixel
        # via a bijective column map and never reads prior contents, so zeroing
        # the buffer first is redundant. (The init/reset zero in sonar_initialize
        # is untouched.)
        wp.launch(
            dim=self.sonar_map.shape,
            kernel=make_sonar_image,
            inputs=[
                self.sonar_map
            ],
            outputs=[
                self.sonar_image
            ]
        )
        return self.sonar_image

    def _ensure_fan_buffers(self):
        """Build the static fan projection once; no host readback per frame."""
        if self._fan_image is not None:
            return
        range_span, beam_span, scale = sonar_scan_math.sonar_fan_lookup(
            self.min_range, self.max_range, self.range_res, self.min_azi,
            np.deg2rad(self.angular_res), self.r.shape[0], self.r.shape[1], self.hori_fov)
        device = self.sonar_map.device
        self._fan_range_span = wp.array(range_span, dtype=wp.int32, device=device)
        self._fan_beam_span = wp.array(beam_span, dtype=wp.int32, device=device)
        self._fan_guides = wp.array(
            sonar_scan_math.sonar_fan_guides(range_span[..., 0] >= 0, self.max_range, self.hori_fov),
            dtype=wp.uint8, device=device)
        self._fan_metres_per_pixel = scale
        self._fan_image = wp.zeros((*range_span.shape[:2], 4), dtype=wp.uint8, device=device)
        self._fan_display_image = wp.zeros_like(self._fan_image)
        self._fan_stale = True

    def get_sonar_fan_image(self, show_grid: bool = False) -> wp.array:
        """RGBA fan with equal metre scales; each pixel shows the brightest bin
        it covers. Reprojected once per make_sonar_data frame.

        show_grid returns a separate display copy with range rings and bearing
        guides, so the plain image published to ROS never carries them."""
        self._ensure_fan_buffers()
        dim = self._fan_image.shape[:2]
        if self._fan_stale:
            wp.launch(make_sonar_fan_image, dim=dim,
                      inputs=[self.sonar_map, self._fan_range_span, self._fan_beam_span],
                      outputs=[self._fan_image], device=self.sonar_map.device)
            self._fan_stale = False
        if not show_grid:
            return self._fan_image
        wp.launch(overlay_sonar_fan_guides, dim=dim, inputs=[self._fan_guides, self._fan_image],
                  outputs=[self._fan_display_image], device=self.sonar_map.device)
        return self._fan_display_image

    def get_semantics_fan_image(self) -> wp.array:
        """Semantic labels on the same Cartesian fan as the intensity image:
        each pixel labels the bin the intensity fan shows."""
        self._ensure_fan_buffers()
        if self._fan_semantics_image is None:
            self._fan_semantics_image = wp.zeros(self._fan_image.shape, dtype=wp.uint8,
                                                device=self.sonar_map.device)
        wp.launch(make_semantics_fan_image, dim=self._fan_image.shape[:2],
                  inputs=[self.sonar_map, self.get_semantics_image(),
                          self._fan_range_span, self._fan_beam_span],
                  outputs=[self._fan_semantics_image], device=self.sonar_map.device)
        return self._fan_semantics_image

    @property
    def azimuth_beamwidth_deg(self) -> float:
        """Azimuth resolving power of the published sonar data, in degrees:
        the beam FWHM make_sonar_data last applied (the device's until the
        first frame), never finer than the beam spacing. With no beam blur
        the bins themselves set the resolution."""
        return max(self._applied_beam_fwhm_deg, float(self.angular_res))

    # --- Upstream OceanSim 0.2 accessors (sonar_data == sonar_map) ----------

    @property
    def sonar_data(self) -> wp.array:
        """Upstream name for sonar_map: (num_r_bin, num_azi_bin) wp.vec3 of
        [x, y, bin_intensity]."""
        return self.sonar_map

    def get_sonar_data(self) -> wp.array:
        """Get GPU array of sonar data 

        Returns: 
            sonar_data (wp.array(dtype=wp.vec3)): shaped (num_r_bin, num_azi_bin), with each entry a wp.vec3 containing [x, y, bin_intensity]
        with (x, y) are cartesian coordinates of the (r, azi) of the bin.
        """
        return self.sonar_map

    def get_sonar_image(self) -> wp.array:
        """Upstream name for make_sonar_image(): render sonar_map into the RGBA
        sonar_image buffer and return it."""
        return self.make_sonar_image()

    def get_semantics_image(self, colormap : str ='jet') -> wp.array:
        """Convert semantic data to a viewable semantic image
        Returns:
            semantic_image (wp.array(dtype=wp.uint8)) : GPU array containing the semantic image (RGBA) 

        Needs sonar_initialize(segmentation=True); bin_semantics is all zeros
        (background) otherwise.
        """
        id_to_labels = self.scan_data.get('idToLabels') or {}
        # Size the palette by the largest semantic id, not the number of labels:
        # ids need not be contiguous, and bin_semantics holds raw ids.
        num_semantics = max((int(k) for k in id_to_labels.keys()), default=0) + 1
        # The palette only depends on (colormap, num_semantics): build + upload
        # it once, not on every viewport frame.
        key = (colormap, num_semantics)
        if getattr(self, "_semantics_palette_key", None) != key:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(colormap)
            colors = cmap(np.linspace(0, 1, num_semantics)) * 255  # Get n colors from the colormap
            self._semantics_palette = wp.array(data=colors.astype(np.uint8), ndim=2,
                                               dtype=wp.uint8, device=self._device)
            self._semantics_palette_key = key

        wp.launch(
            dim=self.bin_semantics.shape,
            kernel=make_semantics_image,
            inputs=[
                self.bin_semantics,
                self._semantics_palette
            ],
            outputs=[
                self.sonar_semantics_image
            ]
        )

        return self.sonar_semantics_image

    @staticmethod
    def get_bbox_3d_corners(bbox_data):
        """Return transformed points in the following order: [LDB, RDB, LUB, RUB, LDF, RDF, LUF, RUF]
        where R=Right, L=Left, D=Down, U=Up, B=Back, F=Front and LR: x-axis, UD: y-axis, FB: z-axis.

        Args:
            bbox_data (numpy.ndarray): A structured numpy array containing the fields: [`x_min`, `y_min`,
                `x_max`, `y_max`, `transform`.

        Returns:
            corners_world (numpy.ndarray): Transformed corner homogeneous coordinates at world frame with shape `(N, 8, 4)`.
            N: number of bbox, 8: eight corners, 4: homogeneous coordinates [x,y,z,1]
        """

        # extend the demension of input data to fit the format of helper method parameter"""
        rdb = [bbox_data["x_max"], bbox_data["y_min"], bbox_data["z_min"]]
        ldb = [bbox_data["x_min"], bbox_data["y_min"], bbox_data["z_min"]]
        lub = [bbox_data["x_min"], bbox_data["y_max"], bbox_data["z_min"]]
        rub = [bbox_data["x_max"], bbox_data["y_max"], bbox_data["z_min"]]
        ldf = [bbox_data["x_min"], bbox_data["y_min"], bbox_data["z_max"]]
        rdf = [bbox_data["x_max"], bbox_data["y_min"], bbox_data["z_max"]]
        luf = [bbox_data["x_min"], bbox_data["y_max"], bbox_data["z_max"]]
        ruf = [bbox_data["x_max"], bbox_data["y_max"], bbox_data["z_max"]]
        tfs = bbox_data["transform"]
        corners = np.stack((ldb, rdb, lub, rub, ldf, rdf, luf, ruf), 0)
        # Homogenize the coordinate
        corners_homo = np.pad(corners, ((0, 0), (0, 1), (0, 0)), constant_values=1.0)
        # local object frame to world frame
        corners_world = np.einsum("jki,ikl->ijl", corners_homo, tfs)

        return corners_world

    def process_bbox_corners(self, bbox_3d_corners):
        """Process the bbox3d data directly from annotator
        Returns:
            corners_min, corners_max : np.ndarray((N,2)), np.ndarray((N,2))
            corners_min is the [(x_min, y_min), ...] that defines all the detected bboxes in the image frame
            corners_max is the [(x_max, y_max), ...] that defines all the detected bboxes in the image frame
        """
        N = bbox_3d_corners.shape[0]
        # world frame to camera frame
        corners_local = np.einsum('ijk,lk->ijl', bbox_3d_corners, self.scan_data['viewTransform'])
        # Rotate axis such that y axis pointing forward for sonar data plotting
        corners_local = np.einsum('ijk,lk->ijl', corners_local, np.array([[1,0,0,0],
                                                                          [0,0,-1,0],
                                                                          [0,1,0,0],
                                                                          [0,0,0,1]]))
        # collapse to 2d sonar grid
        corners_min = np.zeros(shape=(N, 2), dtype=np.int32)
        corners_max = np.zeros(shape=(N, 2), dtype=np.int32)
        corners_local = corners_local[..., :3] # shape: [N,8,3] 
        r = np.linalg.norm(corners_local, axis=-1) # shape: [N, 8]
        azi = np.arctan2(corners_local[..., 1], corners_local[..., 0]) # shape: [N, 8]
        x_pix = np.int32((r - self.min_range) / self.range_res) # shape: [N, 8]
        # angular_res is in degrees on this sensor (upstream stores radians).
        y_pix = np.int32((azi - self.min_azi) / np.deg2rad(self.angular_res)) # shape: [N, 8]
        x_pix = np.clip(x_pix, 0, self.r.shape[0]-1)
        y_pix = np.clip(y_pix, 0, self.r.shape[1]-1)
        corners_min[..., 0] = np.min(x_pix, axis=1)
        corners_min[..., 1] = np.min(y_pix, axis=1)
        corners_max[..., 0] = np.max(x_pix, axis=1)
        corners_max[..., 1] = np.max(y_pix, axis=1)

        return corners_min, corners_max

    def get_priviledged_bbox(self):
        """Priviledged bbox means the bbox computed from the scene, not from the sensor view. It's called priviledged because observer 
        itself won't be able to access this information. For regular bbox, simply use cv2.findCentroid() on semantics information. 

        Returns:
            bbox_id: bbox's id
            bbox_min: is the [(x_min, y_min), ...] that defines all the detected bboxes in the image frame
            bbox_max: is the [(x_max, y_max), ...] that defines all the detected bboxes in the image frame
        """
        if (self._privileged_bbox and getattr(self, "bbox_annot", None) is not None
                and 'viewTransform' in self.scan_data):
            bbox = self.bbox_annot.get_data()
            self.scan_data['bbox'] = bbox['data']
            self.scan_data['bbox_ids'] = bbox['info']['bboxIds']
            # Compute the privileged bbox
            bbox_corners = self.get_bbox_3d_corners(self.scan_data['bbox'])
            bbox_min, bbox_max = self.process_bbox_corners(bbox_corners)
            return self.scan_data['bbox_ids'], bbox_min, bbox_max
        else:
            print(f'[{self._name}] Initialize with priviledged_bbox to true and run a scan before calling this.')
            return

    get_privileged_bbox = get_priviledged_bbox

    @staticmethod
    def draw_bbox_on_image(bboxes_min : np.ndarray, 
                  bboxes_max : np.ndarray,
                  image : wp.array, 
                  colormap : str = 'turbo'):
        """
        Args:
            bbox_min: is the [(x_min, y_min), ...] that defines all the detected bboxes in the image frame
            bbox_max: is the [(x_max, y_max), ...] that defines all the detected bboxes in the image frame          
            image: GPU array containing the image (RGBA) wp.array(dtype=wp.uint8)
            colormap: (str) following plt's cmap standard
        """
        import matplotlib.pyplot as plt

        num_bboxes = bboxes_min.shape[0]
        cmap = plt.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, num_bboxes)) * 255  # Get n colors from the colormap
        wp_min = wp.array(data=bboxes_min, ndim=2, dtype=wp.int32, device=image.device)
        wp_max = wp.array(data=bboxes_max, ndim=2, dtype=wp.int32, device=image.device)
        wp_colors = wp.array(data=colors.astype(np.uint8), ndim=2, dtype=wp.uint8, device=image.device)

        for i in range(num_bboxes):  
            wp.launch(
                kernel = draw_bbox,
                dim=(bboxes_max[i,0]-bboxes_min[i,0], 
                    bboxes_max[i,1]-bboxes_min[i,1]),
                inputs=[
                    i,
                    wp_min,
                    wp_max,
                    wp_colors,
                    image
                ],
                device=image.device
            )


    def make_sonar_viewport(self):
        """Show a metric fan without stretching its range/bearing geometry."""
        self.wrapped_ui_elements = []
        self._ensure_fan_buffers()
        self._sonar_provider = ui.ByteImageProvider()
        segmentation = getattr(self, "_segmentation", False)
        image_h, image_w = self._fan_image.shape[:2]
        # Fit a wide LF fan within a desktop window while retaining its aspect.
        display_scale = min(1.0, 1000.0 / image_w)
        view_w, view_h = int(image_w * display_scale), int(image_h * display_scale)
        panel_w = view_w * (2 if segmentation else 1)
        self._window = ui.Window(self._name, width=panel_w + 20, height=view_h + 100, visible=True)
        with self._window.frame:
            with ui.VStack(spacing=4):
                ui.Label(f"{self.acoustic_frequency / 1e6:g} MHz  |  "
                         f"{self.hori_fov:g}\u00b0 x {self.vert_fov:g}\u00b0  |  "
                         f"{self.min_range:g}\u2013{self.max_range:g} m", height=24)
                with ui.HStack(height=view_h):
                    sonar_image_provider = ui.ImageWithProvider(
                        self._sonar_provider, width=view_w,
                        style={"fill_policy": ui.FillPolicy.PRESERVE_ASPECT_FIT})
                    if segmentation:
                        self._sonar_segmentation_provider = ui.ByteImageProvider()
                        segmentation_image_provider = ui.ImageWithProvider(
                            self._sonar_segmentation_provider, width=view_w,
                            style={"fill_policy": ui.FillPolicy.PRESERVE_ASPECT_FIT})
                ui.Label("PORT / LEFT                         STARBOARD / RIGHT", height=22,
                         alignment=ui.Alignment.CENTER)
                ui.Label(f"Range rings: {self.max_range/4:g} m  |  sensor at bottom centre",
                         height=22, alignment=ui.Alignment.CENTER)
        
        self.wrapped_ui_elements.append(sonar_image_provider)
        self.wrapped_ui_elements.append(self._sonar_provider)
        if segmentation:
            self.wrapped_ui_elements.append(segmentation_image_provider)
            self.wrapped_ui_elements.append(self._sonar_segmentation_provider)
        self.wrapped_ui_elements.append(self._window)

    def get_range(self) -> list[float]:
        """Get the configured operating range of the sonar.
    
        Returns:
            list[float]: [min_range, max_range] in meters
        """
        return [self.min_range, self.max_range]
    
    def get_fov(self) -> list[float]:
        """Get the configured field of view angles.
    
        Returns:
            list[float]: [horizontal_fov, vertical_fov] in degrees
        """
        return [self.hori_fov, self.vert_fov]
    

    
    def close(self):
        """Clean up resources by detaching annotators and clearing caches.
    
        Note:
            - Required for proper shutdown when done using the sensor
            - Also closes viewport window if one was created
        """
        # Stop the async worker first so it isn't mid-kernel when the annotators /
        # render product it reads through scan_data get torn down below.
        self.stop_async()
        # Flush any scheduled-but-unwritten frames before teardown -- the dispatch
        # backend writes asynchronously and dropped trailing frames at shutdown.
        if getattr(self, "writing", False) and getattr(self, "backend", None) is not None:
            try:
                self.backend.wait_until_done()
            except Exception as exc:  # noqa: BLE001
                print(f'[{self._name}] write-backend flush warning: {exc}')
        # If sonar_initialize() never ran (or raised mid-way), cameraParams_annot
        # won't exist; guard so close() on a half-initialized sensor doesn't raise
        # an AttributeError that masks the rest of the scenario teardown.
        if getattr(self, "cameraParams_annot", None) is None:
            if getattr(self, "_viewport", False):
                self.ui_destroy()
            return
        # Same hydra-texture gating as sonar_initialize(): detaching also mutates
        # the SDGPipeline graph, so disable updates first to avoid a teardown-time
        # variant of the partial-graph SIGSEGV. (UNTESTED — see sonar_initialize.)
        _rp = getattr(self, "_render_product", None)
        if _rp is not None:
            _rp.hydra_texture.set_updates_enabled(False)
        # Remove the primitive AOVs (attached via the base Camera helpers) + the
        # manual CameraParams annotator.
        try:
            self.remove_distance_to_image_plane_from_frame()
            self.remove_normals_from_frame()
            self.remove_semantic_segmentation_from_frame()
        except Exception as exc:  # noqa: BLE001
            print(f'[{self._name}] annotator removal warning: {exc}')
        self.cameraParams_annot.detach(self._render_product_path)
        if getattr(self, "bbox_annot", None) is not None:
            try:
                self.bbox_annot.detach(self._render_product_path)
                rep.AnnotatorCache.clear(self.bbox_annot)
            except Exception as exc:  # noqa: BLE001
                print(f'[{self._name}] bbox annotator removal warning: {exc}')
            self.bbox_annot = None
        if _rp is not None:
            _rp.hydra_texture.set_updates_enabled(True)

        rep.AnnotatorCache.clear(self.cameraParams_annot)

        print(f'[{self._name}] Annotator detached. AnnotatorCache cleaned.')

        if self._viewport:
            self.ui_destroy()


    def ui_destroy(self):
        """Explicitly destroy viewport UI elements.
    
        Note:
            - Called automatically by close()
            - Only needed if manually managing UI lifecycle
        """
        for elem in self.wrapped_ui_elements:
            elem.destroy()
