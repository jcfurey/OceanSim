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
                 min_range: float = 0.2, # m
                 max_range: float = 3.0, # m
                 range_res: float = 0.008, # deg
                 hori_fov: float = 130.0, # deg
                 vert_fov: float = 20.0, # deg
                 angular_res: float = 0.5, # deg
                 hori_res: int = 3000, # isaac camera render product only accepts square pixel,
                                      # for now vertical res is automatically set with ratio of hori_fov vs.vert_fov
                 gpu_point_filter: bool = False, # on-device point compaction; skips the
                                      # device->host->device round-trip. Self-heals to the
                                      # numpy path if the AOV outputs aren't on-device Warp arrays.
                 async_compute: bool = False, # run the post-scan kernels + sonar_map readback on
                                      # a worker thread so the sonar doesn't block the sim loop / odom.
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
            min_range (float, optional): Minimum detection range in meters. Defaults to 0.2.
            max_range (float, optional): Maximum detection range in meters. Defaults to 3.0.
            range_res (float, optional): Range resolution in meters. Defaults to 0.008.
            hori_fov (float, optional): Horizontal field of view in degrees. Defaults to 130.0.
            vert_fov (float, optional): Vertical field of view in degrees. Defaults to 20.0.
            angular_res (float, optional): Angular resolution in degrees. Defaults to 0.5.
            hori_res (int, optional): Horizontal pixel resolution. Defaults to 3000.
    
        Note:
            - Vertical resolution is automatically calculated to maintain aspect ratio
            - Uses Warp for GPU-accelerated sonar image generation
            - Creates polar coordinate meshgrid for sonar returns processing
        """


        self._name = name
        # Raw parameters from Oculus M370s\MT370s\MD370s
        self.max_range = max_range # m (max is 200 m in datasheet )
        self.min_range = min_range # m (min is 0.2 m in datasheet)
        self.range_res = range_res # m (datasheet is 0.008 m)
        self.hori_fov = hori_fov # degree (hori_fov is 130 degrees in datasheet)
        self.vert_fov = vert_fov # degree (vert_fov is 20 degrees in datasheet)
        self.angular_res = angular_res # degree (datasheet is 2 deg)
        self.hori_res= hori_res
        # Requested gpu_point_filter; applied in sonar_initialize (which resets the
        # live flag) so it survives (re)initialization.
        self._gpu_point_filter_init = gpu_point_filter
        self._async_compute_init = async_compute
        # Acoustic carrier frequency of the modelled sonar (Hz). The Oculus M370s
        # is a 375 kHz single-frequency unit (Blueprint Subsea datasheet). This is
        # the value reported in ProjectedSonarImage.ping_info.frequency -- kept
        # SEPARATE from the inherited Camera `frequency` attribute, which is the
        # render frame rate, not the acoustic carrier.
        self.acoustic_frequency = 375e3  # Hz (Oculus M370s)

        # self.beam_separation = 0.5 # degree (Not USED FOR NOW)!!
        # self.num_beams = 256 # (max number of beams) (NOT USED FOR NOW)!!
        # self.update_rate = 40 # Hz (max update rate) (NOT USED FOR NOW)!!


        # Generate sonar map's r and z meshgrid
        self.min_azi = np.deg2rad(90-self.hori_fov/2)
        r, azi = np.meshgrid(np.arange(self.min_range,self.max_range,self.range_res),
                                       np.arange(np.deg2rad(90-self.hori_fov/2), np.deg2rad(90+self.hori_fov/2), np.deg2rad(self.angular_res)),
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

        self.AR = self.hori_fov / self.vert_fov
        self.vert_res = int(self.hori_res / self.AR)
        # By doing this, I am assuming the vertical beam separation
        # is the same as the beam horizontal separation. 
        # This is bacause replicator raytracing is specified as resolutions
        # while non-squre pixel is not supported in Isaac sim. See details below.
        
        super().__init__(prim_path=prim_path, 
                         name=name, 
                         frequency=frequency,
                         dt=dt, 
                         resolution=[self.hori_res, self.vert_res],
                         position=position, 
                         orientation=orientation, 
                         translation=translation, 
                         render_product_path=render_product_path)

        self.set_clipping_range(
            near_distance=self.min_range,
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

    def sonar_initialize(self, output_dir : str = None, viewport: bool = True, include_unlabelled = False, if_array_copy: bool = True):
        """Initialize sonar data processing pipeline and annotators.
    
        Args:
            output_dir (str, optional): Directory to save sonar data. Defaults to None.
                                        If set to None, sonar will not write data.
            viewport (bool, optional): Enable viewport visualization. Defaults to True.
                                        Set to False for Sonar running without visualization.
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
        self._device = str(wp.get_preferred_device())
        self.scan_data = {}
        self.id = 0
        self._scan_logged = False  # one-shot shape diagnostic in scan()

        # Optional on-device point compaction (compact_in_range kernel). Default
        # OFF: the kernel is unit tested against the numpy reference
        # (tests/test_imaging_sonar_kernels.py), but whether get_pointcloud /
        # the AOV annotators actually return Warp arrays resident on
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
        self.async_compute = self._async_compute_init
        self._async_busy = False
        self._async_result = None
        self._async_lock = threading.Lock()
        self._async_scan_evt = threading.Event()
        self._async_stop = False
        self._async_params = {}
        self._async_thread = None
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
        self._wp_pcl_local = None
        self._wp_pcl_spher = None
        # Cached reflectivity lookup (semantic id -> reflectivity) keyed on the
        # (idToLabels, query_prop) it was built from, so the GPU upload is only
        # redone when the labelled-mesh set in the FOV changes.
        self._refl_cache_key = None
        self._refl_cache_arr = None
        # Fixed-shape normalization-max buffers, preallocated and re-zeroed each
        # frame (global max -> shape (1,), per-range max -> shape (n_range,)).
        self._max_all = wp.zeros(shape=(1,), dtype=wp.float32, device=self._device)
        self._max_range = wp.zeros(shape=(self.r.shape[0],), dtype=wp.float32, device=self._device)

        # Initialize the camera (creates the render product) HERE -- post
        # world.reset() (deferred from __init__ for the Isaac Sim 6.0.1 port; see
        # __init__). UW_Camera does the same via its scenario-driven initialize().
        # Then set the horizontal aperture (needs initialize() first, per the
        # upstream aperture-ordering bug) so the sonar FOV geometry is correct.
        self.initialize()
        self.focal_length = self.get_focal_length()
        horizontal_aper = 2 * self.focal_length * np.tan(np.deg2rad(self.hori_fov) / 2)
        self.set_horizontal_aperture(horizontal_aper)

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
        sem_data = self._custom_annotators["semantic_segmentation"].get_data()
        id_to_labels = self._annot_get(sem_data, ('info', 'idToLabels'), 'semantic_segmentation')

        # No labels yet (CUDA warmup) or nothing in the FOV -> skip this frame.
        if len(id_to_labels) == 0:
            return False

        # Optional on-device fast path: compact the in-range points with the
        # compact_in_range kernel so the per-pixel depth/pcl/normals/semantics
        # never round-trip device->host->device. Returns True/False on success,
        # or None if it cannot run on-device (in which case it disables itself
        # and we drop through to the numpy path). See sonar_initialize().
        if self.gpu_point_filter:
            result = self._scan_gpu_compact(id_to_labels)
            if result is not None:
                return result
            self.gpu_point_filter = False
            print(f"[{self._name}] gpu_point_filter unavailable (annotator outputs "
                  f"not Warp arrays on '{self._device}'); using numpy scan path.",
                  flush=True)

        return self._scan_numpy(sem_data, id_to_labels)

    def _scan_gpu_compact(self, id_to_labels):
        """On-device point selection via the compact_in_range kernel.

        Keeps the depth / point-cloud / normals / semantics AOVs on
        ``self._device`` and appends the in-range, finite points into reusable
        output buffers with an atomic counter -- the GPU equivalent of
        ``sonar_scan_math.select_in_range_points`` (and proven equal to it in
        tests/test_imaging_sonar_kernels.py).

        Returns:
            True  - in-range points stored in scan_data.
            False - valid frame but no in-range points (skip, same as numpy path).
            None  - the AOV outputs are not on-device Warp arrays (or something
                    threw); caller should fall back to the numpy path.
        """
        try:
            depth = self._custom_annotators["distance_to_image_plane"].get_data(device=self._device)
            pcl = self.get_pointcloud(device=self._device, world_frame=True)
            normals = self._custom_annotators["normals"].get_data(device=self._device)
            sem_dict = self._custom_annotators["semantic_segmentation"].get_data(device=self._device)
            sem = self._annot_get(sem_dict, ('data',), 'semantic_segmentation')

            # The fast path only engages when every AOV is genuinely a Warp array
            # resident on self._device with the dtype the kernel expects. Any
            # mismatch -> None -> numpy fallback (no silent host round-trip).
            depth = self._require_warp(depth, wp.float32)
            pcl = self._require_warp(pcl, wp.float32)
            normals = self._require_warp(normals, wp.float32)
            sem = self._require_warp(sem, wp.uint32)
            if depth is None or pcl is None or normals is None or sem is None:
                return None

            n_px = depth.size
            # Warp's reshape requires C-contiguity, but the annotator AOVs can come
            # back strided/non-contiguous ("Reshaping non-contiguous arrays is
            # unsupported"). Make a contiguous device-resident copy first -- a cheap
            # GPU->GPU copy that still avoids the device->host->device round-trip.
            depth_f = depth.contiguous().reshape((-1,))          # (N,)
            pcl_f = pcl.contiguous().reshape((-1, 3))            # (N,3)
            nm_c = normals.contiguous()
            nm_f = nm_c.reshape((-1, nm_c.shape[-1]))[:, :3]     # (N,3) view
            sem_f = sem.contiguous().reshape((-1,))              # (N,)
            if (pcl_f.shape[0] != n_px or nm_f.shape[0] != n_px
                    or sem_f.shape[0] != n_px):
                return None

            # Reusable, device-resident output buffers sized to the full pixel
            # count (the worst case all-in-range). Allocated once, kept across
            # frames; only the counter is re-zeroed each scan.
            if self._gpu_out_pcl is None or self._gpu_out_pcl.shape[0] != n_px:
                self._gpu_out_pcl = wp.zeros((n_px, 3), dtype=wp.float32, device=self._device)
                self._gpu_out_normals = wp.zeros((n_px, 3), dtype=wp.float32, device=self._device)
                self._gpu_out_sem = wp.zeros(n_px, dtype=wp.uint32, device=self._device)
                self._gpu_counter = wp.zeros(1, dtype=wp.int32, device=self._device)
            self._gpu_counter.zero_()

            wp.launch(kernel=compact_in_range,
                      dim=n_px,
                      inputs=[depth_f, pcl_f, nm_f, sem_f,
                              wp.float32(self.min_range), wp.float32(self.max_range),
                              self._gpu_counter, self._gpu_out_pcl,
                              self._gpu_out_normals, self._gpu_out_sem],
                      device=self._device)
            # counter.numpy() already does a blocking default-stream device->host
            # copy that orders this readback, so a global wp.synchronize() here
            # only adds an unnecessary all-device stall on the sim thread.
            n_valid = int(self._gpu_counter.numpy()[0])

            if not self._scan_logged:
                self._scan_logged = True
                print(f"[{self._name}] scan(gpu): N={n_px} valid={n_valid} "
                      f"idToLabels={id_to_labels}", flush=True)
            if n_valid == 0:
                return False

            cam_data = self.cameraParams_annot.get_data()
            view_tf = self._annot_get(cam_data, ('cameraViewTransform',), 'CameraParams')
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
            self._wp_pcl_local = wp.empty(shape=(num_points,), dtype=wp.vec3, device=self._device)
            self._wp_pcl_spher = wp.empty(shape=(num_points,), dtype=wp.vec3, device=self._device)

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
        # Isaac Sim 6.0.1: reconstruct the point cloud from the depth AOV instead of
        # the (crashing) pointcloud annotator. Camera.get_pointcloud() falls back to
        # a perspective projection of distance_to_image_plane when no pointcloud
        # annotator is attached, returning world points row-major over (H, W) -- so
        # the per-pixel normals / semantics flatten the same way and stay aligned.
        depth = self._custom_annotators["distance_to_image_plane"].get_data(device=self._device)
        depth_np = np.squeeze(self._to_numpy(depth))
        if depth_np.ndim != 2 or depth_np.size == 0:
            return False
        pcl_np = self._to_numpy(self.get_pointcloud(device=self._device, world_frame=True))
        if pcl_np.size == 0:
            return False

        normals_img = self._to_numpy(self._custom_annotators["normals"].get_data(device=self._device))
        sem_img = np.squeeze(self._to_numpy(self._annot_get(sem_data, ('data',), 'semantic_segmentation')))
        cam_data = self.cameraParams_annot.get_data()
        view_tf = self._annot_get(cam_data, ('cameraViewTransform',), 'CameraParams')

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
            depth_flat, pcl_np, normals_flat, sem_flat, self.min_range, self.max_range)
        n_valid = pcl_v.shape[0]
        if not getattr(self, "_scan_logged", False):
            self._scan_logged = True
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
                        normalizing_method: str = "range",
                        query_prop: str ='reflectivity', # Do not modify this if not developing the sensor.
                        attenuation: float = 0.1, # Control the attentuation along distance when computing attenuation
                        gau_noise_param: float = 0.2, # multiplicative noise coefficient 
                        ray_noise_param: float = 0.05, # additive noise parameter
                        intensity_offset: float = 0.0, # offset intensity after normalization
                        intensity_gain: float = 1.0, # scale intensity after normalization
                        central_peak: float = 2, # control the strength of the streak
                        central_std: float = 0.001, # control the spread of the streak
                        _skip_scan: bool = False, # internal: worker has already run scan() on
                                      # the main thread; run only the kernels on the live scan_data.
                        ):
        """Process raw scan data into a sonar image with configurable parameters.

        Args:
            binning_method (str): "sum" or "mean" for intensity accumulation
                                Remember to adjust your noise scale accordingly after changing this.
            normalizing_method (str): "all" (global max) or "range" (per-range max)
                                Remember to adjust your noise scale accordingly after changing this.
            query_prop (str): Material property to query (default 'reflectivity')
                            Don't modify this if not for development.
            attenuation (float): Distance attenuation coefficient (0-1)
            gau_noise_param (float): Gaussian noise multiplier
            ray_noise_param (float): Rayleigh noise scale factor
            intensity_offset (float): Post-normalization intensity offset
            intensity_gain (float): Post-normalization intensity multiplier
            central_peak (float): Central beam streak intensity
            central_std (float): Central beam streak width
    
        """



        if self.async_compute and not _skip_scan:
            # Main thread: scan (reads annotators) + hand off to the worker; the
            # heavy kernels + sonar_map readback run there. Returns immediately.
            self._submit_scan_async(dict(
                binning_method=binning_method, normalizing_method=normalizing_method,
                query_prop=query_prop, attenuation=attenuation,
                gau_noise_param=gau_noise_param, ray_noise_param=ray_noise_param,
                intensity_offset=intensity_offset, intensity_gain=intensity_gain,
                central_peak=central_peak, central_std=central_std))
            return

        if _skip_scan or self.scan():
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
                
        # Transform pointcloud from world cooridates to sonar local
        pcl_local = self._wp_pcl_local[:num_points]
        pcl_spher = self._wp_pcl_spher[:num_points]
        wp.launch(kernel=world2local,
                  dim=num_points,
                  inputs=[
                      viewTransform,
                      pcl
                  ],
                    outputs=[
                      pcl_local,
                      pcl_spher
                    ]
                )
        
        # Collapse three dimensional intensity data to 2D
        # Simply sum intensity return and compute number of return that falls into the same bin
        self.bin_sum.zero_()
        self.bin_count.zero_()
        self.binned_intensity.zero_()

        
        wp.launch(kernel=bin_intensity,
                  dim=num_points,
                  inputs=[
                      pcl_spher,
                      intensity,
                      self.min_range,
                      self.min_azi,
                      self.range_res,
                      wp.radians(self.angular_res),
                  ],
                  outputs=[
                      self.bin_sum,
                      self.bin_count
                  ]
                  )
        
        # Process intensity data by either sum as it is or averaging
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
        
        if binning_method == "sum":
            self.binned_intensity = self.bin_sum


        # gau_noise / range_dependent_ray_noise are fully overwritten every frame
        # by normal_2d / range_dependent_rayleigh_2d (which write every cell), so
        # zeroing them first is redundant. sonar_map.zero_() is kept: an
        # unrecognized normalizing_method runs no map kernel, so it must start clean.
        self.sonar_map.zero_()

        # Calculate multiplicative gaussian noise
        
        wp.launch(
            kernel=normal_2d,
            dim=self.bin_sum.shape,
            inputs=[
                self.id,   # use frame num for RNG seed increment
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
                self.id,   # use frame num for RNG seed increment
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
                kernel=all_max,
                inputs=[
                    self.binned_intensity,
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
                      self.binned_intensity,
                      maximum,
                      self.gau_noise,
                      self.range_dependent_ray_noise,
                      intensity_offset,
                      intensity_gain
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
                kernel=range_max,
                inputs=[
                    self.binned_intensity,
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
                      self.binned_intensity,
                      maximum,
                      self.gau_noise,
                      self.range_dependent_ray_noise,
                      intensity_offset,
                      intensity_gain
                  ],
                  outputs=[
                      self.sonar_map
                  ]
                  )
        
        
        # Write data to the dir
        if self.writing:
            # self.backend.schedule(write_np, f"intensity_{self.id}.npy", data=intensity)
            # self.backend.schedule(write_np, f'pcl_local_{self.id}.npy', data=pcl_local)
            self.backend.schedule(write_np, f'sonar_data_{self.id}.npy', data=self.sonar_map)
            print(f"[{self._name}] [{self.id}] Writing sonar data to {self.backend.output_dir}")
        
        if self._viewport and not self.async_compute:
            # Skip in async mode: this pushes to the Isaac UI byte provider, which
            # must not be touched from the worker thread. ROS consumers read the
            # published sonar_map, not this in-Isaac viewport texture.
            self._sonar_provider.set_bytes_data_from_gpu(self.make_sonar_image().ptr,
                                                    [self.sonar_map.shape[1], self.sonar_map.shape[0]])
            # self.backend.schedule(write_image, f'sonar_{self.id}.png', data = self.make_sonar_image())        
            
        self.id += 1
    

    def _submit_scan_async(self, params):
        """Main-thread half of async sonar: scan() (reads the annotators) then hand
        the device-resident scan_data to the worker. Skips entirely while the worker
        is still processing the previous scan, so scan_data is never overwritten
        mid-process (which also self-throttles scans to the worker's throughput)."""
        if self._async_busy:
            return
        if not self.scan():
            return
        self._async_params = params
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
                with wp.ScopedDevice(self._device):
                    self.make_sonar_data(_skip_scan=True, **self._async_params)
                    grid = self.sonar_map.numpy()
                with self._async_lock:
                    self._async_result = grid
            except Exception as exc:  # noqa: BLE001
                print(f"[{self._name}] async sonar worker error ({exc!r})", flush=True)
            finally:
                self._async_busy = False

    def get_sonar_map_np(self):
        """Host-side (n_range, n_azimuth, 3) sonar_map for the publisher. Async mode
        returns the worker's latest readback (no GPU sync on the caller); sync mode
        reads the device sonar_map directly. None if nothing has been produced yet."""
        if getattr(self, 'async_compute', False):
            with self._async_lock:
                return self._async_result
        sm = getattr(self, 'sonar_map', None)
        if sm is None:
            return None
        return sm.numpy() if hasattr(sm, 'numpy') else np.asarray(sm)

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
        self._async_thread = None

    def make_sonar_image(self):
        """Convert processed sonar data to a viewable grayscale image.
    
        Returns:
            wp.array: GPU array containing the sonar image (RGBA format)
    
        Note:
            - Used internally for viewport display
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
    

    def make_sonar_viewport(self):
        """Create an interactive viewport window for real-time sonar visualization.
    
        Note:
            - Displays live sonar images when simulation is running
            - Includes range and azimuth tick marks
            - Window size is fixed at 800x800 pixels
        """
        self.wrapped_ui_elements = []

        range_tick_num = 10
        range_tick = np.round(np.linspace(self.min_range, self.max_range, range_tick_num), 2)

        azi_tick_num = 10
        azi_tick = np.round(np.linspace(90-self.hori_fov/2, 90+self.hori_fov/2, azi_tick_num))
        self._sonar_provider = ui.ByteImageProvider()
        self._window = ui.Window(self._name, width=800, height=800, visible=True)
        
        with self._window.frame:
            with ui.ZStack(height=720, width = 720):
                ui.Rectangle(style={"background_color": 0xFF000000})
                ui.Label('Run the scenario for image to be received',
                         style={'font_size': 55,'alignment': ui.Alignment.CENTER},
                         word_wrap=True)
                sonar_image_provider = ui.ImageWithProvider(self._sonar_provider, 
                                    style={"width": 720, 
                                        "height": 720, 
                                        "fill_policy" : ui.FillPolicy.STRETCH,
                                        'alignment': ui.Alignment.CENTER})
                
                # ui.Line(alignment=ui.Alignment.LEFT,
                #         style={'border_width': 2,
                #                 'color':ui.color.white })
                # with ui.VGrid(row_height = 720/(range_tick_num-1)):
                #     for i in range(range_tick_num-1):
                #         with ui.ZStack():
                #             ui.Rectangle(style={'border_color': ui.color.white, 'background_color': ui.color.transparent,'border_width': 0.05, 'margin': 0})
                #             ui.Label(str(range_tick[i]) + ' m',style={'font_size': 15,'alignment': ui.Alignment.LEFT, 'margin':2})
                # with ui.HGrid(column_width = 720/(azi_tick_num-1), direction=ui.Direction.RIGHT_TO_LEFT):
                #     for i in range(azi_tick_num-1):
                #         with ui.ZStack():
                #             ui.Rectangle(style={'border_color': ui.color.white, 'background_color': ui.color.transparent,'border_width': 0.05, 'margin': 0})
                #             ui.Label(str(azi_tick[i]) + "°",style={'font_size': 15,'alignment': ui.Alignment.RIGHT, 'margin':2})                           
                # ui.Label(str(range_tick[-1]) +" m", style={'font_size': 15, "alignment":ui.Alignment.LEFT_BOTTOM, 'margin':2})
        
        self.wrapped_ui_elements.append(sonar_image_provider)
        self.wrapped_ui_elements.append(self._sonar_provider)
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