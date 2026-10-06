import warp as wp
from typing import Any


@wp.struct
class sonarGrid:
    x_offset: float
    y_offset: float
    x_res: float
    y_res: float
    x_num: int
    y_num: int


# pcl_bin_idx sentinel for a point that falls outside the binning grid (see
# bin_process). Downstream per-point kernels skip it instead of indexing with it.
INVALID_BIN = wp.constant(wp.uint32(0xFFFFFFFF))


@wp.func
def cartesian_to_spherical(cart: wp.vec3) -> wp.vec3:
    r = wp.sqrt(cart[0]*cart[0] + cart[1]*cart[1] + cart[2]*cart[2])
    # Guard the inclination: at r == 0 (a point at the sensor origin) cart[2]/r
    # is inf/nan, and even for ordinary points floating-point rounding can push
    # cart[2]/r just past +/-1, where acos() returns NaN. Use a floored r and
    # clamp the argument into acos's valid domain so the elevation is always
    # finite. (atan2 is already safe at the origin.)
    cos_incl = cart[2] / wp.max(r, wp.float32(1e-8))
    return wp.vec3(r,
                wp.atan2(cart[1], cart[0]),
                wp.acos(wp.clamp(cos_incl, wp.float32(-1.0), wp.float32(1.0)))
                )
                                    

@wp.kernel
def compute_intensity(pcl: wp.array(ndim=2, dtype=wp.float32),
                    normals: wp.array(ndim=2, dtype=wp.float32),
                    viewTransform: wp.mat44,
                    semantics: wp.array(ndim=1, dtype=wp.uint32),
                    indexToRefl: wp.array(dtype=wp.float32),
                    attenuation: float,
                    intensity: wp.array(dtype=wp.float32)
                    ):
    tid = wp.tid()
    pcl_vec = wp.vec3(pcl[tid,0], pcl[tid,1], pcl[tid,2])
    normal_vec = wp.vec3(normals[tid,0], normals[tid,1],normals[tid,2])
    R = wp.mat33(viewTransform[0,0], viewTransform[0,1], viewTransform[0,2],
                 viewTransform[1,0], viewTransform[1,1], viewTransform[1,2],
                 viewTransform[2,0], viewTransform[2,1], viewTransform[2,2])
    T = wp.vec3(viewTransform[0,3], viewTransform[1,3], viewTransform[2,3])
    sensor_loc = - (wp.transpose(R) @ T)
    incidence = pcl_vec - sensor_loc
    # Will use warp.math.norm_l2() in future release
    dist = wp.length(incidence)
    # Guard the normalization: if the point coincides with the sensor (dist == 0)
    # wp.normalize divides by zero -> NaN, which then poisons the whole bin sum
    # via the atomic_add in bin_intensity. Reuse the already-computed dist instead
    # of recomputing the subtraction + norm inside wp.normalize.
    unit_directs = wp.vec3(0.0, 0.0, 0.0)
    if dist > wp.float32(1e-8):
        unit_directs = incidence / dist
    # |cos|: a surface the render shows returns energy whichever way its normal
    # was authored. A back-facing (double-sided / flipped) normal gave a
    # NEGATIVE contribution that cancelled other returns in the same bin.
    cos_theta = wp.abs(wp.dot(-unit_directs, normal_vec))
    reflectivity = indexToRefl[semantics[tid]]
    intensity[tid] = reflectivity * cos_theta * wp.exp(-attenuation * dist)

@wp.kernel
def world2local(viewTransform: wp.mat44,
                pcl_world: wp.array(ndim=2, dtype=wp.float32),
                pcl_local_spher: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    pcl_world_homogeneous = wp.vec4(pcl_world[tid,0],
                          pcl_world[tid,1],
                          pcl_world[tid,2],
                          wp.float32(1.0)
                          )
    pcl_local_homogeneous = viewTransform @ pcl_world_homogeneous
    # Rotate axis such that y axis pointing forward for sonar data plotting
    pcl_local = wp.vec3(pcl_local_homogeneous[0], -pcl_local_homogeneous[2], pcl_local_homogeneous[1])
    pcl_local_spher[tid] = cartesian_to_spherical(pcl_local)


@wp.kernel
def bin_intensity(pcl: wp.array(dtype=wp.vec3),
                  intensity: wp.array(dtype=wp.float32),
                  x_offset: wp.float32,
                  y_offset: wp.float32,
                  x_res: wp.float32,
                  y_res: wp.float32,
                  bin_sum: wp.array(ndim=2, dtype=wp.float32),
                  bin_count: wp.array(ndim=2, dtype=wp.int32)
                  ):
    tid = wp.tid()

    # Get the range, azimuth, and intensity of the point
    x = pcl[tid][0]
    y = pcl[tid][1]

    # Calculate the bin indices for range and azimuth. FLOOR, not a bare int
    # cast: the cast truncates toward zero, so a point just below the grid
    # origin -- (coord - offset) in (-res, 0), e.g. an azimuth a fraction of a
    # beam outside the FOV or a range a hair under min_range -- truncated to 0,
    # passed the >= 0 bounds check below, and was folded into bin 0 as a
    # spurious bright return. floor() sends it to -1, which the check drops.
    x_bin_idx = wp.int32(wp.floor((x - x_offset) / x_res))
    y_bin_idx = wp.int32(wp.floor((y - y_offset) / y_res))
    # Drop points that fall outside the binning grid. Points at the camera far
    # clip (== max_range) or at the FOV edges land one index past the grid, and
    # Warp does no bounds checking in release mode, so an unchecked atomic_add
    # here is an out-of-bounds write (memory corruption / crash).
    if (x_bin_idx >= 0 and x_bin_idx < bin_sum.shape[0]
            and y_bin_idx >= 0 and y_bin_idx < bin_sum.shape[1]):
        wp.atomic_add(bin_sum, x_bin_idx, y_bin_idx, intensity[tid])
        wp.atomic_add(bin_count, x_bin_idx, y_bin_idx, 1)


@wp.func
def pixel_solid_angle_weight(spher: wp.vec3, beam_gain: wp.array(dtype=wp.float32), beam: wp.int32):
    """cos^3 of the ray's angle off the optical axis (local +y) -- a pinhole
    pixel's relative solid angle -- times the beam's normalising gain
    (sonar_scan_math.beam_solid_angle_gain)."""
    c = wp.sin(spher[2]) * wp.sin(spher[1])
    return c * c * c * beam_gain[beam]


@wp.kernel
def bin_intensity_sa(pcl: wp.array(dtype=wp.vec3),
                     intensity: wp.array(dtype=wp.float32),
                     x_offset: wp.float32,
                     y_offset: wp.float32,
                     x_res: wp.float32,
                     y_res: wp.float32,
                     half_vfov: wp.float32,
                     beam_gain: wp.array(dtype=wp.float32),
                     bin_sum: wp.array(ndim=2, dtype=wp.float32),
                     bin_count: wp.array(ndim=2, dtype=wp.int32)
                     ):
    """bin_intensity with the sonar's real beam geometry:
    - drops points outside elevation +-half_vfov (the render camera is taller
      than the beam -- see sonar_scan_math.sonar_render_height);
    - weights each point by its pixel's solid angle, normalised per beam, so a
      bin integrates intensity over solid angle instead of counting render
      pixels (which grew toward the fan edges and striped beam to beam)."""
    tid = wp.tid()
    p = pcl[tid]
    elev = wp.HALF_PI - p[2]
    if wp.abs(elev) > half_vfov:
        return
    x_bin_idx = wp.int32(wp.floor((p[0] - x_offset) / x_res))
    y_bin_idx = wp.int32(wp.floor((p[1] - y_offset) / y_res))
    if (x_bin_idx >= 0 and x_bin_idx < bin_sum.shape[0]
            and y_bin_idx >= 0 and y_bin_idx < bin_sum.shape[1]):
        w = pixel_solid_angle_weight(p, beam_gain, y_bin_idx)
        wp.atomic_add(bin_sum, x_bin_idx, y_bin_idx, intensity[tid] * w)
        wp.atomic_add(bin_count, x_bin_idx, y_bin_idx, 1)


@wp.kernel
def bin_process(pcl: wp.array(dtype=wp.vec3),
                  intensity: wp.array(dtype=wp.float32),
                  semantics: wp.array(dtype=wp.uint32),
                  sonar_grid: sonarGrid,
                  half_vfov: wp.float32,
                  beam_gain: wp.array(dtype=wp.float32),
                  bin_sum: wp.array(ndim=2, dtype=wp.float32),
                  bin_count: wp.array(ndim=2, dtype=wp.int32),
                  pcl_bin_idx: wp.array(dtype=wp.vec2ui),
                  bin_min_zenith: wp.array(ndim=2, dtype=wp.float32)
                  ):
    """bin_intensity plus the per-point bookkeeping the segmentation kernels
    need: the bin each point landed in (pcl_bin_idx) and the minimum zenith per
    bin (bin_min_zenith), so a bin's label comes from its top-most return.

    Same floor + bounds check, elevation cut and solid-angle weighting as
    bin_intensity_sa. Upstream cast straight to uint32, which wrapped a
    slightly-out-of-grid point to a huge index and wrote out of bounds; here
    such points (and points outside +-half_vfov) get the INVALID_BIN sentinel
    and are dropped by every downstream kernel."""
    tid = wp.tid()

    # Get the range, azimuth of the point
    x = pcl[tid][0]
    y = pcl[tid][1]
    elev = wp.HALF_PI - pcl[tid][2]
    x_bin_idx = wp.int32(wp.floor((x - sonar_grid.x_offset) / sonar_grid.x_res))
    y_bin_idx = wp.int32(wp.floor((y - sonar_grid.y_offset) / sonar_grid.y_res))
    if (wp.abs(elev) <= half_vfov
            and x_bin_idx >= 0 and x_bin_idx < bin_sum.shape[0]
            and y_bin_idx >= 0 and y_bin_idx < bin_sum.shape[1]):
        w = pixel_solid_angle_weight(pcl[tid], beam_gain, y_bin_idx)
        wp.atomic_add(bin_sum, x_bin_idx, y_bin_idx, intensity[tid] * w)
        wp.atomic_add(bin_count, x_bin_idx, y_bin_idx, 1)
        # Store the bin idx that corresponding to this pcl
        pcl_bin_idx[tid] = wp.vec2ui(wp.uint32(x_bin_idx), wp.uint32(y_bin_idx))
        # Store the minimum zenith value recorded for all the pcl that falls
        # into this bin. (Upstream gated this on `semantics[tid] != 0 or 1`,
        # which is always true, so every in-grid point counts.)
        wp.atomic_min(bin_min_zenith, x_bin_idx, y_bin_idx, pcl[tid][2])
    else:
        pcl_bin_idx[tid] = wp.vec2ui(INVALID_BIN, INVALID_BIN)


@wp.kernel
def bin_semantics_process(pcl: wp.array(dtype=wp.vec3),
                          semantics: wp.array(dtype=wp.uint32),
                          pcl_bin_idx: wp.array(dtype=wp.vec2ui),
                          bin_min_zenith: wp.array(ndim=2, dtype=wp.float32),
                          bin_semantics: wp.array(ndim=2, dtype=wp.uint32)
                          ):
    '''
    This kernel is DEPRECATED, use bin_segmentation_process instead. \n
    bin_semantics_process only process semantics information while bin_segmentation_process process both semantics and instances
    '''
    tid = wp.tid()

    # Get the index of the bin in which this pcl falls in
    x_bin_idx = pcl_bin_idx[tid][0]
    y_bin_idx = pcl_bin_idx[tid][1]
    if x_bin_idx == INVALID_BIN:
        return
    # Get the zenith of this pcl
    z = pcl[tid][2]
    # This ensures the semantics of this cell only belongs to the pcl semantics with the smallest zenith value
    if (z <= bin_min_zenith[x_bin_idx, y_bin_idx]):
        bin_semantics[x_bin_idx, y_bin_idx] = semantics[tid]


@wp.kernel
def bin_segmentation_process(pcl: wp.array(dtype=wp.vec3),
                             semantics: wp.array(dtype=wp.uint32),
                             instances: wp.array(dtype=wp.uint32),
                             pcl_bin_idx: wp.array(dtype=wp.vec2ui),
                             bin_min_zenith: wp.array(ndim=2, dtype=wp.float32),
                             bin_semantics: wp.array(ndim=2, dtype=wp.uint8),
                             bin_instances: wp.array(ndim=2, dtype=wp.uint8)
                             ):
    tid = wp.tid()

    # Get the index of the bin in which this pcl falls in
    x_bin_idx = pcl_bin_idx[tid][0]
    y_bin_idx = pcl_bin_idx[tid][1]
    if x_bin_idx == INVALID_BIN:
        return
    # Get the zenith of this pcl
    z = pcl[tid][2]
    # This ensures the semantics of this cell only belongs to the pcl semantics with the smallest zenith value
    if (z <= bin_min_zenith[x_bin_idx, y_bin_idx]):
        bin_semantics[x_bin_idx, y_bin_idx] = wp.uint8(semantics[tid])
        bin_instances[x_bin_idx, y_bin_idx] = wp.uint8(instances[tid])


@wp.kernel
def draw_bbox(n : int,
              aligned_bbox_min: wp.array(ndim=2, dtype=wp.int32),
              aligned_bbox_max: wp.array(ndim=2, dtype=wp.int32),
              bbox_colors: wp.array(ndim=2, dtype=wp.uint8),
              image: wp.array(ndim=3, dtype=wp.uint8),
              ):
    # loop through the horizontal and vertical length, respectively
    i, j = wp.tid()
    width = image.shape[1]

    x_min = aligned_bbox_min[n,0]
    y_min = aligned_bbox_min[n,1]
    x_max = aligned_bbox_max[n,0]
    y_max = aligned_bbox_max[n,1]

    # Columns are mirrored with the same width-1-j map as make_sonar_image (the
    # upstream `width - y` wrote one column past the image for y == 0).
    for c in range(4):
        image[x_min + i, width - 1 - y_min, c] = bbox_colors[n, c]
        image[x_min + i, width - 1 - y_max, c] = bbox_colors[n, c]
        image[x_min, width - 1 - (y_min + j), c] = bbox_colors[n, c]
        image[x_max, width - 1 - (y_min + j), c] = bbox_colors[n, c]

## This kernel is not used
@wp.kernel
def average(sum: wp.array(ndim=2, dtype=wp.float32),
            count: wp.array(ndim=2, dtype=wp.int32),
            avg: wp.array(ndim=2, dtype=wp.float32)):
    i, j = wp.tid()
    if count[i, j] > 0:
        avg[i, j] = sum[i, j] / wp.float32(count[i, j])


@wp.kernel
def compute_max_intensity_all(array: wp.array(ndim=2, dtype=wp.float32), 
              max_value: wp.array(dtype=wp.float32)):
    i,j = wp.tid()  
    wp.atomic_max(max_value, 0, array[i, j])

@wp.kernel
def compute_max_intensity_range(array: wp.array(ndim=2, dtype=wp.float32), 
              max_value: wp.array(dtype=wp.float32)):
    i, j = wp.tid()
    wp.atomic_max(max_value, i, array[i,j])



@wp.kernel
def normal_2d(seed: int,
              mean: float,
              std: float,
              output: wp.array(ndim=2, dtype=wp.float32),

):
    i, j = wp.tid()
    state = wp.rand_init(seed, i * output.shape[1] + j)  
    
    # Generate normal random variable
    output[i,j] = mean + std * wp.randn(state)



@wp.kernel
def range_dependent_rayleigh_2d(seed: int,
                                r: wp.array(ndim=2, dtype=wp.float32),
                                azi: wp.array(ndim=2, dtype=wp.float32),
                                max_range: float,
                                rayleigh_scale: float,
                                central_peak: float,
                                central_std: float,
                                output: wp.array(ndim=2, dtype = wp.float32)
):
    i, j = wp.tid()
    state = wp.rand_init(seed, i * output.shape[1] + j)
    
    # Generate two uniform random numbers
    n1 = wp.randn(state)
    n2 = wp.randn(state)  # Offset for independence
    
    # Transform to Rayleigh distribution
    rayleigh = rayleigh_scale * wp.sqrt(n1*n1 + n2*n2)
    # Flat reverberation floor (range-independent). A real TVG-corrected sonar
    # image has a roughly UNIFORM noise floor at all ranges, near range included.
    # The old wp.pow(r/max_range, 2.0) factor made the floor grow with range^2
    # (dead black at near range, brightest at max_range) -- backwards, and it left
    # near range with no floor at all. central_peak optionally re-adds a boresight
    # streak (OFF by default; see make_sonar_data). r / max_range are retained in
    # the signature for callers/back-compat but no longer scale the floor.
    output[i,j] = (1.0 + central_peak * wp.exp(-wp.pow(azi[i,j] - wp.PI/2.0, 2.0) / central_std)) * rayleigh





# --- Optional sonar model terms (all off by default; see make_sonar_data) -----

@wp.kernel
def apply_range_gain(binned: wp.array(ndim=2, dtype=wp.float32),
                     r: wp.array(ndim=2, dtype=wp.float32),
                     spreading_exponent: wp.float32,
                     absorption: wp.float32,
                     tvg_exponent: wp.float32):
    """In place: binned *= exp(-2 * absorption * r) * r^(tvg_exponent - spreading_exponent).
    Two-way absorption (absorption in Np/m), spreading loss r^-spreading_exponent
    (4 for point targets, ~3 for area-extensive ones) and a time-varied-gain
    r^+tvg_exponent. Per-range-row normalisation cancels any factor that only
    depends on r, so pair this with normalizing_method="all" to see it."""
    i, j = wp.tid()
    rr = wp.max(r[i, j], wp.float32(1e-6))
    g = wp.exp(-wp.float32(2.0) * absorption * rr) * wp.pow(rr, tvg_exponent - spreading_exponent)
    binned[i, j] = binned[i, j] * g


@wp.kernel
def gamma_speckle_2d(seed: int,
                     looks: int,
                     cell_range: int,
                     cell_azimuth: int,
                     output: wp.array(ndim=2, dtype=wp.float32)):
    """Fully developed speckle as a multiplicative field with mean 1: the mean of
    `looks` unit exponentials (Gamma(L, 1/L); L = 1 is single-look, contrast 1),
    constant over cells of cell_range x cell_azimuth bins (a resolution cell /
    beamwidth spans several grid bins). Written as (value - 0.5) so the map
    kernels' (0.5 + noise) multiplier applies it unchanged."""
    i, j = wp.tid()
    ci = i // wp.max(cell_range, 1)
    cj = j // wp.max(cell_azimuth, 1)
    n_cj = (output.shape[1] + wp.max(cell_azimuth, 1) - 1) // wp.max(cell_azimuth, 1)
    state = wp.rand_init(seed, ci * n_cj + cj)
    acc = wp.float32(0.0)
    for _k in range(wp.max(looks, 1)):
        acc = acc - wp.log(wp.float32(1.0) - wp.randf(state))
    output[i, j] = acc / wp.float32(wp.max(looks, 1)) - wp.float32(0.5)


@wp.kernel
def azimuth_gaussian_blur(src: wp.array(ndim=2, dtype=wp.float32),
                          sigma_bins: wp.float32,
                          radius: int,
                          dst: wp.array(ndim=2, dtype=wp.float32)):
    """Beam-pattern blur: Gaussian along azimuth (axis 1) with sigma in bins,
    truncated at +-radius and renormalised at the fan edges."""
    i, j = wp.tid()
    n = src.shape[1]
    acc = wp.float32(0.0)
    wsum = wp.float32(0.0)
    for k in range(-radius, radius + 1):
        jj = j + k
        if jj >= 0 and jj < n:
            w = wp.exp(-wp.float32(0.5) * wp.float32(k * k) / (sigma_bins * sigma_bins))
            acc = acc + w * src[i, jj]
            wsum = wsum + w
    dst[i, j] = acc / wsum


@wp.kernel
def make_sonar_map_all(r: wp.array(ndim=2, dtype=wp.float32),
                       azi: wp.array(ndim=2, dtype=wp.float32),
                       intensity: wp.array(ndim=2, dtype=wp.float32),
                       max_intensity: wp.array(ndim=1, dtype=wp.float32),
                       gau_noise: wp.array(ndim=2, dtype=wp.float32),
                       range_ray_noise: wp.array(ndim=2, dtype=wp.float32),
                       offset: wp.float32,
                       gain: wp.float32,
                       result: wp.array(ndim=2, dtype=wp.vec3)):
    i, j = wp.tid()
    # Guard against an empty frame (no in-grid returns -> global max 0), which
    # would otherwise divide by zero and emit NaN intensities. Mirrors the
    # per-range guard in make_sonar_map_range.
    if max_intensity[0] != 0.0:
        intensity[i,j] = intensity[i,j]/max_intensity[0]
    # Same op order as make_sonar_map_range: noise on the normalized signal
    # first, display offset/gain last. The two modes used to differ ("all"
    # applied offset/gain BEFORE the noise, so gain scaled the noise in one
    # mode and not the other) -- switching normalizing_method then changed the
    # image beyond just the normalization. Identical at the default
    # offset=0/gain=1.
    intensity[i,j] *= (0.5 + gau_noise[i,j])
    intensity[i,j] += range_ray_noise[i,j]
    intensity[i,j] += offset
    intensity[i,j] *= gain
    intensity[i,j] = wp.clamp(intensity[i,j], wp.float32(0.0), wp.float32(1.0))

    result[i,j] = wp.vec3(r[i,j] * wp.cos(azi[i,j]),
                          r[i,j] * wp.sin(azi[i,j]),
                          intensity[i,j])

@wp.kernel
def make_sonar_map_range(r: wp.array(ndim=2, dtype=wp.float32),
                       azi: wp.array(ndim=2, dtype=wp.float32),
                       intensity: wp.array(ndim=2, dtype=wp.float32),
                       max_intensity: wp.array(ndim=1, dtype=wp.float32),
                       gau_noise: wp.array(ndim=2, dtype=wp.float32),
                       range_ray_noise: wp.array(ndim=2, dtype=wp.float32),
                       offset: wp.float32,
                       gain: wp.float32,
                       result: wp.array(ndim=2, dtype=wp.vec3)):
    i, j = wp.tid()

    if max_intensity[i] !=0:
        intensity[i,j] = intensity[i,j]/max_intensity[i]

    intensity[i,j] *= (0.5 + gau_noise[i,j])
    intensity[i,j] += range_ray_noise[i,j]
    intensity[i,j] += offset
    intensity[i,j] *= gain
    intensity[i,j] = wp.clamp(intensity[i,j], wp.float32(0.0), wp.float32(1.0))

    result[i,j] = wp.vec3(r[i,j] * wp.cos(azi[i,j]),
                          r[i,j] * wp.sin(azi[i,j]),
                          intensity[i,j])
    
@wp.kernel
def make_sonar_image(sonar_data: wp.array(ndim=2, dtype=wp.vec3),
                     sonar_image: wp.array(ndim=3, dtype=wp.uint8)):
    i, j = wp.tid()
    width = sonar_data.shape[1]
    # Clamp before the narrowing uint8 cast. Warp does not saturate on a
    # float->uint8 cast, so an intensity > 1 (e.g. if make_sonar_image is fed an
    # un-normalised grid) would wrap modulo 256 and corrupt the pixel instead of
    # showing full white. The normal pipeline already clamps intensity to [0,1],
    # so this is a no-op there and only hardens the standalone use.
    sonar_rgb = wp.uint8(wp.clamp(sonar_data[i,j][2] * wp.float32(255), wp.float32(0.0), wp.float32(255.0)))
    # Flip columns (mirror the image) while keeping the index in [0, width-1];
    # `width - j` would write index `width` (out of bounds) when j == 0.
    col = width - 1 - j
    sonar_image[i,col,0] = sonar_rgb
    sonar_image[i,col,1] = sonar_rgb
    sonar_image[i,col,2] = sonar_rgb
    sonar_image[i,col,3] = wp.uint8(255)



@wp.kernel
def make_semantics_image(bin_semantics: wp.array(ndim=2, dtype=wp.uint32),
                         semantics_color: wp.array(ndim=2, dtype=wp.uint8),
                         semantics_image: wp.array(ndim=3, dtype=wp.uint8),
                         ):
    i, j = wp.tid()
    width = bin_semantics.shape[1]
    # Same column mirror as make_sonar_image (width-1-j, in bounds at j == 0).
    col = width - 1 - j
    # Semantic ids need not be contiguous, so clamp the palette lookup instead
    # of reading past the colour table for an id beyond its last row.
    k = wp.min(wp.int32(bin_semantics[i,j]), semantics_color.shape[0] - 1)
    semantics_image[i,col,0] = semantics_color[k, 0]
    semantics_image[i,col,1] = semantics_color[k, 1]
    semantics_image[i,col,2] = semantics_color[k, 2]
    semantics_image[i,col,3] = semantics_color[k, 3]


@wp.kernel
def compact_in_range(depth: wp.array(ndim=1, dtype=wp.float32),
                     pcl: wp.array(ndim=2, dtype=wp.float32),
                     normals: wp.array(ndim=2, dtype=wp.float32),
                     semantics: wp.array(ndim=1, dtype=wp.uint32),
                     min_range: wp.float32,
                     max_range: wp.float32,
                     counter: wp.array(ndim=1, dtype=wp.int32),
                     out_pcl: wp.array(ndim=2, dtype=wp.float32),
                     out_normals: wp.array(ndim=2, dtype=wp.float32),
                     out_sem: wp.array(ndim=1, dtype=wp.uint32)):
    """On-device stream compaction of the in-range, finite points -- the GPU
    equivalent of sonar_scan_math.select_in_range_points, so the per-pixel
    depth/pointcloud/normals/semantics never round-trip to the CPU. The kept
    points are appended via an atomic counter (so their order is arbitrary, which
    is fine: the downstream per-point + atomic-binning kernels are order
    independent). out_* must be sized >= number of input points; counter[0] holds
    the kept count after the launch."""
    tid = wp.tid()
    d = depth[tid]
    px = pcl[tid, 0]
    py = pcl[tid, 1]
    pz = pcl[tid, 2]
    if (wp.isfinite(d) and d > min_range and d < max_range
            and wp.isfinite(px) and wp.isfinite(py) and wp.isfinite(pz)):
        i = wp.atomic_add(counter, 0, 1)
        if i < out_pcl.shape[0]:
            out_pcl[i, 0] = px
            out_pcl[i, 1] = py
            out_pcl[i, 2] = pz
            out_normals[i, 0] = normals[tid, 0]
            out_normals[i, 1] = normals[tid, 1]
            out_normals[i, 2] = normals[tid, 2]
            out_sem[i] = semantics[tid]



@wp.kernel
def compact_depth_points(depth: wp.array(ndim=2, dtype=wp.float32),
                         normals: wp.array(ndim=3, dtype=wp.float32),
                         semantics: wp.array(ndim=2, dtype=wp.uint32),
                         instances: wp.array(ndim=2, dtype=wp.uint32),
                         exclude_sem: wp.array(ndim=1, dtype=wp.uint8),
                         cam_to_world: wp.mat44,
                         fx: wp.float32,
                         fy: wp.float32,
                         cx: wp.float32,
                         cy: wp.float32,
                         min_range: wp.float32,
                         max_range: wp.float32,
                         counter: wp.array(ndim=1, dtype=wp.int32),
                         out_pcl: wp.array(ndim=2, dtype=wp.float32),
                         out_normals: wp.array(ndim=2, dtype=wp.float32),
                         out_sem: wp.array(ndim=1, dtype=wp.uint32),
                         out_inst: wp.array(ndim=1, dtype=wp.uint32)):
    """Per-pixel replacement for the 'pointcloud' composite annotator (which
    crashed at world.play() on Isaac Sim 6.x): unproject the
    distance_to_image_plane AOV to WORLD points and append the in-range,
    finite ones -- with their normals, semantic and instance ids -- through an
    atomic counter (order arbitrary; the binning kernels don't care).

    Same unprojection as Isaac's Camera.get_pointcloud() depth fallback: pixel
    centres (u+0.5, v+0.5), pinhole intrinsics fx/fy/cx/cy, ROS optical frame
    (+x right, +y down, +z forward) flipped to the USD camera frame (+y up, -z
    forward), then cam_to_world (column-vector form, i.e. the inverse of the
    transposed CameraParams cameraViewTransform). Pixels whose semantic id has
    exclude_sem[id] != 0 are dropped (the old annotator's includeUnlabelled=False);
    pass a one-element zero array to keep everything. counter[0] holds the kept
    count after the launch; out_* must hold >= H*W points."""
    v, u = wp.tid()
    d = depth[v, u]
    sem = semantics[v, u]
    excluded = False
    if sem < wp.uint32(exclude_sem.shape[0]):
        excluded = exclude_sem[wp.int32(sem)] != wp.uint8(0)
    if (not excluded) and wp.isfinite(d) and d > min_range and d < max_range:
        x_ros = (wp.float32(u) + wp.float32(0.5) - cx) * d / fx
        y_ros = (wp.float32(v) + wp.float32(0.5) - cy) * d / fy
        p = cam_to_world @ wp.vec4(x_ros, -y_ros, -d, wp.float32(1.0))
        if wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2]):
            i = wp.atomic_add(counter, 0, 1)
            if i < out_pcl.shape[0]:
                out_pcl[i, 0] = p[0]
                out_pcl[i, 1] = p[1]
                out_pcl[i, 2] = p[2]
                out_normals[i, 0] = normals[v, u, 0]
                out_normals[i, 1] = normals[v, u, 1]
                out_normals[i, 2] = normals[v, u, 2]
                out_sem[i] = sem
                out_inst[i] = instances[v, u]


## THis kernel not used ##

# ImagingSonarSensor.py
#######################################
# bbox_corners = wp.array(bbox_corners, ndim=3, dtype=wp.float32)
# aligned_bbox_min = wp.empty(shape=(bbox_corners.shape[0], 2), dtype=wp.int32)
# aligned_bbox_max = wp.empty(shape=(bbox_corners.shape[0], 2), dtype=wp.int32)
# aligned_bbox_min.fill_(10000)
# aligned_bbox_max.fill_(0)
# wp.launch(kernel=bin_bbox_process,
#             dim=bbox_corners.shape[0:2],
#             inputs=[
#                 bbox_corners,
#                 self.sonar_grid,
#             ],
#             outputs=[
#                 aligned_bbox_min,
#                 aligned_bbox_max
#             ])

# ImagingSonar_kernels.py
##########################################
# @wp.kernel
# def bin_bbox_process(bbox_corners: wp.array(ndim=3, dtype=wp.float32),
#                      sonar_grid: sonarGrid,
#                      aligned_bbox_min: wp.array(ndim=2, dtype=wp.int32),
#                      aligned_bbox_max: wp.array(ndim=2, dtype=wp.int32)
#                     ):
#     i, j = wp.tid()
#     # Convert 8 corners local frame carteisan to local frame spherical
#     bbox_corner_spher = cartesian_to_spherical(wp.vec3(bbox_corners[i,j,0],
#                                                        bbox_corners[i,j,1],
#                                                        bbox_corners[i,j,2]))
#     # collapse 8 corners to the sonar grid
#     x_bin_idx = wp.int32((bbox_corner_spher[0] - sonar_grid.x_offset) / sonar_grid.x_res)
#     y_bin_idx = wp.int32((bbox_corner_spher[1] - sonar_grid.y_offset) / sonar_grid.y_res)

#     x_bin_idx = wp.clamp(x_bin_idx, 0, sonar_grid.x_num-1)
#     y_bin_idx = wp.clamp(y_bin_idx, 0, sonar_grid.y_num-1)
#     # Compute an axis-aligned minimum-area bbox 
#     # that contains all 8 corners of the 3d bbox
#     # x_min
#     wp.atomic_min(aligned_bbox_min, i, 0, x_bin_idx)
#     # y_min
#     wp.atomic_min(aligned_bbox_min, i, 1, y_bin_idx)
#     # x_max
#     wp.atomic_max(aligned_bbox_max, i, 0, x_bin_idx)
#     # y_max
#     wp.atomic_max(aligned_bbox_max, i, 1, y_bin_idx)
