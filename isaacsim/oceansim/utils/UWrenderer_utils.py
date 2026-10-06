import math

import warp as wp


def water_coefficients(values, kind="beta"):
    """Per-channel beta (1/m, as in exp(-beta z)) for one RGB water table entry.
    kind="transmittance": the values are per-metre transmittances N (e.g. the
    Jerlov water types, I = J N^z as in UWCNN), so beta = -ln(N), N in (0, 1].
    kind="beta": the values are already beta and are returned unchanged."""
    values = tuple(float(v) for v in values)
    if kind == "transmittance":
        if any(not (0.0 < v <= 1.0) for v in values):
            raise ValueError(f"transmittance must be in (0, 1], got {values}")
        return tuple(-math.log(v) for v in values)
    if kind != "beta":
        raise ValueError(f"unknown water coefficient kind {kind!r}")
    return values


@wp.func
def vec3_exp(exponent: wp.vec3):
    return wp.vec3(wp.exp(exponent[0]), wp.exp(exponent[1]), wp.exp(exponent[2]), dtype=type(exponent[0]))

@wp.func
def vec3_mul(vec_1: wp.vec3,
            vec_2: wp.vec3):
    return wp.vec3(vec_1[0] * vec_2[0], vec_1[1] * vec_2[1], vec_1[2] * vec_2[2], dtype=type(vec_1[0]))

@wp.func
def srgb_to_linear(c: wp.float32):
    """sRGB transfer function decode, c in [0, 1]."""
    if c <= wp.float32(0.04045):
        return c / wp.float32(12.92)
    return wp.pow((c + wp.float32(0.055)) / wp.float32(1.055), wp.float32(2.4))


@wp.func
def linear_to_srgb(c: wp.float32):
    """sRGB transfer function encode, c clamped to [0, 1]."""
    c = wp.clamp(c, wp.float32(0.0), wp.float32(1.0))
    if c <= wp.float32(0.0031308):
        return c * wp.float32(12.92)
    return wp.float32(1.055) * wp.pow(c, wp.float32(1.0) / wp.float32(2.4)) - wp.float32(0.055)


@wp.func
def srgb8_to_linear3(r: wp.uint8, g: wp.uint8, b: wp.uint8):
    k = wp.float32(1.0) / wp.float32(255.0)
    return wp.vec3(srgb_to_linear(wp.float32(r) * k),
                   srgb_to_linear(wp.float32(g) * k),
                   srgb_to_linear(wp.float32(b) * k))


@wp.func
def srgb_unit_to_linear3(c: wp.vec3):
    return wp.vec3(srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]))


@wp.func
def linear3_to_srgb8(c: wp.vec3, out: wp.array(ndim=3, dtype=wp.uint8), i: int, j: int):
    out[i, j, 0] = wp.uint8(linear_to_srgb(c[0]) * wp.float32(255.0) + wp.float32(0.5))
    out[i, j, 1] = wp.uint8(linear_to_srgb(c[1]) * wp.float32(255.0) + wp.float32(0.5))
    out[i, j, 2] = wp.uint8(linear_to_srgb(c[2]) * wp.float32(255.0) + wp.float32(0.5))


@wp.func
def uw_image_formation(raw_r: wp.uint8, raw_g: wp.uint8, raw_b: wp.uint8,
                       depth: wp.float32,
                       backscatter_value: wp.vec3,
                       atten_coeff: wp.vec3,
                       backscatter_coeff: wp.vec3):
    """Akkaynak-Treibitz (revised) underwater image formation in LINEAR light:
        I = J exp(-beta_D z) + B_inf (1 - exp(-beta_B z)),
    z the range along the ray. LdrColor is sRGB-encoded, so J is decoded
    first; applying the model to the encoded values (the old code) made the
    effective beta ~2x the nominal one. backscatter_value (B_inf) is the
    veiling-light colour as seen, i.e. sRGB in [0, 1] -- infinitely far water
    renders exactly that colour. Returns linear RGB."""
    if not wp.isfinite(depth):
        # Background (no hit): +inf * a zero coefficient would be NaN; a large
        # finite range gives exp(0) = 1 / exp(-large) = 0 per channel instead.
        depth = wp.float32(1.0e4)
    j_lin = srgb8_to_linear3(raw_r, raw_g, raw_b)
    b_inf = srgb_unit_to_linear3(backscatter_value)
    direct = vec3_mul(j_lin, vec3_exp(-depth * atten_coeff))
    veil = vec3_mul(b_inf, wp.vec3(1.0, 1.0, 1.0) - vec3_exp(-depth * backscatter_coeff))
    return direct + veil


@wp.kernel
def UW_render(raw_image: wp.array(ndim=3, dtype=wp.uint8),
             depth_image: wp.array(ndim=2, dtype=wp.float32),
             backscatter_value: wp.vec3,
             atten_coeff: wp.vec3,
             backscatter_coeff: wp.vec3,
             uw_image: wp.array(ndim=3, dtype=wp.uint8)):
    """
    Notice: This kernel is deprecated for UW_render_2, which support caustics.
    Render the UW image.
    """
    i,j = wp.tid()
    # distance_to_camera returns +inf for background (no hit); the model func
    # guards it (a zero coefficient channel would otherwise give NaN).
    lin = uw_image_formation(raw_image[i,j,0], raw_image[i,j,1], raw_image[i,j,2],
                             depth_image[i,j], backscatter_value, atten_coeff, backscatter_coeff)
    linear3_to_srgb8(lin, uw_image, i, j)
    uw_image[i,j,3] = raw_image[i,j,3]


@wp.kernel
def UW_render_2(raw_image: wp.array(ndim=3, dtype=wp.uint8),
             depth_image: wp.array(ndim=2, dtype=wp.float32),
             scale: float,
             backscatter_value: wp.vec3,
             atten_coeff: wp.vec3,
             backscatter_coeff: wp.vec3,
             uw_image: wp.array(ndim=3, dtype=wp.uint8)):
    i,j = wp.tid()
    # Same linear-light model as UW_render; `scale` multiplies both
    # coefficients (scene-scale randomisation in the SDG writer).
    lin = uw_image_formation(raw_image[i,j,0], raw_image[i,j,1], raw_image[i,j,2],
                             depth_image[i,j], backscatter_value,
                             atten_coeff * scale, backscatter_coeff * scale)
    linear3_to_srgb8(lin, uw_image, i, j)
    uw_image[i,j,3] = raw_image[i,j,3]

@wp.func
def fract(x: float):
    return x - wp.floor(x)

@wp.func
def clamp01(x: float):
    return wp.min(1.0, wp.max(0.0, x))

@wp.func
def smoothstep(edge0: float, edge1: float, x: float):
    t = clamp01((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)

@wp.func
def length2(v: wp.vec2f):
    return wp.sqrt(v[0]*v[0] + v[1]*v[1])

@wp.func
def normalize2(v: wp.vec2f):
    l = length2(v)
    return wp.vec2f(v[0]/(l+1e-6), v[1]/(l+1e-6))

@wp.func
def randomVal(inVal: float):
    d = inVal * 12.9898 + 2523.2361 * 78.233
    s = wp.sin(d)
    return fract(s * 43758.5453) - 0.5

@wp.func
def randomVec2(inVal: float):
    v = wp.vec2f(randomVal(inVal), randomVal(inVal + 151.523))
    return normalize2(v)

@wp.func
def makeWaves(uv: wp.vec2f, theTime: float, offset: float, timeSpeed: float):
    result = 0.0
    for n in range(16):
        i = float(n) + offset
        randVec = randomVec2(i)
        direction = uv[0] * randVec[0] + uv[1] * randVec[1]
        s = wp.sin(direction * randomVal(i + 1.6516) + theTime * timeSpeed)
        s = smoothstep(0.0, 1.0, s)
        result += randomVal(i + 123.0) * s
    return result

# NOTE: ndim=3 tells Warp this is a 3D array (H, W, 3)
@wp.kernel
def water_caustics(
    out_img: wp.array(ndim=3, dtype=wp.uint8),  # shape: (H, W, 3)
    width: int, height: int,
    time: float, timeSpeed: float,
):
    x, y = wp.tid()  # (W, H)

    # match your GLSL: uv normalized by width (iResolution.x)
    uv = wp.vec2f(float(x)/float(width), float(y)/float(width))
    uv2 = wp.vec2f(uv[0] * 150.0, uv[1] * 150.0)
    uv  = wp.vec2f(uv[0] * 2.0,   uv[1] * 2.0)

    r1 = makeWaves(wp.vec2f(uv2[0] + time*timeSpeed, uv2[1]),
                   time, 0.1, timeSpeed)
    r2 = makeWaves(wp.vec2f(uv2[0] - time*0.8*timeSpeed, uv2[1]),
                   time*0.8 + 0.06, 0.26, timeSpeed)

    r1 = smoothstep(0.4, 1.1, 1.0 - wp.abs(r1))
    r2 = smoothstep(0.4, 1.1, 1.0 - wp.abs(r2))
    val = 2.0 * smoothstep(0.35, 1.8, (r1 + r2) * 0.5)

    col = wp.uint8(wp.clamp(val * 0.7 * 255.0, 0.0, 255.0))

    # write to (H, W, 3) directly (row-major: y, x, channel)
    out_img[y, x, 0] = col
    out_img[y, x, 1] = col
    out_img[y, x, 2] = col
    out_img[y, x, 3] = wp.uint8(255)

@wp.kernel
def blend_caustics_PyTX(
    rgb_img: wp.array(ndim=3, dtype=wp.uint8),       # (H,W,3)
    depth_img: wp.array(ndim=2, dtype=wp.float32),        # (H,W)
    normals_img: wp.array(ndim=3, dtype=wp.float32),   # (H,W,3)
    caustics_img: wp.array(ndim=3, dtype=wp.uint8),     # (H,W) grayscale
    sun_dir: wp.vec3f,                         # normalized
    blend_weight: float,
    min_depth: float,
    max_depth: float,
    out_img: wp.array(ndim=3, dtype=wp.uint8)        # (H,W,3)
):
    x, y = wp.tid()  # 2D thread indices

    rgb = wp.vec3(wp.float32(rgb_img[y, x, 0]), wp.float32(rgb_img[y, x, 1]), wp.float32(rgb_img[y, x, 2]), dtype=wp.float32)
    depth = depth_img[y, x]
    nml = wp.vec3(wp.float32(normals_img[y, x, 0]), wp.float32(normals_img[y, x, 1]), wp.float32(normals_img[y, x, 2]), dtype=wp.float32)
    caustic_val = wp.vec3(wp.float32(caustics_img[y, x, 0]), wp.float32(caustics_img[y, x, 1]), wp.float32(caustics_img[y, x, 2]), dtype=wp.float32)

    # Lambert term
    dot_normals = wp.max(wp.dot(nml, sun_dir), 0.0)

    # Depth weighting
    norm_depth = (depth - min_depth) / (max_depth - min_depth + 1e-8)
    depth_weight = 1.0 - norm_depth

    # Final blend factor
    blend_factor = blend_weight * dot_normals * depth_weight

    blended = rgb * (1.0 - blend_factor) + caustic_val * blend_factor
    out_img[y, x, 0] = wp.uint8(wp.clamp(blended[0], 0.0, 255.0))
    out_img[y, x, 1] = wp.uint8(wp.clamp(blended[1], 0.0, 255.0))
    out_img[y, x, 2] = wp.uint8(wp.clamp(blended[2], 0.0, 255.0))
    out_img[y, x, 3] = wp.uint8(255)

@wp.kernel
def blend_caustics(
    rgb_aov: wp.array(ndim=3, dtype=wp.uint8),       # (H, W, 3 or 4) base color RGBA
    world_pos_aov: wp.array(ndim=3, dtype=wp.float32), # (H, W, 3) world positions
    world_nml_aov: wp.array(ndim=3, dtype=wp.float32), # (H, W, 3) world normals
    caustics_aov: wp.array(ndim=3, dtype=wp.uint8),  # (H, W, 3 or 4) caustics RGBA
    sun_dir: wp.vec3f,                               # light dir (world space)
    blend_weight: float,                             # base blend weight [0..1]
    uv_scale_x: float,                               # tiling scale for caustics U direction
    uv_scale_y: float,                               # tiling scale for caustics V direction
    depth_min: float,                                # min depth for normalization
    depth_max: float,                                # max depth for normalization
    tex_w: int,                                      # caustics width
    tex_h: int,                                      # caustics height
    out_aov: wp.array(ndim=3, dtype=wp.uint8)        # (H, W, 4) output RGBA
):
    x, y = wp.tid()  # 2D launch index
    


    # Read base color (handle both RGB and RGBA)
    base_col = wp.vec3(wp.float32(rgb_aov[y, x, 0]), wp.float32(rgb_aov[y, x, 1]), wp.float32(rgb_aov[y, x, 2]), dtype=wp.float32)

    # Read world pos & normal
    pos = wp.vec3(world_pos_aov[y, x, 0], world_pos_aov[y, x, 1], world_pos_aov[y, x, 2])
    nml = wp.normalize(wp.vec3(world_nml_aov[y, x, 0], world_nml_aov[y, x, 1], world_nml_aov[y, x, 2]))

    # Lambert shading factor (expects sun_dir to point from surface toward light)
    sun = wp.normalize(sun_dir)
    ndotl = wp.max(wp.dot(nml, sun), 0.0)

    # Depth-based weight
    denom = depth_max - depth_min + 1e-8
    norm_depth = wp.clamp((pos.z - depth_min) / denom, 0.0, 1.0)
    depth_weight = 1.0 - norm_depth

    # Blend factor
    blend_factor = wp.clamp(blend_weight * ndotl * depth_weight, 0.0, 1.0)

    # World-space planar UV projection onto the horizontal (XY) plane -- Isaac
    # stages are Z-up and caustics are cast down from the surface. (Projecting
    # on XZ sampled a single texture row across a flat seafloor: stripes.)
    u = pos.x * uv_scale_x
    v = pos.y * uv_scale_y
    
    # Add aspect ratio correction to prevent stretching
    aspect_ratio = wp.float32(tex_w) / wp.float32(tex_h)
    u = u * aspect_ratio  # Scale U to match texture proportions
    
    u = u - wp.floor(u)
    v = v - wp.floor(v)

    # Map to texture coordinates
    tx = wp.int32(wp.clamp(wp.floor(u * wp.float32(tex_w - 1)), 0.0, wp.float32(tex_w - 1)))
    ty = wp.int32(wp.clamp(wp.floor(v * wp.float32(tex_h - 1)), 0.0, wp.float32(tex_h - 1)))

    # Sample caustics texture (explicit channels)
    tex_r = wp.float32(caustics_aov[ty, tx, 0])
    tex_g = wp.float32(caustics_aov[ty, tx, 1])
    tex_b = wp.float32(caustics_aov[ty, tx, 2])
    tex_a = wp.float32(caustics_aov[ty, tx, 3]) / 255.0
    tex_rgb = wp.vec3(tex_r, tex_g, tex_b, dtype=wp.float32)

    # Scale caustics intensity by blend factor
    caustics_intensity = wp.clamp(tex_a * blend_factor, 0.0, 1.0)
    
    # Add caustics to base color (additive blending)
    out_rgb = base_col + tex_rgb * caustics_intensity

    out_aov[y, x, 0] = wp.uint8(wp.clamp(out_rgb[0], 0.0, 255.0))
    out_aov[y, x, 1] = wp.uint8(wp.clamp(out_rgb[1], 0.0, 255.0))
    out_aov[y, x, 2] = wp.uint8(wp.clamp(out_rgb[2], 0.0, 255.0))
    out_aov[y, x, 3] = wp.uint8(255)


@wp.func
def intrinsics_from_proj(P:wp.mat44f, width:int, height:int):
    fx = P[0,0] * wp.float32(width) / 2.0
    fy = P[1,1] * wp.float32(height) / 2.0
    cx = (1.0 - P[0,2]) * wp.float32(width) / 2.0
    cy = (1.0 + P[1,2]) * wp.float32(height) / 2.0
    return fx, fy, cx, cy

@wp.kernel
def depth_to_world_pos(
    depth: wp.array(ndim=2, dtype=wp.float32),           # (H, W) depth buffer in world units
    proj_matrix: wp.mat44f,     # (4, 4) projection matrix
    view_matrix: wp.mat44f,     # (4, 4) camera->world matrix, column-vector form, i.e.
                                # inv(cameraViewTransform.reshape(4, 4).T) -- NOT the
                                # raw CameraParams cameraViewTransform (world->camera)
    H: int,
    W: int,
    world_points: wp.array(ndim=3, dtype=wp.float32),

):
    x, y = wp.tid()

    d = depth[y, x]

    # Check validity using mathematical operations (avoid branching)
    is_valid = wp.float32(d > 0.0 and wp.isfinite(d))
    
    # Since depth is in world units, we can use it directly to reconstruct world position
    # Extract camera parameters from projection matrix
    fx, fy, cx, cy = intrinsics_from_proj(proj_matrix, W, H)

    # Pixel → ray direction in camera space
    cam_x = (wp.float32(x) - cx) / fx
    cam_y = -(wp.float32(y) - cy) / fy  # flip Y if needed
    cam_z = -1.0  # forward in OpenGL convention

    ray_vec = wp.normalize(wp.vec3(cam_x, cam_y, cam_z))

    # Scale ray by depth (world units)
    cam_pos = ray_vec * d

    # Camera → world
    world_pos_h = view_matrix @ wp.vec4(cam_pos[0], cam_pos[1], cam_pos[2], 1.0)

    # Store world-space position (zero out if invalid depth)
    world_points[y, x, 0] = world_pos_h[0] * is_valid
    world_points[y, x, 1] = world_pos_h[1] * is_valid
    world_points[y, x, 2] = world_pos_h[2] * is_valid


@wp.kernel
def UW_depth_turbidity_attenuator(
    raw_image: wp.array(ndim=3, dtype=wp.uint8),
    depth_image: wp.array(ndim=2, dtype=wp.float32),
    max_range: wp.float32,
    backscatter_value: wp.vec3,
    atten_coeff: wp.vec3,
    backscatter_coeff: wp.vec3,
    sigma: wp.float32,
    min_visibility: wp.float32,
    seed: int,
    adjusted_depth: wp.array(ndim=2, dtype=wp.float32),
):
    """Turbidity-limited depth: keep a pixel's depth (plus Gaussian noise,
    std sigma) only where the surface is visible through the water -- its
    attenuated direct signal D = J exp(-beta_D z) is at least min_visibility of
    what reaches the camera, D / (D + B) with B = B_inf (1 - exp(-beta_B z))
    the backscatter veil (linear light, channel means). The old gate compared
    absolute sRGB brightness to 0.25, so dark surfaces were never measured even
    at 0.1 m and the backscatter / max_range inputs were ignored."""
    i, j = wp.tid()
    depth = depth_image[i, j]

    # No return past the sensor's range or without a hit (inf/NaN).
    if not wp.isfinite(depth) or depth <= wp.float32(0.0) or depth > max_range:
        adjusted_depth[i, j] = wp.float32(0.0)
        return

    j_lin = srgb8_to_linear3(raw_image[i, j, 0], raw_image[i, j, 1], raw_image[i, j, 2])
    b_inf = srgb_unit_to_linear3(backscatter_value)
    direct = vec3_mul(j_lin, vec3_exp(-depth * atten_coeff))
    veil = vec3_mul(b_inf, wp.vec3(1.0, 1.0, 1.0) - vec3_exp(-depth * backscatter_coeff))
    d = (direct[0] + direct[1] + direct[2]) / wp.float32(3.0)
    b = (veil[0] + veil[1] + veil[2]) / wp.float32(3.0)
    visibility = d / (d + b + wp.float32(1e-6))

    if visibility < min_visibility:
        adjusted_depth[i, j] = wp.float32(0.0)
    else:
        state = wp.rand_init(seed, i * depth_image.shape[1] + j)
        sample = wp.randn(state)
        g_noise = sample * sigma
        adjusted_depth[i, j] = wp.max(depth + g_noise, wp.float32(0.0))
