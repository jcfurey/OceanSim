"""Turn payload datasheet parameters (utils.payloads) into the simulated
sensors' constructor arguments, so the runner and the GUI configure them the
same way. Pure Python (unit tested without Isaac Sim).
"""

import math

# Imaging sonar range bins are capped near this many: a real sonar's range
# resolution coarsens with the range setting (fixed sample count), and the
# sim's binning cost grows with the bin count.
MAX_RANGE_BINS = 1024
DEFAULT_SONAR_PAYLOAD = "oculus_m3000d"


def sonar_kwargs(payload, overrides=None):
    """ImagingSonarSensor arguments for a sonar payload.

    max_range is the payload's default working range (an operator setting),
    not the device limit; range_res is the device's resolution or
    working_range / MAX_RANGE_BINS, whichever is coarser; angular_res is the
    beam spacing (FOV / beams). ``overrides`` (the config's sonar_params, with
    hori_fov_deg / vert_fov_deg / min_range / max_range / range_res /
    angular_res keys) wins over the preset."""
    p = payload.params
    o = dict(overrides or {})
    max_range = float(o["max_range"]) if o.get("max_range") is not None else float(p["working_range"])
    kw = dict(
        min_range=float(p["min_range"]),
        max_range=max_range,
        hori_fov=float(p["hori_fov_deg"]),
        vert_fov=float(p["vert_fov_deg"]),
        range_res=max(float(p["range_res"]), max_range / MAX_RANGE_BINS),
        angular_res=float(p["hori_fov_deg"]) / float(p["n_beams"]),
        acoustic_frequency=sonar_frequency(payload),
        beam_fwhm_deg=float(p.get("beamwidth_h_deg", 0.0)),
    )
    for src, dst in (("hori_fov_deg", "hori_fov"), ("vert_fov_deg", "vert_fov"),
                     ("min_range", "min_range"), ("range_res", "range_res"),
                     ("angular_res", "angular_res")):
        if o.get(src) is not None:
            kw[dst] = float(o[src])
    if kw["max_range"] > float(p["max_range"]):
        raise ValueError(f"{payload.name}: max_range {kw['max_range']} m exceeds the device's "
                         f"{p['max_range']} m")
    if not (0.0 <= kw["min_range"] < kw["max_range"]):
        raise ValueError(f"{payload.name}: require 0 <= min_range < max_range")
    if not (0.0 < kw["hori_fov"] < 180.0 and 0.0 < kw["vert_fov"] < 180.0):
        raise ValueError(f"{payload.name}: FOV must be in (0, 180) degrees")
    if kw["range_res"] <= 0.0 or kw["angular_res"] <= 0.0:
        raise ValueError(f"{payload.name}: range_res and angular_res must be positive")
    return kw


def sonar_frequency(payload):
    return float(payload.params["frequency_hz"])


def sonar_model_params(payload, overrides=None):
    """make_sonar_data settings for a sonar payload; explicit model settings
    (the config's sonar_params.model_params) take precedence.

    Every preset normalises a ping by its global maximum: a range-row maximum
    promotes every weak echo to the same brightness and hides relative target
    strength and shadows. A payload with a published azimuth resolving power
    (beamwidth_h_deg) is blurred by a Gaussian with that FWHM, which integrates
    neighbouring output beams rather than giving every bin independent,
    infinitely narrow resolving power; without one, beams are not blurred.
    """
    params = {"normalizing_method": "all"}
    if "beamwidth_h_deg" in payload.params:
        params["beam_fwhm_deg"] = float(payload.params["beamwidth_h_deg"])
    params.update(overrides or {})
    return params


def dvl_kwargs(payload):
    """DVLsensor arguments: Janus beam angle, altitude limits and, when the
    datasheet gives single-ping noise, its variance (beam space)."""
    p = payload.params
    noise = p.get("velocity_noise")
    return dict(elevation=float(p["beam_angle_deg"]), min_range=float(p["min_range"]),
                max_range=float(p["max_range"]),
                vel_cov=float(noise) ** 2 if noise else 0.0)


def altimeter_kwargs(payload):
    p = payload.params
    return dict(min_range=float(p["min_range"]), max_range=float(p["max_range"]),
                beamwidth_deg=float(p["beamwidth_deg"]), frequency_hz=float(p["frequency_hz"]))


def focal_length_for_hfov(hfov_deg, horizontal_aperture):
    """Pinhole focal length giving ``hfov_deg`` for the camera's horizontal
    aperture (same units as the aperture)."""
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError(f"horizontal FOV must be in (0, 180) deg, got {hfov_deg}")
    return float(horizontal_aperture) / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
