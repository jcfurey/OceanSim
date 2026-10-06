#!/usr/bin/env python3
"""OceanSim vehicle / URDF tool (no Isaac Sim needed; Python 3 + numpy).

  list                         platforms and payloads
  show PLATFORM [-p PAYLOAD..] performance summary of a configuration
  export PLATFORM [-p PAYLOAD..] [--bare] [--gazebo] [-o FILE]
                               write the vehicle (with payloads) as a URDF
  inspect FILE.urdf            what OceanSim reads from a URDF (and assumes)

Examples:
  scripts/oceansim_urdf.py export bluerov2 -p oculus_m750d -p waterlinked_a50 -o brov.urdf
  scripts/oceansim_urdf.py export deeptrekker_revolution --gazebo -o rev_gz.urdf
  scripts/oceansim_urdf.py inspect my_rov.urdf
"""

import argparse
import importlib.util
import math
import os
import sys

_UTILS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "isaacsim", "oceansim", "utils")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_UTILS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


platforms = _load("platforms")
payloads = _load("payloads")
vd = _load("vehicle_dynamics")
urdf_export = _load("urdf_export")
urdf_platform = _load("urdf_platform")
urdf_platform.platforms = platforms


def _top_speed(model, axis, sign=1.0):
    import numpy as np
    d = np.zeros(6)
    d[axis] = sign
    f = float(model.max_wrench(d)[axis]) * sign
    q = model.hydro.quadratic_damping[axis]
    lin = model.hydro.linear_damping[axis]
    if f <= 0.0:
        return f, 0.0
    if q <= 0.0:
        return f, (f / lin if lin > 0 else float("inf"))
    return f, (-lin + math.sqrt(lin * lin + 4.0 * q * f)) / (2.0 * q)


def _summary(spec, fitted, rho=1000.0):
    h = vd.resolve_hydro(spec, fitted, rho=rho)
    model = vd.from_platform(spec, payloads=fitted, rho=rho)
    net = (rho * h["displaced_volume"] - h["mass"]) * vd.GRAVITY
    lines = [f"{spec.name}: {spec.description}",
             f"  mass {h['mass']:.2f} kg, net buoyancy {net:+.1f} N (rho {rho:g}), "
             f"CoG offset {tuple(round(v, 3) for v in h['cog'])} m, {model.thrusters.count} thrusters",
             f"  payloads: {[p.name for p in fitted] or 'none'}"]
    if h.get("trim"):
        t = h["trim"]
        lines.append(f"  trim: {t['kind']} {t['mass']:.3f} kg / {t['volume'] * 1e3:.2f} L "
                     f"to restore the bare vehicle's buoyancy")
    for label, axis, sign in (("surge", 0, 1), ("reverse", 0, -1), ("sway", 1, 1),
                              ("heave up", 2, 1), ("heave down", 2, -1), ("yaw", 5, 1)):
        f, v = _top_speed(model, axis, sign)
        unit = "rad/s" if axis >= 3 else "m/s"
        lines.append(f"  {label:10s} {f:7.1f} {'N m' if axis >= 3 else 'N  '} -> {v:5.2f} {unit}")
    lines.append(f"  sources: {h['sources']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("show", "export"):
        p = sub.add_parser(name)
        p.add_argument("platform")
        p.add_argument("-p", "--payload", action="append", default=None,
                       help="payload to fit (repeat); default: the platform's standard set")
        p.add_argument("--bare", action="store_true", help="no payloads at all")
        p.add_argument("--rho", type=float, default=1000.0, help="water density (kg/m^3)")
        if name == "export":
            p.add_argument("--gazebo", action="store_true",
                           help="continuous thruster joints + Gazebo Sim plugins")
            p.add_argument("-o", "--output", help="output file (default: stdout)")
    p = sub.add_parser("inspect")
    p.add_argument("urdf")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        print("Platforms:")
        for n in platforms.available_platforms():
            s = platforms.get_platform(n)
            print(f"  {n:28s} {s.description}")
            if s.default_payloads:
                print(f"  {'':28s}   standard: {', '.join(s.default_payloads)}")
            if s.payload_options:
                print(f"  {'':28s}   options:  {', '.join(s.payload_options)}")
        print("Payloads:")
        for n in payloads.available_payloads():
            p = payloads.get_payload(n)
            print(f"  {n:22s} {p.kind:9s} {p.description}")
        return 0

    if args.cmd == "inspect":
        with open(args.urdf) as f:
            text = f.read()
        spec, notes = urdf_platform.platform_from_urdf(text, urdf_path=args.urdf)
        print(_summary(spec, []))
        for n in notes:
            print(f"  note: {n}")
        return 0

    spec = platforms.get_platform(args.platform)
    fitted = payloads.select_payloads(spec, [] if args.bare else args.payload)
    if args.cmd == "show":
        print(_summary(spec, fitted, args.rho))
        return 0
    text = urdf_export.platform_to_urdf(spec, fitted, gazebo=args.gazebo, rho=args.rho)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"wrote {args.output} ({spec.name}, payloads: {[p.name for p in fitted]})",
              file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
