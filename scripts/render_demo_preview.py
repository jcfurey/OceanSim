#!/usr/bin/env python3
"""Render a demo scene with Isaac Sim's RTX renderer, without ROS or simulation."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--vehicle", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera", default="Overview", choices=["Overview", "Inspection"])
    args = parser.parse_args()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    try:
        import numpy as np
        from PIL import Image
        from pxr import Gf, UsdGeom
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage
        create_new_stage()
        stage = omni.usd.get_context().get_stage()
        add_reference_to_stage(str(args.scene.resolve()), "/World/Environment")
        if args.vehicle:
            prim = add_reference_to_stage(str(args.vehicle.resolve()), "/World/Revolution")
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(-2, 0, -0.8))
        camera = f"/World/Environment/Cameras/{args.camera}"
        if not stage.GetPrimAtPath(camera).IsValid():
            raise ValueError(f"Scene has no {args.camera} camera")
        product = rep.create.render_product(camera, (1440, 900))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(product)
        for _ in range(60):
            app.update()
        rep.orchestrator.step(rt_subframes=8)
        rgba = np.asarray(annotator.get_data())
        if rgba.size == 0:
            raise RuntimeError("RTX renderer returned an empty frame")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba[..., :3]).save(args.output)
        print(f"Rendered {args.output} ({rgba.shape[1]} x {rgba.shape[0]})")
        annotator.detach(product)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
