"""Georeference — pipeline stage 3.

Brings the SfM reconstruction from its arbitrary local frame into a real, metric CRS so that
downstream meters (distances, DSM cells, ortho pixels) are true. v1 supports:

  - gps:   align camera projection centers to EXIF-GPS positions (similarity / Sim3d fit via
           pycolmap.align_reconstruction_to_locations), in a projected CRS (UTM auto-picked).
  - local: no control -> identity transform, flagged non-metric (the Sceaux sample path).
  - (scale-bar and full georeferenced BA are Phase 2.)

To keep float precision sane, georeferenced coordinates are stored relative to a local
`origin` (recorded in georef.json); GeoTIFF/LAS writers add it back. Outputs a transformed
COLMAP model that all later stages consume.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.geo.crs import geodetic_to_crs, utm_epsg_for


@register_stage
class Georef(Stage):
    type = "georef"
    version = "1"
    deterministic = False  # RANSAC alignment

    def default_params(self) -> dict[str, Any]:
        return {
            "method": "auto",          # auto | gps | local
            "crs_epsg": 0,             # 0 = auto UTM from GPS; else explicit projected EPSG
            "min_common_images": 3,
            "ransac_max_error_m": 3.0,
        }

    def run(self, ctx: RunContext) -> StageResult:
        import pycolmap

        model_dir = ctx.input_artifact("sfm", "model")
        rec = pycolmap.Reconstruction(str(model_dir))

        gps = self._gps_table(ctx)
        method = ctx.params["method"]
        if method == "auto":
            method = "gps" if len(gps) >= ctx.params["min_common_images"] else "local"

        out_model = ctx.artifact_path("model")
        out_model.mkdir(parents=True, exist_ok=True)

        if method == "gps":
            info = self._georef_gps(pycolmap, ctx, rec, gps)
        else:
            info = {"method": "local", "crs": "local", "crs_epsg": None,
                    "origin": [0.0, 0.0, 0.0], "num_control": 0, "rms_residual_m": None,
                    "scale": 1.0}

        rec.write(out_model)
        rec.export_PLY(str(ctx.artifact_path("georef.ply")))
        ctx.write_json("georef.json", info)

        return StageResult(
            artifacts={"model": "model", "georef": "georef.json", "ply": "georef.ply"},
            metrics={
                "method": info["method"],
                "crs": info["crs"],
                "num_control": info["num_control"],
                "rms_residual_m": info["rms_residual_m"],
            },
        )

    def _gps_table(self, ctx: RunContext) -> dict[str, tuple[float, float, float]]:
        data = ctx.read_input_json("ingest", "images")
        out: dict[str, tuple[float, float, float]] = {}
        for im in data["images"]:
            if im["excluded"]:
                continue
            if im["lat"] is not None and im["lon"] is not None:
                out[im["name"]] = (im["lat"], im["lon"], im["alt"] if im["alt"] is not None else 0.0)
        return out

    def _georef_gps(self, pycolmap, ctx, rec, gps) -> dict[str, Any]:
        # control = images that are both registered and have GPS
        names, lats, lons, alts = [], [], [], []
        reg_names = {rec.image(i).name for i in rec.reg_image_ids()}
        for name, (la, lo, al) in gps.items():
            if name in reg_names:
                names.append(name)
                lats.append(la)
                lons.append(lo)
                alts.append(al)
        if len(names) < ctx.params["min_common_images"]:
            ctx.logger.warning("insufficient GPS/registered overlap; falling back to local")
            return {"method": "local", "crs": "local", "crs_epsg": None,
                    "origin": [0.0, 0.0, 0.0], "num_control": len(names),
                    "rms_residual_m": None, "scale": 1.0}

        epsg = int(ctx.params["crs_epsg"]) or utm_epsg_for(float(np.median(lats)), float(np.median(lons)))
        world = geodetic_to_crs(np.array(lats), np.array(lons), np.array(alts), epsg)
        origin = world.mean(axis=0)
        target = world - origin  # center for numerical stability

        ransac = pycolmap.RANSACOptions(max_error=float(ctx.params["ransac_max_error_m"]))
        sim3d = pycolmap.align_reconstruction_to_locations(
            rec, names, target, int(ctx.params["min_common_images"]), ransac
        )
        if sim3d is None:
            raise RuntimeError("GPS alignment failed (align_reconstruction_to_locations returned None)")
        rec.transform(sim3d)

        # residuals: registered camera centers vs GPS targets, in meters
        resid = []
        name_to_center = {rec.image(i).name: np.array(rec.image(i).projection_center()) for i in rec.reg_image_ids()}
        for name, t in zip(names, target):
            resid.append(float(np.linalg.norm(name_to_center[name] - t)))
        rms = float(np.sqrt(np.mean(np.square(resid)))) if resid else None

        return {
            "method": "gps",
            "crs": f"EPSG:{epsg}",
            "crs_epsg": epsg,
            "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
            "num_control": len(names),
            "rms_residual_m": round(rms, 4) if rms is not None else None,
            "scale": float(sim3d.scale),
        }

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        m = result.metrics
        if m["method"] == "local":
            return [Issue(Severity.WARNING, "no georeferencing — outputs are in a local, "
                          "non-metric frame", hint="provide EXIF GPS or GCPs for metric products")]
        issues = []
        rms = m["rms_residual_m"]
        if rms is not None and rms > 5.0:
            issues.append(Issue(Severity.WARNING, f"GPS alignment RMS {rms} m is high",
                                hint="check for GPS outliers or too few control images"))
        return issues
