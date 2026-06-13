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

import csv
from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.geo.align import rotmat_to_quat_xyzw, triangulate_dlt, umeyama_similarity
from openreco.geo.crs import geodetic_to_crs, utm_epsg_for


@register_stage
class Georef(Stage):
    type = "georef"
    version = "2"  # v2: adds GCP-based georeferencing
    deterministic = False  # RANSAC alignment

    def default_params(self) -> dict[str, Any]:
        return {
            "method": "auto",          # auto | gcp | gps | local
            "crs_epsg": 0,             # 0 = auto UTM from GPS; else explicit projected EPSG
            "min_common_images": 3,
            "ransac_max_error_m": 3.0,
            "gcp_file": "",            # CSV: name,X,Y,Z,image,u,v (one row per observation)
            "gcp_crs_epsg": 0,         # CRS of the GCP world coords (required when gcp_file set)
        }

    def run(self, ctx: RunContext) -> StageResult:
        import pycolmap

        model_dep = ctx.input_with("model")        # sfm, or a refine stage between sfm and georef
        model_dir = ctx.input_artifact(model_dep, "model")
        rec = pycolmap.Reconstruction(str(model_dir))

        gps = self._gps_table(ctx)
        has_gcp = bool(ctx.params["gcp_file"])
        method = ctx.params["method"]
        if method == "auto":
            if has_gcp:
                method = "gcp"
            elif len(gps) >= ctx.params["min_common_images"]:
                method = "gps"
            else:
                method = "local"

        out_model = ctx.artifact_path("model")
        out_model.mkdir(parents=True, exist_ok=True)

        if method == "gcp":
            info = self._georef_gcp(pycolmap, ctx, rec)
        elif method == "gps":
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
        data = ctx.read_input_json(ctx.input_with("images"), "images")
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

    def _georef_gcp(self, pycolmap, ctx, rec) -> dict[str, Any]:
        epsg = int(ctx.params["gcp_crs_epsg"]) or int(ctx.params["crs_epsg"])
        if not epsg:
            raise ValueError("gcp_crs_epsg (or crs_epsg) must be set when using GCPs")
        gcps = _read_gcp_file((ctx.project_dir / ctx.params["gcp_file"]).resolve())
        if len(gcps) < ctx.params["min_common_images"]:
            raise ValueError(f"need >= {ctx.params['min_common_images']} GCPs, got {len(gcps)}")

        # projection matrices P = K[R|t] for each registered image, by name
        proj: dict[str, np.ndarray] = {}
        for image_id in rec.reg_image_ids():
            img = rec.image(image_id)
            k = np.asarray(rec.camera(img.camera_id).calibration_matrix())
            m = np.asarray(img.cam_from_world().matrix())  # 3x4 [R|t]
            proj[img.name] = k @ m

        local_pts, world_pts, used = [], [], []
        for name, (world, obs) in gcps.items():
            ps = [proj[im] for im, _ in obs if im in proj]
            uvs = [uv for im, uv in obs if im in proj]
            if len(ps) < 2:
                ctx.logger.warning("GCP %s has <2 observations in registered images; skipped", name)
                continue
            local_pts.append(triangulate_dlt(ps, uvs))
            world_pts.append(world)
            used.append(name)

        if len(local_pts) < ctx.params["min_common_images"]:
            raise RuntimeError(f"only {len(local_pts)} GCPs triangulated; cannot georeference")

        world = np.asarray(world_pts)
        origin = world.mean(axis=0)
        target = world - origin
        scale, rot, trans = umeyama_similarity(np.asarray(local_pts), target)

        sim3d = pycolmap.Sim3d(float(scale), pycolmap.Rotation3d(rotmat_to_quat_xyzw(rot)), trans)
        rec.transform(sim3d)

        # residuals: transform local GCP points, compare to target (meters)
        resid = [float(np.linalg.norm((scale * (rot @ lp) + trans) - tg))
                 for lp, tg in zip(local_pts, target)]
        rms = float(np.sqrt(np.mean(np.square(resid)))) if resid else None
        return {
            "method": "gcp",
            "crs": f"EPSG:{epsg}",
            "crs_epsg": epsg,
            "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
            "num_control": len(used),
            "rms_residual_m": round(rms, 4) if rms is not None else None,
            "scale": float(scale),
        }

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        m = result.metrics
        if m["method"] == "local":
            return [Issue(Severity.WARNING, "no georeferencing — outputs are in a local, "
                          "non-metric frame", hint="provide EXIF GPS or GCPs for metric products")]
        issues = []
        rms = m["rms_residual_m"]
        if rms is not None and rms > 5.0:
            hint = ("check for GCP picking errors / wrong CRS" if m["method"] == "gcp"
                    else "check for GPS outliers or too few control images")
            issues.append(Issue(Severity.WARNING, f"{m['method']} alignment RMS {rms} m is high",
                                hint=hint))
        return issues


def _read_gcp_file(path) -> dict[str, tuple[np.ndarray, list[tuple[str, tuple[float, float]]]]]:
    """Parse a GCP CSV: rows of `name,X,Y,Z,image,u,v` (one per image observation; '#' comments,
    optional header). Returns name -> (world_xyz, [(image, (u, v)), ...])."""
    if not path.exists():
        raise FileNotFoundError(f"gcp_file not found: {path}")
    gcps: dict[str, tuple[np.ndarray, list[tuple[str, tuple[float, float]]]]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[0].lstrip().startswith("#"):
                continue
            cells = [c.strip() for c in row]
            if len(cells) < 7 or not _is_number(cells[1]):
                continue  # skip header / malformed
            name, x, y, z, image, u, v = cells[:7]
            world = np.array([float(x), float(y), float(z)])
            gcps.setdefault(name, (world, []))[1].append((image, (float(u), float(v))))
    return gcps


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
