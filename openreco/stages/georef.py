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
    version = "3"  # v3: GCP control/check split + per-GCP residual report
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
        dep = ctx.find_input("images")          # optional: no ingest wired -> no GPS (local frame)
        if dep is None:
            return {}
        data = ctx.read_input_json(dep, "images")
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

        from openreco.geo.crs import crs_info
        return {
            "method": "gps",
            "crs": f"EPSG:{epsg}",
            "crs_epsg": epsg,
            "crs_info": crs_info(epsg),
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

        pts = []   # [(name, local_xyz, world_xyz, type, n_obs)]
        for name, g in gcps.items():
            ps = [proj[im] for im, _ in g["obs"] if im in proj]
            uvs = [uv for im, uv in g["obs"] if im in proj]
            if len(ps) < 2:
                ctx.logger.warning("GCP %s has <2 observations in registered images; skipped", name)
                continue
            pts.append((name, triangulate_dlt(ps, uvs), g["world"], g["type"], len(ps)))

        control = [p for p in pts if p[3] != "check"]
        check = [p for p in pts if p[3] == "check"]
        if len(control) < ctx.params["min_common_images"]:
            raise RuntimeError(f"only {len(control)} control GCPs triangulated; need "
                               f">= {ctx.params['min_common_images']} (mark fewer as check)")

        # fit the similarity on CONTROL points only (check points validate it independently)
        world = np.asarray([p[2] for p in control])
        origin = world.mean(axis=0)
        local = np.asarray([p[1] for p in control])
        scale, rot, trans = umeyama_similarity(local, world - origin)
        sim3d = pycolmap.Sim3d(float(scale), pycolmap.Rotation3d(rotmat_to_quat_xyzw(rot)), trans)
        rec.transform(sim3d)

        # per-GCP residuals (metres) for both control and check points
        def _resid(p):
            est = scale * (rot @ p[1]) + trans + origin     # back to world CRS
            d = est - p[2]
            return {"name": p[0], "type": p[3], "observations": p[4],
                    "error_m": round(float(np.linalg.norm(d)), 4),
                    "dx": round(float(d[0]), 4), "dy": round(float(d[1]), 4), "dz": round(float(d[2]), 4)}
        gcp_report = [_resid(p) for p in (control + check)]

        def _rms(group):
            e = [r["error_m"] for r in gcp_report if r["type"] == group]
            return round(float(np.sqrt(np.mean(np.square(e)))), 4) if e else None
        ctrl_rms = _rms("control")
        from openreco.geo.crs import crs_info
        return {
            "method": "gcp",
            "crs": f"EPSG:{epsg}",
            "crs_epsg": epsg,
            "crs_info": crs_info(epsg),
            "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
            "num_control": len(control),
            "num_check": len(check),
            "rms_residual_m": ctrl_rms,
            "control_rms_m": ctrl_rms,
            "check_rms_m": _rms("check"),
            "gcps": gcp_report,
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


def _read_gcp_file(path) -> dict[str, dict]:
    """Parse a GCP CSV: rows of `name,X,Y,Z,image,u,v[,type]` (one per image observation; '#'
    comments, optional header). `type` is control|check (default control) — check points are held
    out of the fit to validate accuracy. Returns name -> {world, obs:[(image,(u,v))], type}."""
    if not path.exists():
        raise FileNotFoundError(f"gcp_file not found: {path}")
    gcps: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[0].lstrip().startswith("#"):
                continue
            cells = [c.strip() for c in row]
            if len(cells) < 7 or not _is_number(cells[1]):
                continue  # skip header / malformed
            name, x, y, z, image, u, v = cells[:7]
            kind = cells[7].lower() if len(cells) > 7 and cells[7] else "control"
            g = gcps.setdefault(name, {"world": np.array([float(x), float(y), float(z)]),
                                       "obs": [], "type": "control"})
            g["obs"].append((image, (float(u), float(v))))
            if kind == "check":
                g["type"] = "check"
    return gcps


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
