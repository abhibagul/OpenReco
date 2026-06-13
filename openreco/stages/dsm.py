"""DSM (Digital Surface Model) + orthophoto — pipeline stages 6 & 7.

Both rasterize the point cloud top-down at a chosen ground resolution. The DSM stage emits a
single-band elevation GeoTIFF; the ortho stage emits a 3-band RGB GeoTIFF on the same grid.
When georeferenced (georef produced a CRS), outputs carry the project CRS and true-world
coordinates and open directly in QGIS; in the local-frame fallback they're plain rasters with
a warning. (True image-resampled orthorectification with seamlines is Phase 2; v1 is a
point-cloud orthophoto.)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.pointcloud import read_ply_xyzrgb
from openreco.io.raster import grid_topdown, write_geotiff


def _load(ctx: RunContext):
    xyz, rgb = read_ply_xyzrgb(ctx.input_artifact("mvs", "points"))
    if rgb is None:
        rgb = np.full((len(xyz), 3), 200, np.uint8)
    meta = ctx.read_input_json("mvs", "meta")
    origin = np.array(meta.get("origin", [0.0, 0.0, 0.0]))
    return xyz, rgb, meta.get("crs_epsg"), origin


@register_stage
class Dsm(Stage):
    type = "dsm"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {"resolution_m": 0.1, "fill_holes": True}

    def run(self, ctx: RunContext) -> StageResult:
        xyz, rgb, epsg, origin = _load(ctx)
        res = float(ctx.params["resolution_m"])
        dsm, _ortho, west, north = grid_topdown(xyz, xyz[:, 2], rgb, res, ctx.params["fill_holes"])
        # true-world top-left = local corner + CRS origin
        write_geotiff(ctx.artifact_path("dsm.tif"), dsm, west + origin[0], north + origin[1],
                      res, epsg, nodata=float("nan"))
        ctx.write_json("dsm.json", {
            "resolution_m": res, "width": int(dsm.shape[1]), "height": int(dsm.shape[0]),
            "crs_epsg": epsg, "z_min": float(np.nanmin(dsm)), "z_max": float(np.nanmax(dsm)),
        })
        return StageResult(
            artifacts={"dsm": "dsm.tif", "meta": "dsm.json"},
            metrics={"width": int(dsm.shape[1]), "height": int(dsm.shape[0]),
                     "georeferenced": epsg is not None},
        )

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        if not result.metrics["georeferenced"]:
            return [Issue(Severity.WARNING, "DSM is in a local frame (no CRS) — georeference "
                          "with GPS/GCPs for a metric DSM")]
        return []


@register_stage
class Ortho(Stage):
    type = "ortho"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {"resolution_m": 0.05, "fill_holes": True}

    def run(self, ctx: RunContext) -> StageResult:
        xyz, rgb, epsg, origin = _load(ctx)
        res = float(ctx.params["resolution_m"])
        _dsm, ortho, west, north = grid_topdown(xyz, xyz[:, 2], rgb, res, ctx.params["fill_holes"])
        write_geotiff(ctx.artifact_path("ortho.tif"), ortho, west + origin[0], north + origin[1],
                      res, epsg)
        ctx.write_json("ortho.json", {
            "resolution_m": res, "width": int(ortho.shape[1]), "height": int(ortho.shape[0]),
            "crs_epsg": epsg,
        })
        return StageResult(
            artifacts={"ortho": "ortho.tif", "meta": "ortho.json"},
            metrics={"width": int(ortho.shape[1]), "height": int(ortho.shape[0]),
                     "georeferenced": epsg is not None},
        )

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        issues = [Issue(Severity.INFO, "v1 orthophoto is point-cloud-based; true image "
                        "orthorectification + seamlines is Phase 2")]
        if not result.metrics["georeferenced"]:
            issues.append(Issue(Severity.WARNING, "ortho is in a local frame (no CRS)"))
        return issues
