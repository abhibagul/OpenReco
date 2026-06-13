"""Image overlap / coverage map — QA pipeline stage.

Projects each registered image's frame onto the ground plane and counts how many images cover
each cell. The result is the photogrammetric "overlap" map: areas with low overlap (<3 images)
reconstruct poorly, so this is a key QA product. Emits a georeferenced GeoTIFF (overlap count)
and a colormapped PNG preview, plus summary metrics surfaced in the processing report.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.geo.footprint import colormap_overlap, ground_footprint, rasterize_overlap
from openreco.io.raster import write_geotiff


@register_stage
class Coverage(Stage):
    type = "coverage"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {"resolution_m": 2.0, "min_overlap": 3}

    def run(self, ctx: RunContext) -> StageResult:
        import pycolmap

        model_dir = ctx.input_artifact("georef", "model")
        georef = ctx.read_input_json("georef", "georef")
        origin = np.array(georef.get("origin", [0.0, 0.0, 0.0]))
        epsg = georef.get("crs_epsg")
        rec = pycolmap.Reconstruction(str(model_dir))

        xyz = np.array([p.xyz for p in rec.points3D.values()])
        if len(xyz) < 3:
            raise RuntimeError("too few points for a coverage map")
        z0 = float(np.median(xyz[:, 2]))
        minx, miny = xyz[:, 0].min(), xyz[:, 1].min()
        maxx, maxy = xyz[:, 0].max(), xyz[:, 1].max()
        res = float(ctx.params["resolution_m"])
        w = max(1, int(np.ceil((maxx - minx) / res)) + 1)
        h = max(1, int(np.ceil((maxy - miny) / res)) + 1)

        footprints = []
        for image_id in rec.reg_image_ids():
            img = rec.image(image_id)
            cam = rec.camera(img.camera_id)
            k = np.asarray(cam.calibration_matrix())
            r_w2c = np.asarray(img.cam_from_world().matrix())[:, :3]
            center = np.asarray(img.projection_center())
            footprints.append(ground_footprint(k, r_w2c, center, cam.width, cam.height, z0))

        count = rasterize_overlap(footprints, minx, maxy, res, w, h)
        covered = count[count > 0]
        min_ov = int(ctx.params["min_overlap"])
        pct_ge = float((covered >= min_ov).mean()) if covered.size else 0.0

        write_geotiff(ctx.artifact_path("coverage.tif"), count.astype(np.uint16),
                      minx + origin[0], maxy + origin[1], res, epsg, nodata=0)
        self._write_png(ctx, count)

        return StageResult(
            artifacts={"coverage": "coverage.tif", "preview": "coverage.png"},
            metrics={
                "max_overlap": int(count.max()),
                "mean_overlap": round(float(covered.mean()), 2) if covered.size else 0.0,
                f"pct_area_ge_{min_ov}": round(pct_ge * 100, 1),
                "georeferenced": epsg is not None,
            },
        )

    def _write_png(self, ctx: RunContext, count: np.ndarray) -> None:
        from PIL import Image

        Image.fromarray(colormap_overlap(count), mode="RGB").save(ctx.artifact_path("coverage.png"))

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        m = result.metrics
        key = f"pct_area_ge_{int(ctx.params['min_overlap'])}"
        pct = m.get(key, 100.0)
        if pct < 70.0:
            return [Issue(Severity.WARNING,
                          f"only {pct}% of the mapped area has >= {ctx.params['min_overlap']}x overlap",
                          hint="add more images / increase flight overlap for reliable reconstruction")]
        return []
