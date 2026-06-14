"""Contour lines from the DSM — pipeline QA/derived product.

Extracts topographic iso-lines from the DSM at a fixed elevation interval (marching squares),
maps them through the raster's affine transform to world coordinates, reprojects to WGS84, and
writes a GeoJSON FeatureCollection (one MultiLineString per elevation). Contours are a standard
survey deliverable and drop straight into QGIS / web maps.

(v1 emits per-cell segments grouped by level; chaining them into long polylines is a future
polish. Quality follows the DSM — coarse on the CPU sparse-fallback cloud.)
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.geo.contour import contour_levels, contour_segments


@register_stage
class Contours(Stage):
    type = "contours"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {"interval_m": 10.0}

    def run(self, ctx: RunContext) -> StageResult:
        import rasterio

        with rasterio.open(ctx.input_artifact(ctx.input_with("dsm"), "dsm")) as ds:
            z = ds.read(1).astype(np.float64)
            transform = ds.transform
            crs = ds.crs
            nodata = ds.nodata
        if nodata is not None and not np.isnan(nodata):
            z[z == nodata] = np.nan

        finite = z[np.isfinite(z)]
        if finite.size == 0:
            raise RuntimeError("DSM has no valid elevations")
        interval = float(ctx.params["interval_m"])
        levels = contour_levels(float(finite.min()), float(finite.max()), interval)

        # reproject world (CRS) -> WGS84 lon/lat for spec-compliant GeoJSON, if georeferenced
        to_wgs84 = None
        if crs is not None:
            from pyproj import Transformer
            to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        features = []
        total_segments = 0
        for level in levels:
            segs = contour_segments(z, level)
            if not segs:
                continue
            lines = []
            for (c0, r0), (c1, r1) in segs:
                x0, y0 = transform * (c0, r0)
                x1, y1 = transform * (c1, r1)
                if to_wgs84 is not None:
                    x0, y0 = to_wgs84.transform(x0, y0)
                    x1, y1 = to_wgs84.transform(x1, y1)
                lines.append([[x0, y0], [x1, y1]])
            total_segments += len(lines)
            features.append({
                "type": "Feature",
                "properties": {"elevation": round(level, 3)},
                "geometry": {"type": "MultiLineString", "coordinates": lines},
            })

        fc: dict[str, Any] = {"type": "FeatureCollection", "features": features}
        if to_wgs84 is None:  # local frame -> annotate non-WGS84 coordinates
            fc["properties"] = {"crs": "local"}
        ctx.artifact_path("contours.geojson").write_text(json.dumps(fc), encoding="utf-8")
        ctx.write_json("contours.json", {
            "interval_m": interval, "num_levels": len(features),
            "num_segments": total_segments,
            "z_min": float(finite.min()), "z_max": float(finite.max()),
            "wgs84": to_wgs84 is not None,
        })
        return StageResult(
            artifacts={"contours": "contours.geojson", "meta": "contours.json"},
            metrics={"levels": len(features), "segments": total_segments, "interval_m": interval},
        )

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        if result.metrics["levels"] == 0:
            return [Issue(Severity.WARNING, "no contour levels — DSM elevation range < interval",
                          hint="lower contours.interval_m")]
        return []
