"""Vegetation indices — pipeline stage (agriculture / environment).

Computes vegetation indices from the orthophoto: RGB indices (ExG/VARI/GLI) always, and NIR
indices (NDVI/GNDVI) when a NIR band is available — either as band 4+ of a multispectral ortho
or a separate aligned NIR GeoTIFF (params.nir_file). Each index is written as a float GeoTIFF
(same georeferencing as the ortho) plus a colorized PNG preview.

True multispectral capture (per-band alignment, reflectance-panel calibration) is upstream/
future; this stage is the index-computation + mapping layer and works today on RGB orthos.

Inputs: a stage providing "ortho".
Outputs: <index>.tif + <index>.png per index, indices.json
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openreco import indices as veg
from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.raster import write_geotiff


@register_stage
class Indices(Stage):
    type = "indices"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {"indices": ["exg", "vari"], "nir_file": "", "nir_band": 4}

    def run(self, ctx: RunContext) -> StageResult:
        import rasterio

        ortho_path = ctx.input_artifact(ctx.input_with("ortho"), "ortho")
        with rasterio.open(ortho_path) as ds:
            bands = ds.read().astype(np.float64) / 255.0   # (B,H,W) scaled to ~[0,1]
            transform, crs = ds.transform, ds.crs
            west, north = transform.c, transform.f
            res = transform.a
            nbands = ds.count
        r, g, b = bands[0], bands[1], bands[2]
        nir = self._load_nir(ctx, bands, nbands)

        epsg = crs.to_epsg() if crs else None
        produced, skipped = [], []
        for name in ctx.params["indices"]:
            if name not in veg.REGISTRY:
                skipped.append(f"{name} (unknown)")
                continue
            if veg.REGISTRY[name][1] and nir is None:     # needs NIR but none available
                skipped.append(f"{name} (no NIR)")
                continue
            idx = veg.compute(name, r=r, g=g, b=b, nir=nir).astype(np.float32)
            write_geotiff(ctx.artifact_path(f"{name}.tif"), idx, west, north, res, epsg,
                          nodata=float("nan"))
            self._save_png(ctx, f"{name}.png", veg.colorize(idx))
            produced.append(name)

        ctx.write_json("indices.json", {"produced": produced, "skipped": skipped,
                                        "has_nir": nir is not None, "crs_epsg": epsg})
        artifacts = {"meta": "indices.json"}
        for name in produced:
            artifacts[name] = f"{name}.tif"
            artifacts[f"{name}_png"] = f"{name}.png"
        return StageResult(artifacts=artifacts,
                           metrics={"produced": ",".join(produced) or "none",
                                    "count": len(produced), "has_nir": nir is not None})

    def _load_nir(self, ctx, bands, nbands):
        nir_file = ctx.params.get("nir_file")
        if nir_file:
            import rasterio
            with rasterio.open((ctx.project_dir / nir_file).resolve()) as ds:
                return ds.read(1).astype(np.float64) / 255.0
        band = int(ctx.params.get("nir_band", 4))
        if nbands >= band:                                # multispectral ortho with a NIR band
            return bands[band - 1]
        return None

    def _save_png(self, ctx, name, rgb) -> None:
        from PIL import Image

        Image.fromarray(rgb, "RGB").save(ctx.artifact_path(name))

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        if result.metrics["count"] == 0:
            return [Issue(Severity.WARNING, "no indices produced",
                          hint="check the 'indices' list / NIR availability")]
        note = "" if result.metrics["has_nir"] else " (RGB-only; provide a NIR band for NDVI/GNDVI)"
        return [Issue(Severity.INFO, f"computed: {result.metrics['produced']}{note}")]
