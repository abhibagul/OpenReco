"""DTM (Digital Terrain Model) — bare-earth surface, pipeline derived product.

Estimates ground by a grayscale morphological opening of the DSM: erosion then dilation with a
window sized to the largest off-ground object to remove (buildings, trees, vehicles). Positive
features narrower than the window are erased, leaving an approximate bare-earth surface; the
normalized height nDSM = DSM - DTM gives object/canopy heights.

This is the lean, DSM-based approximation (no new dependency — scipy.ndimage). True point-cloud
ground classification (CSF / progressive morphological filter on the 3D cloud) is future work;
on gentle terrain the morphological DTM is a reasonable estimate, but it clips sharp ridges
narrower than the window.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage


def morphological_dtm(dsm: np.ndarray, cells: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Bare-earth estimate via grayscale opening (erosion then dilation) over a `cells`-wide
    window. Removes positive features narrower than the window; NaN/invalid cells (per `mask`)
    are filled with the valid mean before morphology and restored to NaN afterwards."""
    from scipy import ndimage

    if mask is None:
        mask = np.isfinite(dsm)
    filled = dsm.astype(np.float64).copy()
    filled[~mask] = float(dsm[mask].mean())
    dtm = ndimage.grey_opening(filled, size=(cells, cells), mode="nearest")
    dtm[~mask] = np.nan
    return dtm


@register_stage
class Dtm(Stage):
    type = "dtm"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        # window ~ largest off-ground object to remove (meters)
        return {"window_m": 20.0, "write_ndsm": True}

    def run(self, ctx: RunContext) -> StageResult:
        import rasterio

        with rasterio.open(ctx.input_artifact("dsm", "dsm")) as ds:
            dsm = ds.read(1).astype(np.float64)
            res_x, res_y = ds.res
            crs = ds.crs
            transform = ds.transform
            nodata = ds.nodata
        mask = np.isfinite(dsm)
        if nodata is not None and not np.isnan(nodata):
            mask &= dsm != nodata
        if mask.sum() < 4:
            raise RuntimeError("DSM has too few valid cells for a DTM")

        # fill any holes with the cell mean so morphology isn't corrupted, then restore mask
        cells = max(1, int(round(float(ctx.params["window_m"]) / abs(res_x))))
        ctx.progress(0.4, f"morphological opening (window {cells}px)")
        dtm = morphological_dtm(dsm, cells, mask)

        epsg = crs.to_epsg() if crs else None
        self._write(ctx.artifact_path("dtm.tif"), dtm.astype(np.float32), transform, crs, nodata)
        artifacts = {"dtm": "dtm.tif"}
        ndsm_max = None
        if ctx.params["write_ndsm"]:
            ndsm = np.where(mask, dsm - dtm, np.nan).astype(np.float32)
            self._write(ctx.artifact_path("ndsm.tif"), ndsm, transform, crs, nodata)
            artifacts["ndsm"] = "ndsm.tif"
            ndsm_max = round(float(np.nanmax(ndsm)), 3)

        valid = dtm[np.isfinite(dtm)]
        ctx.write_json("dtm.json", {"window_m": ctx.params["window_m"], "crs_epsg": epsg,
                                    "z_min": float(valid.min()), "z_max": float(valid.max()),
                                    "max_object_height_m": ndsm_max})
        return StageResult(
            artifacts=artifacts,
            metrics={"window_m": ctx.params["window_m"], "z_min": round(float(valid.min()), 2),
                     "z_max": round(float(valid.max()), 2), "max_object_height_m": ndsm_max},
        )

    def _write(self, path, arr, transform, crs, nodata) -> None:
        import rasterio

        with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                           count=1, dtype="float32", transform=transform, crs=crs,
                           nodata=nodata if nodata is not None else float("nan"),
                           compress="deflate") as dst:
            dst.write(arr, 1)

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        return [Issue(Severity.INFO, "DTM is a morphological (DSM-based) approximation; "
                      "point-cloud ground classification is future work")]
