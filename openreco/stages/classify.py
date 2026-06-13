"""Point-cloud classification + true DTM — pipeline stage.

Classifies the dense cloud into ground / non-ground (LAS codes 2 / 1), writes a classified LAS,
and rasterises the ground points into a bare-earth DTM (a truer DTM than the morphological
DSM-based one, since it uses actual ground returns). A core the reference tool capability.

Inputs: a stage providing "points" (mvs) + "model"/georef for CRS & origin.
Outputs: classified.las, dtm_ground.tif, classify.json
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openreco.classify_ground import GROUND, classify_ground
from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.pointcloud import read_ply, write_las
from openreco.io.raster import grid_topdown, write_geotiff


@register_stage
class Classify(Stage):
    type = "classify"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {"cell_m": 5.0, "ground_threshold_m": 0.5, "dtm_resolution_m": 1.0}

    def run(self, ctx: RunContext) -> StageResult:
        xyz, rgb, _ = read_ply(ctx.input_artifact(ctx.input_with("points"), "points"))
        meta = ctx.read_input_json(ctx.input_with("points"), "meta")
        origin = np.array(meta.get("origin", [0.0, 0.0, 0.0]))
        epsg = meta.get("crs_epsg")
        if rgb is None:
            rgb = np.full((len(xyz), 3), 200, np.uint8)

        ctx.progress(0.3, "classifying ground / non-ground")
        cls = classify_ground(xyz, float(ctx.params["cell_m"]), float(ctx.params["ground_threshold_m"]))
        n_ground = int((cls == GROUND).sum())

        # classified LAS (true CRS coords); LAS classification field
        try:
            write_las(ctx.artifact_path("classified.las"), xyz + origin, rgb, epsg, origin,
                      classification=cls)
            las_ok = True
        except Exception as exc:  # noqa: BLE001
            ctx.logger.warning("classified LAS export skipped: %r", exc)
            las_ok = False

        # bare-earth DTM from ground points
        ground = xyz[cls == GROUND]
        res = float(ctx.params["dtm_resolution_m"])
        artifacts = {"classified_meta": "classify.json"}
        if len(ground) >= 4:
            gz = ground[:, 2] + origin[2]
            dtm, _o, west, north = grid_topdown(ground, gz, np.zeros((len(ground), 3), np.uint8), res)
            write_geotiff(ctx.artifact_path("dtm_ground.tif"), dtm, west + origin[0], north + origin[1],
                          res, epsg, nodata=float("nan"))
            artifacts["dtm"] = "dtm_ground.tif"
        if las_ok:
            artifacts["las"] = "classified.las"

        ctx.write_json("classify.json", {
            "total": int(len(xyz)), "ground": n_ground, "non_ground": int(len(xyz) - n_ground),
            "ground_pct": round(100.0 * n_ground / max(1, len(xyz)), 1),
            "cell_m": ctx.params["cell_m"], "ground_threshold_m": ctx.params["ground_threshold_m"],
        })
        return StageResult(
            artifacts=artifacts,
            metrics={"total": int(len(xyz)), "ground": n_ground,
                     "ground_pct": round(100.0 * n_ground / max(1, len(xyz)), 1)},
        )

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        pct = result.metrics["ground_pct"]
        if pct == 0 or pct == 100:
            return [Issue(Severity.WARNING, f"{pct}% classified as ground — check cell_m / threshold",
                          hint="cell_m should exceed the largest building; threshold ~0.3-1.0 m")]
        return [Issue(Severity.INFO, f"{result.metrics['ground']:,} ground / "
                      f"{result.metrics['total'] - result.metrics['ground']:,} non-ground ({pct}% ground)")]
