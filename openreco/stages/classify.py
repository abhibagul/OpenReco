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

from openreco.classify_ground import BUILDING, GROUND, VEGETATION, classify_points
from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.pointcloud import read_ply, write_las, write_ply
from openreco.io.raster import grid_topdown, write_geotiff


@register_stage
class Classify(Stage):
    type = "classify"
    version = "3"  # v3: also emit a class-coloured PLY (+ meta) for the viewer
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {"cell_m": 5.0, "ground_threshold_m": 0.5, "dtm_resolution_m": 1.0,
                "knn": 12, "planarity_threshold": 0.04}

    def run(self, ctx: RunContext) -> StageResult:
        xyz, rgb, _ = read_ply(ctx.input_artifact(ctx.input_with("points"), "points"))
        meta = ctx.read_input_json(ctx.input_with("points"), "meta")
        origin = np.array(meta.get("origin", [0.0, 0.0, 0.0]))
        epsg = meta.get("crs_epsg")
        if rgb is None:
            rgb = np.full((len(xyz), 3), 200, np.uint8)

        ctx.progress(0.3, "classifying ground / building / vegetation")
        cls = classify_points(xyz, float(ctx.params["cell_m"]),
                              float(ctx.params["ground_threshold_m"]),
                              int(ctx.params["knn"]), float(ctx.params["planarity_threshold"]))
        counts = {"ground": int((cls == GROUND).sum()), "building": int((cls == BUILDING).sum()),
                  "vegetation": int((cls == VEGETATION).sum())}
        n_ground = counts["ground"]

        # class-coloured point cloud (so the UI can show the classification) + meta for downstream
        ccol = np.full((len(xyz), 3), 170, np.uint8)            # default grey = unclassified
        for code, col in ((GROUND, (160, 120, 80)), (BUILDING, (230, 90, 80)), (VEGETATION, (90, 190, 90))):
            ccol[cls == code] = col
        write_ply(ctx.artifact_path("classified.ply"), xyz, ccol)
        ctx.write_json("points.json", {"mode": "classified", "num_points": int(len(xyz)),
                                       "crs": meta.get("crs", "local"), "crs_epsg": epsg,
                                       "origin": origin.tolist()})

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
        artifacts = {"points": "classified.ply", "meta": "points.json", "classified_meta": "classify.json"}
        if len(ground) >= 4:
            gz = ground[:, 2] + origin[2]
            dtm, _o, west, north = grid_topdown(ground, gz, np.zeros((len(ground), 3), np.uint8), res)
            write_geotiff(ctx.artifact_path("dtm_ground.tif"), dtm, west + origin[0], north + origin[1],
                          res, epsg, nodata=float("nan"))
            artifacts["dtm"] = "dtm_ground.tif"
        if las_ok:
            artifacts["las"] = "classified.las"

        total = int(len(xyz))
        ctx.write_json("classify.json", {
            "total": total, "classes": counts, "non_ground": total - n_ground,
            "ground_pct": round(100.0 * n_ground / max(1, total), 1),
            "cell_m": ctx.params["cell_m"], "ground_threshold_m": ctx.params["ground_threshold_m"],
            "planarity_threshold": ctx.params["planarity_threshold"],
        })
        return StageResult(
            artifacts=artifacts,
            metrics={"total": total, "ground": counts["ground"], "building": counts["building"],
                     "vegetation": counts["vegetation"],
                     "ground_pct": round(100.0 * n_ground / max(1, total), 1)},
        )

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        m = result.metrics
        if m["ground_pct"] in (0, 100):
            return [Issue(Severity.WARNING, f"{m['ground_pct']}% classified as ground — check "
                          "cell_m / threshold", hint="cell_m should exceed the largest building")]
        return [Issue(Severity.INFO, f"ground {m['ground']:,} · building {m['building']:,} · "
                      f"vegetation {m['vegetation']:,} ({m['ground_pct']}% ground)")]
