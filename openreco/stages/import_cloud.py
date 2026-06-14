"""Import / edited point cloud — wrap a PLY file as a first-class layer.

Used by the 3D edit tools (select & delete): the edited cloud is written to <project>/edits/ and a
layer of this type points at it, so it behaves like any other point cloud (viewable, meshable,
exportable) while keeping the pipeline reproducible (the file is the content). Also handy for
importing an external scan.

Inputs:  none (reads `path`).
Outputs: points.ply (+ points.json meta).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.pointcloud import read_ply, write_las, write_ply


@register_stage
class ImportCloud(Stage):
    type = "import_cloud"
    version = "1"
    deterministic = True

    def default_params(self) -> dict[str, Any]:
        return {"path": "", "crs_epsg": 0, "origin": [0.0, 0.0, 0.0]}

    def run(self, ctx: RunContext) -> StageResult:
        raw = ctx.params["path"]
        src = Path(raw) if Path(raw).is_absolute() else (ctx.project_dir / raw)
        if not src.is_file():
            raise FileNotFoundError(f"import_cloud path not found: {src}")
        xyz, rgb, normals = read_ply(src)
        write_ply(ctx.artifact_path("points.ply"), xyz, rgb, normals)
        epsg = int(ctx.params["crs_epsg"]) or None
        origin = np.array(ctx.params["origin"], dtype=np.float64)
        ctx.write_json("points.json", {"mode": "imported", "num_points": int(len(xyz)),
                                       "crs": f"EPSG:{epsg}" if epsg else "local",
                                       "crs_epsg": epsg, "origin": origin.tolist()})
        artifacts = {"points": "points.ply", "meta": "points.json"}
        try:
            write_las(ctx.artifact_path("points.las"), xyz + origin, rgb, epsg, origin)
            artifacts["las"] = "points.las"
        except Exception as exc:  # noqa: BLE001
            ctx.logger.warning("LAS export skipped: %r", exc)
        return StageResult(artifacts=artifacts, metrics={"num_points": int(len(xyz))})

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        return [Issue(Severity.INFO, f"imported {result.metrics['num_points']:,} points")]
