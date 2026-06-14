"""Import / edited mesh — wrap a mesh PLY file as a first-class layer.

The 3D edit tools (select & delete faces) write an edited mesh to <project>/edits/ and add a layer
of this type pointing at it — non-destructive, reproducible (the file is the content), and a normal
mesh layer (viewable, texturable, exportable).

Inputs:  none (reads `path`).
Outputs: mesh.ply.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.pointcloud import read_mesh_ply, write_mesh_ply


@register_stage
class ImportMesh(Stage):
    type = "import_mesh"
    version = "1"
    deterministic = True

    def default_params(self) -> dict[str, Any]:
        return {"path": ""}

    def run(self, ctx: RunContext) -> StageResult:
        raw = ctx.params["path"]
        src = Path(raw) if Path(raw).is_absolute() else (ctx.project_dir / raw)
        if not src.is_file():
            raise FileNotFoundError(f"import_mesh path not found: {src}")
        verts, faces, vcols = read_mesh_ply(src)
        write_mesh_ply(ctx.artifact_path("mesh.ply"), verts, faces, vcols)
        return StageResult(artifacts={"mesh": "mesh.ply"},
                           metrics={"vertices": int(len(verts)), "faces": int(len(faces))})

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        return [Issue(Severity.INFO, f"imported mesh: {result.metrics['faces']:,} faces")]
