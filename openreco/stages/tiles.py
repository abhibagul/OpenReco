"""Tiled model — split a large mesh into streamable 3D Tiles.

Partitions the mesh into an N×N grid of XY tiles, writes each as a glTF, and emits a multi-tile
3D Tiles tileset.json (georeferenced when a CRS is present) so a client (Cesium) streams only the
visible tiles instead of one giant model. Output is a self-contained `tiles/` directory.

Inputs: a stage providing "mesh" + (mvs for CRS/origin).
Outputs: tiles/ (tileset.json + tile_*.glb), tiles.json
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.gltf import write_glb
from openreco.io.pointcloud import read_mesh_ply
from openreco.io.tiles3d import write_tiled_tileset
from openreco.tiling import tile_mesh


@register_stage
class Tiles(Stage):
    type = "tiles"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {"grid": 4}

    def run(self, ctx: RunContext) -> StageResult:
        verts, faces, vcols = read_mesh_ply(ctx.input_artifact(ctx.input_with("mesh"), "mesh"))
        if len(faces) == 0:
            raise RuntimeError("mesh has no faces to tile")
        origin = np.array([0.0, 0.0, 0.0])
        epsg = None
        try:                                            # optional CRS/origin from the dense cloud
            meta = ctx.read_input_json(ctx.input_with("points"), "meta")
            origin = np.array(meta.get("origin", [0.0, 0.0, 0.0]))
            epsg = meta.get("crs_epsg")
        except KeyError:
            pass

        grid = int(ctx.params["grid"])
        tiles = tile_mesh(verts, faces, vcols, grid)
        out_dir = ctx.artifact_path("tiles")
        out_dir.mkdir(parents=True, exist_ok=True)

        children = []
        for t in tiles:
            uri = f"tile_{t['gx']}_{t['gy']}.glb"
            write_glb(out_dir / uri, t["verts"], t["faces"], t["vcolors"])
            children.append({"uri": uri, "bbox_min": t["bbox_min"], "bbox_max": t["bbox_max"]})
        latlon = write_tiled_tileset(out_dir / "tileset.json", children, epsg, origin,
                                     verts.min(axis=0), verts.max(axis=0))
        if latlon:
            ctx.logger.info("tiled model placed at lat=%.5f lon=%.5f", *latlon)

        ctx.write_json("tiles.json", {"tiles": len(children), "grid": grid,
                                      "total_faces": int(len(faces)), "georeferenced": epsg is not None})
        return StageResult(
            artifacts={"tiles": "tiles", "tileset": "tiles/tileset.json", "meta": "tiles.json"},
            metrics={"tiles": len(children), "grid": grid, "total_faces": int(len(faces)),
                     "georeferenced": epsg is not None},
        )

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        m = result.metrics
        return [Issue(Severity.INFO, f"{m['total_faces']:,} faces -> {m['tiles']} streamable tiles "
                      f"({m['grid']}x{m['grid']} grid)")]
