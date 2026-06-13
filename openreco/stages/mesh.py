"""Surface reconstruction — pipeline stage 5.

Default `delaunay_2_5d`: triangulate the point cloud in the ground (x, y) plane and lift to z.
This is the natural surface for UAV/terrain data, is robust on sparse clouds, and feeds the
DSM directly. Long/steep triangles (spanning gaps or vertical jumps) are dropped so the mesh
doesn't web across holes.

`poisson` (optional): estimate normals (scipy KDTree + PCA) and call pycolmap.poisson_meshing —
better for full 3D objects; heavier and needs decent density.

Outputs:
  mesh.ply   — triangle mesh with vertex colors, local-frame coords
  mesh.json  — {method, num_vertices, num_faces}
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.pointcloud import read_mesh_ply, read_ply_xyzrgb, write_mesh_ply, write_ply


@register_stage
class Mesh(Stage):
    type = "mesh"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {
            "method": "delaunay_2_5d",   # delaunay_2_5d | poisson
            "max_edge_factor": 8.0,      # drop triangles whose longest edge > factor * median
            "poisson_depth": 9,
        }

    def run(self, ctx: RunContext) -> StageResult:
        xyz, rgb = read_ply_xyzrgb(ctx.input_artifact("mvs", "points"))
        if rgb is None:
            rgb = np.full((len(xyz), 3), 200, dtype=np.uint8)
        method = ctx.params["method"]
        if method == "poisson":
            verts, faces, vcols = self._poisson(ctx, xyz, rgb)
        else:
            verts, faces, vcols = self._delaunay_2_5d(ctx, xyz, rgb)

        write_mesh_ply(ctx.artifact_path("mesh.ply"), verts, faces, vcols)
        ctx.write_json("mesh.json", {
            "method": method,
            "num_vertices": int(len(verts)),
            "num_faces": int(len(faces)),
        })
        return StageResult(
            artifacts={"mesh": "mesh.ply", "meta": "mesh.json"},
            metrics={"method": method, "vertices": int(len(verts)), "faces": int(len(faces))},
        )

    def _delaunay_2_5d(self, ctx, xyz, rgb):
        from scipy.spatial import Delaunay

        ctx.progress(0.3, "2.5D Delaunay triangulation")
        tri = Delaunay(xyz[:, :2])
        faces = tri.simplices
        # drop triangles with an overly long edge (gap-spanning) using xy distances
        p = xyz[:, :2]
        e0 = np.linalg.norm(p[faces[:, 0]] - p[faces[:, 1]], axis=1)
        e1 = np.linalg.norm(p[faces[:, 1]] - p[faces[:, 2]], axis=1)
        e2 = np.linalg.norm(p[faces[:, 2]] - p[faces[:, 0]], axis=1)
        longest = np.maximum.reduce([e0, e1, e2])
        med = np.median(longest)
        keep = longest <= ctx.params["max_edge_factor"] * med if med > 0 else np.ones(len(faces), bool)
        faces = faces[keep]
        return xyz, faces, rgb

    def _poisson(self, ctx, xyz, rgb):
        import pycolmap

        ctx.progress(0.2, "estimating normals")
        normals = _estimate_normals(xyz)
        tmp = ctx.artifact_path("_oriented.ply")
        write_ply(tmp, xyz, rgb, normals)
        out = ctx.artifact_path("_poisson.ply")
        opts = pycolmap.PoissonMeshingOptions()
        if hasattr(opts, "depth"):
            opts.depth = int(ctx.params["poisson_depth"])
        ctx.progress(0.5, "poisson meshing")
        pycolmap.poisson_meshing(tmp, out, opts)
        verts, faces, vcols = read_mesh_ply(out)
        return verts, faces, vcols

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        if result.metrics["faces"] == 0:
            return [Issue(Severity.ERROR, "mesh has no faces")]
        return []


def _estimate_normals(xyz: np.ndarray, k: int = 16) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=min(k, len(xyz)))
    normals = np.zeros_like(xyz)
    for i, nb in enumerate(idx):
        pts = xyz[nb] - xyz[nb].mean(axis=0)
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        normals[i] = vt[-1]
    # orient consistently: point toward +z hemisphere (good enough for terrain)
    flip = normals[:, 2] < 0
    normals[flip] = -normals[flip]
    return normals
