"""Clean / filter — remove noise from a dense point cloud or a mesh.

Point clouds: statistical outlier removal (SOR) — drop points whose mean distance to their k
nearest neighbours exceeds the global mean + std_ratio·std (the standard the reference tool/PCL filter).
Meshes: connected-component filtering — drop small floating islands, keeping components with at
least `min_component_ratio` of the largest component's face count.

`mode=auto` picks points vs mesh from the wired input. Composable via role-based inputs.

Inputs:  a layer providing "points" (+ "meta") or "mesh".
Outputs: cleaned points.ply (+ points.json) or mesh.ply, plus clean.json (what was removed).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.pointcloud import read_mesh_ply, read_ply, write_mesh_ply, write_ply


@register_stage
class Clean(Stage):
    type = "clean"
    version = "1"
    deterministic = True

    def default_params(self) -> dict[str, Any]:
        return {
            "mode": "auto",               # auto | points | mesh
            "knn": 16,                    # SOR neighbours (points)
            "std_ratio": 2.0,             # SOR threshold = mean + std_ratio*std (points)
            "min_component_ratio": 0.02,  # mesh: keep components >= this fraction of the largest
        }

    def run(self, ctx: RunContext) -> StageResult:
        mode = ctx.params["mode"]
        mesh_dep = ctx.find_input("mesh")
        if mode == "mesh" or (mode == "auto" and mesh_dep):
            return self._clean_mesh(ctx, mesh_dep or ctx.input_with("mesh"))
        return self._clean_points(ctx, ctx.input_with("points"))

    # ---- point cloud: statistical outlier removal --------------------------
    def _clean_points(self, ctx, dep) -> StageResult:
        from scipy.spatial import cKDTree

        xyz, rgb, normals = read_ply(ctx.input_artifact(dep, "points"))
        meta = self._meta(ctx, dep)
        n0 = len(xyz)
        if n0 < 50:
            raise RuntimeError(f"too few points to clean ({n0})")
        k = max(2, int(ctx.params["knn"]))
        ctx.progress(0.3, f"statistical outlier removal (k={k})")
        tree = cKDTree(xyz)
        d, _ = tree.query(xyz, k=k + 1)            # +1: first neighbour is the point itself
        mean_d = d[:, 1:].mean(axis=1)
        thr = float(mean_d.mean() + float(ctx.params["std_ratio"]) * mean_d.std())
        keep = mean_d <= thr
        xyz = xyz[keep]
        rgb = rgb[keep] if rgb is not None else None
        normals = normals[keep] if normals is not None else None

        write_ply(ctx.artifact_path("points.ply"), xyz, rgb, normals)
        ctx.write_json("points.json", {"mode": "cleaned", "num_points": int(len(xyz)),
                                       "crs": meta.get("crs", "local"), "crs_epsg": meta.get("crs_epsg"),
                                       "origin": meta.get("origin", [0.0, 0.0, 0.0])})
        ctx.write_json("clean.json", {"kind": "points", "in": int(n0), "out": int(len(xyz)),
                                      "removed": int(n0 - len(xyz)), "threshold_m": round(thr, 4)})
        return StageResult(
            artifacts={"points": "points.ply", "meta": "points.json", "report": "clean.json"},
            metrics={"kind": "points", "in": int(n0), "out": int(len(xyz)),
                     "removed_pct": round(100.0 * (n0 - len(xyz)) / n0, 2)})

    # ---- mesh: connected-component filtering --------------------------------
    def _clean_mesh(self, ctx, dep) -> StageResult:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        verts, faces, vcols = read_mesh_ply(ctx.input_artifact(dep, "mesh"))
        nf0 = len(faces)
        if nf0 < 4:
            raise RuntimeError(f"too few faces to clean ({nf0})")
        ctx.progress(0.3, "labelling connected components")
        # vertex graph from mesh edges -> components, then label faces by their vertices
        e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(verts), len(verts)))
        ncomp, labels = connected_components(g + g.T, directed=False)
        face_lab = labels[faces[:, 0]]
        counts = np.bincount(face_lab, minlength=ncomp)
        keep_comp = counts >= max(1, counts.max() * float(ctx.params["min_component_ratio"]))
        keep = keep_comp[face_lab]
        faces = faces[keep]

        # drop now-unused vertices and reindex
        used = np.unique(faces)
        remap = np.full(len(verts), -1, np.int64)
        remap[used] = np.arange(len(used))
        verts2 = verts[used]
        vcols2 = vcols[used] if vcols is not None else None
        faces2 = remap[faces]

        write_mesh_ply(ctx.artifact_path("mesh.ply"), verts2, faces2, vcols2)
        ctx.write_json("clean.json", {"kind": "mesh", "components": int(ncomp),
                                      "kept_components": int(keep_comp.sum()),
                                      "faces_in": int(nf0), "faces_out": int(len(faces2))})
        return StageResult(
            artifacts={"mesh": "mesh.ply", "report": "clean.json"},
            metrics={"kind": "mesh", "faces_in": int(nf0), "faces_out": int(len(faces2)),
                     "removed_pct": round(100.0 * (nf0 - len(faces2)) / nf0, 2),
                     "components": int(ncomp)})

    def _meta(self, ctx, dep) -> dict:
        try:
            return ctx.read_input_json(dep, "meta")
        except Exception:  # noqa: BLE001
            return {}

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        m = result.metrics
        return [Issue(Severity.INFO, f"cleaned {m['kind']}: removed {m['removed_pct']}%")]
