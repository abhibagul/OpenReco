"""Align & merge chunks — combine separate reconstructions into one cloud (ICP).

industry-standard "Align Chunks" + "Merge Chunks": takes the point clouds of two or more chunks
(each an independent reconstruction), registers every other chunk onto the first (reference) with
ICP, and outputs a single merged cloud + per-chunk alignment report. The merged cloud is itself a
"points" layer, so it can feed meshing/DSM/etc. downstream. Reuses openreco.register_cloud.

Inputs: 2+ stages each providing "points" (e.g. each chunk's dense cloud / sparse cloud).
Outputs: merged.ply, merged.las, merge.json (per-chunk RMS/fitness/transform).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.pointcloud import read_ply, write_las, write_ply
from openreco.register_cloud import apply_transform, icp


@register_stage
class MergeChunks(Stage):
    type = "merge_chunks"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {"init": "centroid", "max_iter": 40, "max_corr_dist_m": 0.0}

    def run(self, ctx: RunContext) -> StageResult:
        clouds = [(dep, res) for dep, res in ctx.inputs.items() if "points" in res.artifacts]
        if len(clouds) < 2:
            raise ValueError("merge_chunks needs >= 2 inputs each providing a point cloud (chunks)")

        ref_dep = clouds[0][0]
        ref_xyz, ref_rgb, _ = read_ply(ctx.input_artifact(ref_dep, "points"))
        meta = self._meta(ctx, ref_dep)
        origin = np.array(meta.get("origin", [0.0, 0.0, 0.0]))
        epsg = meta.get("crs_epsg")

        merged_xyz = [ref_xyz]
        merged_rgb = [ref_rgb if ref_rgb is not None else np.full((len(ref_xyz), 3), 200, np.uint8)]
        report = [{"chunk": ref_dep, "role": "reference", "points": int(len(ref_xyz))}]
        max_corr = float(ctx.params["max_corr_dist_m"]) or None

        for dep, _res in clouds[1:]:
            xyz, rgb, _ = read_ply(ctx.input_artifact(dep, "points"))
            init = ((np.eye(3), ref_xyz.mean(0) - xyz.mean(0))
                    if ctx.params["init"] == "centroid" else None)
            ctx.progress(0.4, f"aligning chunk {dep} -> {ref_dep}")
            reg = icp(xyz, ref_xyz, max_iter=int(ctx.params["max_iter"]),
                      max_corr_dist=max_corr, init=init)
            merged_xyz.append(apply_transform(xyz, reg["R"], reg["t"]))
            merged_rgb.append(rgb if rgb is not None else np.full((len(xyz), 3), 150, np.uint8))
            report.append({"chunk": dep, "role": "aligned", "points": int(len(xyz)),
                           "rmse_m": round(reg["rmse"], 5), "fitness": round(reg["fitness"], 4)})

        xyz = np.vstack(merged_xyz)
        rgb = np.vstack(merged_rgb)
        write_ply(ctx.artifact_path("merged.ply"), xyz, rgb)
        las_ok = self._try_las(ctx, xyz + origin, rgb, epsg, origin)

        ctx.write_json("merge.json", {"chunks": len(clouds), "total_points": int(len(xyz)),
                                      "crs_epsg": epsg, "origin": origin.tolist(), "alignment": report})
        # also expose as a "points"/"meta" layer so meshing/DSM can consume the merged cloud
        ctx.write_json("points.json", {"mode": "merged", "num_points": int(len(xyz)),
                                       "crs": meta.get("crs", "local"), "crs_epsg": epsg,
                                       "origin": origin.tolist()})
        artifacts = {"merged": "merged.ply", "points": "merged.ply", "meta": "points.json",
                     "report": "merge.json"}
        if las_ok:
            artifacts["las"] = "merged.las"
        return StageResult(artifacts=artifacts,
                           metrics={"chunks": len(clouds), "total_points": int(len(xyz)),
                                    "mean_rmse_m": round(float(np.mean(
                                        [r["rmse_m"] for r in report if "rmse_m" in r] or [0])), 5)})

    def _meta(self, ctx, dep):
        res = ctx.inputs[dep]
        if "meta" in res.artifacts:
            try:
                return ctx.read_input_json(dep, "meta")
            except Exception:  # noqa: BLE001
                pass
        return {}

    def _try_las(self, ctx, xyz_world, rgb, epsg, origin) -> bool:
        try:
            write_las(ctx.artifact_path("merged.las"), xyz_world, rgb, epsg, origin)
            return True
        except Exception as exc:  # noqa: BLE001
            ctx.logger.warning("merged LAS export skipped: %r", exc)
            return False

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        m = result.metrics
        return [Issue(Severity.INFO, f"merged {m['chunks']} chunks -> {m['total_points']:,} points "
                      f"(mean align RMS {m['mean_rmse_m']} m)")]
