"""LiDAR / external point-cloud fusion — pipeline stage.

Imports an external point cloud (LAS/LAZ/PLY — e.g. a terrestrial or airborne LiDAR scan),
co-registers it onto the photogrammetric dense cloud with ICP, and merges the two into a single
cloud. Lets survey-grade LiDAR geometry and photogrammetric coverage/colour reinforce each other.

Inputs: a stage providing "points" (mvs dense cloud, the registration reference).
Params: external_cloud (path, relative to project); init (centroid|none); max_corr_dist.
Outputs: fused.las (dense + registered external, true CRS), fuse.json (transform + RMS/fitness).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.pointcloud import read_ply, write_las
from openreco.register_cloud import apply_transform, icp


@register_stage
class Fuse(Stage):
    type = "fuse"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {"external_cloud": "", "init": "centroid", "max_corr_dist_m": 0.0,
                "max_iter": 40}

    def run(self, ctx: RunContext) -> StageResult:
        ext_param = ctx.params["external_cloud"]
        if not ext_param:
            raise ValueError("fuse requires params.external_cloud (path to a LAS/LAZ/PLY cloud)")

        dst, dst_rgb, _ = read_ply(ctx.input_artifact(ctx.input_with("points"), "points"))
        meta = ctx.read_input_json(ctx.input_with("points"), "meta")
        origin = np.array(meta.get("origin", [0.0, 0.0, 0.0]))
        epsg = meta.get("crs_epsg")

        src, src_rgb = self._read_cloud((ctx.project_dir / ext_param).resolve())
        # external clouds are typically in world coords; bring into the dense LOCAL frame
        src_local = src - origin if epsg else src.copy()

        init = None
        if ctx.params["init"] == "centroid":
            init = (np.eye(3), dst.mean(0) - src_local.mean(0))   # rough translation align
        ctx.progress(0.3, f"ICP registering {len(src):,} external pts to {len(dst):,} dense")
        max_corr = float(ctx.params["max_corr_dist_m"]) or None
        reg = icp(src_local, dst, max_iter=int(ctx.params["max_iter"]),
                  max_corr_dist=max_corr, init=init)

        src_reg = apply_transform(src_local, reg["R"], reg["t"])
        merged = np.vstack([dst, src_reg])
        if dst_rgb is None:
            dst_rgb = np.full((len(dst), 3), 200, np.uint8)
        if src_rgb is None:
            src_rgb = np.full((len(src_reg), 3), 120, np.uint8)
        merged_rgb = np.vstack([dst_rgb, src_rgb])

        las_ok = self._write_las(ctx, merged + origin, merged_rgb, epsg, origin)
        info = {"rmse_m": round(reg["rmse"], 5), "fitness": round(reg["fitness"], 4),
                "iterations": reg["iterations"], "n_dense": int(len(dst)),
                "n_external": int(len(src)), "transform": _t_to_list(reg["R"], reg["t"])}
        ctx.write_json("fuse.json", info)
        artifacts = {"meta": "fuse.json"}
        if las_ok:
            artifacts["fused"] = "fused.las"
        return StageResult(artifacts=artifacts, metrics={
            "n_dense": int(len(dst)), "n_external": int(len(src)),
            "rmse_m": round(reg["rmse"], 5), "fitness": round(reg["fitness"], 4)})

    def _read_cloud(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"external_cloud not found: {path}")
        if path.suffix.lower() in (".las", ".laz"):
            import laspy

            las = laspy.read(str(path))
            xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
            rgb = None
            if hasattr(las, "red"):
                rgb = np.column_stack([las.red, las.green, las.blue]) // 257
                rgb = rgb.astype(np.uint8)
            return xyz.astype(np.float64), rgb
        xyz, rgb, _ = read_ply(path)
        return xyz, rgb

    def _write_las(self, ctx, xyz, rgb, epsg, origin) -> bool:
        try:
            write_las(ctx.artifact_path("fused.las"), xyz, rgb, epsg, origin)
            return True
        except Exception as exc:  # noqa: BLE001
            ctx.logger.warning("fused LAS export skipped: %r", exc)
            return False

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        m = result.metrics
        issues = [Issue(Severity.INFO, f"fused {m['n_external']:,} external + {m['n_dense']:,} dense "
                        f"pts; ICP RMS {m['rmse_m']} m, fitness {m['fitness']}")]
        if m["fitness"] < 0.3:
            issues.append(Issue(Severity.WARNING, "low ICP fitness — clouds may not overlap / be "
                                "mis-scaled", hint="ensure the external cloud overlaps the scene"))
        return issues


def _t_to_list(r, t):
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m.flatten().tolist()
