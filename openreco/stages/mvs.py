"""Dense reconstruction (MVS) — pipeline stage 4.

Default: COLMAP PatchMatch stereo (undistort -> patch_match_stereo -> stereo_fusion), which
is CUDA-only. On a CPU-only machine (no CUDA device) we fall back to the SfM/georef sparse
cloud and flag it loudly: the pipeline still completes end-to-end and stays hardware-agnostic,
but density (and thus mesh/DSM quality) is reduced until a GPU is available.

Outputs (in cache dir):
  points.ply     — dense (or sparse-fallback) point cloud, local-frame coords + colors
  points.las     — same cloud in true CRS meters (origin offset), for GIS
  points.json    — {mode, num_points, crs, crs_epsg, origin}
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.pointcloud import points_from_reconstruction, write_las, write_ply

_QUALITY = {
    "low": 1000,
    "medium": 2000,
    "high": 3200,
}


@register_stage
class Mvs(Stage):
    type = "mvs"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {
            "quality": "medium",          # low | medium | high (undistort image size)
            "geometric_consistency": True,
            "allow_sparse_fallback": True,  # if dense (CUDA) unavailable, use sparse cloud
        }

    def run(self, ctx: RunContext) -> StageResult:
        import pycolmap

        model_dir = ctx.input_artifact("georef", "model")
        georef = ctx.read_input_json("georef", "georef")
        images = ctx.read_input_json("ingest", "images")
        image_dir = images["image_dir"]
        rec = pycolmap.Reconstruction(str(model_dir))
        reg_names = [rec.image(i).name for i in rec.reg_image_ids()]

        mode, xyz, rgb = self._dense_or_fallback(pycolmap, ctx, model_dir, image_dir, reg_names, rec)

        origin = np.array(georef.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        crs_epsg = georef.get("crs_epsg")

        write_ply(ctx.artifact_path("points.ply"), xyz, rgb)
        # LAS in true CRS meters: local frame + origin
        try:
            write_las(ctx.artifact_path("points.las"), xyz + origin, rgb, crs_epsg, origin)
            las_ok = True
        except Exception as exc:  # noqa: BLE001
            ctx.logger.warning("LAS export skipped: %r", exc)
            las_ok = False

        ctx.write_json("points.json", {
            "mode": mode,
            "num_points": int(len(xyz)),
            "crs": georef.get("crs", "local"),
            "crs_epsg": crs_epsg,
            "origin": origin.tolist(),
        })

        artifacts = {"points": "points.ply", "meta": "points.json"}
        if las_ok:
            artifacts["las"] = "points.las"
        return StageResult(
            artifacts=artifacts,
            metrics={"mode": mode, "num_points": int(len(xyz))},
        )

    def _dense_or_fallback(self, pycolmap, ctx, model_dir, image_dir, reg_names, rec):
        if not _has_cuda(pycolmap):
            if not ctx.params["allow_sparse_fallback"]:
                raise RuntimeError("dense MVS requires CUDA and sparse fallback is disabled")
            ctx.logger.warning("no CUDA device — using sparse SfM cloud as dense fallback")
            xyz, rgb = points_from_reconstruction(rec)
            return "sparse_fallback", xyz, rgb

        # CUDA path: undistort -> patch_match -> fusion
        ws = ctx.artifact_path("dense")
        ws.mkdir(parents=True, exist_ok=True)
        ctx.progress(0.1, "undistorting images")
        max_size = _QUALITY[ctx.params["quality"]]
        und = pycolmap.UndistortCameraOptions()
        _set(und, "max_image_size", max_size)
        pycolmap.undistort_images(ws, model_dir, image_dir, image_names=reg_names,
                                  undistort_options=und)
        ctx.progress(0.3, "patch-match stereo")
        pm = pycolmap.PatchMatchOptions()
        _set(pm, "geom_consistency", bool(ctx.params["geometric_consistency"]))
        pycolmap.patch_match_stereo(ws, options=pm)
        ctx.progress(0.8, "stereo fusion")
        fused = pycolmap.stereo_fusion(ctx.artifact_path("dense") / "fused.ply", ws)
        xyz, rgb = points_from_reconstruction(fused)
        return "dense", xyz, rgb

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        issues = []
        if result.metrics["mode"] == "sparse_fallback":
            issues.append(Issue(
                Severity.WARNING,
                "dense MVS unavailable (no CUDA) — used sparse cloud; mesh/DSM will be coarse",
                hint="run on a CUDA GPU for true dense reconstruction",
            ))
        if result.metrics["num_points"] < 1000:
            issues.append(Issue(Severity.WARNING, f"only {result.metrics['num_points']} points"))
        return issues


def _has_cuda(pycolmap) -> bool:
    try:
        return bool(pycolmap.has_cuda)
    except Exception:  # noqa: BLE001
        return False


def _set(obj: Any, attr: str, value: Any) -> None:
    try:
        if hasattr(obj, attr):
            setattr(obj, attr, value)
    except Exception:  # noqa: BLE001
        pass
