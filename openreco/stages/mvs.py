"""Dense reconstruction (MVS) — pipeline stage 4.

Real dense reconstruction via COLMAP PatchMatch stereo (undistort -> patch_match_stereo ->
stereo_fusion). PatchMatch is CUDA-only, and the PyPI pycolmap wheel is CPU-only, so we drive a
CUDA-enabled COLMAP **binary** (see openreco/compute.py) for the dense steps. If no GPU/binary is
available — or dense fails (e.g. out of VRAM) — we fall back to the SfM/georef sparse cloud and
flag it, so the pipeline always completes.

Outputs (in cache dir):
  points.ply  — dense (or sparse-fallback) cloud, local-frame coords + colors
  points.las  — same cloud in true CRS meters (origin offset), for GIS
  points.json — {mode, num_points, crs, crs_epsg, origin}
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from openreco import compute
from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.pointcloud import points_from_reconstruction, read_ply, write_las, write_ply

_QUALITY = {"low": 1000, "medium": 1600, "high": 2400}  # undistort/patch-match max image size


@register_stage
class Mvs(Stage):
    type = "mvs"
    version = "3"  # v3: dense cloud keeps COLMAP MVS normals (for robust Poisson meshing)
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {
            "quality": "medium",            # low | medium | high (image size for dense)
            "geometric_consistency": True,
            "dense_backend": "auto",        # auto | cuda | sparse
            "cache_size_gb": 8,             # PatchMatch RAM cache
            "gpu_index": 0,
            "allow_sparse_fallback": True,
        }

    def run(self, ctx: RunContext) -> StageResult:
        import pycolmap

        model_dir = ctx.input_artifact(ctx.input_with("model"), "model")
        georef = ctx.read_input_json("georef", "georef")
        images = ctx.read_input_json(ctx.input_with("images"), "images")
        image_dir = images["image_dir"]
        rec = pycolmap.Reconstruction(str(model_dir))

        mode, xyz, rgb, normals = self._dense_or_fallback(ctx, model_dir, image_dir, rec)

        origin = np.array(georef.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        crs_epsg = georef.get("crs_epsg")
        write_ply(ctx.artifact_path("points.ply"), xyz, rgb, normals)  # normals (dense) feed meshing
        las_ok = self._try_las(ctx, xyz, rgb, crs_epsg, origin)

        ctx.write_json("points.json", {
            "mode": mode, "num_points": int(len(xyz)),
            "crs": georef.get("crs", "local"), "crs_epsg": crs_epsg, "origin": origin.tolist(),
        })
        artifacts = {"points": "points.ply", "meta": "points.json"}
        if las_ok:
            artifacts["las"] = "points.las"
        return StageResult(artifacts=artifacts, metrics={"mode": mode, "num_points": int(len(xyz))})

    # ---- dense -------------------------------------------------------------------------
    def _dense_or_fallback(self, ctx, model_dir, image_dir, rec):
        backend = ctx.params["dense_backend"]
        want_cuda = backend in ("auto", "cuda")
        if want_cuda and compute.gpu_dense_available():
            try:
                xyz, rgb, normals = self._cuda_dense(ctx, model_dir, image_dir)
                return "dense", xyz, rgb, normals
            except Exception as exc:  # noqa: BLE001
                if backend == "cuda" or not ctx.params["allow_sparse_fallback"]:
                    raise
                ctx.logger.warning("GPU dense failed (%r) — falling back to sparse cloud", exc)
        elif backend == "cuda":
            raise RuntimeError("dense_backend=cuda but no CUDA GPU + COLMAP binary found "
                               "(set OPENRECO_COLMAP or install the CUDA build)")
        if not ctx.params["allow_sparse_fallback"]:
            raise RuntimeError("dense MVS unavailable and sparse fallback disabled")
        ctx.logger.warning("no GPU dense path — using sparse SfM cloud as fallback")
        xyz, rgb = points_from_reconstruction(rec)
        return "sparse_fallback", xyz, rgb, None

    def _cuda_dense(self, ctx, model_dir: Path, image_dir: str):
        colmap = str(compute.find_colmap())
        ws = ctx.artifact_path("dense")
        ws.mkdir(parents=True, exist_ok=True)
        max_size = _QUALITY[ctx.params["quality"]]
        geom = bool(ctx.params["geometric_consistency"])

        ctx.logger.info("GPU dense via %s (max_image_size=%d)", colmap, max_size)
        ctx.progress(0.1, "undistorting images")
        self._colmap(ctx, colmap, [
            "image_undistorter", "--image_path", str(image_dir), "--input_path", str(model_dir),
            "--output_path", str(ws), "--output_type", "COLMAP", "--max_image_size", str(max_size),
        ])
        ctx.progress(0.3, "patch-match stereo (GPU)")
        self._colmap(ctx, colmap, [
            "patch_match_stereo", "--workspace_path", str(ws), "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true" if geom else "false",
            "--PatchMatchStereo.max_image_size", str(max_size),
            "--PatchMatchStereo.cache_size", str(int(ctx.params["cache_size_gb"])),
            "--PatchMatchStereo.gpu_index", str(int(ctx.params["gpu_index"])),
        ])
        ctx.progress(0.8, "stereo fusion")
        fused = ws / "fused.ply"
        self._colmap(ctx, colmap, [
            "stereo_fusion", "--workspace_path", str(ws), "--workspace_format", "COLMAP",
            "--input_type", "geometric" if geom else "photometric", "--output_path", str(fused),
        ])
        if not fused.exists():
            raise RuntimeError("stereo_fusion produced no fused.ply")
        xyz, rgb, normals = read_ply(fused)
        if rgb is None:
            rgb = np.full((len(xyz), 3), 200, np.uint8)
        return xyz, rgb, normals

    def _colmap(self, ctx, colmap: str, args: list[str]) -> None:
        proc = subprocess.run([colmap, *args], capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-800:]
            raise RuntimeError(f"colmap {args[0]} failed (rc={proc.returncode}):\n{tail}")

    def _try_las(self, ctx, xyz, rgb, crs_epsg, origin) -> bool:
        try:
            write_las(ctx.artifact_path("points.las"), xyz + origin, rgb, crs_epsg, origin)
            return True
        except Exception as exc:  # noqa: BLE001
            ctx.logger.warning("LAS export skipped: %r", exc)
            return False

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        issues = []
        if result.metrics["mode"] == "sparse_fallback":
            issues.append(Issue(Severity.WARNING,
                "dense MVS fell back to the sparse cloud — mesh/DSM will be coarse",
                hint="ensure a CUDA GPU + COLMAP binary (OPENRECO_COLMAP) for true dense"))
        elif result.metrics["mode"] == "dense":
            issues.append(Issue(Severity.INFO,
                f"GPU dense reconstruction: {result.metrics['num_points']:,} points"))
        if result.metrics["num_points"] < 1000:
            issues.append(Issue(Severity.WARNING, f"only {result.metrics['num_points']} points"))
        return issues
