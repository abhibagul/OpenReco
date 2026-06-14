"""Structure-from-Motion — pipeline stage 2 (pycolmap, BSD).

Feature extraction (SIFT) -> matching -> incremental mapping with self-calibration and
bundle adjustment. Produces camera poses, refined intrinsics, and a sparse point cloud.
Runs on CPU when no CUDA device is present (slower, but the slice stays hardware-agnostic).

Inputs:  ingest (images.json -> image_dir + kept image names)
Outputs (in cache dir):
  reconstruction/        — COLMAP binary model (cameras/images/points3D); reloadable downstream
  sparse.ply             — sparse point cloud for the viewer
  poses.json             — per-image projection centers + intrinsics + QA metrics
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage

# Determinism note: COLMAP SIFT + incremental mapping are not bit-deterministic across
# threads/runs. We mark this stage non-deterministic so the report/diff compare params,
# not output bytes. (See docs/03 reproducibility section.)


@register_stage
class Sfm(Stage):
    type = "sfm"
    version = "2"  # v2: adds mapper option (incremental | global/GLOMAP)
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {
            "matcher": "exhaustive",   # exhaustive | sequential | spatial(GPS)
            "mapper": "incremental",   # incremental (robust, ordered) | global (GLOMAP: faster, large unordered sets)
            "camera_mode": "auto",     # auto | single | per_folder | per_image
            "max_image_size": 2000,    # downscale long edge for feature extraction
            "max_num_features": 8192,
            "use_gpu": "auto",         # auto | cpu | cuda
        }

    def run(self, ctx: RunContext) -> StageResult:
        import pycolmap

        data = ctx.read_input_json(ctx.input_with("images"), "images")
        image_dir = Path(data["image_dir"])
        names = [im["name"] for im in data["images"] if not im["excluded"]]
        if len(names) < 3:
            raise ValueError(f"SfM needs >=3 usable images, got {len(names)}")

        db_path = ctx.artifact_path("database.db")
        recon_dir = ctx.artifact_path("reconstruction")
        recon_dir.mkdir(parents=True, exist_ok=True)

        device = {
            "auto": pycolmap.Device.auto,
            "cpu": pycolmap.Device.cpu,
            "cuda": pycolmap.Device.cuda,
        }[ctx.params["use_gpu"]]

        gpu = self._extract_and_match(ctx, pycolmap, db_path, image_dir, names, device)
        ctx.logger.info("feature extraction + matching done (%s)", "GPU" if gpu else "CPU")

        mapper = ctx.params["mapper"]
        ctx.progress(0.7, f"{mapper} mapping + bundle adjustment")
        if mapper == "global":
            recons = pycolmap.global_mapping(
                database_path=db_path, image_path=image_dir, output_path=recon_dir
            )
        elif mapper == "incremental":
            recons = pycolmap.incremental_mapping(
                database_path=db_path, image_path=image_dir, output_path=recon_dir
            )
        else:
            raise ValueError(f"unknown mapper {mapper!r} (use 'incremental' or 'global')")
        if not recons:
            raise RuntimeError("SfM produced no reconstruction (insufficient/weak matches)")

        best_idx = max(recons, key=lambda i: recons[i].num_reg_images())
        rec = recons[best_idx]
        # canonical single-model location for downstream stages
        model_dir = ctx.artifact_path("reconstruction") / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        rec.write(model_dir)
        rec.export_PLY(str(ctx.artifact_path("sparse.ply")))

        poses = self._export_poses(rec)
        poses["num_models"] = len(recons)
        ctx.write_json("poses.json", poses)

        ctx.progress(1.0, "sfm done")
        return StageResult(
            artifacts={
                "model": "reconstruction/model",
                "sparse_ply": "sparse.ply",
                "poses": "poses.json",
            },
            metrics={
                "num_models": len(recons),
                "reg_images": rec.num_reg_images(),
                "input_images": len(names),
                "points3D": rec.num_points3D(),
                "mean_reproj_error": round(rec.compute_mean_reprojection_error(), 4),
                "mean_track_length": round(rec.compute_mean_track_length(), 3),
            },
        )

    def _extract_and_match(self, ctx, pycolmap, db_path, image_dir, names, device) -> bool:
        """Extract SIFT features + match. Prefers a CUDA COLMAP binary (GPU SIFT, ~the reference tool speed);
        falls back to the CPU-only pycolmap wheel. Returns True if the GPU path was used."""
        from openreco import compute

        want_gpu = ctx.params["use_gpu"] in ("auto", "cuda")
        colmap = compute.find_colmap()
        if want_gpu and colmap is not None and compute.colmap_has_cuda():
            try:
                self._colmap_gpu(ctx, colmap, db_path, image_dir, names)
                return True
            except Exception as exc:  # noqa: BLE001
                ctx.logger.warning("GPU COLMAP extract/match failed (%r); using CPU pycolmap", exc)
        elif ctx.params["use_gpu"] == "cuda":
            ctx.logger.warning("use_gpu=cuda requested but no CUDA COLMAP binary found; using CPU "
                               "(set OPENRECO_COLMAP to a CUDA-enabled colmap to match GPU speed)")

        # CPU path (PyPI pycolmap wheel has no CUDA SIFT)
        ext = pycolmap.FeatureExtractionOptions()
        _try_set(ext, "max_image_size", int(ctx.params["max_image_size"]))
        _try_set(ext, "max_num_features", int(ctx.params["max_num_features"]))
        camera_mode = getattr(pycolmap.CameraMode, ctx.params["camera_mode"].upper())
        ctx.logger.info("extracting features from %d images (CPU)", len(names))
        ctx.progress(0.05, "feature extraction (CPU)")
        pycolmap.extract_features(database_path=db_path, image_path=image_dir, image_names=names,
                                  camera_mode=camera_mode, extraction_options=ext, device=device)
        ctx.progress(0.4, f"matching ({ctx.params['matcher']}, CPU)")
        self._match(pycolmap, db_path, ctx.params["matcher"], device)
        return False

    def _colmap_gpu(self, ctx, colmap, db_path, image_dir, names) -> None:
        """GPU SIFT extraction + matching via the CUDA COLMAP binary, into the same database."""
        list_path = ctx.artifact_path("image_list.txt")
        list_path.write_text("\n".join(names), encoding="utf-8")
        single = "1" if ctx.params["camera_mode"] in ("auto", "single") else "0"
        ctx.logger.info("extracting features from %d images (GPU SIFT via %s)", len(names), colmap.name)
        ctx.progress(0.05, "feature extraction (GPU)")
        self._run_colmap(ctx, colmap, "feature_extractor",
                         "--database_path", str(db_path), "--image_path", str(image_dir),
                         "--image_list_path", str(list_path),
                         "--ImageReader.single_camera", single,
                         "--FeatureExtraction.use_gpu", "1",
                         "--FeatureExtraction.max_image_size", str(int(ctx.params["max_image_size"])),
                         "--SiftExtraction.max_num_features", str(int(ctx.params["max_num_features"])))
        matcher = {"exhaustive": "exhaustive_matcher", "sequential": "sequential_matcher",
                   "spatial": "spatial_matcher"}.get(ctx.params["matcher"])
        if matcher is None:
            raise ValueError(f"unknown matcher {ctx.params['matcher']!r}")
        ctx.progress(0.4, f"matching ({ctx.params['matcher']}, GPU)")
        self._run_colmap(ctx, colmap, matcher,
                         "--database_path", str(db_path), "--FeatureMatching.use_gpu", "1")

    @staticmethod
    def _run_colmap(ctx, colmap, command, *args) -> None:
        """Invoke a COLMAP subcommand; stdio is inherited so glog streams live to the UI console."""
        import subprocess
        proc = subprocess.run([str(colmap), command, *args], check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"colmap {command} exited {proc.returncode}")

    def _match(self, pycolmap, db_path, matcher: str, device) -> None:
        if matcher == "exhaustive":
            pycolmap.match_exhaustive(database_path=db_path, device=device)
        elif matcher == "sequential":
            pycolmap.match_sequential(database_path=db_path, device=device)
        elif matcher == "spatial":
            pycolmap.match_spatial(database_path=db_path, device=device)
        else:
            raise ValueError(f"unknown matcher {matcher!r}")

    def _export_poses(self, rec) -> dict[str, Any]:
        images = []
        for image_id in rec.reg_image_ids():
            img = rec.image(image_id)
            cam = rec.camera(img.camera_id)
            c = img.projection_center()
            images.append(
                {
                    "name": img.name,
                    "center": [float(c[0]), float(c[1]), float(c[2])],
                    "camera_id": int(img.camera_id),
                    "model": cam.model_name,
                    "focal": float(cam.mean_focal_length()),
                    "width": int(cam.width),
                    "height": int(cam.height),
                }
            )
        images.sort(key=lambda d: d["name"])
        return {"crs": "local", "images": images}

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        m = result.metrics
        issues: list[Issue] = []
        if m["reg_images"] == 0:
            issues.append(Issue(Severity.ERROR, "no images registered"))
            return issues
        unreg = m["input_images"] - m["reg_images"]
        if unreg > 0:
            issues.append(
                Issue(
                    Severity.WARNING,
                    f"{unreg}/{m['input_images']} images not registered",
                    hint="increase overlap, try matcher=sequential for strips, or check for blur",
                )
            )
        if m["num_models"] > 1:
            issues.append(
                Issue(
                    Severity.WARNING,
                    f"{m['num_models']} disconnected models; using the largest "
                    f"({m['reg_images']} images)",
                    hint="more connecting overlap would merge them",
                )
            )
        if m["mean_reproj_error"] > 1.5:
            issues.append(
                Issue(Severity.WARNING, f"high mean reprojection error {m['mean_reproj_error']}px")
            )
        return issues


def _try_set(obj: Any, attr: str, value: Any) -> None:
    """Set an option attribute if this pycolmap build exposes it; ignore otherwise."""
    try:
        if hasattr(obj, attr):
            setattr(obj, attr, value)
    except Exception:  # noqa: BLE001
        pass
