"""Sparse-cloud filtering + camera re-optimization ("gradual selection").

A survey-grade accuracy step (a pro tool's tie-point gradual selection): remove unreliable
3D tie points — those with high reprojection error or short tracks — then re-run bundle
adjustment so the camera parameters and remaining points are re-fit to the cleaner set. This
typically lowers reprojection error and tightens the calibration before georeferencing.

Drop-in between `sfm` and `georef`: it consumes the upstream "model" and emits a refined "model"
(same contract), so georef/mvs just use it transparently.

Inputs:  any upstream providing "model" (sfm)
Outputs: model/ (refined COLMAP model), sparse.ply, refine.json (before/after metrics)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage


def points_to_delete(items: list[tuple[int, float, int]], max_error: float, min_track: int,
                     max_error_percentile: float = 0.0) -> list[int]:
    """Select tie-point ids to remove. `items` = (id, reproj_error, track_length). A point is
    dropped if its error exceeds `max_error`, its track is shorter than `min_track`, or (when
    `max_error_percentile` > 0) its error is in the worst N% (gradual selection)."""
    pct_thresh = np.inf
    if max_error_percentile > 0:
        errs = np.array([e for _, e, _ in items], dtype=float)
        if errs.size:
            pct_thresh = float(np.percentile(errs, 100.0 - max_error_percentile))
    return [pid for pid, err, track in items
            if err > max_error or err > pct_thresh or track < min_track]


@register_stage
class Refine(Stage):
    type = "refine"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {
            "max_reproj_error": 1.0,    # delete points whose mean reprojection error exceeds this (px)
            "min_track_length": 2,      # delete points seen by fewer than this many images
            "max_error_percentile": 0.0,  # also delete the worst N% by error (0 = off; "gradual selection")
            "run_bundle_adjustment": True,
        }

    def run(self, ctx: RunContext) -> StageResult:
        import pycolmap

        model_dir = ctx.input_artifact(ctx.input_with("model"), "model")
        rec = pycolmap.Reconstruction(str(model_dir))

        before_pts = rec.num_points3D()
        before_err = float(rec.compute_mean_reprojection_error())

        max_err = float(ctx.params["max_reproj_error"])
        min_track = int(ctx.params["min_track_length"])
        pct = float(ctx.params["max_error_percentile"])

        items = [(pid, (p.error if p.has_error else 0.0), p.track.length())
                 for pid, p in rec.points3D.items()]
        to_delete = points_to_delete(items, max_err, min_track, pct)
        for pid in to_delete:
            rec.delete_point3D(pid)

        ctx.progress(0.5, f"deleted {len(to_delete)} tie points")
        if ctx.params["run_bundle_adjustment"] and rec.num_points3D() > 0:
            ctx.progress(0.6, "bundle adjustment")
            pycolmap.bundle_adjustment(rec, pycolmap.BundleAdjustmentOptions())

        after_pts = rec.num_points3D()
        after_err = float(rec.compute_mean_reprojection_error())

        out_model = ctx.artifact_path("model")
        out_model.mkdir(parents=True, exist_ok=True)
        rec.write(out_model)
        rec.export_PLY(str(ctx.artifact_path("sparse.ply")))
        ctx.write_json("refine.json", {
            "points_before": before_pts, "points_after": after_pts,
            "deleted": len(to_delete),
            "reproj_error_before": round(before_err, 4),
            "reproj_error_after": round(after_err, 4),
            "reg_images": rec.num_reg_images(),
        })
        return StageResult(
            artifacts={"model": "model", "sparse_ply": "sparse.ply", "refine": "refine.json"},
            metrics={
                "deleted": len(to_delete),
                "points_after": after_pts,
                "reproj_error_before": round(before_err, 4),
                "reproj_error_after": round(after_err, 4),
                "reg_images": rec.num_reg_images(),
            },
        )

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        m = result.metrics
        issues: list[Issue] = []
        if m["reg_images"] == 0 or m["points_after"] == 0:
            issues.append(Issue(Severity.ERROR, "refinement removed all points/images — "
                                "loosen max_reproj_error / min_track_length"))
        elif m["reproj_error_after"] > m["reproj_error_before"] + 1e-6:
            issues.append(Issue(Severity.WARNING, "reprojection error increased after refinement",
                                hint="bundle adjustment may not have converged"))
        else:
            issues.append(Issue(Severity.INFO,
                          f"reprojection error {m['reproj_error_before']} -> {m['reproj_error_after']} px, "
                          f"{m['deleted']} tie points removed"))
        return issues
