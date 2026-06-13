"""Coded-target auto-detection — pipeline stage.

Detects fiducial coded targets (ArUco / AprilTag) in every image and aggregates observations by
marker id across images. Emits a GCP-observation CSV (id,image,u,v) ready for the georef GCP path
once paired with surveyed marker world coordinates — eliminating manual marker picking.

Inputs: none required (reads images from params.image_dir, default 'images').
Outputs: markers.json (per-image + per-marker), gcp_observations.csv, markers.json metrics
"""

from __future__ import annotations

import csv
from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.images import list_images
from openreco.markers import detect_markers


@register_stage
class Markers(Stage):
    type = "markers"
    version = "1"

    def default_params(self) -> dict[str, Any]:
        return {"image_dir": "images", "dictionary": "4x4_50", "max_dim": 2400}

    def run(self, ctx: RunContext) -> StageResult:
        from PIL import Image

        image_dir = (ctx.project_dir / ctx.params["image_dir"]).resolve()
        paths = list_images(image_dir)
        if not paths:
            raise FileNotFoundError(f"no images in {image_dir}")
        dictionary = ctx.params["dictionary"]
        max_dim = int(ctx.params["max_dim"])

        per_image: dict[str, list] = {}
        per_marker: dict[int, list] = {}
        for i, p in enumerate(paths):
            pil = Image.open(p).convert("L")
            w, h = pil.size
            s = min(1.0, max_dim / max(w, h))
            if s < 1.0:
                pil = pil.resize((int(w * s), int(h * s)))
            dets = detect_markers(np.asarray(pil), dictionary)
            # rescale detections back to full-resolution pixel coords
            obs = []
            for d in dets:
                x, y = d["center"][0] / s, d["center"][1] / s
                obs.append({"id": d["id"], "x": round(x, 2), "y": round(y, 2)})
                per_marker.setdefault(d["id"], []).append({"image": p.name, "x": round(x, 2),
                                                            "y": round(y, 2)})
            if obs:
                per_image[p.name] = obs
            ctx.progress((i + 1) / len(paths), f"{p.name}: {len(obs)} markers")

        # GCP-observation CSV (id, image, u, v) — add X,Y,Z per id to use as georef GCPs
        with ctx.artifact_path("gcp_observations.csv").open("w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["marker_id", "image", "u", "v"])
            for mid, obs in sorted(per_marker.items()):
                for o in obs:
                    wr.writerow([mid, o["image"], o["x"], o["y"]])

        ctx.write_json("markers.json", {"dictionary": dictionary, "images": len(paths),
                                        "per_image": per_image,
                                        "per_marker": {str(k): v for k, v in per_marker.items()}})
        total = sum(len(v) for v in per_marker.values())
        return StageResult(
            artifacts={"markers": "markers.json", "observations": "gcp_observations.csv"},
            metrics={"images": len(paths), "unique_markers": len(per_marker),
                     "total_detections": total},
        )

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        m = result.metrics
        if m["unique_markers"] == 0:
            return [Issue(Severity.INFO, "no coded targets detected (none present, or wrong "
                          "dictionary)", hint="set params.dictionary to match your targets")]
        return [Issue(Severity.INFO, f"{m['unique_markers']} unique markers, "
                      f"{m['total_detections']} detections across {m['images']} images")]
