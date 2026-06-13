"""Ingest & validate — pipeline stage 1.

Scans an image directory, reads dimensions/EXIF/GPS, scores blur, and auto-culls images
below a sharpness threshold (non-destructively: culled images stay in the table flagged
`excluded`, so the decision is auditable and reversible). Emits an image table consumed by
SfM, plus QA issues (too few images, no GPS → non-metric warning).

Outputs (in cache dir):
  images.json   — {"crs_hint": ..., "images": [ImageInfo, ...]}
"""

from __future__ import annotations

import statistics
from typing import Any

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.images import ImageInfo, list_images, read_image_info


@register_stage
class Ingest(Stage):
    type = "ingest"
    version = "1"

    def default_params(self) -> dict[str, Any]:
        return {
            "image_dir": "images",     # relative to the project dir
            "blur_threshold": 0.0,     # 0 = disable culling; else absolute variance-of-Laplacian
            "blur_relative": 0.15,     # also cull images below this fraction of the median blur
            "min_images": 3,
        }

    def params_schema(self) -> dict[str, Any]:
        return {
            "image_dir": {"type": "string"},
            "blur_threshold": {"type": "number", "minimum": 0},
            "blur_relative": {"type": "number", "minimum": 0, "maximum": 1},
            "min_images": {"type": "integer", "minimum": 2},
        }

    def run(self, ctx: RunContext) -> StageResult:
        image_dir = (ctx.project_dir / ctx.params["image_dir"]).resolve()
        if not image_dir.is_dir():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")

        paths = list_images(image_dir)
        if not paths:
            raise FileNotFoundError(f"no images in {image_dir}")

        infos: list[ImageInfo] = []
        for i, p in enumerate(paths):
            infos.append(read_image_info(p))
            ctx.progress((i + 1) / len(paths), f"read {p.name}")
            if ctx.is_cancelled():
                raise RuntimeError("cancelled during ingest")

        self._cull(infos, ctx.params)

        kept = [im for im in infos if not im.excluded]
        gps = [im for im in kept if im.has_gps]
        ctx.write_json(
            "images.json",
            {
                "image_dir": str(image_dir),
                "images": [im.to_dict() for im in infos],
            },
        )
        metrics = {
            "total": len(infos),
            "kept": len(kept),
            "excluded": len(infos) - len(kept),
            "with_gps": len(gps),
        }
        return StageResult(artifacts={"images": "images.json"}, metrics=metrics)

    def _cull(self, infos: list[ImageInfo], params: dict[str, Any]) -> None:
        scores = [im.blur_score for im in infos if im.blur_score is not None]
        median = statistics.median(scores) if scores else 0.0
        abs_thr = float(params["blur_threshold"])
        rel_thr = median * float(params["blur_relative"]) if params["blur_relative"] else 0.0
        thr = max(abs_thr, rel_thr)
        if thr <= 0:
            return
        for im in infos:
            if im.blur_score is not None and im.blur_score < thr:
                im.excluded = True
                im.reason = f"blurry (score {im.blur_score:.1f} < {thr:.1f})"

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        issues: list[Issue] = []
        m = result.metrics
        if m["kept"] < ctx.params["min_images"]:
            issues.append(
                Issue(
                    Severity.ERROR,
                    f"only {m['kept']} usable images (min {ctx.params['min_images']})",
                    hint="add more overlapping images or lower blur_threshold",
                )
            )
        if m["excluded"]:
            issues.append(Issue(Severity.INFO, f"auto-culled {m['excluded']} blurry image(s)"))
        if m["with_gps"] == 0:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "no GPS in EXIF — outputs will be non-metric unless GCPs/scale are provided",
                    hint="add GCPs or a scale bar in the georeference stage",
                )
            )
        elif m["with_gps"] < m["kept"]:
            issues.append(
                Issue(Severity.INFO, f"{m['kept'] - m['with_gps']} image(s) lack GPS")
            )
        return issues
