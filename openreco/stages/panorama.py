"""Panorama stitching — pipeline stage.

Stitches a sequence of overlapping images into one panorama: SIFT features + descriptor matching
(scikit-image) between adjacent frames, a robust homography per pair (RANSAC), chained to a middle
reference, then warp+blend into a shared canvas. Best for pure-rotation / planar capture
(homography model); non-planar scenes show parallax. Standalone of the SfM pipeline; no GPU/DB.

Inputs: none required (reads images from params.image_dir, default 'images').
Outputs: panorama.jpg, panorama.json
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openreco import panorama as pano
from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.images import list_images


@register_stage
class Panorama(Stage):
    type = "panorama"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {"image_dir": "images", "max_dim": 1600, "ransac_thresh_px": 4.0,
                "match_ratio": 0.7}

    def run(self, ctx: RunContext) -> StageResult:
        from PIL import Image
        from skimage.color import rgb2gray
        from skimage.feature import SIFT, match_descriptors

        image_dir = (ctx.project_dir / ctx.params["image_dir"]).resolve()
        paths = list_images(image_dir)
        if len(paths) < 2:
            raise ValueError("panorama needs >= 2 images")
        max_dim = int(ctx.params["max_dim"])

        rgbs, kps, descs = [], [], []
        for i, p in enumerate(paths):
            rgb = self._load(Image, p, max_dim)
            rgbs.append(rgb)
            sift = SIFT()
            sift.detect_and_extract(rgb2gray(rgb))
            kps.append(sift.keypoints)               # (N,2) = (row, col)
            descs.append(sift.descriptors)
            ctx.progress(0.1 + 0.4 * (i + 1) / len(paths), f"features {p.name}")

        thr = float(ctx.params["ransac_thresh_px"])
        ratio = float(ctx.params["match_ratio"])
        pair_h, inliers = [], []
        for i in range(len(paths) - 1):
            m = match_descriptors(descs[i], descs[i + 1], cross_check=True, max_ratio=ratio)
            if len(m) < 8:
                raise RuntimeError(f"too few matches ({len(m)}) between images {i} and {i+1} "
                                   "— is this an overlapping sequence?")
            src = kps[i][m[:, 0]][:, ::-1]           # (row,col) -> (x,y)
            dst = kps[i + 1][m[:, 1]][:, ::-1]
            h, inl = pano.ransac_homography(src, dst, thresh=thr)
            pair_h.append(h)
            inliers.append(int(inl.sum()))

        ref = len(paths) // 2
        h_to_ref = self._chain(ref, pair_h)
        ctx.progress(0.8, f"warping + blending {len(rgbs)} images")
        canvas, mask = pano.stitch(rgbs, h_to_ref)

        Image.fromarray(canvas, "RGB").save(ctx.artifact_path("panorama.jpg"), quality=92)
        cov = float(mask.mean())
        ctx.write_json("panorama.json", {"images": len(rgbs), "canvas": list(canvas.shape[:2]),
                                         "coverage": round(cov, 3), "pair_inliers": inliers})
        return StageResult(artifacts={"panorama": "panorama.jpg", "meta": "panorama.json"},
                           metrics={"images": len(rgbs), "width": canvas.shape[1],
                                    "height": canvas.shape[0], "coverage": round(cov, 3)})

    def _chain(self, ref: int, pair_h: list[np.ndarray]) -> list[np.ndarray]:
        n = len(pair_h) + 1
        out = [np.eye(3) for _ in range(n)]
        for j in range(ref - 1, -1, -1):            # left of ref:  H_j = H_{j+1} @ H_{j->j+1}
            out[j] = out[j + 1] @ pair_h[j]
        for j in range(ref + 1, n):                 # right of ref: H_j = H_{j-1} @ inv(H_{j-1->j})
            out[j] = out[j - 1] @ np.linalg.inv(pair_h[j - 1])
        return out

    def _load(self, Image, path, max_dim) -> np.ndarray:
        pil = Image.open(path).convert("RGB")
        w, h = pil.size
        s = min(1.0, max_dim / max(w, h))
        if s < 1.0:
            pil = pil.resize((max(1, int(w * s)), max(1, int(h * s))))
        return np.asarray(pil)

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        m = result.metrics
        issues = [Issue(Severity.INFO, f"stitched {m['images']} images -> {m['width']}x{m['height']} "
                        f"panorama ({int(m['coverage'] * 100)}% filled)")]
        if m["coverage"] < 0.3:
            issues.append(Issue(Severity.WARNING, "sparse panorama — images may not overlap well"))
        return issues
