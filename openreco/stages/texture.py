"""Textured model — pipeline stage (UV atlas + image-projected color).

Turns a (vertex-colored) dense mesh into a properly textured model: decimate to a workable face
count (fast-simplification), UV-unwrap (xatlas), then bake a texture atlas by projecting each
face into its most front-facing source image. Output is a textured OBJ + MTL + atlas PNG — the
flagship familiar visual deliverable, at far higher fidelity than per-vertex colors.

v1 uses pinhole projection (ignores lens distortion) and a single best image per face (no
multi-band blending / de-lighting yet). Inputs resolved by role: mesh, model, images.

Outputs: textured.obj, textured.mtl, texture.png, texture.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.gltf import write_glb_textured
from openreco.io.pointcloud import read_mesh_ply, write_textured_obj
from openreco.texture_bake import bake_face_blend, select_top_k


@register_stage
class Texture(Stage):
    type = "texture"
    version = "4"  # v4: downscale source images for baking + lighter defaults (speed)
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {
            "target_faces": 150000,
            "atlas_resolution": 2048,
            "blend_images": 3,          # blend up to N most front-facing views per face (1 = single best, fastest)
            "equalize_exposure": True,  # per-image gain to a common brightness (radiometric balance)
            "image_max_dim": 2000,      # downscale source images for baking (0 = full res); big speed/memory win
        }

    def run(self, ctx: RunContext) -> StageResult:
        import pycolmap
        import xatlas

        verts, faces, _ = read_mesh_ply(ctx.input_artifact(ctx.input_with("mesh"), "mesh"))
        model_dir = ctx.input_artifact(ctx.input_with("model"), "model")
        images = ctx.read_input_json(ctx.input_with("images"), "images")
        image_dir = Path(images["image_dir"])
        rec = pycolmap.Reconstruction(str(model_dir))

        ctx.logger.info("texturing mesh: %d vertices, %d faces", len(verts), len(faces))
        verts, faces = self._decimate(ctx, verts, faces, int(ctx.params["target_faces"]))
        ctx.progress(0.3, "UV unwrap (xatlas)")
        ctx.logger.info("UV unwrapping %d faces with xatlas (this can take a while) …", len(faces))
        vmapping, indices, uvs = xatlas.parametrize(verts, faces)
        tverts = verts[vmapping]
        faces = indices.astype(np.int64)
        ctx.logger.info("UV unwrap done: %d atlas vertices", len(tverts))

        cams = self._cameras(rec, int(ctx.params.get("image_max_dim", 2000)))
        res = int(ctx.params["atlas_resolution"])
        atlas, filled, used = self._bake(ctx, tverts, faces, uvs, cams, image_dir, res)

        self._save(ctx, tverts, faces, uvs, atlas)
        png_bytes = ctx.artifact_path("texture.png").read_bytes()
        write_glb_textured(ctx.artifact_path("textured.glb"), tverts, faces, uvs, png_bytes)
        cov = float(filled.mean())
        ctx.write_json("texture.json", {
            "vertices": int(len(tverts)), "faces": int(len(faces)),
            "atlas_resolution": res, "images_used": used, "atlas_coverage": round(cov, 3),
        })
        return StageResult(
            artifacts={"obj": "textured.obj", "mtl": "textured.mtl", "texture": "texture.png",
                       "glb": "textured.glb", "meta": "texture.json"},
            metrics={"faces": int(len(faces)), "atlas_resolution": res,
                     "images_used": used, "atlas_coverage": round(cov, 3)},
        )

    def _decimate(self, ctx, verts, faces, target):
        if len(faces) <= target:
            return verts.astype(np.float32), faces.astype(np.int32)
        import fast_simplification

        reduction = float(np.clip(1.0 - target / len(faces), 0.0, 0.99))
        ctx.progress(0.1, f"decimating {len(faces):,} -> ~{target:,} faces")
        v, f = fast_simplification.simplify(verts.astype(np.float32),
                                            faces.astype(np.int32), target_reduction=reduction)
        return v.astype(np.float32), f.astype(np.int32)

    def _cameras(self, rec, max_dim: int = 0) -> list[dict]:
        cams = []
        for image_id in rec.reg_image_ids():
            img = rec.image(image_id)
            cam = rec.camera(img.camera_id)
            k = np.asarray(cam.calibration_matrix(), np.float64)
            m = np.asarray(img.cam_from_world().matrix())          # 3x4 world->cam
            w, h = int(cam.width), int(cam.height)
            # downscale source images for baking (huge speed/memory win on big photos): scale K + size
            f = min(1.0, max_dim / max(w, h)) if max_dim else 1.0
            if f < 1.0:
                k = k.copy()
                k[:2, :] *= f
                w, h = max(1, round(w * f)), max(1, round(h * f))
            cams.append({"P": k @ m, "C": np.asarray(img.projection_center()),
                         "w": w, "h": h, "name": img.name, "wh": (w, h)})
        return cams

    def _bake(self, ctx, tverts, faces, uvs, cams, image_dir, res):
        tri = tverts[faces]                                        # (F,3,3)
        centroids = tri.mean(axis=1)
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

        k = max(1, int(ctx.params["blend_images"]))
        ctx.progress(0.45, f"selecting top-{k} images per face")
        idx, weights = select_top_k(centroids, normals, cams, k)

        used_ids = sorted({int(c) for c in idx.ravel() if c >= 0})
        images, gains = self._load_images(ctx, cams, used_ids, image_dir,
                                          bool(ctx.params["equalize_exposure"]))

        accum = np.zeros((res, res, 3), np.float64)
        wsum = np.zeros((res, res), np.float64)
        auv = uvs * (res - 1)                                      # atlas pixel coords (u=col, v=row)
        nfaces = len(faces)
        ctx.logger.info("baking %d faces into a %d^2 atlas from %d images …", nfaces, res, len(used_ids))
        step = max(1, nfaces // 50)                                # ~50 progress ticks over the bake
        for fi in range(nfaces):
            if fi % step == 0:
                ctx.progress(0.55 + 0.4 * fi / nfaces, f"baking face {fi:,}/{nfaces:,}")
                if ctx.is_cancelled():
                    raise RuntimeError("cancelled during texture bake")
            samples = []
            for j in range(idx.shape[1]):
                ci = int(idx[fi, j])
                if ci < 0 or ci not in images:
                    continue
                samples.append({"P": cams[ci]["P"], "verts3": tverts[faces[fi]],
                                "image": images[ci], "gain": gains[ci], "weight": float(weights[fi, j])})
            if samples:
                bake_face_blend(accum, wsum, auv[faces[fi]], samples)

        filled = wsum > 0
        atlas = np.zeros((res, res, 3), np.uint8)
        atlas[filled] = np.clip(np.round(accum[filled] / wsum[filled, None]), 0, 255).astype(np.uint8)
        return atlas, filled, len(used_ids)

    def _load_images(self, ctx, cams, used_ids, image_dir, equalize):
        """Load (and resize to camera resolution) each used image once; compute a per-image
        exposure gain toward the common median brightness (basic radiometric balancing)."""
        from PIL import Image

        images, means = {}, {}
        ctx.logger.info("loading %d source images for baking …", len(used_ids))
        for n, ci in enumerate(used_ids, 1):
            cam = cams[ci]
            arr = np.asarray(Image.open(image_dir / cam["name"]).convert("RGB").resize(cam["wh"]),
                             np.float32)
            images[ci] = arr
            means[ci] = arr.reshape(-1, 3).mean(axis=0)
            ctx.progress(0.45 + 0.1 * n / len(used_ids), f"loaded image {n}/{len(used_ids)}")
        gains = {ci: np.ones(3, np.float32) for ci in used_ids}
        if equalize and len(means) > 1:
            target = np.median(np.stack(list(means.values())), axis=0)   # per-channel target
            for ci, m in means.items():
                gains[ci] = np.clip(target / np.maximum(m, 1e-3), 0.6, 1.6).astype(np.float32)
        return images, gains

    def _save(self, ctx, tverts, faces, uvs, atlas) -> None:
        from PIL import Image

        Image.fromarray(atlas, "RGB").save(ctx.artifact_path("texture.png"))
        write_textured_obj(ctx.artifact_path("textured.obj"), tverts, faces, uvs, "texture.png")

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        m = result.metrics
        issues = [Issue(Severity.INFO, f"textured {m['faces']:,} faces from {m['images_used']} images, "
                        f"{int(m['atlas_coverage'] * 100)}% atlas coverage")]
        if m["atlas_coverage"] < 0.2:
            issues.append(Issue(Severity.WARNING, "low atlas coverage — few faces saw a valid image",
                                hint="check camera poses / image availability"))
        return issues
