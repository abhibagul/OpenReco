"""Textured model — pipeline stage (UV atlas + image-projected color).

Turns a (vertex-colored) dense mesh into a properly textured model: decimate to a workable face
count (fast-simplification), UV-unwrap (xatlas), then bake a texture atlas by projecting each
face into its most front-facing source image. Output is a textured OBJ + MTL + atlas PNG — the
flagship industry-standard visual deliverable, at far higher fidelity than per-vertex colors.

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
from openreco.io.pointcloud import read_mesh_ply, write_textured_obj
from openreco.texture_bake import bake_face, project, select_best_image


@register_stage
class Texture(Stage):
    type = "texture"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {"target_faces": 200000, "atlas_resolution": 2048}

    def run(self, ctx: RunContext) -> StageResult:
        import pycolmap
        import xatlas

        verts, faces, _ = read_mesh_ply(ctx.input_artifact(ctx.input_with("mesh"), "mesh"))
        model_dir = ctx.input_artifact(ctx.input_with("model"), "model")
        images = ctx.read_input_json(ctx.input_with("images"), "images")
        image_dir = Path(images["image_dir"])
        rec = pycolmap.Reconstruction(str(model_dir))

        verts, faces = self._decimate(ctx, verts, faces, int(ctx.params["target_faces"]))
        ctx.progress(0.3, "UV unwrap (xatlas)")
        vmapping, indices, uvs = xatlas.parametrize(verts, faces)
        tverts = verts[vmapping]
        faces = indices.astype(np.int64)

        cams = self._cameras(rec)
        res = int(ctx.params["atlas_resolution"])
        atlas, filled, used = self._bake(ctx, tverts, faces, uvs, cams, image_dir, res)

        self._save(ctx, tverts, faces, uvs, atlas)
        cov = float(filled.mean())
        ctx.write_json("texture.json", {
            "vertices": int(len(tverts)), "faces": int(len(faces)),
            "atlas_resolution": res, "images_used": used, "atlas_coverage": round(cov, 3),
        })
        return StageResult(
            artifacts={"obj": "textured.obj", "mtl": "textured.mtl", "texture": "texture.png",
                       "meta": "texture.json"},
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

    def _cameras(self, rec) -> list[dict]:
        cams = []
        for image_id in rec.reg_image_ids():
            img = rec.image(image_id)
            cam = rec.camera(img.camera_id)
            k = np.asarray(cam.calibration_matrix(), np.float64)
            m = np.asarray(img.cam_from_world().matrix())          # 3x4 world->cam
            cams.append({"P": k @ m, "C": np.asarray(img.projection_center()),
                         "w": int(cam.width), "h": int(cam.height), "name": img.name,
                         "wh": (int(cam.width), int(cam.height))})
        return cams

    def _bake(self, ctx, tverts, faces, uvs, cams, image_dir, res):
        from PIL import Image

        tri = tverts[faces]                                        # (F,3,3)
        centroids = tri.mean(axis=1)
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        ctx.progress(0.45, "selecting best image per face")
        best = select_best_image(centroids, normals, cams)

        atlas = np.zeros((res, res, 3), np.uint8)
        filled = np.zeros((res, res), bool)
        auv = uvs * (res - 1)                                      # atlas pixel coords (u=col, v=row)
        used = 0
        order = np.argsort(best)                                   # group faces by image -> load once
        cur = -2
        image = None
        for fi in order:
            ci = int(best[fi])
            if ci < 0:
                continue
            if ci != cur:
                cur = ci
                cam = cams[ci]
                pil = Image.open(image_dir / cam["name"]).convert("RGB").resize(cam["wh"])
                image = np.asarray(pil, np.float32)
                used += 1
                ctx.progress(0.5, f"baking from {cam['name']}")
            f = faces[fi]
            px, _z = project(cams[ci]["P"], tverts[f])
            bake_face(atlas, filled, auv[f], px, image)
        return atlas, filled, used

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
