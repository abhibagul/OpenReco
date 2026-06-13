"""3D Gaussian Splatting — neural appearance branch (gsplat, Apache-2.0).

The differentiator: from the SAME SfM camera solution that drives the metric geometry, train a
3D Gaussian Splatting model — a photoreal, view-dependent, real-time-renderable scene. Geometry
(SfM/MVS) stays the source of truth for measurements; the splat is the appearance layer, exported
as a standard 3DGS `.ply` viewable in web/desktop splat viewers.

This is a minimal, dependency-honest trainer built directly on `gsplat.rasterization` (no
nerfstudio): initialise Gaussians from the sparse cloud, optimise position/scale/rotation/opacity/
colour against the input images with an L1 photometric loss, export the splat. CUDA-only; needs
torch + gsplat (compiled CUDA kernels). On a small-VRAM card, keep iterations/scene modest.

Inputs:  a stage providing "model" (sfm/refine/georef) + a stage providing "images"
Outputs: splat.ply (standard 3DGS format), splat.json (metrics)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage

_SH_C0 = 0.28209479177387814  # SH band-0 constant: f_dc = (rgb - 0.5) / C0


@register_stage
class Splat(Stage):
    type = "splat"
    version = "1"
    deterministic = False

    def default_params(self) -> dict[str, Any]:
        return {
            "iterations": 2000,
            "max_image_size": 1000,   # downscale training images (VRAM/speed)
            "lr": 0.01,
            "sh_degree": 0,           # 0 = direct RGB (simplest, view-independent colour)
            "max_gaussians": 500000,
        }

    def run(self, ctx: RunContext) -> StageResult:
        import pycolmap
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("3D Gaussian Splatting requires a CUDA GPU (torch.cuda unavailable)")
        device = torch.device("cuda")

        model_dir = ctx.input_artifact(ctx.input_with("model"), "model")
        images = ctx.read_input_json(ctx.input_with("images"), "images")
        image_dir = Path(images["image_dir"])
        rec = pycolmap.Reconstruction(str(model_dir))

        cams = self._load_cameras(ctx, rec, image_dir, int(ctx.params["max_image_size"]), device)
        gaussians = self._init_gaussians(rec, int(ctx.params["max_gaussians"]), device)
        metrics = self._train(ctx, gaussians, cams, device)

        self._export_ply(ctx.artifact_path("splat.ply"), gaussians)
        ctx.write_json("splat.json", {"num_gaussians": int(gaussians["means"].shape[0]), **metrics})
        return StageResult(
            artifacts={"splat": "splat.ply", "meta": "splat.json"},
            metrics={"num_gaussians": int(gaussians["means"].shape[0]), **metrics},
        )

    # ---- data ---------------------------------------------------------------------------
    def _load_cameras(self, ctx, rec, image_dir: Path, max_size: int, device):
        """Per-image world->camera viewmat (OpenCV), pinhole K (scaled), and the RGB image tensor."""
        import torch
        from PIL import Image

        cams = []
        for image_id in rec.reg_image_ids():
            img = rec.image(image_id)
            cam = rec.camera(img.camera_id)
            path = image_dir / img.name
            if not path.exists():
                continue
            pil = Image.open(path).convert("RGB")
            w0, h0 = pil.size
            scale = min(1.0, max_size / max(w0, h0))
            w, h = int(round(w0 * scale)), int(round(h0 * scale))
            pil = pil.resize((w, h))
            rgb = torch.from_numpy(np.asarray(pil, np.float32) / 255.0).to(device)  # [h,w,3]

            k = np.asarray(cam.calibration_matrix(), np.float64) * scale
            k[2, 2] = 1.0
            viewmat = np.eye(4, dtype=np.float64)
            viewmat[:3, :] = np.asarray(img.cam_from_world().matrix())  # world->camera
            cams.append({
                "viewmat": torch.tensor(viewmat, dtype=torch.float32, device=device),
                "K": torch.tensor(k, dtype=torch.float32, device=device),
                "rgb": rgb, "w": w, "h": h,
            })
        if len(cams) < 2:
            raise RuntimeError("splat training needs >= 2 registered images with files present")
        ctx.logger.info("loaded %d training views", len(cams))
        return cams

    def _init_gaussians(self, rec, max_n: int, device):
        import torch
        from scipy.spatial import cKDTree

        xyz, rgb = [], []
        for p in rec.points3D.values():
            xyz.append(p.xyz)
            rgb.append(np.asarray(p.color, np.float32) / 255.0)
        xyz = np.asarray(xyz, np.float32)
        rgb = np.asarray(rgb, np.float32)
        if len(xyz) > max_n:
            sel = np.random.default_rng(0).choice(len(xyz), max_n, replace=False)
            xyz, rgb = xyz[sel], rgb[sel]
        # init scale = mean distance to 3 nearest neighbours (log-space param)
        d, _ = cKDTree(xyz).query(xyz, k=min(4, len(xyz)))
        nn = np.clip(d[:, 1:].mean(axis=1), 1e-6, None)

        def p(t):
            return torch.nn.Parameter(torch.tensor(t, device=device))
        n = len(xyz)
        quats = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
        return {
            "means": p(xyz),
            "log_scales": p(np.log(np.stack([nn, nn, nn], 1)).astype(np.float32)),
            "quats": p(quats),
            "logit_opac": p(np.full((n,), _logit(0.1), np.float32)),
            "colors": p(rgb),
        }

    # ---- training -----------------------------------------------------------------------
    def _train(self, ctx, g, cams, device) -> dict[str, Any]:
        import torch

        try:
            from gsplat import rasterization
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "gsplat is required for the splat stage: pip install gsplat. Its CUDA kernels are "
                "JIT-compiled on first use and need a working CUDA toolchain (nvcc + matching CUDA "
                "headers + a host C++ compiler). On Windows this means the full CUDA Toolkit + MSVC; "
                "the pip nvcc wheel alone is insufficient."
            ) from exc

        iters = int(ctx.params["iterations"])
        opt = torch.optim.Adam(list(g.values()), lr=float(ctx.params["lr"]))
        rng = np.random.default_rng(0)
        last = 0.0
        for it in range(iters):
            cam = cams[int(rng.integers(len(cams)))]
            colors, _alphas, _info = rasterization(
                means=g["means"],
                quats=g["quats"] / g["quats"].norm(dim=-1, keepdim=True),
                scales=torch.exp(g["log_scales"]),
                opacities=torch.sigmoid(g["logit_opac"]),
                colors=torch.sigmoid(g["colors"]),
                viewmats=cam["viewmat"][None],
                Ks=cam["K"][None],
                width=cam["w"], height=cam["h"],
            )
            pred = colors[0, ..., :3]
            loss = torch.abs(pred - cam["rgb"]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
            if it % 200 == 0:
                ctx.progress(it / iters, f"iter {it} L1={last:.4f}")
                if ctx.is_cancelled():
                    break
        return {"iterations": iters, "final_l1": round(last, 5)}

    # ---- export -------------------------------------------------------------------------
    def _export_ply(self, path: Path, g) -> None:
        import torch

        with torch.no_grad():
            means = g["means"].cpu().numpy()
            scales = g["log_scales"].cpu().numpy()
            quats = (g["quats"] / g["quats"].norm(dim=-1, keepdim=True)).cpu().numpy()
            opac = g["logit_opac"].cpu().numpy()
            rgb = torch.sigmoid(g["colors"]).cpu().numpy()
        f_dc = (rgb - 0.5) / _SH_C0  # store colour as SH band-0 (standard 3DGS .ply)
        n = len(means)
        fields = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                   "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
        arr = np.zeros(n, dtype=[(f, "<f4") for f in fields])
        arr["x"], arr["y"], arr["z"] = means.T
        arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc.T
        arr["opacity"] = opac
        arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales.T
        arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = quats.T
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
        header += [f"property float {f}" for f in fields] + ["end_header", ""]
        with path.open("wb") as fh:
            fh.write("\n".join(header).encode("ascii"))
            fh.write(arr.tobytes())

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        return [Issue(Severity.INFO,
                      f"trained {result.metrics['num_gaussians']:,} gaussians, "
                      f"final L1={result.metrics.get('final_l1')}")]


def _logit(x: float) -> float:
    return float(np.log(x / (1 - x)))
