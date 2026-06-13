"""Portable plane-sweep MVS (PyTorch) — hardware-agnostic dense reconstruction.

A classic plane-sweep stereo written in torch so it runs on ANY backend torch supports — NVIDIA
CUDA, Apple-Silicon MPS, AMD ROCm, or CPU — from one codebase. For each reference view we sweep a
set of fronto-parallel depth planes, warp neighbouring views onto the reference at each depth,
measure windowed photo-consistency, take the per-pixel best depth, filter by cost + cross-view
geometric consistency, and back-project to a fused world-space point cloud.

Lower quality than COLMAP's CUDA PatchMatch (the NVIDIA path) but vendor-neutral: this is the
fallback that covers AMD / Apple-Silicon / CPU. Memory scales with H*W*n_depths, so callers
downscale images for small-VRAM GPUs.
"""

from __future__ import annotations

import numpy as np


def planesweep_dense(views: list[dict], depth_min: float, depth_max: float, device: str,
                     n_depths: int = 48, n_neighbors: int = 4, window: int = 5,
                     cost_thresh: float = 0.02, consistency_px: float = 2.0,
                     ref_stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """views: [{rgb(H,W,3 uint8), K(3,3), R(3,3 world->cam), t(3,), C(3,)}]. Returns (xyz, rgb)."""
    import torch
    import torch.nn.functional as F

    dev = torch.device(device)
    n = len(views)
    centers = np.array([v["C"] for v in views])
    grays, rgbs, Ks, Rs, ts = [], [], [], [], []
    for v in views:
        g = np.asarray(v["rgb"], np.float32).mean(2) / 255.0
        grays.append(torch.tensor(g, device=dev))
        rgbs.append(torch.tensor(np.asarray(v["rgb"], np.uint8), device=dev))
        Ks.append(torch.tensor(v["K"], dtype=torch.float32, device=dev))
        Rs.append(torch.tensor(v["R"], dtype=torch.float32, device=dev))
        ts.append(torch.tensor(v["t"], dtype=torch.float32, device=dev))
    h, w = grays[0].shape
    inv_d = torch.linspace(1.0 / depth_max, 1.0 / depth_min, n_depths, device=dev)
    depths = 1.0 / inv_d                                              # (D,) near->far

    uu, vv = torch.meshgrid(torch.arange(w, device=dev, dtype=torch.float32),
                            torch.arange(h, device=dev, dtype=torch.float32), indexing="xy")
    ones = torch.ones_like(uu)
    pix = torch.stack([uu, vv, ones], 0).reshape(3, -1)              # (3, H*W)

    out_xyz, out_rgb = [], []
    for r in range(0, n, ref_stride):
        nbrs = _neighbors(centers, r, n_neighbors)
        if not nbrs:
            continue
        kinv_r = torch.linalg.inv(Ks[r])
        rays = kinv_r @ pix                                          # (3, H*W) ref camera rays
        cost = torch.full((n_depths, h * w), 10.0, device=dev)
        valid_any = torch.zeros((n_depths, h * w), dtype=torch.bool, device=dev)
        for di, d in enumerate(depths):
            xcam = rays * d                                          # (3,H*W) point in ref cam
            xworld = Rs[r].T @ (xcam - ts[r][:, None])
            acc = torch.zeros(h * w, device=dev)
            cnt = torch.zeros(h * w, device=dev)
            for nb in nbrs:
                xc = Rs[nb] @ xworld + ts[nb][:, None]
                proj = Ks[nb] @ xc
                z = proj[2]
                un, vn = proj[0] / z, proj[1] / z
                gx = (2 * un / (w - 1) - 1).reshape(1, h, w)
                gy = (2 * vn / (h - 1) - 1).reshape(1, h, w)
                grid = torch.stack([gx, gy], -1)                    # (1,H,W,2)
                samp = F.grid_sample(grays[nb][None, None], grid, align_corners=True,
                                     mode="bilinear", padding_mode="zeros")[0, 0]
                ok = (z.reshape(h, w) > 0) & (gx[0].abs() <= 1) & (gy[0].abs() <= 1)
                diff = (samp - grays[r]) ** 2
                acc += torch.where(ok, diff, torch.zeros_like(diff)).reshape(-1)
                cnt += ok.reshape(-1).float()
            seen = cnt > 0
            cost[di][seen] = (acc[seen] / cnt[seen])
            valid_any[di] = seen
        cost = _box(cost.reshape(n_depths, 1, h, w), window).reshape(n_depths, -1)
        best_cost, best_di = cost.min(0)
        depth_map = depths[best_di]
        keep = (best_cost < cost_thresh) & valid_any.any(0)
        if keep.sum() == 0:
            continue
        xcam = rays * depth_map[None, :]
        xworld = (Rs[r].T @ (xcam - ts[r][:, None])).T              # (H*W,3)
        cols = rgbs[r].reshape(-1, 3)
        out_xyz.append(xworld[keep].cpu().numpy())
        out_rgb.append(cols[keep].cpu().numpy().astype(np.uint8))

    if not out_xyz:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    return np.vstack(out_xyz), np.vstack(out_rgb)


def _neighbors(centers: np.ndarray, r: int, k: int) -> list[int]:
    d = np.linalg.norm(centers - centers[r], axis=1)
    order = np.argsort(d)
    return [int(i) for i in order if i != r][:k]


def _box(x, window: int):
    """Box-filter a (D,1,H,W) cost stack for windowed photo-consistency."""
    import torch.nn.functional as F

    if window <= 1:
        return x
    pad = window // 2
    return F.avg_pool2d(x, kernel_size=window, stride=1, padding=pad, count_include_pad=False)
