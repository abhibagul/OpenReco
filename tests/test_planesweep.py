"""Portable plane-sweep MVS — correctness on a synthetic plane, on the CPU device.
Needs torch (slice/neural dep) -> skips otherwise. Validates the vendor-neutral dense path."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from openreco.mvs_planesweep import planesweep_dense  # noqa: E402


def _nadir_cam(center, f=300.0, w=160, h=160):
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    r = np.diag([1.0, -1.0, -1.0])              # looking straight down (-z world)
    t = -r @ np.asarray(center, float)
    return k, r, t, np.asarray(center, float)


def _render_plane(k, r, t, center, texture, ext, w=160, h=160):
    """Render a nadir view of a textured plane at world z=0 (inverse map each pixel)."""
    kinv = np.linalg.inv(k)
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(w * h)])
    rays = r.T @ (kinv @ pix)                    # world-space ray dirs
    s = (0.0 - center[2]) / rays[2]              # intersect z=0
    wx = center[0] + s * rays[0]
    wy = center[1] + s * rays[1]
    tt = texture.shape[0]
    tx = np.clip((wx / ext * tt).astype(int), 0, tt - 1)
    ty = np.clip((wy / ext * tt).astype(int), 0, tt - 1)
    return texture[ty, tx].reshape(h, w, 3)


def test_planesweep_recovers_plane_depth_cpu():
    rng = np.random.default_rng(0)
    ext = 6.0
    texture = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)   # rich texture on the ground
    H = 10.0                                                         # camera height above plane
    views = []
    for cx in (2.5, 3.0, 3.5):                                       # 3 nadir views, small baseline
        k, r, t, c = _nadir_cam([cx, 3.0, H])
        rgb = _render_plane(k, r, t, c, texture, ext)
        views.append({"rgb": rgb, "K": k, "R": r, "t": t, "C": c})

    xyz, rgb = planesweep_dense(views, depth_min=H - 2, depth_max=H + 2, device="cpu",
                                n_depths=32, n_neighbors=2, cost_thresh=0.05)
    assert len(xyz) > 200                                           # produced a dense-ish cloud
    # recovered points lie on the plane (world z ~ 0)
    assert abs(np.median(xyz[:, 2])) < 0.4
