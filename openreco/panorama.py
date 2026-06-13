"""Panorama stitching core: homography estimation (DLT + RANSAC) and multi-image warp/blend.

Pure numpy + scipy (map_coordinates for warping). Adjacent overlapping images are related by a
homography (valid for pure-rotation capture or planar scenes); we estimate pairwise homographies
from feature matches, chain them to a reference frame, then inverse-warp every image into a shared
canvas with feathered (distance-to-edge) weights and blend. The stage supplies matches (pycolmap).
"""

from __future__ import annotations

import numpy as np


def homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Homography mapping src->dst from >=4 correspondences (normalized DLT)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        a.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(a))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def _apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:3]


def ransac_homography(src: np.ndarray, dst: np.ndarray, thresh: float = 3.0,
                      iters: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Robust homography. Returns (H, inlier_mask)."""
    n = len(src)
    if n < 4:
        raise ValueError("need >= 4 correspondences")
    rng = np.random.default_rng(seed)
    best_inl = np.zeros(n, bool)
    best_count = 0
    for _ in range(iters):
        s = rng.choice(n, 4, replace=False)
        try:
            h = homography_dlt(src[s], dst[s])
        except np.linalg.LinAlgError:
            continue
        err = np.linalg.norm(_apply_h(h, src) - dst, axis=1)
        inl = err < thresh
        if inl.sum() > best_count:
            best_count = int(inl.sum())
            best_inl = inl
            if best_count == n:
                break
    h = homography_dlt(src[best_inl], dst[best_inl])    # refit on inliers
    return h, best_inl


def _corners(w: int, h: int) -> np.ndarray:
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], float)


def _feather(h: int, w: int) -> np.ndarray:
    """Per-pixel weight that falls off toward the image border (smooth blending)."""
    yy = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]
    xx = np.minimum(np.arange(w), np.arange(w)[::-1])[None, :]
    wgt = np.minimum(yy, xx).astype(np.float64) + 1.0
    return wgt / wgt.max()


def stitch(images: list[np.ndarray], h_to_ref: list[np.ndarray], max_canvas: int = 6000):
    """Warp `images` (each HxWx3) by `h_to_ref` (img->reference homography) into one blended canvas."""
    from scipy.ndimage import map_coordinates

    all_c = []
    for img, h in zip(images, h_to_ref):
        ih, iw = img.shape[:2]
        all_c.append(_apply_h(h, _corners(iw, ih)))
    pts = np.vstack(all_c)
    minx, miny = np.floor(pts.min(0)).astype(int)
    maxx, maxy = np.ceil(pts.max(0)).astype(int)
    cw, ch = min(maxx - minx, max_canvas), min(maxy - miny, max_canvas)
    cx, cy = np.meshgrid(np.arange(cw) + minx, np.arange(ch) + miny)
    ref_pts = np.stack([cx.ravel(), cy.ravel(), np.ones(cw * ch)])

    accum = np.zeros((ch, cw, 3))
    wsum = np.zeros((ch, cw))
    for img, h in zip(images, h_to_ref):
        ih, iw = img.shape[:2]
        src = np.linalg.inv(h) @ ref_pts                 # ref-frame canvas -> source image coords
        sx = (src[0] / src[2]).reshape(ch, cw)
        sy = (src[1] / src[2]).reshape(ch, cw)
        inside = (sx >= 0) & (sx < iw - 1) & (sy >= 0) & (sy < ih - 1)
        if not inside.any():
            continue
        feat = _feather(ih, iw)
        w = map_coordinates(feat, [sy, sx], order=1, mode="constant", cval=0.0)
        w *= inside
        for c in range(3):
            samp = map_coordinates(img[:, :, c], [sy, sx], order=1, mode="constant", cval=0.0)
            accum[:, :, c] += samp * w
        wsum += w
    out = np.zeros((ch, cw, 3), np.uint8)
    good = wsum > 1e-6
    for c in range(3):
        ch_acc = accum[:, :, c]
        ch_acc[good] /= wsum[good]
        out[:, :, c] = np.clip(ch_acc, 0, 255).astype(np.uint8)
    return out, good
