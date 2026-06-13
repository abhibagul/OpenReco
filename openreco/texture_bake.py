"""Texture baking: project the best source image onto each mesh face and fill a UV atlas.

Given a UV-unwrapped mesh (per-vertex UVs from xatlas) and the SfM cameras, for each face we
pick the most front-facing in-view image, then rasterize the face's atlas triangle and sample
that image (barycentric → image pixel → bilinear) into the atlas. Pinhole projection (K only);
lens distortion is ignored in v1 (small for typical aerial/photo lenses).

Pure numpy so the geometry is unit-testable. The stage wires in xatlas + cameras + images.
"""

from __future__ import annotations

import numpy as np


def project(p3x4: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points (N,3) with a 3x4 pinhole matrix. Returns (pixels[N,2], depth[N])."""
    h = np.c_[pts, np.ones(len(pts))] @ p3x4.T          # (N,3)
    z = h[:, 2]
    px = h[:, :2] / z[:, None]
    return px, z


def select_best_image(centroids: np.ndarray, normals: np.ndarray, cams: list[dict]) -> np.ndarray:
    """For each face, choose the camera index that sees it most head-on and in-bounds, or -1.

    cams: list of {P (3x4), C (3,), w, h}. Score = frontality (normal·view dir) when the centroid
    projects inside the image and in front of the camera."""
    f = len(centroids)
    best_score = np.full(f, -1.0)
    best_idx = np.full(f, -1, dtype=np.int64)
    for i, cam in enumerate(cams):
        px, z = project(cam["P"], centroids)
        view = cam["C"][None, :] - centroids            # face -> camera
        view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-12
        frontal = np.abs((normals * view).sum(axis=1))  # 1 = head-on
        ok = (z > 0) & (px[:, 0] >= 0) & (px[:, 0] < cam["w"]) & (px[:, 1] >= 0) & (px[:, 1] < cam["h"])
        score = np.where(ok, frontal, -1.0)
        take = score > best_score
        best_score[take] = score[take]
        best_idx[take] = i
    return best_idx


def select_top_k(centroids: np.ndarray, normals: np.ndarray, cams: list[dict],
                 k: int) -> tuple[np.ndarray, np.ndarray]:
    """For each face, the top-k cameras by frontality that see it in-bounds. Returns (idx[F,k]
    with -1 padding, weight[F,k] with 0 for invalid)."""
    f = len(centroids)
    scores = np.full((f, len(cams)), -1.0)
    for i, cam in enumerate(cams):
        px, z = project(cam["P"], centroids)
        view = cam["C"][None, :] - centroids
        view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-12
        frontal = np.abs((normals * view).sum(axis=1))
        ok = (z > 0) & (px[:, 0] >= 0) & (px[:, 0] < cam["w"]) & (px[:, 1] >= 0) & (px[:, 1] < cam["h"])
        scores[:, i] = np.where(ok, frontal, -1.0)
    k = min(k, len(cams))
    idx = np.argsort(-scores, axis=1)[:, :k]
    w = np.take_along_axis(scores, idx, axis=1)
    idx = idx.astype(np.int64)
    idx[w <= 0] = -1
    w[w <= 0] = 0.0
    return idx, w


def bake_face_blend(accum: np.ndarray, wsum: np.ndarray, atlas_tri: np.ndarray,
                    samples: list[dict]) -> None:
    """Blend multiple source images into one face's atlas triangle. `samples` is a list of
    {P (3x4), verts3 (3,3), image (H,W,3), gain (3,), weight}. Accumulates weight*gain*color and
    weight into float buffers (per-texel, only where the texel projects in-bounds for that image);
    the caller divides accum/wsum at the end. Reduces seams vs. single-best-image."""
    res = accum.shape[0]
    a = atlas_tri
    minx = max(0, int(np.floor(a[:, 0].min())))
    maxx = min(res - 1, int(np.ceil(a[:, 0].max())))
    miny = max(0, int(np.floor(a[:, 1].min())))
    maxy = min(res - 1, int(np.ceil(a[:, 1].max())))
    if minx > maxx or miny > maxy:
        return
    xs, ys = np.meshgrid(np.arange(minx, maxx + 1), np.arange(miny, maxy + 1))
    pts = np.column_stack([xs.ravel() + 0.5, ys.ravel() + 0.5])
    v0 = a[1] - a[0]
    v1 = a[2] - a[0]
    v2 = pts - a[0]
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < 1e-9:
        return
    l1 = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
    l2 = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
    l0 = 1 - l1 - l2
    inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
    if not inside.any():
        return
    bary = np.column_stack([l0, l1, l2])[inside]
    tx = xs.ravel()[inside].astype(int)
    ty = ys.ravel()[inside].astype(int)
    for s in samples:
        img_pts = (np.c_[s["verts3"], np.ones(3)] @ s["P"].T)        # (3,3)
        img_pts = img_pts[:, :2] / img_pts[:, 2:3]
        img_xy = bary @ img_pts                                      # (M,2)
        h, w = s["image"].shape[:2]
        valid = (img_xy[:, 0] >= 0) & (img_xy[:, 0] < w - 1) & (img_xy[:, 1] >= 0) & (img_xy[:, 1] < h - 1)
        if not valid.any():
            continue
        col = _bilinear(s["image"], img_xy[valid]) * s["gain"]       # (Mv,3) exposure-corrected
        accum[ty[valid], tx[valid]] += s["weight"] * col
        wsum[ty[valid], tx[valid]] += s["weight"]


def _bilinear(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample img (H,W,3) at pixel coords xy (N,2). Clamped."""
    h, w = img.shape[:2]
    x = np.clip(xy[:, 0], 0, w - 1.001)
    y = np.clip(xy[:, 1], 0, h - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    c = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
         + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return c


def bake_face(atlas: np.ndarray, filled: np.ndarray, atlas_tri: np.ndarray,
              img_pts: np.ndarray, image: np.ndarray) -> None:
    """Rasterize one face's atlas triangle and fill texels by sampling `image` via barycentric
    interpolation of the face's projected image points. Mutates atlas/filled in place."""
    res = atlas.shape[0]
    a = atlas_tri
    minx = max(0, int(np.floor(a[:, 0].min())))
    maxx = min(res - 1, int(np.ceil(a[:, 0].max())))
    miny = max(0, int(np.floor(a[:, 1].min())))
    maxy = min(res - 1, int(np.ceil(a[:, 1].max())))
    if minx > maxx or miny > maxy:
        return
    xs, ys = np.meshgrid(np.arange(minx, maxx + 1), np.arange(miny, maxy + 1))
    pts = np.column_stack([xs.ravel() + 0.5, ys.ravel() + 0.5])
    # barycentric of pts wrt atlas triangle
    v0 = a[1] - a[0]
    v1 = a[2] - a[0]
    v2 = pts - a[0]
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < 1e-9:
        return
    l1 = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
    l2 = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
    l0 = 1 - l1 - l2
    inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
    if not inside.any():
        return
    bary = np.column_stack([l0, l1, l2])[inside]
    img_xy = bary @ img_pts                              # (M,2) image pixels
    colors = _bilinear(image, img_xy)
    tx = (xs.ravel()[inside]).astype(int)
    ty = (ys.ravel()[inside]).astype(int)
    atlas[ty, tx] = np.clip(np.round(colors), 0, 255).astype(np.uint8)  # round (avoid uint8 truncation bias)
    filled[ty, tx] = True
