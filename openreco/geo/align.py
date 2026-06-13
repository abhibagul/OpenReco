"""Alignment math for georeferencing from Ground Control Points (pure numpy).

- DLT triangulation: recover a 3D point in the reconstruction frame from its 2D observations
  in >=2 registered images (given each image's 3x4 projection matrix P = K[R|t]).
- Umeyama similarity: the least-squares scale+rotation+translation mapping one set of 3D
  points onto another (local reconstruction points -> world GCP coordinates).

Kept dependency-free (numpy only) so the math is unit-testable anywhere, independent of pycolmap.
"""

from __future__ import annotations

import numpy as np


def triangulate_dlt(projections: list[np.ndarray], uvs: list[tuple[float, float]]) -> np.ndarray:
    """Linear (DLT) triangulation. `projections` are 3x4 matrices, `uvs` the matching pixel
    coordinates. Returns the 3D point. Needs >= 2 views."""
    if len(projections) < 2:
        raise ValueError("triangulation needs >= 2 observations")
    rows = []
    for p, (u, v) in zip(projections, uvs):
        rows.append(u * p[2] - p[0])
        rows.append(v * p[2] - p[1])
    a = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(a)
    x = vt[-1]
    return x[:3] / x[3]


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (scale s, rotation R 3x3, translation t) with dst ≈ s·R·src + t.
    Umeyama (1991). Needs >= 3 non-collinear correspondences."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = src.shape[0]
    if n < 3:
        raise ValueError("similarity alignment needs >= 3 correspondences")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = (xd.T @ xs) / n
    u, d, vt = np.linalg.svd(cov)
    s_diag = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_diag[2, 2] = -1.0
    r = u @ s_diag @ vt
    var_s = (xs ** 2).sum() / n
    scale = float(np.trace(np.diag(d) @ s_diag) / var_s) if var_s > 0 else 1.0
    t = mu_d - scale * (r @ mu_s)
    return scale, r, t


def rotmat_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion in (x, y, z, w) order (pycolmap's convention)."""
    m = np.asarray(r, float)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)
