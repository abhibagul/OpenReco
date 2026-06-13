"""Rigid point-cloud registration (ICP) for fusing an external cloud (e.g. LiDAR) with the
photogrammetric dense cloud. Pure numpy + scipy KDTree.

Kabsch gives the optimal rigid transform (rotation+translation, no scale) for corresponded point
sets; ICP iterates nearest-neighbour correspondence + Kabsch until the RMS settles. Source points
are subsampled for the correspondence search so it scales to millions of points.
"""

from __future__ import annotations

import numpy as np


def kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rigid transform mapping P onto Q (Q ≈ R·P + t). P, Q are corresponded (N,3)."""
    mp, mq = p.mean(0), q.mean(0)
    h = (p - mp).T @ (q - mq)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = mq - r @ mp
    return r, t


def icp(src: np.ndarray, dst: np.ndarray, max_iter: int = 40, tol: float = 1e-5,
        max_corr_dist: float | None = None, sample: int = 20000,
        init: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """Register `src` onto `dst` (rigid). Returns {R, t, rmse, fitness, iterations}, where
    R,t map the ORIGINAL src into the dst frame (dst ≈ R·src + t)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(dst)
    r_tot = np.eye(3)
    t_tot = np.zeros(3)
    if init is not None:
        r_tot, t_tot = init[0].copy(), init[1].copy()
    cur = (src @ r_tot.T) + t_tot

    rng = np.random.default_rng(0)
    idx_s = (rng.choice(len(cur), sample, replace=False) if len(cur) > sample
             else np.arange(len(cur)))
    prev = np.inf
    rmse = np.inf
    fitness = 0.0
    it = 0
    for it in range(1, max_iter + 1):
        d, j = tree.query(cur[idx_s])
        mask = d < max_corr_dist if max_corr_dist else np.ones(len(d), bool)
        if mask.sum() < 3:
            break
        dr, dt = kabsch(cur[idx_s][mask], dst[j[mask]])
        cur = (cur @ dr.T) + dt
        r_tot = dr @ r_tot
        t_tot = dr @ t_tot + dt
        rmse = float(np.sqrt(np.mean(d[mask] ** 2)))
        fitness = float(mask.mean())
        if abs(prev - rmse) < tol:
            break
        prev = rmse
    return {"R": r_tot, "t": t_tot, "rmse": rmse, "fitness": fitness, "iterations": it}


def apply_transform(pts: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (pts @ r.T) + t
