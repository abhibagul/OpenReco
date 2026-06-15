"""Measurements on derived products — currently DSM-based volumes (cut/fill).

Volume above a reference surface is the classic earthwork/stockpile measurement: integrate
(elevation - base) over the DSM cells. The base can be a fixed elevation, the DSM minimum, or
its mean. Reported in cubic meters when the DSM is in a metric CRS (as our georeferenced DSMs
are). Distance/area/cross-section measurements will join this module as they're added.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def measure_volume(dsm_path: str | Path, base: str | float = "min") -> dict[str, Any]:
    """Cut/fill volume of a DSM relative to a reference `base`.

    base: "min" (volume of everything above the lowest point — stockpile-style),
          "mean", or an explicit elevation (float, in the DSM's vertical units).
    Returns volumes in m**3 (cut = above base, fill = below base, net = cut - fill),
    plus the base elevation, measured area, and cell count.
    """
    import rasterio

    with rasterio.open(dsm_path) as ds:
        z = ds.read(1).astype(np.float64)
        rx, ry = ds.res
        nodata = ds.nodata
    if nodata is not None and not np.isnan(nodata):
        z[z == nodata] = np.nan
    valid = np.isfinite(z)
    if not valid.any():
        raise ValueError("DSM has no valid cells")
    zv = z[valid]

    if base == "min":
        base_z = float(zv.min())
    elif base == "mean":
        base_z = float(zv.mean())
    else:
        base_z = float(base)

    cell = abs(rx * ry)
    diff = zv - base_z
    cut = float(np.clip(diff, 0, None).sum() * cell)
    fill = float(np.clip(-diff, 0, None).sum() * cell)
    return {
        "base": base if isinstance(base, str) else "fixed",
        "base_elevation": round(base_z, 4),
        "cut_m3": round(cut, 3),
        "fill_m3": round(fill, 3),
        "net_m3": round(cut - fill, 3),
        "area_m2": round(cell * int(valid.sum()), 3),
        "cells": int(valid.sum()),
        "cell_size_m": round(cell, 6),
    }


def _point_in_poly(pts_xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized even-odd ray-cast test. pts_xy: (N,2), poly: (M,2) ring. Returns (N,) bool."""
    x, y = pts_xy[:, 0], pts_xy[:, 1]
    inside = np.zeros(len(pts_xy), dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = (yi > y) != (yj > y)
        denom = yj - yi
        if denom == 0:
            denom = 1e-12
        xints = (xj - xi) * (y - yi) / denom + xi
        inside ^= cond & (x < xints)
        j = i
    return inside


def _boundary_cells(mask: np.ndarray) -> np.ndarray:
    """Cells in `mask` that touch a non-mask cell or the grid edge (4-neighbourhood)."""
    b = np.zeros_like(mask)
    b[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
    b[1:, :] |= mask[1:, :] & ~mask[:-1, :]
    b[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
    b[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
    b[0, :] |= mask[0, :]
    b[-1, :] |= mask[-1, :]
    b[:, 0] |= mask[:, 0]
    b[:, -1] |= mask[:, -1]
    return b & mask


def measure_volume_region(xyz: np.ndarray, polygon, base: str | float = "plane",
                          cell_size: float | None = None) -> dict[str, Any]:
    """Polygon-bounded cut/fill volume of a surface (the interactive "stockpile" measurement).

    `xyz` is an (N,3) point set in metric world coordinates (a mesh's vertices or a dense cloud).
    `polygon` is a list of [x, y] (or [x, y, z]) vertices in the same frame outlining the footprint.
    The top surface is gridded as the max Z per cell. `base` is the reference the volume is taken
    above: "plane" (best-fit plane through the footprint boundary — handles sloped ground), "mean",
    "min", or an explicit elevation. Returns cut/fill/net m**3, the measured area, and grid info.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    poly = np.asarray(polygon, dtype=np.float64)[:, :2]
    if len(poly) < 3:
        raise ValueError("polygon needs at least 3 vertices")
    minx, miny = poly.min(0)
    maxx, maxy = poly.max(0)
    span = float(max(maxx - minx, maxy - miny))
    if span <= 0:
        raise ValueError("polygon has zero extent")
    if cell_size is None:
        cell_size = max(span / 200.0, 1e-6)
    cell_size = float(cell_size)

    inb = ((xyz[:, 0] >= minx) & (xyz[:, 0] <= maxx)
           & (xyz[:, 1] >= miny) & (xyz[:, 1] <= maxy))
    pts = xyz[inb]
    if len(pts) < 3:
        raise ValueError("not enough surface points inside the region")

    nx = max(int(np.ceil((maxx - minx) / cell_size)), 1)
    ny = max(int(np.ceil((maxy - miny) / cell_size)), 1)
    ix = np.clip(((pts[:, 0] - minx) / cell_size).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - miny) / cell_size).astype(int), 0, ny - 1)
    grid = np.full((ny, nx), -np.inf)
    np.maximum.at(grid, (iy, ix), pts[:, 2])      # top surface = highest point per cell
    filled = np.isfinite(grid)

    cx = minx + (np.arange(nx) + 0.5) * cell_size
    cy = miny + (np.arange(ny) + 0.5) * cell_size
    cxg, cyg = np.meshgrid(cx, cy)
    inside = _point_in_poly(np.column_stack([cxg.ravel(), cyg.ravel()]), poly).reshape(ny, nx)
    use = inside & filled
    if not use.any():
        raise ValueError("no surface cells fall inside the region")

    zc = grid[use]
    xc, yc = cxg[use], cyg[use]
    if base == "plane":
        bnd = _boundary_cells(use)
        if int(bnd.sum()) >= 3:
            a = np.column_stack([cxg[bnd], cyg[bnd], np.ones(int(bnd.sum()))])
            coef, *_ = np.linalg.lstsq(a, grid[bnd], rcond=None)
            base_z = coef[0] * xc + coef[1] * yc + coef[2]
            base_label = "plane"
        else:
            base_z = np.full(zc.shape, float(zc.min()))
            base_label = "min"
    elif base == "mean":
        base_z = float(zc.mean())
        base_label = "mean"
    elif base == "min":
        base_z = float(zc.min())
        base_label = "min"
    else:
        base_z = float(base)
        base_label = "fixed"

    cell = cell_size * cell_size
    diff = zc - base_z
    cut = float(np.clip(diff, 0, None).sum() * cell)
    fill = float(np.clip(-diff, 0, None).sum() * cell)
    base_summary = float(np.mean(base_z)) if isinstance(base_z, np.ndarray) else float(base_z)
    return {
        "base": base_label,
        "base_elevation": round(base_summary, 4),
        "cut_m3": round(cut, 3),
        "fill_m3": round(fill, 3),
        "net_m3": round(cut - fill, 3),
        "area_m2": round(cell * int(use.sum()), 3),
        "cells": int(use.sum()),
        "cell_size_m": round(cell_size, 6),
    }


def measure_profile_region(xyz: np.ndarray, p_from, p_to, n: int = 200,
                           radius: float | None = None) -> dict[str, Any]:
    """Elevation cross-section along p_from -> p_to sampled directly from a point set.

    `xyz` is (N,3) in metric world coordinates (a mesh's vertices or a dense cloud); the line
    endpoints are in the same frame (as picked in the 3D view). Each of `n` samples takes the
    nearest point's elevation; samples whose nearest point is farther than `radius` (a coverage
    gap) are returned as null. Returns per-sample dist/x/y/z plus length/min/max/relief/slope.
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 2:
        raise ValueError("not enough points to sample a profile")
    a = np.asarray(p_from, dtype=np.float64)[:2]
    b = np.asarray(p_to, dtype=np.float64)[:2]
    seg = b - a
    seg_len = float(np.hypot(seg[0], seg[1]))
    if seg_len <= 0:
        raise ValueError("profile endpoints coincide")
    t = np.linspace(0.0, 1.0, max(int(n), 2))
    xs = a[0] + t * seg[0]
    ys = a[1] + t * seg[1]
    step = seg_len / (len(t) - 1)
    if radius is None:
        radius = max(step * 5.0, 1e-6)

    tree = cKDTree(xyz[:, :2])
    dist, idx = tree.query(np.column_stack([xs, ys]), k=1)
    zs = np.where(dist <= radius, xyz[idx, 2], np.nan)
    dvals = t * seg_len
    samples = [{"dist_m": round(float(d), 3), "x": round(float(x), 3), "y": round(float(y), 3),
                "z": (round(float(z), 3) if np.isfinite(z) else None)}
               for d, x, y, z in zip(dvals, xs, ys, zs)]
    finite = zs[np.isfinite(zs)]
    relief = float(finite.max() - finite.min()) if finite.size else None
    return {
        "length_m": round(seg_len, 3),
        "samples": samples,
        "z_min": round(float(finite.min()), 3) if finite.size else None,
        "z_max": round(float(finite.max()), 3) if finite.size else None,
        "relief_m": round(relief, 3) if relief is not None else None,
        "slope_pct": round(relief / seg_len * 100.0, 2) if relief is not None and seg_len else None,
        "covered": int(finite.size),
    }


def measure_profile(dsm_path: str | Path, p_from: tuple[float, float],
                    p_to: tuple[float, float], n: int = 200) -> dict[str, Any]:
    """Elevation cross-section along the segment p_from -> p_to (both in the DSM's CRS units).
    Samples `n` points and returns per-sample distance/x/y/z plus min/max/length summary."""
    import rasterio

    with rasterio.open(dsm_path) as ds:
        xs = np.linspace(p_from[0], p_to[0], n)
        ys = np.linspace(p_from[1], p_to[1], n)
        zs = np.array([v[0] for v in ds.sample(np.column_stack([xs, ys]))], dtype=np.float64)
        nodata = ds.nodata
    if nodata is not None and not np.isnan(nodata):
        zs[zs == nodata] = np.nan
    seg_len = float(np.hypot(p_to[0] - p_from[0], p_to[1] - p_from[1]))
    dist = np.linspace(0, seg_len, n)
    samples = [{"dist_m": round(float(d), 3), "x": round(float(x), 3), "y": round(float(y), 3),
                "z": (round(float(z), 3) if np.isfinite(z) else None)}
               for d, x, y, z in zip(dist, xs, ys, zs)]
    finite = zs[np.isfinite(zs)]
    return {
        "length_m": round(seg_len, 3),
        "samples": samples,
        "z_min": round(float(finite.min()), 3) if finite.size else None,
        "z_max": round(float(finite.max()), 3) if finite.size else None,
        "relief_m": round(float(finite.max() - finite.min()), 3) if finite.size else None,
    }
