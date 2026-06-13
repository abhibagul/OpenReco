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
