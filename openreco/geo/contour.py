"""Marching-squares contour extraction from a raster grid (pure numpy).

Given a 2D elevation grid and an iso-level, returns line segments where the (bilinearly
interpolated) surface crosses that level, in fractional (col, row) grid coordinates. The
caller maps those to world coordinates via the raster's affine transform. NaN cells are
skipped. Pure numpy so it has no plotting/skimage dependency and is unit-testable.
"""

from __future__ import annotations

import numpy as np

Segment = tuple[tuple[float, float], tuple[float, float]]  # ((col,row),(col,row))

# Marching-squares case table: for each of the 16 corner-sign patterns, the pairs of edges the
# contour crosses. Edges: 0=top, 1=right, 2=bottom, 3=left. Corner bits (tl,tr,br,bl) -> index.
_CASES: dict[int, list[tuple[int, int]]] = {
    0: [], 15: [],
    1: [(2, 3)], 14: [(2, 3)],
    2: [(1, 2)], 13: [(1, 2)],
    3: [(1, 3)], 12: [(1, 3)],
    4: [(0, 1)], 11: [(0, 1)],
    6: [(0, 2)], 9: [(0, 2)],
    7: [(0, 3)], 8: [(0, 3)],
    5: [(0, 3), (1, 2)],   # saddle
    10: [(0, 1), (2, 3)],  # saddle
}


def _interp(level: float, a: float, b: float) -> float:
    if a == b:
        return 0.5
    return (level - a) / (b - a)


def contour_segments(z: np.ndarray, level: float) -> list[Segment]:
    """Iso-line segments at `level` in fractional (col, row) coordinates."""
    h, w = z.shape
    segs: list[Segment] = []
    for r in range(h - 1):
        for c in range(w - 1):
            tl, tr = z[r, c], z[r, c + 1]
            bl, br = z[r + 1, c], z[r + 1, c + 1]
            if not np.isfinite(tl + tr + bl + br):
                continue
            idx = ((tl > level) << 3) | ((tr > level) << 2) | ((br > level) << 1) | (bl > level)
            edges = _CASES.get(idx, [])
            if not edges:
                continue
            # edge crossing points (col, row), fractional
            pts = {
                0: (c + _interp(level, tl, tr), float(r)),         # top
                1: (float(c + 1), r + _interp(level, tr, br)),     # right
                2: (c + _interp(level, bl, br), float(r + 1)),     # bottom
                3: (float(c), r + _interp(level, tl, bl)),         # left
            }
            for e0, e1 in edges:
                segs.append((pts[e0], pts[e1]))
    return segs


def contour_levels(zmin: float, zmax: float, interval: float) -> list[float]:
    """Iso-levels at multiples of `interval` strictly inside (zmin, zmax)."""
    if interval <= 0 or not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        return []
    start = np.ceil(zmin / interval) * interval
    levels = np.arange(start, zmax, interval)
    return [float(x) for x in levels]
