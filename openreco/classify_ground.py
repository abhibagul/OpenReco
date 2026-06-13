"""Point-cloud ground classification (grid-minimum + height-threshold).

A pragmatic, dependency-free ground filter: grid the cloud in XY, take the lowest point per cell
as the local ground level, then classify each point as GROUND if it sits within `ground_thresh`
metres of the (gap-filled) local ground surface, else NON-GROUND (buildings, vegetation). Cell
size should exceed the largest off-ground object so each cell still contains real ground.

Returns LAS classification codes (2 = ground, 1 = unclassified/non-ground). A true bare-earth
DTM is then just the ground points rasterised. Full CSF/progressive-densification is future work.
"""

from __future__ import annotations

import numpy as np

GROUND = 2
NON_GROUND = 1


def _ground_grid(xyz: np.ndarray, cell_m: float):
    minx, miny = xyz[:, 0].min(), xyz[:, 1].min()
    maxx, maxy = xyz[:, 0].max(), xyz[:, 1].max()
    nc = max(1, int(np.ceil((maxx - minx) / cell_m)) + 1)
    nr = max(1, int(np.ceil((maxy - miny) / cell_m)) + 1)
    col = np.clip(((xyz[:, 0] - minx) / cell_m).astype(int), 0, nc - 1)
    row = np.clip(((xyz[:, 1] - miny) / cell_m).astype(int), 0, nr - 1)
    grid = np.full((nr, nc), np.inf)
    np.minimum.at(grid, (row, col), xyz[:, 2])      # lowest z per cell
    grid[~np.isfinite(grid)] = np.nan
    return grid, row, col, minx, miny


def classify_ground(xyz: np.ndarray, cell_m: float = 5.0, ground_thresh: float = 0.5) -> np.ndarray:
    """Per-point LAS class codes (2 ground, 1 non-ground)."""
    grid, row, col, _minx, _miny = _ground_grid(xyz, cell_m)
    grid = _fill_nan_nearest(grid)
    ground_z = grid[row, col]                        # local ground level under each point
    height = xyz[:, 2] - ground_z
    cls = np.where(height <= ground_thresh, GROUND, NON_GROUND).astype(np.uint8)
    return cls


def _fill_nan_nearest(grid: np.ndarray) -> np.ndarray:
    """Fill empty cells with the nearest valid value (so the ground surface is continuous)."""
    mask = np.isnan(grid)
    if not mask.any():
        return grid
    if mask.all():
        return np.zeros_like(grid)
    from scipy.ndimage import distance_transform_edt

    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return grid[tuple(idx)]
