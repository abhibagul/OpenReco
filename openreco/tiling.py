"""Mesh tiling: partition a mesh into an N×N grid of XY tiles for streaming.

Each face is assigned to a tile by its centroid; per tile we gather the used vertices and reindex
the faces, so every tile is a standalone mesh. Pure numpy — the stage writes each tile as a glTF
and a multi-tile 3D Tiles tileset references them.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def tile_mesh(verts: np.ndarray, faces: np.ndarray, vcolors: np.ndarray | None,
              grid_n: int) -> list[dict]:
    """Split into up to grid_n*grid_n XY tiles. Returns a list of
    {verts, faces, vcolors, bbox_min, bbox_max, gx, gy} for each non-empty tile."""
    if vcolors is None:
        vcolors = np.full((len(verts), 3), 200, np.uint8)
    centroids = verts[faces].mean(axis=1)
    minx, miny = verts[:, 0].min(), verts[:, 1].min()
    maxx, maxy = verts[:, 0].max(), verts[:, 1].max()
    gx = np.clip(((centroids[:, 0] - minx) / (maxx - minx + _EPS) * grid_n).astype(int), 0, grid_n - 1)
    gy = np.clip(((centroids[:, 1] - miny) / (maxy - miny + _EPS) * grid_n).astype(int), 0, grid_n - 1)
    tile_id = gx * grid_n + gy

    out = []
    for tid in np.unique(tile_id):
        fmask = tile_id == tid
        ftile = faces[fmask]
        used = np.unique(ftile.ravel())
        lookup = np.full(int(used.max()) + 1, -1, np.int64)
        lookup[used] = np.arange(len(used))
        vt = verts[used]
        out.append({
            "verts": vt,
            "faces": lookup[ftile],
            "vcolors": vcolors[used],
            "bbox_min": vt.min(axis=0),
            "bbox_max": vt.max(axis=0),
            "gx": int(tid // grid_n),
            "gy": int(tid % grid_n),
        })
    return out
