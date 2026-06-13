"""Mesh tiling — partition correctness (pure numpy)."""

from __future__ import annotations

import numpy as np

from openreco.tiling import tile_mesh


def _grid_mesh(n=20):
    """A flat n×n grid of vertices -> 2 triangles per cell."""
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    verts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)]).astype(float)
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            faces.append([a, a + 1, a + n])
            faces.append([a + 1, a + n + 1, a + n])
    return verts, np.array(faces), np.full((n * n, 3), 100, np.uint8)


def test_tiles_partition_preserves_all_faces():
    verts, faces, vcols = _grid_mesh(20)
    tiles = tile_mesh(verts, faces, vcols, grid_n=3)
    assert sum(len(t["faces"]) for t in tiles) == len(faces)   # every face assigned exactly once
    assert 1 < len(tiles) <= 9


def test_tile_faces_reindex_within_bounds():
    verts, faces, vcols = _grid_mesh(16)
    for t in tile_mesh(verts, faces, vcols, grid_n=4):
        assert t["faces"].max() < len(t["verts"])              # reindexed to local vertices
        assert t["faces"].min() >= 0
        assert len(t["vcolors"]) == len(t["verts"])
        assert np.all(t["bbox_max"] >= t["bbox_min"])


def test_single_tile_when_grid_one():
    verts, faces, vcols = _grid_mesh(10)
    tiles = tile_mesh(verts, faces, vcols, grid_n=1)
    assert len(tiles) == 1 and len(tiles[0]["faces"]) == len(faces)
