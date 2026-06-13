"""Footprint projection + overlap rasterization (pure numpy — runs in CI)."""

from __future__ import annotations

import numpy as np

from openreco.geo.footprint import colormap_overlap, ground_footprint, rasterize_overlap


def test_nadir_footprint_geometry():
    # nadir camera at 100 m looking straight down; f=1000, 1000x1000 sensor -> cx=cy=500
    f = 1000.0
    k = np.array([[f, 0, 500], [0, f, 500], [0, 0, 1]])
    r_w2c = np.diag([1.0, -1.0, -1.0])     # 180 deg about x: world -Z maps to cam +Z (looking down)
    center = np.array([0.0, 0.0, 100.0])
    fp = ground_footprint(k, r_w2c, center, 1000, 1000, z0=0.0)
    assert fp is not None and fp.shape == (4, 2)
    # half-width = (alt - z0) * cx / f = 100 * 500 / 1000 = 50  -> full extent 100 m, centered at 0
    assert np.allclose(fp.mean(axis=0), [0, 0], atol=1e-6)
    assert np.isclose(fp[:, 0].max() - fp[:, 0].min(), 100.0, atol=1e-6)
    assert np.isclose(fp[:, 1].max() - fp[:, 1].min(), 100.0, atol=1e-6)


def test_camera_not_looking_at_plane_returns_none():
    k = np.array([[1000.0, 0, 500], [0, 1000, 500], [0, 0, 1]])
    # camera looking up (+z): plane below is behind it -> None
    r_w2c = np.eye(3)
    center = np.array([0.0, 0.0, 100.0])
    assert ground_footprint(k, r_w2c, center, 1000, 1000, z0=0.0) is None


def test_rasterize_overlap_counts_intersection():
    # two unit-ish squares overlapping in the middle of a 10x10 m grid at 1 m/px
    sq1 = np.array([[1, 1], [6, 1], [6, 6], [1, 6]], dtype=float)
    sq2 = np.array([[4, 4], [9, 4], [9, 9], [4, 9]], dtype=float)
    count = rasterize_overlap([sq1, sq2], west=0.0, north=10.0, res=1.0, width=10, height=10)
    assert count.shape == (10, 10)
    assert count.max() == 2                      # overlap region seen by both
    assert count.min() == 0                       # corners covered by neither
    assert (count == 2).sum() > 0


def test_colormap_shape_and_uncovered_color():
    count = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint16)
    rgb = colormap_overlap(count)
    assert rgb.shape == (2, 3, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == (15, 15, 22)       # uncovered cell is dark
