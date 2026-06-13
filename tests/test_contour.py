"""Marching-squares contour extraction (pure numpy — runs in CI)."""

from __future__ import annotations

import numpy as np

from openreco.geo.contour import contour_levels, contour_segments


def test_levels_strictly_inside_range():
    assert contour_levels(1567.8, 1947.4, 10.0)[0] == 1570.0
    assert contour_levels(1567.8, 1947.4, 10.0)[-1] == 1940.0
    assert contour_levels(5.0, 5.0, 1.0) == []          # zero range
    assert contour_levels(0.0, 100.0, 0.0) == []        # bad interval


def test_planar_ramp_produces_one_segment_per_row():
    # z increases with column: contour at a mid value is a vertical line crossing every row band
    z = np.tile(np.arange(5, dtype=float), (4, 1))      # shape (4,5), cols 0..4
    segs = contour_segments(z, 2.5)
    assert len(segs) == 3                                # 4 rows -> 3 cell-row bands
    # each segment should be near column 2.5 (where value crosses 2.5)
    for (c0, _), (c1, _) in segs:
        assert abs(c0 - 2.5) < 1e-6 and abs(c1 - 2.5) < 1e-6


def test_no_crossing_returns_empty():
    z = np.full((4, 4), 7.0)
    assert contour_segments(z, 3.0) == []               # level below all -> no crossing
    assert contour_segments(z, 99.0) == []              # level above all


def test_nan_cells_skipped():
    z = np.array([[0.0, 1.0], [np.nan, 1.0]])
    # the only cell contains a NaN corner -> skipped, no segments regardless of level
    assert contour_segments(z, 0.5) == []


def test_single_corner_above_emits_segment():
    # bottom-left corner above the level (case 1) -> one segment cutting the bottom-left cell
    z = np.array([[0.0, 0.0], [5.0, 0.0]])              # bl high
    segs = contour_segments(z, 2.5)
    assert len(segs) == 1
    (a, b) = segs[0]
    # crossing points lie on the left edge (col 0) and bottom edge (row 1)
    cols = {round(a[0], 6), round(b[0], 6)}
    rows = {round(a[1], 6), round(b[1], 6)}
    assert 0.0 in cols and 1.0 in rows
