"""Ground classification (grid-min + height threshold) — pure numpy/scipy."""

from __future__ import annotations

import numpy as np

from openreco.classify_ground import GROUND, NON_GROUND, classify_ground


def test_flat_ground_all_ground():
    rng = np.random.default_rng(0)
    xy = rng.random((2000, 2)) * 50
    z = np.full(2000, 100.0) + rng.normal(0, 0.05, 2000)   # flat-ish ground
    xyz = np.column_stack([xy, z])
    cls = classify_ground(xyz, cell_m=5.0, ground_thresh=0.5)
    assert (cls == GROUND).mean() > 0.95


def test_building_points_are_non_ground():
    rng = np.random.default_rng(1)
    # ground plane at z=0 over 60x60 m
    g = np.column_stack([rng.random((4000, 2)) * 60, rng.normal(0, 0.05, 4000)])
    # a 10x10 m building, 8 m tall, centred — its roof/walls are well above local ground
    bx = 25 + rng.random((800, 1)) * 10
    by = 25 + rng.random((800, 1)) * 10
    bz = np.full((800, 1), 8.0) + rng.normal(0, 0.05, (800, 1))
    building = np.column_stack([bx, by, bz])
    xyz = np.vstack([g, building])
    cls = classify_ground(xyz, cell_m=5.0, ground_thresh=0.5)
    # the building points (last 800) should be non-ground
    assert (cls[-800:] == NON_GROUND).mean() > 0.9
    # most ground points should be ground
    assert (cls[:4000] == GROUND).mean() > 0.9


def test_codes_are_las_values():
    xyz = np.column_stack([np.arange(10), np.zeros(10), np.zeros(10)]).astype(float)
    cls = classify_ground(xyz, cell_m=5.0, ground_thresh=0.5)
    assert set(np.unique(cls)).issubset({GROUND, NON_GROUND})
