"""Ground classification (grid-min + height threshold) — pure numpy/scipy."""

from __future__ import annotations

import numpy as np

from openreco.classify_ground import (
    BUILDING,
    GROUND,
    NON_GROUND,
    VEGETATION,
    classify_ground,
    classify_points,
)


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


def test_multiclass_building_vs_vegetation():
    rng = np.random.default_rng(3)
    # ground plane
    g = np.column_stack([rng.random((4000, 2)) * 60, rng.normal(0, 0.03, 4000)])
    # a building: a flat (planar, low-roughness) roof slab 8 m up
    bx, by = (20 + rng.random((1500, 1)) * 12), (20 + rng.random((1500, 1)) * 12)
    roof = np.column_stack([bx, by, np.full((1500, 1), 8.0) + rng.normal(0, 0.02, (1500, 1))])
    # vegetation: a rough/scattered cloud (high local roughness) 5 m up
    veg = np.column_stack([40 + rng.random((1500, 1)) * 8, 40 + rng.random((1500, 1)) * 8,
                           5 + rng.normal(0, 1.2, (1500, 1))])
    xyz = np.vstack([g, roof, veg])
    cls = classify_points(xyz, cell_m=8.0, ground_thresh=0.5, knn=12, planarity_thresh=0.04)
    roof_cls = cls[4000:5500]
    veg_cls = cls[5500:7000]
    assert (roof_cls == BUILDING).mean() > 0.7         # planar roof -> building
    assert (veg_cls == VEGETATION).mean() > 0.7        # rough cloud -> vegetation
    assert (cls[:4000] == GROUND).mean() > 0.9
