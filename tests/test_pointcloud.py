"""Point-cloud / mesh I/O round-trips (numpy only — runs in CI without slice deps)."""

from __future__ import annotations

import numpy as np

from openreco.io.pointcloud import (
    read_mesh_ply,
    read_ply_xyzrgb,
    write_mesh_ply,
    write_obj,
    write_ply,
)


def test_ply_xyzrgb_roundtrip(tmp_path):
    xyz = np.array([[0, 0, 0], [1.5, 2.5, 3.5], [-4, 5, 6]], dtype=np.float64)
    rgb = np.array([[10, 20, 30], [255, 0, 128], [1, 2, 3]], dtype=np.uint8)
    p = tmp_path / "c.ply"
    write_ply(p, xyz, rgb)
    rxyz, rrgb = read_ply_xyzrgb(p)
    assert np.allclose(rxyz, xyz, atol=1e-4)
    assert np.array_equal(rrgb, rgb)


def test_ply_without_color(tmp_path):
    xyz = np.random.default_rng(0).random((50, 3))
    p = tmp_path / "c.ply"
    write_ply(p, xyz)
    rxyz, rrgb = read_ply_xyzrgb(p)
    assert rrgb is None
    assert np.allclose(rxyz, xyz, atol=1e-4)


def test_mesh_ply_roundtrip(tmp_path):
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    vcols = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], dtype=np.uint8)
    p = tmp_path / "m.ply"
    write_mesh_ply(p, verts, faces, vcols)
    rv, rf, rc = read_mesh_ply(p)
    assert np.allclose(rv, verts, atol=1e-4)
    assert np.array_equal(rf, faces)
    assert np.array_equal(rc, vcols)


def test_obj_is_one_indexed_with_colors(tmp_path):
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    vcols = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    p = tmp_path / "m.obj"
    write_obj(p, verts, faces, vcols)
    text = p.read_text()
    assert "f 1 2 3" in text          # OBJ faces are 1-indexed
    assert text.count("\nv ") + text.startswith("v ") >= 3
