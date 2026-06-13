"""Export system: detection, registry, and the new permissive writers."""

from __future__ import annotations

import struct

import numpy as np

from openreco.exporters import detect_kind, export_product, list_formats
from openreco.io.pointcloud import write_mesh_ply, write_ply


def _mesh(tmp_path):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    f = np.array([[0, 1, 2], [1, 3, 2]])
    c = np.full((4, 3), 100, np.uint8)
    p = tmp_path / "mesh.ply"
    write_mesh_ply(p, v, f, c)
    return p


def test_detect_kind(tmp_path):
    assert detect_kind(_mesh(tmp_path)) == "mesh"
    cloud = tmp_path / "c.ply"
    write_ply(cloud, np.random.default_rng(0).random((20, 3)))
    assert detect_kind(cloud) == "pointcloud"
    assert detect_kind(tmp_path / "x.tif") == "raster"
    assert detect_kind(tmp_path / "x.geojson") == "vector"


def test_list_formats(tmp_path):
    fmts = list_formats(_mesh(tmp_path))
    assert {"ply", "obj", "glb", "stl", "dxf"} <= set(fmts)


def test_export_mesh_stl(tmp_path):
    out = export_product(_mesh(tmp_path), "stl", tmp_path / "m.stl")
    data = out.read_bytes()
    assert len(data) == 80 + 4 + 2 * 50                  # header + count + 2 triangles * 50 bytes
    (n,) = struct.unpack_from("<I", data, 80)
    assert n == 2


def test_export_mesh_dxf(tmp_path):
    out = export_product(_mesh(tmp_path), "dxf", tmp_path / "m.dxf")
    text = out.read_text()
    assert "3DFACE" in text and text.count("3DFACE") == 2 and text.strip().endswith("EOF")


def test_export_cloud_csv(tmp_path):
    cloud = tmp_path / "c.ply"
    write_ply(cloud, np.array([[1.0, 2, 3], [4, 5, 6]]), np.array([[10, 20, 30], [40, 50, 60]], np.uint8))
    out = export_product(cloud, "csv", tmp_path / "c.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "x,y,z,r,g,b" and lines[1].startswith("1.0") and len(lines) == 3


def test_unsupported_format_message(tmp_path):
    try:
        export_product(_mesh(tmp_path), "usd", tmp_path / "x.usd")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "usd-core" in str(e)
