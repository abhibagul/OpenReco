"""3D Tiles ECEF transform + tileset generation. Needs pyproj (slice dep)."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pyproj")

from openreco.io.tiles3d import enu_to_ecef_transform, write_tileset  # noqa: E402


def test_enu_origin_maps_back_to_site_latlon():
    # a UTM 13N easting/northing near the Colorado aerial site
    origin = np.array([246779.0, 4309989.0, 1994.0])
    m, (lat, lon) = enu_to_ecef_transform(32613, origin)
    assert 38.7 < lat < 39.1 and -108.1 < lon < -107.7      # back-projects to the site
    # transform maps local origin (0,0,0) to the ECEF point, and is a rigid ENU basis
    o_ecef = m @ np.array([0, 0, 0, 1.0])
    assert np.linalg.norm(o_ecef[:3]) > 6_000_000           # ~Earth radius magnitude (ECEF)
    east = (m @ np.array([1, 0, 0, 1.0]))[:3] - o_ecef[:3]  # 1 m east
    assert np.isclose(np.linalg.norm(east), 1.0, atol=1e-6)


def test_write_tileset_structure(tmp_path):
    p = tmp_path / "tileset.json"
    lat, lon = write_tileset(p, "model.glb", 32613, np.array([246779.0, 4309989.0, 1994.0]),
                             np.array([-50.0, -40.0, -5.0]), np.array([50.0, 40.0, 10.0]))
    t = json.loads(p.read_text())
    assert t["asset"]["gltfUpAxis"] == "Z"
    assert len(t["root"]["transform"]) == 16
    assert len(t["root"]["boundingVolume"]["box"]) == 12
    assert t["root"]["content"]["uri"] == "model.glb"
    assert 38.7 < lat < 39.1
