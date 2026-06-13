"""Geo + raster helpers. Needs pyproj/rasterio (slice deps) — skips where unavailable."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyproj")
pytest.importorskip("rasterio")

from openreco.geo.crs import geodetic_to_crs, utm_epsg_for  # noqa: E402
from openreco.io.raster import grid_topdown, write_geotiff  # noqa: E402


def test_utm_zone_selection():
    assert utm_epsg_for(51.5, -0.1) == 32630   # London, N hemisphere
    assert utm_epsg_for(-33.87, 151.2) == 32756  # Sydney, S hemisphere


def test_geodetic_to_crs_shape_and_finiteness():
    lat = np.array([51.50, 51.51])
    lon = np.array([-0.10, -0.11])
    alt = np.array([10.0, 12.0])
    out = geodetic_to_crs(lat, lon, alt, utm_epsg_for(51.5, -0.1))
    assert out.shape == (2, 3)
    assert np.isfinite(out).all()
    assert abs(out[0, 2] - 10.0) < 1e-6  # z passes through


def test_grid_topdown_and_geotiff(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.random((500, 3)) * np.array([10, 10, 2])
    rgb = rng.integers(0, 256, (500, 3), dtype=np.uint8)
    dsm, ortho, west, north = grid_topdown(xyz, xyz[:, 2], rgb, res=0.5)
    assert dsm.ndim == 2 and ortho.shape[2] == 3
    assert not np.isnan(dsm).any()  # holes filled

    import rasterio

    p = tmp_path / "dsm.tif"
    write_geotiff(p, dsm, west, north, 0.5, crs_epsg=32630, nodata=float("nan"))
    with rasterio.open(p) as ds:
        assert ds.count == 1
        assert str(ds.crs) == "EPSG:32630"
        assert ds.width == dsm.shape[1] and ds.height == dsm.shape[0]
