"""Coordinate-system introspection, search, and output-CRS reprojection. Needs pyproj/rasterio."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyproj")

from openreco.geo.crs import crs_info, search_crs  # noqa: E402


def test_wgs84_components():
    i = crs_info(4326)
    assert i["code"] == "EPSG:4326" and i["name"] == "WGS 84"
    assert i["is_geographic"] and not i["is_projected"]
    assert i["datum"]["code"] == "EPSG:6326"            # WGS84 ensemble
    assert i["ellipsoid"]["code"] == "EPSG:7030"
    assert i["prime_meridian"]["code"] == "EPSG:8901"
    assert i["unit"]["name"] == "degree"
    assert [a["abbrev"] for a in i["axes"]] == ["Lat", "Lon"]


def test_projected_crs_components():
    i = crs_info(32613)                                 # WGS84 / UTM 13N
    assert i["is_projected"] and i["unit"]["name"] == "metre"
    assert i["projection"] == "Transverse Mercator"
    assert i["base_crs"]["code"] == "EPSG:4326"


def test_crs_info_accepts_strings():
    assert crs_info("EPSG:4326")["name"] == "WGS 84"
    assert crs_info("WGS 84")["code"] == "EPSG:4326"


def test_search_by_code_and_name():
    assert any(r["code"] == "EPSG:32613" for r in search_crs("32613"))
    res = search_crs("UTM zone 13N", kind="projected", limit=80)
    assert any(r["code"] == "EPSG:32613" for r in res)


def test_output_crs_reprojection(tmp_path):
    pytest.importorskip("rasterio")
    import rasterio
    from rasterio.transform import from_origin

    from openreco.exporters import export_product
    # a small UTM 13N raster -> export to WGS84 (4326)
    src = tmp_path / "dsm.tif"
    arr = np.random.default_rng(0).random((20, 20)).astype("float32")
    with rasterio.open(src, "w", driver="GTiff", height=20, width=20, count=1, dtype="float32",
                       crs="EPSG:32613", transform=from_origin(246700, 4310000, 1, 1)) as d:
        d.write(arr, 1)
    out = export_product(src, "tif", tmp_path / "dsm_wgs84.tif", crs=4326)
    with rasterio.open(out) as r:
        assert str(r.crs) == "EPSG:4326"
