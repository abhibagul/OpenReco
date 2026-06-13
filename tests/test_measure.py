"""Volume measurement — verified against analytic shapes on synthetic DSMs.
Needs rasterio (slice dep) -> skips in CI without it."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rasterio")

from openreco.measure import measure_volume  # noqa: E402


def _write_dsm(path, z, res=1.0, nodata=-9999.0):
    import rasterio
    from rasterio.transform import from_origin

    z = np.asarray(z, dtype=np.float32)
    profile = {"driver": "GTiff", "height": z.shape[0], "width": z.shape[1], "count": 1,
               "dtype": "float32", "transform": from_origin(0, z.shape[0] * res, res, res),
               "crs": "EPSG:32613", "nodata": nodata}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(z, 1)


def test_flat_slab_volume_above_zero(tmp_path):
    # 10x10 cells at 2 m, flat elevation 5 m -> volume above base 0 = 5 * (20*20) = 2000 m^3
    p = tmp_path / "dsm.tif"
    _write_dsm(p, np.full((10, 10), 5.0), res=2.0)
    r = measure_volume(p, base=0.0)
    assert r["area_m2"] == pytest.approx(400.0)         # 100 cells * 4 m^2
    assert r["cut_m3"] == pytest.approx(2000.0)
    assert r["fill_m3"] == pytest.approx(0.0)


def test_base_min_makes_stockpile(tmp_path):
    # ground at 100, a single 1 m^2-cell pillar +10 m -> volume above min = 10 * 1 = 10 m^3
    z = np.full((5, 5), 100.0)
    z[2, 2] = 110.0
    p = tmp_path / "dsm.tif"
    _write_dsm(p, z, res=1.0)
    r = measure_volume(p, base="min")
    assert r["base_elevation"] == pytest.approx(100.0)
    assert r["cut_m3"] == pytest.approx(10.0)
    assert r["fill_m3"] == pytest.approx(0.0)


def test_cut_and_fill_split_around_base(tmp_path):
    z = np.array([[0.0, 0.0], [10.0, 10.0]])            # half at 0, half at 10
    p = tmp_path / "dsm.tif"
    _write_dsm(p, z, res=1.0)
    r = measure_volume(p, base=5.0)
    assert r["cut_m3"] == pytest.approx(10.0)            # two cells 5 above
    assert r["fill_m3"] == pytest.approx(10.0)           # two cells 5 below
    assert r["net_m3"] == pytest.approx(0.0)


def test_nodata_excluded(tmp_path):
    z = np.full((4, 4), 3.0)
    z[0, :] = -9999.0                                    # a nodata row
    p = tmp_path / "dsm.tif"
    _write_dsm(p, z, res=1.0)
    r = measure_volume(p, base=0.0)
    assert r["cells"] == 12                              # 4x4 minus one row
    assert r["cut_m3"] == pytest.approx(36.0)            # 12 cells * 3 m
