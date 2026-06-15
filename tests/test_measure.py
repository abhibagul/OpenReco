"""Volume measurement — verified against analytic shapes on synthetic DSMs.
The DSM-based tests need rasterio (slice dep) -> skip in CI without it; the
polygon-bounded region tests are pure numpy and always run."""

from __future__ import annotations

import numpy as np
import pytest

from openreco.measure import measure_profile_region, measure_volume_region


def _grid_points(nx, ny, step, zfn):
    """A regular XY lawn of points with elevation zfn(x, y) — a synthetic surface."""
    xs = np.arange(nx) * step
    ys = np.arange(ny) * step
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    return np.column_stack([gx, gy, zfn(gx, gy)])


def test_region_flat_slab_above_zero():
    # flat surface at z=5; a 9x9 m footprint, base 0 -> ~5 * 81 = 405 m^3
    pts = _grid_points(41, 41, 0.25, lambda x, y: np.full_like(x, 5.0))
    poly = [[0.5, 0.5], [9.5, 0.5], [9.5, 9.5], [0.5, 9.5]]
    r = measure_volume_region(pts, poly, base=0.0, cell_size=0.5)
    assert r["area_m2"] == pytest.approx(81.0, rel=0.05)
    assert r["cut_m3"] == pytest.approx(405.0, rel=0.05)
    assert r["fill_m3"] == pytest.approx(0.0, abs=1e-6)
    assert r["net_m3"] == pytest.approx(405.0, rel=0.05)


def test_region_pyramid_above_plane_base():
    # cone/pyramid on flat ground at z=100; best-fit plane base sits at the ground,
    # so the measured volume is just the mound above it (positive, no fill).
    def zfn(x, y):
        d = np.hypot(x - 5.0, y - 5.0)
        return 100.0 + np.clip(5.0 - d, 0, None)        # 5 m peak, radius 5
    pts = _grid_points(101, 101, 0.1, zfn)
    poly = [[0.5, 0.5], [9.5, 0.5], [9.5, 9.5], [0.5, 9.5]]
    r = measure_volume_region(pts, poly, base="plane", cell_size=0.25)
    assert r["base"] == "plane"
    assert r["base_elevation"] == pytest.approx(100.0, abs=0.3)
    assert r["fill_m3"] == pytest.approx(0.0, abs=5.0)   # tiny plane-fit residual only
    assert r["cut_m3"] > 50.0                            # a real mound, not noise


def test_region_too_few_points_raises():
    with pytest.raises(ValueError):
        measure_volume_region(np.zeros((1, 3)), [[0, 0], [1, 0], [1, 1]], cell_size=0.5)


def test_profile_ramp_relief_and_slope():
    # surface tilting in x: z = 2*x over a 10 m line -> relief 20 m, slope 200%
    pts = _grid_points(101, 11, 0.1, lambda x, y: 2.0 * x)
    r = measure_profile_region(pts, [0.0, 0.5], [10.0, 0.5], n=50)
    assert r["length_m"] == pytest.approx(10.0, rel=1e-3)
    assert r["z_min"] == pytest.approx(0.0, abs=0.3)
    assert r["z_max"] == pytest.approx(20.0, abs=0.3)
    assert r["relief_m"] == pytest.approx(20.0, abs=0.5)
    assert r["slope_pct"] == pytest.approx(200.0, abs=5.0)
    assert len(r["samples"]) == 50
    assert r["covered"] > 40


def test_profile_coincident_endpoints_raise():
    pts = _grid_points(10, 10, 1.0, lambda x, y: np.zeros_like(x))
    with pytest.raises(ValueError):
        measure_profile_region(pts, [1.0, 1.0], [1.0, 1.0])


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
