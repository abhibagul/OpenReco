"""DTM morphological ground filter + cross-section profile.
DTM math is pure numpy/scipy; profile needs rasterio (skips if absent)."""

from __future__ import annotations

import numpy as np
import pytest

from openreco.stages.dtm import morphological_dtm


def test_dtm_removes_narrow_spike_keeps_ground():
    # flat ground at 100 with a 1-cell +20 spike; window 5 -> spike erased, ground kept
    z = np.full((21, 21), 100.0)
    z[10, 10] = 120.0
    dtm = morphological_dtm(z, cells=5)
    assert dtm[10, 10] == pytest.approx(100.0)        # spike removed
    assert dtm[0, 0] == pytest.approx(100.0)          # ground preserved
    assert dtm.max() <= z.max()                        # opening never raises the surface


def test_dtm_preserves_broad_hill():
    # a broad smooth hill wider than the window should survive opening
    yy, xx = np.mgrid[0:40, 0:40]
    z = 100 + 30 * np.exp(-(((xx - 20) ** 2 + (yy - 20) ** 2) / (2 * 8.0 ** 2)))
    dtm = morphological_dtm(z, cells=3)
    assert dtm[20, 20] > 120.0                          # hill crest largely retained
    assert dtm.max() <= z.max() + 1e-9


def test_dtm_le_dsm_everywhere():
    rng = np.random.default_rng(0)
    z = 50 + rng.random((30, 30)) * 10
    dtm = morphological_dtm(z, cells=4)
    assert np.all(dtm <= z + 1e-9)                      # ground estimate never above the DSM


# ---- cross-section profile -------------------------------------------------------------

rasterio = pytest.importorskip("rasterio")


def _write_ramp(path, res=1.0):
    from rasterio.transform import from_origin

    # elevation ramps with x (column): z = col * 1.0
    h, w = 20, 30
    z = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
                       transform=from_origin(0, h * res, res, res), crs="EPSG:32613") as dst:
        dst.write(z, 1)


def test_profile_along_ramp_is_linear(tmp_path):
    from openreco.measure import measure_profile

    p = tmp_path / "dsm.tif"
    _write_ramp(p)
    # horizontal line across the ramp at mid height (world y stays inside raster)
    r = measure_profile(p, p_from=(0.5, 10.0), p_to=(29.5, 10.0), n=30)
    assert r["length_m"] == pytest.approx(29.0)
    zs = [s["z"] for s in r["samples"] if s["z"] is not None]
    assert zs[0] < zs[-1]                               # increases along +x
    assert r["relief_m"] == pytest.approx(zs[-1] - zs[0], abs=1.0)
