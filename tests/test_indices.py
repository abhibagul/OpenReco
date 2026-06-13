"""Vegetation index math (pure numpy)."""

from __future__ import annotations

import numpy as np

from openreco import indices as veg


def test_exg_highlights_green():
    # ExG = 2G - R - B: pure green = 2, neutral gray = 0, green > gray
    assert veg.exg(0.0, 1.0, 0.0) == 2.0
    assert veg.exg(0.5, 0.5, 0.5) == 0.0
    assert veg.exg(0.1, 0.8, 0.1) > veg.exg(0.5, 0.5, 0.5)


def test_vari_range_and_sign():
    assert np.isclose(veg.vari(0.0, 1.0, 0.0), 1.0)        # all green
    assert veg.vari(1.0, 0.0, 0.0) < 0                     # all red -> negative


def test_ndvi_known_value():
    assert np.isclose(veg.ndvi(0.8, 0.2), 0.6, atol=1e-4)  # (0.8-0.2)/(0.8+0.2)
    assert np.isclose(veg.gndvi(0.8, 0.4), (0.4) / (1.2), atol=1e-4)


def test_compute_requires_nir_for_ndvi():
    try:
        veg.compute("ndvi", r=np.zeros(4), g=np.zeros(4), b=np.zeros(4), nir=None)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_colorize_shape_and_dtype():
    idx = np.linspace(-1, 1, 100).reshape(10, 10)
    rgb = veg.colorize(idx)
    assert rgb.shape == (10, 10, 3) and rgb.dtype == np.uint8
    # high index -> greenish (G dominant), low -> reddish (R dominant); cast to int (avoid uint8 wrap)
    hi, lo = rgb[-1, -1].astype(int), rgb[0, 0].astype(int)
    assert hi[1] > hi[0]      # green channel dominates at high index
    assert lo[0] > lo[1]      # red channel dominates at low index
