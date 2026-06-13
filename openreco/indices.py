"""Vegetation indices (pure numpy).

Two families:
  - RGB-only indices (ExG, VARI, GLI) — usable on ordinary RGB drone orthos, widely used in
    agriculture when no NIR sensor is available.
  - NIR-based indices (NDVI, GNDVI) — the classic multispectral indices, computed when a NIR
    band is supplied (e.g. a multispectral ortho or an aligned NIR raster).

Inputs are per-band float arrays (reflectance-like; RGB scaled to [0,1]). Each function returns a
float index array. The stage picks bands from a raster and colorizes the result.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def exg(r, g, b):
    """Excess Green (2G - R - B) — highlights vegetation; RGB only."""
    return 2.0 * g - r - b


def vari(r, g, b):
    """Visible Atmospherically Resistant Index (G-R)/(G+R-B) — RGB only, ~[-1, 1]."""
    return (g - r) / (g + r - b + _EPS)


def gli(r, g, b):
    """Green Leaf Index (2G-R-B)/(2G+R+B) — RGB only."""
    return (2.0 * g - r - b) / (2.0 * g + r + b + _EPS)


def ndvi(nir, r):
    """Normalized Difference Vegetation Index (NIR-R)/(NIR+R) — needs NIR, ~[-1, 1]."""
    return (nir - r) / (nir + r + _EPS)


def gndvi(nir, g):
    """Green NDVI (NIR-G)/(NIR+G) — needs NIR."""
    return (nir - g) / (nir + g + _EPS)


# name -> (function, needs_nir, band-arg order)
REGISTRY = {
    "exg":   (exg,   False),
    "vari":  (vari,  False),
    "gli":   (gli,   False),
    "ndvi":  (ndvi,  True),
    "gndvi": (gndvi, True),
}


def compute(name: str, r=None, g=None, b=None, nir=None) -> np.ndarray:
    fn, needs_nir = REGISTRY[name]
    if needs_nir:
        if nir is None:
            raise ValueError(f"index {name!r} requires a NIR band")
        return fn(nir, r if name == "ndvi" else g)
    return fn(r, g, b)


def colorize(index: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Map an index array to an RGB uint8 image with a red->yellow->green ramp (low->high).
    Range defaults to the 2nd/98th percentiles of finite values."""
    finite = index[np.isfinite(index)]
    if finite.size == 0:
        return np.zeros((*index.shape, 3), np.uint8)
    if lo is None:
        lo = float(np.percentile(finite, 2))
    if hi is None:
        hi = float(np.percentile(finite, 98))
    norm = np.clip((index - lo) / (hi - lo + _EPS), 0, 1)
    stops = np.array([[0.0, 165, 0, 38], [0.5, 255, 255, 191], [1.0, 0, 104, 55]])  # RdYlGn
    rgb = np.zeros((*index.shape, 3), np.uint8)
    for c in range(3):
        rgb[:, :, c] = np.interp(norm, stops[:, 0], stops[:, c + 1]).astype(np.uint8)
    rgb[~np.isfinite(index)] = 0
    return rgb
