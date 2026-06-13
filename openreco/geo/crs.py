"""Coordinate reference systems and transforms.

For UAV data we georeference into a projected, metric CRS (UTM by default) so that distances,
areas, DSM cells, and orthomosaic pixels are all in meters. pyproj/PROJ (MIT) handles the
datum/grid transforms; this module just picks a sensible CRS and converts geodetic GPS to it.
"""

from __future__ import annotations

import numpy as np


def utm_epsg_for(lat: float, lon: float) -> int:
    """EPSG code of the WGS84/UTM zone containing (lat, lon). North: 326xx, South: 327xx."""
    zone = int((lon + 180.0) / 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def geodetic_to_crs(
    lat: np.ndarray, lon: np.ndarray, alt: np.ndarray, epsg: int
) -> np.ndarray:
    """Transform WGS84 lat/lon/alt arrays to (x, y, z) in the target projected CRS (meters).
    Returns an [N, 3] array. Uses always_xy so inputs are (lon, lat)."""
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = tf.transform(np.asarray(lon, float), np.asarray(lat, float))
    z = np.asarray(alt, float)
    return np.column_stack([np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)])
