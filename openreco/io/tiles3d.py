"""3D Tiles (Cesium) tileset generation for streaming a georeferenced model in the browser.

The tile content is the model's glTF (.glb) in the project's local metric frame (UTM minus the
georef origin). The tileset's root `transform` places that local frame on the globe: an
East-North-Up basis at the origin's geodetic position, expressed in Earth-Centered-Earth-Fixed
(ECEF, EPSG:4978). The georeferencing math is numerically verifiable (origin -> lat/lon) without
a renderer; final Cesium placement also relies on gltfUpAxis=Z (our glb is Z-up).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def enu_to_ecef_transform(crs_epsg: int, origin: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """4x4 (row-major) mapping local ENU meters (relative to `origin` in the projected CRS) to ECEF.
    Returns (M, (lat_deg, lon_deg)) so the caller can verify placement."""
    from pyproj import Transformer

    to_geo = Transformer.from_crs(crs_epsg, 4326, always_xy=True)
    to_ecef = Transformer.from_crs(4326, 4978, always_xy=True)
    lon, lat, h = to_geo.transform(origin[0], origin[1], origin[2])
    ox, oy, oz = to_ecef.transform(lon, lat, h)

    lam, phi = math.radians(lon), math.radians(lat)
    east = np.array([-math.sin(lam), math.cos(lam), 0.0])
    north = np.array([-math.sin(phi) * math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi)])
    up = np.array([math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)])
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = east, north, up
    m[:3, 3] = [ox, oy, oz]
    return m, (lat, lon)


def _box(bmin: np.ndarray, bmax: np.ndarray) -> list[float]:
    """3D Tiles oriented bounding box [center, x-half-axis, y-half-axis, z-half-axis]."""
    c = (bmin + bmax) / 2.0
    h = (bmax - bmin) / 2.0
    return [float(c[0]), float(c[1]), float(c[2]),
            float(h[0]), 0.0, 0.0, 0.0, float(h[1]), 0.0, 0.0, 0.0, float(h[2])]


def write_tiled_tileset(out_path: Path, children: list[dict], crs_epsg: int | None,
                        origin: np.ndarray, root_min: np.ndarray, root_max: np.ndarray):
    """Multi-tile tileset: a root placing the local frame on the globe (ENU->ECEF when
    georeferenced, else identity) with one child per tile. `children` items:
    {uri, bbox_min, bbox_max}. Returns (lat, lon) | None."""
    if crs_epsg:
        m, latlon = enu_to_ecef_transform(crs_epsg, np.asarray(origin, float))
        transform = m.T.flatten().tolist()
    else:
        transform, latlon = np.eye(4).flatten().tolist(), None
    child_tiles = [{
        "boundingVolume": {"box": _box(np.asarray(c["bbox_min"]), np.asarray(c["bbox_max"]))},
        "geometricError": 0.0,
        "content": {"uri": c["uri"]},
    } for c in children]
    tileset = {
        "asset": {"version": "1.1", "gltfUpAxis": "Z"},
        "geometricError": float(np.linalg.norm(root_max - root_min)),
        "root": {
            "transform": transform,
            "boundingVolume": {"box": _box(np.asarray(root_min), np.asarray(root_max))},
            "geometricError": float(np.linalg.norm(root_max - root_min)) / 2.0,
            "refine": "ADD",
            "children": child_tiles,
        },
    }
    out_path.write_text(json.dumps(tileset, indent=2), encoding="utf-8")
    return latlon


def write_tileset(out_path: Path, glb_name: str, crs_epsg: int, origin: np.ndarray,
                  bbox_min: np.ndarray, bbox_max: np.ndarray) -> tuple[float, float]:
    """Write a 3D Tiles 1.1 tileset.json placing `glb_name` on the globe. Returns the (lat, lon)
    the local origin maps to (for verification/logging)."""
    m, (lat, lon) = enu_to_ecef_transform(crs_epsg, np.asarray(origin, float))
    center = (bbox_min + bbox_max) / 2.0
    half = (bbox_max - bbox_min) / 2.0
    box = [float(center[0]), float(center[1]), float(center[2]),
           float(half[0]), 0.0, 0.0, 0.0, float(half[1]), 0.0, 0.0, 0.0, float(half[2])]
    geom_err = float(np.linalg.norm(bbox_max - bbox_min))
    tileset = {
        "asset": {"version": "1.1", "gltfUpAxis": "Z"},   # our glb local frame is Z-up (ENU)
        "geometricError": geom_err,
        "root": {
            "transform": m.T.flatten().tolist(),          # column-major per the 3D Tiles spec
            "boundingVolume": {"box": box},
            "geometricError": 0.0,
            "refine": "ADD",
            "content": {"uri": glb_name},
        },
    }
    out_path.write_text(json.dumps(tileset, indent=2), encoding="utf-8")
    return lat, lon
