"""Persisted measurements -> GeoJSON / DXF / CSV.

Measurements are picked in the viewport's local/world frame; the georef convention is
``CRS = world + origin`` (origin in projected-CRS meters, recorded in georef.json). So we add the
origin to get projected E/N, and for GeoJSON convert those to WGS84 lon/lat. DXF keeps projected
E/N (CAD expects meters). When the project is ungeoreferenced (no EPSG) coordinates stay local.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np


def _to_crs(points, origin) -> np.ndarray:
    o = np.asarray(origin if origin is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3) + o


def _lonlat(en: np.ndarray, epsg: int) -> np.ndarray:
    from openreco.geo.crs import crs_to_geodetic
    return crs_to_geodetic(en[:, 0], en[:, 1], int(epsg))


def _coords(points, epsg, origin) -> list[list[float]]:
    en = _to_crs(points, origin)
    if epsg:
        ll = _lonlat(en, epsg)
        return [[float(ll[i, 0]), float(ll[i, 1]), float(en[i, 2])] for i in range(len(en))]
    return [[float(en[i, 0]), float(en[i, 1]), float(en[i, 2])] for i in range(len(en))]


def measurements_to_geojson(measurements: list[dict], epsg: int | None = None,
                            origin=None) -> dict[str, Any]:
    """A FeatureCollection: distance/profile -> LineString, area/volume -> Polygon. Coordinates are
    WGS84 lon/lat when `epsg` is set, else local. Metrics ride along in feature properties."""
    feats = []
    for m in measurements:
        pts = m.get("points") or []
        if len(pts) < 2:
            continue
        coords = _coords(pts, epsg, origin)
        if m.get("type") in ("area", "vol"):
            geom = {"type": "Polygon", "coordinates": [coords + [coords[0]]]}
        else:
            geom = {"type": "LineString", "coordinates": coords}
        props = {"name": m.get("name"), "type": m.get("type")}
        for k, v in (m.get("result") or {}).items():
            if isinstance(v, (int, float, str)):
                props[k] = v
        feats.append({"type": "Feature", "geometry": geom, "properties": props})
    return {"type": "FeatureCollection", "crs_epsg": (int(epsg) if epsg else None), "features": feats}


_CSV_COLS = ["id", "name", "type", "length_m", "area_m2", "perimeter_m", "net_m3", "cut_m3",
             "fill_m3", "relief_m", "slope_pct", "center_e", "center_n", "center_lon", "center_lat"]


def measurements_to_csv(measurements: list[dict], epsg: int | None = None, origin=None) -> bytes:
    """One summary row per measurement: metrics + projected centroid (and lon/lat when georeferenced)."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(_CSV_COLS)
    for m in measurements:
        pts = m.get("points") or []
        ce = cn = clon = clat = ""
        if pts:
            c = _to_crs(pts, origin).mean(axis=0)
            ce, cn = round(float(c[0]), 3), round(float(c[1]), 3)
            if epsg:
                ll = _lonlat(c[None, :], epsg)[0]
                clon, clat = round(float(ll[0]), 8), round(float(ll[1]), 8)
        r = m.get("result") or {}
        w.writerow([m.get("id"), m.get("name"), m.get("type"),
                    r.get("length_m", ""), r.get("area_m2", ""), r.get("perimeter_m", ""),
                    r.get("net_m3", ""), r.get("cut_m3", ""), r.get("fill_m3", ""),
                    r.get("relief_m", ""), r.get("slope_pct", ""), ce, cn, clon, clat])
    return out.getvalue().encode("utf-8")


def profile_samples_csv(measurement: dict, epsg: int | None = None, origin=None) -> bytes:
    """Per-station CSV for one profile: distance, projected E/N (+ lon/lat), elevation."""
    r = measurement.get("result") or {}
    samples = r.get("samples") or []
    o = np.asarray(origin if origin is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["dist_m", "easting", "northing", "lon", "lat", "elev_m"])
    for s in samples:
        e, n = float(s["x"]) + o[0], float(s["y"]) + o[1]
        lon = lat = ""
        if epsg:
            ll = _lonlat(np.array([[e, n]]), epsg)[0]
            lon, lat = round(float(ll[0]), 8), round(float(ll[1]), 8)
        z = "" if s.get("z") is None else round(float(s["z"]) + o[2], 3)
        w.writerow([s.get("dist_m"), round(e, 3), round(n, 3), lon, lat, z])
    return out.getvalue().encode("utf-8")


def write_measurements(measurements: list[dict], fmt: str, path: str | Path,
                       epsg: int | None = None, origin=None) -> Path:
    path = Path(path)
    if fmt == "geojson":
        path.write_text(json.dumps(measurements_to_geojson(measurements, epsg, origin), indent=2),
                        encoding="utf-8")
    elif fmt == "dxf":
        from openreco.exporters import _vector_dxf
        _vector_dxf(measurements_to_geojson(measurements, None, origin), path)   # CAD = projected meters
    elif fmt == "csv":
        path.write_bytes(measurements_to_csv(measurements, epsg, origin))
    else:
        raise ValueError(f"unsupported measurement export format: {fmt!r}")
    return path
