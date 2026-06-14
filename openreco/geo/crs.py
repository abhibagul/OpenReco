"""Coordinate reference systems and transforms.

For UAV data we georeference into a projected, metric CRS (UTM by default) so that distances,
areas, DSM cells, and orthomosaic pixels are all in meters. pyproj/PROJ (MIT) handles the
datum/grid transforms; this module just picks a sensible CRS and converts geodetic GPS to it.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_crs(crs):
    from pyproj import CRS

    if hasattr(crs, "to_epsg"):
        return crs
    if isinstance(crs, int):
        return CRS.from_epsg(crs)
    return CRS.from_user_input(crs)                      # EPSG:xxxx, WKT, PROJ string, name


def _auth(obj) -> str | None:
    try:                                                # most reliable: the object's JSON id block
        ident = (obj.to_json_dict() or {}).get("id") or {}
        if ident.get("authority") and ident.get("code") is not None:
            return f"{ident['authority']}:{ident['code']}"
    except Exception:  # noqa: BLE001
        pass
    try:
        a = obj.to_authority()
        return f"{a[0]}:{a[1]}" if a else None
    except Exception:  # noqa: BLE001
        return None


def crs_info(crs) -> dict[str, Any]:
    """Full component breakdown of any coordinate reference system (EPSG code / WKT / PROJ / name).
    Mirrors a industry-standard CRS dialog: name, kind, datum, ellipsoid, prime meridian, units, axes,
    and (for projected CRS) the projection method + base geographic CRS."""
    c = _as_crs(crs)
    ax = c.axis_info[0] if c.axis_info else None
    info: dict[str, Any] = {
        "code": (f"EPSG:{c.to_epsg()}" if c.to_epsg() else _auth(c)),
        "name": c.name,
        "kind": c.type_name,                            # e.g. 'Geographic 2D CRS', 'Projected CRS'
        "is_geographic": c.is_geographic,
        "is_projected": c.is_projected,
        "datum": {"name": c.datum.name, "code": _auth(c.datum)} if c.datum else None,
        "ellipsoid": ({"name": c.ellipsoid.name, "code": _auth(c.ellipsoid),
                       "semi_major_m": c.ellipsoid.semi_major_metre,
                       "inverse_flattening": c.ellipsoid.inverse_flattening}
                      if c.ellipsoid else None),
        "prime_meridian": ({"name": c.prime_meridian.name, "code": _auth(c.prime_meridian),
                            "longitude": c.prime_meridian.longitude}
                           if c.prime_meridian else None),
        "unit": ({"name": ax.unit_name,
                  "code": (f"EPSG:{ax.unit_code}" if getattr(ax, "unit_code", None) else None)}
                 if ax else None),
        "axes": [{"name": a.name, "abbrev": a.abbrev, "direction": a.direction,
                  "unit": a.unit_name} for a in c.axis_info],
    }
    if c.is_projected:
        try:
            info["projection"] = c.coordinate_operation.method_name
        except Exception:  # noqa: BLE001
            info["projection"] = None
        geo = c.geodetic_crs
        info["base_crs"] = {"name": geo.name, "code": _auth(geo)} if geo else None
    return info


def search_crs(query: str, kind: str = "all", limit: int = 50) -> list[dict[str, Any]]:
    """Search the EPSG catalog for a CRS picker. `query` matches the name (case-insensitive) or an
    EPSG code; `kind` in {all, geographic, projected}. Returns [{code, name, kind}]."""
    from pyproj.database import query_crs_info

    types = None
    if kind == "geographic":
        from pyproj.enums import PJType
        types = [PJType.GEOGRAPHIC_2D_CRS, PJType.GEOGRAPHIC_3D_CRS]
    elif kind == "projected":
        from pyproj.enums import PJType
        types = [PJType.PROJECTED_CRS]
    q = query.strip().lower()
    out = []
    for ci in query_crs_info(auth_name="EPSG", pj_types=types):
        if q.isdigit():
            if ci.code == q:
                out.append({"code": f"EPSG:{ci.code}", "name": ci.name, "kind": ci.type.name})
        elif q in ci.name.lower():
            out.append({"code": f"EPSG:{ci.code}", "name": ci.name, "kind": ci.type.name})
        if len(out) >= limit:
            break
    return out


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
