"""Rasterization helpers for DSM and orthophoto: grid a point cloud top-down and write GeoTIFF.

Cells take the highest point (DSM = top surface); the ortho takes that same top point's color.
Empty cells are filled by interpolation so small gaps don't punch holes. GeoTIFFs are written
with the project CRS when georeferenced, else as a plain (local-frame) raster with a warning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def grid_topdown(xy: np.ndarray, z: np.ndarray, rgb: np.ndarray, res: float, fill: bool = True):
    """Bin points onto a north-up grid at `res` meters/pixel.

    Returns (dsm[H,W] float32 with NaN holes, ortho[H,W,3] uint8, west, north) where west/north
    are the local-frame coordinates of the grid's top-left corner."""
    minx, miny = xy[:, 0].min(), xy[:, 1].min()
    maxx, maxy = xy[:, 0].max(), xy[:, 1].max()
    w = max(1, int(np.ceil((maxx - minx) / res)) + 1)
    h = max(1, int(np.ceil((maxy - miny) / res)) + 1)
    col = np.clip(((xy[:, 0] - minx) / res).astype(int), 0, w - 1)
    row = np.clip(((maxy - xy[:, 1]) / res).astype(int), 0, h - 1)  # north-up: y inverted

    dsm = np.full((h, w), np.nan, dtype=np.float32)
    ortho = np.zeros((h, w, 3), dtype=np.uint8)
    order = np.argsort(z)  # ascending; highest written last -> wins the cell
    dsm[row[order], col[order]] = z[order].astype(np.float32)
    ortho[row[order], col[order]] = rgb[order]

    if fill:
        _fill_holes(dsm, ortho)
    return dsm, ortho, float(minx), float(maxy)


def _fill_holes(dsm: np.ndarray, ortho: np.ndarray) -> None:
    from scipy.interpolate import griddata

    h, w = dsm.shape
    valid = ~np.isnan(dsm)
    if valid.sum() < 4 or valid.all():
        return
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.column_stack([xx[valid], yy[valid]])
    holes = np.column_stack([xx[~valid], yy[~valid]])
    # DSM: linear then nearest for anything outside the convex hull
    z = griddata(pts, dsm[valid], holes, method="linear")
    nn = griddata(pts, dsm[valid], holes, method="nearest")
    z[np.isnan(z)] = nn[np.isnan(z)]
    dsm[~valid] = z
    for c in range(3):
        ch = ortho[:, :, c]
        ortho[~valid, c] = griddata(pts, ch[valid], holes, method="nearest")


def reproject_geotiff(src: Path, dst: Path, dst_epsg: int) -> None:
    """Reproject a GeoTIFF to another CRS (output-CRS selection). Uses rasterio.warp."""
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(src) as s:
        dst_crs = f"EPSG:{dst_epsg}"
        transform, w, h = calculate_default_transform(s.crs, dst_crs, s.width, s.height, *s.bounds)
        profile = s.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=w, height=h)
        with rasterio.open(dst, "w", **profile) as d:
            for b in range(1, s.count + 1):
                reproject(source=rasterio.band(s, b), destination=rasterio.band(d, b),
                          src_transform=s.transform, src_crs=s.crs,
                          dst_transform=transform, dst_crs=dst_crs,
                          resampling=Resampling.bilinear)


def raster_to_png(path: Path, max_dim: int = 2000) -> bytes:
    """Render a GeoTIFF to PNG bytes for the 2D viewer (browsers can't show GeoTIFF directly).

    RGB(A) rasters (ortho) pass through; single-band rasters (DSM / vegetation index) get a 2–98
    percentile grayscale stretch with nodata made transparent. Large rasters are downscaled so the
    2D canvas stays responsive. Returns PNG bytes."""
    import io

    import rasterio
    from PIL import Image

    with rasterio.open(path) as ds:
        scale = min(1.0, max_dim / max(ds.width, ds.height))
        out_w, out_h = max(1, int(ds.width * scale)), max(1, int(ds.height * scale))
        nodata = ds.nodata
        if ds.count >= 3:
            arr = ds.read([1, 2, 3], out_shape=(3, out_h, out_w)).transpose(1, 2, 0)
            rgb = arr.astype(np.uint8) if arr.dtype == np.uint8 else _stretch(arr)
            alpha = np.full((out_h, out_w), 255, np.uint8)
            if ds.count >= 4:
                alpha = ds.read(4, out_shape=(out_h, out_w)).astype(np.uint8)
            img = Image.fromarray(np.dstack([rgb, alpha]))      # HxWx4 uint8 -> RGBA
        else:
            band = ds.read(1, out_shape=(out_h, out_w)).astype(np.float32)
            valid = np.isfinite(band)
            if nodata is not None:
                valid &= band != nodata
            gray = _stretch(band, valid)
            alpha = np.where(valid, 255, 0).astype(np.uint8)
            img = Image.fromarray(np.dstack([gray, gray, gray, alpha]))   # RGBA
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def raster_to_overlay(src: Path) -> tuple[bytes, list[list[float]]]:
    """Reproject a georeferenced raster to WGS84 and render it to PNG for a web-map image overlay.

    Returns (png_bytes, bounds) where bounds = [[south, west], [north, east]] in lat/lon (Leaflet
    order). Raises if the raster has no CRS (georeference it first)."""
    import os
    import tempfile

    import rasterio

    src = Path(src)
    with rasterio.open(src) as s:
        if s.crs is None:
            raise ValueError("raster has no CRS — georeference it first")
        if str(s.crs).upper() in ("EPSG:4326",):       # already WGS84
            png = raster_to_png(src)
            b = s.bounds
            return png, [[b.bottom, b.left], [b.top, b.right]]
    fd, name = tempfile.mkstemp(suffix=".tif")
    os.close(fd)                                        # close the handle so Windows can rewrite/delete
    tmp = Path(name)
    try:
        reproject_geotiff(src, tmp, 4326)
        png = raster_to_png(tmp)
        with rasterio.open(tmp) as d:
            b = d.bounds
        return png, [[b.bottom, b.left], [b.top, b.right]]
    finally:
        tmp.unlink(missing_ok=True)


def _stretch(arr: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Percentile (2–98) contrast stretch to uint8; per-channel for RGB, single for grayscale."""
    a = arr.astype(np.float32)
    if a.ndim == 3:
        out = np.zeros_like(a, np.uint8)
        for c in range(a.shape[2]):
            out[:, :, c] = _stretch(a[:, :, c])
        return out
    m = valid if valid is not None else np.isfinite(a)
    if not m.any():
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(a[m], [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    a = np.where(m, a, lo)                       # neutralize nodata/NaN before the cast
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def write_geotiff(path: Path, array: np.ndarray, west: float, north: float, res: float,
                  crs_epsg: int | None, nodata=None) -> None:
    import rasterio
    from rasterio.transform import from_origin

    if array.ndim == 2:
        array = array[np.newaxis, :, :]          # (1, H, W)
    else:
        array = np.moveaxis(array, 2, 0)         # (bands, H, W)
    bands, h, w = array.shape
    transform = from_origin(west, north, res, res)
    crs = f"EPSG:{crs_epsg}" if crs_epsg else None
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": bands,
        "dtype": array.dtype, "transform": transform, "crs": crs,
        "compress": "deflate",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
