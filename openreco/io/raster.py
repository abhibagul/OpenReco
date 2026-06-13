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
