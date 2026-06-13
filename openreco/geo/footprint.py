"""Image ground footprints and overlap rasterization (pure numpy).

For an overlap/coverage map we project each image's frame corners onto a flat ground plane
(z = z0) using the camera pose + intrinsics, giving a ground quad per image, then count how
many quads cover each grid cell. The count is the classic photogrammetry "overlap" (how many
images see each spot) — the input that determines reconstruction reliability.
"""

from __future__ import annotations

import numpy as np


def ground_footprint(k: np.ndarray, r_w2c: np.ndarray, center: np.ndarray,
                     width: int, height: int, z0: float) -> np.ndarray | None:
    """Project the four image corners onto the plane z=z0. `k` is the 3x3 intrinsics,
    `r_w2c` the world->camera rotation, `center` the camera projection center (world).
    Returns a (4,2) array of ground XY corners, or None if the camera doesn't look at the plane."""
    kinv = np.linalg.inv(k)
    rt = r_w2c.T  # camera->world rotation
    corners_px = [(0, 0), (width, 0), (width, height), (0, height)]
    out = []
    for u, v in corners_px:
        d_world = rt @ (kinv @ np.array([u, v, 1.0]))
        if abs(d_world[2]) < 1e-9:
            return None
        s = (z0 - center[2]) / d_world[2]
        if s <= 0:  # plane is behind the camera
            return None
        p = center + s * d_world
        out.append(p[:2])
    return np.asarray(out)


def _inside_convex(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Vectorized point-in-convex-polygon for ordered `poly` (M,2) and `pts` (N,2)."""
    n = len(poly)
    inside = np.ones(len(pts), dtype=bool)
    sign = None
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        edge = b - a
        cross = edge[0] * (pts[:, 1] - a[1]) - edge[1] * (pts[:, 0] - a[0])
        # require all cross products to share a sign (consistent winding)
        s = cross >= 0
        if sign is None:
            sign = s.mean() >= 0.5  # dominant orientation for this polygon
        inside &= (cross >= 0) if sign else (cross <= 0)
    return inside


def rasterize_overlap(footprints: list[np.ndarray], west: float, north: float,
                      res: float, width: int, height: int) -> np.ndarray:
    """Count overlapping footprints per cell on a north-up grid whose top-left local corner is
    (west, north). Returns a (height, width) uint16 array."""
    count = np.zeros((height, width), dtype=np.uint16)
    # cell-center coordinates (north-up: row 0 is the top / max y)
    cols = west + (np.arange(width) + 0.5) * res
    rows = north - (np.arange(height) + 0.5) * res
    for poly in footprints:
        if poly is None or len(poly) < 3:
            continue
        # limit the test to the polygon's bounding box for speed
        minx, miny = poly.min(0)
        maxx, maxy = poly.max(0)
        c0 = max(0, int((minx - west) / res))
        c1 = min(width, int((maxx - west) / res) + 2)
        r0 = max(0, int((north - maxy) / res))
        r1 = min(height, int((north - miny) / res) + 2)
        if c0 >= c1 or r0 >= r1:
            continue
        sub_cols = cols[c0:c1]
        sub_rows = rows[r0:r1]
        sx, sy = np.meshgrid(sub_cols, sub_rows)
        sub_pts = np.column_stack([sx.ravel(), sy.ravel()])
        mask = _inside_convex(poly, sub_pts).reshape(len(sub_rows), len(sub_cols))
        count[r0:r1, c0:c1] += mask.astype(np.uint16)
    return count


def colormap_overlap(count: np.ndarray) -> np.ndarray:
    """Map an overlap count grid to an RGB uint8 image (dark -> blue -> green -> yellow -> red).
    Cells with zero coverage are dark. Used for the report/preview PNG (Pillow writes it)."""
    cmax = max(1, int(count.max()))
    norm = np.clip(count.astype(np.float64) / cmax, 0, 1)
    # control points: (pos, R, G, B)
    stops = np.array([
        [0.00, 30, 30, 45],
        [0.25, 40, 80, 200],
        [0.50, 40, 180, 120],
        [0.75, 240, 220, 60],
        [1.00, 220, 50, 50],
    ])
    rgb = np.zeros((*count.shape, 3), dtype=np.uint8)
    for c in range(3):
        rgb[:, :, c] = np.interp(norm, stops[:, 0], stops[:, c + 1]).astype(np.uint8)
    rgb[count == 0] = (15, 15, 22)  # uncovered
    return rgb
