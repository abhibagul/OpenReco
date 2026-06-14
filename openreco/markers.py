"""Coded-target (fiducial marker) detection via OpenCV ArUco/AprilTag.

Detects printed coded targets in images and returns each marker's id + sub-pixel center +
corners. Because every marker carries a unique id, the same target is trivially identified across
images — giving automatic, robust tie points / GCP observations without manual picking (paired
with surveyed marker coordinates, these feed the georef GCP path).
"""

from __future__ import annotations

import numpy as np

# friendly name -> OpenCV predefined dictionary attribute
_DICTS = {
    "4x4_50": "DICT_4X4_50", "5x5_100": "DICT_5X5_100", "6x6_250": "DICT_6X6_250",
    "apriltag_36h11": "DICT_APRILTAG_36h11", "aruco_original": "DICT_ARUCO_ORIGINAL",
}


def detect_markers(gray: np.ndarray, dictionary: str = "4x4_50") -> list[dict]:
    """Detect coded targets in a grayscale image. Returns [{id, center[x,y], corners[4][2]}]."""
    import cv2

    if dictionary not in _DICTS:
        raise ValueError(f"unknown dictionary {dictionary!r}; choices: {sorted(_DICTS)}")
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, _DICTS[dictionary]))
    detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    out: list[dict] = []
    if ids is not None:
        for c, i in zip(corners, ids.ravel()):
            pts = np.asarray(c).reshape(4, 2)
            ctr = pts.mean(axis=0)
            out.append({"id": int(i), "center": [float(ctr[0]), float(ctr[1])],
                        "corners": pts.tolist()})
    return out


def dictionaries() -> list[str]:
    """Friendly names of the supported coded-target dictionaries."""
    return list(_DICTS)


def marker_sheet_png(dictionary: str = "4x4_50", count: int = 24, marker_px: int = 240,
                     cols: int = 4) -> bytes:
    """A printable sheet of coded targets (ids 0..count-1) with labels + quiet zones. Returns PNG.

    Print at 100% (no scaling), measure the printed marker side, and survey each target's centre —
    then 'Detect markers' auto-fills the GCP table and you enter the surveyed world coordinates."""
    import cv2
    from PIL import Image, ImageDraw

    if dictionary not in _DICTS:
        raise ValueError(f"unknown dictionary {dictionary!r}; choices: {sorted(_DICTS)}")
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, _DICTS[dictionary]))
    quiet, gap, label, margin = marker_px // 6, 36, 26, 44
    cellw = marker_px + 2 * quiet + gap
    cellh = marker_px + 2 * quiet + gap + label
    rows = (count + cols - 1) // cols
    W, H = margin * 2 + cols * cellw, margin * 2 + rows * cellh + 30
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), f"OpenReco coded targets — {dictionary} — print at 100%", fill="black")
    for i in range(count):
        img = cv2.aruco.generateImageMarker(d, i, marker_px)
        tile = Image.new("RGB", (marker_px + 2 * quiet, marker_px + 2 * quiet), "white")
        tile.paste(Image.fromarray(img).convert("RGB"), (quiet, quiet))
        r, c = divmod(i, cols)
        x, y = margin + c * cellw, margin + 30 + r * cellh
        canvas.paste(tile, (x, y))
        draw.rectangle([x, y, x + tile.width - 1, y + tile.height - 1], outline="#cccccc")
        draw.text((x + 4, y + tile.height + 4), f"ID {i}", fill="black")
    import io
    buf = io.BytesIO()
    canvas.save(buf, "PNG")
    return buf.getvalue()
