"""Image metadata + quality reading for the ingest stage.

Pulls dimensions, camera make/model, focal length, and GPS (lat/lon/alt) from EXIF, and
computes a blur score (variance of the Laplacian) used for auto-culling. Pillow only —
permissive, no native build. Designed to degrade gracefully: a missing EXIF tag yields
None, never an exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ExifTags

_GPS_IFD = ExifTags.IFD.GPSInfo
_TAG = {v: k for k, v in ExifTags.TAGS.items()}  # name -> id
_GPSTAG = {v: k for k, v in ExifTags.GPSTAGS.items()}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class ImageInfo:
    name: str               # filename relative to the image dir
    width: int
    height: int
    make: str | None = None
    model: str | None = None
    focal_mm: float | None = None
    lat: float | None = None
    lon: float | None = None
    alt: float | None = None
    blur_score: float | None = None
    excluded: bool = False
    reason: str | None = None

    @property
    def has_gps(self) -> bool:
        return self.lat is not None and self.lon is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "make": self.make,
            "model": self.model,
            "focal_mm": self.focal_mm,
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "blur_score": self.blur_score,
            "excluded": self.excluded,
            "reason": self.reason,
        }


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _rational_to_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        try:
            return x.numerator / x.denominator
        except Exception:  # noqa: BLE001
            return float("nan")


def _dms_to_deg(dms: Any, ref: str | None) -> float | None:
    try:
        d, m, s = (_rational_to_float(v) for v in dms)
    except Exception:  # noqa: BLE001
        return None
    deg = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        deg = -deg
    return deg


def _read_gps(exif: Image.Exif) -> tuple[float | None, float | None, float | None]:
    try:
        gps = exif.get_ifd(_GPS_IFD)
    except Exception:  # noqa: BLE001
        return None, None, None
    if not gps:
        return None, None, None
    lat = _dms_to_deg(gps.get(_GPSTAG["GPSLatitude"]), gps.get(_GPSTAG["GPSLatitudeRef"]))
    lon = _dms_to_deg(gps.get(_GPSTAG["GPSLongitude"]), gps.get(_GPSTAG["GPSLongitudeRef"]))
    alt = None
    if _GPSTAG["GPSAltitude"] in gps:
        alt = _rational_to_float(gps[_GPSTAG["GPSAltitude"]])
        if gps.get(_GPSTAG.get("GPSAltitudeRef")) == 1:  # below sea level
            alt = -alt
    return lat, lon, alt


def blur_score(img: Image.Image, max_side: int = 512) -> float:
    """Variance of the Laplacian on a downsampled grayscale image. Higher = sharper.
    Downsampling keeps it fast and roughly scale-invariant across resolutions."""
    g = img.convert("L")
    w, h = g.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        g = g.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    a = np.asarray(g, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 3:
        return 0.0
    # 4-neighbour Laplacian via slicing (no SciPy dependency)
    lap = (
        -4 * a[1:-1, 1:-1]
        + a[:-2, 1:-1]
        + a[2:, 1:-1]
        + a[1:-1, :-2]
        + a[1:-1, 2:]
    )
    return float(lap.var())


def read_image_info(path: Path, *, compute_blur: bool = True) -> ImageInfo:
    with Image.open(path) as img:
        w, h = img.size
        info = ImageInfo(name=path.name, width=w, height=h)
        try:
            exif = img.getexif()
        except Exception:  # noqa: BLE001
            exif = None
        if exif:
            info.make = _clean(exif.get(_TAG.get("Make")))
            info.model = _clean(exif.get(_TAG.get("Model")))
            # focal length lives in the Exif sub-IFD
            try:
                sub = exif.get_ifd(ExifTags.IFD.Exif)
                fl = sub.get(_TAG.get("FocalLength"))
                if fl is not None:
                    info.focal_mm = _rational_to_float(fl)
            except Exception:  # noqa: BLE001
                pass
            info.lat, info.lon, info.alt = _read_gps(exif)
        if compute_blur:
            info.blur_score = blur_score(img)
    return info


def _clean(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip().strip("\x00").strip()
    return s or None
