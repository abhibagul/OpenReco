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
