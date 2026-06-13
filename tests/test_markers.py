"""Coded-target (ArUco) detection. Needs opencv-contrib (slice dep) — skips otherwise."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from openreco.markers import detect_markers  # noqa: E402


def _canvas_with_markers(ids_positions, dict_name="4x4_50", size=120):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    canvas = np.full((800, 800), 255, np.uint8)
    centers = {}
    for mid, (cx, cy) in ids_positions:
        m = cv2.aruco.generateImageMarker(d, mid, size)
        x0, y0 = cx - size // 2, cy - size // 2
        canvas[y0:y0 + size, x0:x0 + size] = m
        centers[mid] = (cx, cy)
    return canvas, centers


def test_detects_ids_and_centers():
    canvas, centers = _canvas_with_markers([(3, (200, 200)), (11, (550, 300)), (42, (400, 600))])
    dets = detect_markers(canvas, "4x4_50")
    found = {d["id"]: d["center"] for d in dets}
    assert set(found) == {3, 11, 42}
    for mid, (cx, cy) in centers.items():
        assert abs(found[mid][0] - cx) < 2 and abs(found[mid][1] - cy) < 2   # sub-pixel-ish center


def test_no_markers_returns_empty():
    blank = np.full((300, 300), 255, np.uint8)
    assert detect_markers(blank, "4x4_50") == []


def test_unknown_dictionary_raises():
    with pytest.raises(ValueError):
        detect_markers(np.zeros((50, 50), np.uint8), "not_a_dict")
