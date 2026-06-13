"""Panorama homography + stitch core (pure numpy/scipy)."""

from __future__ import annotations

import numpy as np

from openreco.panorama import _apply_h, homography_dlt, ransac_homography, stitch


def _grid(n=6, step=20):
    xs, ys = np.meshgrid(np.arange(n) * step, np.arange(n) * step)
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def test_homography_dlt_recovers_known():
    h_true = np.array([[1.1, 0.05, 12.0], [0.02, 0.95, -8.0], [1e-4, 2e-4, 1.0]])
    src = _grid()
    dst = _apply_h(h_true, src)
    h = homography_dlt(src, dst)
    assert np.allclose(_apply_h(h, src), dst, atol=1e-6)


def test_ransac_homography_rejects_outliers():
    rng = np.random.default_rng(0)
    h_true = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]])  # translation
    src = _grid(8)
    dst = _apply_h(h_true, src)
    # corrupt 25% with gross outliers
    bad = rng.choice(len(src), len(src) // 4, replace=False)
    dst[bad] += rng.uniform(-200, 200, (len(bad), 2))
    h, inl = ransac_homography(src, dst, thresh=2.0)
    assert inl.sum() >= len(src) - len(bad) - 1
    assert np.allclose(h[:, 2][:2], [10, 5], atol=1.0)      # recovered translation


def test_stitch_single_image_identity():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)
    canvas, mask = stitch([img], [np.eye(3)])
    assert canvas.shape[0] >= 39 and canvas.shape[1] >= 59
    assert mask.mean() > 0.9                                 # nearly all filled
    # an interior pixel should match the source
    assert np.allclose(canvas[20, 30], img[20, 30], atol=2)


def test_stitch_two_overlapping_blend():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)
    # second image shifted right by 30 px in the reference frame
    t = np.array([[1.0, 0, 30], [0, 1, 0], [0, 0, 1.0]])
    canvas, mask = stitch([img, img], [np.eye(3), t])
    assert canvas.shape[1] >= 89                             # widened by the shift
    assert mask.mean() > 0.5
