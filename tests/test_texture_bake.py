"""Texture-baking geometry: projection, best-image selection, atlas rasterization (pure numpy)."""

from __future__ import annotations

import numpy as np

from openreco.texture_bake import bake_face, project, select_best_image


def _nadir_cam(alt=100.0, f=1000.0, w=1000, h=1000):
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    r = np.diag([1.0, -1.0, -1.0])          # nadir: looking down -z world
    t = -r @ np.array([0.0, 0.0, alt])
    p = k @ np.hstack([r, t[:, None]])
    return {"P": p, "C": np.array([0.0, 0.0, alt]), "w": w, "h": h}


def test_project_principal_point():
    cam = _nadir_cam()
    px, z = project(cam["P"], np.array([[0.0, 0.0, 0.0]]))   # point under the camera
    assert np.allclose(px[0], [500, 500], atol=1e-6)        # projects to principal point
    assert z[0] > 0


def test_select_best_image_prefers_frontal():
    # one nadir camera; an up-facing ground face should be selected (frontal), index 0
    centroids = np.array([[0.0, 0.0, 0.0]])
    normals = np.array([[0.0, 0.0, 1.0]])                   # facing up toward the nadir cam
    best = select_best_image(centroids, normals, [_nadir_cam()])
    assert best[0] == 0


def test_select_best_image_out_of_view_is_minus_one():
    centroids = np.array([[10000.0, 10000.0, 0.0]])         # far outside the frustum
    normals = np.array([[0.0, 0.0, 1.0]])
    best = select_best_image(centroids, normals, [_nadir_cam()])
    assert best[0] == -1


def test_bake_face_fills_atlas_from_image():
    res = 16
    atlas = np.zeros((res, res, 3), np.uint8)
    filled = np.zeros((res, res), bool)
    # a face covering most of the atlas; image is solid red
    atlas_tri = np.array([[1.0, 1.0], [14.0, 1.0], [1.0, 14.0]])
    img_pts = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    image = np.zeros((64, 64, 3), np.float32)
    image[..., 0] = 200.0                                   # red
    bake_face(atlas, filled, atlas_tri, img_pts, image)
    assert filled.any()
    assert (atlas[filled][:, 0] == 200).all()               # filled texels are red
    assert (atlas[filled][:, 1] == 0).all()
