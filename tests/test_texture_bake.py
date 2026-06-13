"""Texture-baking geometry: projection, best-image selection, atlas rasterization (pure numpy)."""

from __future__ import annotations

import numpy as np

from openreco.texture_bake import (
    bake_face,
    bake_face_blend,
    project,
    select_best_image,
    select_top_k,
)


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


def test_select_top_k_orders_and_pads():
    # two nadir cameras; an up-facing face is seen by both -> top-2 both valid
    cams = [_nadir_cam(), _nadir_cam()]
    centroids = np.array([[0.0, 0.0, 0.0]])
    normals = np.array([[0.0, 0.0, 1.0]])
    idx, w = select_top_k(centroids, normals, cams, k=2)
    assert idx.shape == (1, 2)
    assert set(idx[0].tolist()) == {0, 1} and (w[0] > 0).all()
    # k larger than #cams that see it, with a far face -> padded with -1 / 0 weight
    far = np.array([[1e5, 1e5, 0.0]])
    idx2, w2 = select_top_k(far, normals, cams, k=2)
    assert (idx2[0] == -1).all() and (w2[0] == 0).all()


def test_bake_face_blend_averages_two_images():
    res = 16
    accum = np.zeros((res, res, 3), np.float64)
    wsum = np.zeros((res, res), np.float64)
    atlas_tri = np.array([[1.0, 1.0], [14.0, 1.0], [1.0, 14.0]])
    verts3 = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    # identity-ish projection: P maps the 3 verts onto image pixels directly
    P = np.array([[40.0, 0, 0, 5], [0, 40.0, 0, 5], [0, 0, 0, 1]])
    red = np.zeros((64, 64, 3), np.float32)
    red[..., 0] = 200
    blue = np.zeros((64, 64, 3), np.float32)
    blue[..., 2] = 100
    samples = [
        {"P": P, "verts3": verts3, "image": red, "gain": np.ones(3, np.float32), "weight": 1.0},
        {"P": P, "verts3": verts3, "image": blue, "gain": np.ones(3, np.float32), "weight": 1.0},
    ]
    bake_face_blend(accum, wsum, atlas_tri, samples)
    filled = wsum > 0
    assert filled.any()
    blended = accum[filled] / wsum[filled, None]
    # equal-weight blend of red(200,0,0) and blue(0,0,100) -> (100,0,50)
    assert np.allclose(blended.mean(axis=0), [100, 0, 50], atol=5)
