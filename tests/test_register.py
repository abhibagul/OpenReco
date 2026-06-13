"""ICP / Kabsch rigid registration (pure numpy/scipy)."""

from __future__ import annotations

import numpy as np

from openreco.register_cloud import apply_transform, icp, kabsch


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_kabsch_recovers_rigid_transform():
    rng = np.random.default_rng(0)
    p = rng.random((100, 3)) * 10
    r_true, t_true = _rot_z(0.4), np.array([2.0, -1.0, 0.5])
    q = apply_transform(p, r_true, t_true)
    r, t = kabsch(p, q)
    assert np.allclose(r, r_true, atol=1e-9)
    assert np.allclose(t, t_true, atol=1e-9)
    assert np.allclose(apply_transform(p, r, t), q, atol=1e-9)


def test_icp_converges_small_transform():
    rng = np.random.default_rng(1)
    src = rng.random((3000, 3)) * 20
    r_true, t_true = _rot_z(0.06), np.array([0.4, -0.3, 0.2])
    dst = apply_transform(src, r_true, t_true)
    res = icp(src, dst)                              # identity init OK for a small transform
    assert res["rmse"] < 1e-3
    assert np.allclose(res["R"], r_true, atol=1e-2)
    assert np.allclose(res["t"], t_true, atol=1e-2)


def test_icp_with_centroid_init_handles_large_offset():
    rng = np.random.default_rng(2)
    src = rng.random((3000, 3)) * 20
    r_true, t_true = _rot_z(0.05), np.array([50.0, -40.0, 10.0])   # big translation
    dst = apply_transform(src, r_true, t_true)
    init = (np.eye(3), dst.mean(0) - src.mean(0))    # centroid pre-alignment
    res = icp(src, dst, init=init)
    assert res["rmse"] < 1e-2
    assert np.allclose(res["t"], t_true, atol=1.0)
