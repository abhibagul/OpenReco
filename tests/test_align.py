"""Georeferencing math: DLT triangulation, Umeyama similarity, rotation->quaternion.
Pure numpy — runs in CI."""

from __future__ import annotations

import numpy as np

from openreco.geo.align import rotmat_to_quat_xyzw, triangulate_dlt, umeyama_similarity


def _quat_to_rotmat(q):  # xyzw -> 3x3, for round-trip checking
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_triangulate_dlt_recovers_point():
    k = np.array([[800.0, 0, 320], [0, 800, 240], [0, 0, 1]])
    # two cameras looking at the origin region from different positions
    def proj(cam_center):
        r = np.eye(3)
        t = -r @ cam_center
        return k @ np.hstack([r, t[:, None]])
    p1 = proj(np.array([0.0, 0, -5]))
    p2 = proj(np.array([2.0, 0, -5]))
    x = np.array([0.3, -0.2, 0.0])
    uvs = []
    for p in (p1, p2):
        h = p @ np.append(x, 1.0)
        uvs.append((h[0] / h[2], h[1] / h[2]))
    rec = triangulate_dlt([p1, p2], uvs)
    assert np.allclose(rec, x, atol=1e-6)


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.random((10, 3)) * 10
    # known transform: scale 2.5, 30 deg about z, translation
    th = np.deg2rad(30)
    r = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    s, t = 2.5, np.array([100.0, -50.0, 7.0])
    dst = (s * (r @ src.T).T) + t
    rs, rr, rt = umeyama_similarity(src, dst)
    assert np.isclose(rs, s, atol=1e-6)
    assert np.allclose(rr, r, atol=1e-6)
    assert np.allclose(rt, t, atol=1e-5)
    # and it actually maps src onto dst
    assert np.allclose((rs * (rr @ src.T).T) + rt, dst, atol=1e-6)


def test_rotmat_to_quat_roundtrip():
    for axis_angle in ([0, 0, 0], [0, 0, np.pi / 2], [np.pi / 3, 0, 0], [0.4, -0.7, 1.1]):
        # build a rotation via Rodrigues from axis*angle
        v = np.array(axis_angle, float)
        ang = np.linalg.norm(v)
        if ang < 1e-9:
            r = np.eye(3)
        else:
            ax = v / ang
            kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            r = np.eye(3) + np.sin(ang) * kx + (1 - np.cos(ang)) * (kx @ kx)
        q = rotmat_to_quat_xyzw(r)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-9)
        assert np.allclose(_quat_to_rotmat(q), r, atol=1e-6)


def test_gcp_file_parsing(tmp_path):
    from openreco.stages.georef import _read_gcp_file

    p = tmp_path / "gcps.csv"
    p.write_text(
        "# name,X,Y,Z,image,u,v\n"
        "name,X,Y,Z,image,u,v\n"          # header row -> skipped (non-numeric X)
        "g1,246700.0,4310000.0,1900.0,DJI_0001.JPG,1234.5,678.9\n"
        "g1,246700.0,4310000.0,1900.0,DJI_0002.JPG,1100.0,700.0\n"
        "g2,246750.0,4310050.0,1905.0,DJI_0003.JPG,500.0,400.0\n",
        encoding="utf-8",
    )
    gcps = _read_gcp_file(p)
    assert set(gcps) == {"g1", "g2"}
    world, obs = gcps["g1"]
    assert np.allclose(world, [246700.0, 4310000.0, 1900.0])
    assert len(obs) == 2 and obs[0][0] == "DJI_0001.JPG"
