"""Clean stage — statistical outlier removal (points) + component filtering (mesh)."""

from __future__ import annotations

import json
import logging

import numpy as np

from openreco.engine.context import DeviceInfo, RunContext, StageResult
from openreco.engine.stage import registered_types
from openreco.io.pointcloud import write_mesh_ply, write_ply
from openreco.stages.clean import Clean
from openreco.workflow import operations


def _ctx(tmp_path, params, inputs, dirs):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return RunContext(stage_id="cl", stage_type="clean", params={**Clean().default_params(), **params},
                      cache_dir=out, inputs=inputs, input_dirs=dirs, project_dir=tmp_path,
                      device=DeviceInfo(), logger=logging.getLogger("t"))


def test_clean_registered():
    assert "clean" in registered_types()
    assert {"Clean Point Cloud", "Clean Mesh"} <= {o["op"] for o in operations()}


def test_clean_points_removes_outliers(tmp_path):
    rng = np.random.default_rng(0)
    blob = rng.normal(0, 1, (4000, 3))
    outliers = rng.uniform(-40, 40, (250, 3))
    xyz = np.vstack([blob, outliers]).astype("float32")
    src = tmp_path / "src"
    src.mkdir()
    write_ply(src / "points.ply", xyz, np.full((len(xyz), 3), 200, np.uint8))
    (src / "points.json").write_text(json.dumps({"origin": [0, 0, 0], "crs_epsg": None}), "utf-8")
    res = Clean().run(_ctx(tmp_path, {"mode": "points"},
                           {"src": StageResult(artifacts={"points": "points.ply", "meta": "points.json"})},
                           {"src": src}))
    assert res.metrics["kind"] == "points"
    assert res.metrics["out"] < res.metrics["in"]          # some outliers removed
    assert 1.0 < res.metrics["removed_pct"] < 20.0         # but not most of the cloud


def test_clean_mesh_drops_small_islands(tmp_path):
    gv, gf = [], []
    for i in range(11):
        for j in range(11):
            gv.append([i, j, 0])
    def idx(i, j):
        return i * 11 + j
    for i in range(10):
        for j in range(10):
            a, b, c, e = idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1)
            gf += [[a, b, c], [b, e, c]]
    base = len(gv)
    gv += [[500, 500, 9], [501, 500, 9], [500, 501, 9]]      # a far-away 1-face island
    gf.append([base, base + 1, base + 2])
    ms = tmp_path / "ms"
    ms.mkdir()
    write_mesh_ply(ms / "mesh.ply", np.array(gv, float), np.array(gf))
    res = Clean().run(_ctx(tmp_path, {"mode": "mesh"},
                           {"m": StageResult(artifacts={"mesh": "mesh.ply"})}, {"m": ms}))
    assert res.metrics["kind"] == "mesh" and res.metrics["components"] == 2
    assert res.metrics["faces_out"] == 200                  # island dropped, main grid kept
