"""Align & merge chunks — ICP-register a second chunk onto the first and merge."""

from __future__ import annotations

import json
import logging

import numpy as np

from openreco.engine.context import DeviceInfo, RunContext, StageResult
from openreco.engine.stage import registered_types
from openreco.io.pointcloud import write_ply
from openreco.stages.merge import MergeChunks
from openreco.workflow import operations


def test_merge_chunks_registered():
    assert "merge_chunks" in registered_types()
    assert "Merge Chunks" in {o["op"] for o in operations()}


def _chunk(d, xyz, rgb, origin=(0.0, 0.0, 0.0)):
    d.mkdir(parents=True, exist_ok=True)
    write_ply(d / "points.ply", xyz, rgb)
    (d / "points.json").write_text(json.dumps({"origin": list(origin), "crs_epsg": None}), "utf-8")
    return StageResult(artifacts={"points": "points.ply", "meta": "points.json"})


def _ctx(tmp_path, inputs, input_dirs):
    return RunContext(stage_id="merge", stage_type="merge_chunks",
                      params=MergeChunks().default_params(), cache_dir=tmp_path / "out",
                      inputs=inputs, input_dirs=input_dirs, project_dir=tmp_path,
                      device=DeviceInfo(), logger=logging.getLogger("t"))


def test_merge_requires_two_clouds(tmp_path):
    ctx = _ctx(tmp_path, {}, {})
    (tmp_path / "out").mkdir()
    try:
        MergeChunks().run(ctx)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert ">= 2" in str(e)


def test_merge_aligns_and_combines(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.random((4000, 3)) * 10
    rgb = np.full((4000, 3), 100, np.uint8)
    # chunk B is chunk A rotated 4 deg about z + translated + noise
    th = np.deg2rad(4.0)
    rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    b = (a @ rz.T) + np.array([3.0, -2.0, 1.0]) + rng.normal(0, 0.02, a.shape)

    dirs = {"chunkA": tmp_path / "A", "chunkB": tmp_path / "B"}
    inputs = {"chunkA": _chunk(dirs["chunkA"], a, rgb), "chunkB": _chunk(dirs["chunkB"], b, rgb)}
    (tmp_path / "out").mkdir()
    res = MergeChunks().run(_ctx(tmp_path, inputs, dirs))
    assert res.metrics["chunks"] == 2
    assert res.metrics["total_points"] == 8000
    assert res.metrics["mean_rmse_m"] < 0.1           # ICP recovered the chunk offset
    assert (tmp_path / "out" / "merged.ply").exists()
