"""Python API tests — programmatic build, run, diff, save/round-trip. Uses dummy stages so
they run in CI without pycolmap/GDAL."""

from __future__ import annotations

import openreco
from openreco import Project
from openreco.engine.manifest import load_manifest
from openreco.engine.runner import StageStatus


def _demo(tmp_path):
    return (Project.create(tmp_path, name="api-demo")
            .add_stage("gen_a", "dummy_generate", params={"n": 5, "start": 0})
            .add_stage("gen_b", "dummy_generate", params={"n": 3, "start": 100})
            .add_stage("total", "dummy_sum", inputs=["gen_a", "gen_b"]))


def test_registered_stages_exposed():
    names = openreco.registered_stages()
    assert {"ingest", "sfm", "georef", "mvs", "mesh", "dsm", "ortho", "export"} <= set(names)


def test_programmatic_build_and_run(tmp_path):
    proj = _demo(tmp_path)
    outcome = proj.run()
    assert outcome.ok
    assert outcome.stage("total").metrics["total"] == 313  # sum(0..4)+sum(100..102)
    assert outcome.report.exists()
    # re-run is a no-op
    assert all(s.status == StageStatus.CACHED for s in proj.run().stages)


def test_duplicate_stage_id_rejected(tmp_path):
    proj = Project.create(tmp_path).add_stage("a", "dummy_generate")
    try:
        proj.add_stage("a", "dummy_generate")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_force_recomputes_one_stage(tmp_path):
    proj = _demo(tmp_path)
    proj.run()
    out = proj.run(force=["gen_b"])
    status = {s.id: s.status for s in out.stages}
    assert status["gen_b"] == StageStatus.EXECUTED
    assert status["gen_a"] == StageStatus.CACHED


def test_diff_detects_param_change(tmp_path):
    a = _demo(tmp_path / "a")
    b = _demo(tmp_path / "b")
    b.stages[0] = b.stages[0].__class__(id="gen_a", type="dummy_generate",
                                        params={"n": 7, "start": 0}, inputs=[])
    d = a.diff(b)
    assert d["gen_a"]["change"] == "modified"
    assert d["gen_b"]["change"] == "same"
    assert d["total"]["change"] == "modified"   # downstream of gen_a


def test_save_roundtrip(tmp_path):
    proj = _demo(tmp_path)
    proj.manifest.crs = "EPSG:32633"
    path = proj.save()
    assert path.exists()
    reloaded = load_manifest(path)
    assert reloaded.name == "api-demo"
    assert reloaded.crs == "EPSG:32633"
    assert [s.id for s in reloaded.stages] == ["gen_a", "gen_b", "total"]
    assert reloaded.stages[0].params == {"n": 5, "start": 0}
    assert reloaded.stages[2].inputs == ["gen_a", "gen_b"]
