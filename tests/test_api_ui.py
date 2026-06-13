"""Engine<->UI contract: list-param serialization, stage introspection, run events/cancel,
artifact paths. All on dummy stages (CI-safe)."""

from __future__ import annotations

import openreco
from openreco import Project, stage_info
from openreco.engine.manifest import load_manifest


def _demo(tmp_path):
    return (Project.create(tmp_path, name="ui")
            .add_stage("gen_a", "dummy_generate", params={"n": 5})
            .add_stage("gen_b", "dummy_generate", params={"n": 3, "start": 100})
            .add_stage("total", "dummy_sum", inputs=["gen_a", "gen_b"]))


def test_list_param_roundtrips(tmp_path):
    # A: the serialization bug — list/nested params must survive save -> load
    p = Project.create(tmp_path, name="lp")
    p.add_stage("indices", "indices", inputs=["ortho"],
                params={"indices": ["exg", "vari", "gli"], "nir_band": 4})
    reloaded = load_manifest(p.save())
    assert reloaded.stages[0].params["indices"] == ["exg", "vari", "gli"]
    assert reloaded.stages[0].params["nir_band"] == 4


def test_stage_info_schema():
    # B: UI palette + parameter panels
    infos = {s["type"]: s for s in stage_info()}
    assert "sfm" in infos and "ingest" in infos
    sfm = infos["sfm"]
    assert "default_params" in sfm and "matcher" in sfm["default_params"]
    assert "version" in sfm and "deterministic" in sfm
    # single-stage form
    one = openreco.stage_info("ingest")
    assert one["type"] == "ingest" and "image_dir" in one["default_params"]


def test_run_emits_events(tmp_path):
    # C: live progress events
    events = []
    out = _demo(tmp_path).run(on_event=events.append)
    kinds = [e["event"] for e in events]
    assert "stage_start" in kinds and "stage_done" in kinds and kinds[-1] == "run_done"
    done = [e for e in events if e["event"] == "stage_done"]
    assert {e["id"] for e in done} == {"gen_a", "gen_b", "total"}
    assert out.ok


def test_cancel_stops_run(tmp_path):
    # C: cooperative cancellation -> downstream stages are cancelled
    out = _demo(tmp_path).run(cancel=lambda: True)
    statuses = {s.id: s.status.value for s in out.stages}
    assert all(v == "cancelled" for v in statuses.values())
    assert not out.ok


def test_artifacts_exposed_with_paths(tmp_path):
    # D: layer-tree needs per-stage artifacts as absolute paths
    out = _demo(tmp_path).run()
    total = out.stage("total")
    assert "sum" in total.artifacts
    import os
    assert os.path.isabs(total.artifacts["sum"]) and total.artifacts["sum"].endswith("sum.json")
    assert os.path.exists(total.artifacts["sum"])
