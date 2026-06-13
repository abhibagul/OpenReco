"""Phase 0 engine tests: DAG ordering/cycles, caching, resume no-op, invalidation, diff."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from openreco import stages  # noqa: F401 — registers dummy stages
from openreco.engine.cache import compute_key
from openreco.engine.dag import Dag, DagError
from openreco.engine.manifest import StageSpec, load_manifest
from openreco.engine.runner import Runner, StageStatus


def write_project(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "project.toml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


# ---- DAG --------------------------------------------------------------------------------

def test_topo_order_respects_dependencies():
    specs = [
        StageSpec(id="c", type="dummy_sum", inputs=["a", "b"]),
        StageSpec(id="a", type="dummy_generate"),
        StageSpec(id="b", type="dummy_generate"),
    ]
    dag = Dag.build(specs)
    assert dag.order.index("a") < dag.order.index("c")
    assert dag.order.index("b") < dag.order.index("c")


def test_cycle_detected():
    specs = [
        StageSpec(id="a", type="dummy_sum", inputs=["b"]),
        StageSpec(id="b", type="dummy_sum", inputs=["a"]),
    ]
    with pytest.raises(DagError, match="cycle"):
        Dag.build(specs)


def test_unknown_dependency_rejected():
    specs = [StageSpec(id="a", type="dummy_sum", inputs=["ghost"])]
    with pytest.raises(DagError, match="unknown stage"):
        Dag.build(specs)


# ---- cache keys -------------------------------------------------------------------------

def test_cache_key_changes_with_params():
    k1 = compute_key("dummy_generate", "1", {"n": 5}, [])
    k2 = compute_key("dummy_generate", "1", {"n": 6}, [])
    assert k1 != k2


def test_cache_key_propagates_from_upstream():
    up1 = compute_key("dummy_generate", "1", {"n": 5}, [])
    up2 = compute_key("dummy_generate", "1", {"n": 6}, [])
    down1 = compute_key("dummy_sum", "1", {}, [up1])
    down2 = compute_key("dummy_sum", "1", {}, [up2])
    assert down1 != down2  # upstream change must invalidate downstream


def test_cache_key_order_independent_for_inputs():
    a = compute_key("x", "1", {}, ["k1", "k2"])
    b = compute_key("x", "1", {}, ["k2", "k1"])
    assert a == b


# ---- runner: execute, cache, resume, invalidate -----------------------------------------

DEMO = """
    [project]
    name = "t"
    [[stage]]
    id = "gen_a"
    type = "dummy_generate"
    params = { n = 5, start = 0 }
    [[stage]]
    id = "gen_b"
    type = "dummy_generate"
    params = { n = 3, start = 100 }
    [[stage]]
    id = "total"
    type = "dummy_sum"
    inputs = ["gen_a", "gen_b"]
"""


def test_first_run_executes_and_is_correct(tmp_path):
    write_project(tmp_path, DEMO)
    outcome = Runner(load_manifest(tmp_path)).run()
    assert outcome.ok
    statuses = {s.id: s.status for s in outcome.stages}
    assert all(st == StageStatus.EXECUTED for st in statuses.values())
    total = next(s for s in outcome.stages if s.id == "total")
    # sum(0..4)=10  +  sum(100..102)=303  = 313
    assert total.metrics["total"] == 313


def test_rerun_is_a_noop_all_cached(tmp_path):
    write_project(tmp_path, DEMO)
    Runner(load_manifest(tmp_path)).run()
    outcome2 = Runner(load_manifest(tmp_path)).run()
    assert outcome2.ok
    assert all(s.status == StageStatus.CACHED for s in outcome2.stages)


def test_param_change_invalidates_only_affected_stages(tmp_path):
    write_project(tmp_path, DEMO)
    Runner(load_manifest(tmp_path)).run()
    # change gen_a's n -> gen_a and total recompute; gen_b stays cached
    write_project(tmp_path, DEMO.replace("n = 5", "n = 7"))
    outcome = Runner(load_manifest(tmp_path)).run()
    status = {s.id: s.status for s in outcome.stages}
    assert status["gen_a"] == StageStatus.EXECUTED
    assert status["gen_b"] == StageStatus.CACHED
    assert status["total"] == StageStatus.EXECUTED
    total = next(s for s in outcome.stages if s.id == "total")
    # sum(0..6)=21 + 303 = 324
    assert total.metrics["total"] == 324


def test_force_recomputes(tmp_path):
    write_project(tmp_path, DEMO)
    Runner(load_manifest(tmp_path)).run()
    outcome = Runner(load_manifest(tmp_path), force=["gen_b"]).run()
    status = {s.id: s.status for s in outcome.stages}
    assert status["gen_b"] == StageStatus.EXECUTED
    assert status["gen_a"] == StageStatus.CACHED


def test_failed_upstream_skips_downstream(tmp_path):
    write_project(
        tmp_path,
        """
        [project]
        name = "fail"
        [[stage]]
        id = "bad"
        type = "dummy_sum"
        [[stage]]
        id = "downstream"
        type = "dummy_sum"
        inputs = ["bad"]
        """,
    )
    outcome = Runner(load_manifest(tmp_path)).run()
    assert not outcome.ok
    status = {s.id: s.status for s in outcome.stages}
    assert status["bad"] == StageStatus.FAILED       # dummy_sum with no inputs raises
    assert status["downstream"] == StageStatus.SKIPPED


def test_report_written(tmp_path):
    write_project(tmp_path, DEMO)
    outcome = Runner(load_manifest(tmp_path)).run()
    assert (outcome.run_dir / "report.html").exists()
    assert (outcome.run_dir / "run.json").exists()
