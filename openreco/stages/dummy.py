"""Dummy stages — Phase 0 proof that the engine schedules, caches, and resumes a DAG.

`dummy_generate` produces a list of integers; `dummy_sum` consumes an upstream generator
and writes their sum. They exercise: params -> cache key, upstream artifact wiring,
cache hits on re-run, and invalidation when a param changes. They will be deleted once the
real Phase 1 stages exist (kept only in tests as a fixture).
"""

from __future__ import annotations

import time
from typing import Any

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage


@register_stage
class DummyGenerate(Stage):
    type = "dummy_generate"
    version = "1"

    def default_params(self) -> dict[str, Any]:
        return {"n": 5, "start": 0, "sleep_ms": 0}

    def run(self, ctx: RunContext) -> StageResult:
        n = int(ctx.params["n"])
        start = int(ctx.params["start"])
        if ctx.params.get("sleep_ms"):
            time.sleep(ctx.params["sleep_ms"] / 1000.0)
        values = list(range(start, start + n))
        ctx.progress(1.0, f"generated {n} values")
        name = ctx.write_json("values.json", values)
        return StageResult(artifacts={"values": name}, metrics={"count": len(values)})

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        if result.metrics.get("count", 0) == 0:
            return [Issue(Severity.WARNING, "generated zero values", hint="increase params.n")]
        return []


@register_stage
class DummySum(Stage):
    type = "dummy_sum"
    version = "1"

    def run(self, ctx: RunContext) -> StageResult:
        if not ctx.inputs:
            raise ValueError("dummy_sum requires one upstream dummy_generate via inputs=[...]")
        total = 0
        per_input: dict[str, int] = {}
        for dep in ctx.inputs:
            values = ctx.read_input_json(dep, "values")
            per_input[dep] = sum(values)
            total += per_input[dep]
        ctx.write_json("sum.json", {"total": total, "per_input": per_input})
        return StageResult(artifacts={"sum": "sum.json"}, metrics={"total": total})
