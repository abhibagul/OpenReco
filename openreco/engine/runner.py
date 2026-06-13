"""The Runner — orchestrates a manifest into a reproducible run.

Walks the DAG in topological order, computes each stage's content-addressed key from its
params + upstream keys, skips on cache hit (checkpoint/resume), otherwise executes the
stage into its cache dir and finalizes it. Records a run.json + report for auditability.
"""

from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from openreco import __version__
from openreco.engine.cache import Cache, compute_key
from openreco.engine.context import DeviceInfo, Issue, RunContext, Severity, StageResult
from openreco.engine.dag import Dag
from openreco.engine.manifest import Manifest
from openreco.engine.stage import get_stage

logger = logging.getLogger("openreco")


class StageStatus(str, Enum):
    CACHED = "cached"      # skipped — content-address already computed
    EXECUTED = "executed"  # ran fresh
    FAILED = "failed"
    SKIPPED = "skipped"    # not run because an upstream failed
    CANCELLED = "cancelled"


@dataclass
class StageRun:
    id: str
    type: str
    key: str
    status: StageStatus
    seconds: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)  # resolved params (reproducibility)
    error: str | None = None


@dataclass
class RunOutcome:
    project: str
    started: str
    finished: str
    ok: bool
    stages: list[StageRun]
    run_dir: Path

    @property
    def report(self) -> Path:
        """Path to this run's HTML report."""
        return self.run_dir / "report.html"

    def stage(self, stage_id: str) -> StageRun:
        """Look up a stage's run record by id."""
        for s in self.stages:
            if s.id == stage_id:
                return s
        raise KeyError(f"no stage {stage_id!r} in this run")

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "openreco_version": __version__,
            "started": self.started,
            "finished": self.finished,
            "ok": self.ok,
            "platform": {
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "stages": [
                {
                    "id": s.id,
                    "type": s.type,
                    "key": s.key,
                    "status": s.status.value,
                    "seconds": round(s.seconds, 4),
                    "metrics": s.metrics,
                    "issues": [i.to_dict() for i in s.issues],
                    "params": s.params,
                    "error": s.error,
                }
                for s in self.stages
            ],
        }


def _detect_device() -> DeviceInfo:
    # Phase 0: CPU only. CUDA/Metal probes land with openreco/compute in Phase 1+.
    return DeviceInfo(has_cuda=False, has_metal=False, cpu_count=os.cpu_count() or 1)


def compute_keys(manifest: Manifest) -> dict[str, dict[str, Any]]:
    """Compute each stage's content-address key for a manifest WITHOUT executing anything.

    Shared by the CLI `diff`, the Python API, and anything that needs to predict which stages
    would (re)compute. Returns id -> {type, params, inputs, key} in topological order."""
    dag = Dag.build(manifest.stages)
    keys: dict[str, str] = {}
    info: dict[str, dict[str, Any]] = {}
    for sid in dag.order:
        spec = dag.specs[sid]
        stage = get_stage(spec.type)
        params = {**stage.default_params(), **spec.params}
        input_keys = [keys[d] for d in spec.inputs]
        key = compute_key(spec.type, stage.version, params, input_keys)
        keys[sid] = key
        info[sid] = {"type": spec.type, "params": params, "inputs": list(spec.inputs), "key": key}
    return info


class Runner:
    def __init__(self, manifest: Manifest, *, force: list[str] | None = None):
        self.manifest = manifest
        self.cache = Cache(manifest.cache_dir)
        self.device = _detect_device()
        # stage ids to force-recompute even on cache hit (or ["*"] for all)
        self.force = set(force or [])

    def _forced(self, stage_id: str) -> bool:
        return "*" in self.force or stage_id in self.force

    def run(self) -> RunOutcome:
        dag = Dag.build(self.manifest.stages)
        started = datetime.now(timezone.utc)
        run_id = started.strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.manifest.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        keys: dict[str, str] = {}
        results: dict[str, StageResult] = {}
        result_dirs: dict[str, Path] = {}
        stage_runs: list[StageRun] = []
        failed_upstream: set[str] = set()

        for sid in dag.order:
            spec = dag.specs[sid]
            stage = get_stage(spec.type)

            # resolve params: defaults <- user params
            params = {**stage.default_params(), **spec.params}

            input_keys = [keys[dep] for dep in spec.inputs]
            key = compute_key(spec.type, stage.version, params, input_keys)
            keys[sid] = key

            # if any upstream failed/was skipped, skip this one
            if any(dep in failed_upstream for dep in spec.inputs):
                failed_upstream.add(sid)
                stage_runs.append(StageRun(sid, spec.type, key, StageStatus.SKIPPED, params=params))
                logger.warning("skip %s (upstream failed)", sid)
                continue

            entry = self.cache.entry(key)
            if entry.done and not self._forced(sid):
                result = entry.load_result()
                results[sid] = result
                result_dirs[sid] = entry.dir
                issues = self._safe_validate(stage, result, spec, params, entry.dir, results, result_dirs)
                stage_runs.append(
                    StageRun(sid, spec.type, key, StageStatus.CACHED, 0.0, result.metrics, issues,
                             params=params)
                )
                logger.info("cached %s  [%s]", sid, key[:12])
                continue

            # execute
            cache_dir = self.cache.open_for_write(key)
            ctx = self._make_context(sid, spec, params, cache_dir, results, result_dirs)
            t0 = time.perf_counter()
            try:
                logger.info("run    %s (%s)", sid, spec.type)
                result = stage.run(ctx)
                dt = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001 — surface any stage failure into the report
                dt = time.perf_counter() - t0
                failed_upstream.add(sid)
                stage_runs.append(
                    StageRun(sid, spec.type, key, StageStatus.FAILED, dt, params=params,
                             error=repr(exc))
                )
                logger.error("FAILED %s: %r", sid, exc)
                continue

            issues = self._safe_validate(stage, result, spec, params, cache_dir, results, result_dirs)
            result.issues = list(result.issues) + issues
            self.cache.finalize(
                key,
                result,
                keyinfo={
                    "id": sid,
                    "type": spec.type,
                    "version": stage.version,
                    "deterministic": stage.deterministic,
                    "params": params,
                    "input_keys": input_keys,
                },
            )
            results[sid] = result
            result_dirs[sid] = cache_dir
            stage_runs.append(
                StageRun(sid, spec.type, key, StageStatus.EXECUTED, dt, result.metrics, issues,
                         params=params)
            )

        finished = datetime.now(timezone.utc)
        ok = not any(s.status in (StageStatus.FAILED, StageStatus.SKIPPED) for s in stage_runs)
        outcome = RunOutcome(
            project=self.manifest.name,
            started=started.isoformat(),
            finished=finished.isoformat(),
            ok=ok,
            stages=stage_runs,
            run_dir=run_dir,
        )
        self._write_run(outcome, run_dir)
        return outcome

    def _make_context(
        self,
        sid: str,
        spec,
        params: dict[str, Any],
        cache_dir: Path,
        results: dict[str, StageResult],
        result_dirs: dict[str, Path],
    ) -> RunContext:
        inputs = {dep: results[dep] for dep in spec.inputs}
        input_dirs = {dep: result_dirs[dep] for dep in spec.inputs}
        stage_logger = logger.getChild(sid)
        return RunContext(
            stage_id=sid,
            stage_type=spec.type,
            params=params,
            cache_dir=cache_dir,
            inputs=inputs,
            input_dirs=input_dirs,
            project_dir=self.manifest.project_dir,
            device=self.device,
            logger=stage_logger,
            progress=lambda frac, msg, _l=stage_logger: _l.debug("progress %.0f%% %s", frac * 100, msg),
        )

    def _safe_validate(self, stage, result, spec, params, cache_dir, results, result_dirs) -> list[Issue]:
        try:
            ctx = self._make_context(spec.id, spec, params, cache_dir, results, result_dirs)
            return list(stage.validate(result, ctx))
        except Exception as exc:  # noqa: BLE001
            return [Issue(Severity.WARNING, f"validate() raised: {exc!r}")]

    def _write_run(self, outcome: RunOutcome, run_dir: Path) -> None:
        import json

        (run_dir / "run.json").write_text(
            json.dumps(outcome.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        # latest pointer
        (self.manifest.runs_dir / "latest.json").write_text(
            json.dumps(outcome.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        from openreco.engine.report import write_report

        write_report(outcome, run_dir / "report.html")
