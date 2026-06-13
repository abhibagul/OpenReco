"""Batch processing — run many OpenReco projects in one invocation.

Discovers project.toml manifests under a directory and runs each (sequentially, or across
processes with --jobs). Because every project is content-addressed and reproducible, a batch is
just a fan-out over the engine; results aggregate into one summary. Distributed/network execution
(remote workers) is future work — this is the local multi-project / multi-core path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def discover_projects(root: str | Path) -> list[Path]:
    """Project directories under `root` (or `[root]` itself if it holds a project.toml)."""
    root = Path(root)
    if (root / "project.toml").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("project.toml"))


def run_one(path: str) -> dict[str, Any]:
    """Run a single project (module-level so it's picklable for process pools)."""
    from openreco import stages  # noqa: F401 — register stages in worker processes
    from openreco.engine.manifest import load_manifest
    from openreco.engine.runner import Runner, StageStatus

    try:
        outcome = Runner(load_manifest(path)).run()
        failed = [s.id for s in outcome.stages
                  if s.status in (StageStatus.FAILED, StageStatus.SKIPPED)]
        return {"project": outcome.project, "path": str(path), "ok": outcome.ok,
                "stages": len(outcome.stages),
                "seconds": round(sum(s.seconds for s in outcome.stages), 2),
                "failed": failed}
    except Exception as exc:  # noqa: BLE001
        return {"project": Path(path).name, "path": str(path), "ok": False, "error": repr(exc)}


def run_batch(paths: list[str | Path], jobs: int = 1) -> list[dict[str, Any]]:
    """Run each project. jobs>1 runs them across processes (beware GPU oversubscription)."""
    paths = [str(p) for p in paths]
    if jobs <= 1 or len(paths) <= 1:
        return [run_one(p) for p in paths]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(run_one, paths))
