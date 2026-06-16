"""Project manifest — the project IS this file plus its content-addressed cache.

A project.toml describes the project and an ordered list of stages. Example:

    [project]
    name = "demo"
    crs = "EPSG:32633"          # target coordinate system (Phase 1+)

    [[stage]]
    id = "generate"
    type = "dummy_generate"
    params = { n = 5 }

    [[stage]]
    id = "sum"
    type = "dummy_sum"
    inputs = ["generate"]

The engine reads (never writes) this file. tomllib is read-only stdlib (3.11+), which is
exactly the contract we want: humans/CLI author manifests, the engine consumes them.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    chunk: str = "Chunk 1"        # workspace grouping (familiar chunks); not part of the cache key
    enabled: bool = True          # disabled layers (+ their dependents) are excluded from a run; not in cache key


@dataclass
class Manifest:
    name: str
    crs: str | None
    stages: list[StageSpec]
    project_dir: Path
    raw: dict[str, Any]
    chunks: list[str] = field(default_factory=lambda: ["Chunk 1"])

    def chunk_names(self) -> list[str]:
        """All chunks: the registered list plus any referenced by a stage, order-preserving."""
        names = list(self.chunks)
        for s in self.stages:
            if s.chunk not in names:
                names.append(s.chunk)
        return names or ["Chunk 1"]

    @property
    def openreco_dir(self) -> Path:
        return self.project_dir / ".openreco"

    @property
    def cache_dir(self) -> Path:
        return self.openreco_dir / "cache"

    @property
    def runs_dir(self) -> Path:
        return self.openreco_dir / "runs"


def load_manifest(path: str | Path) -> Manifest:
    p = Path(path).resolve()
    if p.is_dir():
        p = p / "project.toml"
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {p}")

    with p.open("rb") as fh:
        raw = tomllib.load(fh)

    project = raw.get("project", {})
    name = project.get("name", p.parent.name)
    crs = project.get("crs")
    chunks = list(project.get("chunks", ["Chunk 1"]))

    stage_entries = raw.get("stage", [])
    if not isinstance(stage_entries, list):
        raise ValueError("[[stage]] must be an array of tables")

    seen: set[str] = set()
    stages: list[StageSpec] = []
    for i, entry in enumerate(stage_entries):
        sid = entry.get("id")
        stype = entry.get("type")
        if not sid:
            raise ValueError(f"stage #{i} missing 'id'")
        if not stype:
            raise ValueError(f"stage {sid!r} missing 'type'")
        if sid in seen:
            raise ValueError(f"duplicate stage id: {sid!r}")
        seen.add(sid)
        stages.append(
            StageSpec(
                id=sid,
                type=stype,
                params=dict(entry.get("params", {})),
                inputs=list(entry.get("inputs", [])),
                chunk=entry.get("chunk", "Chunk 1"),
                enabled=bool(entry.get("enabled", True)),
            )
        )

    return Manifest(name=name, crs=crs, stages=stages, project_dir=p.parent, raw=raw, chunks=chunks)
