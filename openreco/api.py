"""Python API — mirrors the CLI 1:1 and adds programmatic project construction.

The API is intentionally a thin, documented wrapper over the same engine the CLI uses, so a
script and a command line produce identical, reproducible runs. Two ways in:

    import openreco

    # 1) open an existing project-as-code manifest
    proj = openreco.Project.open("samples/sceaux")
    outcome = proj.run()                      # cache-aware; re-run is a no-op
    print(outcome.ok, outcome.stage("sfm").metrics, outcome.report)

    # 2) build a pipeline programmatically (no TOML needed)
    proj = (openreco.Project.create("/tmp/job", name="demo", crs="EPSG:32633")
            .add_stage("ingest", "ingest", params={"image_dir": "images"})
            .add_stage("sfm", "sfm", inputs=["ingest"]))
    proj.save()                               # optional: write project.toml
    proj.run(force=["sfm"])

`Project.diff(other)` predicts which stages would recompute, without running anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openreco import stages as _stages  # noqa: F401 — importing registers built-in stages
from openreco.engine.manifest import Manifest, StageSpec, load_manifest
from openreco.engine.runner import RunOutcome, Runner, compute_keys
from openreco.engine.stage import registered_types

__all__ = ["Project", "registered_stages"]


def registered_stages() -> list[str]:
    """Names of all registered stage types."""
    return registered_types()


class Project:
    """A photogrammetry project: an ordered set of typed stages plus its on-disk cache."""

    def __init__(self, manifest: Manifest):
        self.manifest = manifest

    # ---- construction ------------------------------------------------------------------
    @classmethod
    def open(cls, path: str | Path) -> "Project":
        """Load an existing project.toml (or a directory containing one)."""
        return cls(load_manifest(path))

    @classmethod
    def create(cls, directory: str | Path, name: str | None = None,
               crs: str | None = None) -> "Project":
        """Create a new, empty in-memory project rooted at `directory`."""
        d = Path(directory).resolve()
        d.mkdir(parents=True, exist_ok=True)
        return cls(Manifest(name=name or d.name, crs=crs, stages=[], project_dir=d, raw={}))

    # ---- editing -----------------------------------------------------------------------
    def add_stage(self, id: str, type: str, inputs: list[str] | None = None,
                  params: dict[str, Any] | None = None) -> "Project":
        """Append a stage. Returns self for chaining."""
        if any(s.id == id for s in self.manifest.stages):
            raise ValueError(f"duplicate stage id: {id!r}")
        self.manifest.stages.append(
            StageSpec(id=id, type=type, params=dict(params or {}), inputs=list(inputs or []))
        )
        return self

    @property
    def stages(self) -> list[StageSpec]:
        return self.manifest.stages

    @property
    def project_dir(self) -> Path:
        return self.manifest.project_dir

    # ---- execution ---------------------------------------------------------------------
    def run(self, force: list[str] | None = None, force_all: bool = False) -> RunOutcome:
        """Execute the pipeline (cache-aware). `force` recomputes named stages; `force_all`
        recomputes everything."""
        return Runner(self.manifest, force=(["*"] if force_all else force)).run()

    def resume(self) -> RunOutcome:
        """Alias of run() — the cache provides checkpoint/resume semantics."""
        return self.run()

    # ---- planning / introspection ------------------------------------------------------
    def plan(self) -> dict[str, dict[str, Any]]:
        """Per-stage content-address keys + resolved params, without executing."""
        return compute_keys(self.manifest)

    def diff(self, other: "Project") -> dict[str, dict[str, Any]]:
        """Compare two projects by content-address. Returns id -> {change, a, b} where change
        is 'same' | 'added' | 'removed' | 'modified' (would recompute)."""
        a, b = self.plan(), other.plan()
        out: dict[str, dict[str, Any]] = {}
        for sid in sorted(set(a) | set(b)):
            ka = a.get(sid, {}).get("key")
            kb = b.get(sid, {}).get("key")
            if ka == kb:
                change = "same"
            elif ka is None:
                change = "added"
            elif kb is None:
                change = "removed"
            else:
                change = "modified"
            out[sid] = {"change": change, "a": a.get(sid), "b": b.get(sid)}
        return out

    # ---- persistence -------------------------------------------------------------------
    def save(self, path: str | Path | None = None) -> Path:
        """Write the project to a TOML manifest (default: <project_dir>/project.toml)."""
        target = Path(path) if path else self.manifest.project_dir / "project.toml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._to_toml(), encoding="utf-8")
        return target

    def _to_toml(self) -> str:
        lines = ["[project]", f'name = "{self.manifest.name}"']
        if self.manifest.crs:
            lines.append(f'crs = "{self.manifest.crs}"')
        for s in self.manifest.stages:
            lines += ["", "[[stage]]", f'id = "{s.id}"', f'type = "{s.type}"']
            if s.inputs:
                inputs = ", ".join(f'"{i}"' for i in s.inputs)
                lines.append(f"inputs = [{inputs}]")
            if s.params:
                lines.append(f"params = {_toml_inline(s.params)}")
        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:
        return f"Project(name={self.manifest.name!r}, stages={[s.id for s in self.manifest.stages]})"


def _toml_inline(d: dict[str, Any]) -> str:
    parts = []
    for k, v in d.items():
        if isinstance(v, bool):
            parts.append(f"{k} = {str(v).lower()}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k} = {v}")
        else:
            parts.append(f'{k} = "{v}"')
    return "{ " + ", ".join(parts) + " }"
