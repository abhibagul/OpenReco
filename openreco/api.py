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

from dataclasses import replace
from pathlib import Path
from typing import Any

from openreco import stages as _stages  # noqa: F401 — importing registers built-in stages
from openreco.engine.manifest import Manifest, StageSpec, load_manifest
from openreco.engine.runner import RunOutcome, Runner, compute_keys
from openreco.engine.stage import get_stage, registered_types

__all__ = ["Project", "registered_stages", "stage_info"]


def registered_stages() -> list[str]:
    """Names of all registered stage types."""
    return registered_types()


def stage_info(type_name: str | None = None) -> list[dict[str, Any]] | dict[str, Any]:
    """Introspect registered stages — for building a UI stage palette and parameter panels.
    Returns, per stage: type, version, deterministic, default_params, params_schema. Pass a
    `type_name` for a single stage's info."""
    def info(t: str) -> dict[str, Any]:
        s = get_stage(t)
        return {"type": t, "version": s.version, "deterministic": s.deterministic,
                "default_params": s.default_params(), "params_schema": s.params_schema()}

    if type_name is not None:
        return info(type_name)
    return [info(t) for t in registered_types()]


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
                  params: dict[str, Any] | None = None, chunk: str = "Chunk 1") -> "Project":
        """Append a stage (layer) to a workspace chunk. Returns self for chaining."""
        if any(s.id == id for s in self.manifest.stages):
            raise ValueError(f"duplicate stage id: {id!r}")
        self.manifest.stages.append(
            StageSpec(id=id, type=type, params=dict(params or {}), inputs=list(inputs or []),
                      chunk=chunk))
        return self

    def add_chunk(self, name: str) -> "Project":
        """Register a (possibly empty) workspace chunk."""
        if name not in self.manifest.chunks:
            self.manifest.chunks.append(name)
        return self

    def remove_stage(self, id: str) -> "Project":
        """Delete a layer and drop it from any other layer's inputs."""
        kept = [s for s in self.manifest.stages if s.id != id]
        self.manifest.stages = [replace(s, inputs=[i for i in s.inputs if i != id])
                                if id in s.inputs else s for s in kept]
        return self

    def rename_stage(self, id: str, new_id: str) -> "Project":
        """Rename a layer, updating downstream input references."""
        if id == new_id or not new_id:
            return self
        if any(s.id == new_id for s in self.manifest.stages):
            raise ValueError(f"duplicate stage id: {new_id!r}")
        out = []
        for s in self.manifest.stages:
            inputs = [new_id if i == id else i for i in s.inputs]
            out.append(replace(s, id=new_id if s.id == id else s.id, inputs=inputs))
        self.manifest.stages = out
        return self

    def rename_chunk(self, name: str, new_name: str) -> "Project":
        """Rename a chunk and re-point its layers."""
        if name == new_name or not new_name:
            return self
        self.manifest.chunks = [new_name if c == name else c for c in self.manifest.chunks]
        if new_name not in self.manifest.chunks:
            self.manifest.chunks.append(new_name)
        self.manifest.stages = [replace(s, chunk=new_name) if s.chunk == name else s
                                for s in self.manifest.stages]
        return self

    def remove_chunk(self, name: str) -> "Project":
        """Delete a chunk and all of its layers."""
        self.manifest.stages = [s for s in self.manifest.stages if s.chunk != name]
        self.manifest.chunks = [c for c in self.manifest.chunks if c != name]
        return self

    def move_stage(self, id: str, chunk: str) -> "Project":
        """Move a layer to another chunk."""
        self.manifest.stages = [replace(s, chunk=chunk) if s.id == id else s
                                for s in self.manifest.stages]
        return self

    def set_stage_enabled(self, id: str, enabled: bool) -> "Project":
        """Enable/disable a layer (disabled layers + their dependents are skipped on Run)."""
        self.manifest.stages = [replace(s, enabled=enabled) if s.id == id else s
                                for s in self.manifest.stages]
        return self

    def set_chunk_enabled(self, name: str, enabled: bool) -> "Project":
        """Enable/disable every layer in a chunk."""
        self.manifest.stages = [replace(s, enabled=enabled) if s.chunk == name else s
                                for s in self.manifest.stages]
        return self

    def _runnable_stages(self) -> list[StageSpec]:
        """Enabled layers whose entire input closure is also enabled (disabled branches skipped)."""
        excluded = {s.id for s in self.manifest.stages if not s.enabled}
        changed = True
        while changed:
            changed = False
            for s in self.manifest.stages:
                if s.id not in excluded and any(i in excluded for i in s.inputs):
                    excluded.add(s.id)
                    changed = True
        return [s for s in self.manifest.stages if s.id not in excluded]

    def add_operation(self, op: str, id: str, inputs: list[str] | None = None,
                      values: dict[str, Any] | None = None, chunk: str = "Chunk 1") -> "Project":
        """Add a stage from a familiar workflow operation (e.g. 'Build Dense Cloud') + field values,
        translated to the underlying stage/params via openreco.workflow."""
        from openreco.workflow import to_stage
        spec = to_stage(op, values)
        return self.add_stage(id, spec["stage_type"], inputs=inputs, params=spec["params"], chunk=chunk)

    @property
    def stages(self) -> list[StageSpec]:
        return self.manifest.stages

    @property
    def project_dir(self) -> Path:
        return self.manifest.project_dir

    # ---- execution ---------------------------------------------------------------------
    def run(self, force: list[str] | None = None, force_all: bool = False,
            on_event: Any = None, cancel: Any = None) -> RunOutcome:
        """Execute the pipeline (cache-aware). `force` recomputes named stages; `force_all`
        recomputes everything. `on_event(dict)` receives live run events (stage_start/progress/
        stage_done/run_done) and `cancel()->bool` enables cooperative cancellation — both for UIs.
        Disabled layers (and any layer depending on one) are excluded from this run."""
        manifest = self.manifest
        runnable = self._runnable_stages()
        if len(runnable) != len(manifest.stages):
            manifest = replace(manifest, stages=runnable)
        return Runner(manifest, force=(["*"] if force_all else force),
                      on_event=on_event, cancel=cancel).run()

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
        lines = ["[project]", f"name = {_toml_str(self.manifest.name)}"]
        if self.manifest.crs:
            lines.append(f"crs = {_toml_str(self.manifest.crs)}")
        chunks = self.manifest.chunk_names()
        if chunks != ["Chunk 1"]:
            lines.append("chunks = [" + ", ".join(_toml_str(c) for c in chunks) + "]")
        for s in self.manifest.stages:
            lines += ["", "[[stage]]", f"id = {_toml_str(s.id)}", f"type = {_toml_str(s.type)}"]
            if s.chunk and s.chunk != "Chunk 1":
                lines.append(f"chunk = {_toml_str(s.chunk)}")
            if not s.enabled:
                lines.append("enabled = false")
            if s.inputs:
                inputs = ", ".join(_toml_str(i) for i in s.inputs)
                lines.append(f"inputs = [{inputs}]")
            if s.params:
                lines.append(f"params = {_toml_inline(s.params)}")
        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:
        return f"Project(name={self.manifest.name!r}, stages={[s.id for s in self.manifest.stages]})"


def _toml_str(s: Any) -> str:
    """A TOML basic string with backslashes and quotes escaped (Windows paths, etc.)."""
    esc = str(s).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{esc}"'


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + " }"
    return _toml_str(v)


def _toml_inline(d: dict[str, Any]) -> str:
    return "{ " + ", ".join(f"{k} = {_toml_value(v)}" for k, v in d.items()) + " }"
