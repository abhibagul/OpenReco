"""Runtime types passed to stages: RunContext, StageResult, Issue.

A stage receives a RunContext (resolved params, upstream results, its own cache dir,
device info, logging/progress hooks) and returns a StageResult (named artifacts +
metrics). Stages never reference each other directly — only via ctx.inputs, wired
by the engine from the DAG.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Issue:
    """A QA finding surfaced by a stage's validate() — feeds the report and, later, the
    'alignment doctor'."""

    severity: Severity
    message: str
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"severity": self.severity.value, "message": self.message, "hint": self.hint}


@dataclass
class StageResult:
    """What a stage produces. `artifacts` maps a logical name to a path inside the stage's
    cache dir; `metrics` is JSON-serializable summary data for the report."""

    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "issues": [i.to_dict() for i in self.issues],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageResult:
        return cls(
            artifacts=dict(d.get("artifacts", {})),
            metrics=dict(d.get("metrics", {})),
            issues=[
                Issue(Severity(i["severity"]), i["message"], i.get("hint"))
                for i in d.get("issues", [])
            ],
        )


@dataclass
class DeviceInfo:
    """Compute capabilities. Phase 0 reports CPU only; the wgpu/CUDA probe lands in
    openreco/compute later. Stages branch on this rather than detecting hardware themselves."""

    has_cuda: bool = False
    has_metal: bool = False
    cpu_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"has_cuda": self.has_cuda, "has_metal": self.has_metal, "cpu_count": self.cpu_count}


@dataclass
class RunContext:
    """Everything a stage needs to run, supplied by the engine.

    - stage_id / stage_type: identity within this DAG.
    - params: resolved, validated parameters for this stage.
    - cache_dir: the stage writes ALL its outputs here. It is keyed by content address,
      so writing here is what makes the stage cacheable/resumable.
    - inputs: results of upstream stages, keyed by upstream stage id.
    - project_dir: root of the project (where project.toml lives); resolve user paths against it.
    - device: compute capabilities.
    - logger / progress / is_cancelled: observability and cooperative cancellation.
    """

    stage_id: str
    stage_type: str
    params: dict[str, Any]
    cache_dir: Path
    inputs: dict[str, StageResult]
    input_dirs: dict[str, Path]
    project_dir: Path
    device: DeviceInfo
    logger: logging.Logger
    progress: Callable[[float, str], None] = lambda frac, msg: None
    is_cancelled: Callable[[], bool] = lambda: False

    def artifact_path(self, name: str) -> Path:
        """Absolute path for a new artifact this stage will write."""
        return self.cache_dir / name

    def input_artifact(self, stage_id: str, name: str) -> Path:
        """Absolute path to a named artifact produced by an upstream stage."""
        result = self.inputs[stage_id]
        rel = result.artifacts[name]
        return self.input_dirs[stage_id] / rel

    def input_with(self, artifact: str) -> str:
        """Id of the (first) upstream input that produces `artifact`. Lets a stage consume e.g.
        a "model" without hardcoding whether it came from `sfm` or a `refine` stage in between —
        keeping the pipeline composable."""
        for dep, result in self.inputs.items():
            if artifact in result.artifacts:
                return dep
        raise KeyError(f"no input provides artifact {artifact!r} (have: "
                       f"{[d for d in self.inputs]})")

    def find_input(self, artifact: str) -> str | None:
        """Like input_with, but returns None instead of raising when no input provides `artifact`
        (for optional dependencies, e.g. mvs georeferencing)."""
        for dep, result in self.inputs.items():
            if artifact in result.artifacts:
                return dep
        return None

    def write_json(self, name: str, data: Any) -> str:
        path = self.artifact_path(name)
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return name

    def read_input_json(self, stage_id: str, name: str) -> Any:
        return json.loads(self.input_artifact(stage_id, name).read_text(encoding="utf-8"))
