"""OpenReco pipeline engine.

The engine knows nothing about photogrammetry. It schedules a DAG of typed stages,
caches their outputs by content address, and produces a reproducible run report.
"""

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.dag import Dag, DagError
from openreco.engine.manifest import Manifest, StageSpec, load_manifest
from openreco.engine.runner import RunOutcome, Runner, StageStatus
from openreco.engine.stage import Stage, get_stage, register_stage

__all__ = [
    "Dag",
    "DagError",
    "Issue",
    "Manifest",
    "RunContext",
    "RunOutcome",
    "Runner",
    "Severity",
    "Stage",
    "StageResult",
    "StageSpec",
    "StageStatus",
    "get_stage",
    "load_manifest",
    "register_stage",
]
