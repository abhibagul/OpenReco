"""The Stage protocol and registry — the one interface that matters.

A Stage is a unit of computation. Implementations register a `type` name; the manifest
references stages by that type. The engine (not the stage) computes cache keys, so caching
behavior is uniform and stages stay simple.

Design rule: stages are idempotent and write only into ctx.cache_dir. Given identical
params + identical upstream outputs + identical impl version, a stage must produce
equivalent output. Nondeterministic stages declare it via `deterministic = False`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from openreco.engine.context import Issue, RunContext, StageResult


class Stage(ABC):
    """Base class for all pipeline stages.

    Subclasses set the class attributes `type` and `version`, and implement `run`.
    `version` participates in the cache key: bump it when the implementation changes in a
    way that should invalidate cached outputs.
    """

    type: str = ""
    version: str = "1"
    deterministic: bool = True

    def params_schema(self) -> dict[str, Any]:
        """JSON-schema-ish description of tunables. Drives validation now and presets/GUI later.
        Default: accept anything."""
        return {}

    def default_params(self) -> dict[str, Any]:
        """Default parameter values, merged under user-supplied params."""
        return {}

    @abstractmethod
    def run(self, ctx: RunContext) -> StageResult:
        """Execute the stage. Must write all outputs into ctx.cache_dir and return a
        StageResult referencing them by relative path."""

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        """QA hook run after a successful execution (and after cache hits). Returns issues
        for the report. Default: no issues."""
        return []


# ---- registry ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Stage]] = {}


def register_stage(cls: type[Stage]) -> type[Stage]:
    """Class decorator: register a Stage implementation by its `type`."""
    if not cls.type:
        raise ValueError(f"{cls.__name__} must set a non-empty `type`")
    if cls.type in _REGISTRY:
        raise ValueError(f"stage type {cls.type!r} already registered by {_REGISTRY[cls.type].__name__}")
    _REGISTRY[cls.type] = cls
    return cls


def get_stage(type_name: str) -> Stage:
    """Instantiate a registered stage by type name."""
    try:
        cls = _REGISTRY[type_name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown stage type {type_name!r}; registered: {known}") from None
    return cls()


def registered_types() -> list[str]:
    return sorted(_REGISTRY)
