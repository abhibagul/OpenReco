"""Directed acyclic graph: build from stage specs, detect cycles, topologically order.

Edges come from each stage's `inputs` (the ids of stages it depends on). The DAG is pure
structure — no execution, no caching. The Runner walks the topo order.
"""

from __future__ import annotations

from dataclasses import dataclass

from openreco.engine.manifest import StageSpec


class DagError(Exception):
    pass


@dataclass
class Dag:
    specs: dict[str, StageSpec]          # id -> spec
    order: list[str]                     # topologically sorted ids
    dependents: dict[str, list[str]]     # id -> ids that depend on it

    @classmethod
    def build(cls, specs: list[StageSpec]) -> "Dag":
        by_id: dict[str, StageSpec] = {}
        for spec in specs:
            if spec.id in by_id:
                raise DagError(f"duplicate stage id: {spec.id!r}")
            by_id[spec.id] = spec

        # validate edges
        for spec in specs:
            for dep in spec.inputs:
                if dep not in by_id:
                    raise DagError(f"stage {spec.id!r} depends on unknown stage {dep!r}")
                if dep == spec.id:
                    raise DagError(f"stage {spec.id!r} depends on itself")

        order = _topo_sort(by_id)
        dependents: dict[str, list[str]] = {sid: [] for sid in by_id}
        for spec in specs:
            for dep in spec.inputs:
                dependents[dep].append(spec.id)
        return cls(specs=by_id, order=order, dependents=dependents)


def _topo_sort(by_id: dict[str, StageSpec]) -> list[str]:
    """Kahn's algorithm. Raises DagError on a cycle, naming the involved stages."""
    indegree = {sid: 0 for sid in by_id}
    for spec in by_id.values():
        for _dep in spec.inputs:
            indegree[spec.id] += 1

    # deterministic order: process ready nodes sorted by id
    ready = sorted(sid for sid, d in indegree.items() if d == 0)
    order: list[str] = []
    while ready:
        sid = ready.pop(0)
        order.append(sid)
        # decrement dependents
        newly_ready = []
        for spec in by_id.values():
            if sid in spec.inputs:
                indegree[spec.id] -= 1
                if indegree[spec.id] == 0:
                    newly_ready.append(spec.id)
        for n in sorted(newly_ready):
            # insert keeping the ready list sorted for determinism
            ready.append(n)
        ready.sort()

    if len(order) != len(by_id):
        stuck = sorted(set(by_id) - set(order))
        raise DagError(f"cycle detected among stages: {', '.join(stuck)}")
    return order
