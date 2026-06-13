"""OpenReco — clean-room photogrammetry & 3D reconstruction platform.

Phase 0: the pipeline engine. A typed DAG of stages with content-addressed caching,
checkpoint/resume, deterministic re-runs, and a project-as-code manifest.
"""

__version__ = "0.0.1"


def __getattr__(name: str):
    # Lazily expose the Python API (openreco.Project / openreco.registered_stages) without
    # importing heavy stage deps at package import time.
    if name in ("Project", "registered_stages"):
        from openreco import api

        return getattr(api, name)
    if name == "measure_volume":
        from openreco.measure import measure_volume

        return measure_volume
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
