"""OpenReco — clean-room photogrammetry & 3D reconstruction platform.

Phase 0: the pipeline engine. A typed DAG of stages with content-addressed caching,
checkpoint/resume, deterministic re-runs, and a project-as-code manifest.
"""

__version__ = "0.0.1"


def __getattr__(name: str):
    # Lazily expose the Python API (openreco.Project / openreco.registered_stages) without
    # importing heavy stage deps at package import time.
    if name in ("Project", "registered_stages", "stage_info"):
        from openreco import api

        return getattr(api, name)
    if name in ("measure_volume", "measure_profile", "measure_volume_region"):
        from openreco import measure

        return getattr(measure, name)
    if name in ("export_product", "list_formats"):
        from openreco import exporters

        return getattr(exporters, name)
    if name in ("crs_info", "search_crs"):
        from openreco.geo import crs

        return getattr(crs, name)
    if name in ("workflow_operations", "to_stage"):
        from openreco import workflow

        return workflow.operations if name == "workflow_operations" else workflow.to_stage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
