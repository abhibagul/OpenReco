"""Familiar workflow layer — industry-standard operations + field names mapped to OpenReco stages.

Clean-room: this maps widely-used, *functional* photogrammetry option labels (Accuracy, Quality,
Face count, Texture size, …) onto OpenReco's own stages/parameters so the workflow feels familiar.
It contains no the reference tool code, assets, or trademarks — just descriptive field names and our own
parameter translation. The UI presents these operations; `to_stage()` converts a chosen operation
+ field values into a stage spec the engine runs.
"""

from __future__ import annotations

from typing import Any

# Each operation: a familiar name, the OpenReco stage it builds, and fields with familiar labels.
# An "enum" field maps a friendly choice -> the underlying parameter value.
OPERATIONS: list[dict[str, Any]] = [
    {
        "op": "Align Photos", "stage": "sfm",
        "desc": "Detect features, match, and solve camera poses (sparse cloud).",
        "fields": [
            {"label": "Accuracy", "param": "max_image_size", "type": "enum", "default": "Medium",
             "options": {"Highest": 4000, "High": 2400, "Medium": 1600, "Low": 1000, "Lowest": 700}},
            {"label": "Matching", "param": "matcher", "type": "enum", "default": "Exhaustive",
             "options": {"Exhaustive": "exhaustive", "Sequential": "sequential", "Spatial (GPS)": "spatial"}},
            {"label": "Key point limit", "param": "max_num_features", "type": "int", "default": 8192},
            {"label": "Camera model", "param": "camera_mode", "type": "enum", "default": "Auto",
             "options": {"Auto": "auto", "Single": "single", "Per camera": "per_image"}},
            {"label": "Method", "param": "mapper", "type": "enum", "default": "Incremental",
             "options": {"Incremental": "incremental", "Global": "global"}},
        ],
    },
    {
        "op": "Build Dense Cloud", "stage": "mvs",
        "desc": "Dense multi-view stereo depth maps fused into a dense point cloud.",
        "fields": [
            {"label": "Quality", "param": "quality", "type": "enum", "default": "Medium",
             "options": {"Ultra high": "high", "High": "high", "Medium": "medium", "Low": "low", "Lowest": "low"}},
            {"label": "Depth filtering", "param": "geometric_consistency", "type": "enum", "default": "Mild",
             "options": {"Aggressive": True, "Moderate": True, "Mild": True, "Disabled": False}},
            {"label": "Backend", "param": "dense_backend", "type": "enum", "default": "Auto",
             "options": {"Auto": "auto", "GPU (COLMAP CUDA)": "colmap_cuda",
                         "Portable (any GPU/CPU)": "planesweep"}},
        ],
    },
    {
        "op": "Build Model", "stage": "mesh",
        "desc": "Reconstruct a polygonal mesh surface from the dense cloud.",
        "fields": [
            {"label": "Surface type", "param": "method", "type": "enum", "default": "Arbitrary (3D)",
             "options": {"Arbitrary (3D)": "poisson", "Height field (2.5D)": "delaunay_2_5d"}},
            {"label": "Face count", "param": "poisson_depth", "type": "enum", "default": "Medium",
             "options": {"High": 12, "Medium": 11, "Low": 9}},
        ],
    },
    {
        "op": "Build Texture", "stage": "texture",
        "desc": "Bake a UV texture atlas onto the model from the source images.",
        "fields": [
            {"label": "Texture size", "param": "atlas_resolution", "type": "enum", "default": "2048",
             "options": {"8192": 8192, "4096": 4096, "2048": 2048, "1024": 1024}},
            {"label": "Blending images", "param": "blend_images", "type": "int", "default": 4},
            {"label": "Color balancing", "param": "equalize_exposure", "type": "bool", "default": True},
        ],
    },
    {
        "op": "Build DEM", "stage": "dsm",
        "desc": "Rasterize a digital elevation model (DSM).",
        "fields": [{"label": "Resolution (m/px)", "param": "resolution_m", "type": "float", "default": 1.0}],
    },
    {
        "op": "Build Orthomosaic", "stage": "ortho",
        "desc": "Orthorectified, georeferenced image mosaic.",
        "fields": [{"label": "Resolution (m/px)", "param": "resolution_m", "type": "float", "default": 0.5}],
    },
    {
        "op": "Classify Points", "stage": "classify",
        "desc": "Classify the dense cloud into ground / building / vegetation.",
        "fields": [
            {"label": "Cell size (m)", "param": "cell_m", "type": "float", "default": 5.0},
            {"label": "Max distance (m)", "param": "ground_threshold_m", "type": "float", "default": 0.5},
        ],
    },
    {
        "op": "Build Tiled Model", "stage": "tiles",
        "desc": "Split the model into streamable 3D Tiles.",
        "fields": [{"label": "Grid", "param": "grid", "type": "int", "default": 4}],
    },
    {
        "op": "Build Contours", "stage": "contours",
        "desc": "Generate contour lines from the DEM.",
        "fields": [{"label": "Interval (m)", "param": "interval_m", "type": "float", "default": 10.0}],
    },
]

_BY_OP = {o["op"]: o for o in OPERATIONS}


def operations() -> list[dict[str, Any]]:
    """The familiar workflow operations + their fields (for a UI workflow menu)."""
    return OPERATIONS


def to_stage(op: str, values: dict[str, Any] | None = None) -> dict[str, Any]:
    """Translate a workflow operation + chosen field values into {stage_type, params}."""
    if op not in _BY_OP:
        raise KeyError(f"unknown operation {op!r}; choices: {list(_BY_OP)}")
    spec = _BY_OP[op]
    values = values or {}
    params: dict[str, Any] = {}
    for f in spec["fields"]:
        label, param = f["label"], f["param"]
        v = values.get(label, f.get("default"))
        if f["type"] == "enum":
            opts = f["options"]
            params[param] = opts.get(v, v if v in opts.values() else f["options"][f["default"]])
        else:
            params[param] = v
    return {"stage_type": spec["stage"], "params": params}
