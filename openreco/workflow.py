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
        "op": "Add Photos", "stage": "ingest",
        "desc": "Add a folder of images to this chunk (ingest + QC). The first step of any project.",
        "fields": [
            {"label": "Image folder", "param": "image_dir", "type": "path", "default": "images"},
            {"label": "Blur culling", "param": "blur_relative", "type": "enum", "default": "Mild",
             "options": {"Off": 0.0, "Mild": 0.15, "Moderate": 0.3, "Aggressive": 0.5}},
        ],
    },
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
        "op": "Georeference", "stage": "georef",
        "desc": "Place the model in a CRS from EXIF GPS or GCPs (use the marker tool, then 'Use these GCPs').",
        "fields": [
            {"label": "Source", "param": "method", "type": "enum", "default": "Auto",
             "options": {"Auto": "auto", "GCPs": "gcp", "GPS (EXIF)": "gps", "Local": "local"}},
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
    {
        "op": "Merge Chunks", "stage": "merge_chunks",
        "desc": "Align separate chunks (ICP) and merge their point clouds into one.",
        "fields": [{"label": "Initial alignment", "param": "init", "type": "enum", "default": "Centroid",
                    "options": {"Centroid": "centroid", "None": "none"}}],
    },
]

# Which artifact each stage type produces, and which each operation consumes — used by the UI to
# auto-wire a new layer to existing layers that provide what it needs (so a click-built pipeline runs).
STAGE_PROVIDES: dict[str, list[str]] = {
    "ingest": ["images"],
    "sfm": ["model", "sparse_ply", "poses"],
    "refine": ["model"],
    "georef": ["model", "georef", "ply"],
    "mvs": ["points", "meta", "las", "model"],
    "merge_chunks": ["points", "meta", "merged"],
    "fuse": ["points", "meta"],
    "classify": ["points", "meta"],
    "mesh": ["mesh"],
    "texture": ["mesh", "glb"],
    "dsm": ["dsm", "meta"],
    "ortho": ["ortho", "meta"],
    "dtm": ["dtm"],
    "contours": ["contours"],
    "tiles": ["tiles"],
    "indices": ["meta"],
}
# artifacts each operation needs wired as input(s)
OP_NEEDS: dict[str, list[str]] = {
    "Add Photos": [],
    "Align Photos": ["images"],
    "Georeference": ["model", "images"],
    "Build Dense Cloud": ["model", "images"],
    "Build Model": ["points"],
    "Build Texture": ["mesh", "model", "images"],
    "Build DEM": ["points"],
    "Build Orthomosaic": ["points"],
    "Classify Points": ["points"],
    "Build Tiled Model": ["mesh"],
    "Build Contours": ["dsm"],
    "Merge Chunks": ["points"],
}
for _o in OPERATIONS:
    _o["needs"] = OP_NEEDS.get(_o["op"], [])

_BY_OP = {o["op"]: o for o in OPERATIONS}

# Quality/speed presets: per-stage-type parameter overrides applied in one click.
PRESETS: dict[str, dict[str, Any]] = {
    "Low": {
        "_speed": "fastest — quick preview",
        "ingest": {"blur_relative": 0.15},
        "sfm": {"max_image_size": 1200, "max_num_features": 4096, "matcher": "sequential"},
        "mvs": {"quality": "low", "geometric_consistency": False, "planesweep_max_dim": 600},
        "mesh": {"poisson_depth": 9},
        "texture": {"target_faces": 80000, "atlas_resolution": 1024, "blend_images": 1, "image_max_dim": 1500},
        "dsm": {"resolution_m": 1.0}, "ortho": {"resolution_m": 0.5},
    },
    "Medium": {
        "_speed": "balanced (default)",
        "sfm": {"max_image_size": 2000, "max_num_features": 8192, "matcher": "exhaustive"},
        "mvs": {"quality": "medium", "geometric_consistency": True, "planesweep_max_dim": 700},
        "mesh": {"poisson_depth": 10},
        "texture": {"target_faces": 150000, "atlas_resolution": 2048, "blend_images": 3, "image_max_dim": 2000},
        "dsm": {"resolution_m": 0.5}, "ortho": {"resolution_m": 0.25},
    },
    "High": {
        "_speed": "slow — detailed",
        "sfm": {"max_image_size": 3200, "max_num_features": 12000, "matcher": "exhaustive"},
        "mvs": {"quality": "high", "geometric_consistency": True, "planesweep_max_dim": 900},
        "mesh": {"poisson_depth": 11},
        "texture": {"target_faces": 300000, "atlas_resolution": 4096, "blend_images": 4, "image_max_dim": 3000},
        "dsm": {"resolution_m": 0.25}, "ortho": {"resolution_m": 0.1},
    },
    "Ultra": {
        "_speed": "slowest — maximum detail",
        "sfm": {"max_image_size": 4000, "max_num_features": 16000, "matcher": "exhaustive"},
        "mvs": {"quality": "high", "geometric_consistency": True, "planesweep_max_dim": 1100},
        "mesh": {"poisson_depth": 12},
        "texture": {"target_faces": 500000, "atlas_resolution": 8192, "blend_images": 5, "image_max_dim": 4000},
        "dsm": {"resolution_m": 0.1}, "ortho": {"resolution_m": 0.05},
    },
}


def presets() -> list[dict[str, str]]:
    """Preset names + a short speed/quality hint (for the UI selector)."""
    return [{"name": n, "speed": p["_speed"]} for n, p in PRESETS.items()]


def preset_params(name: str) -> dict[str, dict[str, Any]]:
    """The per-stage-type parameter overrides for a preset (no metadata keys)."""
    p = PRESETS.get(name)
    if p is None:
        raise KeyError(f"unknown preset {name!r}; choices: {list(PRESETS)}")
    return {k: dict(v) for k, v in p.items() if not k.startswith("_")}


def provides(stage_type: str) -> list[str]:
    """Artifacts a stage type produces (for input auto-wiring in the UI)."""
    return STAGE_PROVIDES.get(stage_type, [])


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
