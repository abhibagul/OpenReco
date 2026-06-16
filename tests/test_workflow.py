"""Familiar workflow layer: common operations -> OpenReco stages/params."""

from __future__ import annotations

import openreco
from openreco import Project
from openreco.workflow import operations, to_stage, validate_pipeline


def test_operations_listed():
    names = {o["op"] for o in operations()}
    assert {"Align Photos", "Build Dense Cloud", "Build Model", "Build Texture",
            "Build DEM", "Build Orthomosaic", "Classify Points"} <= names
    assert openreco.workflow_operations() == operations()


def test_align_photos_translation():
    s = to_stage("Align Photos", {"Accuracy": "High", "Matching": "Sequential", "Method": "Global"})
    assert s["stage_type"] == "sfm"
    assert s["params"]["max_image_size"] == 2400          # "High" -> 2400 px
    assert s["params"]["matcher"] == "sequential"
    assert s["params"]["mapper"] == "global"
    assert s["params"]["max_num_features"] == 8192        # default applied


def test_dense_cloud_backend_and_defaults():
    s = to_stage("Build Dense Cloud", {"Backend": "Portable (any GPU/CPU)"})
    assert s["stage_type"] == "mvs" and s["params"]["dense_backend"] == "planesweep"
    assert s["params"]["quality"] == "medium"             # default
    assert s["params"]["geometric_consistency"] is True   # "Mild" default -> True


def test_build_model_surface_type():
    assert to_stage("Build Model", {"Surface type": "Height field (2.5D)"})["params"]["method"] == "delaunay_2_5d"
    assert to_stage("Build Model", {"Surface type": "Arbitrary (3D)"})["params"]["method"] == "poisson"


def test_project_add_operation(tmp_path):
    p = (Project.create(tmp_path, name="wf")
         .add_operation("Align Photos", "align", inputs=["ingest"], values={"Accuracy": "Low"}))
    s = p.stages[0]
    assert s.id == "align" and s.type == "sfm" and s.params["max_image_size"] == 1000


def test_vegetation_indices_operation():
    ops = {o["op"] for o in operations()}
    assert "Vegetation Indices" in ops                  # surfaced in the Workflow menu
    s = to_stage("Vegetation Indices", {"Indices": "RGB + NDVI · GNDVI (needs NIR)", "NIR band #": 5})
    assert s["stage_type"] == "indices"
    assert s["params"]["indices"] == ["exg", "vari", "gli", "ndvi", "gndvi"]
    assert s["params"]["nir_band"] == 5
    assert to_stage("Vegetation Indices")["params"]["indices"] == ["exg", "vari", "gli"]


def test_validate_clean_pipeline_has_no_issues(tmp_path):
    p = (Project.create(tmp_path, name="ok", crs="EPSG:32613")
         .add_stage("ing", "ingest", params={"image_dir": "imgs"})
         .add_stage("sfm", "sfm", inputs=["ing"])
         .add_stage("geo", "georef", inputs=["sfm", "ing"])
         .add_stage("mvs", "mvs", inputs=["ing", "geo"])
         .add_stage("mesh", "mesh", inputs=["mvs"])
         .add_stage("tex", "texture", inputs=["mesh", "geo", "ing"]))
    assert validate_pipeline(p.manifest.stages) == []


def test_validate_catches_texture_missing_model(tmp_path):
    # texture wired without a 'model' provider (the bug the end-to-end run hit)
    p = (Project.create(tmp_path, name="bad")
         .add_stage("ing", "ingest", params={"image_dir": "imgs"})
         .add_stage("sfm", "sfm", inputs=["ing"])
         .add_stage("mvs", "mvs", inputs=["ing", "sfm"])
         .add_stage("mesh", "mesh", inputs=["mvs"])
         .add_stage("tex", "texture", inputs=["mesh", "ing"]))   # no model provider
    issues = validate_pipeline(p.manifest.stages)
    assert any(i["stage"] == "tex" and i["severity"] == "error" and "model" in i["message"]
               for i in issues)


def test_validate_warns_georef_local_fallback(tmp_path):
    # georef without ingest (no GPS) and no GCP -> warns about a local frame
    p = (Project.create(tmp_path, name="loc")
         .add_stage("ing", "ingest", params={"image_dir": "imgs"})
         .add_stage("sfm", "sfm", inputs=["ing"])
         .add_stage("geo", "georef", inputs=["sfm"]))            # no ingest input
    issues = validate_pipeline(p.manifest.stages)
    assert any(i["stage"] == "geo" and i["severity"] == "warning" and "local" in i["message"]
               for i in issues)


def test_validate_flags_unknown_input(tmp_path):
    p = (Project.create(tmp_path, name="u")
         .add_stage("sfm", "sfm", inputs=["nope"]))
    issues = validate_pipeline(p.manifest.stages)
    assert any(i["severity"] == "error" and "does not exist" in i["message"] for i in issues)
