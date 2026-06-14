"""Familiar workflow layer: industry-standard operations -> OpenReco stages/params."""

from __future__ import annotations

import openreco
from openreco import Project
from openreco.workflow import operations, to_stage


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
