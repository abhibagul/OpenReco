"""SfM integration test on the real Sceaux Castle sample.

Skipped unless the sample images are present (run scripts/fetch_sample.py) and pycolmap is
installed — keeps CI fast/offline while validating the real reconstruction path locally.
This is slow (~1 min CPU); deselect with `-m "not slow"`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openreco import stages  # noqa: F401
from openreco.engine.manifest import load_manifest
from openreco.engine.runner import Runner, StageStatus

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sceaux"
HAVE_IMAGES = SAMPLE.joinpath("images").is_dir() and any(SAMPLE.joinpath("images").glob("*.JPG"))

pycolmap = pytest.importorskip("pycolmap")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not HAVE_IMAGES, reason="run scripts/fetch_sample.py to enable"),
]


def test_sceaux_reconstructs(tmp_path):
    outcome = Runner(load_manifest(SAMPLE)).run()
    assert outcome.ok
    sfm = next(s for s in outcome.stages if s.id == "sfm")
    # the sample is well-connected: expect all 11 images, healthy points, sub-pixel error
    assert sfm.metrics["reg_images"] >= 10
    assert sfm.metrics["points3D"] > 1000
    assert sfm.metrics["mean_reproj_error"] < 1.5
    assert sfm.metrics["num_models"] == 1


def test_rerun_is_cached():
    Runner(load_manifest(SAMPLE)).run()
    outcome = Runner(load_manifest(SAMPLE)).run()
    assert all(s.status == StageStatus.CACHED for s in outcome.stages)


def test_full_pipeline_publishes_shareable_bundle():
    outcome = Runner(load_manifest(SAMPLE)).run()
    assert outcome.ok
    out = SAMPLE / "output"
    for f in ("index.html", "points.ply", "mesh.ply", "dsm.tif", "ortho.tif", "summary.json"):
        assert (out / f).exists(), f"missing {f} in published bundle"
    # viewer fully templated (no placeholders left)
    import re
    assert not re.findall(r"__[A-Z]+__", (out / "index.html").read_text(encoding="utf-8"))
