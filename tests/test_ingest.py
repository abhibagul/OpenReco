"""Tests for the ingest stage and image-IO helpers (no network, no real EXIF needed)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from openreco import stages  # noqa: F401
from openreco.engine.manifest import load_manifest
from openreco.engine.runner import Runner, StageStatus
from openreco.io.images import _dms_to_deg, blur_score


def _sharp(size=(128, 128), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)  # high-freq noise = sharp
    return Image.fromarray(a)


def _blurry(size=(128, 128), seed=0) -> Image.Image:
    img = _sharp(size, seed)
    return img.resize((8, 8)).resize(size)  # destroy high frequencies


# ---- pure helpers -----------------------------------------------------------------------

def test_dms_to_deg_north_and_south():
    # 51°28'48" N  ->  51.48
    assert _dms_to_deg((51, 28, 48), "N") == pytest.approx(51.48, abs=1e-6)
    assert _dms_to_deg((51, 28, 48), "S") == pytest.approx(-51.48, abs=1e-6)


def test_blur_score_orders_sharp_above_blurry():
    assert blur_score(_sharp(seed=1)) > blur_score(_blurry(seed=1)) * 5


# ---- ingest stage -----------------------------------------------------------------------

def _make_project(tmp_path: Path, n_sharp: int, n_blurry: int, params: str = "") -> Path:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(n_sharp):
        _sharp(seed=i).save(img_dir / f"sharp_{i:02d}.png")
    for i in range(n_blurry):
        _blurry(seed=100 + i).save(img_dir / f"blur_{i:02d}.png")
    (tmp_path / "project.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "ingest-test"
            [[stage]]
            id = "ingest"
            type = "ingest"
            params = {{ {params} }}
            """
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_ingest_builds_table_and_warns_no_gps(tmp_path):
    proj = _make_project(tmp_path, n_sharp=5, n_blurry=0, params="blur_relative = 0.0")
    outcome = Runner(load_manifest(proj)).run()
    assert outcome.ok
    stage = outcome.stages[0]
    assert stage.status == StageStatus.EXECUTED
    assert stage.metrics == {"total": 5, "kept": 5, "excluded": 0, "with_gps": 0}
    # generated PNGs have no GPS -> non-metric warning
    assert any("non-metric" in i.message for i in stage.issues)


def test_ingest_auto_culls_blurry(tmp_path):
    proj = _make_project(tmp_path, n_sharp=5, n_blurry=3, params="blur_relative = 0.3")
    outcome = Runner(load_manifest(proj)).run()
    stage = outcome.stages[0]
    assert stage.metrics["excluded"] == 3
    assert stage.metrics["kept"] == 5


def test_ingest_errors_when_too_few_usable(tmp_path):
    proj = _make_project(tmp_path, n_sharp=2, n_blurry=0, params="min_images = 3, blur_relative = 0.0")
    outcome = Runner(load_manifest(proj)).run()
    stage = outcome.stages[0]
    # stage still 'executes' (produces a table) but validate raises an ERROR-severity issue
    assert any(i.severity.value == "error" for i in stage.issues)
