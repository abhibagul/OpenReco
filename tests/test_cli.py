"""CLI packaging surface: `openreco doctor` (env probe) and `openreco init` (project scaffold)."""

from __future__ import annotations

import pytest

from openreco.api import Project
from openreco.cli import main
from openreco.workflow import validate_pipeline


def test_doctor_runs(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "openreco" in out and "Compute" in out and "auto dense backend" in out


def test_init_empty_project(tmp_path, capsys):
    d = tmp_path / "proj"
    assert main(["init", str(d), "--name", "demo"]) == 0
    assert (d / "project.toml").is_file()
    proj = Project.open(d)
    assert proj.manifest.name == "demo" and proj.manifest.stages == []


def test_init_full_pipeline_is_valid(tmp_path):
    d = tmp_path / "full"
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    assert main(["init", str(d), "--crs", "EPSG:32613", "--images", str(imgs)]) == 0
    proj = Project.open(d)
    ids = [s.id for s in proj.manifest.stages]
    assert ids == ["ingest", "sfm", "georef", "mvs", "mesh", "texture", "dsm", "ortho"]
    # the scaffolded wiring must be validation-clean (no errors, no georef-local warning)
    assert validate_pipeline(proj.manifest.stages) == []
    geo = next(s for s in proj.manifest.stages if s.type == "georef")
    assert geo.params["crs_epsg"] == 32613


def test_init_refuses_overwrite(tmp_path, capsys):
    d = tmp_path / "p"
    assert main(["init", str(d)]) == 0
    assert main(["init", str(d)]) == 1                 # exists -> refuse
    assert main(["init", str(d), "--force"]) == 0      # --force overwrites


def test_lightweight_commands_dont_need_stage_deps(tmp_path, monkeypatch):
    # simulate a bare install (reconstruction deps absent): registering stages would fail.
    import openreco.cli as cli

    def boom():
        raise SystemExit("deps missing")
    monkeypatch.setattr(cli, "_register_stages", boom)
    # doctor and init must still work (they don't touch stage implementations)...
    assert cli.main(["doctor"]) == 0
    assert cli.main(["init", str(tmp_path / "p"), "--name", "x"]) == 0
    # ...while run surfaces the missing-deps message instead of an opaque ImportError
    with pytest.raises(SystemExit):
        cli.main(["run", str(tmp_path / "p")])
