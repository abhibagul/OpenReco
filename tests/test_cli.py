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


def test_bootstrap_module_detection():
    from openreco import bootstrap
    # mapping is import-name -> pip-name; PIL/skimage are the classic mismatches
    assert bootstrap.SLICE_DEPS["PIL"] == "pillow"
    assert bootstrap.SLICE_DEPS["skimage"] == "scikit-image"
    assert bootstrap.install([]) == 0                   # no-op, never shells out
    assert isinstance(bootstrap.missing_deps(), list)


def test_bootstrap_installs_only_missing(monkeypatch):
    import openreco.bootstrap as bs
    calls = {}
    monkeypatch.setattr(bs, "missing_deps", lambda: ["rasterio", "scipy"])
    def fake_install(pkgs, upgrade=False):
        calls["pkgs"] = pkgs
        return 0
    monkeypatch.setattr(bs, "install", fake_install)
    assert main(["bootstrap", "--yes"]) == 0
    assert calls["pkgs"] == ["rasterio", "scipy"]       # only the missing ones, with -y (no prompt)


def test_bootstrap_noop_when_all_present(monkeypatch, capsys):
    import openreco.bootstrap as bs
    monkeypatch.setattr(bs, "missing_deps", lambda: [])
    assert main(["bootstrap"]) == 0
    assert "already installed" in capsys.readouterr().out


def test_no_args_launches_ui(monkeypatch):
    # double-click / no command -> open the GUI (cmd_ui), not an argparse error
    import openreco.cli as cli
    called = {}

    def fake_ui(args):
        called["ran"] = True
        return 0
    monkeypatch.setattr(cli, "cmd_ui", fake_ui)
    assert cli.main([]) == 0
    assert called.get("ran")


def test_ui_creates_project_when_manifest_absent(tmp_path, monkeypatch):
    # `openreco ui <dir>` on a folder without project.toml must create it, not crash
    import openreco.ui.desktop as desktop
    captured = {}
    monkeypatch.setattr(desktop, "launch", lambda proj, **k: captured.setdefault("proj", proj))
    d = tmp_path / "newproj"
    d.mkdir()                                          # dir exists but has no project.toml
    assert main(["ui", str(d), "--no-browser"]) == 0
    assert captured["proj"].manifest.project_dir == d.resolve()


def test_ui_opens_existing_project(tmp_path, monkeypatch):
    import openreco.ui.desktop as desktop
    assert main(["init", str(tmp_path / "p"), "--name", "keep"]) == 0
    captured = {}
    monkeypatch.setattr(desktop, "launch", lambda proj, **k: captured.setdefault("proj", proj))
    assert main(["ui", str(tmp_path / "p")]) == 0
    assert captured["proj"].manifest.name == "keep"


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
