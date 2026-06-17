"""Offline tests for the COLMAP provisioner (no network / no download)."""

from __future__ import annotations

from pathlib import Path

from openreco import provision


def test_pick_windows_cuda_asset_prefers_cuda_zip():
    assets = [
        {"name": "colmap-3.11.1-windows-nocuda.zip"},
        {"name": "colmap-3.11.1-windows-cuda.zip", "browser_download_url": "u"},
        {"name": "colmap-3.11.1-linux.tar.gz"},
        {"name": "source.zip"},
    ]
    picked = provision._pick_windows_cuda_asset(assets)
    assert picked is not None and picked["name"] == "colmap-3.11.1-windows-cuda.zip"


def test_pick_windows_cuda_asset_none_when_absent():
    assert provision._pick_windows_cuda_asset([{"name": "colmap-windows-nocuda.zip"}]) is None
    assert provision._pick_windows_cuda_asset([]) is None


def test_data_dirs_are_paths_under_openreco():
    d = provision.user_data_dir()
    assert isinstance(d, Path) and d.name.lower() == "openreco"
    assert provision.colmap_dir().parts[-2:] == ("bin", "colmap")


def test_find_user_colmap_discovers_extracted_exe(tmp_path, monkeypatch):
    nested = tmp_path / "colmap-3.11.1" / "bin"
    nested.mkdir(parents=True)
    exe = nested / ("colmap.exe" if provision.sys.platform == "win32" else "colmap")
    exe.write_text("")
    monkeypatch.setattr(provision, "colmap_dir", lambda: tmp_path)
    assert provision.find_user_colmap() == exe


def test_find_user_colmap_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(provision, "colmap_dir", lambda: tmp_path / "nope")
    assert provision.find_user_colmap() is None
