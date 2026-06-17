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


def _patch_caps(monkeypatch, *, nvidia, colmap, torch):
    from openreco import compute
    monkeypatch.setattr(compute, "has_nvidia_gpu", lambda: nvidia)
    monkeypatch.setattr(compute, "find_colmap", lambda: (provision.Path("colmap") if colmap else None))
    monkeypatch.setattr(compute, "torch_device", lambda: ("cpu" if torch else None))
    # treat as a normal (non-frozen) Python install for these plan tests
    monkeypatch.delenv("OPENRECO_NO_AUTOSETUP", raising=False)
    monkeypatch.setattr(provision.sys, "frozen", False, raising=False)
    import openreco.bootstrap as bs
    monkeypatch.setattr(bs, "missing_deps", lambda: [])


def test_plan_empty_when_nvidia_and_colmap_present(monkeypatch):
    _patch_caps(monkeypatch, nvidia=True, colmap=True, torch=False)
    assert provision.dependency_plan() == []


def test_plan_offers_colmap_when_nvidia_without_colmap(monkeypatch):
    _patch_caps(monkeypatch, nvidia=True, colmap=False, torch=False)
    titles = " ".join(it["title"] for it in provision.dependency_plan())
    assert "COLMAP" in titles


def test_plan_offers_torch_when_no_gpu_and_no_torch(monkeypatch):
    _patch_caps(monkeypatch, nvidia=False, colmap=False, torch=False)
    plan = provision.dependency_plan()
    assert any("PyTorch" in it["title"] and it["action"] for it in plan)
