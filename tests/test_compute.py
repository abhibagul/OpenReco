"""Compute/GPU detection — the locator logic (CI-safe; doesn't require a GPU)."""

from __future__ import annotations

from openreco import compute


def _clear():
    # guarded: monkeypatch may have replaced a cached fn with a plain lambda (no cache_clear)
    for fn in (compute.find_colmap, compute.has_nvidia_gpu, compute.colmap_has_cuda):
        clear = getattr(fn, "cache_clear", None)
        if clear:
            clear()


def test_find_colmap_env_override(tmp_path, monkeypatch):
    fake = tmp_path / "colmap.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv("OPENRECO_COLMAP", str(fake))
    _clear()
    assert compute.find_colmap() == fake
    _clear()


def test_gpu_dense_requires_both_gpu_and_binary(monkeypatch):
    _clear()
    monkeypatch.setattr(compute, "has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(compute, "find_colmap", lambda: None)
    assert compute.gpu_dense_available() is False
    _clear()


def test_gpu_dense_available_when_both_present(tmp_path, monkeypatch):
    _clear()
    fake = tmp_path / "colmap"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(compute, "has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(compute, "find_colmap", lambda: fake)
    assert compute.gpu_dense_available() is True
    _clear()
