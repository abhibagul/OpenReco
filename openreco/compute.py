"""Compute capability detection: NVIDIA GPU presence and the COLMAP CUDA binary.

The PyPI `pycolmap` wheel is CPU-only (`pycolmap.has_cuda == False`), so GPU dense MVS
(PatchMatch stereo + fusion, which are CUDA-only) is driven by an external, CUDA-enabled
COLMAP executable. This module finds that executable and reports whether a usable GPU dense
path exists, so stages can branch (GPU dense vs. CPU sparse fallback) without probing hardware
themselves.

Locating colmap.exe, in order:
  1. $OPENRECO_COLMAP (explicit path to the binary)
  2. a bundled copy under <repo>/tools/**/colmap(.exe)
  3. `colmap` on PATH
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def has_nvidia_gpu() -> bool:
    """True if an NVIDIA GPU + driver is present (nvidia-smi runs successfully)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        return subprocess.run([smi], capture_output=True, timeout=15).returncode == 0
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=1)
def find_colmap() -> Path | None:
    """Path to a COLMAP executable, or None. Prefers an explicit env var, then a bundled copy."""
    env = os.environ.get("OPENRECO_COLMAP")
    if env and Path(env).exists():
        return Path(env)
    names = ("colmap.exe", "colmap.bat", "colmap")
    tools = _REPO_ROOT / "tools"
    if tools.is_dir():
        for name in names:
            hits = sorted(tools.rglob(name))
            if hits:
                return hits[0]
    found = shutil.which("colmap")
    return Path(found) if found else None


@lru_cache(maxsize=1)
def colmap_has_cuda() -> bool:
    """Whether the located COLMAP binary supports CUDA (best-effort: GPU present + binary found).
    COLMAP's CUDA build only exposes patch_match_stereo at runtime when a GPU is available."""
    return has_nvidia_gpu() and find_colmap() is not None


def gpu_dense_available() -> bool:
    """True when real GPU dense MVS can run (CUDA GPU + a COLMAP binary to drive it)."""
    return colmap_has_cuda()


@lru_cache(maxsize=1)
def torch_device() -> str | None:
    """Best available torch device for the portable (plane-sweep) dense backend, or None if torch
    is absent. Covers every vendor from one codebase: 'cuda' (NVIDIA), 'mps' (Apple Silicon),
    'cpu' (anywhere; AMD ROCm also presents as 'cuda' on torch-rocm builds)."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dense_backend(prefer: str = "auto") -> str:
    """Pick a dense MVS backend by capability:
       colmap_cuda  — best quality, NVIDIA + CUDA COLMAP binary
       planesweep   — portable torch plane-sweep (CUDA/MPS/ROCm/CPU)
       sparse       — no dense; reuse the SfM sparse cloud
    `prefer` forces a specific backend (validated against availability by the caller)."""
    if prefer != "auto":
        return prefer
    if colmap_has_cuda():
        return "colmap_cuda"
    if torch_device() is not None:
        return "planesweep"
    return "sparse"


def describe() -> dict:
    """Human/JSON-friendly capability snapshot (for reports / the 'doctor')."""
    return {
        "nvidia_gpu": has_nvidia_gpu(),
        "colmap": str(find_colmap()) if find_colmap() else None,
        "colmap_cuda": colmap_has_cuda(),
        "torch_device": torch_device(),
        "auto_dense_backend": select_dense_backend(),
    }
