"""Runtime dependency bootstrap for the `slice` extra.

For users who run OpenReco from a Python install (not a frozen binary): detect which optional
reconstruction dependencies are missing and pip-install them into the running interpreter. This
deliberately does NOT auto-install silently — installing into someone's environment is a side
effect they should consent to (a virtualenv is recommended) — so callers confirm first.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

# import name -> pip distribution name (the `slice` optional-dependency set)
SLICE_DEPS: dict[str, str] = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pycolmap": "pycolmap",
    "pyproj": "pyproj",
    "rasterio": "rasterio",
    "laspy": "laspy",
    "PIL": "pillow",
    "xatlas": "xatlas",
    "fast_simplification": "fast-simplification",
    "skimage": "scikit-image",
}


def missing_deps() -> list[str]:
    """pip names of the slice deps that aren't importable (checked without importing them)."""
    out = []
    for imp, dist in SLICE_DEPS.items():
        try:
            found = importlib.util.find_spec(imp) is not None
        except Exception:  # noqa: BLE001 — a broken/partial install also counts as missing
            found = False
        if not found:
            out.append(dist)
    return out


def install(packages: list[str], upgrade: bool = False) -> int:
    """pip-install `packages` into the current interpreter. Returns the pip exit code."""
    if not packages:
        return 0
    cmd = [sys.executable, "-m", "pip", "install", *(["--upgrade"] if upgrade else []), *packages]
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd).returncode
