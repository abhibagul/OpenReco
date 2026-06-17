"""Provision the external CUDA COLMAP binary used for GPU dense MVS.

The packaged binary ships no COLMAP (PatchMatch stereo is CUDA-only and the PyPI pycolmap wheel is
CPU-only), so on an NVIDIA machine we fetch an official prebuilt CUDA COLMAP at runtime and drop it
in a per-user data dir, where `compute.find_colmap()` then picks it up automatically.

Platform reality (why this can't be one-size-fits-all):
  Windows — official ``colmap-*-windows-cuda.zip`` on GitHub releases: download + unzip (automatic).
  Linux   — no official prebuilt CUDA binary; we print apt/conda/build guidance (manual step).
  macOS   — no CUDA on Apple hardware; GPU dense isn't available (use the torch plane-sweep backend).

Nothing here runs without consent — downloading hundreds of MB and writing an executable is a side
effect the user opts into (``openreco fetch-colmap`` or the launch prompt's [Y/n]).
"""

from __future__ import annotations

import json
import os
import shutil
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

_GH_LATEST = "https://api.github.com/repos/colmap/colmap/releases/latest"


def user_data_dir() -> Path:
    """Per-user, writable data dir for OpenReco (where a fetched COLMAP is stored)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "OpenReco"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "OpenReco"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "openreco"


def colmap_dir() -> Path:
    return user_data_dir() / "bin" / "colmap"


def find_user_colmap() -> Path | None:
    """A COLMAP executable previously fetched into the per-user data dir, or None."""
    d = colmap_dir()
    if d.is_dir():
        for name in ("colmap.exe", "colmap.bat", "colmap"):
            hits = sorted(d.rglob(name))
            if hits:
                return hits[0]
    return None


def _pick_windows_cuda_asset(assets: list[dict]) -> dict | None:
    """The Windows CUDA .zip from a GitHub release's asset list (skips the -nocuda build)."""
    for a in assets:
        name = (a.get("name") or "").lower()
        if name.endswith(".zip") and "windows" in name and "cuda" in name and "nocuda" not in name:
            return a
    return None


def _fetch_latest_release() -> dict:
    req = urllib.request.Request(
        _GH_LATEST, headers={"User-Agent": "OpenReco", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=30) as r:
        return json.load(r)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "OpenReco"})
    with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  downloading {done / 1e6:6.1f} / {total / 1e6:6.1f} MB", end="")
    print()


def install_colmap_windows() -> Path | None:
    """Download + extract the official CUDA COLMAP into the per-user data dir. Returns the exe path."""
    try:
        rel = _fetch_latest_release()
    except Exception as exc:  # noqa: BLE001
        print(f"could not reach the COLMAP releases API ({exc!r}).")
        print("download manually from https://github.com/colmap/colmap/releases and set OPENRECO_COLMAP")
        return None
    asset = _pick_windows_cuda_asset(rel.get("assets", []))
    if not asset:
        print("no windows-cuda COLMAP asset in the latest release — install manually and set OPENRECO_COLMAP")
        return None

    target = colmap_dir()
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / asset["name"]
    print(f"COLMAP {rel.get('tag_name', '')}: {asset['name']} (~{asset.get('size', 0) / 1e6:.0f} MB)")
    try:
        _download(asset["browser_download_url"], zip_path)
        print("  extracting…")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target)
    except Exception as exc:  # noqa: BLE001
        print(f"download/extract failed ({exc!r}).")
        return None
    finally:
        zip_path.unlink(missing_ok=True)

    exe = find_user_colmap()
    if exe:
        print(f"  installed COLMAP -> {exe}")
    else:
        print("  extracted, but no colmap executable was found in the archive.")
    return exe


def _linux_guidance() -> None:
    print("No official prebuilt CUDA COLMAP exists for Linux. Options:")
    print("  • Ubuntu/Debian:  sudo apt-get install colmap   (may be CPU-only / older)")
    print("  • conda-forge:    conda install -c conda-forge colmap")
    print("  • build with CUDA from source: https://colmap.github.io/install.html")
    print("Then set OPENRECO_COLMAP to the colmap binary, or put `colmap` on PATH.")


def _macos_guidance() -> None:
    print("macOS has no CUDA, so GPU dense (COLMAP PatchMatch) isn't available on this hardware.")
    print("Use the portable plane-sweep backend instead — run from a Python install with torch:")
    print('  pip install -e ".[slice]" && pip install torch   (dense_backend = "auto" or "planesweep")')


def _confirm(prompt: str, yes: bool) -> bool:
    if yes:
        return True
    if not (sys.stdin and sys.stdin.isatty()):
        return False                       # never block a non-interactive / windowed launch
    try:
        return input(f"{prompt} [Y/n] ").strip().lower() in ("", "y", "yes")
    except EOFError:
        return False


def _clear_caches() -> None:
    from openreco import compute
    for fn in (compute.find_colmap, compute.colmap_has_cuda, compute.has_nvidia_gpu):
        clear = getattr(fn, "cache_clear", None)
        if clear:
            clear()


def _pip_install(args: list[str]) -> bool:
    """pip-install into the current interpreter. No-op (False) inside the frozen binary, which has
    no importable site-packages to install into."""
    if getattr(sys, "frozen", False):
        return False
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print("  $ " + " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def install_torch() -> bool:
    """Install PyTorch for the portable plane-sweep dense backend. CPU build off the NVIDIA path
    (Apple Silicon gets MPS from the default wheel)."""
    if sys.platform == "darwin":
        return _pip_install(["torch"])
    return _pip_install(["torch", "--index-url", "https://download.pytorch.org/whl/cpu"])


def _install_slice(pkgs: list[str]) -> bool:
    from openreco.bootstrap import install
    return install(pkgs) == 0


def dependency_plan() -> list[dict]:
    """What this machine is missing for a fully working pipeline (esp. dense reconstruction).

    Each item: {title, detail, size, action}. `action` is a no-arg callable that installs it, or
    None when the step can't be automated here (manual setup; we still surface it)."""
    from openreco import compute
    from openreco.bootstrap import missing_deps

    frozen = bool(getattr(sys, "frozen", False))
    plan: list[dict] = []

    miss = [] if frozen else missing_deps()           # bundled in the frozen app; pip-only otherwise
    if miss:
        plan.append({
            "title": "Reconstruction libraries",
            "detail": f"{len(miss)} Python package(s): {', '.join(miss)}",
            "size": "~80 MB", "action": lambda m=miss: _install_slice(m)})

    has_colmap = compute.find_colmap() is not None
    nvidia = compute.has_nvidia_gpu()
    torch_ok = compute.torch_device() is not None

    if nvidia and not has_colmap:
        if sys.platform == "win32":
            plan.append({
                "title": "CUDA COLMAP — GPU dense MVS",
                "detail": "official NVIDIA build; enables high-quality dense 3D reconstruction",
                "size": "~250 MB", "action": install_colmap_windows})
        else:
            plan.append({
                "title": "CUDA COLMAP — GPU dense MVS",
                "detail": "no auto-download on this OS — install via apt/conda or a CUDA source build",
                "size": "", "action": None})
    elif not nvidia and not torch_ok:
        if frozen:
            plan.append({
                "title": "Dense MVS on this (non-NVIDIA) machine",
                "detail": "needs PyTorch, which can't be added to the packaged app — run the Python "
                          'install: pip install -e ".[slice]" && pip install torch',
                "size": "", "action": None})
        else:
            plan.append({
                "title": "PyTorch — portable (CPU/Apple-GPU) dense MVS",
                "detail": "plane-sweep dense reconstruction without an NVIDIA GPU",
                "size": "~200 MB", "action": install_torch})
    return plan


def run_preflight(yes: bool = False) -> None:
    """Show what's needed for full functionality and, with consent, install it. Always returns
    (skipping never blocks the app); manual-only items are surfaced but don't prompt."""
    try:
        plan = dependency_plan()
    except Exception:  # noqa: BLE001
        return
    if not plan:
        return

    bar = "=" * 66
    print(f"\n{bar}\nOpenReco — setup\n")
    print("For full functionality (including dense 3D reconstruction), the following")
    print("need to be installed on this machine:\n")
    for it in plan:
        size = f"   [{it['size']}]" if it["size"] else ""
        print(f"  • {it['title']}{size}\n      {it['detail']}")

    actionable = [it for it in plan if it["action"]]
    if not actionable:
        print("\nThese need manual setup (see notes above). Starting OpenReco…\n" + bar + "\n")
        return

    print()
    if not _confirm("Download & install these now?", yes):
        print("\nSkipped — starting OpenReco without them. Install later with `openreco setup`,")
        print("`openreco fetch-colmap`, or `openreco bootstrap`. (Disable this with OPENRECO_NO_AUTOSETUP=1.)")
        print(bar + "\n")
        return

    for it in actionable:
        print(f"\n-> {it['title']}")
        try:
            it["action"]()
        except Exception as exc:  # noqa: BLE001 — one failed item shouldn't abort the rest / the app
            print(f"   failed: {exc!r}")
    _clear_caches()
    print("\nSetup done. Starting OpenReco…\n" + bar + "\n")


def ensure_colmap(yes: bool = False) -> Path | None:
    """Make a CUDA COLMAP available for GPU dense MVS, downloading it on Windows (with consent).

    Returns the executable path if one is (now) available, else None. Idempotent: if COLMAP is
    already found, or there's no NVIDIA GPU, it does nothing destructive.
    """
    from openreco import compute

    existing = compute.find_colmap()
    if existing:
        print(f"COLMAP already available: {existing}")
        return existing

    if not compute.has_nvidia_gpu():
        print("no NVIDIA GPU detected (nvidia-smi not found) — GPU dense MVS needs an NVIDIA GPU.")
        if sys.platform == "darwin":
            _macos_guidance()
        return None

    if sys.platform == "win32":
        if not _confirm("Download the official CUDA COLMAP (~250 MB) now?", yes):
            print("skipped — run `openreco fetch-colmap` when you're ready.")
            return None
        exe = install_colmap_windows()
        if exe:
            _clear_caches()
        return exe

    if sys.platform.startswith("linux"):
        _linux_guidance()
        return None

    _macos_guidance()
    return None
