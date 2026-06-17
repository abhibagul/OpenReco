# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: build a single self-contained `openreco` executable per OS.

  pip install pyinstaller
  pyinstaller packaging/openreco.spec        # -> dist/openreco(.exe)

The CPU torch build is bundled so the portable plane-sweep dense backend works out of the box
(no end-user pip step). The CUDA torch build (~4.7 GB) and the Gaussian-splat path stay out; NVIDIA
users get higher-quality dense via an external CUDA COLMAP (`openreco fetch-colmap`). A single OS
cannot cross-build the others — run this on each (see CI).
"""
import glob
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# absolute paths derived from the spec location, so the build works from any CWD
ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 — SPECPATH is injected by PyInstaller

# application icon: Windows wants .ico, macOS wants .icns; Linux onefile embeds none.
if sys.platform == "win32":
    icon = os.path.join(ROOT, "packaging/openreco.ico")
elif sys.platform == "darwin":
    icon = os.path.join(ROOT, "packaging/openreco.icns")
else:
    icon = None

datas, binaries, hiddenimports = [], [], []

# heavy native packages: grab their data files, dynamic libs and submodules.
# torch (CPU build) is bundled so the portable plane-sweep dense MVS backend works out of the box on
# any machine — no pip step for end users. cv2 powers ArUco/AprilTag (GCP marker) detection.
for pkg in ("rasterio", "pyproj", "pycolmap", "scipy", "skimage",
            "laspy", "xatlas", "fast_simplification", "PIL", "numpy", "torch", "cv2"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # a missing optional pkg shouldn't abort the build
        print(f"[spec] collect_all({pkg}) skipped: {exc}")

# our own package data + the stages that are imported dynamically (registration by import).
# Paths are relative to the repo root (run `pyinstaller packaging/openreco.spec` from there).
datas += [(os.path.join(ROOT, "openreco/ui/web"), "openreco/ui/web"),
          (os.path.join(ROOT, "openreco/viewer/template"), "openreco/viewer/template"),
          (os.path.join(ROOT, "openreco/engine/report_template.html"), "openreco/engine")]
hiddenimports += collect_submodules("openreco")

# bundle a CUDA COLMAP binary when one is present (Windows: tools/bin/colmap.exe). Kept under
# tools/ so compute.find_colmap() locates it inside the unpacked bundle (sys._MEIPASS/tools/**).
for exe in glob.glob(os.path.join(ROOT, "tools/**/colmap*"), recursive=True):
    dest = os.path.join("tools", os.path.relpath(os.path.dirname(exe), os.path.join(ROOT, "tools")))
    binaries.append((exe, dest))

a = Analysis(
    [os.path.join(ROOT, "packaging/openreco_entry.py")],
    pathex=[ROOT],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    excludes=["torchvision", "torchaudio", "gsplat", "matplotlib",
              "tkinter", "pytest", "IPython", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)

# splash screen shown by the bootloader *during* the one-file unpack (so launch isn't a blank
# screen). Windows/Linux only — PyInstaller has no macOS splash. Text is updated from Python via
# the `pyi_splash` module and closed once the UI is up.
splash = None
if sys.platform in ("win32", "linux"):
    try:
        splash = Splash(  # noqa: F821 — Splash is injected by PyInstaller
            os.path.join(ROOT, "packaging/splash.png"),
            binaries=a.binaries, datas=a.datas,
            text_pos=(140, 278), text_size=11, text_color="#0b1a2b",
            text_default="Starting OpenReco…",
        )
    except Exception as exc:
        print(f"[spec] splash skipped: {exc}")

exe_inputs = [pyz, a.scripts]
if splash:
    exe_inputs += [splash, splash.binaries]
exe_inputs += [a.binaries, a.datas, []]

exe = EXE(
    *exe_inputs,
    name="openreco",
    console=True,
    onefile=True,
    upx=False,
    disable_windowed_traceback=False,
    icon=icon,
)
