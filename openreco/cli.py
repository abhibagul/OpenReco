"""openreco command-line interface.

    openreco run    <project>     run the pipeline (cache-aware: re-run is a no-op)
    openreco resume <project>     alias of run — continues from the cache/checkpoints
    openreco diff   <a> <b>       show which stages differ (and would recompute) between manifests
    openreco report <project>     print the path to the latest run report
    openreco stages               list registered stage types

The headless CLI is the primary interface for Phase 0/1 and the basis for CI/batch use.
The Python API and GUI (Phase 2) will mirror it 1:1.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from openreco import __version__
from openreco.engine.manifest import load_manifest
from openreco.engine.runner import Runner, StageStatus, compute_keys
from openreco.engine.stage import registered_types


def _register_stages() -> None:
    """Import the stage implementations (registration side-effect). Deferred so the lightweight
    commands (doctor / init / crs / --version) run on a bare install without the heavy `slice`
    deps — `doctor` can then report exactly which deps are missing instead of crashing on import."""
    try:
        from openreco import stages  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"missing reconstruction dependency ({e.name}). Install the runtime extras with:\n"
            "    openreco bootstrap          # detect & pip-install them, or\n"
            "    pip install 'openreco[slice]'\n"
            "(run `openreco doctor` to see exactly what's available)") from None


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


_BANNER = r"""
   ___                    ___
  / _ \ _ __  ___ _ _    | _ \ ___ __ ___
 | (_) | '_ \/ -_) ' \   |   / -_) _/ _ \
  \___/| .__/\___|_||_|  |_|_\___\__\___/
       |_|
"""


def _print_banner() -> None:
    """Show the OpenReco ASCII logo on launch (only to an interactive terminal — keeps pipes/CI clean)."""
    if not (sys.stdout and sys.stdout.isatty()):
        return
    print(_BANNER)
    print(f"  OpenReco {__version__} — open-source photogrammetry & 3D reconstruction\n")


def _splash(text: str | None = None, *, close: bool = False) -> None:
    """Drive the PyInstaller boot splash (frozen build only): update its status line, or close it.
    A no-op everywhere else (the `pyi_splash` module exists only inside a splash-enabled bundle)."""
    try:
        import pyi_splash  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return
    try:
        if close:
            pyi_splash.close()
        elif text:
            pyi_splash.update_text(text)
    except Exception:  # noqa: BLE001
        pass


def cmd_run(args: argparse.Namespace) -> int:
    _splash("Loading engine…")
    _print_banner()
    _splash(close=True)
    _preflight_dependencies()
    _register_stages()
    manifest = load_manifest(args.project)
    force = ["*"] if args.force_all else (args.force or [])
    outcome = Runner(manifest, force=force).run()
    print()
    for s in outcome.stages:
        print(f"  {s.status.value:9s}  {s.id:20s}  {s.type:18s}  {s.seconds:7.3f}s  {s.key[:12]}")
    report = outcome.run_dir / "report.html"
    print(f"\nreport: {report}")
    if outcome.ok:
        print("status: OK")
        return 0
    print("status: FAILED")
    for s in outcome.stages:
        if s.status == StageStatus.FAILED:
            print(f"  ! {s.id}: {s.error}")
    return 1


def cmd_resume(args: argparse.Namespace) -> int:
    # resume == run; the cache provides checkpoint/resume semantics.
    return cmd_run(args)


def cmd_diff(args: argparse.Namespace) -> int:
    a = compute_keys(load_manifest(args.a))
    b = compute_keys(load_manifest(args.b))
    ids = sorted(set(a) | set(b))
    changed = 0
    print(f"diff {args.a}  ->  {args.b}\n")
    for sid in ids:
        ka = a.get(sid, {}).get("key")
        kb = b.get(sid, {}).get("key")
        if ka == kb:
            print(f"  =  {sid:20s}  {ka[:12] if ka else '-'}")
            continue
        changed += 1
        if ka is None:
            print(f"  +  {sid:20s}  added              -> {kb[:12]}")
        elif kb is None:
            print(f"  -  {sid:20s}  {ka[:12]} -> removed")
        else:
            print(f"  ~  {sid:20s}  {ka[:12]} -> {kb[:12]}  (would recompute)")
            if args.verbose:
                pa, pb = a[sid]["params"], b[sid]["params"]
                for k in sorted(set(pa) | set(pb)):
                    if pa.get(k) != pb.get(k):
                        print(f"        param {k}: {pa.get(k)!r} -> {pb.get(k)!r}")
    print(f"\n{changed} stage(s) differ" if changed else "\nidentical")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.project)
    latest = manifest.runs_dir / "latest.json"
    if not latest.exists():
        print("no runs yet — run `openreco run` first", file=sys.stderr)
        return 1
    data = json.loads(latest.read_text(encoding="utf-8"))
    # find the run dir holding this report
    runs = sorted([p for p in manifest.runs_dir.glob("*/report.html")], key=lambda p: p.stat().st_mtime)
    if runs:
        print(runs[-1])
    print(f"project={data['project']} ok={data['ok']} stages={len(data['stages'])}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from openreco.batch import discover_projects, run_batch

    _register_stages()
    projects = discover_projects(args.root)
    if not projects:
        print(f"no project.toml found under {args.root}", file=sys.stderr)
        return 1
    print(f"running {len(projects)} project(s) with jobs={args.jobs}\n")
    results = run_batch(projects, jobs=args.jobs)
    ok = 0
    for r in results:
        status = "OK   " if r["ok"] else "FAIL "
        ok += r["ok"]
        detail = r.get("error") or (f"{r['stages']} stages, {r['seconds']}s"
                                    + (f", failed={r['failed']}" if r.get("failed") else ""))
        print(f"  {status} {r['project']:24s} {detail}")
    out = Path(args.root) / "batch_report.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{ok}/{len(results)} succeeded · report: {out}")
    return 0 if ok == len(results) else 1


def cmd_export(args: argparse.Namespace) -> int:
    from openreco.exporters import export_product, list_formats

    avail = list_formats(args.src)
    if not args.to:
        print(f"{args.src}: available formats -> {', '.join(avail)}")
        return 0
    if args.to.lower() not in avail:
        print(f"cannot export as {args.to!r}; choices: {', '.join(avail)}", file=sys.stderr)
        return 1
    crs = int(args.crs) if args.crs else None
    out = export_product(args.src, args.to, args.out or f"{Path(args.src).stem}.{args.to}", crs=crs)
    print(f"wrote {out}" + (f" (reprojected to EPSG:{crs})" if crs else ""))
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from openreco.api import Project
    from openreco.ui.desktop import launch

    _print_banner()
    _splash("Loading engine…")
    _register_stages()

    # no project given (e.g. double-clicked the app) -> a friendly default workspace in the home dir
    project = args.project or str(Path.home() / "OpenReco")
    # open an existing project (a dir with project.toml, or the toml itself), else create a new one
    path = Path(project)
    manifest = path if path.suffix == ".toml" else path / "project.toml"
    proj = Project.open(project) if manifest.is_file() else Project.create(project)
    _preflight_dependencies()
    mode = "browser" if args.browser else "window" if args.window else "auto"
    _splash(close=True)                              # hand off to the UI window/browser
    launch(proj, host=args.host, port=args.port, mode=mode, open_browser=not args.no_browser)
    return 0


def _preflight_dependencies() -> None:
    """On launch, list what's missing for full functionality and (with consent) install it.
    Silent when disabled or non-interactive, so it never blocks startup."""
    if os.environ.get("OPENRECO_NO_AUTOSETUP"):
        return
    if not (sys.stdin and sys.stdin.isatty()):     # don't print/prompt in scripted/windowed launches
        return
    try:
        from openreco import provision
        provision.run_preflight(yes=False)
    except Exception:  # noqa: BLE001 — setup must never stop the UI from launching
        pass


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Print the compute capability probe + dependency status (install diagnostics)."""
    import importlib.util
    import platform

    from openreco import __version__, compute
    print(f"openreco {__version__}  ·  python {platform.python_version()}  ·  "
          f"{platform.system()} {platform.machine()}")

    def mark(ok: bool) -> str:
        return "[ok]" if ok else "[--]"

    d = compute.describe()
    print("\nCompute")
    gpu = d["gpu_name"]
    if gpu:
        extra = (f"  ·  CUDA {d['cuda_version']}  ·  {d['vram_mb'] / 1024:.1f} GB"
                 if d.get("vram_mb") else "")
        print(f"  {mark(True)} GPU: {gpu}{extra}")
    elif d["nvidia_gpu"]:
        print("  [ok] GPU: NVIDIA GPU present (install torch for name/VRAM)")
    else:
        print("  [--] GPU: none detected — CPU / sparse fallback")
    print(f"  {mark(d['colmap_cuda'])} dense MVS: CUDA COLMAP "
          + (f"({d['colmap']})" if d['colmap'] else "binary not found"))
    print(f"  {mark(bool(d['pycolmap_version']))} SfM / matching: pycolmap {d['pycolmap_version'] or '(missing)'}")
    print(f"  {mark(d['torch_device'] is not None)} torch {d['torch_version'] or '(missing)'} "
          f"· device {d['torch_device'] or '-'}")
    print(f"     CPU: {d['cpu_count']} cores  ·  auto dense backend: {d['auto_dense_backend']}")
    if d["nvidia_gpu"] and not d["colmap"]:
        print("     -> run `openreco fetch-colmap` to download GPU dense support (CUDA COLMAP)")

    from openreco.bootstrap import SLICE_DEPS, missing_deps
    print("\nDependencies (the 'slice' extra)")
    for imp in SLICE_DEPS:
        print(f"  {mark(importlib.util.find_spec(imp) is not None)} {imp}")
    print(f"  {mark(importlib.util.find_spec('webview') is not None)} pywebview "
          "(optional — the UI falls back to a browser)")

    missing = missing_deps()
    if missing:
        print(f"\n{len(missing)} reconstruction dep(s) missing — install with:  openreco bootstrap")
    else:
        print("\nall reconstruction dependencies present.")
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Detect missing reconstruction deps (the 'slice' extra) and pip-install them."""
    from openreco.bootstrap import SLICE_DEPS, install, missing_deps
    missing = missing_deps()
    if not missing and not args.upgrade:
        print("all reconstruction dependencies already installed. (run `openreco doctor` to confirm)")
        return 0
    targets = sorted(set(SLICE_DEPS.values())) if args.upgrade else missing
    print(("upgrading: " if args.upgrade else "missing, will install: ") + ", ".join(targets))
    print(f"target interpreter: {sys.executable}")
    if not args.yes:
        try:
            ans = input("proceed with pip install? [y/N] ")
        except EOFError:
            ans = ""
        if ans.strip().lower() not in ("y", "yes"):
            print("cancelled.")
            return 1
    rc = install(targets, upgrade=args.upgrade)
    if rc != 0:
        print(f"pip exited with {rc}", file=sys.stderr)
        return rc
    still = missing_deps()
    print("done — all dependencies present." if not still else f"still missing: {', '.join(still)}")
    return 0


def cmd_fetch_colmap(args: argparse.Namespace) -> int:
    """Download a CUDA COLMAP binary for GPU dense MVS (Windows), or print setup guidance."""
    from openreco import provision
    exe = provision.ensure_colmap(yes=args.yes)
    return 0 if exe else 1


def cmd_setup(args: argparse.Namespace) -> int:
    """Detect everything missing for full functionality (dense MVS deps) and install it, with consent."""
    from openreco import provision
    if not provision.dependency_plan():
        print("everything needed is already present. (run `openreco doctor` for details)")
        return 0
    provision.run_preflight(yes=args.yes)
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Scaffold a new project. With --images, wires the full (validated) photogrammetry chain so
    `openreco run` / `openreco ui` work out of the box; otherwise creates an empty project."""
    from openreco.api import Project
    from openreco.workflow import validate_pipeline

    p = Path(args.project)
    if (p / "project.toml").exists() and not args.force:
        print(f"project already exists at {p} (use --force to overwrite)", file=sys.stderr)
        return 1
    epsg = 0
    if args.crs and str(args.crs).upper().startswith("EPSG:"):
        try:
            epsg = int(str(args.crs).split(":", 1)[1])
        except ValueError:
            epsg = 0
    proj = Project.create(p, name=args.name or p.name, crs=args.crs)
    if args.images:
        (proj.add_stage("ingest", "ingest", params={"image_dir": args.images})
             .add_stage("sfm", "sfm", inputs=["ingest"])
             .add_stage("georef", "georef", inputs=["sfm", "ingest"], params={"crs_epsg": epsg})
             .add_stage("mvs", "mvs", inputs=["ingest", "georef"])
             .add_stage("mesh", "mesh", inputs=["mvs"])
             .add_stage("texture", "texture", inputs=["mesh", "georef", "ingest"])
             .add_stage("dsm", "dsm", inputs=["mvs"])
             .add_stage("ortho", "ortho", inputs=["mvs"]))
    proj.save()
    issues = [i for i in validate_pipeline(proj.manifest.stages) if i["severity"] == "error"]
    print(f"created project at {p / 'project.toml'}"
          + (f" with a {len(proj.manifest.stages)}-stage pipeline" if args.images else " (empty)"))
    if issues:
        print(f"warning: {len(issues)} wiring issue(s) — run `openreco ui` and check", file=sys.stderr)
    print(f"next:  openreco {'run' if args.images else 'ui'} {p}"
          + ("" if args.images else "    # then Add Photos in the UI"))
    return 0


def cmd_crs(args: argparse.Namespace) -> int:
    from openreco.geo.crs import crs_info, search_crs

    if args.search:
        for r in search_crs(args.search, kind=args.kind):
            print(f"  {r['code']:14s} {r['name']}  [{r['kind']}]")
        return 0
    if not args.code:
        print("specify a CRS (e.g. `openreco crs 4326`) or --search <text>", file=sys.stderr)
        return 1
    i = crs_info(args.code)
    print(f"{i['code']}  {i['name']}  ({i['kind']})")
    for key in ("datum", "ellipsoid", "prime_meridian", "unit", "base_crs"):
        v = i.get(key)
        if v:
            print(f"  {key:15s} {v.get('name', '')}  {v.get('code') or ''}".rstrip())
    if i.get("projection"):
        print(f"  {'projection':15s} {i['projection']}")
    print(f"  {'axes':15s} " + ", ".join(f"{a['abbrev']}({a['unit']})" for a in i["axes"]))
    return 0


def cmd_stages(_args: argparse.Namespace) -> int:
    _register_stages()
    for t in registered_types():
        print(t)
    return 0


def cmd_volume(args: argparse.Namespace) -> int:
    from openreco.measure import measure_volume

    base: str | float = args.base
    if base not in ("min", "mean"):
        base = float(base)
    result = measure_volume(args.dsm, base)
    for k, v in result.items():
        print(f"  {k:16s} {v}")
    return 0


def _xy(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


def cmd_profile(args: argparse.Namespace) -> int:
    import json as _json

    from openreco.measure import measure_profile

    result = measure_profile(args.dsm, _xy(getattr(args, "from")), _xy(args.to), args.n)
    print(f"  length_m  {result['length_m']}")
    print(f"  z_min     {result['z_min']}")
    print(f"  z_max     {result['z_max']}")
    print(f"  relief_m  {result['relief_m']}")
    print(f"  samples   {len(result['samples'])}")
    if args.out:
        coords = [[s["x"], s["y"], s["z"]] for s in result["samples"] if s["z"] is not None]
        geo = {"type": "Feature", "properties": {"length_m": result["length_m"]},
               "geometry": {"type": "LineString", "coordinates": coords}}
        with open(args.out, "w", encoding="utf-8") as f:
            _json.dump(geo, f)
        print(f"  wrote     {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openreco", description="OpenReco photogrammetry pipeline")
    p.add_argument("--version", action="version", version=f"openreco {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging / detailed diff")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run the pipeline (cache-aware)")
    pr.add_argument("project", help="path to project.toml or its directory")
    pr.add_argument("--force", action="append", metavar="STAGE_ID", help="force-recompute a stage")
    pr.add_argument("--force-all", action="store_true", help="recompute every stage")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("resume", help="alias of run (continues from cache)")
    ps.add_argument("project")
    ps.add_argument("--force", action="append", metavar="STAGE_ID")
    ps.add_argument("--force-all", action="store_true")
    ps.set_defaults(func=cmd_resume)

    pd = sub.add_parser("diff", help="compare two manifests by content-address keys")
    pd.add_argument("a")
    pd.add_argument("b")
    pd.set_defaults(func=cmd_diff)

    prep = sub.add_parser("report", help="locate the latest run report")
    prep.add_argument("project")
    prep.set_defaults(func=cmd_report)

    pst = sub.add_parser("stages", help="list registered stage types")
    pst.set_defaults(func=cmd_stages)

    pc = sub.add_parser("crs", help="inspect or search coordinate reference systems")
    pc.add_argument("code", nargs="?", help="EPSG code / WKT / PROJ / name to describe")
    pc.add_argument("--search", help="search the EPSG catalog by name or code")
    pc.add_argument("--kind", default="all", choices=["all", "geographic", "projected"])
    pc.set_defaults(func=cmd_crs)

    pb = sub.add_parser("batch", help="run all projects under a directory")
    pb.add_argument("root", help="directory containing project.toml manifests (recursively)")
    pb.add_argument("--jobs", type=int, default=1, help="parallel processes (default 1)")
    pb.set_defaults(func=cmd_batch)

    pe = sub.add_parser("export", help="convert a product (mesh/cloud/raster/...) to another format")
    pe.add_argument("src", help="source product file (e.g. output/mesh.ply, output/dsm.tif)")
    pe.add_argument("--to", help="target format (omit to list available formats)")
    pe.add_argument("--out", help="output path (default: <src stem>.<fmt>)")
    pe.add_argument("--crs", help="reproject raster output to this EPSG (output-CRS selection)")
    pe.set_defaults(func=cmd_export)

    pdoc = sub.add_parser("doctor", help="print compute (GPU/COLMAP/torch) + dependency status")
    pdoc.set_defaults(func=cmd_doctor)

    pboot = sub.add_parser("bootstrap", help="detect & pip-install the reconstruction deps (slice extra)")
    pboot.add_argument("-y", "--yes", action="store_true", help="install without confirmation")
    pboot.add_argument("--upgrade", action="store_true", help="reinstall/upgrade all slice deps")
    pboot.set_defaults(func=cmd_bootstrap)

    pfc = sub.add_parser("fetch-colmap",
                         help="download a CUDA COLMAP for GPU dense MVS (Windows; guidance on Linux/macOS)")
    pfc.add_argument("-y", "--yes", action="store_true", help="download without confirmation")
    pfc.set_defaults(func=cmd_fetch_colmap)

    pset = sub.add_parser("setup", help="install everything needed for full functionality (dense MVS)")
    pset.add_argument("-y", "--yes", action="store_true", help="install without confirmation")
    pset.set_defaults(func=cmd_setup)

    pin = sub.add_parser("init", help="scaffold a new project (optionally a full pipeline)")
    pin.add_argument("project", help="directory to create the project in")
    pin.add_argument("--name", help="project name (default: directory name)")
    pin.add_argument("--crs", help="coordinate system, e.g. EPSG:32613")
    pin.add_argument("--images", help="image folder — wires the full photogrammetry pipeline")
    pin.add_argument("--force", action="store_true", help="overwrite an existing project.toml")
    pin.set_defaults(func=cmd_init)

    pu = sub.add_parser("ui", help="launch the UI (native window if pywebview present, else browser)")
    pu.add_argument("project", nargs="?", default=None,
                    help="project dir/toml (created if absent; defaults to ~/OpenReco)")
    pu.add_argument("--host", default="127.0.0.1")
    pu.add_argument("--port", type=int, default=8000)
    pu.add_argument("--window", action="store_true", help="force a native desktop window")
    pu.add_argument("--browser", action="store_true", help="force the system browser")
    pu.add_argument("--no-browser", action="store_true", help="don't auto-open a browser")
    pu.set_defaults(func=cmd_ui)

    pv = sub.add_parser("volume", help="cut/fill volume of a DSM GeoTIFF")
    pv.add_argument("dsm", help="path to a DSM GeoTIFF (e.g. output/dsm.tif)")
    pv.add_argument("--base", default="min", help="reference: min | mean | <elevation>")
    pv.set_defaults(func=cmd_volume)

    pp = sub.add_parser("profile", help="elevation cross-section across a DSM")
    pp.add_argument("dsm", help="path to a DSM GeoTIFF")
    pp.add_argument("--from", required=True, metavar="X,Y", help="start point in DSM CRS units")
    pp.add_argument("--to", required=True, metavar="X,Y", help="end point in DSM CRS units")
    pp.add_argument("--n", type=int, default=200, help="number of samples")
    pp.add_argument("--out", help="optional GeoJSON LineString output path")
    pp.set_defaults(func=cmd_profile)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw = list(argv) if argv is not None else sys.argv[1:]
    if not raw:                       # double-clicked / launched with no command -> open the GUI
        raw = ["ui"]
    args = parser.parse_args(raw)
    _setup_logging(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
