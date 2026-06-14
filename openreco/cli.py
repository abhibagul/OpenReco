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
import sys
from pathlib import Path

from openreco import __version__, stages  # noqa: F401 — import registers stages
from openreco.engine.manifest import load_manifest
from openreco.engine.runner import Runner, StageStatus, compute_keys
from openreco.engine.stage import registered_types


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_run(args: argparse.Namespace) -> int:
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
    out = export_product(args.src, args.to, args.out or f"{Path(args.src).stem}.{args.to}")
    print(f"wrote {out}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    import webbrowser

    from openreco.api import Project
    from openreco.ui.server import serve

    proj = Project.open(args.project) if Path(args.project).exists() else Project.create(args.project)
    httpd = serve(proj, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"OpenReco UI for '{proj.manifest.name}' -> {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
    return 0


def cmd_stages(_args: argparse.Namespace) -> int:
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

    pb = sub.add_parser("batch", help="run all projects under a directory")
    pb.add_argument("root", help="directory containing project.toml manifests (recursively)")
    pb.add_argument("--jobs", type=int, default=1, help="parallel processes (default 1)")
    pb.set_defaults(func=cmd_batch)

    pe = sub.add_parser("export", help="convert a product (mesh/cloud/raster/...) to another format")
    pe.add_argument("src", help="source product file (e.g. output/mesh.ply, output/dsm.tif)")
    pe.add_argument("--to", help="target format (omit to list available formats)")
    pe.add_argument("--out", help="output path (default: <src stem>.<fmt>)")
    pe.set_defaults(func=cmd_export)

    pu = sub.add_parser("ui", help="launch the local web UI (layer tree, params, run, 3D viewport)")
    pu.add_argument("project", nargs="?", default=".", help="project dir/toml (created if absent)")
    pu.add_argument("--host", default="127.0.0.1")
    pu.add_argument("--port", type=int, default=8000)
    pu.add_argument("--no-browser", action="store_true")
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
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
