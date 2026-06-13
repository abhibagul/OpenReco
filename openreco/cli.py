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

from openreco import __version__, stages  # noqa: F401 — import registers stages
from openreco.engine.cache import compute_key
from openreco.engine.dag import Dag
from openreco.engine.manifest import load_manifest
from openreco.engine.runner import Runner, StageStatus
from openreco.engine.stage import get_stage, registered_types


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _keys_for_manifest(manifest) -> dict[str, dict]:
    """Compute the content-address key for every stage in a manifest (no execution)."""
    dag = Dag.build(manifest.stages)
    keys: dict[str, str] = {}
    info: dict[str, dict] = {}
    for sid in dag.order:
        spec = dag.specs[sid]
        stage = get_stage(spec.type)
        params = {**stage.default_params(), **spec.params}
        input_keys = [keys[d] for d in spec.inputs]
        key = compute_key(spec.type, stage.version, params, input_keys)
        keys[sid] = key
        info[sid] = {"type": spec.type, "params": params, "inputs": spec.inputs, "key": key}
    return info


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
    a = _keys_for_manifest(load_manifest(args.a))
    b = _keys_for_manifest(load_manifest(args.b))
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


def cmd_stages(_args: argparse.Namespace) -> int:
    for t in registered_types():
        print(t)
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
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
