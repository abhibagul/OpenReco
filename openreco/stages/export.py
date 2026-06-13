"""Export & publish — pipeline stage 8 (terminal).

Assembles a self-contained, shareable output: the point cloud (PLY/LAS), mesh (PLY/OBJ),
DSM/ortho GeoTIFFs, a processing summary, and a static three.js web viewer that loads the
mesh + cloud and supports distance measurement. Copies the bundle to <project>/output/ so
it's easy to find and zip/host. Whatever upstream stages are wired as inputs get included;
missing ones are simply skipped.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from openreco.engine.context import Issue, RunContext, Severity, StageResult
from openreco.engine.stage import Stage, register_stage
from openreco.io.gltf import write_glb
from openreco.io.pointcloud import read_mesh_ply, write_obj
from openreco.viewer import TEMPLATE_DIR


@register_stage
class Export(Stage):
    type = "export"
    version = "2"  # v2: adds portable glTF (.glb) mesh export

    def default_params(self) -> dict[str, Any]:
        return {"output_dir": "output", "copy_to_project": True}

    def run(self, ctx: RunContext) -> StageResult:
        site = ctx.artifact_path("site")
        site.mkdir(parents=True, exist_ok=True)
        included: list[str] = []
        crs = "local"
        unit = "units"

        # point cloud
        if "mvs" in ctx.inputs:
            self._copy(ctx, "mvs", "points", site / "points.ply")
            included.append("points.ply")
            meta = ctx.read_input_json("mvs", "meta")
            crs = meta.get("crs", "local")
            unit = "m" if meta.get("crs_epsg") else "units"
            if "las" in ctx.inputs["mvs"].artifacts:
                self._copy(ctx, "mvs", "las", site / "points.las")
                included.append("points.las")

        # mesh -> PLY + OBJ
        has_mesh = "mesh" in ctx.inputs
        if has_mesh:
            mesh_src = ctx.input_artifact("mesh", "mesh")
            shutil.copyfile(mesh_src, site / "mesh.ply")
            included.append("mesh.ply")
            v, fcs, vc = read_mesh_ply(mesh_src)
            write_obj(site / "mesh.obj", v, fcs, vc)
            included.append("mesh.obj")
            write_glb(site / "mesh.glb", v, fcs, vc)   # portable glTF for VFX/AEC/web pipelines
            included.append("mesh.glb")

        # rasters
        for dep, name, fn in (("dsm", "dsm", "dsm.tif"), ("ortho", "ortho", "ortho.tif"),
                              ("coverage", "coverage", "coverage.tif")):
            if dep in ctx.inputs:
                self._copy(ctx, dep, name, site / fn)
                included.append(fn)
        if "coverage" in ctx.inputs and "preview" in ctx.inputs["coverage"].artifacts:
            self._copy(ctx, "coverage", "preview", site / "coverage.png")
            included.append("coverage.png")

        # viewer
        self._write_viewer(ctx, site, crs, unit,
                           points="points.ply" if "mvs" in ctx.inputs else "",
                           mesh="mesh.ply" if has_mesh else "")
        included += ["index.html", "serve.py"]

        summary = self._summary(ctx, included, crs)
        (site / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # publish copy to <project>/output for convenience (outside the cache)
        published = None
        if ctx.params["copy_to_project"]:
            published = ctx.project_dir / ctx.params["output_dir"]
            if published.exists():
                shutil.rmtree(published, ignore_errors=True)
            shutil.copytree(site, published)
            ctx.logger.info("published shareable bundle -> %s", published)

        return StageResult(
            artifacts={"site": "site"},
            metrics={"files": len(included), "crs": crs,
                     "published_to": str(published) if published else None},
        )

    def _copy(self, ctx, dep, artifact, dst: Path) -> None:
        shutil.copyfile(ctx.input_artifact(dep, artifact), dst)

    def _write_viewer(self, ctx, site: Path, crs: str, unit: str, points: str, mesh: str) -> None:
        html = (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
        for key, val in {
            "__PROJECT__": ctx.project_dir.name, "__CRS__": crs,
            "__POINTS__": points, "__MESH__": mesh, "__UNIT__": unit,
        }.items():
            html = html.replace(key, val)
        (site / "index.html").write_text(html, encoding="utf-8")
        (site / "serve.py").write_text(
            "import http.server\n"
            "import socketserver\n"
            "PORT = 8000\n"
            "print(f'Open http://localhost:{PORT}/  (Ctrl+C to stop)')\n"
            "socketserver.TCPServer(('', PORT), http.server.SimpleHTTPRequestHandler).serve_forever()\n",
            encoding="utf-8",
        )

    def _summary(self, ctx, included, crs) -> dict[str, Any]:
        s: dict[str, Any] = {"project": ctx.project_dir.name, "crs": crs, "files": included,
                             "stages": {}}
        for dep in ctx.inputs:
            try:
                # surface each upstream stage's small JSON metadata if present
                for art in ("meta", "georef"):
                    if art in ctx.inputs[dep].artifacts:
                        s["stages"][dep] = json.loads(ctx.input_artifact(dep, art).read_text("utf-8"))
                        break
            except Exception:  # noqa: BLE001
                pass
        return s

    def validate(self, result: StageResult, ctx: RunContext) -> list[Issue]:
        if result.metrics["files"] <= 2:
            return [Issue(Severity.WARNING, "export bundle has little content — wire mvs/mesh/dsm/ortho as inputs")]
        return [Issue(Severity.INFO, "view it: cd into the output dir and run `python serve.py`, "
                      "then open http://localhost:8000/")]
