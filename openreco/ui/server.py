"""Local UI server (stdlib http.server) exposing the engine to the web frontend.

Endpoints (JSON unless noted):
  GET  /                      -> the single-page app (web/index.html)
  GET  /app.js, /viewer.js    -> static frontend assets
  GET  /api/stages            -> stage_info() (palette + parameter-panel schemas)
  GET  /api/project           -> {name, crs, stages[], layers[]} (layer tree + last-run status/artifacts)
  POST /api/project           -> set project metadata {crs}; re-saves project.toml (CRS picker)
  POST /api/stage             -> add/update a stage {id,type,inputs,params}; re-saves project.toml
  POST /api/run               -> start a run in a background thread (force? in body); 202
  GET  /api/events            -> Server-Sent Events: live run events (stage_start/progress/.../run_done)
  GET  /api/file?path=...      -> serve an artifact file (sandboxed to the project dir) for the viewer
  GET  /api/images?chunk=...   -> source images of a chunk (for the Photos pane + GCP picking)
  GET  /api/browse?path=...    -> list sub-folders + image files of a dir (Add-Photos file picker)
  GET  /api/thumb?path=...     -> serve an image file from anywhere (picker previews; image-only)
  POST /api/add_photos         -> create an ingest layer from chosen image paths {paths,chunk,id}
  POST /api/remove_photo       -> drop one image from an ingest layer {layer,name} (select list)
  GET  /api/markers            -> saved GCP/markers (markers.json)
  POST /api/markers            -> save GCP/markers; also writes gcps.csv consumable by the georef stage
  POST /api/use_gcps           -> point a chunk's georef stage(s) at gcps.csv (method=gcp + CRS)
  GET  /api/raster_png?path=... -> render a GeoTIFF (ortho/DSM/index) to PNG for the 2D Ortho view

Zero third-party deps; ThreadingHTTPServer + a per-run event queue for SSE.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from openreco import stage_info
from openreco.api import Project
from openreco.engine.runner import compute_keys
from openreco.ui import WEB_DIR

_CT = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
       ".json": "application/json", ".ply": "application/octet-stream",
       ".glb": "model/gltf-binary", ".tif": "image/tiff", ".png": "image/png",
       ".jpg": "image/jpeg", ".geojson": "application/json", ".las": "application/octet-stream"}


def _unique_id(base: str, taken: set[str]) -> str:
    n = 1
    while f"{base}{n}" in taken:
        n += 1
    return f"{base}{n}"


class AppState:
    """Server-wide state: the open project + the current run's event stream."""

    def __init__(self, project: Project):
        self.project = project
        self.events: queue.Queue = queue.Queue()
        self.running = False
        self.lock = threading.Lock()

    # ---- data for the frontend ----
    def project_json(self) -> dict:
        m = self.project.manifest
        keys = compute_keys(m)
        last = self._last_run()
        layers = []
        for s in m.stages:
            run = last.get(s.id, {})
            layers.append({
                "id": s.id, "type": s.type, "inputs": s.inputs, "params": s.params, "chunk": s.chunk,
                "status": run.get("status"), "metrics": run.get("metrics", {}),
                "artifacts": run.get("artifacts", {}), "key": keys.get(s.id, {}).get("key"),
            })
        return {"name": m.name, "crs": m.crs, "project_dir": str(m.project_dir),
                "chunks": m.chunk_names(), "layers": layers}

    def images_for_chunk(self, chunk: str | None) -> dict:
        """Source images of a chunk's ingest layer(s) — for the Photos pane + GCP picking.

        Uses the ingest output (with GPS/cull flags) once it has run; before that, scans the
        configured image folder directly so photos appear as soon as they're added."""
        from openreco.io.images import list_images

        last = self._last_run()
        out: list[dict] = []
        image_dir = ""
        proj_dir = self.project.manifest.project_dir
        for s in self.project.manifest.stages:
            if s.type != "ingest" or (chunk and s.chunk != chunk):
                continue
            art = last.get(s.id, {}).get("artifacts", {}).get("images")
            if art and Path(art).is_file():
                data = json.loads(Path(art).read_text(encoding="utf-8"))
                image_dir = data.get("image_dir", "")
                for im in data.get("images", []):
                    out.append({"name": im["name"], "path": str(Path(image_dir) / im["name"]),
                                "lat": im.get("lat"), "lon": im.get("lon"),
                                "excluded": im.get("excluded", False), "layer": s.id})
                continue
            # not run yet: scan the configured folder so the user sees the photos immediately
            idir = s.params.get("image_dir", "images")
            p = Path(idir) if Path(idir).is_absolute() else (proj_dir / idir)
            if p.is_dir():
                image_dir = str(p)
                select = set(s.params.get("select") or [])
                for f in list_images(p):
                    if select and f.name not in select:
                        continue
                    out.append({"name": f.name, "path": str(f), "lat": None, "lon": None,
                                "excluded": False, "layer": s.id})
        return {"image_dir": image_dir, "images": out}

    def browse(self, path: str | None) -> dict:
        """List sub-folders + image files of a directory (a local file picker for Add Photos).

        With no path: Windows drive roots (else '/'). Local-first desktop tool bound to 127.0.0.1."""
        from openreco.io.images import IMAGE_SUFFIXES
        if not path:
            if os.name == "nt":
                import string
                drives = [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]
                return {"path": "", "parent": None, "dirs": drives, "images": []}
            path = "/"
        p = Path(path)
        if not p.is_dir():
            return {"error": f"not a directory: {path}"}
        dirs, images = [], []
        try:
            for e in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                try:
                    if e.is_dir():
                        dirs.append(str(e))
                    elif e.suffix.lower() in IMAGE_SUFFIXES:
                        images.append({"name": e.name, "path": str(e)})
                except OSError:
                    continue
        except PermissionError:
            return {"error": f"permission denied: {path}"}
        parent = str(p.parent) if p.parent != p else None
        return {"path": str(p), "parent": parent, "dirs": dirs, "images": images}

    def add_photos(self, paths: list[str], chunk: str, layer_id: str | None) -> dict:
        """Create an ingest layer from chosen image paths. One folder -> image_dir (+ select for a
        subset); spanning folders -> stage copies into the project so SfM keeps a single image root."""
        import shutil

        from openreco.io.images import IMAGE_SUFFIXES
        files = [Path(p) for p in paths]
        files = [f for f in files if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
        if not files:
            raise ValueError("no valid image files selected")
        ids = {s.id for s in self.project.manifest.stages}
        lid = layer_id or _unique_id("photos", ids)
        params: dict
        dirs = {f.parent for f in files}
        if len(dirs) == 1:
            folder = next(iter(dirs))
            all_names = {p.name for p in folder.iterdir()
                         if p.suffix.lower() in IMAGE_SUFFIXES}
            chosen = {f.name for f in files}
            params = {"image_dir": str(folder)}
            if chosen != all_names:                      # a subset -> whitelist it
                params["select"] = sorted(chosen)
        else:                                            # multiple folders -> stage into the project
            staged = self.project.manifest.project_dir / f"{lid}_photos"
            staged.mkdir(parents=True, exist_ok=True)
            for f in files:
                dst = staged / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
            params = {"image_dir": str(staged)}
        self.project.add_stage(lid, "ingest", params=params, chunk=chunk)
        self.project.save()
        return {"ok": True, "id": lid, "count": len(files),
                "image_dir": params["image_dir"], "staged": len(dirs) > 1}

    def remove_photo(self, layer_id: str, name: str) -> dict:
        """Drop one image from an ingest layer by adding it to (the complement of) the select list.

        If the layer had no select (whole folder), we materialize the current folder listing minus
        this image, so the removal sticks without touching the source files."""
        from openreco.io.images import list_images

        stage = next((s for s in self.project.manifest.stages if s.id == layer_id), None)
        if stage is None or stage.type != "ingest":
            raise ValueError(f"no ingest layer {layer_id!r}")
        idir = stage.params.get("image_dir", "images")
        p = Path(idir) if Path(idir).is_absolute() else (self.project.manifest.project_dir / idir)
        current = set(stage.params.get("select") or [f.name for f in list_images(p)])
        current.discard(name)
        stage.params["select"] = sorted(current)
        self.project.save()
        return {"ok": True, "id": layer_id, "remaining": len(current)}

    def allowed_roots(self) -> list[Path]:
        """Dirs the viewer may read from: the project dir + each chunk's ingest image folder
        (source photos commonly live outside the project)."""
        roots = [self.project.manifest.project_dir.resolve()]
        proj_dir = self.project.manifest.project_dir
        for s in self.project.manifest.stages:
            if s.type == "ingest":
                idir = s.params.get("image_dir", "images")
                p = Path(idir) if Path(idir).is_absolute() else (proj_dir / idir)
                try:
                    roots.append(p.resolve())
                except OSError:
                    pass
        return roots

    def markers_path(self) -> Path:
        return self.project.manifest.project_dir / "markers.json"

    def load_markers(self) -> dict:
        p = self.markers_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"markers": []}

    def save_markers(self, markers: list[dict]) -> Path:
        """Persist markers.json + a georef-ready gcps.csv (name,X,Y,Z,image,u,v per observation)."""
        self.markers_path().write_text(json.dumps({"markers": markers}, indent=2), encoding="utf-8")
        rows = ["# name,X,Y,Z,image,u,v  (one row per image observation; written by the UI marker tool)"]
        for mk in markers:
            w = mk.get("world") or [0.0, 0.0, 0.0]
            for ob in mk.get("observations", []):
                rows.append(f"{mk['name']},{w[0]},{w[1]},{w[2]},{ob['image']},{ob['u']},{ob['v']}")
        csv = self.project.manifest.project_dir / "gcps.csv"
        csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return csv

    def _last_run(self) -> dict:
        latest = self.project.manifest.runs_dir / "latest.json"
        if not latest.exists():
            return {}
        data = json.loads(latest.read_text(encoding="utf-8"))
        return {s["id"]: s for s in data.get("stages", [])}

    # ---- run control ----
    def start_run(self, force=None) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
        self.events = queue.Queue()

        def worker():
            try:
                self.project.run(force=force, on_event=self.events.put)
            except Exception as exc:  # noqa: BLE001
                self.events.put({"event": "run_error", "error": repr(exc)})
            finally:
                self.events.put({"event": "_eof"})
                self.running = False

        threading.Thread(target=worker, daemon=True).start()
        return True


class _Handler(BaseHTTPRequestHandler):
    state: AppState = None  # set on the server class

    def log_message(self, *_a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        route = u.path
        if route in ("/", "/index.html"):
            return self._static("index.html")
        if route in ("/app.js", "/viewer.js", "/style.css"):
            return self._static(route.lstrip("/"))
        if route == "/api/stages":
            return self._send(200, stage_info())
        if route == "/api/workflows":
            from openreco.workflow import operations
            return self._send(200, operations())
        if route == "/api/project":
            return self._send(200, self.state.project_json())
        if route == "/api/events":
            return self._sse()
        if route == "/api/file":
            return self._file(parse_qs(u.query).get("path", [""])[0])
        if route == "/api/formats":
            return self._formats(parse_qs(u.query).get("path", [""])[0])
        if route == "/api/crs":
            return self._crs(parse_qs(u.query))
        if route == "/api/images":
            return self._send(200, self.state.images_for_chunk(
                parse_qs(u.query).get("chunk", [None])[0]))
        if route == "/api/markers":
            return self._send(200, self.state.load_markers())
        if route == "/api/raster_png":
            return self._raster_png(parse_qs(u.query).get("path", [""])[0])
        if route == "/api/browse":
            return self._send(200, self.state.browse(parse_qs(u.query).get("path", [None])[0]))
        if route == "/api/thumb":
            return self._thumb(parse_qs(u.query).get("path", [""])[0])
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if u.path == "/api/run":
            started = self.state.start_run(force=body.get("force"))
            return self._send(202 if started else 409, {"started": started})
        if u.path == "/api/project":
            return self._set_project(body)
        if u.path == "/api/stage":
            return self._add_stage(body)
        if u.path == "/api/markers":
            return self._set_markers(body)
        if u.path == "/api/use_gcps":
            return self._use_gcps(body)
        if u.path == "/api/add_photos":
            return self._add_photos(body)
        if u.path == "/api/remove_photo":
            return self._remove_photo(body)
        if u.path == "/api/operation":
            return self._operation(body)
        if u.path == "/api/chunk":
            return self._add_chunk(body)
        if u.path == "/api/export":
            return self._export(body)
        return self._send(404, {"error": "not found"})

    # ---- handlers ----
    def _static(self, name):
        p = WEB_DIR / name
        if not p.exists():
            return self._send(404, {"error": f"{name} missing"})
        self._send(200, p.read_bytes(), _CT.get(p.suffix, "application/octet-stream"))

    def _add_chunk(self, body):
        name = (body.get("name") or "").strip()
        if not name:
            return self._send(400, {"error": "chunk name required"})
        self.state.project.add_chunk(name)
        self.state.project.save()
        return self._send(200, {"ok": True})

    def _set_project(self, body):
        """Set project-level metadata from the UI (CRS picker)."""
        m = self.state.project.manifest
        if "crs" in body:
            m.crs = (body["crs"] or "").strip() or None
        self.state.project.save()
        return self._send(200, {"ok": True, "crs": m.crs})

    def _set_markers(self, body):
        markers = body.get("markers", [])
        if not isinstance(markers, list):
            return self._send(400, {"error": "markers must be a list"})
        try:
            csv = self.state.save_markers(markers)
            return self._send(200, {"ok": True, "count": len(markers), "gcp_csv": str(csv)})
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": repr(exc)})

    def _add_photos(self, body):
        try:
            res = self.state.add_photos(body.get("paths", []), body.get("chunk", "Chunk 1"),
                                        body.get("id"))
            return self._send(200, res)
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": repr(exc)})

    def _remove_photo(self, body):
        try:
            return self._send(200, self.state.remove_photo(body.get("layer", ""), body.get("name", "")))
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": repr(exc)})

    def _use_gcps(self, body):
        """Wire the picked GCPs (gcps.csv) into a chunk's georef stage(s): method=gcp + CRS."""
        chunk = body.get("chunk")
        m = self.state.project.manifest
        epsg = int(body.get("crs_epsg") or self._epsg_from_crs(m.crs) or 0)
        if not epsg:
            return self._send(400, {"error": "set a projected EPSG CRS first (crs_epsg or project CRS)"})
        if not (m.project_dir / "gcps.csv").exists():
            return self._send(400, {"error": "no gcps.csv yet — pick & save markers first"})
        updated = []
        for s in m.stages:
            if s.type == "georef" and (not chunk or s.chunk == chunk):
                s.params.update({"method": "gcp", "gcp_file": "gcps.csv", "gcp_crs_epsg": epsg})
                updated.append(s.id)
        if not updated:
            return self._send(400, {"error": f"no georef stage in chunk {chunk!r}; add one first"})
        self.state.project.save()
        return self._send(200, {"ok": True, "updated": updated, "gcp_crs_epsg": epsg})

    @staticmethod
    def _epsg_from_crs(crs):
        if crs and str(crs).upper().startswith("EPSG:"):
            try:
                return int(str(crs).split(":", 1)[1])
            except ValueError:
                return 0
        return 0

    def _add_stage(self, body):
        try:
            stages = self.state.project.manifest.stages
            ids = {s.id for s in stages}
            if body["id"] in ids:                       # update -> replace params/inputs/chunk
                from openreco.engine.manifest import StageSpec
                for i, s in enumerate(stages):
                    if s.id == body["id"]:
                        stages[i] = StageSpec(id=body["id"], type=body["type"],
                                              params=body.get("params", {}),
                                              inputs=body.get("inputs", []),
                                              chunk=body.get("chunk", s.chunk))
            else:
                self.state.project.add_stage(body["id"], body["type"], inputs=body.get("inputs"),
                                             params=body.get("params"),
                                             chunk=body.get("chunk", "Chunk 1"))
            self.state.project.save()
            return self._send(200, {"ok": True})
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": repr(exc)})

    def _operation(self, body):
        """Add/update a layer from a familiar workflow operation (Workflow menu)."""
        from openreco.workflow import to_stage
        try:
            spec = to_stage(body["op"], body.get("values"))
            return self._add_stage({"id": body["id"], "type": spec["stage_type"],
                                    "inputs": body.get("inputs", []), "params": spec["params"],
                                    "chunk": body.get("chunk", "Chunk 1")})
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": repr(exc)})

    def _file(self, path):
        if not path:
            return self._send(400, {"error": "path required"})
        p = Path(path).resolve()
        # sandbox: the project dir or any registered ingest image folder
        if not any(r == p or r in p.parents for r in self.state.allowed_roots()):
            return self._send(403, {"error": "outside project"})
        if not p.is_file():
            return self._send(404, {"error": "not found"})
        self._send(200, p.read_bytes(), _CT.get(p.suffix.lower(), "application/octet-stream"))

    def _thumb(self, path):
        """Serve an image file from anywhere (image suffixes only) for the Add-Photos picker."""
        from openreco.io.images import IMAGE_SUFFIXES
        p = Path(path)
        if not path or not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            return self._send(400, {"error": "image file required"})
        self._send(200, p.read_bytes(), _CT.get(p.suffix.lower(), "image/jpeg"))

    def _raster_png(self, path):
        from openreco.io.raster import raster_to_png
        p = Path(path)
        if not path or not self._in_project(p) or not p.is_file():
            return self._send(400, {"error": "valid in-project raster path required"})
        try:
            return self._send(200, raster_to_png(p), "image/png")
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": repr(exc)})

    def _in_project(self, p: Path) -> bool:
        root = self.state.project.manifest.project_dir.resolve()
        p = p.resolve()
        return p == root or root in p.parents

    def _formats(self, path):
        from openreco.exporters import list_formats
        if not path or not self._in_project(Path(path)):
            return self._send(400, {"error": "valid in-project path required"})
        try:
            return self._send(200, {"formats": list_formats(path)})
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": repr(exc)})

    def _export(self, body):
        from openreco.exporters import export_product
        src = Path(body.get("path", ""))
        fmt = body.get("fmt", "")
        if not self._in_project(src):
            return self._send(403, {"error": "outside project"})
        out_dir = self.state.project.manifest.project_dir / "exports"
        out = out_dir / f"{src.stem}.{fmt}"
        try:
            export_product(src, fmt, out)
            return self._send(200, {"out": str(out)})
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": repr(exc)})

    def _crs(self, q):
        from openreco.geo.crs import crs_info, search_crs
        try:
            if q.get("search"):
                return self._send(200, {"results": search_crs(q["search"][0],
                                        kind=q.get("kind", ["all"])[0])})
            if q.get("code"):
                return self._send(200, crs_info(q["code"][0]))
            return self._send(400, {"error": "pass ?code= or ?search="})
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": repr(exc)})

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                ev = self.state.events.get()
                if ev.get("event") == "_eof":
                    self.wfile.write(b"event: eof\ndata: {}\n\n")
                    self.wfile.flush()
                    break
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve(project: Project, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"state": AppState(project)})
    httpd = ThreadingHTTPServer((host, port), handler)
    return httpd
