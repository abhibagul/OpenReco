"""Local UI server (stdlib http.server) exposing the engine to the web frontend.

Endpoints (JSON unless noted):
  GET  /                      -> the single-page app (web/index.html)
  GET  /app.js, /viewer.js    -> static frontend assets
  GET  /api/stages            -> stage_info() (palette + parameter-panel schemas)
  GET  /api/project           -> {name, crs, stages[], layers[]} (layer tree + last-run status/artifacts)
  POST /api/stage             -> add/update a stage {id,type,inputs,params}; re-saves project.toml
  POST /api/run               -> start a run in a background thread (force? in body); 202
  GET  /api/events            -> Server-Sent Events: live run events (stage_start/progress/.../run_done)
  GET  /api/file?path=...      -> serve an artifact file (sandboxed to the project dir) for the viewer

Zero third-party deps; ThreadingHTTPServer + a per-run event queue for SSE.
"""

from __future__ import annotations

import json
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
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if u.path == "/api/run":
            started = self.state.start_run(force=body.get("force"))
            return self._send(202 if started else 409, {"started": started})
        if u.path == "/api/stage":
            return self._add_stage(body)
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
        root = self.state.project.manifest.project_dir.resolve()
        if root not in p.parents and p != root:         # sandbox to the project dir
            return self._send(403, {"error": "outside project"})
        if not p.is_file():
            return self._send(404, {"error": "not found"})
        self._send(200, p.read_bytes(), _CT.get(p.suffix.lower(), "application/octet-stream"))

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
