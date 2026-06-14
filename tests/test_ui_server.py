"""UI server endpoints — headless (no browser): static, stages, project, run+SSE, file sandbox."""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from openreco.api import Project
from openreco.ui.server import serve


@pytest.fixture()
def server(tmp_path):
    proj = (Project.create(tmp_path, name="ui-test")
            .add_stage("gen", "dummy_generate", params={"n": 4})
            .add_stage("total", "dummy_sum", inputs=["gen"]))
    httpd = serve(proj, port=0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", tmp_path
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read())


def test_static_and_api_stages(server):
    base, _ = server
    _, html = _get(base + "/")
    assert b"OpenReco" in html
    _, appjs = _get(base + "/app.js")
    assert b"OrbitControls" in appjs
    _, raw = _get(base + "/api/stages")
    types = {s["type"] for s in json.loads(raw)}
    assert {"ingest", "sfm", "classify", "dummy_generate"} <= types


def test_project_tree(server):
    base, _ = server
    _, raw = _get(base + "/api/project")
    proj = json.loads(raw)
    assert proj["name"] == "ui-test"
    assert [layer["id"] for layer in proj["layers"]] == ["gen", "total"]


def test_run_streams_events_and_updates_status(server):
    base, _ = server
    status, body = _post(base + "/api/run", {})
    assert status == 202 and body["started"]
    # read the SSE stream until eof
    events = []
    with urllib.request.urlopen(base + "/api/events", timeout=15) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
            if line.startswith("event: eof"):
                break
    kinds = [e["event"] for e in events]
    assert "stage_done" in kinds and any(e.get("event") == "run_done" for e in events)
    # project now reports completed layers
    _, raw = _get(base + "/api/project")
    statuses = {layer["id"]: layer["status"] for layer in json.loads(raw)["layers"]}
    assert statuses["total"] in ("executed", "cached")


def test_file_sandbox_rejects_outside_project(server):
    base, _ = server
    req = base + "/api/file?path=" + urllib.request.quote("C:/Windows/system32/drivers/etc/hosts")
    try:
        _get(req)
        raise AssertionError("expected non-200")
    except urllib.error.HTTPError as e:
        assert e.code in (403, 404)


def test_export_endpoint(server, tmp_path):
    base, root = server
    # make a small in-project mesh to export
    import numpy as np
    from openreco.io.pointcloud import write_mesh_ply
    mesh = root / "m.ply"
    write_mesh_ply(mesh, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float),
                   np.array([[0, 1, 2]]), np.full((3, 3), 100, np.uint8))
    _, fmts = _get(base + "/api/formats?path=" + urllib.request.quote(str(mesh)))
    assert "stl" in json.loads(fmts)["formats"]
    status, body = _post(base + "/api/export", {"path": str(mesh), "fmt": "stl"})
    assert status == 200 and body["out"].endswith(".stl")
    from pathlib import Path
    assert Path(body["out"]).exists()


def test_workflows_and_operation(server):
    base, _ = server
    _, raw = _get(base + "/api/workflows")
    ops = {o["op"] for o in json.loads(raw)}
    assert {"Align Photos", "Build Dense Cloud", "Build Model"} <= ops
    # build a layer via a familiar operation -> creates an mvs stage with translated params
    status, body = _post(base + "/api/operation",
                         {"op": "Build Dense Cloud", "id": "dense1", "inputs": [],
                          "values": {"Backend": "Portable (any GPU/CPU)"}})
    assert status == 200 and body["ok"]
    _, praw = _get(base + "/api/project")
    layer = next(layer for layer in json.loads(praw)["layers"] if layer["id"] == "dense1")
    assert layer["type"] == "mvs" and layer["params"]["dense_backend"] == "planesweep"


def test_frontend_has_workflow_ui(server):
    base, _ = server
    _, appjs = _get(base + "/app.js")
    assert b"loadWorkflows" in appjs and b"/api/operation" in appjs
    _, html = _get(base + "/")
    assert b"Workflow" in html and b"modal" in html


def test_chunks_workspace(server):
    base, _ = server
    # project exposes chunks + per-layer chunk
    _, raw = _get(base + "/api/project")
    proj = json.loads(raw)
    assert "Chunk 1" in proj["chunks"]
    assert all("chunk" in layer for layer in proj["layers"])
    # create a new chunk and build a layer into it
    assert _post(base + "/api/chunk", {"name": "Site B"})[0] == 200
    _post(base + "/api/operation", {"op": "Align Photos", "id": "alignB", "inputs": [],
                                    "values": {}, "chunk": "Site B"})
    proj = json.loads(_get(base + "/api/project")[1])
    assert "Site B" in proj["chunks"]
    assert next(layer for layer in proj["layers"] if layer["id"] == "alignB")["chunk"] == "Site B"


def test_frontend_has_workspace_chunks(server):
    base, _ = server
    _, appjs = _get(base + "/app.js")
    assert b"renderWorkspace" in appjs and b"ACTIVE_CHUNK" in appjs and b"/api/chunk" in appjs


def test_set_project_crs(server):
    base, _ = server
    status, body = _post(base + "/api/project", {"crs": "EPSG:32613"})
    assert status == 200 and body["crs"] == "EPSG:32613"
    proj = json.loads(_get(base + "/api/project")[1])
    assert proj["crs"] == "EPSG:32613"


def test_markers_roundtrip_writes_gcp_csv(server):
    base, root = server
    markers = [{"name": "GCP1", "world": [500000.0, 4000000.0, 1500.0],
                "observations": [{"image": "a.jpg", "u": 100.0, "v": 200.0},
                                 {"image": "b.jpg", "u": 110.0, "v": 210.0}]}]
    status, body = _post(base + "/api/markers", {"markers": markers})
    assert status == 200 and body["count"] == 1
    assert json.loads(_get(base + "/api/markers")[1])["markers"][0]["name"] == "GCP1"
    from pathlib import Path
    csv = Path(body["gcp_csv"]).read_text()
    assert "GCP1,500000.0,4000000.0,1500.0,a.jpg,100.0,200.0" in csv
    assert csv.count("GCP1,") == 2          # one row per observation


def test_images_endpoint_lists_chunk_photos(tmp_path):
    proj = Project.create(tmp_path, name="img-test").add_stage(
        "ing", "ingest", params={"image_dir": "images"})
    # fake ingest output + a last-run record pointing at it (no real run needed)
    imgs = tmp_path / "images.json"
    imgs.write_text(json.dumps({"image_dir": str(tmp_path / "images"),
                                "images": [{"name": "DJI_1.JPG", "lat": 40.1, "lon": -105.2,
                                            "excluded": False}]}), "utf-8")
    proj.manifest.runs_dir.mkdir(parents=True, exist_ok=True)
    (proj.manifest.runs_dir / "latest.json").write_text(json.dumps(
        {"stages": [{"id": "ing", "status": "executed", "artifacts": {"images": str(imgs)}}]}), "utf-8")
    httpd = serve(proj, port=0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        data = json.loads(_get(base + "/api/images?chunk=Chunk 1".replace(" ", "%20"))[1])
        assert [im["name"] for im in data["images"]] == ["DJI_1.JPG"]
        assert data["images"][0]["path"].endswith("DJI_1.JPG")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_frontend_has_crs_and_marker_ui(server):
    base, _ = server
    _, appjs = _get(base + "/app.js")
    assert b"/api/markers" in appjs and b"openCrsPicker" in appjs


def test_desktop_mode_resolution(monkeypatch):
    from openreco.ui import desktop
    monkeypatch.setattr(desktop, "_have_webview", lambda: True)
    assert desktop.resolve_mode("auto") == "window"
    assert desktop.resolve_mode("browser") == "browser"
    monkeypatch.setattr(desktop, "_have_webview", lambda: False)
    assert desktop.resolve_mode("auto") == "browser"
    assert desktop.resolve_mode("window") == "browser"   # downgrades when pywebview absent
