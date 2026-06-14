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


def test_presets_apply_to_layers(tmp_path):
    proj = (Project.create(tmp_path, name="pre")
            .add_stage("a", "sfm")
            .add_stage("d", "mvs", inputs=["a"]))
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        names = {p["name"] for p in json.loads(_get(base + "/api/presets")[1])}
        assert {"Low", "Medium", "High", "Ultra"} <= names
        status, body = _post(base + "/api/preset", {"name": "Low"})
        assert status == 200 and body["preset"] == "Low" and body["updated"] == 2
        layers = {L["id"]: L for L in json.loads(_get(base + "/api/project")[1])["layers"]}
        assert layers["a"]["params"]["max_image_size"] == 1200      # sfm Low
        assert layers["d"]["params"]["quality"] == "low"            # mvs Low
        # new ops built afterwards inherit the active preset's base params
        _post(base + "/api/operation", {"op": "Build Texture", "id": "tx", "inputs": [], "values": {}})
        tx = next(L for L in json.loads(_get(base + "/api/project")[1])["layers"] if L["id"] == "tx")
        assert tx["params"]["target_faces"] == 80000               # texture Low base
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_layer_provides_and_op_needs(server):
    base, _ = server
    # operations declare what they need (for input auto-wiring)
    ops = {o["op"]: o for o in json.loads(_get(base + "/api/workflows")[1])}
    assert ops["Align Photos"]["needs"] == ["images"]
    assert "model" in ops["Build Dense Cloud"]["needs"] and "images" in ops["Build Dense Cloud"]["needs"]
    # layers report what they provide
    layers = {L["id"]: L for L in json.loads(_get(base + "/api/project")[1])["layers"]}
    assert "model" not in layers["gen"]["provides"]   # dummy stage -> no provides


def test_input_resolution_by_artifact(tmp_path):
    # a stage referencing inputs by artifact must work regardless of the upstream layer's id
    from openreco.engine.context import DeviceInfo, RunContext, StageResult
    import logging
    res = StageResult(artifacts={"images": "images.json"})
    ctx = RunContext(stage_id="x", stage_type="sfm", params={}, cache_dir=tmp_path,
                     inputs={"ingest_custom_id": res},
                     input_dirs={"ingest_custom_id": tmp_path}, project_dir=tmp_path,
                     device=DeviceInfo(), logger=logging.getLogger("t"))
    assert ctx.input_with("images") == "ingest_custom_id"
    assert ctx.find_input("images") == "ingest_custom_id"
    assert ctx.find_input("nonexistent") is None


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
    # modals/menus must start hidden — .hidden has to beat .modal's display:flex
    assert b"display:none !important" in html
    assert b'id="crsModal" class="modal hidden"' in html and b'id="modal" class="modal hidden"' in html


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
    assert b"/api/layer" in appjs and b"showCtx" in appjs       # context menu + layer ops
    assert b"set_enabled" in appjs and b"openLayer" in appjs and b"ondragstart" in appjs
    assert b"targets:" in appjs and b"Run up to here" in appjs       # per-stage run
    assert b"/api/new_project" in appjs and b"/api/save_project" in appjs
    assert b"/api/cameras" in appjs and b"buildCameras" in appjs
    assert b"/api/geo_overlay" in appjs and b"showOnMap" in appjs        # web map overlay
    assert b"/api/edit_cloud" in appjs and b"selectInPoly" in appjs       # 3D edit (box)
    assert b"/api/edit_mesh" in appjs and b"lassoBtn" in appjs and b"frontMostFilter" in appjs  # lasso/mesh/depth
    assert b"/api/cesium" in appjs                                        # Cesium 3D-Tiles viewer
    assert b"setupSplitters" in appjs and b"snapView" in appjs and b"gridline" in appjs  # infinite grid shader
    assert b"runPipeline" in appjs and b"camera.up.set(0, 0, 1)" in appjs   # Z-up world
    assert b"rotateGizmo" in appjs and b"contourView" in appjs               # navcube nav + contour overlay
    assert b"progShow" in appjs and b"/api/cancel" in appjs and b"event === 'log'" in appjs
    _, html = _get(base + "/")
    assert b'id="gizmo"' in html and b'class="split' in html and b"mAddOnly" in html
    assert b'id="progress"' in html and b'id="progBar"' in html


def test_cancel_when_idle(server):
    base, _ = server
    status, body = _post(base + "/api/cancel", {})
    assert status == 200 and body["cancelling"] is False   # nothing running


def test_chunk_rename_and_remove(tmp_path):
    proj = (Project.create(tmp_path, name="cx")
            .add_stage("a", "dummy_generate", chunk="Old")
            .add_stage("b", "dummy_generate", chunk="Keep"))
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        assert _post(base + "/api/chunk", {"action": "rename", "name": "Old", "to": "New"})[0] == 200
        proj_json = json.loads(_get(base + "/api/project")[1])
        assert "New" in proj_json["chunks"] and "Old" not in proj_json["chunks"]
        assert next(L for L in proj_json["layers"] if L["id"] == "a")["chunk"] == "New"
        # remove a chunk -> its layers go too
        assert _post(base + "/api/chunk", {"action": "remove", "name": "New"})[0] == 200
        proj_json = json.loads(_get(base + "/api/project")[1])
        assert "New" not in proj_json["chunks"]
        assert [L["id"] for L in proj_json["layers"]] == ["b"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_layer_rename_remove_move(tmp_path):
    proj = (Project.create(tmp_path, name="lx")
            .add_chunk("Other")
            .add_stage("gen", "dummy_generate")
            .add_stage("total", "dummy_sum", inputs=["gen"]))
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        # rename gen -> src, downstream input reference updates
        assert _post(base + "/api/layer", {"action": "rename", "id": "gen", "to": "src"})[0] == 200
        layers = {L["id"]: L for L in json.loads(_get(base + "/api/project")[1])["layers"]}
        assert "src" in layers and layers["total"]["inputs"] == ["src"]
        # move to another chunk
        assert _post(base + "/api/layer", {"action": "move", "id": "src", "to": "Other"})[0] == 200
        assert json.loads(_get(base + "/api/project")[1])
        # remove -> drops from downstream inputs
        assert _post(base + "/api/layer", {"action": "remove", "id": "src"})[0] == 200
        layers = {L["id"]: L for L in json.loads(_get(base + "/api/project")[1])["layers"]}
        assert "src" not in layers and layers["total"]["inputs"] == []
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_new_and_save_project(tmp_path):
    proj = Project.create(tmp_path / "first", name="first")
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        newdir = tmp_path / "second"
        status, body = _post(base + "/api/new_project", {"path": str(newdir), "name": "second"})
        assert status == 200 and body["name"] == "second"
        assert (newdir / "project.toml").exists()
        # the server now serves the new project
        assert json.loads(_get(base + "/api/project")[1])["name"] == "second"
        # explicit save returns the manifest path
        assert _post(base + "/api/save_project", {})[1]["path"].endswith("project.toml")
    finally:
        httpd.shutdown()
        httpd.server_close()


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


def test_edit_cloud_creates_import_layer(tmp_path):
    import numpy as np
    from openreco.io.pointcloud import write_ply
    ply = tmp_path / "cloud.ply"
    write_ply(ply, np.random.rand(500, 3).astype("float32"), np.full((500, 3), 200, np.uint8))
    proj = Project.create(tmp_path / "proj", name="ed").add_stage("pc", "import_cloud",
                                                                  params={"path": str(ply)})
    proj.run()                                  # materialize the source cloud
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        status, body = _post(base + "/api/edit_cloud",
                             {"layer": "pc", "removed": list(range(100)), "chunk": "Chunk 1"})
        assert status == 200 and body["ok"] and body["kept"] == 400 and body["removed"] == 100
        layers = {L["id"]: L for L in json.loads(_get(base + "/api/project")[1])["layers"]}
        assert body["id"] in layers and layers[body["id"]]["type"] == "import_cloud"
        # the edited cloud materializes to the kept count
        out = proj.run(targets=[body["id"]])
        assert next(s for s in out.stages if s.id == body["id"]).metrics["num_points"] == 400
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_edit_mesh_creates_import_mesh(tmp_path):
    import numpy as np
    from openreco.io.pointcloud import write_mesh_ply
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [5, 5, 5], [6, 5, 5], [5, 6, 5]], float)
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])         # 2 quad faces + 1 stray triangle
    mp = tmp_path / "m.ply"
    write_mesh_ply(mp, v, f)
    proj = Project.create(tmp_path / "proj", name="em").add_stage("ms", "import_mesh",
                                                                  params={"path": str(mp)})
    proj.run()
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        status, body = _post(base + "/api/edit_mesh", {"layer": "ms", "removed": [2], "chunk": "Chunk 1"})
        assert status == 200 and body["ok"] and body["kept_faces"] == 2 and body["removed_faces"] == 1
        out = proj.run(targets=[body["id"]])
        m = next(s for s in out.stages if s.id == body["id"]).metrics
        assert m["faces"] == 2 and m["vertices"] == 4      # stray face + its verts dropped
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_use_gcps_wires_georef_stage(tmp_path):
    proj = (Project.create(tmp_path, name="gcp", crs="EPSG:32613")
            .add_stage("align", "sfm")
            .add_stage("ref", "georef", inputs=["align"]))
    (tmp_path / "gcps.csv").write_text("GCP1,1,2,3,a.jpg,10,20\n", "utf-8")
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        status, body = _post(base + "/api/use_gcps", {"chunk": "Chunk 1"})
        assert status == 200 and body["updated"] == ["ref"] and body["gcp_crs_epsg"] == 32613
        layer = next(layer for layer in json.loads(_get(base + "/api/project")[1])["layers"]
                     if layer["id"] == "ref")
        assert layer["params"]["method"] == "gcp" and layer["params"]["gcp_file"] == "gcps.csv"
        assert layer["params"]["gcp_crs_epsg"] == 32613
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_use_gcps_needs_georef_stage(server):
    base, root = server
    (root / "gcps.csv").write_text("GCP1,1,2,3,a.jpg,10,20\n", "utf-8")
    _post(base + "/api/project", {"crs": "EPSG:32613"})
    try:
        _post(base + "/api/use_gcps", {"chunk": "Chunk 1"})
        raise AssertionError("expected 400")
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_export_dxf_vector(tmp_path):
    from openreco.exporters import export_product, list_formats
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": [[[0, 0], [1, 1], [2, 0]]]}}]}
    src = tmp_path / "contours.geojson"
    src.write_text(json.dumps(gj), "utf-8")
    assert "dxf" in list_formats(src)
    out = export_product(src, "dxf", tmp_path / "c.dxf")
    assert "LWPOLYLINE" in out.read_text()


def test_cesium_and_tiles3d(tmp_path):
    tdir = tmp_path / "tiles"
    tdir.mkdir()
    (tdir / "tileset.json").write_text('{"asset":{"version":"1.1"}}', "utf-8")
    (tdir / "tile_0_0.glb").write_bytes(b"glTF\x02\x00\x00\x00")
    proj = Project.create(tmp_path / "proj", name="t3d").add_stage("tm", "tiles")
    proj.manifest.runs_dir.mkdir(parents=True, exist_ok=True)
    (proj.manifest.runs_dir / "latest.json").write_text(json.dumps(
        {"stages": [{"id": "tm", "status": "executed", "artifacts": {"tiles": str(tdir)}}]}), "utf-8")
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        _, html = _get(base + "/api/cesium?layer=tm")
        assert b"Cesium" in html and b"/tiles3d/tm/tileset.json" in html
        status, body = _get(base + "/tiles3d/tm/tileset.json")
        assert status == 200 and b'"asset"' in body
        # sandbox: escaping the tiles dir is refused
        try:
            _get(base + "/tiles3d/tm/" + urllib.request.quote("../../secret"))
            raise AssertionError("expected non-200")
        except urllib.error.HTTPError as e:
            assert e.code in (403, 404)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_report_pdf_builds():
    from openreco.engine.report_pdf import write_report_pdf
    assert write_report_pdf(None)[:5] == b"%PDF-"            # placeholder
    data = {"project": "demo", "openreco_version": "0.1", "started": "2026", "ok": True,
            "platform": {"python": "3.13", "system": "Windows", "machine": "AMD64"},
            "stages": [{"id": "align", "type": "sfm", "key": "abc", "status": "executed", "seconds": 5.0,
                        "metrics": {"reg_images": 36, "input_images": 36, "points3D": 5200},
                        "issues": [{"severity": "info", "message": "ok"}], "params": {"matcher": "exhaustive"},
                        "artifacts": {}}]}
    assert write_report_pdf(data)[:5] == b"%PDF-"


def test_report_endpoint_serves_pdf(server):
    base, _ = server
    status, body = _get(base + "/api/report")
    assert status == 200 and body[:5] == b"%PDF-"            # PDF even before any run (placeholder)


def test_geo_overlay_endpoint(server, tmp_path):
    base, root = server
    import numpy as np
    from openreco.io.raster import write_geotiff
    ortho = root / "ortho.tif"
    write_geotiff(ortho, (np.random.rand(40, 50, 3) * 255).astype("uint8"),
                  691000.0, 5334000.0, 0.5, 32632)          # UTM 32N near Munich
    _, body = _get(base + "/api/geo_overlay?path=" + urllib.request.quote(str(ortho)))
    data = json.loads(body)
    assert data["ok"] and data["image"].startswith("data:image/png;base64,")
    (s, w), (n, e) = data["bounds"]
    assert 47 < s < 49 and 11 < w < 12 and n > s and e > w   # reprojected to lat/lon


def test_raster_png_endpoint(server, tmp_path):
    base, root = server
    import numpy as np
    from openreco.io.raster import write_geotiff
    dsm = root / "dsm.tif"
    write_geotiff(dsm, np.linspace(0, 100, 64*64).reshape(64, 64).astype("float32"),
                  500000.0, 4000000.0, 1.0, 32613, nodata=float("nan"))
    status, png = _get(base + "/api/raster_png?path=" + urllib.request.quote(str(dsm)))
    assert status == 200 and png[:8] == b"\x89PNG\r\n\x1a\n"


def test_images_scan_before_run_and_serve_external(tmp_path):
    # source photos live OUTSIDE the project dir (common for drone sets)
    ext = tmp_path / "flight"
    ext.mkdir()
    (ext / "DJI_1.JPG").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    (ext / "DJI_2.JPG").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    proj = Project.create(tmp_path / "proj", name="scan").add_stage(
        "photos", "ingest", params={"image_dir": str(ext)})
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        data = json.loads(_get(base + "/api/images?chunk=Chunk%201")[1])
        names = sorted(im["name"] for im in data["images"])
        assert names == ["DJI_1.JPG", "DJI_2.JPG"]          # listed before any run
        # the external photo is serveable (sandbox widened to ingest folders)
        status, body = _get(base + "/api/file?path=" + urllib.request.quote(data["images"][0]["path"]))
        assert status == 200 and body.startswith(b"\xff\xd8")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_browse_lists_dirs_and_images(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.JPG").write_bytes(b"\xff\xd8x")
    (tmp_path / "notes.txt").write_text("x")
    proj = Project.create(tmp_path / "proj", name="b")
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        d = json.loads(_get(base + "/api/browse?path=" + urllib.request.quote(str(tmp_path)))[1])
        assert str(tmp_path / "sub") in d["dirs"]
        assert [i["name"] for i in d["images"]] == ["a.JPG"]   # .txt excluded
        assert d["parent"] == str(tmp_path.parent)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_add_photos_subset_uses_select(tmp_path):
    folder = tmp_path / "flight"
    folder.mkdir()
    for n in ("a.JPG", "b.JPG", "c.JPG"):
        (folder / n).write_bytes(b"\xff\xd8x")
    proj = Project.create(tmp_path / "proj", name="ap")
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        # pick 2 of 3 -> a select whitelist, image_dir = the folder
        body = _post(base + "/api/add_photos",
                     {"paths": [str(folder / "a.JPG"), str(folder / "b.JPG")], "chunk": "Chunk 1"})[1]
        assert body["ok"] and body["count"] == 2 and not body["staged"]
        layer = next(layer for layer in json.loads(_get(base + "/api/project")[1])["layers"]
                     if layer["id"] == body["id"])
        assert layer["type"] == "ingest" and layer["params"]["image_dir"] == str(folder)
        assert sorted(layer["params"]["select"]) == ["a.JPG", "b.JPG"]
        # Photos pane reflects the subset before any run
        imgs = json.loads(_get(base + "/api/images?chunk=Chunk%201")[1])["images"]
        assert sorted(i["name"] for i in imgs) == ["a.JPG", "b.JPG"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_remove_photo_updates_select(tmp_path):
    folder = tmp_path / "flight"
    folder.mkdir()
    for n in ("a.JPG", "b.JPG", "c.JPG"):
        (folder / n).write_bytes(b"\xff\xd8x")
    proj = Project.create(tmp_path / "proj", name="rm").add_stage(
        "photos", "ingest", params={"image_dir": str(folder)})    # no select = whole folder
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        status, body = _post(base + "/api/remove_photo", {"layer": "photos", "name": "b.JPG"})
        assert status == 200 and body["remaining"] == 2
        params = json.loads(_get(base + "/api/project")[1])["layers"][0]["params"]
        assert sorted(params["select"]) == ["a.JPG", "c.JPG"]       # materialized minus b
        imgs = json.loads(_get(base + "/api/images?chunk=Chunk%201")[1])["images"]
        assert sorted(i["name"] for i in imgs) == ["a.JPG", "c.JPG"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_add_photos_multi_folder_stages_copies(tmp_path):
    f1 = tmp_path / "d1"
    f2 = tmp_path / "d2"
    f1.mkdir()
    f2.mkdir()
    (f1 / "a.JPG").write_bytes(b"\xff\xd8x")
    (f2 / "b.JPG").write_bytes(b"\xff\xd8x")
    proj = Project.create(tmp_path / "proj", name="multi")
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        body = _post(base + "/api/add_photos",
                     {"paths": [str(f1 / "a.JPG"), str(f2 / "b.JPG")], "chunk": "Chunk 1"})[1]
        assert body["ok"] and body["staged"] and body["count"] == 2
        from pathlib import Path
        staged = Path(body["image_dir"])
        assert (staged / "a.JPG").exists() and (staged / "b.JPG").exists()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_marker_template_and_autodetect(tmp_path):
    from openreco.markers import marker_sheet_png
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    (imgs / "sheet.png").write_bytes(marker_sheet_png("4x4_50", count=6))   # a photo of targets
    proj = Project.create(tmp_path / "proj", name="mk").add_stage(
        "photos", "ingest", params={"image_dir": str(imgs)})
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        status, png = _get(base + "/api/marker_template?dictionary=4x4_50&count=6")
        assert status == 200 and png[:4] == b"\x89PNG"
        status, body = _post(base + "/api/detect_markers", {"chunk": "Chunk 1", "dictionary": "4x4_50"})
        assert status == 200 and body["ok"] and len(body["markers"]) == 6
        m0 = body["markers"][0]
        assert m0["name"].startswith("marker_") and m0["type"] == "control"
        assert m0["observations"][0]["image"] == "sheet.png" and "u" in m0["observations"][0]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cameras_gps_fallback(tmp_path):
    proj = Project.create(tmp_path, name="cam").add_stage("ing", "ingest",
                                                          params={"image_dir": "images"})
    imgs = tmp_path / "images.json"
    imgs.write_text(json.dumps({"image_dir": str(tmp_path / "images"), "images": [
        {"name": "a.JPG", "lat": 40.000, "lon": -105.000, "alt": 1500.0, "excluded": False},
        {"name": "b.JPG", "lat": 40.001, "lon": -105.001, "alt": 1510.0, "excluded": False}]}), "utf-8")
    proj.manifest.runs_dir.mkdir(parents=True, exist_ok=True)
    (proj.manifest.runs_dir / "latest.json").write_text(json.dumps(
        {"stages": [{"id": "ing", "status": "executed", "artifacts": {"images": str(imgs)}}]}), "utf-8")
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        data = json.loads(_get(base + "/api/cameras?chunk=Chunk%201")[1])
        assert data["frame"] == "gps" and len(data["cameras"]) == 2
        # ENU metres, centred on the set: ~111 m north, ~85 m west between the two
        cs = {c["name"]: c["c"] for c in data["cameras"]}
        assert abs(cs["b.JPG"][1] - cs["a.JPG"][1] - 110.54) < 1.0     # 0.001 deg lat ~ 111 m
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cameras_from_solved_poses(tmp_path):
    proj = Project.create(tmp_path, name="cam2").add_stage("align", "sfm")
    poses = tmp_path / "poses.json"
    poses.write_text(json.dumps({"images": [
        {"name": "a.JPG", "center": [1.0, 2.0, 3.0]},
        {"name": "b.JPG", "center": [4.0, 5.0, 6.0]}]}), "utf-8")
    proj.manifest.runs_dir.mkdir(parents=True, exist_ok=True)
    (proj.manifest.runs_dir / "latest.json").write_text(json.dumps(
        {"stages": [{"id": "align", "status": "executed",
                     "artifacts": {"poses": str(poses)}}]}), "utf-8")
    httpd = serve(proj, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        data = json.loads(_get(base + "/api/cameras?chunk=Chunk%201")[1])
        assert data["frame"] == "model" and data["source"] == "align"
        assert data["cameras"][0]["c"] == [1.0, 2.0, 3.0]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_frontend_has_crs_and_marker_ui(server):
    base, _ = server
    _, appjs = _get(base + "/app.js")
    assert b"/api/markers" in appjs and b"openCrsPicker" in appjs
    assert b"/api/use_gcps" in appjs and b"/api/raster_png" in appjs
    assert b"/api/add_photos" in appjs and b"openBrowse" in appjs and b"/api/browse" in appjs
    assert b"/api/remove_photo" in appjs and b"brAlign" in appjs
    assert b"showGcpAccuracy" in appjs and b"control_rms" in appjs       # GCP control/check accuracy
    assert b"/api/detect_markers" in appjs and b"/api/marker_template" in appjs  # auto markers
    assert b"const ic =" in appjs                                        # line-icon helper
    assert b"loadPresets" in appjs and b"/api/preset" in appjs           # quality presets
    _, html2 = _get(base + "/")
    assert b"backdrop-filter" in html2 and b'id="i-play"' in html2       # glass theme + icon sprite
    assert b"#300a24" in html2                                           # Ubuntu-style console


def test_desktop_mode_resolution(monkeypatch):
    from openreco.ui import desktop
    monkeypatch.setattr(desktop, "_have_webview", lambda: True)
    assert desktop.resolve_mode("auto") == "window"
    assert desktop.resolve_mode("browser") == "browser"
    monkeypatch.setattr(desktop, "_have_webview", lambda: False)
    assert desktop.resolve_mode("auto") == "browser"
    assert desktop.resolve_mode("window") == "browser"   # downgrades when pywebview absent
