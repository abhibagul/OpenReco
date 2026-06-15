"""Local UI server (stdlib http.server) exposing the engine to the web frontend.

Endpoints (JSON unless noted):
  GET  /                      -> the single-page app (web/index.html)
  GET  /app.js, /viewer.js    -> static frontend assets
  GET  /api/stages            -> stage_info() (palette + parameter-panel schemas)
  GET  /api/project           -> {name, crs, stages[], layers[]} (layer tree + last-run status/artifacts)
  POST /api/new_project        -> create/open a project at {path,name,crs} and load it
  POST /api/save_project       -> write project.toml now (explicit Save)
  POST /api/project           -> set project metadata {crs}; re-saves project.toml (CRS picker)
  POST /api/stage             -> add/update a stage {id,type,inputs,params}; re-saves project.toml
  POST /api/run               -> start a run in a background thread (force? in body); 202
  GET  /api/events            -> Server-Sent Events: live run events (stage_start/progress/.../run_done)
  GET  /api/file?path=...      -> serve an artifact file (sandboxed to the project dir) for the viewer
  GET  /api/images?chunk=...   -> source images of a chunk (for the Photos pane + GCP picking)
  GET  /api/browse?path=...    -> list sub-folders + image files of a dir (Add-Photos file picker)
  GET  /api/thumb?path=...     -> serve an image file from anywhere (picker previews; image-only)
  GET  /api/cameras?chunk=...  -> camera positions (solved poses, else EXIF-GPS) for the 3D view
  POST /api/add_photos         -> create an ingest layer from chosen image paths {paths,chunk,id}
  POST /api/remove_photo       -> drop one image from an ingest layer {layer,name} (select list)
  POST /api/chunk              -> chunk ops {action: add|rename|remove, name, to}
  POST /api/layer              -> layer ops {action: remove|rename|move, id, to}
  GET  /api/markers            -> saved GCP/markers (markers.json)
  POST /api/markers            -> save GCP/markers; also writes gcps.csv consumable by the georef stage
  POST /api/use_gcps           -> point a chunk's georef stage(s) at gcps.csv (method=gcp + CRS)
  GET  /api/raster_png?path=... -> render a GeoTIFF (ortho/DSM/index) to PNG for the 2D Ortho view

Zero third-party deps; ThreadingHTTPServer + a per-run event queue for SSE.
"""

from __future__ import annotations

import json
import logging
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

_CESIUM = "https://cesium.com/downloads/cesiumjs/releases/1.119/Build/Cesium"
_CESIUM_HTML = ("""<!doctype html><html><head><meta charset="utf-8"><title>OpenReco — Cesium</title>
<script src="%s/Cesium.js"></script>
<link href="%s/Widgets/widgets.css" rel="stylesheet">
<style>html,body,#c{margin:0;height:100%%;width:100%%;background:#0b0d12}
#err{color:#fff;font:15px system-ui;padding:2rem}</style></head>
<body><div id="c"></div><script>
Cesium.Ion.defaultAccessToken = '';
const viewer = new Cesium.Viewer('c', { baseLayer:false, baseLayerPicker:false, geocoder:false,
  timeline:false, animation:false, homeButton:false, sceneModePicker:false,
  navigationHelpButton:false, infoBox:false, selectionIndicator:false, fullscreenButton:false });
(async () => {
  try {
    const osm = await Cesium.ImageryLayer.fromProviderAsync(
      Cesium.OpenStreetMapImageryProvider.fromUrl('https://tile.openstreetmap.org/'));
    viewer.imageryLayers.add(osm);
  } catch (e) { /* basemap optional */ }
  try {
    const ts = await Cesium.Cesium3DTileset.fromUrl('/tiles3d/__LAYER__/tileset.json');
    viewer.scene.primitives.add(ts); await viewer.zoomTo(ts);
  } catch (e) {
    document.body.innerHTML = '<div id="err">Could not load 3D Tiles: ' + e +
      '<br>The Tiled Model must be georeferenced (add a Georeference step, rebuild tiles).</div>';
  }
})();
</script></body></html>""" % (_CESIUM, _CESIUM))


def _cameras_from_gps(imgs: list[dict]) -> list[dict]:
    """Project EXIF GPS to a local ENU frame (metres) centred on the set — a pre-alignment preview."""
    import math
    lat0 = sum(i["lat"] for i in imgs) / len(imgs)
    lon0 = sum(i["lon"] for i in imgs) / len(imgs)
    alts = [i.get("alt") or 0.0 for i in imgs]
    alt0 = sum(alts) / len(alts)
    k = math.cos(math.radians(lat0))
    out = []
    for i in imgs:
        x = (i["lon"] - lon0) * k * 111320.0
        y = (i["lat"] - lat0) * 110540.0
        out.append({"name": i["name"], "c": [x, y, (i.get("alt") or 0.0) - alt0]})
    return out


class _QueueLogHandler(logging.Handler):
    """Forward 'openreco' log records to the SSE queue so the UI Console shows detailed run output."""

    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        try:
            self.q.put({"event": "log", "level": record.levelname, "msg": self.format(record)})
        except Exception:  # noqa: BLE001
            pass


class _StderrCapture:
    """Tee the OS-level stderr (fd 2) into the SSE queue during a run.

    COLMAP/glog (and other native libs) write straight to file descriptor 2, bypassing Python's
    logging, so the only way to surface their per-image output in the UI Console is to redirect the
    fd through a pipe. Lines are still echoed to the real terminal. Best-effort: if the platform
    refuses the redirect, the run proceeds without native-log capture."""

    def __init__(self, q: queue.Queue):
        self.q = q
        self.ok = False

    def __enter__(self):
        try:
            self._r, self._w = os.pipe()
            self._saved = os.dup(2)
            os.dup2(self._w, 2)
            os.close(self._w)
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()
            self.ok = True
        except Exception:  # noqa: BLE001 — never let log capture break a run
            self.ok = False
        return self

    def _pump(self):
        buf = b""
        while True:
            try:
                chunk = os.read(self._r, 4096)
            except OSError:
                break
            if not chunk:
                break
            try:
                os.write(self._saved, chunk)        # keep echoing to the real terminal
            except OSError:
                pass
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", "replace").rstrip("\r")
                if text:
                    self.q.put({"event": "log", "level": "INFO", "msg": text})

    def __exit__(self, *_exc):
        if not self.ok:
            return
        try:
            os.dup2(self._saved, 2)                  # restore -> pipe write side fully closed -> EOF
            os.close(self._saved)
            os.close(self._r)
        except OSError:
            pass


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
        self.cancel_requested = False
        self.preset = None          # active quality/speed preset (applied to new layers too)
        self.lock = threading.Lock()

    def apply_preset(self, name: str) -> dict:
        from openreco.workflow import preset_params
        pm = preset_params(name)
        n = sum(1 for s in self.project.manifest.stages if s.type in pm)
        self.preset = name
        self.project.apply_preset(pm)
        self.project.save()
        return {"ok": True, "preset": name, "updated": n}

    # ---- data for the frontend ----
    def new_project(self, path: str, name: str | None, crs: str | None) -> dict:
        """Create (or open) a project at `path` and make it the loaded project."""
        from openreco.api import Project
        p = Path(path).expanduser()
        if (p / "project.toml").exists():
            self.project = Project.open(p)
        else:
            self.project = Project.create(p, name=name or p.name, crs=crs or None)
            self.project.save()
        self.running = False
        self.events = queue.Queue()
        return {"ok": True, "project_dir": str(self.project.manifest.project_dir),
                "name": self.project.manifest.name}

    def save_project(self) -> dict:
        return {"ok": True, "path": str(self.project.save())}

    def project_json(self) -> dict:
        from openreco.workflow import provides
        m = self.project.manifest
        keys = compute_keys(m)
        last = self._last_run()
        layers = []
        for s in m.stages:
            run = last.get(s.id, {})
            layers.append({
                "id": s.id, "type": s.type, "inputs": s.inputs, "params": s.params, "chunk": s.chunk,
                "enabled": s.enabled, "provides": provides(s.type),
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

    def cameras_for_chunk(self, chunk: str | None) -> dict:
        """Camera positions for the 3D view (the reference tool 'show cameras'). Prefers solved poses from
        an sfm/georef reconstruction (centers + orientation); falls back to EXIF GPS as a local
        ENU preview before alignment."""
        last = self._last_run()
        # use the model the dense/mesh are actually in, so cameras align with the displayed cloud:
        # prefer mvs (exact dense frame) > georef > sfm
        order = {"mvs": 0, "georef": 1, "sfm": 2}
        cands = []
        for s in self.project.manifest.stages:
            if chunk and s.chunk != chunk:
                continue
            model = last.get(s.id, {}).get("artifacts", {}).get("model")
            if s.type in order and model and Path(model).is_dir():
                cands.append((order[s.type], s.id, model))
        cands.sort()
        if cands:
            cams = self._cameras_from_model(cands[0][2])
            if cams:
                return {"frame": "model", "source": cands[0][1], "cameras": cams}
        # fallback: poses.json centers (no orientation) if the model couldn't be loaded
        for s in self.project.manifest.stages:
            if chunk and s.chunk != chunk:
                continue
            poses = last.get(s.id, {}).get("artifacts", {}).get("poses")
            if poses and Path(poses).is_file():
                data = json.loads(Path(poses).read_text(encoding="utf-8"))
                cams = [{"name": im["name"], "c": im["center"]} for im in data.get("images", [])]
                if cams:
                    return {"frame": "model", "source": s.id, "cameras": cams}
        gps = [im for im in self.images_for_chunk(chunk)["images"] if im.get("lat") is not None]
        if gps:
            return {"frame": "gps", "cameras": _cameras_from_gps(gps)}
        return {"frame": "none", "cameras": []}

    @staticmethod
    def _cameras_from_model(model_dir) -> list[dict]:
        """Camera centers + orientation (forward/up) from a COLMAP reconstruction."""
        try:
            import numpy as np
            import pycolmap
            rec = pycolmap.Reconstruction(str(model_dir))
            out = []
            for i in rec.reg_image_ids():
                img = rec.image(i)
                c = np.asarray(img.projection_center())
                r = np.asarray(img.cam_from_world().matrix())[:, :3]   # world->cam rotation
                out.append({"name": img.name, "c": c.tolist(),
                            "fwd": (r.T @ np.array([0, 0, 1.0])).tolist(),
                            "up": (r.T @ np.array([0, -1.0, 0])).tolist()})
            return out
        except Exception:  # noqa: BLE001
            return []

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

    def edit_cloud(self, layer_id: str, removed: list[int], chunk: str) -> dict:
        """Write a copy of a point-cloud layer with `removed` point indices deleted, and add it as an
        import_cloud layer (non-destructive: the source layer is untouched)."""
        import numpy as np

        from openreco.io.pointcloud import read_ply, write_ply
        last = self._last_run()
        art = last.get(layer_id, {}).get("artifacts", {})
        ply = art.get("points") or art.get("merged")
        if not ply or not Path(ply).is_file():
            raise ValueError(f"layer {layer_id!r} has no point cloud (run it first)")
        xyz, rgb, normals = read_ply(Path(ply))
        mask = np.ones(len(xyz), bool)
        rem = np.asarray([i for i in removed if 0 <= i < len(xyz)], dtype=np.int64)
        mask[rem] = False
        xyz, rgb = xyz[mask], (rgb[mask] if rgb is not None else None)
        normals = normals[mask] if normals is not None else None

        meta = {}
        mp = art.get("meta")
        if mp and Path(mp).is_file():
            meta = json.loads(Path(mp).read_text(encoding="utf-8"))
        edits = self.project.manifest.project_dir / "edits"
        edits.mkdir(parents=True, exist_ok=True)
        ids = {s.id for s in self.project.manifest.stages}
        new_id = _unique_id(f"{layer_id}_edit", ids)
        out = edits / f"{new_id}.ply"
        write_ply(out, xyz, rgb, normals)
        self.project.add_stage(new_id, "import_cloud", chunk=chunk, params={
            "path": str(out), "crs_epsg": int(meta.get("crs_epsg") or 0),
            "origin": meta.get("origin", [0.0, 0.0, 0.0])})
        self.project.save()
        return {"ok": True, "id": new_id, "kept": int(mask.sum()), "removed": int((~mask).sum())}

    def edit_mesh(self, layer_id: str, removed: list[int], chunk: str) -> dict:
        """Write a copy of a mesh layer with `removed` face indices deleted (unused verts dropped),
        added as an import_mesh layer. Non-destructive."""
        import numpy as np

        from openreco.io.pointcloud import read_mesh_ply, write_mesh_ply
        art = self._last_run().get(layer_id, {}).get("artifacts", {})
        mesh = art.get("mesh")
        if not mesh or not Path(mesh).is_file():
            raise ValueError(f"layer {layer_id!r} has no mesh (run it first)")
        verts, faces, vcols = read_mesh_ply(Path(mesh))
        mask = np.ones(len(faces), bool)
        rem = np.asarray([i for i in removed if 0 <= i < len(faces)], dtype=np.int64)
        mask[rem] = False
        faces = faces[mask]
        used = np.unique(faces)
        remap = np.full(len(verts), -1, np.int64)
        remap[used] = np.arange(len(used))
        verts2, faces2 = verts[used], remap[faces]
        vcols2 = vcols[used] if vcols is not None else None

        edits = self.project.manifest.project_dir / "edits"
        edits.mkdir(parents=True, exist_ok=True)
        new_id = _unique_id(f"{layer_id}_edit", {s.id for s in self.project.manifest.stages})
        out = edits / f"{new_id}.ply"
        write_mesh_ply(out, verts2, faces2, vcols2)
        self.project.add_stage(new_id, "import_mesh", chunk=chunk, params={"path": str(out)})
        self.project.save()
        return {"ok": True, "id": new_id, "kept_faces": int(len(faces2)), "removed_faces": int(rem.size)}

    def _layer_xyz(self, layer_id: str):
        """A layer's surface points (mesh vertices preferred, else dense/point cloud) + the source
        kind. Both are in the world frame the viewport picks in, so picked coords map directly."""
        from openreco.io.pointcloud import read_mesh_ply, read_ply
        art = self._last_run().get(layer_id, {}).get("artifacts", {})
        mesh = art.get("mesh")
        if mesh and Path(mesh).is_file():
            xyz, _, _ = read_mesh_ply(Path(mesh))
            return xyz, "mesh"
        ply = art.get("points") or art.get("merged")
        if not ply or not Path(ply).is_file():
            raise ValueError(f"layer {layer_id!r} has no mesh or point cloud (run it first)")
        xyz, _, _ = read_ply(Path(ply))
        return xyz, "cloud"

    def measure_volume(self, layer_id: str, polygon: list, base: str | float = "plane") -> dict:
        """Polygon-bounded cut/fill volume of a layer's surface (interactive stockpile measurement)."""
        from openreco.measure import measure_volume_region
        if len(polygon) < 3:
            raise ValueError("draw at least 3 points to outline the region")
        xyz, source = self._layer_xyz(layer_id)
        res = measure_volume_region(xyz, polygon, base=base)
        res["layer"] = layer_id
        res["source"] = source
        return res

    def measure_profile(self, layer_id: str, p_from, p_to, n: int = 200) -> dict:
        """Elevation cross-section along p_from -> p_to, sampled from a layer's surface."""
        from openreco.measure import measure_profile_region
        if not p_from or not p_to:
            raise ValueError("a profile needs two endpoints")
        xyz, source = self._layer_xyz(layer_id)
        res = measure_profile_region(xyz, p_from, p_to, n=n)
        res["layer"] = layer_id
        res["source"] = source
        return res

    def detect_markers(self, chunk: str | None, dictionary: str) -> dict:
        """Auto-detect coded targets across a chunk's photos -> GCP markers (id -> observations)."""
        import numpy as np
        from PIL import Image

        from openreco.markers import detect_markers as _detect
        imgs = self.images_for_chunk(chunk)["images"]
        if not imgs:
            return {"ok": False, "error": "no photos in this chunk (add/ingest photos first)"}
        per: dict[int, list] = {}
        scanned = 0
        for im in imgs:
            p = Path(im["path"])
            if not p.is_file():
                continue
            pil = Image.open(p).convert("L")
            w, h = pil.size
            s = min(1.0, 2400 / max(w, h))
            if s < 1.0:
                pil = pil.resize((int(w * s), int(h * s)))
            for det in _detect(np.asarray(pil), dictionary):
                per.setdefault(det["id"], []).append(
                    {"image": im["name"], "u": round(det["center"][0] / s, 1),
                     "v": round(det["center"][1] / s, 1)})
            scanned += 1
        markers = [{"name": f"marker_{mid}", "world": None, "type": "control", "observations": obs}
                   for mid, obs in sorted(per.items())]
        return {"ok": True, "markers": markers, "images_scanned": scanned,
                "detections": sum(len(o) for o in per.values())}

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
        rows = ["# name,X,Y,Z,image,u,v,type  (type=control|check; written by the UI marker tool)"]
        for mk in markers:
            w = mk.get("world") or [0.0, 0.0, 0.0]
            kind = "check" if mk.get("type") == "check" else "control"
            for ob in mk.get("observations", []):
                rows.append(f"{mk['name']},{w[0]},{w[1]},{w[2]},{ob['image']},{ob['u']},{ob['v']},{kind}")
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
    def start_run(self, force=None, targets=None) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
        self.cancel_requested = False
        self.events = queue.Queue()

        def worker():
            handler = _QueueLogHandler(self.events)
            handler.setFormatter(logging.Formatter("%(message)s"))
            log = logging.getLogger("openreco")
            old_level = log.level
            log.setLevel(logging.INFO)
            log.addHandler(handler)
            try:
                with _StderrCapture(self.events):       # surface COLMAP/glog native output too
                    self.project.run(force=force, targets=targets, on_event=self.events.put,
                                     cancel=lambda: self.cancel_requested)
            except Exception as exc:  # noqa: BLE001
                self.events.put({"event": "run_error", "error": repr(exc)})
            finally:
                log.removeHandler(handler)
                log.setLevel(old_level)
                self.events.put({"event": "_eof"})
                self.running = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    def cancel_run(self) -> bool:
        self.cancel_requested = True
        return self.running


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
        if route == "/api/presets":
            from openreco.workflow import presets
            return self._send(200, presets())
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
        if route == "/api/marker_template":
            return self._marker_template(parse_qs(u.query))
        if route == "/api/raster_png":
            return self._raster_png(parse_qs(u.query).get("path", [""])[0])
        if route == "/api/geo_overlay":
            return self._geo_overlay(parse_qs(u.query).get("path", [""])[0])
        if route == "/api/browse":
            return self._send(200, self.state.browse(parse_qs(u.query).get("path", [None])[0]))
        if route == "/api/thumb":
            return self._thumb(parse_qs(u.query).get("path", [""])[0])
        if route == "/api/cameras":
            return self._send(200, self.state.cameras_for_chunk(
                parse_qs(u.query).get("chunk", [None])[0]))
        if route == "/api/report":
            return self._report()
        if route == "/api/cesium":
            return self._cesium(parse_qs(u.query).get("layer", [""])[0])
        if route.startswith("/tiles3d/"):
            return self._tiles3d(route[len("/tiles3d/"):])
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if u.path == "/api/run":
            started = self.state.start_run(force=body.get("force"), targets=body.get("targets"))
            return self._send(202 if started else 409, {"started": started})
        if u.path == "/api/cancel":
            return self._send(200, {"cancelling": self.state.cancel_run()})
        if u.path == "/api/preset":
            try:
                return self._send(200, self.state.apply_preset(body.get("name", "")))
            except Exception as exc:  # noqa: BLE001
                return self._send(400, {"error": repr(exc)})
        if u.path == "/api/new_project":
            return self._new_project(body)
        if u.path == "/api/save_project":
            return self._send(200, self.state.save_project())
        if u.path == "/api/project":
            return self._set_project(body)
        if u.path == "/api/stage":
            return self._add_stage(body)
        if u.path == "/api/markers":
            return self._set_markers(body)
        if u.path == "/api/detect_markers":
            try:
                return self._send(200, self.state.detect_markers(body.get("chunk"),
                                  body.get("dictionary", "4x4_50")))
            except Exception as exc:  # noqa: BLE001
                return self._send(400, {"error": repr(exc)})
        if u.path == "/api/edit_cloud":
            try:
                return self._send(200, self.state.edit_cloud(body.get("layer", ""),
                                  body.get("removed", []), body.get("chunk", "Chunk 1")))
            except Exception as exc:  # noqa: BLE001
                return self._send(400, {"error": repr(exc)})
        if u.path == "/api/edit_mesh":
            try:
                return self._send(200, self.state.edit_mesh(body.get("layer", ""),
                                  body.get("removed", []), body.get("chunk", "Chunk 1")))
            except Exception as exc:  # noqa: BLE001
                return self._send(400, {"error": repr(exc)})
        if u.path == "/api/measure_volume":
            try:
                return self._send(200, self.state.measure_volume(body.get("layer", ""),
                                  body.get("polygon", []), body.get("base", "plane")))
            except Exception as exc:  # noqa: BLE001
                return self._send(400, {"error": str(exc)})
        if u.path == "/api/measure_profile":
            try:
                return self._send(200, self.state.measure_profile(body.get("layer", ""),
                                  body.get("from"), body.get("to"), int(body.get("n", 200))))
            except Exception as exc:  # noqa: BLE001
                return self._send(400, {"error": str(exc)})
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
        if u.path == "/api/layer":
            return self._layer(body)
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
        """Chunk operations: add | rename | remove (Workspace context menu)."""
        action = body.get("action", "add")
        p = self.state.project
        try:
            if action == "add":
                name = (body.get("name") or "").strip()
                if not name:
                    return self._send(400, {"error": "chunk name required"})
                p.add_chunk(name)
            elif action == "rename":
                p.rename_chunk(body["name"], (body.get("to") or "").strip())
            elif action == "remove":
                p.remove_chunk(body["name"])
            elif action == "set_enabled":
                p.set_chunk_enabled(body["name"], bool(body.get("enabled", True)))
            else:
                return self._send(400, {"error": f"unknown action {action!r}"})
            p.save()
            return self._send(200, {"ok": True})
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": repr(exc)})

    def _layer(self, body):
        """Layer operations: remove | rename | move | set_enabled (Workspace context menu)."""
        action = body.get("action", "")
        p = self.state.project
        try:
            if action == "remove":
                p.remove_stage(body["id"])
            elif action == "rename":
                p.rename_stage(body["id"], (body.get("to") or "").strip())
            elif action == "move":
                p.move_stage(body["id"], body["to"])
            elif action == "set_enabled":
                p.set_stage_enabled(body["id"], bool(body.get("enabled", True)))
            else:
                return self._send(400, {"error": f"unknown action {action!r}"})
            p.save()
            return self._send(200, {"ok": True})
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": repr(exc)})

    def _new_project(self, body):
        path = (body.get("path") or "").strip()
        if not path:
            return self._send(400, {"error": "project path required"})
        try:
            return self._send(200, self.state.new_project(path, body.get("name"), body.get("crs")))
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": repr(exc)})

    def _set_project(self, body):
        """Set project-level metadata from the UI (CRS picker)."""
        m = self.state.project.manifest
        if "crs" in body:
            m.crs = (body["crs"] or "").strip() or None
        self.state.project.save()
        return self._send(200, {"ok": True, "crs": m.crs})

    def _marker_template(self, q):
        from openreco.markers import marker_sheet_png
        try:
            png = marker_sheet_png(q.get("dictionary", ["4x4_50"])[0],
                                   count=int(q.get("count", ["24"])[0]))
            return self._send(200, png, "image/png")
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": repr(exc)})

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
        from openreco.workflow import preset_params, to_stage
        try:
            spec = to_stage(body["op"], body.get("values"))
            params = spec["params"]
            if self.state.preset:           # apply preset defaults under the op's chosen values
                base = preset_params(self.state.preset).get(spec["stage_type"], {})
                params = {**base, **params}
            return self._add_stage({"id": body["id"], "type": spec["stage_type"],
                                    "inputs": body.get("inputs", []), "params": params,
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

    def _report(self):
        """Serve the most recent run's processing report as a PDF (built from latest.json)."""
        from openreco.engine.report_pdf import write_report_pdf
        latest = self.state.project.manifest.runs_dir / "latest.json"
        data = None
        if latest.exists():
            try:
                data = json.loads(latest.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                data = None
        try:
            pdf = write_report_pdf(data)
        except Exception as exc:  # noqa: BLE001
            return self._send(500, {"error": f"report build failed: {exc!r}"})
        name = (self.state.project.manifest.name or "openreco").replace(" ", "_")
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Disposition", f'inline; filename="{name}_report.pdf"')
        self.send_header("Content-Length", str(len(pdf)))
        self.end_headers()
        self.wfile.write(pdf)

    def _tiles3d(self, rest):
        """Serve a 3D-Tiles file (tileset.json / tile_*.glb) of a tiles layer for Cesium streaming."""
        lid, _, fname = rest.partition("/")
        tdir = self.state._last_run().get(lid, {}).get("artifacts", {}).get("tiles")
        if not tdir:
            return self._send(404, {"error": "no tiles for that layer"})
        base = Path(tdir).resolve()
        p = (base / fname).resolve()
        if base != p and base not in p.parents:
            return self._send(403, {"error": "outside tiles dir"})
        if not p.is_file():
            return self._send(404, {"error": "not found"})
        self._send(200, p.read_bytes(), _CT.get(p.suffix.lower(), "application/octet-stream"))

    def _cesium(self, lid):
        """A self-contained CesiumJS page that streams a tiles layer over /tiles3d/<id>/."""
        if "tiles" not in self.state._last_run().get(lid, {}).get("artifacts", {}):
            return self._send(404, {"error": "run a Build Tiled Model layer first"})
        self._send(200, _CESIUM_HTML.replace("__LAYER__", lid), "text/html")

    def _thumb(self, path):
        """Serve an image file from anywhere (image suffixes only) for the Add-Photos picker."""
        from openreco.io.images import IMAGE_SUFFIXES
        p = Path(path)
        if not path or not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            return self._send(400, {"error": "image file required"})
        self._send(200, p.read_bytes(), _CT.get(p.suffix.lower(), "image/jpeg"))

    def _geo_overlay(self, path):
        """Reproject a georeferenced raster to WGS84 -> {bounds, image(data-url)} for the web map."""
        import base64

        from openreco.io.raster import raster_to_overlay
        p = Path(path)
        if not path or not self._in_project(p) or not p.is_file():
            return self._send(400, {"error": "valid in-project raster path required"})
        try:
            png, bounds = raster_to_overlay(p)
            return self._send(200, {"ok": True, "bounds": bounds,
                                    "image": "data:image/png;base64," + base64.b64encode(png).decode()})
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": str(exc)})       # clean message, not repr

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
