"""Export system — convert any product to many formats (format registry + export_product).

A product is detected by file type (mesh / point cloud / raster / vector / splat); each kind has
a set of permissively-implementable export formats. This backs the UI's "Export layer as…" and a
headless `openreco export`. Heavier formats (USD, 3D-PDF, FBX-binary, COPC) are intentionally left
out for now — noted in UNSUPPORTED with the reason.

    export_product("output/mesh.ply", "stl", "out.stl")
    list_formats("output/dsm.tif")        # -> ['tif', 'png', 'kmz', 'asc']
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import numpy as np

from openreco.io.pointcloud import (
    read_mesh_ply,
    read_ply,
    write_mesh_ply,
    write_obj,
    write_ply,
    write_las,
)

UNSUPPORTED = {
    "usd": "needs usd-core (large Apache dep)",
    "usdz": "needs usd-core",
    "3dpdf": "needs a 3D-PDF/PRC toolchain",
    "fbx": "binary FBX needs the FBX SDK; ascii FBX is lossy/complex",
    "copc": "no permissive COPC writer (laspy reads only; needs native PDAL)",
}


# ---- product detection ------------------------------------------------------------------

def detect_kind(path: str | Path) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".tif", ".tiff"):
        return "raster"
    if ext in (".geojson", ".json"):
        return "vector"
    if ext in (".las", ".laz"):
        return "pointcloud"
    if ext == ".ply":
        return _ply_kind(p)
    raise ValueError(f"unknown product type for {p.name}")


def _ply_kind(p: Path) -> str:
    import re
    head = p.read_bytes()[:4096].decode("latin1", "ignore")
    if "f_dc_0" in head and "opacity" in head:
        return "splat"                                  # 3D Gaussian Splatting PLY
    m = re.search(r"element face (\d+)", head)
    return "mesh" if (m and int(m.group(1)) > 0) else "pointcloud"


# ---- loaders ----------------------------------------------------------------------------

def _load_mesh(src):
    v, f, c = read_mesh_ply(Path(src))
    return {"verts": v, "faces": f, "vcolors": c}


def _load_cloud(src):
    src = Path(src)
    if src.suffix.lower() in (".las", ".laz"):
        import laspy
        las = laspy.read(str(src))
        xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
        rgb = None
        if hasattr(las, "red"):
            rgb = (np.column_stack([las.red, las.green, las.blue]) // 257).astype(np.uint8)
        return {"xyz": xyz, "rgb": rgb}
    xyz, rgb, _ = read_ply(src)
    return {"xyz": xyz, "rgb": rgb}


def _load_raster(src):
    import rasterio
    with rasterio.open(src) as ds:
        return {"array": ds.read(), "transform": ds.transform, "crs": ds.crs, "nodata": ds.nodata}


def _load_vector(src):
    return json.loads(Path(src).read_text(encoding="utf-8"))


def _load_splat(src):
    xyz, _, _ = read_ply(Path(src))
    # full attribute read for the .splat conversion
    return {"path": Path(src), "n": len(xyz)}


# ---- mesh writers -----------------------------------------------------------------------

def _mesh_stl(m, out):
    v, f = m["verts"], m["faces"]
    tris = v[f].astype(np.float64)                      # (M,3,3)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = nrm
    rec["v"] = tris
    with open(out, "wb") as fh:                          # vectorized: handles million-face meshes
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(f)))
        fh.write(rec.tobytes())


def _mesh_dxf(m, out):
    v, f = m["verts"], m["faces"]
    lines = ["0", "SECTION", "2", "ENTITIES"]
    for a, b, c in f:
        lines += ["0", "3DFACE", "8", "0"]
        for j, vi in enumerate((a, b, c, c)):          # 4th vertex repeats the 3rd (triangle)
            x, y, z = v[vi]
            lines += [str(10 + j), f"{x:.6f}", str(20 + j), f"{y:.6f}", str(30 + j), f"{z:.6f}"]
    lines += ["0", "ENDSEC", "0", "EOF"]
    Path(out).write_text("\n".join(lines), encoding="ascii")


def _mesh_glb(m, out):
    from openreco.io.gltf import write_glb
    write_glb(Path(out), m["verts"], m["faces"], m["vcolors"])


# ---- point-cloud writers ----------------------------------------------------------------

def _cloud_csv(c, out):
    xyz, rgb = c["xyz"], c["rgb"]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("x,y,z,r,g,b\n" if rgb is not None else "x,y,z\n")
        if rgb is not None:
            for (x, y, z), (r, g, b) in zip(xyz, rgb):
                fh.write(f"{x:.4f},{y:.4f},{z:.4f},{r},{g},{b}\n")
        else:
            for x, y, z in xyz:
                fh.write(f"{x:.4f},{y:.4f},{z:.4f}\n")


def _cloud_las(c, out):
    write_las(Path(out), c["xyz"], c["rgb"], None, c["xyz"].mean(0))


# ---- raster writers ---------------------------------------------------------------------

def _raster_png(r, out):
    from PIL import Image
    a = r["array"]
    if a.shape[0] == 1:                                # single band -> normalized grayscale
        b = a[0].astype(np.float64)
        finite = np.isfinite(b)
        lo, hi = np.percentile(b[finite], [2, 98]) if finite.any() else (0, 1)
        g = np.clip((b - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(g, "L").save(out)
    else:
        Image.fromarray(np.moveaxis(a[:3], 0, 2).astype(np.uint8), "RGB").save(out)


def _raster_asc(r, out):
    import rasterio
    a = r["array"][0].astype(np.float32)
    t = r["transform"]
    nd = r["nodata"] if r["nodata"] is not None else -9999.0
    a = np.where(np.isfinite(a), a, nd)
    with rasterio.open(out, "w", driver="AAIGrid", height=a.shape[0], width=a.shape[1], count=1,
                       dtype="float32", transform=t, crs=r["crs"], nodata=nd) as dst:
        dst.write(a, 1)


def _raster_kmz(r, out):
    """Ground-overlay KMZ: a PNG of the raster + a KML LatLonBox (WGS84 bounds)."""
    import io

    from PIL import Image
    from pyproj import Transformer

    a = r["array"]
    png = np.moveaxis(a[:3], 0, 2).astype(np.uint8) if a.shape[0] >= 3 else None
    if png is None:                                    # colorize single band
        b = a[0].astype(np.float64)
        m = np.isfinite(b)
        lo, hi = np.percentile(b[m], [2, 98]) if m.any() else (0, 1)
        png = np.clip((b - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
        png = np.dstack([png, png, png])
    t = r["transform"]
    h, w = a.shape[1], a.shape[2]
    west, north = t.c, t.f
    east, south = t.c + t.a * w, t.f + t.e * h
    if r["crs"] is not None:
        tf = Transformer.from_crs(r["crs"], 4326, always_xy=True)
        west, north = tf.transform(west, north)
        east, south = tf.transform(east, south)
    out = Path(out)
    buf = io.BytesIO()
    Image.fromarray(png, "RGB").save(buf, "PNG")
    kml = (f'<?xml version="1.0"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><GroundOverlay>'
           f'<name>{out.stem}</name><Icon><href>overlay.png</href></Icon>'
           f'<LatLonBox><north>{north}</north><south>{south}</south>'
           f'<east>{east}</east><west>{west}</west></LatLonBox></GroundOverlay></kml>')
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
        z.writestr("overlay.png", buf.getvalue())


# ---- vector writers ---------------------------------------------------------------------

def _vector_kml(g, out):
    placemarks = []
    for feat in g.get("features", []):
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})
        name = ", ".join(f"{k}={v}" for k, v in props.items())
        for line in _iter_linestrings(geom):
            coords = " ".join(f"{x},{y},{z if len(c) > 2 else 0}"
                              for c in line for x, y, *z_ in [c] for z in [z_[0] if z_ else 0])
            placemarks.append(f"<Placemark><name>{name}</name><LineString><coordinates>"
                              f"{coords}</coordinates></LineString></Placemark>")
    Path(out).write_text('<?xml version="1.0"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">'
                         f"<Document>{''.join(placemarks)}</Document></kml>", encoding="utf-8")


def _iter_linestrings(geom):
    t = geom.get("type")
    if t == "LineString":
        yield geom["coordinates"]
    elif t == "MultiLineString":
        yield from geom["coordinates"]


# ---- splat writer -----------------------------------------------------------------------

def _splat_splat(s, out):
    """Convert a 3DGS .ply to the compact antimatter15 .splat (32 bytes/gaussian)."""
    arr = _read_full_ply(s["path"])
    n = len(arr["x"])
    sh_c0 = 0.28209479177387814
    rgb = np.clip((np.column_stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]]) * sh_c0 + 0.5)
                  * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(1 / (1 + np.exp(-arr["opacity"])) * 255, 0, 255).astype(np.uint8)
    scale = np.exp(np.column_stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]])).astype("<f4")
    quat = np.column_stack([arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"]])
    quat = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-9)
    rot = np.clip(quat * 128 + 128, 0, 255).astype(np.uint8)
    pos = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype("<f4")
    with open(out, "wb") as fh:
        for i in range(n):
            fh.write(pos[i].tobytes())
            fh.write(scale[i].tobytes())
            fh.write(bytes([*rgb[i], alpha[i]]))
            fh.write(rot[i].tobytes())


def _read_full_ply(path: Path):
    with path.open("rb") as f:
        assert f.readline().strip() == b"ply"
        f.readline()
        fields, n = [], 0
        tmap = {"float": "<f4", "uchar": "u1", "double": "<f8", "int": "<i4"}
        while True:
            ln = f.readline().strip()
            if ln.startswith(b"element vertex"):
                n = int(ln.split()[-1])
            elif ln.startswith(b"property") and b"list" not in ln:
                _, t, name = ln.split()[:3]
                fields.append((name.decode(), tmap[t.decode()]))
            elif ln == b"end_header":
                break
        return np.frombuffer(f.read(n * np.dtype(fields).itemsize), dtype=fields, count=n)


# ---- registry ---------------------------------------------------------------------------

_PASSTHROUGH = {"ply", "obj", "tif", "geojson", "las"}  # native forms handled specially

REGISTRY = {
    "mesh": {"ply": lambda m, o: write_mesh_ply(Path(o), m["verts"], m["faces"], m["vcolors"]),
             "obj": lambda m, o: write_obj(Path(o), m["verts"], m["faces"], m["vcolors"]),
             "glb": _mesh_glb, "stl": _mesh_stl, "dxf": _mesh_dxf},
    "pointcloud": {"ply": lambda c, o: write_ply(Path(o), c["xyz"], c["rgb"]),
                   "las": _cloud_las, "csv": _cloud_csv},
    "raster": {"tif": None, "png": _raster_png, "asc": _raster_asc, "kmz": _raster_kmz},
    "vector": {"geojson": None, "kml": _vector_kml, "csv": None},
    "splat": {"ply": None, "splat": _splat_splat},
}
_LOADERS = {"mesh": _load_mesh, "pointcloud": _load_cloud, "raster": _load_raster,
            "vector": _load_vector, "splat": _load_splat}


def list_formats(src: str | Path) -> list[str]:
    return sorted(REGISTRY[detect_kind(src)])


def export_product(src: str | Path, fmt: str, out: str | Path, kind: str | None = None) -> Path:
    src, out = Path(src), Path(out)
    fmt = fmt.lower()
    if fmt in UNSUPPORTED:
        raise ValueError(f"format {fmt!r} not supported: {UNSUPPORTED[fmt]}")
    kind = kind or detect_kind(src)
    if kind not in REGISTRY or fmt not in REGISTRY[kind]:
        raise ValueError(f"cannot export a {kind} as {fmt!r}; choices: {sorted(REGISTRY[kind])}")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = REGISTRY[kind][fmt]
    if writer is None:                                  # passthrough: same representation, just copy
        import shutil
        shutil.copyfile(src, out)
        return out
    writer(_LOADERS[kind](src), out)
    return out
