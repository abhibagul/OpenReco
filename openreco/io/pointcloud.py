"""Point-cloud readers/writers: binary PLY (viewer/mesh) and LAS (GIS).

Pure-numpy PLY (no plyfile dep). LAS via laspy. Coordinates are stored in true CRS meters
(local frame + georef origin), with the LAS header offset set to the origin for float
precision. CRS is embedded when available so the cloud opens correctly in QGIS/CloudCompare.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def points_from_reconstruction(rec) -> tuple[np.ndarray, np.ndarray]:
    """Extract (xyz [N,3] float64, rgb [N,3] uint8) from a pycolmap Reconstruction."""
    pts = rec.points3D
    n = len(pts)
    xyz = np.empty((n, 3), dtype=np.float64)
    rgb = np.empty((n, 3), dtype=np.uint8)
    for i, p in enumerate(pts.values()):
        xyz[i] = p.xyz
        rgb[i] = np.asarray(p.color, dtype=np.uint8)
    return xyz, rgb


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None,
              normals: np.ndarray | None = None) -> None:
    n = len(xyz)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if rgb is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.zeros(n, dtype=fields)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if normals is not None:
        arr["nx"], arr["ny"], arr["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    if rgb is not None:
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    typ = {"<f4": "float", "u1": "uchar"}
    header += [f"property {typ[t]} {name}" for name, t in fields]
    header += ["end_header", ""]
    with path.open("wb") as f:
        f.write("\n".join(header).encode("ascii"))
        f.write(arr.tobytes())


def write_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray,
                   vcolors: np.ndarray | None = None) -> None:
    """Binary PLY triangle mesh."""
    nv, nf = len(vertices), len(faces)
    vfields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if vcolors is not None:
        vfields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    varr = np.zeros(nv, dtype=vfields)
    varr["x"], varr["y"], varr["z"] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    if vcolors is not None:
        varr["red"], varr["green"], varr["blue"] = vcolors[:, 0], vcolors[:, 1], vcolors[:, 2]
    typ = {"<f4": "float", "u1": "uchar"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {nv}"]
    header += [f"property {typ[t]} {name}" for name, t in vfields]
    header += [f"element face {nf}", "property list uchar int vertex_indices", "end_header", ""]
    # face records: uchar count(3) + 3 int32
    frec = np.zeros(nf, dtype=[("c", "u1"), ("v", "<i4", (3,))])
    frec["c"] = 3
    frec["v"] = faces.astype("<i4")
    with path.open("wb") as f:
        f.write("\n".join(header).encode("ascii"))
        f.write(varr.tobytes())
        f.write(frec.tobytes())


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray,
              vcolors: np.ndarray | None = None) -> None:
    """Wavefront OBJ mesh. Vertex colors are written as the common 'v x y z r g b' extension."""
    lines = []
    if vcolors is not None:
        c = vcolors.astype(np.float64) / 255.0
        for (x, y, z), (r, g, b) in zip(vertices, c):
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}")
    else:
        for x, y, z in vertices:
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for a, b, c in faces + 1:  # OBJ is 1-indexed
        lines.append(f"f {a} {b} {c}")
    path.write_text("\n".join(lines), encoding="ascii")


def read_ply_xyzrgb(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Minimal reader for the binary-little-endian PLY this module writes."""
    with path.open("rb") as f:
        assert f.readline().strip() == b"ply"
        assert b"binary_little_endian" in f.readline()
        fields: list[tuple[str, str]] = []
        n = 0
        typemap = {"float": "<f4", "uchar": "u1", "int": "<i4"}
        while True:
            line = f.readline().strip()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line.startswith(b"property") and b"list" not in line:
                _, t, name = line.split()[:3]
                fields.append((name.decode(), typemap[t.decode()]))
            elif line == b"end_header":
                break
        arr = np.frombuffer(f.read(n * np.dtype(fields).itemsize), dtype=fields, count=n)
    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
    rgb = None
    if "red" in arr.dtype.names:
        rgb = np.column_stack([arr["red"], arr["green"], arr["blue"]]).astype(np.uint8)
    return xyz, rgb


_PLY_TYPE = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
_STRUCT = {"i1": "b", "u1": "B", "i2": "h", "u2": "H", "i4": "i", "u4": "I", "f4": "f", "f8": "d"}


def read_mesh_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an ASCII or binary-little-endian PLY triangle mesh with arbitrary property and
    face-list types (handles both our writer and pycolmap's). Returns (verts[N,3] float64,
    faces[M,3] int64, vcolors[N,3] uint8)."""
    import struct

    with path.open("rb") as f:
        fmt = None
        nv = nf = 0
        vprops: list[tuple[str, str]] = []
        count_t, index_t = "u1", "i4"
        in_face = False
        while True:
            line = f.readline().strip()
            if line.startswith(b"format"):
                fmt = line.split()[1].decode()
            elif line.startswith(b"element vertex"):
                nv = int(line.split()[-1])
            elif line.startswith(b"element face"):
                nf = int(line.split()[-1])
                in_face = True
            elif line.startswith(b"property list"):
                ct, it = line.split()[2].decode(), line.split()[3].decode()
                count_t, index_t = _PLY_TYPE[ct], _PLY_TYPE[it]
            elif line.startswith(b"property") and not in_face:
                t, name = line.split()[1].decode(), line.split()[2].decode()
                vprops.append((name, _PLY_TYPE[t]))
            elif line == b"end_header":
                break

        names = [n for n, _ in vprops]
        if fmt == "ascii":
            verts = np.empty((nv, 3))
            vcols = np.full((nv, 3), 200, np.uint8)
            ci = [names.index(c) for c in ("red", "green", "blue")] if "red" in names else None
            for i in range(nv):
                tok = f.readline().split()
                verts[i] = [float(tok[0]), float(tok[1]), float(tok[2])]
                if ci:
                    vcols[i] = [int(tok[ci[0]]), int(tok[ci[1]]), int(tok[ci[2]])]
            faces = np.array([f.readline().split()[1:4] for _ in range(nf)], dtype=np.int64)
            return verts, faces, vcols

        vd = np.dtype([(n, "<" + t) for n, t in vprops])
        varr = np.frombuffer(f.read(nv * vd.itemsize), dtype=vd, count=nv)
        verts = np.column_stack([varr["x"], varr["y"], varr["z"]]).astype(np.float64)
        vcols = (np.column_stack([varr["red"], varr["green"], varr["blue"]]).astype(np.uint8)
                 if "red" in names else np.full((nv, 3), 200, np.uint8))
        ct_sz, it_sz = np.dtype(count_t).itemsize, np.dtype(index_t).itemsize
        ct_c, it_c = _STRUCT[count_t], _STRUCT[index_t]
        faces = np.empty((nf, 3), dtype=np.int64)
        for i in range(nf):
            (cnt,) = struct.unpack("<" + ct_c, f.read(ct_sz))
            idx = struct.unpack("<" + it_c * cnt, f.read(it_sz * cnt))
            faces[i] = idx[:3]
        return verts, faces, vcols


def write_las(path: Path, xyz_world: np.ndarray, rgb: np.ndarray | None,
              crs_epsg: int | None, origin: np.ndarray) -> None:
    import laspy

    header = laspy.LasHeader(point_format=2 if rgb is not None else 0, version="1.4")
    header.offsets = origin
    header.scales = np.array([0.001, 0.001, 0.001])
    if crs_epsg:
        try:
            from pyproj import CRS

            header.add_crs(CRS.from_epsg(crs_epsg))
        except Exception:  # noqa: BLE001
            pass
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz_world[:, 0], xyz_world[:, 1], xyz_world[:, 2]
    if rgb is not None:
        # LAS color is 16-bit
        las.red = rgb[:, 0].astype(np.uint16) * 257
        las.green = rgb[:, 1].astype(np.uint16) * 257
        las.blue = rgb[:, 2].astype(np.uint16) * 257
    las.write(str(path))
