"""Minimal binary glTF (.glb) writer for a colored triangle mesh.

glTF 2.0 is the portable, widely-supported runtime 3D format (three.js, Blender, USD pipelines,
game engines). We write it by hand — a GLB is just a 12-byte header + a JSON chunk + a binary
chunk — so we add no dependency (consistent with our hand-written PLY/OBJ writers).

Layout of the single binary buffer: POSITION (f32 vec3) | COLOR_0 (u8 vec4, normalized) |
indices (u32 scalar). Offsets are 4-byte aligned by construction.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

# glTF component types / targets / primitive mode
_FLOAT, _UBYTE, _UINT = 5126, 5121, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_TRIANGLES = 4


def _pad(buf: bytes, fill: bytes) -> bytes:
    rem = (-len(buf)) % 4
    return buf + fill * rem


def write_glb(path: Path, vertices: np.ndarray, faces: np.ndarray,
              vcolors: np.ndarray | None = None) -> None:
    n, m = len(vertices), len(faces)
    pos = np.ascontiguousarray(vertices, dtype="<f4")
    idx = np.ascontiguousarray(faces.reshape(-1), dtype="<u4")
    if vcolors is None:
        vcolors = np.full((n, 3), 200, np.uint8)
    rgba = np.empty((n, 4), dtype=np.uint8)
    rgba[:, :3] = vcolors
    rgba[:, 3] = 255

    pos_b = pos.tobytes()
    col_b = rgba.tobytes()
    idx_b = idx.tobytes()
    pos_off, col_off = 0, len(pos_b)
    idx_off = col_off + len(col_b)
    bin_buf = _pad(pos_b + col_b + idx_b, b"\x00")

    vmin = pos.min(axis=0).tolist()
    vmax = pos.max(axis=0).tolist()
    gltf = {
        "asset": {"version": "2.0", "generator": "OpenReco"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2, "mode": _TRIANGLES,
        }]}],
        "buffers": [{"byteLength": len(bin_buf)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos_b), "target": _ARRAY_BUFFER},
            {"buffer": 0, "byteOffset": col_off, "byteLength": len(col_b), "target": _ARRAY_BUFFER},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx_b),
             "target": _ELEMENT_ARRAY_BUFFER},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": _FLOAT, "count": n, "type": "VEC3",
             "min": vmin, "max": vmax},
            {"bufferView": 1, "componentType": _UBYTE, "normalized": True, "count": n, "type": "VEC4"},
            {"bufferView": 2, "componentType": _UINT, "count": m * 3, "type": "SCALAR"},
        ],
    }

    json_b = _pad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(json_b) + 8 + len(bin_buf)
    with path.open("wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))            # 'glTF', version 2, length
        f.write(struct.pack("<II", len(json_b), 0x4E4F534A))          # JSON chunk header ('JSON')
        f.write(json_b)
        f.write(struct.pack("<II", len(bin_buf), 0x004E4942))         # BIN chunk header ('BIN\0')
        f.write(bin_buf)
