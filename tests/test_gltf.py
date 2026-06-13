"""GLB writer: produce a colored triangle mesh and parse it back to validate the container."""

from __future__ import annotations

import json
import struct

import numpy as np

from openreco.io.gltf import write_glb


def _parse_glb(path):
    data = path.read_bytes()
    magic, version, total = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67 and version == 2 and total == len(data)
    jlen, jtype = struct.unpack_from("<II", data, 12)
    assert jtype == 0x4E4F534A
    gltf = json.loads(data[20:20 + jlen])
    blen, btype = struct.unpack_from("<II", data, 20 + jlen)
    assert btype == 0x004E4942
    bin_buf = data[20 + jlen + 8: 20 + jlen + 8 + blen]
    return gltf, bin_buf


def test_glb_roundtrip_positions_and_indices(tmp_path):
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.5]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    vcols = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], dtype=np.uint8)
    p = tmp_path / "m.glb"
    write_glb(p, verts, faces, vcols)

    gltf, buf = _parse_glb(p)
    acc = gltf["accessors"]
    assert acc[0]["type"] == "VEC3" and acc[0]["count"] == 4      # positions
    assert acc[1]["type"] == "VEC4" and acc[1]["normalized"]       # colors
    assert acc[2]["count"] == 6                                    # 2 triangles -> 6 indices
    assert gltf["meshes"][0]["primitives"][0]["mode"] == 4         # TRIANGLES

    # read positions back out of the binary chunk via the bufferView offsets
    bv = gltf["bufferViews"][0]
    pos = np.frombuffer(buf[bv["byteOffset"]:bv["byteOffset"] + bv["byteLength"]],
                        dtype="<f4").reshape(-1, 3)
    assert np.allclose(pos, verts, atol=1e-6)
    # indices
    iv = gltf["bufferViews"][2]
    idx = np.frombuffer(buf[iv["byteOffset"]:iv["byteOffset"] + iv["byteLength"]], dtype="<u4")
    assert np.array_equal(idx.reshape(-1, 3), faces)


def test_textured_glb_roundtrip(tmp_path):
    import io as _io

    from PIL import Image

    from openreco.io.gltf import write_glb_textured

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    buf = _io.BytesIO()
    Image.new("RGB", (4, 4), (200, 100, 50)).save(buf, format="PNG")
    png = buf.getvalue()

    p = tmp_path / "t.glb"
    write_glb_textured(p, verts, faces, uvs, png)
    gltf, binblob = _parse_glb(p)
    assert gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    assert gltf["images"][0]["mimeType"] == "image/png"
    assert gltf["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] == 1
    # the embedded PNG bytes are present in the binary chunk
    assert png in binblob
    assert p.stat().st_size % 4 == 0


def test_glb_total_length_is_4byte_aligned(tmp_path):
    verts = np.random.default_rng(0).random((5, 3))
    faces = np.array([[0, 1, 2], [2, 3, 4]])
    p = tmp_path / "m.glb"
    write_glb(p, verts, faces)
    assert p.stat().st_size % 4 == 0
