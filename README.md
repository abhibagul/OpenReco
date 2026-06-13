# OpenReco (working name) — a next-generation photogrammetry & 3D reconstruction platform

A clean-room, local-first photogrammetry platform: drone/aerial photos → georeferenced
mesh + DSM + orthomosaic + a shareable 3D web view. Built on permissively-licensed
open source (BSD/MIT/Apache only), designed to grow toward professional-class parity and to
fuse classic survey-grade geometry (SfM/MVS) with neural appearance (3D Gaussian Splatting).

> **Clean-room note.** Built from published research and permissively-licensed OSS only.
> No the reference photogrammetry suite code, UI assets, or trademarks. Capability parity, not implementation copying.

## Project context (decided)

- **Team:** solo / learning build — slice-first, lean on mature OSS.
- **Deployment:** local-first, **permissive licenses only** (no AGPL → no OpenMVS).
- **v1 use case:** UAV / drone mapping.
- **Intent:** architecture exercise + one working end-to-end slice.

## Documents (read in order)

1. [docs/01-discovery.md](docs/01-discovery.md) — goal, assumptions, open questions, user conflicts
2. [docs/02-product-plan.md](docs/02-product-plan.md) — vision, feature map, MVP definition
3. [docs/03-architecture.md](docs/03-architecture.md) — system design, stack + tradeoffs, GPU & out-of-core strategy
4. [docs/04-pipeline-spec.md](docs/04-pipeline-spec.md) — end-to-end pipeline, algorithm choices + fallbacks
5. [docs/05-roadmap.md](docs/05-roadmap.md) — phased milestones, effort sizing, risk register

## The vertical slice (MVP) — working end-to-end

```
photos/ ──▶ ingest ──▶ SfM (poses+sparse) ──▶ georeference ──▶ dense MVS
                                                                   │
   shareable web view ◀── orthomosaic ◀── DSM ◀── mesh ◀──────────┘
```

All eight stages run on the engine (DAG + content-addressed cache + checkpoint/resume).
Verified end-to-end on a real 11-image dataset: 11/11 cameras registered, 0.63 px mean
reprojection error, producing a georeferenced point cloud, mesh, DSM, orthomosaic, and a
static three.js viewer.

### Quickstart

```bash
pip install -e ".[slice]"          # permissive deps: pycolmap, pyproj, rasterio, scipy, laspy...
python scripts/fetch_sample.py     # 11-image Sceaux Castle sample (~13 MB)
openreco run samples/sceaux        # photos -> ... -> shareable bundle in samples/sceaux/output/
cd samples/sceaux/output && python serve.py   # then open http://localhost:8000/
```

Other commands: `openreco resume <proj>` (continue from cache), `openreco diff a.toml b.toml`
(which stages would recompute), `openreco stages` (list stage types).

### Status & honest caveats
- **Dense MVS needs CUDA.** On a CPU-only machine the `mvs` stage falls back to the SfM sparse
  cloud (loudly flagged); mesh/DSM are correspondingly coarse. True dense needs a GPU.
- **Georeferencing** in v1 is a post-hoc GPS/Sim3d fit (or local-frame fallback when there's no
  GPS, as in the Sceaux sample). Full georeferenced bundle adjustment + GCP auto-detection is Phase 2.
- **Orthophoto** is point-cloud-based; true image-resampled orthorectification + seamlines is Phase 2.
- No GUI / cloud / neural (3DGS) branch yet — see [docs/05-roadmap.md](docs/05-roadmap.md).

Phase status: **Phase 0 (engine) ✅ · Phase 1 (UAV slice) ✅ functional.** Roadmap in [docs/05-roadmap.md](docs/05-roadmap.md).
