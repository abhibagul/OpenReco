# 05 — Roadmap, Effort Sizing & Risk Register

Sizing is for **one developer**, in ideal focused weeks (calendar will be longer). "≈" = rough order.

## North star (product vision)

Match **every** the reference photogrammetry suite capability and exceed it — each feature working *better*, on
**any hardware** (NVIDIA/AMD/Apple-Silicon GPU **and** CPU-only, with graceful fallback), wrapped
in a **layered desktop UI** that surpasses the reference tool's. Pillars, in priority order:

1. **Hardware-agnostic compute** — today: CUDA dense (external COLMAP binary) + CPU fallback. Next:
   AMD/Apple-Silicon GPU paths (Vulkan/Metal via wgpu, or HIP/Metal COLMAP builds), auto-selected.
2. **Full processing parity** — alignment, dense, mesh, texture (now multi-image blended), tiled
   models, DSM/DTM/ortho, classification, LiDAR fusion, multispectral/NDVI, panoramas, 4D — plus
   our neural 3DGS branch. (See per-phase items below; most geometry/survey items are ✅.)
3. ◑ **Desktop UI with layers** — **shipped & growing**: `openreco ui` (native window via pywebview,
   else browser) with a **industry-standard layout** — menu bar (File / Workflow / Model / Tools /
   Help), toolbar, **Workspace/Reference** left tabs, a **industry-standard Workspace tree**
   (Workspace → Chunks → category nodes → items) with expand/collapse carets, the active chunk
   in bold (double-click to activate), data-count badges (points / faces / images), per-layer
   **visibility (👁) toggles**, and **right-click context menus** — chunks: set-active / add photos /
   rename / remove; layers: show-hide / rename / move-to-chunk / remove (/api/chunk, /api/layer).
   Categories: Cameras / Tie Points / Dense Cloud / Point Cloud / 3D Model / DEM / Orthomosaic /
   Shapes. Layers/chunks have **enable/disable checkboxes** (disabled branches skipped on Run),
   **drag-and-drop** between chunks, and **double-click opens** in the right view. **File** menu does
   **New / Save project**. **Cameras in 3D** (/api/cameras): camera frustums from solved poses, or an
   EXIF-GPS ENU preview before alignment. Panes are **resizable** (draggable splitters, persisted);
   the 3D view has an CAD-style **navigation cube** (click a face to snap the view) plus a
   **grid + axis triad**. Click-built pipelines **auto-wire** inputs (each op declares the artifacts
   it needs; layers declare what they provide). A **minimizable progress popup** (progress bar +
   live log + Cancel) and a **streaming Console** (the engine's INFO logs forwarded over SSE,
   /api/cancel for cooperative cancellation). Cameras render in the sky with drop-lines to a ground
   plane. **Model/Photo** viewport
   tabs, a **Properties** pane, and a **Console / Photos / Jobs** bottom dock. Features: active-chunk
   selector + "＋ Chunk"; **Workflow menu** (Align Photos / Build Dense Cloud / … Build dialogs →
   engine) incl. **Merge Chunks** (ICP align+merge); schema-driven parameter panels; Run with live
   SSE progress; three.js viewport with **multi-layer visibility, distance & area measurement, and
   Gaussian-splat rendering**; per-layer **Export as…**; a **CRS picker widget** (search /api/crs →
   set project CRS); **Photos thumbnail pane** + **GCP/marker picking** (click photos to place
   observations → writes a georef-ready `gcps.csv`, and **"Use these GCPs"** wires them straight
   into the chunk's Georeference step). An **Ortho 2D view** pans/zooms any raster product
   (ortho / DSM / vegetation index) rendered server-side to PNG (/api/raster_png). UI edits save
   the manifest (undo-via-diff). **Add Photos** is a file picker (/api/browse + /api/thumb) — navigate
   folders, multi-select specific images (thumbnails), and add them to a chunk: a subset of one folder
   becomes an ingest `select` whitelist; images spanning folders are staged-copied into the project so
   SfM keeps a single image root (/api/add_photos). Remaining: drag-to-reorder, depth-map viewer.
   Original plan: a
   cross-platform app (evaluated:
   **Tauri + web frontend** for the WebGPU 3D viewport we already have, vs Qt) presenting the
   project as a **layer tree** (chunks, cameras, tie/dense points, mesh, texture, DSM/DTM, ortho,
   contours, splats), a 3D/ortho/photo workspace, parameter panels driven by each stage's
   `params_schema`, a job queue over the DAG engine, and live progress — but *better* than
   the reference tool via our reproducible pipeline-as-code (every UI action edits the manifest; undo =
   diff; re-run = cache-aware). Effort ≈ a phase of its own (≈2–4 months).
4. **Reproducibility & openness as the moat** — already ahead: pipeline-as-code, content-addressed
   caching, run diffing, headless CLI + Python API, permissive/MIT.

## Phase 0 — Foundations (the engine) · ≈2–3 wks
Goal: the DAG engine exists and a no-op pipeline runs with caching, before any real algorithm.
- DAG scheduler, content-addressed cache, checkpoint/resume, manifest (TOML) loader.
- `Stage` protocol + `RunContext`; CLI `run/resume/diff/report`.
- License-check CI; tiny synthetic sample dataset; test harness.
- **Exit:** `openreco run` executes a 2-stage dummy DAG, caches, resumes, and re-run is a no-op.

## Phase 1 — The vertical slice (MVP) · ≈4–6 wks
Goal: real drone photos → georeferenced mesh + DSM + ortho + web view.
- Stages: ingest/QC → SfM (pycolmap) → georef (EXIF-GPS + GCP, pyproj) → MVS (PatchMatch) →
  mesh (Open3D Poisson) → DSM → orthomosaic (GDAL) → export + `report.html`.
- Static three.js viewer with distance measurement.
- **Exit (success criteria from 02):** a public drone set runs end-to-end on the Windows laptop;
  ortho+DSM open correctly georeferenced in QGIS; re-run reproduces manifests; viewer measures within tol.

## Phase 2 — Parity wave 1 · ≈3–5 months
- ✅ **Python API mirroring CLI 1:1** (`openreco.Project`: open/create/add_stage/run/resume/diff/save).
- ✅ **Richer processing report** (summary cards, QA by severity, GPS/GCP residuals, repro block).
- ✅ **GCP-based georeferencing** (file → triangulation → Umeyama; validated on real aerial data).
  Now survey-grade: **control vs check points** (check points held out of the fit to validate),
  **per-GCP residuals** (dx/dy/dz/total) + separate control/check RMSE in georef.json, surfaced in
  the UI Reference pane (accuracy table), and the marker tool tags each GCP control|check.
- ✅ **glTF (.glb) export** (portable colored mesh; hand-written container, no dependency).
- ✅ **Coverage / overlap map** (per-image ground footprints → overlap GeoTIFF + PNG; QA + report cards).
- ✅ **GLOMAP global SfM** option (`mapper=global`) alongside incremental.
- ✅ **Contour lines** from the DSM (marching squares → WGS84 GeoJSON; standard survey product).
- ✅ **Volume measurement** (cut/fill from the DSM; `openreco volume` + `openreco.measure_volume`).
- ✅ **DTM** (morphological ground filter on the DSM, + nDSM object heights).
- ✅ **Multi-class point classification** (`classify` v2: ground via grid-min+height; non-ground split
  into building [planar] / vegetation [rough] by PCA surface variation → ASPRS LAS codes 2/6/5) +
  bare-earth DTM from ground points. Validated on 1.68M aerial pts. Next: CSF, roughness tuning.
- ✅ **General export system** (`exporters.py` + `openreco export` + `export_product` API): format
  registry per product kind — mesh ply/obj/glb/STL/DXF · cloud ply/las/CSV · raster tif/png/ASC/KMZ ·
  vector geojson/KML · splat ply/.splat. (USD/3D-PDF/FBX/COPC flagged unsupported.) Backs the UI's
  "Export layer as…".
- ✅ **Cross-section profiles** (`openreco profile` + `openreco.measure_profile`).
- ✅ **Sparse-cloud filtering + camera re-optimization** (`refine` stage — "gradual selection":
  drop high-error/short-track tie points, re-run BA; composable via role-based inputs).
- Coded-target auto-detection + sub-pixel refine; USD/COPC/3D Tiles; learned matching;
  point-cloud ground classification (CSF/PMF) for a true DTM.
- Learned matching (LightGlue/ALIKED); GLOMAP global SfM; hierarchical for large sets.
- ✅ **Texturing**: mesh decimation (fast-simplification) + UV unwrap (xatlas) + atlas bake from the
  best source image per face → textured OBJ/MTL/PNG. Validated on the Sceaux dense mesh (150k faces,
  2048² atlas, 11 images, 60% coverage). Next: multi-image blending + de-lighting → PBR; textured glTF.
- Coded/non-coded GCP auto-detect + sub-pixel; geoid/NTv2; georeferenced BA; DTM; seamline+inpaint.
- Point-cloud classification (ground/veg/building); volumes, cross-sections.
- Exports: USD/USDZ, COPC, 3D Tiles, FBX, DXF, KML. Desktop GUI (evaluate Tauri vs Qt at that point).

- ✅ **Vegetation indices** (`indices`): RGB indices (ExG/VARI/GLI) on plain RGB orthos + NDVI/GNDVI
  when a NIR band is present → georeferenced index GeoTIFFs + colorized previews. Validated on the
  aerial RGB ortho. Next: reflectance-panel calibration, true multispectral band alignment.
- ✅ **LiDAR / external point-cloud fusion** (`fuse`): ICP (Kabsch) co-registration of a LAS/PLY
  cloud onto the dense cloud + merge. Validated: 120k-pt offset copy re-registers, RMS 0.24 m.
- ✅ **Panorama stitching** (`panorama`): skimage SIFT + RANSAC homography + warp/blend. Validated
  on 3 real images → 1493×1008, 83% coverage.
- ✅ **Tiled models** (`tiles`): mesh → N×N streamable 3D-Tiles (georeferenced). 3.36M faces → 16 tiles.
- ✅ **Batch processing** (`openreco batch`): run many projects (sequential/parallel) + aggregate
  report. (Distributed/network workers still future.)
- ✅ **Coded-target auto-detection** (`markers`): OpenCV ArUco/AprilTag → per-marker observations +
  GCP-observation CSV for georef. Validated on synthetic multi-view markers.
- ✅ **Coordinate-system selection + introspection** (PROJ/pyproj): every EPSG CRS supported;
  `crs_info` (datum/ellipsoid/prime-meridian/units/axes + sub-codes), `search_crs` catalog,
  `openreco crs` CLI, /api/crs, and output-CRS reprojection (`export --crs`, e.g. DSM→WGS84).
  Next: vertical/geoid datums (orthometric heights via grids), NTv2 grid shifts, UI CRS picker.

## Phase 3 — Parity wave 2 + differentiation · ≈6–12 months
- ✅ **GPU dense MVS** — real COLMAP PatchMatch stereo + fusion via a CUDA-enabled COLMAP binary
  (`openreco/compute.py` detection, `mvs` stage drives it). Validated on an NVIDIA GTX 1650 Ti:
  265k dense points + a 1.4M-face dense Poisson mesh on the 11-image sample.
- ✅ **Hardware-agnostic dense (portable backend)** — a **PyTorch plane-sweep** MVS
  (`mvs_planesweep.py`) that runs on **CUDA / Apple-Silicon MPS / AMD ROCm / CPU** from one
  codebase; `compute.select_dense_backend()` auto-picks colmap_cuda → planesweep → sparse.
  Validated on CUDA (2.6M pts) and CPU (synthetic-plane correctness). Lower quality than COLMAP
  CUDA but vendor-neutral — covers the non-NVIDIA gap without AGPL OpenMVS.
- Future: a **Rust + wgpu** kernel rewrite for a native (non-torch) cross-vendor HAL; quality
  parity (NCC windows, better consistency filtering, normal estimation) for the plane-sweep path.
- Out-of-core proven at scale; tiled models + streaming.
- ◑ **3DGS branch (gsplat)** on shared poses — `splat` stage **implemented** (init from sparse cloud,
  train via gsplat.rasterization, export standard 3DGS .ply). Runs in a gsplat-capable CUDA env;
  on this Windows box it's blocked by the CUDA compile toolchain (no full Toolkit/nvcc). Next:
  densification, SH view-dependence, splat↔mesh fusion, difficult-surface mode.
- AI assists: auto-masking (segmentation), smart culling, "alignment doctor".
- Multispectral/thermal + NDVI; LiDAR fusion + co-registration; 4D time-series; panorama.
- Cloud bursting, multi-user roles, collaborative browser viewer, plugin SDK, air-gapped tier.
- Capture-time mobile guidance app.

## Dependencies (critical path)
```
Engine (P0) ─▶ SfM ─▶ Georef ─▶ MVS ─▶ Mesh ─▶ DSM ─▶ Ortho ─▶ Viewer/Export
                                  └▶ (3DGS branch, P3, reuses SfM poses)
GPU HAL (wgpu, P3) is independent until hot-stage rewrites; not on the slice critical path.
```

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Permissive-only blocks a needed algorithm (e.g. best dense MVS is AGPL OpenMVS) | Med | High | Use COLMAP PatchMatch (BSD) for v1; benchmark; reimplement specific kernels in Rust if quality gap matters |
| pycolmap build/install friction on Windows | Med | Med | Pin wheels / conda-forge; document a reproducible env; CI on Windows |
| Georeferencing accuracy without full georeferenced BA in v1 | Med | Med | Ship post-hoc similarity fit for slice; flag as non-survey-grade until P2 georeferenced BA |
| Solo bandwidth — scope creep into breadth | High | High | Hard gate: no breadth until slice success criteria pass; roadmap phases are firm |
| GPU portability debt if we lean on CUDA-only paths | Med | Med | Keep CPU fallback for every stage; isolate GPU calls behind compute/ ; wgpu in P3 |
| Determinism claims undercut by nondeterministic libs (SIFT/BA threading) | Med | Low | Record nondeterministic stages explicitly; pin seeds/threads where possible; diff on params not bytes for those |
| Out-of-core complexity added too early | Med | Med | v1 targets a dataset that fits; design tiling seams but don't over-build until P3 |
| Apple-Silicon promise without test hardware | Low | Med | Design for Metal via wgpu; defer the claim until a device is available to validate |

## What "done" means at each gate
- **P0 done:** engine runs + caches + resumes a dummy DAG.
- **P1 done (MVP):** the four success criteria in 02 pass on a real dataset.
- **P2 done:** Python API + report + texturing + GCP auto + classification + broad exports + GUI.
- **P3 done:** GPU HAL on 3 vendors + neural branch + cloud + collaboration.
