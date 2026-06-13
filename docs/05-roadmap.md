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
3. **Desktop UI with layers** (the big remaining piece) — a cross-platform app (evaluated:
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
- ✅ **glTF (.glb) export** (portable colored mesh; hand-written container, no dependency).
- ✅ **Coverage / overlap map** (per-image ground footprints → overlap GeoTIFF + PNG; QA + report cards).
- ✅ **GLOMAP global SfM** option (`mapper=global`) alongside incremental.
- ✅ **Contour lines** from the DSM (marching squares → WGS84 GeoJSON; standard survey product).
- ✅ **Volume measurement** (cut/fill from the DSM; `openreco volume` + `openreco.measure_volume`).
- ✅ **DTM** (morphological ground filter on the DSM, + nDSM object heights).
- ✅ **Point-cloud ground classification** (`classify`: grid-min + height threshold → classified LAS
  [ground/non-ground] + true bare-earth DTM from ground points). Validated: 64.7% ground on 1.68M
  aerial points. Next: CSF/progressive densification, building/vegetation sub-classes.
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

## Phase 3 — Parity wave 2 + differentiation · ≈6–12 months
- ✅ **GPU dense MVS** — real COLMAP PatchMatch stereo + fusion via a CUDA-enabled COLMAP binary
  (`openreco/compute.py` detection, `mvs` stage drives it; CPU sparse fallback). Validated on an
  NVIDIA GTX 1650 Ti: 265k dense points + a 1.4M-face dense Poisson mesh on the 11-image sample.
- **Rust + wgpu** hot-stage rewrites → GPU HAL → **native Apple Silicon / AMD** acceleration.
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
