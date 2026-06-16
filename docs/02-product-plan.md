# 02 — Product Plan

## Vision

> One capture, two truths: a **metrically accurate** model you can measure and survey, and a
> **photorealistic, real-time** model you can show anyone in a browser — reproducible to the last
> parameter, runnable on the hardware you already own.

## Positioning vs commercial suites

| Axis | commercial suites | OpenReco intent |
|---|---|---|
| Hardware | Local GPU/RAM-bound; weak on laptops; Rosetta on Mac | Out-of-core + GPU-agnostic (wgpu) + native ARM64 path |
| Collaboration | Single-user, license-per-seat, giant file hand-off | Local-first now; cloud-native projects + browser review later |
| Manual work | Hand-placed GCPs, tie-point nudging, manual masking | AI-assisted target detection, masking, culling, "alignment doctor" |
| Reproducibility | Parameters live in a project blob | Pipeline-as-code: versioned, diffable, re-runnable DAG |
| Appearance | Classic textured mesh | + 3DGS/NeRF branch on shared poses; hybrid fusion |
| Licensing | Rigid per-seat | Flexible later (deferred); open documented project format |

## Target users (priority for this build)

1. **UAV / drone mapping (v1 hero).** Metric ortho/DEM, georeferencing, volumes.
2. Researchers/students (scriptable, reproducible, runs on modest hardware).
3. Heritage/close-range objects (high-fidelity texture) — Phase 2+.
4. AEC / digital twins, VFX/e-commerce, agriculture/multispectral — later phases.

## Feature map (prioritized)

Legend: **[MVP]** v1 slice · **[P2]** parity wave 1 · **[P3]** parity wave 2 · **[DIFF]** differentiation

### Ingest & cameras
- **[MVP]** Frame-camera import, EXIF/GPS ingest, basic image-quality (blur) scoring + auto-cull
- **[P2]** Fisheye, spherical/360, multi-camera rigs, rolling-shutter correction, IMU ingest
- **[DIFF]** Capture-time mobile guidance (coverage/overlap/parallax warnings)

### Alignment (SfM)
- **[MVP]** SIFT detect/match, incremental SfM, self-calibration, sparse cloud, bundle adjustment (COLMAP)
- **[P2]** Global SfM option (GLOMAP), hierarchical for large sets, learned matching (LightGlue/ALIKED)
- **[DIFF]** "Alignment doctor": diagnose failed chunks, explain why, suggest fixes

### Dense / surface / texture
- **[MVP]** PatchMatch MVS depth maps + dense cloud (COLMAP, BSD); Poisson mesh; basic texture
- **[P2]** Decimation, hole-fill, smoothing; UV atlas, multi-image blend, de-lighting, PBR; tiled models
- **[DIFF]** Difficult-surface mode (monocular priors); 3DGS/NeRF branch; splat↔mesh fusion

### Georeferencing & geo products
- **[MVP]** EXIF-GPS georef, manual GCP/marker entry, CRS transforms (PROJ), scale; **DSM + orthomosaic** (GDAL)
- **[P2]** Coded/non-coded target auto-detection + sub-pixel refine, geoid/datum (NTv2), DTM, seamline editing/inpainting
- **[P2]** Point-cloud classification (ground/building/veg), filtering, height/class selection

### Measurement & analysis
- **[MVP]** Distance/area on the web view
- **[P2]** Volumes, cross-sections, stereoscopic measurement; processing report (reproj error, GCP residuals, coverage)
- **[P3]** LiDAR fusion + co-registration; multispectral/thermal band align + NDVI; 4D time-series; panorama stitching

### Platform & interop
- **[MVP]** Pipeline-as-code project format; headless CLI; PLY/OBJ/LAS/GeoTIFF/glTF export; static shareable web viewer
- **[P2]** Python API mirroring CLI 1:1; USD/USDZ, COPC, 3D Tiles, DXF/KML; desktop GUI
- **[P3/DIFF]** Cloud bursting, multi-user roles, collaborative viewer, plugin SDK, BYO-cloud / air-gapped

## MVP definition (the contract)

**"Smallest thing that takes a drone photo set → georeferenced mesh + DSM + orthomosaic + a shareable view."**

In scope:
- CLI: `openreco run <project.toml>` executes the full DAG with checkpoint/resume.
- Stages: ingest → SfM → georeference (EXIF-GPS + optional GCP file) → dense MVS → mesh → DSM → orthomosaic.
- Outputs: georeferenced `mesh.ply/obj`, `dense.las`, `dsm.tif` (GeoTIFF), `ortho.tif` (GeoTIFF), `report.html`.
- A static web bundle (three.js) to view the mesh/point cloud + measure distance, shareable as a folder.
- Reproducibility: project TOML records every parameter + tool versions; re-run is deterministic given inputs.

Explicitly **out** of MVP: GUI, cloud, neural branch, learned matching, classification, multispectral,
LiDAR fusion, USD/3D-Tiles streaming. All are roadmap (see 05).

## Success criteria for the slice

- A real public drone dataset (e.g. an ODM/COLMAP sample) runs end-to-end on a Windows laptop.
- Produced orthomosaic + DSM open correctly in QGIS with correct georeferencing.
- Re-running the project reproduces byte-comparable intermediate manifests (parameters/versions matched).
- The web viewer loads the result and measures a known distance within tolerance.
