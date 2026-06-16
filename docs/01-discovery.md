# 01 — Discovery & Assumptions

## Goal (restated)

A clean-room photogrammetry / 3D-reconstruction platform — desktop & (eventually) cloud — that:

1. **Matches** commercial-suite Standard + Pro capability over a multi-year horizon.
2. **Eliminates** its structural pain points (hardware lock-in, no native Apple Silicon, weak
   collaboration, manual drudgery, slow large-job SfM/MVS, hard-surface failure, steep UI,
   reproducibility gaps).
3. **Leapfrogs** it by fusing survey-grade SfM/MVS geometry with neural appearance (3DGS/NeRF)
   on a shared camera solution, plus AI-assisted capture/processing.

**MVP slice:** photo set → georeferenced mesh + orthomosaic + shareable web view.

## Decided constraints

| Dimension | Decision | Why it matters |
|---|---|---|
| Team | Solo / learning build | Slice-first; minimize moving parts; reuse OSS aggressively |
| Deployment | Local-first | Desktop is where v1 proves value; cloud is roadmap |
| Licensing | **Permissive only (BSD/MIT/Apache)** | Closed-source-friendly; **excludes AGPL (OpenMVS, some splat code)** |
| v1 use case | UAV / drone mapping | Drives the slice: aerial imagery, GPS/GCP, DSM, orthomosaic |
| Intent | Architecture exercise + working slice | Honest, single-person roadmap; real end-to-end run as proof |

## Assumptions (challenge any)

- A1. Clean-room: published research + permissive OSS only. Hard constraint.
- A2. "Parity" = capability parity over time, not a v1 promise. v1 = the slice.
- A3. Metric accuracy is the geometry path's job; neural (3DGS/NeRF) augments appearance &
  hard surfaces and is never the source of truth for measurements.
- A4. Compute core eventually in Rust; GPU work behind a hardware-abstraction layer (wgpu →
  Vulkan/Metal/DX12). For the slice, orchestrate Python over native OSS.
- A5. Pipeline = typed DAG with checkpointing, caching, deterministic re-runs. "Pipeline-as-code"
  is the project format and the core reproducibility differentiator. Foundational, not bolted on.
- A6. Build *on* OSS where the license permits; reimplement only to differentiate or where licenses block.
- A7. Out-of-core / tiled data model assumed from day one — never assume dataset fits in RAM/VRAM.
- A8. Beginner recipes + expert full-control coexist over one parameter model.
- A9. Commercial model deferred (architecture exercise), but nothing in the design precludes
  perpetual + subscription + PAYG-credits later.

## User conflicts (architecture-shaping)

| Tension | Pull A | Pull B | Resolution lever |
|---|---|---|---|
| Accuracy vs photoreal | Survey/UAV/AEC: metric truth, watertight geometry | VFX/e-comm: photoreal, real-time splats | Dual branch on shared poses; surface the tradeoff |
| Local vs cloud | Heritage/gov: offline, residency | AEC/collab: elastic cloud + browser review | Same DAG runs local or on remote workers |
| Simple vs control | Students/field: one-click recipes | Pro surveyors: every param + Python | Beginner/expert toggle over one param model |
| Cost vs compute | Researchers: cheap on a laptop / Apple Silicon | Heavy MVS/neural is GPU-hungry | Out-of-core + cloud burst + native Metal |
| Geometry generality | Mapping: 2.5D terrain (DEM/ortho) | Heritage/VFX: full 3D close-range | Selectable stages per recipe; no single fixed flow |

## Open questions still parked (non-blocking for the slice)

- Q-A. Eventual commercial model specifics (deferred — architecture exercise).
- Q-B. Apple-Silicon hardware availability for testing the Metal path (the *design* supports it via
  wgpu regardless; physical test device TBD).
- Q-C. Whether learned matching (LightGlue/ALIKED) ships in v1 slice or Phase 2 — leaning Phase 2
  (classic SIFT via COLMAP is enough to prove the slice).
- Q-D. Target dataset size for the first real run (drives whether out-of-core tiling is exercised in v1).
