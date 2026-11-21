# Project: visualizeIT — Unified >=15 FPS Video & Interactive Surface Design Studio

## Architecture

### Overview & Product Goal
`visualizeIT` maps procedural or custom uploaded 2D designs/artwork onto real-world objects and curved/planar/3D surfaces across recorded video files and live camera streams with realistic perspective, cylindrical/3D surface wrapping, occlusion handling, and scene-luminance shading preservation.

Prior to this consolidation, the repository contained five disconnected experimental video/camera subsystems (`bench/workbench_*`, `bench/web/design-studio/phone-camera.js`, `bench/prototype_service.py`, `bench/*_experiment.py`, and `reference/qhacks2025/`), none of which achieved sustained **>= 15 FPS (`<= 66.67 ms/frame`)** end-to-end artwork mapping on recorded video.

To fulfill `ORIGINAL_REQUEST.md` (*"decide on what to do and what is needed and the structure of the app and then delegate agents to tackle each task"* and *"get it working at 15 fps at least for the video version"*) without breaking existing repository tests or SHA-256-pinned research files (`bench/workbench_neural.py:94-133`), `visualizeIT` unifies around a clean, modular production stack:

```text
                       ┌──────────────────────────────────────────────────────┐
                       │         Unified Web Studio (visualizeit/web/)        │
                       │  • Design Studio (Prompts, Patterns, RGBA Upload)    │
                       │  • Recorded Video Mode (.mp4/.webm/.mov, >=15-30 FPS)│
                       │  • Live Camera Mode (Uncapped WebWorker + WebGL/2D)  │
                       │  • Live FPS / P95 ms / Inliers / State Telemetry HUD │
                       └───────────────┬──────────────────────┬───────────────┘
                                       │ REST (HTTP/1.1)      │ WebSocket (/ws/...)
                                       ▼                      ▼
┌──────────────────────┐   ┌──────────────────────────────────────────────────┐
│ CLI (`visualizeit`)  │──▶│      FastAPI Backend (`visualizeit/server.py`)   │
│ demo-assets | info   │   │  Upload, Range Media, Select, Render MP4, Stream │
│ segment | process    │   └────────────────────────┬─────────────────────────┘
│ benchmark | serve    │                            │
└──────────┬───────────┘                            ▼
           │               ┌──────────────────────────────────────────────────┐
           └──────────────▶│   Streaming Video Pipeline (`visualizeit/`)      │
                           │  1. Sequential VideoCapture / Frame Ingest       │
                           │  2. One-Time GrabCut + Edge Mask Onboarding      │
                           │  3. Causal KLT + ORB/LSH Tracker + Re-Acquire    │
                           │     (Planar / Cylinder / Grid / 6-DoF 3D Mesh)   │
                           │  4. Fast Surface Compositor (C++17 + OpenCV ROI) │
                           │  5. FFmpeg libx264 Ultrafast / Stream Output     │
                           └────────────────────────┬─────────────────────────┘
                                                    │ ctypes (extern "C" ABI)
                                                    ▼
                           ┌──────────────────────────────────────────────────┐
                           │   Native C++17 AVX-512 Engine (`native/`)        │
                           │  • `pose.cpp` (GLTF <-> Unity pose transform)    │
                           │  • `raster.cpp` (Z-buffer triangle rasterizer,   │
                           │    perspective UV, luminance shading, ROI blend) │
                           └──────────────────────────────────────────────────┘
```

### Host Hardware & Operational Constraints
- **Target Host**: Debian 12 Linux x86_64, **2 vCPUs** (`Intel Xeon Platinum 8481C` with AVX-512 & 105 MiB L3), **7.8 GiB RAM**, 0 swap, **CPU-only (no GPU)**.
- **Filesystem Rules (MANDATORY)**:
  - **NEVER** place temporary files, builds, virtualenvs, datasets, or `TMPDIR` under `/tmp` or inside `.agents/teamwork/`.
  - `.agents/teamwork/` is strictly for `.md` agent coordination metadata — never source code, tests, or data, and never create `AGENTS.md` or `GEMINI.md` there.
  - `/home/antigravity` (including `/home/antigravity/achitaan/visualizeIT`) is **XFS** (19 GiB free, full POSIX `chmod` support). All persistent local virtualenvs, CMake build directories, and generated demo video caches MUST reside under gitignored `/home/antigravity/achitaan/visualizeIT/.cache/` (`.cache/venv`, `.cache/native-build`, `.cache/demo-videos`, `.cache/tmp`).
  - `/home/antigravity/.gemini/jetski` is `virtiofs` (`chmod` returns `EPERM`); use `/home/antigravity/.gemini/jetski/scratch/teamwork_tmp/<conversation-id>` for per-agent ad-hoc scratch files/probes, and `/home/antigravity/achitaan/visualizeIT/.cache/tmp` when `chmod`-requiring build tools (`cmake`, `uv`) need `TMPDIR`.
- **Anti-Cheating & >=15 FPS Performance Invariant**:
  - Every video run MUST process **100% of input frames** (`render_stride == 1`, `input_frame_count == processed_frame_count == rendered_frame_count == output_frame_count`).
  - No frame skipping masquerading as processed frames, no pre-baked/cached output frames, no ground-truth poses leaked into tracking, and no synthetic sleeps/mocks.
  - End-to-end throughput (`processed_frame_count / wall_time_seconds`, covering decode + preprocess + tracking + surface warp/compositing + output encode/stream) MUST sustain **`>= 15.0 FPS` (`<= 66.67 ms/frame`)**, targeting **25–40+ FPS** on this 2-vCPU VM.
- **Legacy Preservation**:
  - Do NOT modify SHA-256-pinned research files (`bench/quality_gotrack.py`, `bench/workbench_neural.py`, `bench/workbench_cli.py`, etc.). Existing `native/tests/pose_tests.cpp`, standalone Node tests (`tools/test_*.js`, `bench/test_*.js`), and core `bench/test_*.py` suites must continue to pass.

---

## Code Layout

Exclusive write ownership per milestone/track (concurrent workers MUST NOT edit files outside their owned scope):

```text
/home/antigravity/achitaan/visualizeIT/
├── PROJECT.md                        # Owner: Project Orchestrator
├── TEST_INFRA.md                     # Owner: E2E Testing Orchestrator
├── TEST_READY.md                     # Owner: E2E Testing Orchestrator
├── pyproject.toml                    # Owner: M1 (Python package & pytest config)
├── requirements-video.txt            # Owner: M1 (Pinned CPU video dependencies)
├── scripts/
│   └── setup_env.sh                  # Owner: M1 (Repeatable .cache/venv + native C++ + demo video bootstrap)
├── tools/
│   └── prototype_bootstrap.py        # Owner: M1 (Linux site-packages portability fix only)
├── native/                           # Owner: M1 (Preserves pose.h/pose.cpp/pose_tests.cpp; adds raster.h/raster.cpp/raster_tests.cpp)
│   ├── CMakeLists.txt
│   ├── include/visualizeit/
│   │   ├── pose.h                    # Existing — preserve ABI
│   │   └── raster.h                  # New — C++17 triangle rasterizer & ROI compositor ABI
│   ├── src/
│   │   ├── pose.cpp                  # Existing — preserve
│   │   └── raster.cpp                # New — AVX-512/-O3 z-buffer perspective UV rasterizer
│   └── tests/
│       ├── pose_tests.cpp            # Existing — preserve
│       └── raster_tests.cpp          # New — C++ unit & parity tests
├── visualizeit/
│   ├── __init__.py                   # Owner: M1
│   ├── contracts.py                  # Owner: M1 (Typed schemas, configs, VideoRunReport)
│   ├── designs.py                    # Owner: M1 (Procedural recipes, prompt parser, RGBA image loader)
│   ├── native_raster.py              # Owner: M1 (ctypes loader + fallback for libvisualizeit_pose.so)
│   ├── demo_assets.py                # Owner: M1 (Deterministic realistic MP4 & 3D PLY fixture generator)
│   ├── segmentation.py               # Owner: M2 (Fast GrabCut + point/box/polygon mask onboarding)
│   ├── tracker.py                    # Owner: M2 (Pyramidal KLT + ORB/LSH Homography/Cylinder/Grid/6-DoF + Re-Acquisition)
│   ├── compositor.py                 # Owner: M2 (Planar, Cylinder, Grid & 3D Mesh surface compositor + luminance blend)
│   ├── pipeline.py                   # Owner: M2 (Streaming VideoPipeline, render_stride=1, FFmpeg pipe, FPS telemetry)
│   ├── cli.py                        # Owner: M2 (M3 wires `serve` subcommand to `visualizeit.server`)
│   ├── __main__.py                   # Owner: M2 (`python -m visualizeit` entrypoint)
│   ├── server.py                     # Owner: M3 (FastAPI HTTP/1.1 REST + Range media + WebSocket streaming server)
│   └── web/                          # Owner: M3 (Unified Design + Recorded Video + Live Camera Web Studio)
│       ├── index.html
│       ├── styles.css
│       ├── design-engine.js
│       ├── video-tracker-worker.js
│       ├── surface-renderer.js
│       └── app.js
├── README.md                         # Owner: M3 (Updated architecture & quickstart documentation)
└── tests/
    ├── unit/
    │   ├── test_m1_foundation.py     # Owner: M1
    │   ├── test_m2_video_engine.py   # Owner: M2
    │   └── test_m3_server_and_web.py # Owner: M3
    └── e2e/                          # Owner: E2E Testing Track (Tiers 1-4; Tier 5 adds adversarial tests in M4)
```

---

## Feature Inventory

Every feature discovered during Step 0 Survey (Explorers 1, 2, and 3) is assigned to an explicit implementation milestone (none unassigned):

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Repeatable Linux CPU Env & Legacy Compatibility | `uv`-provisioned `.cache/venv` (Python 3.11, `opencv-python-headless`, `numpy>=2.1`, `scipy`, `Pillow`, `fastapi`, `uvicorn`, `websockets`, `python-multipart`, `httpx`, `pytest`), `scripts/setup_env.sh`, Linux `site-packages` fix in `tools/prototype_bootstrap.py`, preserving existing C++/Node/Python tests | M1 | survey_1, survey_3 |
| F2 | Native C++17 AVX-512 Surface Rasterizer | `vi_rasterize_mesh_rgbd` in `native/` (`-O3 -march=native`): perspective-correct z-buffer triangle rasterization, bilinear RGBA repeat/decal sampling, diffuse/luminance shading, and ROI alpha compositing in `~2–4 ms/frame` (`50–150x` faster than `bench/renderer.py`) | M1 | survey_2 |
| F3 | Procedural Design Engine & Prompt Parser | Procedural 2D artwork generation (`solid`, `stripes`, `checker`, `dots`, `waves`, `grid`), 2–5 color palettes, custom text overlay, scale/rotation/seed, and natural-language prompt parsing (Python + JS parity) | M1 | survey_1, survey_2 |
| F4 | Custom RGBA Artwork Upload & Loader | Load/upload PNG, JPEG, and WebP custom artwork preserving alpha transparency, aspect ratio, and validation against corrupt/non-image inputs | M1 | survey_1 |
| F5 | Realistic Multi-Scenario Demo Video & 3D Corpus | Deterministic generator (`demo-assets`) creating `>=30 FPS` H.264 `.mp4` clips (`planar_card_640x480`, `planar_hd_1280x720`, `cylinder_mug_640x480`, `occlusion_recovery_640x480`, and `mesh_box_640x480` + `.ply` + ground-truth poses) | M1 | survey_1, survey_3 |
| F6 | Fast Point & Box Object Segmentation Onboarding | Interactive positive/negative click points and bounding-box segmentation on any frame (`GrabCut` + edge-aware morphological feathering + interior feature extraction in `< 80 ms` warm on CPU, one-time onboarding) | M2 | survey_1, survey_2 |
| F7 | Real-Time KLT + ORB Homography & Grid Video Tracker | Causal per-frame `360p/480p` `resize_first` pyramidal Lucas-Kanade optical flow (forward-backward gated) + compact ORB keyframe matching + RANSAC 8-DoF Homography / Cylindrical / Deformable Grid estimation + temporal smoothing (`<= 12 ms/frame`) | M2 | survey_1, survey_2 |
| F8 | Calibrated 6-DoF 3D Mesh Fast Tracker | Integration of compact LSH/ORB descriptor bank (`<= 1500` descriptors) + expanding ROI search + KLT-first `solvePnPRansac` + `solvePnPRefineLM` (`mesh_6dof` mode) on `.ply` and parametric 3D surfaces (`plane`, `cylinder`, `box`, `curved_panel`) | M2 | survey_2, survey_3 |
| F9 | Occlusion Detection, Suppression & Auto Re-Acquisition | Automatic detection of object exit/occlusion (`state="lost"` or `"recovering"`), immediate suppression of stale overlays (`inliers < min_inliers`), and automatic keyframe descriptor re-acquisition upon object re-entry | M2 | survey_1, survey_2, survey_3 |
| F10 | Multi-Mode Surface Wrapping & Luminance Blending | Vectorized/native rendering across `planar`, `cylinder` (360° bottle/mug UV wrap), `mesh_grid`, and `mesh_6dof` modes with `luminance` (scene shading preservation), `normal`, and `ink` blend modes + placement transforms (`scale`, `rotation_deg`, `offset_x`, `offset_y`, `opacity`) | M2 | survey_1, survey_2 |
| F11 | Streaming >=15 FPS Video CLI (`process`, `info`, `segment`) | Headless `python -m visualizeit` CLI supporting `.mp4`/`.webm`/`.mov`/`.avi` probing, segmentation, and full-frame-count (`render_stride=1`) H.264 video processing via streaming `cv2.VideoCapture` + `ffmpeg libx264 -preset ultrafast` | M2 | survey_1, survey_2 |
| F12 | Honest >=15 FPS Benchmark & Telemetry Gate (`benchmark`) | `python -m visualizeit benchmark` measuring end-to-end wall-clock & steady-state FPS, P50/P95/P99 latency, per-stage breakdown, frame-count equality (`input == processed == rendered == output`), tracking availability, and attachment error | M2 | survey_1, survey_2, survey_3 |
| F13 | FastAPI HTTP/1.1 Video, Design & Render REST API | Loopback/configurable `HTTP/1.1` keep-alive FastAPI server (`visualizeit/server.py`) with video upload, HTTP `Range` media streaming, frame extraction, interactive selection, design generation/upload, MP4 batch render, and strict input validation | M3 | survey_1, survey_2, survey_3 |
| F14 | Real-Time WebSocket Video Processing Stream | `/ws/videos/{id}/stream` streaming live processed frames + per-frame tracking/FPS telemetry JSON at `>= 15 FPS` over persistent WebSocket without HTTP/1.0 polling overhead | M3 | survey_1, survey_2 |
| F15 | Unified Browser Video Studio (>=15 FPS Playback & MP4 Export) | Single-page web app (`visualizeit/web/`) unifying Design Studio + Recorded Video File Mode (demo or uploaded video, click/box object selection, uncapped `>= 15–30+ FPS` Web Worker KLT tracker + WebGL2/Canvas2D surface renderer, live placement sliders, and MP4 export) | M3 | survey_1, survey_2 |
| F16 | Uncapped Live Camera Mode & Telemetry HUD | Browser `getUserMedia` camera mode sharing the Web Worker tracker & surface renderer with zero `66 ms` throttle (`>= 25–30+ FPS` capable), auto re-acquisition, and live HUD (`window.__VISUALIZEIT_TELEMETRY__`) | M3 | survey_1, survey_2 |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Foundation, Env, Native C++17 Rasterizer & Demo Corpus | Provision `.cache/venv` & `scripts/setup_env.sh`; fix `tools/prototype_bootstrap.py` Linux path; implement `native/` C++17 `vi_rasterize_mesh_rgbd` + `ctest`; implement `visualizeit/{contracts,designs,native_raster,demo_assets}.py` + `tests/unit/test_m1_foundation.py` | none | IN_PROGRESS (`2a795cde954fa3a09d6456af5bd83054`) |
| M2 | High-Speed >=15 FPS Video Engine & CLI | Implement `visualizeit/{segmentation,tracker,compositor,pipeline,cli,__main__}.py` + `tests/unit/test_m2_video_engine.py`; achieve sustained `>= 15.0 FPS` (target `25–40+ FPS`) end-to-end on 640×480 & 1280×720 MP4s (`render_stride=1`) across planar, cylinder, grid, occlusion, and 6-DoF mesh modes | M1 | PLANNED |
| M3 | FastAPI Backend, WebSocket Stream & Unified Web Studio | Implement `visualizeit/server.py` + `visualizeit/web/*` + wire `cli.py serve` + update `README.md` + `tests/unit/test_m3_server_and_web.py` (including headless `/usr/bin/chromium` >=15 FPS browser video playback verification) | M2 | PLANNED |
| E2E_Track | Opaque-Box 4-Tier E2E Test Suite | Design `TEST_INFRA.md`, build comprehensive opaque-box CLI/REST/WebSocket/Browser E2E tests in `tests/e2e/` covering F1–F16 across Tiers 1–4, and publish `TEST_READY.md` | none (uses `.cache/venv` read-only after M1; zero code overlap) | IN_PROGRESS (`7f581f83ceffea9c8bff65df1c5e096d`) |
| M4 | Final Milestone: E2E Pass (Tiers 1–4) & Adversarial Hardening (Tier 5) | Phase 1: Pass 100% of E2E tests sequentially across Tiers 1 -> 2 -> 3 -> 4 + all legacy suites. Phase 2: Tier-5 white-box adversarial Challenger -> Worker -> Reviewer + whole-project Forensic Audit verifying genuine `>= 15 FPS` video performance | M3, E2E_Track | PLANNED |

---

## Interface Contracts

### 1. Environment & Execution Paths
- **Python Interpreter**: `/home/antigravity/achitaan/visualizeIT/.cache/venv/bin/python`
- **Setup Script**: `bash /home/antigravity/achitaan/visualizeIT/scripts/setup_env.sh` (idempotent; exits `0`)
- **Native Shared Library**: `/home/antigravity/achitaan/visualizeIT/.cache/native-build/libvisualizeit_pose.so`
- **Demo Video Directory**: `/home/antigravity/achitaan/visualizeIT/.cache/demo-videos`

### 2. Native C++17 Rasterizer ABI (`native/include/visualizeit/raster.h` <-> `visualizeit/native_raster.py`)
Preserves existing `vi_cv_gltf_pose_to_unity` in `pose.h` and adds `extern "C"`:
```c
int vi_rasterize_mesh_rgbd(
    const float* vertices_cam,     // (V * 3) camera-space XYZ (z > 0 in front)
    int num_vertices,
    const int32_t* triangles,      // (T * 3) vertex indices
    int num_triangles,
    const float* vertex_uvs,       // (V * 2) UV coordinates
    const uint8_t* texture_rgba,   // (tex_h * tex_w * 4) RGBA design texture
    int tex_w,
    int tex_h,
    int wrap_repeat,               // 1 = tile/repeat UV, 0 = clamp/decal [0,1]
    float fx, float fy, float cx, float cy,
    float opacity,                 // [0.0, 1.0]
    int blend_mode,                // 0 = normal, 1 = luminance_shading, 2 = ink
    uint8_t* frame_rgb_inout,      // (frame_h * frame_w * 3) RGB uint8 buffer mutated in place
    float* depth_buf_inout,        // (frame_h * frame_w) float32 z-buffer mutated in place
    uint8_t* mask_out,             // (frame_h * frame_w) uint8 rendered pixel mask (optional, may be NULL)
    int frame_w,
    int frame_h
);
```
Returns `>= 0` (number of rendered pixels) on success, negative error code on invalid arguments/null pointers/non-positive dimensions.

### 3. Core Python Contracts (`visualizeit/contracts.py`)
- **`DesignSpec`**:
  - `source: Literal["recipe", "image"]`
  - `pattern: Literal["solid", "stripes", "checker", "dots", "waves", "grid"] = "stripes"`
  - `colors: list[str] = ["#ff595e", "#1982c4", "#ffca3a"]`
  - `text: str = ""`
  - `prompt: str = ""`
  - `image_path: str | None = None`
  - `width: int = 512`, `height: int = 512`, `seed: int = 42`
- **`PlacementSpec`**:
  - `wrap_mode: Literal["planar", "cylinder", "mesh_grid", "mesh_6dof"] = "planar"`
  - `blend_mode: Literal["luminance", "normal", "ink"] = "luminance"`
  - `opacity: float = 0.90` (`0.0 <= opacity <= 1.0`)
  - `scale: float = 1.0` (`0.1 <= scale <= 5.0`)
  - `rotation_deg: float = 0.0`
  - `offset_x: float = 0.0`, `offset_y: float = 0.0`
  - `cylinder_radius_frac: float = 0.5`
  - `tile_repeat: bool = False`
- **`SelectionSpec`**:
  - `frame_index: int = 0`
  - `positive_points: list[tuple[float, float]] = []` (pixel coordinates `(x, y)`)
  - `negative_points: list[tuple[float, float]] = []`
  - `bbox: tuple[float, float, float, float] | None = None` (`(x, y, w, h)` in pixels)
- **`TrackFrameResult`**:
  - `frame_index: int`
  - `timestamp_s: float`
  - `state: Literal["tracking", "recovering", "lost"]`
  - `homography_3x3: np.ndarray | None` (strictly `None` when `state != "tracking"`)
  - `pose_rvec: np.ndarray | None`, `pose_tvec: np.ndarray | None` (for `mesh_6dof`)
  - `grid_src_pts: np.ndarray | None`, `grid_dst_pts: np.ndarray | None` (for `mesh_grid` / `cylinder`)
  - `inliers: int`
  - `confidence: float`
  - `track_ms: float`
- **`VideoRunReport`** (serializable via `.to_dict()` to exact JSON keys below):
  - `input_path: str`, `output_path: str | None`
  - `width: int`, `height: int`, `source_fps: float`
  - `wrap_mode: str`, `blend_mode: str`, `raster_backend: str`
  - `render_stride: int` (**MUST always equal `1`**)
  - `input_frame_count: int`, `processed_frame_count: int`, `rendered_frame_count: int`, `output_frame_count: int` (**all four MUST be equal when writing output video**)
  - `tracked_frame_count: int`, `lost_frame_count: int`, `recovered_count: int`, `tracking_availability: float`
  - `wall_time_seconds: float`, `effective_fps: float`, `steady_state_fps: float`
  - `frame_latency_ms: dict[str, float]` (`mean`, `p50`, `p95`, `p99`, `max`)
  - `stage_latency_mean_ms: dict[str, float]` (`decode_ms`, `prep_ms`, `track_ms`, `composite_ms`, `encode_ms`)
  - `inlier_count_median: float`
  - `reprojection_rmse_px: float | None`

### 4. CLI Contract (`python -m visualizeit <subcommand>`)
Exit codes: `0` = success, `1` = runtime/benchmark gate failure, `2` = invalid input/usage error (clean error message to `stderr`, never uncaught stack trace on bad user inputs).
- **`demo-assets`**:
  `python -m visualizeit demo-assets [--output-dir .cache/demo-videos] [--json]`
  Generates `planar_card_640x480.mp4`, `planar_hd_1280x720.mp4`, `cylinder_mug_640x480.mp4`, `occlusion_recovery_640x480.mp4`, and `mesh_box_640x480.mp4` (+ `mesh_box.ply` & `mesh_box_meta.json`). With `--json`, prints `{ "clips": [ { "id", "path", "width", "height", "fps", "frame_count", "default_selection", "recommended_wrap" } ] }`.
- **`info`**:
  `python -m visualizeit info --input <path> [--json]`
  Outputs `{ "path", "width", "height", "fps", "frame_count", "duration_s", "codec" }`. Exits `2` if file missing/empty/not a valid video.
- **`segment`**:
  `python -m visualizeit segment --input <path> [--frame 0] [--point X,Y ...] [--neg-point X,Y ...] [--box X,Y,W,H] [--output-mask <png_path>] [--json]`
  Outputs `{ "frame_index", "width", "height", "foreground_pixels", "coverage_ratio", "bbox", "polygon", "anchor_count", "init_ms", "mask_path" }`. Exits `2` on out-of-range frame or invalid coordinates/box.
- **`process`**:
  `python -m visualizeit process --input <path> --output <mp4_path> [--report <json_path>] [--prompt <text>] [--pattern {solid,stripes,checker,dots,waves,grid}] [--colors C1,C2,...] [--text <str>] [--design-image <img_path>] [--wrap-mode {planar,cylinder,mesh_grid,mesh_6dof}] [--blend-mode {luminance,normal,ink}] [--opacity <float>] [--scale <float>] [--rotation-deg <float>] [--offset-x <float>] [--offset-y <float>] [--select-frame <int>] [--point X,Y ...] [--neg-point X,Y ...] [--box X,Y,W,H] [--mesh <ply_or_shape>] [--intrinsics fx,fy,cx,cy] [--tracking-height {360,480}] [--threads <int>] [--raster-backend {native_cpp,opencv}] [--json]`
  Processes 100% of frames (`render_stride=1`), writes valid `.mp4` to `--output`, writes `VideoRunReport` JSON to `--report` (if provided), and prints JSON to `stdout` when `--json` is set.
- **`benchmark`**:
  `python -m visualizeit benchmark [--input <path>] [--preset {quick,standard,hd,cylinder,occlusion,mesh_6dof,all}] [--min-fps 15.0] [--output-dir <dir>] [--report <json_path>] [--json]`
  Runs end-to-end video processing (`decode + track + composite + encode`), asserts `effective_fps >= min_fps` and `input_frame_count == processed_frame_count == rendered_frame_count == output_frame_count`, writes report JSON, and exits `0` on pass or `1` if `effective_fps < min_fps`.
- **`serve`**:
  `python -m visualizeit serve [--host 127.0.0.1] [--port 8790] [--storage-dir .cache/studio-runtime]`
  Starts FastAPI + WebSocket + Unified Web Studio server.

### 5. HTTP REST & WebSocket Contract (`visualizeit/server.py`)
- `GET /` -> `200 text/html` (Unified Web Studio)
- `GET /api/health` -> `200 application/json` (`{ "status": "ok", "version": "1.0.0", "native_raster_available": bool, "demo_clips_ready": bool }`)
- `GET /api/demos` -> `200 application/json` (`{ "clips": [...] }`)
- `GET /api/designs/presets` -> `200 application/json` (`{ "presets": [...] }`)
- `POST /api/designs/render` -> JSON `{ "pattern", "colors", "text", "prompt", "width", "height" }` -> `200` `{ "design_id", "image_url", "recipe" }` (`400`/`422` on invalid params)
- `POST /api/designs/upload` -> `multipart/form-data` field `file` -> `200` `{ "design_id", "image_url", "width", "height", "has_alpha" }` (`400`/`415`/`422` on non-image/corrupt upload)
- `POST /api/videos/upload` -> `multipart/form-data` field `file` -> `200` `{ "video_id", "width", "height", "fps", "frame_count", "duration_s", "media_url" }` (`400`/`415`/`422` on corrupt/non-video upload)
- `GET /api/videos/{video_id}` -> `200` metadata (`404` if not found)
- `GET /api/videos/{video_id}/media` -> `200` or `206 Partial Content` (`Accept-Ranges: bytes`, video bytes)
- `GET /api/videos/{video_id}/frame/{frame_index}` -> `200 image/jpeg` (`400`/`404` if out of bounds)
- `POST /api/videos/{video_id}/select` -> JSON `{ "frame_index": 0, "positive_points": [[x, y]], "negative_points": [], "bbox": [x, y, w, h] | null }` -> `200` `{ "selection_id", "frame_index", "foreground_pixels", "coverage_ratio", "bbox", "polygon", "anchor_count", "init_ms", "mask_png_base64" }`
- `POST /api/videos/{video_id}/render` -> JSON `{ "selection_id"?: str, "selection"?: dict, "design_id"?: str, "design_recipe"?: dict, "placement"?: dict }` -> `200` `{ "render_id", "status": "completed", "video_url": "/api/renders/{render_id}/video.mp4", "report_url": "/api/renders/{render_id}/report.json", "report": VideoRunReport }`
- `GET /api/renders/{render_id}/video.mp4` -> `200`/`206` `video/mp4`
- `GET /api/renders/{render_id}/report.json` -> `200 application/json` (`VideoRunReport`)
- `WS /ws/videos/{video_id}/stream` -> Client sends JSON `{ "selection"?: dict, "selection_id"?: str, "design_id"?: str, "design_recipe"?: dict, "placement"?: dict, "max_frames"?: int }`; server sends per-frame JSON `{ "type": "frame", "frame_index": int, "state": str, "fps": float, "latency_ms": float, "inliers": int, "jpeg_base64": str }` (or binary frame following metadata) and finishes with `{ "type": "complete", "report": VideoRunReport }`.

### 6. Browser Web Studio DOM & Telemetry Contract (`visualizeit/web/`)
Required `data-testid` attributes and global read-only telemetry object:
- Mode selectors: `mode-video`, `mode-camera`
- Video controls: `demo-clip-select`, `load-demo-btn`, `video-file-input`, `play-pause-btn`, `export-mp4-btn`, `export-download-link`
- Selection controls: `stage-canvas`, `select-mode-point`, `select-mode-box`, `auto-select-btn`, `clear-selection-btn`
- Design & placement controls: `design-prompt-input`, `apply-prompt-btn`, `pattern-select`, `design-image-input`, `wrap-mode-select`, `blend-mode-select`, `opacity-slider`, `scale-slider`, `rotation-slider`
- Telemetry HUD: `hud-fps`, `hud-latency-ms`, `hud-tracking-state`, `hud-processed-frames`
- Global telemetry inspection hook: `window.__VISUALIZEIT_TELEMETRY__` returning `{ mode, playing, currentFps, medianFrameMs, p95FrameMs, processedFrames, trackingState, inliers, wrapMode, blendMode, lastExportReport }`.
