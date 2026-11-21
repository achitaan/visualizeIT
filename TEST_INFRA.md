# E2E Test Infra: visualizeIT

## Test Philosophy

The `visualizeIT` end-to-end (E2E) test suite (`tests/e2e/`) is a **requirement-driven, strictly opaque-box 4-tier verification suite** built directly from `PROJECT.md` (`## Feature Inventory` `F1`–`F16` and `## Interface Contracts` §1–§6) and `ORIGINAL_REQUEST.md` (*"get it working at 15 fps at least for the video version"*).

Core principles:
1. **Strictly Opaque-Box Boundaries**:
   - No file in `tests/e2e/` imports `visualizeit.*` internal modules at collection or module scope.
   - All tests interact with the system solely as an external user/caller across documented public boundaries:
     - Environment bootstrap script (`bash scripts/setup_env.sh`) and `.cache/venv/bin/python`
     - Native C++17 `extern "C"` shared library ABI (`.cache/native-build/libvisualizeit_pose.so` via `ctypes`: `vi_pose_abi_version`, `vi_cv_gltf_pose_to_unity`, `vi_rasterize_mesh_rgbd`)
     - CLI subprocess invocations (`python -m visualizeit {demo-assets, info, segment, process, benchmark, serve}`)
     - Live HTTP/1.1 Keep-Alive REST & `Range` requests against a real spawned `visualizeit serve` process bound to an ephemeral `127.0.0.1:<port>`
     - Live WebSocket frame + telemetry streaming (`ws://127.0.0.1:<port>/ws/videos/{id}/stream`)
     - Real headless `/usr/bin/chromium` (`156.x`) automation over Chrome DevTools Protocol (CDP) exercising documented `[data-testid="..."]` controls, `<input type="file">` uploads, MP4 downloads, fake/synthetic `getUserMedia` camera feeds, and `window.__VISUALIZEIT_TELEMETRY__`
     - Independent media & pixel verification (`ffprobe`, OpenCV `cv2.VideoCapture`, Pillow `Image`, Moller-Trumbore ray-triangle UV math, and SHA-256 integrity checks).
2. **Honest `>= 15.0 FPS` & Anti-Cheating Enforcement**:
   - Every video processing & benchmark workload asserts:
     - `render_stride == 1`
     - `input_frame_count == processed_frame_count == rendered_frame_count == output_frame_count` (cross-checked independently via `ffprobe -count_frames`)
     - End-to-end throughput `effective_fps >= 15.0` (`<= 66.67 ms/frame`), cross-checked against an external `time.perf_counter()` wall-clock timer
     - Genuine non-trivial RGB pixel delta inside the tracked target mask (`mean_abs_diff >= 10.0`), strict background preservation outside the object ROI (`mean_abs_diff <= 3.5` H.264 quantization tolerance), and strict overlay suppression (`mean_abs_diff <= 2.5`) during occluded/lost frames (`state != "tracking"`).
3. **Progressive Testability & Clean Fast Failure**:
   - `pytest --collect-only tests/e2e` succeeds in `< 1.0 s` with zero collection errors even before Milestones M1–M3 are complete.
   - Every subprocess, HTTP request, WebSocket read, and CDP command enforces explicit finite timeouts (`5s`–`120s`), and server/browser launchers check `proc.poll()` every `50–100 ms` during startup so unimplemented commands fail cleanly in `< 0.25 s` rather than hanging.
   - Invalid CLI inputs must exit with code `2` and print a clean diagnostic to `stderr` with zero `Traceback (most recent call last)`.
4. **Strict Filesystem & Process Hygiene (Zero `/tmp` Leakage, Zero Orphans/Zombies)**:
   - All temporary files, synthetic clips, server storage dirs, `TMPDIR`, `pytest` base temp dirs, and Chromium `--user-data-dir`/`--disk-cache-dir` reside strictly on XFS under `/home/antigravity/achitaan/visualizeIT/.cache/tmp/` (never `/tmp`, never `.agents/teamwork/`, never `virtiofs`).
   - Chromium sessions allocate short XFS directories (`tempfile.mkdtemp(prefix="c", dir=".cache/tmp")`, `<= 61` chars to satisfy Linux `sockaddr_un` `108`-byte `SingletonSocket` path limits) and launch with `--no-sandbox --no-zygote` + graceful CDP `Browser.close` + process-group cleanup (`os.killpg`) so zero orphan or `<defunct>` zombie processes remain.

---

## Feature Inventory

All **16 features (`F1`–`F16`)** from `PROJECT.md § Feature Inventory` are mapped across Tiers 1–4 with zero unassigned features:

| ID | Feature | Milestone | Primary Opaque-Box Channels | Tier 1 (`-m tier1`) | Tier 2 (`-m tier2`) | Tier 3 (`-m tier3`) | Tier 4 (`-m tier4`) |
|----|---------|-----------|-----------------------------|---------------------|---------------------|---------------------|---------------------|
| **F1** | Repeatable Linux CPU Env & Legacy Compatibility | M1 | `scripts/setup_env.sh`, `.cache/venv/bin/python`, `tools/prototype_bootstrap.py`, `ctypes` `pose.h`, `node`, `unittest` | 6 tests | 6 tests | `T3-01`, `T3-10` | `T4-08` (`T4-10`) |
| **F2** | Native C++17 AVX-512 Surface Rasterizer | M1 | `ctypes` `libvisualizeit_pose.so` (`vi_rasterize_mesh_rgbd`), CLI `--raster-backend native_cpp` | 6 tests | 6 tests | `T3-01`, `T3-04`, `T3-08`, `T3-10` | `T4-01`, `T4-04`, `T4-08` |
| **F3** | Procedural Design Engine & Prompt Parser | M1 | CLI `--pattern`/`--colors`/`--text`/`--prompt`, REST `POST /api/designs/render`, Browser prompt/pattern UI | 6 tests | 6 tests | `T3-02`, `T3-03`, `T3-08`, `T3-11`, `T3-15`, `T3-17`, `T3-19` | `T4-01`, `T4-03`, `T4-06`, `T4-09` |
| **F4** | Custom RGBA Artwork Upload & Loader | M1 | CLI `--design-image`, REST `POST /api/designs/upload`, Browser `design-image-input` | 6 tests | 6 tests | `T3-04`, `T3-05`, `T3-12`, `T3-14`, `T3-18`, `T3-20` | `T4-02`, `T4-05`, `T4-07` |
| **F5** | Realistic Multi-Scenario Demo Video & 3D Corpus | M1 | CLI `demo-assets --json`, `ffprobe`, `.ply` & `mesh_box_meta.json` | 6 tests | 6 tests | `T3-02`..`T3-09`, `T3-13`, `T3-17` | `T4-01`..`T4-07` |
| **F6** | Fast Point & Box Object Segmentation Onboarding | M2 | CLI `segment` (`--point`, `--neg-point`, `--box`, `--output-mask`), REST `POST /api/videos/{id}/select`, Browser canvas selection | 6 tests | 6 tests | `T3-03`, `T3-06`, `T3-11`, `T3-12`, `T3-18` | `T4-01`, `T4-02`, `T4-05`, `T4-06` |
| **F7** | Real-Time KLT + ORB Homography & Grid Video Tracker | M2 | CLI `process` (`planar`, `cylinder`, `mesh_grid`, `--tracking-height`), REST/WS/Browser tracking telemetry | 6 tests | 6 tests | `T3-02`, `T3-03`, `T3-05`, `T3-07`, `T3-13` | `T4-01`, `T4-02`, `T4-05`, `T4-09` |
| **F8** | Calibrated 6-DoF 3D Mesh Fast Tracker | M2 | CLI `process --wrap-mode mesh_6dof --mesh <ply|shape> --intrinsics`, `benchmark --preset mesh_6dof`, WS `mesh_6dof` | 6 tests | 6 tests | `T3-01`, `T3-04`, `T3-08`, `T3-14` | `T4-04` |
| **F9** | Occlusion Detection, Suppression & Auto Re-Acquisition | M2 | CLI `process` & `benchmark --preset occlusion`, WS `state` stream, Browser camera occlusion recovery | 6 tests | 6 tests | `T3-06`, `T3-09`, `T3-15`, `T3-20` | `T4-03`, `T4-07` |
| **F10** | Multi-Mode Surface Wrapping & Luminance Blending | M2 | CLI/REST/WS/Browser `wrap_mode` (`planar`,`cylinder`,`mesh_grid`,`mesh_6dof`), `blend_mode` (`luminance`,`normal`,`ink`), placement transforms | 6 tests | 6 tests | `T3-01`..`T3-08`, `T3-11`..`T3-20` | `T4-01`..`T4-07`, `T4-09` |
| **F11** | Streaming >=15 FPS Video CLI (`process`, `info`, `segment`) | M2 | `python -m visualizeit {info, segment, process}` across `.mp4`/`.webm`/`.mov`/`.avi` | 6 tests | 6 tests | `T3-01`..`T3-09` | `T4-01`..`T4-04`, `T4-09` |
| **F12** | Honest >=15 FPS Benchmark & Telemetry Gate (`benchmark`) | M2 | `python -m visualizeit benchmark` presets, `--min-fps` exit `0`/`1`, `VideoRunReport` telemetry | 6 tests | 6 tests | `T3-07`, `T3-09`, `T3-10` | `T4-01`..`T4-04`, `T4-10` |
| **F13** | FastAPI HTTP/1.1 Video, Design & Render REST API | M3 | `visualizeit serve`, HTTP/1.1 Keep-Alive, `Range` (`206`), upload, frame JPEG, select, batch MP4 render | 6 tests | 6 tests | `T3-11`..`T3-16`, `T3-18` | `T4-05`, `T4-09`, `T4-10` |
| **F14** | Real-Time WebSocket Video Processing Stream | M3 | `WS /ws/videos/{id}/stream` per-frame JPEGs + telemetry + `complete` report | 6 tests | 6 tests | `T3-13`..`T3-16` | `T4-05`, `T4-09` |
| **F15** | Unified Browser Video Studio (>=15 FPS Playback & MP4 Export) | M3 | Headless `/usr/bin/chromium` CDP against `GET /`, `data-testid` controls, `window.__VISUALIZEIT_TELEMETRY__`, MP4 download | 6 tests | 6 tests | `T3-17`, `T3-18`, `T3-19` | `T4-06`, `T4-09` |
| **F16** | Uncapped Live Camera Mode & Telemetry HUD | M3 | Headless `/usr/bin/chromium` fake/Y4M `getUserMedia` camera mode, HUD DOM, `window.__VISUALIZEIT_TELEMETRY__` | 6 tests | 6 tests | `T3-19`, `T3-20` | `T4-07` |

---

## Test Architecture

### Test Runner Commands

Full E2E test suite execution (using the read-only `.cache/venv` provisioned by Milestone 1):
```bash
/home/antigravity/achitaan/visualizeIT/.cache/venv/bin/python -m pytest tests/e2e -v
```

Sequential tier-by-tier execution (used by Final Milestone M4 Phase 1; markers registered in `tests/e2e/conftest.py` with zero unknown-marker warnings):
```bash
# Tier 1: Feature Happy-Path Coverage (96 tests)
/home/antigravity/achitaan/visualizeIT/.cache/venv/bin/python -m pytest tests/e2e -m tier1 -v
# or by directory:
/home/antigravity/achitaan/visualizeIT/.cache/venv/bin/python -m pytest tests/e2e/tier1_features -v

# Tier 2: Boundary, Corner-Case & Negative Error-Handling Coverage (96 tests)
/home/antigravity/achitaan/visualizeIT/.cache/venv/bin/python -m pytest tests/e2e -m tier2 -v
# or by directory:
/home/antigravity/achitaan/visualizeIT/.cache/venv/bin/python -m pytest tests/e2e/tier2_boundaries -v

# Tier 3: Cross-Feature Pairwise & Multi-Surface Combinations (20 tests)
/home/antigravity/achitaan/visualizeIT/.cache/venv/bin/python -m pytest tests/e2e -m tier3 -v
# or by directory:
/home/antigravity/achitaan/visualizeIT/.cache/venv/bin/python -m pytest tests/e2e/tier3_combinations -v

# Tier 4: Real-World End-to-End Application Workloads & >=15 FPS Gates (10 tests)
/home/antigravity/achitaan/visualizeIT/.cache/venv/bin/python -m pytest tests/e2e -m tier4 -v
# or by directory:
/home/antigravity/achitaan/visualizeIT/.cache/venv/bin/python -m pytest tests/e2e/tier4_workloads -v

# Syntax & Collection Check:
/home/antigravity/achitaan/visualizeIT/.cache/venv/bin/python -m pytest tests/e2e --collect-only -q
```

### Directory & Module Layout

```text
tests/e2e/
├── __init__.py
├── conftest.py                                         # Registers tier1..tier4 & f01..f16 markers, enforces XFS TMPDIR, shared fixtures
├── _helpers/
│   ├── __init__.py
│   ├── paths_and_proc.py                               # REPO_ROOT, VENV_PYTHON, E2E_TMP_ROOT, run_cli(), assert_cli_usage_error(), spawn_live_server()
│   ├── native_abi.py                                   # ctypes loader & callers for vi_pose_abi_version, vi_cv_gltf_pose_to_unity, vi_rasterize_mesh_rgbd
│   ├── media_and_oracle.py                             # make_textured_tracking_clip(), make_occlusion_clip(), make_y4m_camera_clip(), probe_video(), decode_frames(), compute_roi_outside_delta(), make_rgba_png(), ray_triangle_uv_oracle()
│   ├── http_and_ws.py                                  # StudioAPIClient (HTTP/1.1 keep-alive, Range 206, multipart upload, dual-mode WS /ws/videos/{id}/stream)
│   └── cdp_browser.py                                  # StdlibWebSocket + HeadlessChromiumSession (zero-dep CDP, <=61-char XFS TMPDIR, --no-zygote, Browser.close, file upload, MP4 download, telemetry polling)
├── tier1_features/
│   ├── __init__.py
│   ├── test_t1_f01_f04_foundation.py                   # F1 (6), F2 (6), F3 (6), F4 (6) = 24 tests
│   ├── test_t1_f05_f08_assets_and_tracking.py          # F5 (6), F6 (6), F7 (6), F8 (6) = 24 tests
│   ├── test_t1_f09_f12_compositor_and_cli.py           # F9 (6), F10 (6), F11 (6), F12 (6) = 24 tests
│   └── test_t1_f13_f16_server_and_studio.py            # F13 (6), F14 (6), F15 (6), F16 (6) = 24 tests
├── tier2_boundaries/
│   ├── __init__.py
│   ├── test_t2_f01_f04_foundation_bounds.py            # F1 (6), F2 (6), F3 (6), F4 (6) = 24 tests
│   ├── test_t2_f05_f08_assets_and_tracking_bounds.py   # F5 (6), F6 (6), F7 (6), F8 (6) = 24 tests
│   ├── test_t2_f09_f12_compositor_and_cli_bounds.py    # F9 (6), F10 (6), F11 (6), F12 (6) = 24 tests
│   └── test_t2_f13_f16_server_and_studio_bounds.py     # F13 (6), F14 (6), F15 (6), F16 (6) = 24 tests
├── tier3_combinations/
│   ├── __init__.py
│   ├── test_t3_cli_and_engine_combinations.py          # 10 multi-feature CLI/native/engine combination tests
│   └── test_t3_api_ws_and_browser_combinations.py      # 10 multi-feature REST/Range/WS/Browser Studio combination tests
└── tier4_workloads/
    ├── __init__.py
    ├── test_t4_video_pipeline_workloads.py             # 5 realistic end-to-end video pipeline workloads (HD >=180f, Cylinder, Occlusion, 6-DoF, Planar)
    └── test_t4_studio_live_and_legacy_workloads.py     # 5 realistic end-to-end server/WS, Browser Studio, Live Camera, Cross-Surface & Legacy workloads
```

---

## Real-World Application Scenarios (Tier 4)

Tier 4 (`tests/e2e/tier4_workloads/`, `-m tier4`) validates **10 full end-to-end realistic user workloads** (covering all 8 mandatory workloads plus 2 cross-surface & planar production gates):

1. **`test_t4_01_workload_full_hd_1280x720_180frame_video_branding_15fps`** *(Mandatory Workload 1)*:
   - End-to-end branding workflow on a Full HD `1280x720` **`>= 180`-frame** recorded H.264 video (`demo-assets` / 180-frame HD clip -> `info` -> `segment` -> `process` + `benchmark --min-fps 15.0`).
   - Verifies sustained **`effective_fps >= 15.0`**, `render_stride == 1`, `input_frame_count == processed_frame_count == rendered_frame_count == output_frame_count >= 180`, `ffprobe` H.264 `1280x720` stream integrity, strong foreground pixel delta inside tracked target, and untouched background pixels outside target ROI.
2. **`test_t4_02_workload_cylindrical_mug_360_label_wrap_luminance_15fps`** *(Mandatory Workload 2)*:
   - 360° cylindrical bottle/mug label wrap workflow (`cylinder_mug_640x480.mp4`) with custom die-cut RGBA artwork and `luminance` shading preservation at sustained **`>= 15.0 FPS`** (`render_stride == 1`, cylindrical foreshortening + highlight/shadow ratio preserved, alpha-0 cutouts showing raw mug surface).
3. **`test_t4_03_workload_multi_second_occlusion_suppression_and_reacquisition_15fps`** *(Mandatory Workload 3)*:
   - Multi-second object occlusion workflow (`occlusion_recovery_640x480.mp4` / >=90-frame multi-second occlusion clip) verifying strong pre-occlusion overlay, **immediate overlay suppression during occlusion** (`lost_frame_count >= 5`, zero stale ghost overlay pixels), and **automatic descriptor re-acquisition** upon object re-entry (`recovered_count >= 1`) at sustained **`>= 15.0 FPS`**.
4. **`test_t4_04_workload_calibrated_6dof_rigid_3d_mesh_surface_design_15fps`** *(Mandatory Workload 4)*:
   - Calibrated 6-DoF rigid 3D mesh (`mesh_6dof`) surface design workflow on `mesh_box_640x480.mp4` + `mesh_box.ply` + `mesh_box_meta.json` intrinsics with native C++17 z-buffer rasterizer, verifying geometric attachment accuracy (`reprojection_rmse_px < 5.0` and high overlap with ground-truth projected 3D silhouette) AND sustained **`>= 15.0 FPS`**.
5. **`test_t4_05_workload_live_websocket_streaming_video_session_15fps`** *(Mandatory Workload 5)*:
   - End-to-end live WebSocket streaming session (`POST /api/videos/upload` -> `POST /api/designs/upload` -> `POST /api/videos/{id}/select` -> `WS /ws/videos/{id}/stream`) delivering real-time composited JPEG frames + per-frame tracking/FPS telemetry in strict frame order at sustained **`>= 15.0 FPS`** (`complete.report.render_stride == 1`).
6. **`test_t4_06_workload_browser_unified_studio_interactive_playback_and_mp4_export`** *(Mandatory Workload 6)*:
   - Full Browser Unified Studio E2E workflow in headless `/usr/bin/chromium` via CDP: open Studio (`GET /`) -> load demo/uploaded video -> select object on `stage-canvas` -> customize design prompt, wrap mode, blend mode, and opacity -> play interactive video verifying `window.__VISUALIZEIT_TELEMETRY__.currentFps >= 15.0`, `medianFrameMs <= 66.67`, live HUD updates, and zero console/page errors -> click `export-mp4-btn` & download valid H.264 MP4 verified via `ffprobe` + pixel check.
7. **`test_t4_07_workload_browser_live_camera_uncapped_tracking_and_rendering_15fps`** *(Mandatory Workload 7)*:
   - Browser Live Camera Mode E2E workflow in headless `/usr/bin/chromium` (`--use-fake-ui-for-media-stream --use-fake-device-for-media-stream --use-file-for-fake-video-capture=<moving_card_30fps.y4m>`): activate `mode-camera` -> select moving camera target -> customize artwork -> verify uncapped live camera tracking & rendering (`window.__VISUALIZEIT_TELEMETRY__.mode == "camera"`, `playing == true`, `currentFps >= 15.0`, `processedFrames` advancing continuously, non-zero stage canvas overlay, zero console/page errors).
8. **`test_t4_08_workload_legacy_coexistence_cpp_node_python_and_sha256_regression`** *(Mandatory Workload 8)*:
   - Full legacy coexistence & non-regression check verifying native C++ `pose_tests` (`ctest`), all 9 standalone Node.js suites (`bench/test_*.js`, `tools/test_*.js`, `tools/profile_phone_tracker.js`), `tools/prototype_bootstrap.py` under Linux `-S`, standalone `tools/test_evaluate_tracking.py`, core legacy Python `unittest` suites (`tools.test_design_studio`, `tools.test_phone_static`, `bench.test_renderer`, `bench.test_workbench_neural`, etc.), and exact SHA-256 preservation of pinned research files (`bench/quality_gotrack.py`, `bench/quality_runner.py`).
9. **`test_t4_09_workload_cross_surface_parity_cli_rest_websocket_and_browser_export`**:
   - Processes the exact same video, object selection, procedural design recipe, and surface placement across all 4 product interfaces (CLI `process`, REST `POST /api/videos/{id}/render`, WebSocket `/ws/videos/{id}/stream`, and Browser Studio `export-mp4-btn`), asserting `render_stride == 1`, `>= 15.0 FPS`, exact frame count equality, and visual frame-level agreement across all 4 surfaces.
10. **`test_t4_10_workload_sustained_planar_640x480_studio_and_benchmark_gate`**:
    - Complete `640x480` planar production pipeline (`demo-assets` -> `info` -> `segment` -> `process` -> `benchmark --preset standard --min-fps 15.0` -> HTTP `Range` media streaming verification) verifying external wall-clock honesty, stage latency breakdown consistency, foreground artwork attachment, and background preservation.

---

## Coverage Thresholds

For $N = 16$ features (`F1`–`F16`):

| Tier | Pytest Marker | Subdirectory | Required Minimum | Implemented Target | Per-Feature Guarantee |
|------|---------------|--------------|------------------|--------------------|-----------------------|
| **Tier 1 — Feature Coverage** | `-m tier1` | `tests/e2e/tier1_features/` | `>= 80` (`>= 5 × 16`) | **96 tests** | `6` distinct `def test_*` functions for **each** of `F1`–`F16` |
| **Tier 2 — Boundary & Corner Cases** | `-m tier2` | `tests/e2e/tier2_boundaries/` | `>= 80` (`>= 5 × 16`) | **96 tests** | `6` distinct `def test_*` functions for **each** of `F1`–`F16` |
| **Tier 3 — Cross-Feature Combinations** | `-m tier3` | `tests/e2e/tier3_combinations/` | `>= 16` (`>= 1 × 16`) | **20 tests** | Every `F1`–`F16` participates in `>= 2` multi-feature tests |
| **Tier 4 — Real-World Workloads** | `-m tier4` | `tests/e2e/tier4_workloads/` | `>= 8` scenarios | **10 tests** | Covers all 8 mandatory workloads + cross-surface parity & planar gate |
| **Total Suite** | `tests/e2e` | `tests/e2e/` | **`>= 184` tests** | **`222` tests** | 100% genuine, non-trivial opaque-box `def test_*` functions |
