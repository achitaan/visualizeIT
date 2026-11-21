"""Tier 4 Real-World End-to-End Application Workloads & >=15.0 FPS Gates (Scenarios 5-9).

Covers:
- T4-05: Live WebSocket streaming video session at >= 15.0 FPS
- T4-06: Browser Unified Studio interactive >= 15.0 FPS playback & MP4 export
- T4-07: Browser Live Camera uncapped tracking & rendering at >= 15.0 FPS
- T4-08: Legacy coexistence (C++ ctest, 9 Node suites, prototype_bootstrap -S, Python unittest, SHA-256 pins)
- T4-09: Cross-surface parity across CLI, REST, WebSocket, and Browser Studio export

Zero imports of `visualizeit.*` or top-level non-stdlib third-party libraries (other than `pytest`).
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.e2e._helpers import (
    NATIVE_BUILD_DIR,
    REPO_ROOT,
    HeadlessChromiumSession,
    StudioAPIClient,
    assert_video_run_report_invariants,
    bbox_at_frame,
    compute_psnr,
    compute_roi_and_bg_delta,
    decode_frames,
    decode_image_bytes,
    get_venv_python,
    make_rgba_png,
    make_textured_tracking_clip,
    make_y4m_camera_clip,
    probe_video,
    run_cli_json,
    run_subproc,
)

pytestmark = pytest.mark.tier4

PINNED_RESEARCH_SHA256: dict[str, tuple[str, str]] = {
    # (workbench_neural.py Windows-CRLF pinned SHA-256, Linux-LF git checkout SHA-256)
    "bench/quality_gotrack.py": (
        "0b18516894690de0fbf3565d5d542ced17865c161e646369fa4f7e5412a28c29",
        "a07bdfd6df72ad41e861d790c64cd6f96b4eca27b2b1f34469cb7dd476969f38",
    ),
    "bench/quality_runner.py": (
        "bca306d1342dce2566c786fed94a7564ae090e11bf42243da2ea7bc9ce841ad1",
        "af0998eb7bb984723e660dca80543e41a6a042e341e1288184851dd06da151aa",
    ),
}


def _wait_for_stage_ready(
    browser: HeadlessChromiumSession,
    *,
    min_non_black: float = 0.12,
    timeout: float = 15.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_sample: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_sample = browser.sample_stage_canvas(timeout=5.0)
        if (
            int(last_sample.get("width", 0)) > 0
            and int(last_sample.get("height", 0)) > 0
            and float(last_sample.get("non_black_ratio", 0.0)) >= min_non_black
        ):
            return last_sample
        time.sleep(0.08)
    raise AssertionError(f"stage-canvas failed to render non-blank pixels within {timeout}s: {last_sample}")


def _ensure_playing(browser: HeadlessChromiumSession, *, timeout: float = 10.0) -> dict[str, Any]:
    tel = browser.get_telemetry(timeout=5.0)
    if not tel or not tel.get("playing"):
        browser.click_testid("play-pause-btn", timeout=timeout)
    return browser.wait_for_telemetry(lambda t: bool(t.get("playing")), timeout=timeout)


def _select_y4m_target_in_camera(
    browser: HeadlessChromiumSession,
    y4m_meta: dict[str, Any],
    *,
    timeout: float = 12.0,
) -> dict[str, Any]:
    px = float(y4m_meta["x0"]) + (float(y4m_meta["span_x"]) + float(y4m_meta["box_w"])) * 0.5
    py = float(y4m_meta["y0"]) + (float(y4m_meta["span_y"]) + float(y4m_meta["box_h"])) * 0.5
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sample = browser.sample_stage_canvas(timeout=5.0)
        if max(float(v) for v in (sample.get("std_rgb") or [0.0])) > 15.0:
            browser.click_testid("select-mode-point", timeout=5.0)
            browser.click_canvas_point(px, py, timeout=5.0)
            try:
                return browser.wait_for_telemetry(
                    lambda t: t.get("trackingState") == "tracking",
                    timeout=2.0,
                    poll_interval=0.08,
                )
            except TimeoutError:
                browser.click_testid("auto-select-btn", timeout=5.0)
                try:
                    return browser.wait_for_telemetry(
                        lambda t: t.get("trackingState") == "tracking",
                        timeout=2.0,
                        poll_interval=0.08,
                    )
                except TimeoutError:
                    pass
        time.sleep(0.12)
    return browser.wait_for_telemetry(lambda t: t.get("trackingState") == "tracking", timeout=3.0)


def _export_and_fetch_mp4(
    browser: HeadlessChromiumSession,
    client: StudioAPIClient,
    out_dir: Path,
    *,
    timeout: float = 60.0,
) -> tuple[Path, dict[str, Any]]:
    browser.click_testid("export-mp4-btn", timeout=10.0)
    tel = browser.wait_for_telemetry(
        lambda t: isinstance(t.get("lastExportReport"), dict),
        timeout=timeout,
        poll_interval=0.15,
    )
    report = dict(tel["lastExportReport"])

    link_deadline = time.monotonic() + 10.0
    link_info: dict[str, Any] = {}
    while time.monotonic() < link_deadline:
        link_info = browser.evaluate(
            """(() => {
                const el = document.querySelector('[data-testid="export-download-link"]');
                if (!el) return { exists: false };
                const rawHref = el.getAttribute("href") || "";
                return {
                    exists: true,
                    href: el.href || "",
                    rawHref,
                    hasDownload: el.hasAttribute("download")
                };
            })()""",
            await_promise=False,
            timeout=5.0,
        )
        if link_info.get("exists") and link_info.get("rawHref") and link_info.get("rawHref") != "#":
            break
        time.sleep(0.1)

    assert link_info.get("exists") and link_info.get("href") and link_info.get("rawHref") != "#", (
        f"export-download-link missing valid populated href: {link_info}"
    )

    existing_mp4s = [
        p
        for p in browser.download_dir.iterdir()
        if p.is_file() and p.name.lower().endswith(".mp4") and p.stat().st_size > 0
    ]
    if existing_mp4s:
        return max(existing_mp4s, key=lambda p: p.stat().st_mtime), report

    href = str(link_info["href"])
    if link_info.get("hasDownload"):
        browser.click_testid("export-download-link", timeout=10.0)
        return browser.wait_for_download(timeout=25.0, expected_suffix=".mp4"), report

    dst = out_dir / f"studio_export_{int(time.time() * 1000)}.mp4"
    if href.startswith("blob:"):
        b64_data = browser.evaluate(
            f"""(async () => {{
                const r = await fetch({json.dumps(href)});
                const buf = new Uint8Array(await r.arrayBuffer());
                let bin = "";
                for (let i = 0; i < buf.length; i += 8192) {{
                    bin += String.fromCharCode(...buf.subarray(i, i + 8192));
                }}
                return btoa(bin);
            }})()""",
            await_promise=True,
            timeout=25.0,
        )
        dst.write_bytes(base64.b64decode(b64_data))
        return dst, report

    parsed = urllib.parse.urlsplit(href)
    req_path = f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path
    dl_resp = client.get(req_path, expected_status=200, timeout=30.0)
    dst.write_bytes(dl_resp["body"])
    return dst, report


# ==============================================================================
# Scenario 5 — Live WebSocket Streaming Video Session (>= 15.0 FPS)
# ==============================================================================


@pytest.mark.f04
@pytest.mark.f05
@pytest.mark.f06
@pytest.mark.f07
@pytest.mark.f10
@pytest.mark.f13
@pytest.mark.f14
def test_t4_05_workload_live_websocket_streaming_video_session_15fps(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """End-to-end live WebSocket video streaming session delivering all frames in order at >= 15.0 FPS."""
    live_studio_server.close()
    port_idx = len(live_studio_server.client_ports_seen)

    health = live_studio_server.get_json("/api/health", expected_status=200)
    assert health.get("status") == "ok" and health.get("native_raster_available") is True

    clip_path = e2e_tmp_dir / "t4_05_stream_90f.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path, width=640, height=480, frames=90, fps=30, shaded_card=True, tmp_dir=e2e_tmp_dir
    )
    vid_up = live_studio_server.upload(
        "/api/videos/upload",
        filename=clip_path.name,
        content=clip_path.read_bytes(),
        content_type="video/mp4",
        expected_status=200,
        timeout=30.0,
    )
    vid_json = json.loads(vid_up["body"].decode("utf-8"))
    video_id = str(vid_json["video_id"])
    assert int(vid_json["frame_count"]) == 90

    range_resp = live_studio_server.get_range(
        f"/api/videos/{video_id}/media", start=0, end=4095, expected_status=206
    )
    assert len(range_resp["body"]) == 4096

    frame0_resp = live_studio_server.get(f"/api/videos/{video_id}/frame/0", expected_status=200)
    f0_arr, _ = decode_image_bytes(frame0_resp["body"])
    assert f0_arr.shape == (480, 640, 3)

    rgba_path = make_rgba_png(
        e2e_tmp_dir / "t4_05_label.png", width=320, height=240, mode="cutout_label", alpha_value=255
    )
    design_up = live_studio_server.upload(
        "/api/designs/upload",
        filename=rgba_path.name,
        content=rgba_path.read_bytes(),
        content_type="image/png",
        expected_status=200,
    )
    design_json = json.loads(design_up["body"].decode("utf-8"))
    design_id = str(design_json["design_id"])
    assert design_json["has_alpha"] is True

    sel_json = live_studio_server.post_json(
        f"/api/videos/{video_id}/select",
        {
            "frame_index": 0,
            "positive_points": [list(clip_meta["center_frame0"])],
            "negative_points": [],
            "bbox": list(clip_meta["bbox_frame0"]),
        },
        expected_status=200,
    )
    selection_id = str(sel_json["selection_id"])

    # Confirm HTTP/1.1 Keep-Alive reused a single TCP socket across all setup REST calls
    setup_ports = live_studio_server.client_ports_seen[port_idx:]
    assert len(set(setup_ports)) == 1, f"Expected HTTP/1.1 keep-alive reuse, saw ports {setup_ports}"

    outcome = live_studio_server.stream_video_ws_raw(
        video_id,
        {
            "selection_id": selection_id,
            "design_id": design_id,
            "placement": {
                "wrap_mode": "planar",
                "blend_mode": "luminance",
                "opacity": 0.92,
                "scale": 1.0,
            },
        },
        frame_timeout=10.0,
        total_timeout=60.0,
    )
    assert outcome["upgraded"] is True
    assert outcome["error_message"] is None
    report = outcome["complete_report"]
    assert isinstance(report, dict), f"WebSocket stream missing terminal complete report: {outcome}"

    frames = outcome["frames"]
    assert len(frames) == 90
    assert [int(f["frame_index"]) for f in frames] == list(range(90))
    assert_video_run_report_invariants(report, require_min_fps=15.0, expect_output_video=False)
    assert report["input_frame_count"] == 90
    assert report["processed_frame_count"] == 90
    assert report["rendered_frame_count"] == 90
    assert float(report["tracking_availability"]) >= 0.90

    client_fps = len(frames) / max(1e-6, float(outcome["wall_time_s"]))
    assert client_fps >= 15.0, (
        f"External WebSocket streaming throughput {client_fps:.2f} FPS < 15.0 FPS "
        f"(90 frames in {outcome['wall_time_s']:.3f}s)"
    )

    raw_samples = decode_frames(clip_path, [0, 30, 60, 85])
    for idx in (0, 30, 60, 85):
        ws_rgb, _ = decode_image_bytes(bytes(frames[idx]["jpeg_bytes"]))
        roi_diff, bg_diff = compute_roi_and_bg_delta(
            raw_samples[idx], ws_rgb, bbox_at_frame(clip_meta, idx)
        )
        assert roi_diff >= 8.0, f"Streamed frame {idx} missing composited RGBA label inside ROI: {roi_diff:.2f}"
        assert bg_diff <= 4.5, f"Streamed frame {idx} corrupted background outside ROI: {bg_diff:.2f}"


# ==============================================================================
# Scenario 6 — Browser Unified Studio Interactive Playback & MP4 Export
# ==============================================================================


@pytest.mark.f03
@pytest.mark.f05
@pytest.mark.f06
@pytest.mark.f10
@pytest.mark.f15
def test_t4_06_workload_browser_unified_studio_interactive_playback_and_mp4_export(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """Full Browser Unified Studio user journey: load clip, select, prompt, >=15 FPS playback, MP4 export."""
    import numpy as np

    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)

    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    browser.click_testid("auto-select-btn")

    browser.fill_testid("design-prompt-input", 'crimson and gold stripes text "STUDIO"')
    browser.click_testid("apply-prompt-btn")
    browser.select_testid("wrap-mode-select", "planar")
    browser.select_testid("blend-mode-select", "luminance")
    browser.set_slider_testid("opacity-slider", 0.90)

    _ensure_playing(browser)
    time.sleep(3.0)
    browser.pump_events(0.05)

    tel = browser.get_telemetry(timeout=5.0)
    assert tel is not None
    assert tel.get("mode") == "video"
    assert tel.get("playing") is True
    assert float(tel.get("currentFps", 0.0)) >= 15.0, f"Studio playback currentFps < 15.0: {tel}"
    assert float(tel.get("medianFrameMs", 999.0)) <= 66.67, f"Studio medianFrameMs > 66.67ms: {tel}"
    assert int(tel.get("processedFrames", 0)) >= 45, f"Expected >=45 processedFrames in 3.0s: {tel}"
    assert tel.get("trackingState") == "tracking"
    assert int(tel.get("inliers", 0)) > 0

    for hud_id in ("hud-fps", "hud-latency-ms", "hud-tracking-state", "hud-processed-frames"):
        hud_text = browser.get_testid_text(hud_id)
        assert hud_text, f"HUD element {hud_id} was empty during active video playback"

    mp4_path, export_report = _export_and_fetch_mp4(
        browser, live_studio_server, e2e_tmp_dir, timeout=60.0
    )
    assert_video_run_report_invariants(export_report, require_min_fps=15.0, expect_output_video=True)

    probed = probe_video(mp4_path, tmp_dir=e2e_tmp_dir)
    assert probed["codec_name"] == "h264"
    assert probed["width"] == 640 and probed["height"] == 480
    assert probed["frame_count"] == export_report["output_frame_count"]

    demo_meta = demo_corpus_manifest["planar_card_640x480"]
    src_mp4 = Path(demo_meta.get("path") or (demo_corpus_dir / "planar_card_640x480.mp4"))
    if not src_mp4.is_file():
        src_mp4 = demo_corpus_dir / "planar_card_640x480.mp4"
    raw_f0 = decode_frames(src_mp4, [0])[0]
    exp_f0 = decode_frames(mp4_path, [0])[0]
    p99_diff = float(np.percentile(np.abs(exp_f0.astype(np.float32) - raw_f0.astype(np.float32)), 99.0))
    assert p99_diff >= 12.0, f"Exported MP4 missing visible prompt artwork overlay: p99_diff={p99_diff:.2f}"

    default_bbox = (demo_meta.get("default_selection") or {}).get("bbox")
    if isinstance(default_bbox, list) and len(default_bbox) == 4:
        roi_diff, bg_diff = compute_roi_and_bg_delta(
            raw_f0,
            exp_f0,
            (int(default_bbox[0]), int(default_bbox[1]), int(default_bbox[2]), int(default_bbox[3])),
        )
        assert roi_diff >= 8.0
        assert bg_diff <= 3.5

    browser.assert_zero_errors()


# ==============================================================================
# Scenario 7 — Browser Live Camera Uncapped Tracking & Rendering (>= 15.0 FPS)
# ==============================================================================


@pytest.mark.f04
@pytest.mark.f05
@pytest.mark.f09
@pytest.mark.f10
@pytest.mark.f16
def test_t4_07_workload_browser_live_camera_uncapped_tracking_and_rendering_15fps(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Browser Live Camera Mode E2E workflow on moving .y4m feed at sustained >= 15.0 FPS."""
    y4m_path = e2e_tmp_dir / "moving_card_30fps.y4m"
    y4m_meta = make_y4m_camera_clip(
        y4m_path, width=640, height=480, frames=90, fps=30, tmp_dir=e2e_tmp_dir
    )
    rgba_path = make_rgba_png(
        e2e_tmp_dir / "t4_07_cam_emblem.png", width=256, height=256, mode="emblem", alpha_value=255
    )

    browser = chromium_session_factory(fake_video_y4m=y4m_path)
    browser.navigate(live_studio_server.base_url)

    browser.click_testid("mode-camera")
    unselected_sample = _wait_for_stage_ready(browser, min_non_black=0.20, timeout=12.0)

    browser.set_file_input("design-image-input", rgba_path)
    browser.select_testid("blend-mode-select", "luminance")
    browser.set_slider_testid("opacity-slider", 0.92)
    _select_y4m_target_in_camera(browser, y4m_meta, timeout=12.0)

    tel_start = browser.get_telemetry(timeout=5.0)
    assert tel_start is not None
    f_start = int(tel_start.get("processedFrames", 0))

    time.sleep(2.5)
    browser.pump_events(0.05)

    tel_end = browser.get_telemetry(timeout=5.0)
    assert tel_end is not None
    assert tel_end.get("mode") == "camera"
    assert tel_end.get("playing") is True
    assert float(tel_end.get("currentFps", 0.0)) >= 15.0, f"Live camera currentFps < 15.0: {tel_end}"
    assert float(tel_end.get("medianFrameMs", 999.0)) <= 66.67, f"Live camera medianFrameMs > 66.67: {tel_end}"
    assert (int(tel_end.get("processedFrames", 0)) - f_start) >= 35, (
        f"Expected >=35 frames processed in 2.5s of live camera mode: start={f_start}, end={tel_end}"
    )
    assert tel_end.get("trackingState") == "tracking"
    assert int(tel_end.get("inliers", 0)) > 0

    active_sample = browser.sample_stage_canvas(timeout=8.0)
    assert float(active_sample["non_black_ratio"]) > 0.25
    assert active_sample["png_base64"] != unselected_sample["png_base64"]

    hud_fps = browser.get_testid_text("hud-fps")
    assert re.search(r"\d", hud_fps), f"Expected numeric FPS in hud-fps, got {hud_fps!r}"
    browser.assert_zero_errors()


# ==============================================================================
# Scenario 8 — Legacy Coexistence (C++, Node.js, Python & SHA-256 Regression)
# ==============================================================================


@pytest.mark.f01
@pytest.mark.f02
def test_t4_08_workload_legacy_coexistence_cpp_node_python_and_sha256_regression(
    e2e_tmp_dir: Path,
) -> None:
    """Full legacy non-regression check across C++ ctest, 9 Node suites, Python suites, and SHA-256 pins."""
    venv_py = str(get_venv_python())

    # 1. Native C++ CTest suite (pose_tests + raster_tests)
    ctest_res = run_subproc(
        ["ctest", "--test-dir", str(NATIVE_BUILD_DIR), "--output-on-failure"],
        timeout=60.0,
        cwd=REPO_ROOT,
        tmp_dir=e2e_tmp_dir,
    )
    assert ctest_res.returncode == 0, (
        f"Native CTest suite failed (rc={ctest_res.returncode}):\n"
        f"STDOUT:\n{ctest_res.stdout}\nSTDERR:\n{ctest_res.stderr}"
    )

    # 2. Node --test suites (3 files)
    node_runner_res = run_subproc(
        [
            "node",
            "--test",
            "bench/test_prototype_client.js",
            "tools/test_design_studio_engine.js",
            "tools/test_phone_glass_helpers.js",
        ],
        timeout=45.0,
        cwd=REPO_ROOT,
        tmp_dir=e2e_tmp_dir,
    )
    assert node_runner_res.returncode == 0, (
        f"`node --test` suites failed (rc={node_runner_res.returncode}):\n"
        f"STDOUT:\n{node_runner_res.stdout}\nSTDERR:\n{node_runner_res.stderr}"
    )

    # 3. Standalone Node scripts (6 files)
    standalone_node_scripts = (
        "bench/test_quality_editor_state_smoke.js",
        "bench/test_workbench_resource_labels.js",
        "tools/test_phone_selection.js",
        "tools/test_phone_tracker.js",
        "tools/test_phone_feature_coverage.js",
        "tools/profile_phone_tracker.js",
    )
    for script_rel in standalone_node_scripts:
        res = run_subproc(
            ["node", script_rel],
            timeout=45.0,
            cwd=REPO_ROOT,
            tmp_dir=e2e_tmp_dir,
        )
        assert res.returncode == 0, (
            f"Standalone Node suite {script_rel} failed (rc={res.returncode}):\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )

    # 4. Linux `-S` site-packages portability via tools/prototype_bootstrap.py
    boot_res = run_subproc(
        [venv_py, "-B", "-S", "tools/prototype_bootstrap.py", "unittest", "tools.test_phone_static"],
        timeout=45.0,
        cwd=REPO_ROOT,
        tmp_dir=e2e_tmp_dir,
    )
    assert boot_res.returncode == 0, (
        f"prototype_bootstrap.py -S check failed (rc={boot_res.returncode}):\n"
        f"STDOUT:\n{boot_res.stdout}\nSTDERR:\n{boot_res.stderr}"
    )

    # 5. Standalone Python script tools/test_evaluate_tracking.py
    eval_res = run_subproc(
        [venv_py, "-B", "tools/test_evaluate_tracking.py"],
        timeout=30.0,
        cwd=REPO_ROOT,
        tmp_dir=e2e_tmp_dir,
    )
    assert eval_res.returncode == 0, (
        f"tools/test_evaluate_tracking.py failed (rc={eval_res.returncode}):\n"
        f"STDOUT:\n{eval_res.stdout}\nSTDERR:\n{eval_res.stderr}"
    )

    # 6. Core legacy Python unittest suites (with TMPDIR on XFS e2e_tmp_dir)
    unittest_modules = [
        "tools.test_design_studio",
        "tools.test_phone_static",
        "tools.test_quality_preview",
        "bench.test_renderer",
        "bench.test_scan_mapping",
        "bench.test_prototype_dataset",
        "bench.test_prototype_replay",
        "bench.test_prototype_tracking",
        "bench.test_surface_tracker",
        "bench.test_workbench_contract",
        "bench.test_workbench_geometry",
        "bench.test_workbench_evaluate",
        "bench.test_workbench_neural",
    ]
    unit_res = run_subproc(
        [venv_py, "-B", "-m", "unittest", *unittest_modules],
        timeout=120.0,
        cwd=REPO_ROOT,
        tmp_dir=e2e_tmp_dir,
    )
    assert unit_res.returncode == 0, (
        f"Legacy Python unittest regression battery failed (rc={unit_res.returncode}):\n"
        f"STDOUT:\n{unit_res.stdout}\nSTDERR:\n{unit_res.stderr}"
    )

    # 7. Exact SHA-256 preservation of pinned research files (workbench_neural.py:113-126)
    for rel_path, (expected_crlf_sha, expected_lf_sha) in PINNED_RESEARCH_SHA256.items():
        target = REPO_ROOT / rel_path
        assert target.is_file(), f"Pinned research file missing: {target}"
        raw_bytes = target.read_bytes()
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
        crlf_bytes = raw_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        crlf_sha = hashlib.sha256(crlf_bytes).hexdigest()
        assert crlf_sha == expected_crlf_sha, (
            f"Canonical pinned SHA-256 violation on {rel_path}: "
            f"got {crlf_sha}, expected {expected_crlf_sha}"
        )
        assert raw_sha in {expected_crlf_sha, expected_lf_sha}, (
            f"Raw checkout SHA-256 violation on {rel_path}: "
            f"got {raw_sha}, expected {expected_lf_sha} (LF) or {expected_crlf_sha} (CRLF)"
        )


# ==============================================================================
# Scenario 9 — Cross-Surface Parity (CLI vs REST vs WebSocket vs Browser Export)
# ==============================================================================


@pytest.mark.f03
@pytest.mark.f07
@pytest.mark.f10
@pytest.mark.f11
@pytest.mark.f13
@pytest.mark.f14
@pytest.mark.f15
def test_t4_09_workload_cross_surface_parity_cli_rest_websocket_and_browser_export(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Process the exact same video, selection, stripes recipe & planar placement across all 4 surfaces."""
    src_mp4 = e2e_tmp_dir / "t4_09_parity_45f.mp4"
    clip_meta = make_textured_tracking_clip(
        src_mp4, width=640, height=480, frames=45, fps=30, tmp_dir=e2e_tmp_dir
    )
    bx, by, bw, bh = clip_meta["bbox_frame0"]
    cx, cy = clip_meta["center_frame0"]
    colors = ["#ff595e", "#1982c4", "#ffca3a"]
    prompt_str = "#ff595e #1982c4 #ffca3a stripes"
    placement = {
        "wrap_mode": "planar",
        "blend_mode": "normal",
        "opacity": 0.90,
        "scale": 1.0,
        "rotation_deg": 0.0,
    }
    selection_dict = {
        "frame_index": 0,
        "positive_points": [[cx, cy]],
        "negative_points": [],
        "bbox": [bx, by, bw, bh],
    }
    recipe_dict = {
        "pattern": "stripes",
        "colors": colors,
        "prompt": prompt_str,
        "text": "",
        "width": 512,
        "height": 512,
    }

    # Surface 1: CLI `process`
    cli_mp4 = e2e_tmp_dir / "t4_09_cli_out.mp4"
    _, cli_report = run_cli_json(
        [
            "process",
            "--input",
            str(src_mp4),
            "--output",
            str(cli_mp4),
            "--pattern",
            "stripes",
            "--colors",
            ",".join(colors),
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "normal",
            "--opacity",
            "0.90",
            "--scale",
            "1.0",
            "--rotation-deg",
            "0.0",
            "--select-frame",
            "0",
            "--point",
            f"{cx},{cy}",
            "--box",
            f"{bx},{by},{bw},{bh}",
        ],
        timeout=60.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(cli_report, require_min_fps=15.0, expect_output_video=True)
    assert cli_report["output_frame_count"] == 45

    # Surface 2: REST `POST /api/videos/{id}/render`
    vid_up = live_studio_server.upload(
        "/api/videos/upload",
        filename=src_mp4.name,
        content=src_mp4.read_bytes(),
        content_type="video/mp4",
        expected_status=200,
    )
    video_id = str(json.loads(vid_up["body"].decode("utf-8"))["video_id"])

    rest_resp = live_studio_server.post_json(
        f"/api/videos/{video_id}/render",
        {
            "selection": selection_dict,
            "design_recipe": recipe_dict,
            "placement": placement,
        },
        expected_status=200,
        timeout=60.0,
    )
    rest_report = rest_resp["report"]
    assert_video_run_report_invariants(rest_report, require_min_fps=15.0, expect_output_video=True)
    assert rest_report["output_frame_count"] == 45
    rest_mp4 = e2e_tmp_dir / "t4_09_rest_out.mp4"
    rest_mp4.write_bytes(
        live_studio_server.get(str(rest_resp["video_url"]), expected_status=200)["body"]
    )

    # Surface 3: WebSocket `WS /ws/videos/{id}/stream`
    ws_frames, ws_report = live_studio_server.stream_video_ws(
        video_id,
        {
            "selection": selection_dict,
            "design_recipe": recipe_dict,
            "placement": placement,
        },
        frame_timeout=10.0,
        total_timeout=60.0,
    )
    assert_video_run_report_invariants(ws_report, require_min_fps=15.0, expect_output_video=False)
    assert len(ws_frames) == ws_report["processed_frame_count"] == 45

    # Surface 4: Browser Studio upload + selection + design + `export-mp4-btn`
    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)
    browser.set_file_input("video-file-input", src_mp4)
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)

    browser.click_testid("select-mode-box")
    browser.drag_canvas_box(float(bx), float(by), float(bw), float(bh))
    browser.fill_testid("design-prompt-input", prompt_str)
    browser.click_testid("apply-prompt-btn")
    browser.select_testid("pattern-select", "stripes")
    browser.select_testid("wrap-mode-select", "planar")
    browser.select_testid("blend-mode-select", "normal")
    browser.set_slider_testid("opacity-slider", 0.90)
    browser.set_slider_testid("scale-slider", 1.0)
    browser.set_slider_testid("rotation-slider", 0.0)

    browser_mp4, browser_report = _export_and_fetch_mp4(
        browser, live_studio_server, e2e_tmp_dir, timeout=60.0
    )
    assert_video_run_report_invariants(browser_report, require_min_fps=15.0, expect_output_video=True)
    assert browser_report["output_frame_count"] == 45
    browser.assert_zero_errors()

    # Cross-surface frame-15 visual & foreground/background parity check
    f_idx = 15
    raw_f15 = decode_frames(src_mp4, [f_idx])[f_idx]
    cli_f15 = decode_frames(cli_mp4, [f_idx])[f_idx]
    rest_f15 = decode_frames(rest_mp4, [f_idx])[f_idx]
    ws_f15, _ = decode_image_bytes(bytes(ws_frames[f_idx]["jpeg_bytes"]))
    brow_f15 = decode_frames(browser_mp4, [f_idx])[f_idx]

    gt_bbox_15 = bbox_at_frame(clip_meta, f_idx)
    for surf_name, frame_rgb in (
        ("CLI", cli_f15),
        ("REST", rest_f15),
        ("WebSocket", ws_f15),
        ("Browser", brow_f15),
    ):
        roi_diff, bg_diff = compute_roi_and_bg_delta(raw_f15, frame_rgb, gt_bbox_15)
        assert roi_diff >= 10.0, f"{surf_name} frame 15 missing stripes overlay inside ROI: {roi_diff:.2f}"
        assert bg_diff <= 4.5, f"{surf_name} frame 15 altered background outside ROI: {bg_diff:.2f}"

    assert compute_psnr(cli_f15, rest_f15) >= 24.0, "CLI vs REST frame-15 visual divergence"
    assert compute_psnr(rest_f15, ws_f15) >= 24.0, "REST vs WebSocket frame-15 visual divergence"
    assert compute_psnr(rest_f15, brow_f15) >= 20.0, "REST vs Browser export frame-15 visual divergence"
