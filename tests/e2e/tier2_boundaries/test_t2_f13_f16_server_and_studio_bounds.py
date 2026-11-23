"""Tier 2 Boundary, Corner-Case & Negative Error-Handling E2E Tests for F13-F16.

Verifies structured 4xx HTTP errors, RFC Range handling, path-traversal safety, WebSocket error/close
hygiene and concurrent isolation, and Browser Studio / Live Camera resilience under invalid files,
permission denials, abrupt track loss, narrow mobile viewports, and rapid epoch/mode switching.
Zero imports of `visualizeit.*` or top-level non-stdlib third-party libraries (other than `pytest`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.e2e._helpers import (
    HeadlessChromiumSession,
    StudioAPIClient,
    WebSocketClosedError,
    assert_no_python_traceback,
    decode_image_bytes,
    make_solid_clip,
    make_textured_tracking_clip,
    make_y4m_camera_clip,
)

pytestmark = pytest.mark.tier2


def _upload_valid_video(client: StudioAPIClient, video_path: Path) -> dict[str, Any]:
    resp = client.upload(
        "/api/videos/upload",
        filename=video_path.name,
        content=video_path.read_bytes(),
        content_type="video/mp4",
        expected_status=200,
        timeout=30.0,
    )
    return json.loads(resp["body"].decode("utf-8"))


def _resolve_or_upload_demo(
    client: StudioAPIClient,
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    clip_id: str,
) -> tuple[str, dict[str, Any]]:
    meta = demo_corpus_manifest[clip_id]
    probe = client.get(f"/api/videos/{clip_id}", timeout=10.0)
    if probe["status"] == 200:
        return clip_id, meta
    clip_path = Path(meta.get("path") or (demo_corpus_dir / f"{clip_id}.mp4"))
    if not clip_path.is_file():
        clip_path = demo_corpus_dir / f"{clip_id}.mp4"
    up = _upload_valid_video(client, clip_path)
    return str(up["video_id"]), meta


def _wait_for_stage_ready(
    browser: HeadlessChromiumSession,
    *,
    min_non_black: float = 0.10,
    timeout: float = 12.0,
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
    raise AssertionError(f"stage-canvas failed to become ready within {timeout}s: {last_sample}")


def _ensure_playing(browser: HeadlessChromiumSession, *, timeout: float = 10.0) -> dict[str, Any]:
    tel = browser.get_telemetry(timeout=5.0)
    if not tel or not tel.get("playing"):
        browser.click_testid("play-pause-btn", timeout=timeout)
    return browser.wait_for_telemetry(lambda t: bool(t.get("playing")), timeout=timeout)


# ==============================================================================
# F13 — FastAPI HTTP/1.1 REST API Boundaries (6 tests)
# ==============================================================================


@pytest.mark.f13
def test_f13_bound_unknown_video_id_and_render_id_return_404(
    live_studio_server: StudioAPIClient,
) -> None:
    """Verify all 7 video and render routes return structured HTTP 404 on unknown IDs without 500 crashes."""
    get_routes = (
        "/api/videos/unknown_id_999",
        "/api/videos/unknown_id_999/media",
        "/api/videos/unknown_id_999/frame/0",
        "/api/renders/unknown_render_999/video.mp4",
        "/api/renders/unknown_render_999/report.json",
    )
    for route in get_routes:
        resp = live_studio_server.get(route, expected_status=404)
        text = resp["body"].decode("utf-8", "replace")
        assert_no_python_traceback(text)

    sel_resp = live_studio_server.post_json(
        "/api/videos/unknown_id_999/select",
        {"frame_index": 0, "positive_points": [[120, 120]], "negative_points": [], "bbox": None},
        expected_status=404,
    )
    assert isinstance(sel_resp, dict)

    ren_resp = live_studio_server.post_json(
        "/api/videos/unknown_id_999/render",
        {"design_recipe": {"pattern": "solid", "colors": ["#ff595e", "#1982c4"]}},
        expected_status=404,
    )
    assert isinstance(ren_resp, dict)


@pytest.mark.f13
def test_f13_bound_path_traversal_ids_do_not_escape_storage(
    live_studio_server: StudioAPIClient,
) -> None:
    """Verify URL-encoded path traversal IDs cannot escape server storage or leak host files."""
    traversal_paths = (
        "/api/videos/..%2F..%2F..%2Fetc%2Fpasswd",
        "/api/videos/..%2Fpyproject.toml/media",
        "/api/renders/..%2F..%2F..%2Fetc%2Fpasswd/report.json",
        "/api/renders/..%2Fsecret/video.mp4",
    )
    for path in traversal_paths:
        resp = live_studio_server.get(path, expected_status=(400, 404, 422))
        body_text = resp["body"].decode("utf-8", "replace")
        assert "root:x:" not in body_text
        assert "[project]" not in body_text
        assert_no_python_traceback(body_text)

    health = live_studio_server.get_json("/api/health", expected_status=200)
    assert health.get("status") == "ok"


@pytest.mark.f13
def test_f13_bound_corrupt_empty_and_non_video_uploads_return_4xx(
    live_studio_server: StudioAPIClient,
) -> None:
    """Verify empty 0-byte, plain-text, and truncated corrupt MP4 uploads return HTTP 400/415/422."""
    bad_payloads = (
        ("empty.mp4", b"", "video/mp4"),
        ("notes.txt", b"plain text is not a video container\n", "text/plain"),
        ("disguised.mp4", b"NOT_AN_MP4_CONTAINER_PAYLOAD", "video/mp4"),
        ("truncated.mp4", b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom_truncated", "video/mp4"),
    )
    for fname, content, ctype in bad_payloads:
        resp = live_studio_server.upload(
            "/api/videos/upload",
            filename=fname,
            content=content,
            content_type=ctype,
            expected_status=(400, 415, 422),
        )
        assert_no_python_traceback(resp["body"].decode("utf-8", "replace"))

    health = live_studio_server.get_json("/api/health", expected_status=200)
    assert health.get("status") == "ok"


@pytest.mark.f13
def test_f13_bound_frame_extraction_negative_and_out_of_bounds_index_returns_400_or_404(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify GET /api/videos/{id}/frame/{idx} accepts N-1 and rejects -1, N, and 99999 cleanly."""
    clip_path = e2e_tmp_dir / "bounds_12f.mp4"
    make_textured_tracking_clip(clip_path, width=640, height=480, frames=12, fps=30, tmp_dir=e2e_tmp_dir)
    up = _upload_valid_video(live_studio_server, clip_path)
    vid = str(up["video_id"])
    assert int(up["frame_count"]) == 12

    last_valid = live_studio_server.get(f"/api/videos/{vid}/frame/11", expected_status=200)
    arr11, _ = decode_image_bytes(last_valid["body"])
    assert arr11.shape == (480, 640, 3)

    live_studio_server.get(f"/api/videos/{vid}/frame/-1", expected_status=(400, 404, 422))
    live_studio_server.get(f"/api/videos/{vid}/frame/12", expected_status=(400, 404))
    live_studio_server.get(f"/api/videos/{vid}/frame/99999", expected_status=(400, 404))


@pytest.mark.f13
def test_f13_bound_invalid_http_range_headers_handled_per_rfc(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify out-of-bounds and inverted Range headers on /media return 416/400 (never 500)."""
    clip_path = e2e_tmp_dir / "range_bounds.mp4"
    make_textured_tracking_clip(clip_path, width=640, height=480, frames=15, fps=30, tmp_dir=e2e_tmp_dir)
    up = _upload_valid_video(live_studio_server, clip_path)
    vid = str(up["video_id"])

    unsatisfiable = live_studio_server.get_range(
        f"/api/videos/{vid}/media",
        range_header="bytes=999999999-999999999",
        expected_status=(400, 416),
    )
    assert unsatisfiable["status"] in (400, 416)

    inverted = live_studio_server.get_range(
        f"/api/videos/{vid}/media",
        range_header="bytes=500-100",
        expected_status=(200, 400, 416),
    )
    assert inverted["status"] != 500

    malformed = live_studio_server.get_range(
        f"/api/videos/{vid}/media",
        range_header="bytes=not-a-number",
        expected_status=(200, 400, 416),
    )
    assert malformed["status"] != 500

    health = live_studio_server.get_json("/api/health", expected_status=200)
    assert health.get("status") == "ok"


@pytest.mark.f13
def test_f13_bound_malformed_json_and_unknown_design_or_selection_id_in_render(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify malformed JSON bodies and unknown design_id/selection_id references return 4xx."""
    clip_path = e2e_tmp_dir / "bad_render_req.mp4"
    make_textured_tracking_clip(clip_path, width=640, height=480, frames=12, fps=30, tmp_dir=e2e_tmp_dir)
    up = _upload_valid_video(live_studio_server, clip_path)
    vid = str(up["video_id"])

    for endpoint in ("/api/designs/render", f"/api/videos/{vid}/render"):
        resp = live_studio_server.request(
            "POST",
            endpoint,
            headers={"Content-Type": "application/json"},
            body=b"{bad_json_syntax",
        )
        assert resp["status"] in (400, 422), f"Expected 400/422 on malformed JSON to {endpoint}, got {resp['status']}"
        assert_no_python_traceback(resp["body"].decode("utf-8", "replace"))

    missing_refs = live_studio_server.post_json(
        f"/api/videos/{vid}/render",
        {
            "design_id": "nonexistent_design_999",
            "selection_id": "nonexistent_selection_999",
        },
        expected_status=(400, 404, 422),
    )
    assert isinstance(missing_refs, dict)


# ==============================================================================
# F14 — Real-Time WebSocket Stream Boundaries (6 tests)
# ==============================================================================


@pytest.mark.f14
def test_f14_bound_websocket_unknown_video_id_closes_or_errors_cleanly(
    live_studio_server: StudioAPIClient,
) -> None:
    """Verify WS /ws/videos/nonexistent_video_999/stream rejects handshake or closes cleanly without hanging."""
    outcome = live_studio_server.stream_video_ws_raw(
        "nonexistent_video_999",
        {"max_frames": 5},
        frame_timeout=4.0,
        total_timeout=6.0,
    )
    assert len(outcome["frames"]) == 0
    assert outcome["complete_report"] is None
    if not outcome["upgraded"]:
        assert outcome["handshake_status"] in (400, 403, 404)
    else:
        assert outcome["error_message"] is not None or outcome["close_code"] in (
            1000,
            1003,
            1008,
            1011,
            4000,
            4004,
        )


@pytest.mark.f14
def test_f14_bound_websocket_malformed_non_json_payload_handled_without_server_crash(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify sending malformed non-JSON text over /ws/videos/{id}/stream closes cleanly and server stays healthy."""
    clip_path = e2e_tmp_dir / "ws_bad_json.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path, width=640, height=480, frames=15, fps=30, tmp_dir=e2e_tmp_dir
    )
    up = _upload_valid_video(live_studio_server, clip_path)
    vid = str(up["video_id"])

    bad_outcome = live_studio_server.stream_video_ws_raw(
        vid,
        "not valid json {{{",
        frame_timeout=3.0,
        total_timeout=5.0,
    )
    assert len(bad_outcome["frames"]) == 0
    assert bad_outcome["complete_report"] is None
    assert bad_outcome["error_message"] is not None or bad_outcome["close_code"] is not None
    assert float(bad_outcome["wall_time_s"]) < 3.5

    frames, report = live_studio_server.stream_video_ws(
        vid,
        {
            "selection": {"frame_index": 0, "bbox": list(clip_meta["bbox_frame0"])},
            "design_recipe": {"pattern": "solid", "colors": ["#ff595e", "#1982c4"]},
            "max_frames": 5,
        },
        frame_timeout=8.0,
        total_timeout=20.0,
    )
    assert len(frames) == 5 and report["processed_frame_count"] == 5


@pytest.mark.f14
def test_f14_bound_websocket_invalid_placement_opacity_and_scale_rejected(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify out-of-range placement opacity=2.5 and scale=-1.0 over WebSocket are rejected with zero streamed frames."""
    clip_path = e2e_tmp_dir / "ws_bad_placement.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path, width=640, height=480, frames=12, fps=30, tmp_dir=e2e_tmp_dir
    )
    up = _upload_valid_video(live_studio_server, clip_path)
    vid = str(up["video_id"])

    outcome = live_studio_server.stream_video_ws_raw(
        vid,
        {
            "selection": {"frame_index": 0, "bbox": list(clip_meta["bbox_frame0"])},
            "design_recipe": {"pattern": "stripes", "colors": ["#ff595e", "#1982c4"]},
            "placement": {"wrap_mode": "planar", "opacity": 2.5, "scale": -1.0},
        },
        frame_timeout=4.0,
        total_timeout=8.0,
    )
    assert len(outcome["frames"]) == 0
    assert outcome["complete_report"] is None
    assert outcome["error_message"] is not None or outcome["close_code"] is not None


@pytest.mark.f14
def test_f14_bound_websocket_max_frames_zero_or_negative_rejected(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify max_frames=0 and max_frames=-5 are cleanly rejected on /ws/videos/{id}/stream."""
    clip_path = e2e_tmp_dir / "ws_bad_maxframes.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path, width=640, height=480, frames=12, fps=30, tmp_dir=e2e_tmp_dir
    )
    up = _upload_valid_video(live_studio_server, clip_path)
    vid = str(up["video_id"])

    for invalid_max in (0, -5):
        outcome = live_studio_server.stream_video_ws_raw(
            vid,
            {
                "selection": {"frame_index": 0, "bbox": list(clip_meta["bbox_frame0"])},
                "max_frames": invalid_max,
            },
            frame_timeout=4.0,
            total_timeout=8.0,
        )
        assert len(outcome["frames"]) == 0
        assert outcome["complete_report"] is None
        assert outcome["error_message"] is not None or outcome["close_code"] is not None

    health = live_studio_server.get_json("/api/health", expected_status=200)
    assert health.get("status") == "ok"


@pytest.mark.f14
def test_f14_bound_websocket_abrupt_client_disconnect_mid_stream_cleans_up_reader(
    live_studio_server: StudioAPIClient,
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
) -> None:
    """Verify 5 rapid mid-stream TCP disconnects on an HD video do not deadlock or exhaust server resources."""
    vid, demo_meta = _resolve_or_upload_demo(
        live_studio_server, demo_corpus_dir, demo_corpus_manifest, "planar_hd_1280x720"
    )
    sel = demo_meta.get("default_selection") or {}

    for _ in range(5):
        ws = live_studio_server.open_ws(f"/ws/videos/{vid}/stream", timeout=8.0)
        try:
            ws.send_json(
                {
                    "selection": sel,
                    "design_recipe": {"pattern": "stripes", "colors": ["#ff595e", "#1982c4"]},
                    "placement": {"wrap_mode": "planar", "blend_mode": "normal"},
                }
            )
            ws.settimeout(8.0)
            frames_read = 0
            while frames_read < 3:
                try:
                    op, raw = ws.recv_message()
                except WebSocketClosedError:
                    break
                if op == 0x1:
                    msg = json.loads(raw.decode("utf-8"))
                    if msg.get("type") == "frame" and msg.get("jpeg_base64"):
                        frames_read += 1
                elif op == 0x2:
                    frames_read += 1
            assert frames_read >= 1
        finally:
            with contextlib.suppress(OSError):
                ws._sock.close()

    health = live_studio_server.get_json("/api/health", expected_status=200)
    assert health.get("status") == "ok"

    frames_ok, report_ok = live_studio_server.stream_video_ws(
        vid,
        {"selection": sel, "max_frames": 8},
        frame_timeout=10.0,
        total_timeout=30.0,
    )
    assert len(frames_ok) == 8 and report_ok["processed_frame_count"] == 8


@pytest.mark.f14
def test_f14_bound_websocket_two_concurrent_streams_on_same_video_isolated(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify two simultaneous WebSocket streams on the same video_id with different wrap_modes stay isolated."""
    clip_path = e2e_tmp_dir / "ws_concurrent.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path, width=640, height=480, frames=20, fps=30, tmp_dir=e2e_tmp_dir
    )
    up = _upload_valid_video(live_studio_server, clip_path)
    vid = str(up["video_id"])

    def _run_stream(wrap_mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        with StudioAPIClient(
            host=live_studio_server.host,
            port=live_studio_server.port,
            server_proc=live_studio_server.server_proc,
        ) as sub_client:
            return sub_client.stream_video_ws(
                vid,
                {
                    "selection": {"frame_index": 0, "bbox": list(clip_meta["bbox_frame0"])},
                    "design_recipe": {"pattern": "checker", "colors": ["#ff595e", "#ffca3a"]},
                    "placement": {"wrap_mode": wrap_mode, "blend_mode": "luminance"},
                    "max_frames": 12,
                },
                frame_timeout=12.0,
                total_timeout=45.0,
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_planar = pool.submit(_run_stream, "planar")
        fut_cylinder = pool.submit(_run_stream, "cylinder")
        frames_planar, rep_planar = fut_planar.result(timeout=50.0)
        frames_cyl, rep_cyl = fut_cylinder.result(timeout=50.0)

    assert [int(f["frame_index"]) for f in frames_planar] == list(range(12))
    assert [int(f["frame_index"]) for f in frames_cyl] == list(range(12))
    assert rep_planar["wrap_mode"] == "planar"
    assert rep_cyl["wrap_mode"] == "cylinder"


# ==============================================================================
# F15 — Unified Browser Video Studio Boundaries (6 tests)
# ==============================================================================


@pytest.mark.f15
def test_f15_bound_play_and_export_before_any_selection_or_video_safe_ux(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
) -> None:
    """Verify clicking play-pause-btn and export-mp4-btn before loading a video causes zero uncaught JS errors."""
    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)
    browser.wait_for_testid("play-pause-btn")
    browser.wait_for_testid("export-mp4-btn")

    for tid in ("play-pause-btn", "export-mp4-btn"):
        browser.evaluate(
            f"""(() => {{
                const el = document.querySelector('[data-testid={json.dumps(tid)}]');
                if (el && !el.disabled) el.click();
            }})()""",
            await_promise=False,
        )
    browser.pump_events(0.35)

    tel = browser.get_telemetry(timeout=5.0)
    assert isinstance(tel, dict)
    browser.assert_zero_errors(allow_console_substrings=["No video", "load", "select", "400", "404"])


@pytest.mark.f15
def test_f15_bound_invalid_corrupt_video_file_input_shows_error_without_page_crash(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify attaching a corrupt file to video-file-input causes zero uncaught page crashes and Studio recovers."""
    bad_video = e2e_tmp_dir / "corrupt_upload.mp4"
    bad_video.write_bytes(b"corrupt_not_a_valid_h264_mp4_stream")

    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)
    browser.wait_for_testid("video-file-input")
    browser.set_file_input("video-file-input", bad_video)
    browser.pump_events(0.5)
    assert not browser.page_exceptions, f"Uncaught exception on corrupt video upload: {browser.page_exceptions}"

    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    browser.click_testid("auto-select-btn")
    _ensure_playing(browser)
    tel = browser.wait_for_telemetry(lambda t: int(t.get("processedFrames", 0)) >= 5, timeout=10.0)
    assert tel["playing"] is True
    assert not browser.page_exceptions


@pytest.mark.f15
def test_f15_bound_invalid_non_image_file_in_design_image_input_rejected_cleanly(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify attaching a .txt file to design-image-input is rejected cleanly and procedural design stays usable."""
    bad_img = e2e_tmp_dir / "not_an_image.txt"
    bad_img.write_text("plain text payload instead of PNG/JPEG/WebP\n", encoding="utf-8")

    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)
    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)

    browser.set_file_input("design-image-input", bad_img)
    browser.pump_events(0.5)
    assert not browser.page_exceptions, f"Uncaught exception on bad design image: {browser.page_exceptions}"

    browser.select_testid("pattern-select", "checker")
    browser.click_testid("auto-select-btn")
    _ensure_playing(browser)
    tel = browser.wait_for_telemetry(lambda t: int(t.get("processedFrames", 0)) >= 5, timeout=10.0)
    assert tel["playing"] is True
    assert not browser.page_exceptions


@pytest.mark.f15
def test_f15_bound_clear_selection_during_active_playback_suppresses_overlay_safely(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
) -> None:
    """Verify clicking clear-selection-btn mid-playback resets trackingState cleanly with zero Worker crashes."""
    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)

    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    browser.click_testid("auto-select-btn")
    _ensure_playing(browser)
    browser.wait_for_telemetry(
        lambda t: t.get("trackingState") == "tracking" and int(t.get("processedFrames", 0)) >= 5,
        timeout=12.0,
    )

    browser.click_testid("clear-selection-btn")
    tel_after = browser.wait_for_telemetry(
        lambda t: t.get("trackingState") != "tracking" and int(t.get("inliers", 0)) == 0,
        timeout=8.0,
    )
    assert tel_after["trackingState"] != "tracking"
    assert int(tel_after.get("inliers", 0)) == 0
    time.sleep(0.5)
    browser.assert_zero_errors()


@pytest.mark.f15
def test_f15_bound_slider_min_max_clamping_opacity_scale_rotation(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
) -> None:
    """Verify min, max, out-of-range, and NaN slider values clamp safely with finite telemetry and canvas."""
    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)

    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    browser.click_testid("auto-select-btn")

    slider_matrix = (
        ("opacity-slider", ("0", "1", "-999", "999", "NaN", "0.8")),
        ("scale-slider", ("0.1", "5.0", "-50", "999", "NaN", "1.2")),
        ("rotation-slider", ("-180", "180", "-9999", "9999", "NaN", "15")),
    )
    for testid, values in slider_matrix:
        for val in values:
            browser.set_slider_testid(testid, val)

    _ensure_playing(browser)
    tel = browser.wait_for_telemetry(lambda t: int(t.get("processedFrames", 0)) >= 5, timeout=10.0)
    for k in ("currentFps", "medianFrameMs", "p95FrameMs"):
        assert math.isfinite(float(tel[k])), f"Non-finite telemetry {k} after extreme slider inputs: {tel}"

    sample = browser.sample_stage_canvas(timeout=8.0)
    for ch_val in sample["mean_rgb"]:
        assert math.isfinite(float(ch_val))
    browser.assert_zero_errors()


@pytest.mark.f15
def test_f15_bound_rapid_demo_clip_switching_cancels_stale_worker_epochs(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
) -> None:
    """Verify rapid switching across 3 demo clips within ~300ms cancels stale worker epochs cleanly."""
    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)

    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    browser.click_testid("auto-select-btn")
    _ensure_playing(browser)

    for clip_id in ("cylinder_mug_640x480", "occlusion_recovery_640x480", "planar_card_640x480"):
        browser.select_testid("demo-clip-select", clip_id)
        browser.click_testid("load-demo-btn")
        time.sleep(0.06)

    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    browser.click_testid("auto-select-btn")
    _ensure_playing(browser)
    f0 = int((browser.get_telemetry() or {}).get("processedFrames", 0))
    tel_final = browser.wait_for_telemetry(
        lambda t: bool(t.get("playing")) and int(t.get("processedFrames", 0)) >= f0 + 8,
        timeout=12.0,
    )
    assert tel_final["playing"] is True
    browser.assert_zero_errors()


# ==============================================================================
# F16 — Live Camera & Telemetry HUD Boundaries (6 tests)
# ==============================================================================


@pytest.mark.f16
def test_f16_bound_getusermedia_permission_denied_notallowederror_graceful_state(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
) -> None:
    """Verify getUserMedia NotAllowedError leaves playing=false with zero unhandled rejections and recovers."""
    browser = chromium_session_factory()
    browser.add_init_script(
        """(() => {
            if (!navigator.mediaDevices) navigator.mediaDevices = {};
            navigator.mediaDevices.getUserMedia = () =>
                Promise.reject(new DOMException("Permission denied", "NotAllowedError"));
        })();"""
    )
    browser.navigate(live_studio_server.base_url)
    browser.click_testid("mode-camera")
    browser.pump_events(0.5)

    assert not browser.page_exceptions, (
        f"Unhandled exception/rejection on NotAllowedError: {browser.page_exceptions}"
    )
    tel = browser.get_telemetry(timeout=5.0)
    assert tel is not None and tel.get("playing") is False

    browser.click_testid("mode-video")
    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    _ensure_playing(browser)
    tel_recovered = browser.wait_for_telemetry(
        lambda t: t.get("mode") == "video" and bool(t.get("playing")),
        timeout=10.0,
    )
    assert tel_recovered["playing"] is True
    assert not browser.page_exceptions


@pytest.mark.f16
def test_f16_bound_getusermedia_missing_device_notfounderror_graceful_state(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
) -> None:
    """Verify getUserMedia NotFoundError is caught gracefully with playing=false and zero uncaught JS errors."""
    browser = chromium_session_factory()
    browser.add_init_script(
        """(() => {
            if (!navigator.mediaDevices) navigator.mediaDevices = {};
            navigator.mediaDevices.getUserMedia = () =>
                Promise.reject(new DOMException("Requested device not found", "NotFoundError"));
        })();"""
    )
    browser.navigate(live_studio_server.base_url)
    browser.click_testid("mode-camera")
    browser.pump_events(0.5)

    assert not browser.page_exceptions, (
        f"Unhandled exception/rejection on NotFoundError: {browser.page_exceptions}"
    )
    tel = browser.get_telemetry(timeout=5.0)
    assert tel is not None and tel.get("playing") is False


@pytest.mark.f16
def test_f16_bound_camera_track_unexpected_ended_event_stops_loop_cleanly(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify unexpected MediaStreamTrack stop/ended mid-camera session stops cleanly without infinite errors."""
    y4m_path = e2e_tmp_dir / "cam_ended.y4m"
    make_y4m_camera_clip(y4m_path, width=640, height=480, frames=45, fps=30, tmp_dir=e2e_tmp_dir)

    browser = chromium_session_factory(fake_video_y4m=y4m_path)
    browser.add_init_script(
        """(() => {
            window.__E2E_LAST_STREAM__ = null;
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                const orig = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
                navigator.mediaDevices.getUserMedia = async (c) => {
                    const s = await orig(c);
                    window.__E2E_LAST_STREAM__ = s;
                    return s;
                };
            }
        })();"""
    )
    browser.navigate(live_studio_server.base_url)
    browser.click_testid("mode-camera")
    browser.wait_for_telemetry(
        lambda t: t.get("mode") == "camera" and bool(t.get("playing")) and int(t.get("processedFrames", 0)) >= 5,
        timeout=12.0,
    )

    stopped_count = browser.evaluate(
        """(() => {
            const s = window.__E2E_LAST_STREAM__;
            if (!s) return 0;
            const tracks = s.getTracks();
            for (const t of tracks) {
                t.stop();
                t.dispatchEvent(new Event("ended"));
            }
            return tracks.length;
        })()""",
        await_promise=False,
    )
    assert int(stopped_count) >= 1
    time.sleep(0.6)
    browser.pump_events(0.1)
    assert not browser.page_exceptions, (
        f"Uncaught exception after camera track ended event: {browser.page_exceptions}"
    )


@pytest.mark.f16
def test_f16_bound_camera_pitch_black_feed_reports_lost_and_finite_telemetry(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify pure-black featureless camera feed keeps finite FPS/latency telemetry and never falsely locks."""
    black_y4m = e2e_tmp_dir / "cam_pitch_black.y4m"
    make_solid_clip(
        black_y4m, width=640, height=480, frames=45, fps=30, color="0x000000", tmp_dir=e2e_tmp_dir
    )

    browser = chromium_session_factory(fake_video_y4m=black_y4m)
    browser.navigate(live_studio_server.base_url)
    browser.click_testid("mode-camera")
    browser.wait_for_telemetry(
        lambda t: t.get("mode") == "camera" and bool(t.get("playing")) and int(t.get("processedFrames", 0)) >= 5,
        timeout=12.0,
    )

    browser.click_testid("auto-select-btn")
    browser.click_testid("select-mode-point")
    browser.click_canvas_point(320, 240)
    time.sleep(1.0)
    browser.pump_events(0.05)

    tel = browser.get_telemetry(timeout=5.0)
    assert tel is not None
    for k in ("currentFps", "medianFrameMs", "p95FrameMs"):
        assert math.isfinite(float(tel[k])), f"Non-finite {k} on pitch-black camera feed: {tel}"
    assert tel.get("trackingState") != "tracking", (
        f"Tracker falsely reported trackingState='tracking' on zero-gradient black feed: {tel}"
    )
    assert not browser.page_exceptions


@pytest.mark.f16
def test_f16_bound_rapid_mode_toggle_10x_video_camera_stress(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify toggling mode-camera <-> mode-video 10 times in ~1s leaves accurate mode and zero JS exceptions."""
    y4m_path = e2e_tmp_dir / "cam_toggle_stress.y4m"
    make_y4m_camera_clip(y4m_path, width=640, height=480, frames=45, fps=30, tmp_dir=e2e_tmp_dir)

    browser = chromium_session_factory(fake_video_y4m=y4m_path)
    browser.navigate(live_studio_server.base_url)
    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)

    for i in range(10):
        target_btn = "mode-camera" if (i % 2 == 0) else "mode-video"
        browser.click_testid(target_btn)
        time.sleep(0.08)

    tel_video = browser.wait_for_telemetry(lambda t: t.get("mode") == "video", timeout=8.0)
    assert tel_video["mode"] == "video"

    browser.click_testid("mode-camera")
    tel_cam = browser.wait_for_telemetry(
        lambda t: t.get("mode") == "camera" and bool(t.get("playing")),
        timeout=12.0,
    )
    assert tel_cam["mode"] == "camera" and tel_cam["playing"] is True
    assert not browser.page_exceptions


@pytest.mark.f16
def test_f16_bound_narrow_mobile_viewport_360x640_canvas_click_mapping_valid(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify 360x640 dpr=2 mobile viewport maps stage-canvas selection clicks accurately onto the target."""
    y4m_path = e2e_tmp_dir / "cam_mobile.y4m"
    y4m_meta = make_y4m_camera_clip(
        y4m_path, width=640, height=480, frames=60, fps=30, tmp_dir=e2e_tmp_dir
    )

    browser = chromium_session_factory(fake_video_y4m=y4m_path)
    browser.call(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": 360,
            "height": 640,
            "deviceScaleFactor": 2,
            "mobile": True,
        },
    )
    browser.navigate(live_studio_server.base_url)
    browser.click_testid("mode-camera")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=12.0)

    px = float(y4m_meta["x0"]) + (float(y4m_meta["span_x"]) + float(y4m_meta["box_w"])) * 0.5
    py = float(y4m_meta["y0"]) + (float(y4m_meta["span_y"]) + float(y4m_meta["box_h"])) * 0.5
    browser.click_testid("select-mode-point")
    browser.click_canvas_point(px, py)

    tel = browser.wait_for_telemetry(
        lambda t: t.get("trackingState") == "tracking" and int(t.get("inliers", 0)) > 0,
        timeout=12.0,
    )
    assert tel["trackingState"] == "tracking"
    assert int(tel["inliers"]) > 0
    browser.assert_zero_errors()
