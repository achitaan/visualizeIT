"""Tier 1 Feature Happy-Path E2E Tests for F13-F16 (REST API, WebSocket Stream, Browser Studio, Live Camera).

Strictly opaque-box: interacts only with a live `python -m visualizeit serve` subprocess via HTTP/1.1,
RFC 6455 WebSocket, and headless `/usr/bin/chromium` over CDP (`PROJECT.md §5 & §6`).
Zero imports of `visualizeit.*` or top-level non-stdlib third-party libraries (other than `pytest`).
"""

from __future__ import annotations

import base64
import json
import math
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.e2e._helpers import (
    HeadlessChromiumSession,
    StudioAPIClient,
    assert_video_run_report_invariants,
    bbox_at_frame,
    compute_roi_and_bg_delta,
    decode_frames,
    decode_image_bytes,
    make_rgba_png,
    make_textured_tracking_clip,
    make_y4m_camera_clip,
    probe_video,
)

pytestmark = pytest.mark.tier1

REQUIRED_STUDIO_TESTIDS: tuple[str, ...] = (
    # Mode selectors (2)
    "mode-video",
    "mode-camera",
    # Video controls (6)
    "demo-clip-select",
    "load-demo-btn",
    "video-file-input",
    "play-pause-btn",
    "export-mp4-btn",
    "export-download-link",
    # Selection controls (5)
    "stage-canvas",
    "select-mode-point",
    "select-mode-box",
    "auto-select-btn",
    "clear-selection-btn",
    # Design & placement controls (9)
    "design-prompt-input",
    "apply-prompt-btn",
    "pattern-select",
    "design-image-input",
    "wrap-mode-select",
    "blend-mode-select",
    "opacity-slider",
    "scale-slider",
    "rotation-slider",
    # Telemetry HUD (4)
    "hud-fps",
    "hud-latency-ms",
    "hud-tracking-state",
    "hud-processed-frames",
)

REQUIRED_TELEMETRY_KEYS: tuple[str, ...] = (
    "mode",
    "playing",
    "currentFps",
    "medianFrameMs",
    "p95FrameMs",
    "processedFrames",
    "trackingState",
    "inliers",
    "wrapMode",
    "blendMode",
    "lastExportReport",
)


def _upload_video_file(client: StudioAPIClient, video_path: Path) -> dict[str, Any]:
    resp = client.upload(
        "/api/videos/upload",
        filename=video_path.name,
        content=video_path.read_bytes(),
        content_type="video/mp4",
        expected_status=200,
        timeout=30.0,
    )
    data = json.loads(resp["body"].decode("utf-8"))
    assert isinstance(data, dict) and data.get("video_id"), f"Invalid upload response: {data!r}"
    return data


def _resolve_or_upload_demo_video(
    client: StudioAPIClient,
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    clip_id: str,
) -> tuple[str, dict[str, Any]]:
    meta = demo_corpus_manifest[clip_id]
    probe_resp = client.get(f"/api/videos/{clip_id}", timeout=10.0)
    if probe_resp["status"] == 200:
        return clip_id, meta
    clip_path = Path(meta.get("path") or (demo_corpus_dir / f"{clip_id}.mp4"))
    if not clip_path.is_file():
        clip_path = demo_corpus_dir / f"{clip_id}.mp4"
    up = _upload_video_file(client, clip_path)
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
        time.sleep(0.1)
    raise AssertionError(
        f"stage-canvas did not render non-blank content within {timeout}s: {last_sample}"
    )


def _ensure_playing(browser: HeadlessChromiumSession, *, timeout: float = 10.0) -> dict[str, Any]:
    tel = browser.get_telemetry(timeout=5.0)
    if not tel or not tel.get("playing"):
        browser.click_testid("play-pause-btn", timeout=timeout)
    return browser.wait_for_telemetry(lambda t: bool(t.get("playing")), timeout=timeout)


def _always_inside_y4m_point(y4m_meta: dict[str, Any]) -> tuple[float, float]:
    """Return `(x, y)` guaranteed inside the moving `.y4m` card at every visible frame of the loop."""
    x = float(y4m_meta["x0"]) + (float(y4m_meta["span_x"]) + float(y4m_meta["box_w"])) * 0.5
    y = float(y4m_meta["y0"]) + (float(y4m_meta["span_y"]) + float(y4m_meta["box_h"])) * 0.5
    return (x, y)


def _select_y4m_target_in_camera(
    browser: HeadlessChromiumSession,
    y4m_meta: dict[str, Any],
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    px, py = _always_inside_y4m_point(y4m_meta)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sample = browser.sample_stage_canvas(timeout=5.0)
        std_rgb = sample.get("std_rgb") or [0.0, 0.0, 0.0]
        if max(float(v) for v in std_rgb) > 15.0:
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
    return browser.wait_for_telemetry(
        lambda t: t.get("trackingState") == "tracking",
        timeout=3.0,
    )


def _export_and_fetch_mp4(
    browser: HeadlessChromiumSession,
    client: StudioAPIClient,
    out_dir: Path,
    *,
    timeout: float = 45.0,
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
        f"export-download-link missing valid populated href after export: {link_info}"
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
        dl_path = browser.wait_for_download(timeout=20.0, expected_suffix=".mp4")
        return dl_path, report

    dst = out_dir / "browser_exported.mp4"
    if href.startswith("blob:"):
        b64_data = browser.evaluate(
            f"""(async () => {{
                const r = await fetch({json.dumps(href)});
                if (!r.ok) throw new Error("Blob fetch failed: " + r.status);
                const buf = new Uint8Array(await r.arrayBuffer());
                let bin = "";
                for (let i = 0; i < buf.length; i += 8192) {{
                    bin += String.fromCharCode(...buf.subarray(i, i + 8192));
                }}
                return btoa(bin);
            }})()""",
            await_promise=True,
            timeout=20.0,
        )
        dst.write_bytes(base64.b64decode(b64_data))
        return dst, report

    parsed = urllib.parse.urlsplit(href)
    req_path = f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path
    dl_resp = client.get(req_path, expected_status=200, timeout=30.0)
    dst.write_bytes(dl_resp["body"])
    return dst, report


def _extract_first_float(text: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    assert m is not None, f"Expected numeric value in HUD text, got {text!r}"
    return float(m.group(0))


# ==============================================================================
# F13 — FastAPI HTTP/1.1 Video, Design & Render REST API (6 tests)
# ==============================================================================


@pytest.mark.f13
def test_f13_health_demos_and_design_presets_keepalive_http11(
    live_studio_server: StudioAPIClient,
) -> None:
    """Verify HTTP/1.1 Keep-Alive connection reuse across /api/health, /api/demos, and /api/designs/presets."""
    live_studio_server.close()
    start_idx = len(live_studio_server.client_ports_seen)

    health = live_studio_server.get_json("/api/health", expected_status=200)
    demos = live_studio_server.get_json("/api/demos", expected_status=200)
    presets = live_studio_server.get_json("/api/designs/presets", expected_status=200)

    ports_used = live_studio_server.client_ports_seen[start_idx:]
    assert len(ports_used) == 3, f"Expected 3 recorded requests on keep-alive client, got {ports_used}"
    assert len(set(ports_used)) == 1, (
        f"Expected persistent HTTP/1.1 keep-alive single TCP socket reuse, got distinct ports {ports_used}"
    )

    assert health.get("status") == "ok"
    assert health.get("version") == "1.0.0"
    assert health.get("native_raster_available") is True
    assert health.get("demo_clips_ready") is True

    clips = demos.get("clips")
    assert isinstance(clips, list) and len(clips) >= 5, f"Expected >=5 demo clips, got {demos!r}"
    clip_ids = {str(c.get("id")) for c in clips if isinstance(c, dict)}
    assert "planar_card_640x480" in clip_ids

    preset_list = presets.get("presets")
    assert isinstance(preset_list, list) and len(preset_list) >= 1, (
        f"Expected non-empty design presets, got {presets!r}"
    )


@pytest.mark.f13
def test_f13_video_upload_and_metadata_retrieval_roundtrip(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify multipart MP4 upload to POST /api/videos/upload and metadata lookup via GET /api/videos/{id}."""
    clip_path = e2e_tmp_dir / "upload_roundtrip.mp4"
    make_textured_tracking_clip(clip_path, width=640, height=480, frames=30, fps=30, tmp_dir=e2e_tmp_dir)

    uploaded = _upload_video_file(live_studio_server, clip_path)
    video_id = str(uploaded["video_id"])
    assert uploaded["width"] == 640
    assert uploaded["height"] == 480
    assert abs(float(uploaded["fps"]) - 30.0) < 0.5
    assert int(uploaded["frame_count"]) == 30
    assert float(uploaded["duration_s"]) > 0.8
    assert isinstance(uploaded.get("media_url"), str) and uploaded["media_url"].startswith("/")

    fetched = live_studio_server.get_json(f"/api/videos/{video_id}", expected_status=200)
    assert fetched["width"] == 640
    assert fetched["height"] == 480
    assert int(fetched["frame_count"]) == 30
    assert abs(float(fetched["fps"]) - 30.0) < 0.5


@pytest.mark.f13
def test_f13_media_endpoint_supports_full_200_and_byte_range_206(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify GET /api/videos/{id}/media supports full 200 download with Accept-Ranges and 206 Partial Content."""
    clip_path = e2e_tmp_dir / "range_test.mp4"
    make_textured_tracking_clip(clip_path, width=640, height=480, frames=24, fps=30, tmp_dir=e2e_tmp_dir)
    raw_bytes = clip_path.read_bytes()
    uploaded = _upload_video_file(live_studio_server, clip_path)
    video_id = str(uploaded["video_id"])

    full_resp = live_studio_server.get(f"/api/videos/{video_id}/media", expected_status=200)
    assert "bytes" in full_resp["headers"].get("accept-ranges", "").lower(), (
        f"Missing Accept-Ranges: bytes header on media response: {full_resp['headers']}"
    )
    assert "video" in full_resp["headers"].get("content-type", "").lower()
    full_body = full_resp["body"]
    assert len(full_body) == len(raw_bytes) and full_body == raw_bytes

    r1 = live_studio_server.get_range(f"/api/videos/{video_id}/media", start=0, end=1023, expected_status=206)
    cr1 = r1["headers"].get("content-range", "")
    assert cr1.startswith(f"bytes 0-1023/{len(full_body)}"), f"Unexpected Content-Range: {cr1!r}"
    assert len(r1["body"]) == 1024
    assert r1["body"] == full_body[:1024]

    r2 = live_studio_server.get_range(f"/api/videos/{video_id}/media", start=512, end=1535, expected_status=206)
    assert len(r2["body"]) == 1024
    assert r2["body"] == full_body[512:1536]


@pytest.mark.f13
def test_f13_frame_extraction_endpoint_returns_valid_jpeg_at_exact_dimensions(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify GET /api/videos/{id}/frame/{idx} returns valid JPEG images matching source dimensions and motion."""
    import numpy as np

    clip_path = e2e_tmp_dir / "frames_extract.mp4"
    make_textured_tracking_clip(clip_path, width=640, height=480, frames=24, fps=30, tmp_dir=e2e_tmp_dir)
    uploaded = _upload_video_file(live_studio_server, clip_path)
    video_id = str(uploaded["video_id"])

    resp0 = live_studio_server.get(f"/api/videos/{video_id}/frame/0", expected_status=200)
    resp15 = live_studio_server.get(f"/api/videos/{video_id}/frame/15", expected_status=200)
    assert "image/jpeg" in resp0["headers"].get("content-type", "").lower()
    assert "image/jpeg" in resp15["headers"].get("content-type", "").lower()

    arr0, mode0 = decode_image_bytes(resp0["body"])
    arr15, mode15 = decode_image_bytes(resp15["body"])
    assert mode0 == "RGB" and mode15 == "RGB"
    assert arr0.shape == (480, 640, 3)
    assert arr15.shape == (480, 640, 3)

    inter_frame_diff = float(np.abs(arr15.astype(np.float32) - arr0.astype(np.float32)).mean())
    assert inter_frame_diff > 2.0, (
        f"Expected distinct extracted frames 0 vs 15 on moving clip, got mean_abs_diff={inter_frame_diff:.3f}"
    )


@pytest.mark.f13
def test_f13_design_render_and_select_then_batch_mp4_render(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify POST /api/designs/render -> POST /api/videos/{id}/select -> POST /api/videos/{id}/render workflow."""
    clip_path = e2e_tmp_dir / "batch_render_src.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path, width=640, height=480, frames=30, fps=30, tmp_dir=e2e_tmp_dir
    )
    uploaded = _upload_video_file(live_studio_server, clip_path)
    video_id = str(uploaded["video_id"])

    design_resp = live_studio_server.post_json(
        "/api/designs/render",
        {
            "pattern": "stripes",
            "colors": ["#ff595e", "#1982c4", "#ffca3a"],
            "text": "STUDIO",
            "prompt": "",
            "width": 512,
            "height": 512,
        },
        expected_status=200,
    )
    design_id = str(design_resp["design_id"])
    assert design_id
    assert isinstance(design_resp.get("recipe"), dict)
    img_resp = live_studio_server.get(str(design_resp["image_url"]), expected_status=200)
    design_arr, _ = decode_image_bytes(img_resp["body"])
    assert design_arr.shape[:2] == (512, 512)

    cx, cy = clip_meta["center_frame0"]
    bx, by, bw, bh = clip_meta["bbox_frame0"]
    sel_resp = live_studio_server.post_json(
        f"/api/videos/{video_id}/select",
        {
            "frame_index": 0,
            "positive_points": [[cx, cy]],
            "negative_points": [],
            "bbox": [bx, by, bw, bh],
        },
        expected_status=200,
    )
    selection_id = str(sel_resp["selection_id"])
    assert selection_id
    assert int(sel_resp["frame_index"]) == 0
    assert int(sel_resp["foreground_pixels"]) > 500
    assert 0.02 <= float(sel_resp["coverage_ratio"]) <= 0.90
    assert int(sel_resp["anchor_count"]) >= 4
    assert float(sel_resp["init_ms"]) >= 0.0
    mask_arr, _ = decode_image_bytes(base64.b64decode(sel_resp["mask_png_base64"]))
    assert mask_arr.shape[:2] == (480, 640)

    render_resp = live_studio_server.post_json(
        f"/api/videos/{video_id}/render",
        {
            "selection_id": selection_id,
            "design_id": design_id,
            "placement": {
                "wrap_mode": "planar",
                "blend_mode": "luminance",
                "opacity": 0.90,
                "scale": 1.0,
            },
        },
        expected_status=200,
        timeout=60.0,
    )
    render_id = str(render_resp["render_id"])
    assert render_id
    assert render_resp["status"] == "completed"
    assert render_resp["video_url"] == f"/api/renders/{render_id}/video.mp4"
    assert render_resp["report_url"] == f"/api/renders/{render_id}/report.json"
    assert_video_run_report_invariants(render_resp["report"], expect_output_video=True)
    assert render_resp["report"]["output_frame_count"] == 30


@pytest.mark.f13
def test_f13_rendered_video_mp4_and_report_json_download_and_range(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify GET /api/renders/{id}/report.json and /video.mp4 (200 & 206 Range) deliver valid composited H.264."""
    clip_path = e2e_tmp_dir / "render_dl_src.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path, width=640, height=480, frames=24, fps=30, tmp_dir=e2e_tmp_dir
    )
    uploaded = _upload_video_file(live_studio_server, clip_path)
    video_id = str(uploaded["video_id"])

    bx, by, bw, bh = clip_meta["bbox_frame0"]
    render_resp = live_studio_server.post_json(
        f"/api/videos/{video_id}/render",
        {
            "selection": {
                "frame_index": 0,
                "positive_points": [list(clip_meta["center_frame0"])],
                "bbox": [bx, by, bw, bh],
            },
            "design_recipe": {
                "pattern": "checker",
                "colors": ["#ff006e", "#ffbe0b"],
                "width": 512,
                "height": 512,
            },
            "placement": {
                "wrap_mode": "planar",
                "blend_mode": "normal",
                "opacity": 0.95,
            },
        },
        expected_status=200,
        timeout=60.0,
    )
    render_id = str(render_resp["render_id"])

    fetched_report = live_studio_server.get_json(
        f"/api/renders/{render_id}/report.json", expected_status=200
    )
    assert_video_run_report_invariants(fetched_report, expect_output_video=True)
    assert fetched_report["output_frame_count"] == render_resp["report"]["output_frame_count"] == 24

    mp4_full = live_studio_server.get(f"/api/renders/{render_id}/video.mp4", expected_status=200)
    assert "video/mp4" in mp4_full["headers"].get("content-type", "").lower()
    mp4_range = live_studio_server.get_range(
        f"/api/renders/{render_id}/video.mp4", start=0, end=511, expected_status=206
    )
    assert len(mp4_range["body"]) == 512
    assert mp4_range["body"] == mp4_full["body"][:512]

    out_mp4 = e2e_tmp_dir / "downloaded_render.mp4"
    out_mp4.write_bytes(mp4_full["body"])
    probed = probe_video(out_mp4, tmp_dir=e2e_tmp_dir)
    assert probed["codec_name"] == "h264"
    assert probed["width"] == 640 and probed["height"] == 480
    assert probed["frame_count"] == 24

    raw_frames = decode_frames(clip_path, [0, 12])
    out_frames = decode_frames(out_mp4, [0, 12])
    for f_idx in (0, 12):
        roi_diff, bg_diff = compute_roi_and_bg_delta(
            raw_frames[f_idx], out_frames[f_idx], bbox_at_frame(clip_meta, f_idx)
        )
        assert roi_diff >= 10.0, f"Frame {f_idx} missing rendered artwork delta inside ROI: {roi_diff:.2f}"
        assert bg_diff <= 3.5, f"Frame {f_idx} background unexpectedly altered: {bg_diff:.2f}"


# ==============================================================================
# F14 — Real-Time WebSocket Video Processing Stream (6 tests)
# ==============================================================================


@pytest.mark.f14
def test_f14_websocket_stream_delivers_all_frames_in_order_and_complete_msg(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify WS /ws/videos/{id}/stream emits frames 0..N-1 in strict order followed by complete VideoRunReport."""
    clip_path = e2e_tmp_dir / "ws_order.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path, width=640, height=480, frames=24, fps=30, tmp_dir=e2e_tmp_dir
    )
    uploaded = _upload_video_file(live_studio_server, clip_path)
    video_id = str(uploaded["video_id"])

    frames, report = live_studio_server.stream_video_ws(
        video_id,
        {
            "selection": {
                "frame_index": 0,
                "positive_points": [list(clip_meta["center_frame0"])],
                "bbox": list(clip_meta["bbox_frame0"]),
            },
            "design_recipe": {
                "pattern": "stripes",
                "colors": ["#ff595e", "#ffca3a"],
            },
            "placement": {
                "wrap_mode": "planar",
                "blend_mode": "normal",
                "opacity": 0.90,
            },
        },
        frame_timeout=10.0,
        total_timeout=45.0,
    )

    assert len(frames) == 24, f"Expected 24 streamed frames, got {len(frames)}"
    indices = [int(f["frame_index"]) for f in frames]
    assert indices == list(range(24)), f"Out-of-order WebSocket frame indices: {indices}"

    assert_video_run_report_invariants(report, expect_output_video=False)
    assert report["input_frame_count"] == 24
    assert report["processed_frame_count"] == 24
    assert report["rendered_frame_count"] == 24


@pytest.mark.f14
def test_f14_websocket_frame_payloads_decode_to_valid_composited_jpegs(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify streamed WebSocket frame payloads decode to valid 640x480 JPEGs with composited artwork."""
    clip_path = e2e_tmp_dir / "ws_jpeg.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path, width=640, height=480, frames=24, fps=30, tmp_dir=e2e_tmp_dir
    )
    uploaded = _upload_video_file(live_studio_server, clip_path)
    video_id = str(uploaded["video_id"])

    frames, _ = live_studio_server.stream_video_ws(
        video_id,
        {
            "selection": {
                "frame_index": 0,
                "positive_points": [list(clip_meta["center_frame0"])],
                "bbox": list(clip_meta["bbox_frame0"]),
            },
            "design_recipe": {
                "pattern": "checker",
                "colors": ["#ff006e", "#00f5d4"],
                "width": 512,
                "height": 512,
            },
            "placement": {
                "wrap_mode": "planar",
                "blend_mode": "normal",
                "opacity": 0.95,
            },
        },
        frame_timeout=10.0,
        total_timeout=45.0,
    )

    raw_frames = decode_frames(clip_path, [0, 10, 20])
    for idx in (0, 10, 20):
        jpeg_bytes = frames[idx].get("jpeg_bytes")
        assert isinstance(jpeg_bytes, (bytes, bytearray)) and len(jpeg_bytes) > 500
        arr, mode = decode_image_bytes(bytes(jpeg_bytes))
        assert mode == "RGB" and arr.shape == (480, 640, 3)
        roi_diff, bg_diff = compute_roi_and_bg_delta(
            raw_frames[idx], arr, bbox_at_frame(clip_meta, idx)
        )
        assert roi_diff >= 12.0, f"Streamed frame {idx} missing artwork delta inside ROI: {roi_diff:.2f}"
        assert bg_diff <= 4.5, f"Streamed frame {idx} background corrupted: {bg_diff:.2f}"


@pytest.mark.f14
def test_f14_websocket_per_frame_telemetry_fields_valid_and_finite(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify every WebSocket frame message carries valid state, finite fps > 0, latency_ms > 0, and inliers >= 0."""
    clip_path = e2e_tmp_dir / "ws_telemetry.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path, width=640, height=480, frames=20, fps=30, tmp_dir=e2e_tmp_dir
    )
    uploaded = _upload_video_file(live_studio_server, clip_path)
    video_id = str(uploaded["video_id"])

    frames, _ = live_studio_server.stream_video_ws(
        video_id,
        {
            "selection": {
                "frame_index": 0,
                "positive_points": [list(clip_meta["center_frame0"])],
                "bbox": list(clip_meta["bbox_frame0"]),
            },
            "design_recipe": {"pattern": "dots", "colors": ["#ff595e", "#1982c4"]},
            "placement": {"wrap_mode": "planar", "blend_mode": "luminance"},
        },
        frame_timeout=10.0,
        total_timeout=45.0,
    )

    assert len(frames) == 20
    tracked_count = 0
    for f in frames:
        assert f.get("state") in {"tracking", "recovering", "lost"}, f"Invalid frame state: {f}"
        fps_val = float(f["fps"])
        lat_val = float(f["latency_ms"])
        inliers = f["inliers"]
        assert math.isfinite(fps_val) and fps_val > 0.0, f"Invalid frame fps: {f}"
        assert math.isfinite(lat_val) and lat_val > 0.0, f"Invalid frame latency_ms: {f}"
        assert isinstance(inliers, int) and not isinstance(inliers, bool) and inliers >= 0
        if f["state"] == "tracking":
            tracked_count += 1
    assert tracked_count >= 17, f"Expected >=17/20 tracked frames on clean planar clip, got {tracked_count}"


@pytest.mark.f14
def test_f14_websocket_stream_sustains_at_least_15_fps_over_loopback(
    live_studio_server: StudioAPIClient,
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
) -> None:
    """Verify full >=60-frame 640x480 WebSocket streaming sustains >= 15.0 FPS both reported and wall-clock."""
    video_id, demo_meta = _resolve_or_upload_demo_video(
        live_studio_server, demo_corpus_dir, demo_corpus_manifest, "planar_card_640x480"
    )
    default_sel = demo_meta.get("default_selection") or {}

    outcome = live_studio_server.stream_video_ws_raw(
        video_id,
        {
            "selection": default_sel,
            "design_recipe": {
                "pattern": "stripes",
                "colors": ["#ff595e", "#1982c4", "#ffca3a"],
            },
            "placement": {
                "wrap_mode": "planar",
                "blend_mode": "luminance",
                "opacity": 0.90,
            },
        },
        frame_timeout=10.0,
        total_timeout=60.0,
    )
    assert outcome["upgraded"] is True
    assert outcome["error_message"] is None
    report = outcome["complete_report"]
    assert isinstance(report, dict), f"Missing complete report: {outcome}"

    frames = outcome["frames"]
    assert_video_run_report_invariants(report, require_min_fps=15.0, expect_output_video=False)
    assert len(frames) == report["input_frame_count"] >= 60
    assert report["processed_frame_count"] == report["input_frame_count"]

    client_fps = len(frames) / max(1e-6, float(outcome["wall_time_s"]))
    assert client_fps >= 15.0, (
        f"Client-observed WebSocket streaming FPS {client_fps:.2f} fell below 15.0 FPS "
        f"({len(frames)} frames in {outcome['wall_time_s']:.3f}s)"
    )


@pytest.mark.f14
def test_f14_websocket_max_frames_parameter_bounds_stream_cleanly(
    live_studio_server: StudioAPIClient,
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
) -> None:
    """Verify max_frames=15 bounds a >=60-frame video stream to exact frames 0..14 and clean complete/close."""
    video_id, demo_meta = _resolve_or_upload_demo_video(
        live_studio_server, demo_corpus_dir, demo_corpus_manifest, "planar_card_640x480"
    )
    outcome = live_studio_server.stream_video_ws_raw(
        video_id,
        {
            "selection": demo_meta.get("default_selection") or {},
            "design_recipe": {"pattern": "waves", "colors": ["#1982c4", "#8ac926"]},
            "placement": {"wrap_mode": "planar", "blend_mode": "normal"},
            "max_frames": 15,
        },
        frame_timeout=10.0,
        total_timeout=30.0,
    )
    assert outcome["upgraded"] is True
    assert outcome["error_message"] is None
    frames = outcome["frames"]
    report = outcome["complete_report"]
    assert len(frames) == 15, f"Expected exactly 15 frames with max_frames=15, got {len(frames)}"
    assert [int(f["frame_index"]) for f in frames] == list(range(15))
    assert isinstance(report, dict)
    assert_video_run_report_invariants(report, expect_output_video=False)
    assert report["processed_frame_count"] == 15
    assert report["rendered_frame_count"] == 15
    assert outcome["close_code"] in (1000, None)


@pytest.mark.f14
def test_f14_websocket_supports_precreated_selection_id_and_design_id(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify WebSocket streaming accepts pre-created REST selection_id and uploaded RGBA design_id."""
    clip_path = e2e_tmp_dir / "ws_ids_src.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path, width=640, height=480, frames=20, fps=30, tmp_dir=e2e_tmp_dir
    )
    uploaded_vid = _upload_video_file(live_studio_server, clip_path)
    video_id = str(uploaded_vid["video_id"])

    png_path = make_rgba_png(e2e_tmp_dir / "ws_emblem.png", width=256, height=256, mode="emblem")
    design_up = live_studio_server.upload(
        "/api/designs/upload",
        filename="ws_emblem.png",
        content=png_path.read_bytes(),
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
            "bbox": list(clip_meta["bbox_frame0"]),
        },
        expected_status=200,
    )
    selection_id = str(sel_json["selection_id"])

    frames, report = live_studio_server.stream_video_ws(
        video_id,
        {
            "selection_id": selection_id,
            "design_id": design_id,
            "placement": {
                "wrap_mode": "planar",
                "blend_mode": "normal",
                "opacity": 0.95,
            },
        },
        frame_timeout=10.0,
        total_timeout=45.0,
    )
    assert len(frames) == 20
    assert_video_run_report_invariants(report, expect_output_video=False)

    raw_f5 = decode_frames(clip_path, [5])[5]
    ws_f5, _ = decode_image_bytes(bytes(frames[5]["jpeg_bytes"]))
    roi_diff, bg_diff = compute_roi_and_bg_delta(raw_f5, ws_f5, bbox_at_frame(clip_meta, 5))
    assert roi_diff >= 8.0, f"Uploaded RGBA emblem missing inside streamed ROI on frame 5: {roi_diff:.2f}"
    assert bg_diff <= 4.5, f"Streamed background altered outside ROI: {bg_diff:.2f}"


# ==============================================================================
# F15 — Unified Browser Video Studio (>=15 FPS Playback & MP4 Export) (6 tests)
# ==============================================================================


@pytest.mark.f15
def test_f15_studio_index_loads_all_required_data_testids_with_zero_errors(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
) -> None:
    """Verify GET / serves Unified Web Studio with all 26 required data-testid controls and zero JS errors."""
    index_resp = live_studio_server.get("/", expected_status=200)
    assert "text/html" in index_resp["headers"].get("content-type", "").lower()

    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)

    for testid in REQUIRED_STUDIO_TESTIDS:
        info = browser.wait_for_testid(testid, timeout=8.0)
        assert info["tag"], f"Empty tag for required data-testid={testid!r}"

    tel = browser.get_telemetry(timeout=5.0)
    assert isinstance(tel, dict), "window.__VISUALIZEIT_TELEMETRY__ missing or null on Studio load"
    browser.assert_zero_errors()


@pytest.mark.f15
def test_f15_load_demo_clip_and_auto_select_renders_stage_canvas(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
) -> None:
    """Verify selecting planar_card_640x480, clicking load-demo-btn, and auto-select-btn renders stage-canvas."""
    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)

    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)

    browser.click_testid("auto-select-btn")
    sample = browser.sample_stage_canvas(timeout=8.0)
    assert int(sample["width"]) >= 320 and int(sample["height"]) >= 240
    assert float(sample["non_black_ratio"]) > 0.25
    assert max(float(s) for s in sample["std_rgb"]) > 10.0, (
        f"Expected non-uniform rendered stage-canvas, got std_rgb={sample['std_rgb']}"
    )
    browser.assert_zero_errors()


@pytest.mark.f15
def test_f15_interactive_point_and_box_selection_controls_on_stage_canvas(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    demo_corpus_manifest: dict[str, dict[str, Any]],
) -> None:
    """Verify select-mode-point click, clear-selection-btn, and select-mode-box drag on stage-canvas."""
    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)

    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    stage = _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    cw, ch = float(stage["width"]), float(stage["height"])

    default_sel = demo_corpus_manifest["planar_card_640x480"].get("default_selection") or {}
    pts = default_sel.get("positive_points") or [[cw * 0.5, ch * 0.5]]
    px, py = float(pts[0][0]), float(pts[0][1])
    bbox = default_sel.get("bbox") or [cw * 0.25, ch * 0.25, cw * 0.4, ch * 0.4]

    browser.click_testid("select-mode-point")
    browser.click_canvas_point(px, py)
    _ensure_playing(browser)
    tel_pt = browser.wait_for_telemetry(
        lambda t: t.get("trackingState") == "tracking" or int(t.get("inliers", 0)) > 0,
        timeout=12.0,
    )
    assert tel_pt["trackingState"] in {"tracking", "recovering"}

    browser.click_testid("clear-selection-btn")
    tel_cleared = browser.wait_for_telemetry(
        lambda t: t.get("trackingState") != "tracking" and int(t.get("inliers", 0)) == 0,
        timeout=8.0,
    )
    assert tel_cleared["trackingState"] != "tracking"

    browser.click_testid("select-mode-box")
    browser.drag_canvas_box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    tel_box = browser.wait_for_telemetry(
        lambda t: t.get("trackingState") == "tracking",
        timeout=12.0,
    )
    assert tel_box["trackingState"] == "tracking"
    browser.assert_zero_errors()


@pytest.mark.f15
def test_f15_design_prompt_pattern_and_placement_sliders_update_live_state(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
) -> None:
    """Verify prompt input, pattern select, wrap/blend selects, and placement sliders update live state & canvas."""
    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)

    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    browser.click_testid("auto-select-btn")
    browser.pump_events(0.25)
    before_sample = browser.sample_stage_canvas(timeout=8.0)

    browser.select_testid("pattern-select", "checker")
    browser.fill_testid("design-prompt-input", 'orange and teal waves text "LIVE"')
    browser.click_testid("apply-prompt-btn")
    browser.select_testid("wrap-mode-select", "cylinder")
    browser.select_testid("blend-mode-select", "luminance")
    browser.set_slider_testid("opacity-slider", 0.85)
    browser.set_slider_testid("scale-slider", 1.35)
    browser.set_slider_testid("rotation-slider", 25)

    tel = browser.wait_for_telemetry(
        lambda t: t.get("wrapMode") == "cylinder" and t.get("blendMode") == "luminance",
        timeout=10.0,
    )
    assert tel["wrapMode"] == "cylinder"
    assert tel["blendMode"] == "luminance"

    _ensure_playing(browser)
    browser.wait_for_telemetry(lambda t: int(t.get("processedFrames", 0)) >= 5, timeout=10.0)
    after_sample = browser.sample_stage_canvas(timeout=8.0)
    assert int(after_sample["width"]) > 0 and float(after_sample["non_black_ratio"]) > 0.25
    assert after_sample["png_base64"] != before_sample["png_base64"]
    browser.assert_zero_errors()


@pytest.mark.f15
def test_f15_recorded_video_playback_sustains_at_least_15fps_in_worker(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
) -> None:
    """Verify recorded video playback in Browser Studio sustains >= 15.0 FPS (medianFrameMs <= 66.67)."""
    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)

    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    browser.click_testid("auto-select-btn")
    _ensure_playing(browser)

    time.sleep(2.5)
    browser.pump_events(0.05)
    tel = browser.get_telemetry(timeout=5.0)
    assert tel is not None
    assert tel.get("mode") == "video"
    assert tel.get("playing") is True
    assert int(tel.get("processedFrames", 0)) >= 35, f"Expected >=35 processedFrames in 2.5s, got {tel}"
    assert float(tel.get("currentFps", 0.0)) >= 15.0, f"Browser video FPS < 15.0: {tel}"
    assert float(tel.get("medianFrameMs", 999.0)) <= 66.67, f"medianFrameMs > 66.67ms: {tel}"
    assert tel.get("trackingState") == "tracking"
    browser.assert_zero_errors()


@pytest.mark.f15
def test_f15_export_mp4_button_produces_valid_downloadable_rendered_video(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify clicking export-mp4-btn populates export-download-link & lastExportReport and yields valid H.264 MP4."""
    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)

    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    browser.click_testid("auto-select-btn")
    browser.select_testid("pattern-select", "checker")

    mp4_path, report = _export_and_fetch_mp4(
        browser, live_studio_server, e2e_tmp_dir, timeout=60.0
    )
    assert_video_run_report_invariants(report, expect_output_video=True)
    probed = probe_video(mp4_path, tmp_dir=e2e_tmp_dir)
    assert probed["codec_name"] == "h264"
    assert probed["width"] == 640 and probed["height"] == 480
    assert probed["frame_count"] == report["output_frame_count"]
    browser.assert_zero_errors()


# ==============================================================================
# F16 — Uncapped Live Camera Mode & Telemetry HUD (6 tests)
# ==============================================================================


@pytest.mark.f16
def test_f16_mode_camera_activates_fake_getusermedia_stream_cleanly(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify clicking mode-camera starts getUserMedia camera feed with playing=true and zero errors."""
    y4m_path = e2e_tmp_dir / "cam_activate.y4m"
    make_y4m_camera_clip(y4m_path, width=640, height=480, frames=60, fps=30, tmp_dir=e2e_tmp_dir)

    browser = chromium_session_factory(fake_video_y4m=y4m_path)
    browser.navigate(live_studio_server.base_url)
    browser.click_testid("mode-camera")

    tel = browser.wait_for_telemetry(
        lambda t: t.get("mode") == "camera" and bool(t.get("playing")) and int(t.get("processedFrames", 0)) >= 5,
        timeout=12.0,
    )
    assert tel["mode"] == "camera"
    assert tel["playing"] is True

    sample = _wait_for_stage_ready(browser, min_non_black=0.20, timeout=8.0)
    assert int(sample["width"]) >= 320 and int(sample["height"]) >= 240
    browser.assert_zero_errors()


@pytest.mark.f16
def test_f16_live_camera_runs_uncapped_without_legacy_66ms_throttle(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify live camera mode feeds a 60fps stream uncapped (>=15.0 FPS, >=30 frames, no 66ms fixed timer cap)."""
    y4m_path = e2e_tmp_dir / "cam_60fps.y4m"
    y4m_meta = make_y4m_camera_clip(
        y4m_path, width=320, height=240, frames=90, fps=60, tmp_dir=e2e_tmp_dir
    )

    browser = chromium_session_factory(fake_video_y4m=y4m_path)
    browser.navigate(live_studio_server.base_url)
    browser.click_testid("mode-camera")
    _wait_for_stage_ready(browser, min_non_black=0.20, timeout=10.0)
    _select_y4m_target_in_camera(browser, y4m_meta, timeout=10.0)

    t0 = browser.get_telemetry(timeout=5.0)
    assert t0 is not None
    start_frames = int(t0.get("processedFrames", 0))

    time.sleep(2.2)
    browser.pump_events(0.05)
    t1 = browser.get_telemetry(timeout=5.0)
    assert t1 is not None
    assert t1.get("mode") == "camera"
    assert t1.get("playing") is True

    delta_frames = int(t1.get("processedFrames", 0)) - start_frames
    assert int(t1.get("processedFrames", 0)) >= 30 and delta_frames >= 30, (
        f"Expected continuous camera frame advancement >=30 frames, got delta={delta_frames}, t1={t1}"
    )
    assert float(t1.get("currentFps", 0.0)) >= 15.0, f"Live camera FPS below 15.0 floor: {t1}"
    assert float(t1.get("medianFrameMs", 999.0)) < 66.0, (
        f"Live camera appears throttled by legacy 66ms timer cap: medianFrameMs={t1.get('medianFrameMs')}"
    )
    browser.assert_zero_errors()


@pytest.mark.f16
def test_f16_hud_dom_elements_reflect_live_telemetry_hook_values(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify hud-fps, hud-latency-ms, hud-tracking-state, and hud-processed-frames reflect live telemetry."""
    y4m_path = e2e_tmp_dir / "cam_hud.y4m"
    y4m_meta = make_y4m_camera_clip(
        y4m_path, width=640, height=480, frames=60, fps=30, tmp_dir=e2e_tmp_dir
    )

    browser = chromium_session_factory(fake_video_y4m=y4m_path)
    browser.navigate(live_studio_server.base_url)
    browser.click_testid("mode-camera")
    _wait_for_stage_ready(browser, min_non_black=0.20, timeout=10.0)
    _select_y4m_target_in_camera(browser, y4m_meta, timeout=10.0)

    tel = browser.wait_for_telemetry(
        lambda t: int(t.get("processedFrames", 0)) >= 15 and float(t.get("currentFps", 0.0)) > 0.0,
        timeout=12.0,
    )

    hud_fps_txt = browser.get_testid_text("hud-fps")
    hud_lat_txt = browser.get_testid_text("hud-latency-ms")
    hud_state_txt = browser.get_testid_text("hud-tracking-state")
    hud_frames_txt = browser.get_testid_text("hud-processed-frames")

    assert hud_fps_txt and _extract_first_float(hud_fps_txt) > 0.0
    assert hud_lat_txt and _extract_first_float(hud_lat_txt) > 0.0
    assert hud_frames_txt and _extract_first_float(hud_frames_txt) >= 10.0
    assert hud_state_txt and str(tel["trackingState"]).lower() in hud_state_txt.lower(), (
        f"hud-tracking-state={hud_state_txt!r} inconsistent with telemetry state={tel['trackingState']!r}"
    )
    browser.assert_zero_errors()


@pytest.mark.f16
def test_f16_telemetry_global_object_exposes_all_eleven_contract_keys(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify window.__VISUALIZEIT_TELEMETRY__ directly exposes all 11 documented PROJECT.md §6 keys."""
    y4m_path = e2e_tmp_dir / "cam_keys.y4m"
    make_y4m_camera_clip(y4m_path, width=640, height=480, frames=45, fps=30, tmp_dir=e2e_tmp_dir)

    browser = chromium_session_factory(fake_video_y4m=y4m_path)
    browser.navigate(live_studio_server.base_url)
    browser.click_testid("mode-camera")
    browser.wait_for_telemetry(lambda t: int(t.get("processedFrames", 0)) >= 5, timeout=12.0)

    raw_inspection = browser.evaluate(
        f"""(() => {{
            const raw = window.__VISUALIZEIT_TELEMETRY__;
            if (!raw) return null;
            const obj = typeof raw === "function" ? raw() : raw;
            const required = {json.dumps(list(REQUIRED_TELEMETRY_KEYS))};
            const present = {{}};
            for (const k of required) {{
                present[k] = (k in obj);
            }}
            return {{ present, snapshot: JSON.parse(JSON.stringify(obj)) }};
        }})()""",
        await_promise=False,
        timeout=5.0,
    )
    assert isinstance(raw_inspection, dict), "window.__VISUALIZEIT_TELEMETRY__ is missing"
    missing = [k for k, ok in raw_inspection["present"].items() if not ok]
    assert not missing, f"window.__VISUALIZEIT_TELEMETRY__ missing required keys: {missing}"

    snap = raw_inspection["snapshot"]
    assert snap["mode"] in {"video", "camera"}
    assert isinstance(snap["playing"], bool)
    for num_key in ("currentFps", "medianFrameMs", "p95FrameMs"):
        assert isinstance(snap[num_key], (int, float)) and not isinstance(snap[num_key], bool)
        assert math.isfinite(float(snap[num_key]))
    for int_key in ("processedFrames", "inliers"):
        assert isinstance(snap[int_key], int) and not isinstance(snap[int_key], bool) and snap[int_key] >= 0
    assert isinstance(snap["trackingState"], str) and snap["trackingState"]
    assert isinstance(snap["wrapMode"], str) and snap["wrapMode"]
    assert isinstance(snap["blendMode"], str) and snap["blendMode"]
    assert snap["lastExportReport"] is None or isinstance(snap["lastExportReport"], dict)
    browser.assert_zero_errors()


@pytest.mark.f16
def test_f16_camera_mode_auto_reacquires_after_synthetic_occlusion_wipe(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify camera mode suppresses tracking during .y4m blackout and automatically re-acquires upon return."""
    y4m_path = e2e_tmp_dir / "cam_occlusion.y4m"
    y4m_meta = make_y4m_camera_clip(
        y4m_path,
        width=640,
        height=480,
        frames=75,
        fps=30,
        occlude_ranges=[(35, 52)],
        tmp_dir=e2e_tmp_dir,
    )

    browser = chromium_session_factory(fake_video_y4m=y4m_path)
    browser.navigate(live_studio_server.base_url)
    browser.click_testid("mode-camera")
    _wait_for_stage_ready(browser, min_non_black=0.10, timeout=10.0)
    _select_y4m_target_in_camera(browser, y4m_meta, timeout=12.0)

    saw_loss = False
    saw_recovery = False
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        tel = browser.get_telemetry(timeout=4.0)
        if tel is not None:
            st = str(tel.get("trackingState", ""))
            if not saw_loss and st in {"lost", "recovering"}:
                saw_loss = True
            elif saw_loss and st == "tracking" and int(tel.get("inliers", 0)) > 0:
                saw_recovery = True
                break
        time.sleep(0.06)

    assert saw_loss, "Camera tracker never transitioned to lost/recovering during full blackout occlusion"
    assert saw_recovery, "Camera tracker failed to auto-reacquire target after occlusion wipe ended"
    browser.assert_zero_errors()


@pytest.mark.f16
def test_f16_switching_camera_to_video_and_back_releases_tracks_without_leak(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify switching mode-video -> mode-camera -> mode-video -> mode-camera stops tracks and avoids leaks."""
    y4m_path = e2e_tmp_dir / "cam_switch.y4m"
    make_y4m_camera_clip(y4m_path, width=640, height=480, frames=60, fps=30, tmp_dir=e2e_tmp_dir)

    browser = chromium_session_factory(fake_video_y4m=y4m_path)
    browser.add_init_script(
        """(() => {
            window.__E2E_TRACKS__ = [];
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                const orig = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
                navigator.mediaDevices.getUserMedia = async (constraints) => {
                    const stream = await orig(constraints);
                    for (const t of stream.getTracks()) window.__E2E_TRACKS__.push(t);
                    return stream;
                };
            }
        })();"""
    )
    browser.navigate(live_studio_server.base_url)

    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    _ensure_playing(browser)
    browser.wait_for_telemetry(
        lambda t: t.get("mode") == "video" and int(t.get("processedFrames", 0)) >= 5,
        timeout=10.0,
    )

    browser.click_testid("mode-camera")
    tel_cam1 = browser.wait_for_telemetry(
        lambda t: t.get("mode") == "camera" and bool(t.get("playing")) and int(t.get("processedFrames", 0)) >= 5,
        timeout=12.0,
    )
    assert tel_cam1["mode"] == "camera"

    browser.click_testid("mode-video")
    browser.wait_for_telemetry(lambda t: t.get("mode") == "video", timeout=10.0)
    live_tracks_in_video_mode = browser.evaluate(
        "window.__E2E_TRACKS__.filter(t => t.readyState === 'live').length",
        await_promise=False,
    )
    assert int(live_tracks_in_video_mode) == 0, (
        f"Expected 0 live camera MediaStreamTracks after switching back to mode-video, got {live_tracks_in_video_mode}"
    )

    browser.click_testid("mode-camera")
    tel_cam2 = browser.wait_for_telemetry(
        lambda t: t.get("mode") == "camera" and bool(t.get("playing")),
        timeout=12.0,
    )
    f_before = int(tel_cam2.get("processedFrames", 0))
    tel_cam2_after = browser.wait_for_telemetry(
        lambda t: int(t.get("processedFrames", 0)) >= f_before + 5,
        timeout=10.0,
    )
    assert tel_cam2_after["mode"] == "camera"
    browser.assert_zero_errors()
