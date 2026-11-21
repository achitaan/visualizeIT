"""Tier 3 Pairwise & Multi-Feature Combination E2E Tests (T3-11 .. T3-20: REST, Range, WS, Browser Studio).

Exercises multi-feature interactions across procedural recipes, prompts, custom RGBA uploads, point/box
segmentation, planar/cylinder/mesh_grid/mesh_6dof surface wrapping, luminance/normal/ink blending,
occlusion suppression + auto-recovery, concurrent HTTP/1.1 Range + WebSocket streaming, and headless
Chromium Unified Studio + Live Camera workflows.
Zero imports of `visualizeit.*` or top-level non-stdlib third-party libraries (other than `pytest`).
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import json
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
    compute_psnr,
    compute_roi_and_bg_delta,
    decode_frames,
    decode_image_bytes,
    make_rgba_png,
    make_textured_tracking_clip,
    make_y4m_camera_clip,
    probe_video,
)

pytestmark = pytest.mark.tier3


def _upload_video(client: StudioAPIClient, video_path: Path) -> dict[str, Any]:
    resp = client.upload(
        "/api/videos/upload",
        filename=video_path.name,
        content=video_path.read_bytes(),
        content_type="video/mp4",
        expected_status=200,
        timeout=30.0,
    )
    return json.loads(resp["body"].decode("utf-8"))


def _upload_png(client: StudioAPIClient, png_path: Path) -> dict[str, Any]:
    resp = client.upload(
        "/api/designs/upload",
        filename=png_path.name,
        content=png_path.read_bytes(),
        content_type="image/png",
        expected_status=200,
        timeout=20.0,
    )
    return json.loads(resp["body"].decode("utf-8"))


def _resolve_or_upload_demo(
    client: StudioAPIClient,
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    clip_id: str,
) -> tuple[str, Path, dict[str, Any]]:
    meta = demo_corpus_manifest[clip_id]
    clip_path = Path(meta.get("path") or (demo_corpus_dir / f"{clip_id}.mp4"))
    if not clip_path.is_file():
        clip_path = demo_corpus_dir / f"{clip_id}.mp4"
    probe = client.get(f"/api/videos/{clip_id}", timeout=10.0)
    if probe["status"] == 200:
        return clip_id, clip_path, meta
    up = _upload_video(client, clip_path)
    return str(up["video_id"]), clip_path, meta


def _wait_for_stage_ready(
    browser: HeadlessChromiumSession,
    *,
    min_non_black: float = 0.10,
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
        f"export-download-link missing populated href: {link_info}"
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

    dst = out_dir / f"exported_{int(time.time() * 1000)}.mp4"
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
            timeout=20.0,
        )
        dst.write_bytes(base64.b64decode(b64_data))
        return dst, report

    parsed = urllib.parse.urlsplit(href)
    req_path = f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path
    dl_resp = client.get(req_path, expected_status=200, timeout=30.0)
    dst.write_bytes(dl_resp["body"])
    return dst, report


# ==============================================================================
# T3-11 .. T3-20 — REST, Range, WebSocket & Browser Studio Combinations
# ==============================================================================


@pytest.mark.f03
@pytest.mark.f06
@pytest.mark.f10
@pytest.mark.f13
def test_t3_11_rest_prompt_design_plus_point_neg_select_and_planar_render(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Combine REST natural-language prompt design + pos/neg point selection + planar luminance render + Range."""
    clip_path = e2e_tmp_dir / "t3_11_planar_shaded.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path, width=640, height=480, frames=30, fps=30, shaded_card=True, tmp_dir=e2e_tmp_dir
    )
    up = _upload_video(live_studio_server, clip_path)
    video_id = str(up["video_id"])

    design_resp = live_studio_server.post_json(
        "/api/designs/render",
        {
            "prompt": 'navy and yellow stripes text "BRAND" scale 1.2',
            "width": 512,
            "height": 512,
        },
        expected_status=200,
    )
    design_id = str(design_resp["design_id"])
    assert design_resp["recipe"].get("pattern") == "stripes"

    cx, cy = clip_meta["center_frame0"]
    sel_resp = live_studio_server.post_json(
        f"/api/videos/{video_id}/select",
        {
            "frame_index": 0,
            "positive_points": [[cx, cy]],
            "negative_points": [[24, 24], [616, 456]],
            "bbox": list(clip_meta["bbox_frame0"]),
        },
        expected_status=200,
    )
    selection_id = str(sel_resp["selection_id"])

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
    assert_video_run_report_invariants(render_resp["report"], expect_output_video=True)
    assert render_resp["report"]["wrap_mode"] == "planar"
    assert render_resp["report"]["blend_mode"] == "luminance"

    range_resp = live_studio_server.get_range(
        f"/api/renders/{render_id}/video.mp4", start=0, end=2047, expected_status=206
    )
    full_resp = live_studio_server.get(f"/api/renders/{render_id}/video.mp4", expected_status=200)
    assert range_resp["body"] == full_resp["body"][:2048]

    out_mp4 = e2e_tmp_dir / "t3_11_out.mp4"
    out_mp4.write_bytes(full_resp["body"])
    raw_f10 = decode_frames(clip_path, [10])[10]
    out_f10 = decode_frames(out_mp4, [10])[10]
    roi_diff, bg_diff = compute_roi_and_bg_delta(raw_f10, out_f10, bbox_at_frame(clip_meta, 10))
    assert roi_diff >= 8.0
    assert bg_diff <= 3.5


@pytest.mark.f04
@pytest.mark.f06
@pytest.mark.f10
@pytest.mark.f13
def test_t3_12_rest_rgba_png_upload_plus_box_select_and_cylinder_ink_render(
    live_studio_server: StudioAPIClient,
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """Combine custom die-cut RGBA PNG upload + bbox selection + cylindrical ink wrap via REST batch render."""
    rgba_path = make_rgba_png(
        e2e_tmp_dir / "t3_12_cutout.png", width=320, height=240, mode="cutout_label", alpha_value=255
    )
    design_up = _upload_png(live_studio_server, rgba_path)
    assert design_up["has_alpha"] is True
    design_id = str(design_up["design_id"])

    video_id, mug_path, mug_meta = _resolve_or_upload_demo(
        live_studio_server, demo_corpus_dir, demo_corpus_manifest, "cylinder_mug_640x480"
    )
    default_sel = mug_meta.get("default_selection") or {}
    bbox = default_sel.get("bbox") or [180, 100, 260, 280]

    sel_resp = live_studio_server.post_json(
        f"/api/videos/{video_id}/select",
        {
            "frame_index": int(default_sel.get("frame_index", 0)),
            "positive_points": default_sel.get("positive_points") or [],
            "negative_points": [],
            "bbox": bbox,
        },
        expected_status=200,
    )
    selection_id = str(sel_resp["selection_id"])

    render_resp = live_studio_server.post_json(
        f"/api/videos/{video_id}/render",
        {
            "selection_id": selection_id,
            "design_id": design_id,
            "placement": {
                "wrap_mode": "cylinder",
                "blend_mode": "ink",
                "scale": 1.35,
                "opacity": 0.85,
            },
        },
        expected_status=200,
        timeout=60.0,
    )
    report = render_resp["report"]
    assert_video_run_report_invariants(report, expect_output_video=True)
    assert report["wrap_mode"] == "cylinder"
    assert report["blend_mode"] == "ink"

    out_mp4 = e2e_tmp_dir / "t3_12_mug_ink.mp4"
    out_mp4.write_bytes(
        live_studio_server.get(str(render_resp["video_url"]), expected_status=200)["body"]
    )
    probed = probe_video(out_mp4, tmp_dir=e2e_tmp_dir)
    assert probed["frame_count"] == report["input_frame_count"]

    raw_f0 = decode_frames(mug_path, [0])[0]
    out_f0 = decode_frames(out_mp4, [0])[0]
    roi_diff, bg_diff = compute_roi_and_bg_delta(
        raw_f0, out_f0, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    )
    assert roi_diff >= 4.0, f"Expected cylindrical ink darkening inside mug bbox, got {roi_diff:.2f}"
    assert bg_diff <= 3.5


@pytest.mark.f05
@pytest.mark.f07
@pytest.mark.f10
@pytest.mark.f13
@pytest.mark.f14
def test_t3_13_websocket_stream_vs_rest_batch_render_telemetry_and_frame_parity(
    live_studio_server: StudioAPIClient,
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """Verify WebSocket streaming and REST batch render agree on frame counts, tracking, and visual frame 10."""
    video_id, _, demo_meta = _resolve_or_upload_demo(
        live_studio_server, demo_corpus_dir, demo_corpus_manifest, "planar_card_640x480"
    )
    default_sel = demo_meta.get("default_selection") or {}
    sel_resp = live_studio_server.post_json(
        f"/api/videos/{video_id}/select",
        default_sel,
        expected_status=200,
    )
    selection_id = str(sel_resp["selection_id"])

    design_resp = live_studio_server.post_json(
        "/api/designs/render",
        {
            "pattern": "checker",
            "colors": ["#ff595e", "#1982c4", "#ffca3a"],
            "width": 512,
            "height": 512,
        },
        expected_status=200,
    )
    design_id = str(design_resp["design_id"])
    placement = {"wrap_mode": "mesh_grid", "blend_mode": "luminance", "opacity": 0.90}

    rest_render = live_studio_server.post_json(
        f"/api/videos/{video_id}/render",
        {"selection_id": selection_id, "design_id": design_id, "placement": placement},
        expected_status=200,
        timeout=60.0,
    )
    rest_report = rest_render["report"]
    assert_video_run_report_invariants(rest_report, expect_output_video=True)

    ws_frames, ws_report = live_studio_server.stream_video_ws(
        video_id,
        {"selection_id": selection_id, "design_id": design_id, "placement": placement},
        frame_timeout=10.0,
        total_timeout=60.0,
    )
    assert_video_run_report_invariants(ws_report, expect_output_video=False)

    assert ws_report["input_frame_count"] == rest_report["input_frame_count"]
    assert ws_report["processed_frame_count"] == rest_report["processed_frame_count"] == len(ws_frames)
    assert abs(int(ws_report["tracked_frame_count"]) - int(rest_report["tracked_frame_count"])) <= 3

    rest_mp4 = e2e_tmp_dir / "t3_13_rest.mp4"
    rest_mp4.write_bytes(
        live_studio_server.get(str(rest_render["video_url"]), expected_status=200)["body"]
    )
    rest_f10 = decode_frames(rest_mp4, [10])[10]
    ws_f10, _ = decode_image_bytes(bytes(ws_frames[10]["jpeg_bytes"]))
    psnr_db = compute_psnr(ws_f10, rest_f10)
    assert psnr_db >= 24.0, (
        f"Expected close visual parity between WS frame 10 and REST MP4 frame 10, got PSNR={psnr_db:.2f} dB"
    )


@pytest.mark.f04
@pytest.mark.f08
@pytest.mark.f10
@pytest.mark.f13
@pytest.mark.f14
def test_t3_14_websocket_stream_custom_rgba_artwork_in_mesh_6dof_mode(
    live_studio_server: StudioAPIClient,
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """Stream mesh_box_640x480 over WebSocket with custom RGBA PNG in mesh_6dof mode at >= 15.0 FPS."""
    rgba_path = make_rgba_png(e2e_tmp_dir / "t3_14_emblem.png", width=256, height=256, mode="emblem")
    design_up = _upload_png(live_studio_server, rgba_path)
    design_id = str(design_up["design_id"])

    video_id, _, mesh_meta = _resolve_or_upload_demo(
        live_studio_server, demo_corpus_dir, demo_corpus_manifest, "mesh_box_640x480"
    )
    frames, report = live_studio_server.stream_video_ws(
        video_id,
        {
            "selection": mesh_meta.get("default_selection") or {},
            "design_id": design_id,
            "placement": {
                "wrap_mode": "mesh_6dof",
                "blend_mode": "normal",
                "opacity": 0.90,
            },
        },
        frame_timeout=12.0,
        total_timeout=60.0,
    )

    assert_video_run_report_invariants(report, require_min_fps=15.0, expect_output_video=False)
    assert report["wrap_mode"] == "mesh_6dof"
    assert len(frames) == report["input_frame_count"]
    assert any(int(f.get("inliers", 0)) > 0 and f.get("state") == "tracking" for f in frames)


@pytest.mark.f03
@pytest.mark.f09
@pytest.mark.f10
@pytest.mark.f13
@pytest.mark.f14
def test_t3_15_websocket_stream_prompt_recipe_across_occlusion_and_recovery(
    live_studio_server: StudioAPIClient,
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
) -> None:
    """Verify prompt recipe over WebSocket renders before occlusion, suppresses during blackout, and recovers."""
    import numpy as np

    design_resp = live_studio_server.post_json(
        "/api/designs/render",
        {
            "prompt": 'red and yellow checker text "LOCK"',
            "width": 512,
            "height": 512,
        },
        expected_status=200,
    )
    design_id = str(design_resp["design_id"])

    video_id, occ_mp4_path, occ_meta = _resolve_or_upload_demo(
        live_studio_server, demo_corpus_dir, demo_corpus_manifest, "occlusion_recovery_640x480"
    )
    frames, report = live_studio_server.stream_video_ws(
        video_id,
        {
            "selection": occ_meta.get("default_selection") or {},
            "design_id": design_id,
            "placement": {
                "wrap_mode": "planar",
                "blend_mode": "normal",
                "opacity": 0.95,
            },
        },
        frame_timeout=10.0,
        total_timeout=60.0,
    )
    assert_video_run_report_invariants(report, expect_output_video=False)
    assert int(report["lost_frame_count"]) >= 1
    assert int(report["recovered_count"]) >= 1

    lost_indices = [int(f["frame_index"]) for f in frames if f.get("state") != "tracking"]
    assert lost_indices, "Expected occluded non-tracking frames in occlusion_recovery_640x480 stream"
    mid_lost_idx = lost_indices[len(lost_indices) // 2]

    post_tracked = [
        int(f["frame_index"])
        for f in frames
        if int(f["frame_index"]) > lost_indices[-1] and f.get("state") == "tracking"
    ]
    assert post_tracked, "Expected post-occlusion recovered tracking frames"
    rec_idx = post_tracked[len(post_tracked) // 2]

    raw_map = decode_frames(occ_mp4_path, [2, mid_lost_idx, rec_idx])
    ws_f2, _ = decode_image_bytes(bytes(frames[2]["jpeg_bytes"]))
    ws_lost, _ = decode_image_bytes(bytes(frames[mid_lost_idx]["jpeg_bytes"]))
    ws_rec, _ = decode_image_bytes(bytes(frames[rec_idx]["jpeg_bytes"]))

    p99_f2 = float(np.percentile(np.abs(ws_f2.astype(np.float32) - raw_map[2].astype(np.float32)), 99.0))
    mad_lost = float(np.abs(ws_lost.astype(np.float32) - raw_map[mid_lost_idx].astype(np.float32)).mean())
    p99_rec = float(
        np.percentile(np.abs(ws_rec.astype(np.float32) - raw_map[rec_idx].astype(np.float32)), 99.0)
    )

    assert p99_f2 >= 20.0, f"Expected strong prompt artwork on frame 2 before occlusion, got p99={p99_f2:.2f}"
    assert mad_lost <= 4.0, f"Stale overlay leaked during occluded frame {mid_lost_idx}: MAD={mad_lost:.2f}"
    assert p99_rec >= 20.0, (
        f"Expected restored prompt artwork on recovered frame {rec_idx}, got p99={p99_rec:.2f}"
    )


@pytest.mark.f13
@pytest.mark.f14
def test_t3_16_concurrent_rest_range_download_during_active_websocket_stream(
    live_studio_server: StudioAPIClient,
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
) -> None:
    """Verify concurrent HTTP/1.1 206 Range fetches and /api/health calls alongside an active WS stream."""
    video_id, _, demo_meta = _resolve_or_upload_demo(
        live_studio_server, demo_corpus_dir, demo_corpus_manifest, "planar_card_640x480"
    )
    sel = demo_meta.get("default_selection") or {}

    def _ws_worker() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        with StudioAPIClient(
            host=live_studio_server.host,
            port=live_studio_server.port,
            server_proc=live_studio_server.server_proc,
        ) as ws_client:
            return ws_client.stream_video_ws(
                video_id,
                {
                    "selection": sel,
                    "design_recipe": {"pattern": "waves", "colors": ["#ff595e", "#1982c4"]},
                    "placement": {"wrap_mode": "planar", "blend_mode": "luminance"},
                },
                frame_timeout=12.0,
                total_timeout=60.0,
            )

    def _http_range_worker() -> list[int]:
        statuses: list[int] = []
        with StudioAPIClient(
            host=live_studio_server.host,
            port=live_studio_server.port,
            server_proc=live_studio_server.server_proc,
        ) as http_client:
            for i in range(10):
                start = i * 256
                end = start + 1023
                r = http_client.get_range(
                    f"/api/videos/{video_id}/media",
                    start=start,
                    end=end,
                    expected_status=206,
                )
                assert len(r["body"]) == 1024
                statuses.append(int(r["status"]))
                h = http_client.get_json("/api/health", expected_status=200)
                assert h.get("status") == "ok"
        return statuses

    with ThreadPoolExecutor(max_workers=2) as pool:
        ws_fut = pool.submit(_ws_worker)
        http_fut = pool.submit(_http_range_worker)
        ws_frames, ws_report = ws_fut.result(timeout=65.0)
        range_statuses = http_fut.result(timeout=65.0)

    assert range_statuses == [206] * 10
    assert len(ws_frames) == ws_report["input_frame_count"] >= 60
    assert_video_run_report_invariants(ws_report, expect_output_video=False)


@pytest.mark.f03
@pytest.mark.f05
@pytest.mark.f10
@pytest.mark.f15
def test_t3_17_browser_studio_prompt_cylinder_wrap_playback_and_mp4_export(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify Browser Studio cylinder_mug_640x480 + prompt + cylinder luminance >=15 FPS playback + MP4 export."""
    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)

    browser.select_testid("demo-clip-select", "cylinder_mug_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    browser.click_testid("auto-select-btn")

    browser.fill_testid("design-prompt-input", 'pink and navy dots text "MUG"')
    browser.click_testid("apply-prompt-btn")
    browser.select_testid("wrap-mode-select", "cylinder")
    browser.select_testid("blend-mode-select", "luminance")

    _ensure_playing(browser)
    time.sleep(2.2)
    browser.pump_events(0.05)
    tel = browser.get_telemetry(timeout=5.0)
    assert tel is not None
    assert tel.get("wrapMode") == "cylinder"
    assert tel.get("blendMode") == "luminance"
    assert float(tel.get("currentFps", 0.0)) >= 15.0, f"Cylinder playback FPS < 15.0: {tel}"
    assert float(tel.get("medianFrameMs", 999.0)) <= 66.67

    mp4_path, export_report = _export_and_fetch_mp4(
        browser, live_studio_server, e2e_tmp_dir, timeout=60.0
    )
    assert_video_run_report_invariants(export_report, expect_output_video=True)
    assert export_report["wrap_mode"] == "cylinder"
    assert export_report["blend_mode"] == "luminance"
    probed = probe_video(mp4_path, tmp_dir=e2e_tmp_dir)
    assert probed["codec_name"] == "h264"
    assert probed["frame_count"] == export_report["output_frame_count"]
    browser.assert_zero_errors()


@pytest.mark.f04
@pytest.mark.f06
@pytest.mark.f10
@pytest.mark.f13
@pytest.mark.f15
def test_t3_18_browser_studio_custom_video_and_rgba_png_upload_with_box_select(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify custom MP4 file upload + custom RGBA PNG upload + select-mode-box drag + >=15 FPS playback."""
    local_mp4 = e2e_tmp_dir / "t3_18_custom.mp4"
    clip_meta = make_textured_tracking_clip(
        local_mp4, width=640, height=480, frames=60, fps=30, tmp_dir=e2e_tmp_dir
    )
    local_png = make_rgba_png(e2e_tmp_dir / "t3_18_logo.png", width=256, height=256, mode="emblem")

    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)

    browser.set_file_input("video-file-input", local_mp4)
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    browser.set_file_input("design-image-input", local_png)
    browser.pump_events(0.3)

    bx, by, bw, bh = clip_meta["bbox_frame0"]
    browser.click_testid("select-mode-box")
    browser.drag_canvas_box(float(bx), float(by), float(bw), float(bh))
    browser.set_slider_testid("scale-slider", 1.2)
    browser.set_slider_testid("rotation-slider", 20)

    _ensure_playing(browser)
    time.sleep(2.2)
    browser.pump_events(0.05)
    tel = browser.get_telemetry(timeout=5.0)
    assert tel is not None
    assert tel.get("playing") is True
    assert tel.get("trackingState") == "tracking"
    assert float(tel.get("currentFps", 0.0)) >= 15.0, f"Custom upload playback FPS < 15.0: {tel}"
    assert float(tel.get("medianFrameMs", 999.0)) <= 66.67
    browser.assert_zero_errors()


@pytest.mark.f03
@pytest.mark.f10
@pytest.mark.f15
@pytest.mark.f16
def test_t3_19_browser_studio_design_persists_across_video_and_live_camera_switch(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify procedural waves + ink blend + opacity persist across mode-video -> mode-camera -> mode-video export."""
    y4m_path = e2e_tmp_dir / "t3_19_cam.y4m"
    make_y4m_camera_clip(y4m_path, width=640, height=480, frames=60, fps=30, tmp_dir=e2e_tmp_dir)

    browser = chromium_session_factory(fake_video_y4m=y4m_path)
    browser.navigate(live_studio_server.base_url)

    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    browser.click_testid("auto-select-btn")

    browser.select_testid("pattern-select", "waves")
    browser.select_testid("wrap-mode-select", "planar")
    browser.select_testid("blend-mode-select", "ink")
    browser.set_slider_testid("opacity-slider", 0.75)

    browser.click_testid("mode-camera")
    tel_cam = browser.wait_for_telemetry(
        lambda t: t.get("mode") == "camera" and bool(t.get("playing")) and int(t.get("processedFrames", 0)) >= 5,
        timeout=12.0,
    )
    assert tel_cam["wrapMode"] == "planar"
    assert tel_cam["blendMode"] == "ink"
    assert browser.get_testid_text("hud-fps") != ""
    assert browser.get_testid_text("hud-tracking-state") != ""

    browser.click_testid("mode-video")
    browser.wait_for_telemetry(lambda t: t.get("mode") == "video", timeout=10.0)
    browser.select_testid("demo-clip-select", "planar_card_640x480")
    browser.click_testid("load-demo-btn")
    _wait_for_stage_ready(browser, min_non_black=0.15, timeout=15.0)
    browser.click_testid("auto-select-btn")

    mp4_path, export_report = _export_and_fetch_mp4(
        browser, live_studio_server, e2e_tmp_dir, timeout=60.0
    )
    assert_video_run_report_invariants(export_report, expect_output_video=True)
    assert export_report["blend_mode"] == "ink"
    assert probe_video(mp4_path, tmp_dir=e2e_tmp_dir)["codec_name"] == "h264"
    browser.assert_zero_errors()


@pytest.mark.f04
@pytest.mark.f09
@pytest.mark.f10
@pytest.mark.f16
def test_t3_20_browser_live_camera_rgba_upload_occlusion_suppression_and_recovery(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
    e2e_tmp_dir: Path,
) -> None:
    """Verify custom RGBA upload in mode-camera suppresses during occlusion and recovers at >= 15.0 FPS."""
    y4m_path = e2e_tmp_dir / "t3_20_cam_occ.y4m"
    y4m_meta = make_y4m_camera_clip(
        y4m_path,
        width=640,
        height=480,
        frames=75,
        fps=30,
        occlude_ranges=[(35, 52)],
        tmp_dir=e2e_tmp_dir,
    )
    rgba_path = make_rgba_png(e2e_tmp_dir / "t3_20_ring.png", width=256, height=256, mode="ring")

    browser = chromium_session_factory(fake_video_y4m=y4m_path)
    browser.navigate(live_studio_server.base_url)
    browser.set_file_input("design-image-input", rgba_path)
    browser.pump_events(0.25)

    browser.click_testid("mode-camera")
    _wait_for_stage_ready(browser, min_non_black=0.10, timeout=12.0)
    _select_y4m_target_in_camera(browser, y4m_meta, timeout=12.0)

    saw_loss = False
    saw_recovery = False
    max_fps_seen = 0.0
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        tel = browser.get_telemetry(timeout=4.0)
        if tel is not None:
            max_fps_seen = max(max_fps_seen, float(tel.get("currentFps", 0.0)))
            st = str(tel.get("trackingState", ""))
            if not saw_loss and st in {"lost", "recovering"}:
                saw_loss = True
            elif saw_loss and st == "tracking" and int(tel.get("inliers", 0)) > 0:
                saw_recovery = True
                break
        time.sleep(0.06)

    assert saw_loss, "Expected trackingState to enter lost/recovering during camera occlusion"
    assert saw_recovery, "Expected trackingState to recover to tracking after camera occlusion"
    assert max_fps_seen >= 15.0, f"Expected live camera FPS >= 15.0, observed max {max_fps_seen:.2f}"
    hud_state = browser.get_testid_text("hud-tracking-state").lower()
    assert "tracking" in hud_state
    browser.assert_zero_errors()
