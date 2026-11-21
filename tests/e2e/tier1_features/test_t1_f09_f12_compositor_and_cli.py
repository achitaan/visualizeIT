"""Tier 1 Feature Happy-Path E2E Tests for Features F9–F12 (24 distinct tests: 6 per feature).

Strictly opaque-box:
- Zero imports of `visualizeit.*` anywhere.
- Zero top-level third-party imports (besides `pytest`); `numpy` / `PIL` loaded lazily inside helpers/tests.
- Exercises documented `PROJECT.md §3–§5` CLI (`info`, `segment`, `process`, `benchmark`) and WebSocket
  (`/ws/videos/{id}/stream`) contracts with finite timeouts and XFS `.cache/tmp` isolation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import pytest

from tests.e2e._helpers import (
    CACHE_DIR,
    assert_no_python_traceback,
    assert_video_run_report_invariants,
    bbox_at_frame,
    compute_psnr,
    compute_roi_and_bg_delta,
    decode_video_frames,
    make_occlusion_clip,
    make_textured_tracking_clip,
    probe_video,
    run_cli,
    run_cli_json,
    run_subproc,
    spawn_live_server,
    transcode_clip,
)

pytestmark = pytest.mark.tier1


def _selection_to_process_args(selection: dict[str, Any] | None) -> list[str]:
    """Convert a `SelectionSpec` JSON dict into `visualizeit process` CLI flags."""
    if not selection:
        return []
    args: list[str] = []
    if "frame_index" in selection and selection["frame_index"] is not None:
        args.extend(["--select-frame", str(int(selection["frame_index"]))])
    for pt in selection.get("positive_points") or []:
        args.extend(["--point", f"{float(pt[0]):g},{float(pt[1]):g}"])
    for pt in selection.get("negative_points") or []:
        args.extend(["--neg-point", f"{float(pt[0]):g},{float(pt[1]):g}"])
    bbox = selection.get("bbox")
    if bbox is not None and len(bbox) == 4:
        args.extend(["--box", ",".join(f"{float(v):g}" for v in bbox)])
    return args


def _bbox_tuple_from_selection(
    selection: dict[str, Any] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Extract an integer `(x, y, w, h)` bounding box from a selection dict."""
    if selection and selection.get("bbox") and len(selection["bbox"]) == 4:
        bx, by, bw, bh = selection["bbox"]
        return (int(round(bx)), int(round(by)), int(round(bw)), int(round(bh)))
    return (width // 4, height // 4, width // 2, height // 2)


def _find_occlusion_phase_indices(
    raw_mp4: Path,
    bbox: tuple[int, int, int, int],
) -> tuple[list[int], list[int], list[int]]:
    """Opaque-box physical phase detector on raw video: returns `(pre_visible, occluded, post_visible)` frame indices."""
    import numpy as np

    meta = probe_video(raw_mp4)
    total = int(meta["frame_count"])
    sample_step = max(1, total // 36)
    candidate_indices = sorted(set(range(0, total, sample_step)) | {0, total // 2, max(0, total - 2)})
    frames = decode_video_frames(raw_mp4, candidate_indices)

    x, y, bw, bh = bbox
    stds: dict[int, float] = {}
    for idx, rgb in frames.items():
        h, w = rgb.shape[:2]
        x1 = max(0, min(w - 2, x + bw // 6))
        y1 = max(0, min(h - 2, y + bh // 6))
        x2 = max(x1 + 2, min(w, x + bw - bw // 6))
        y2 = max(y1 + 2, min(h, y + bh - bh // 6))
        stds[idx] = float(np.std(rgb[y1:y2, x1:x2].astype(np.float32)))

    occluded = [idx for idx in sorted(stds) if stds[idx] < 10.0]
    assert occluded, f"Expected featureless occluded frames in {raw_mp4}, got spatial stds={stds}"
    first_occ = min(occluded)
    last_occ = max(occluded)
    pre_visible = [idx for idx in sorted(stds) if idx < first_occ - sample_step and stds[idx] >= 18.0]
    post_visible = [idx for idx in sorted(stds) if idx > last_occ + sample_step and stds[idx] >= 18.0]
    assert pre_visible and post_visible, (
        f"Expected visible phases before and after occlusion in {raw_mp4}; "
        f"pre={pre_visible}, occ={occluded}, post={post_visible}"
    )
    return pre_visible, occluded, post_visible


def _make_compact_disk_png(
    out_path: Path,
    *,
    size: int = 256,
    radius_px: float = 34.0,
    rgb: tuple[int, int, int] = (0, 255, 136),
) -> Path:
    """Create a centered opaque disk on a fully transparent RGBA canvas for scale/offset tests."""
    import numpy as np
    from PIL import Image

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    ys, xs = np.ogrid[:size, :size]
    cx = (size - 1) * 0.5
    cy = (size - 1) * 0.5
    mask = ((xs - cx) ** 2 + (ys - cy) ** 2) <= (radius_px ** 2)
    rgba[mask, 0] = rgb[0]
    rgba[mask, 1] = rgb[1]
    rgba[mask, 2] = rgb[2]
    rgba[mask, 3] = 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_path, format="PNG")
    return out_path


def _extract_benchmark_reports(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract one or more `VideoRunReport` dicts from `benchmark --json` output."""
    if "render_stride" in payload and "effective_fps" in payload:
        return [payload]
    found: list[dict[str, Any]] = []
    for key in ("presets", "results", "runs", "scenarios", "reports"):
        val = payload.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    if "render_stride" in item and "effective_fps" in item:
                        found.append(item)
                    elif isinstance(item.get("report"), dict):
                        found.append(item["report"])
        elif isinstance(val, dict):
            for item in val.values():
                if isinstance(item, dict):
                    if "render_stride" in item and "effective_fps" in item:
                        found.append(item)
                    elif isinstance(item.get("report"), dict):
                        found.append(item["report"])
    if not found:
        for val in payload.values():
            if isinstance(val, dict) and "render_stride" in val and "effective_fps" in val:
                found.append(val)
    return found


# ==============================================================================
# F9 — Occlusion Detection, Suppression & Auto Re-Acquisition (6 tests)
# ==============================================================================


@pytest.mark.f09
def test_f09_occlusion_clip_reports_lost_and_recovered_counts(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """F9-1: `process` on `occlusion_recovery_640x480.mp4` reports genuine lost and recovered frame counts."""
    occ_mp4 = demo_corpus_dir / "occlusion_recovery_640x480.mp4"
    meta = demo_corpus_manifest["occlusion_recovery_640x480"]
    sel_args = _selection_to_process_args(meta.get("default_selection"))
    out_mp4 = e2e_tmp_dir / "f09_occ_counts.mp4"

    _, report = run_cli_json(
        [
            "process",
            "--input",
            str(occ_mp4),
            "--output",
            str(out_mp4),
            "--pattern",
            "checker",
            "--colors",
            "#ff006e,#ffbe0b",
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "normal",
            "--opacity",
            "0.92",
            *sel_args,
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, expect_output_video=True)
    assert report["lost_frame_count"] >= 3, f"Expected >=3 lost frames during occlusion, got {report}"
    assert report["recovered_count"] >= 1, f"Expected >=1 auto re-acquisition recovery, got {report}"
    assert report["tracked_frame_count"] > 10, f"Expected >10 tracked visible frames, got {report}"
    assert 0.15 <= float(report["tracking_availability"]) < 0.98, (
        f"Expected bounded tracking_availability reflecting occluded window, got {report['tracking_availability']}"
    )


@pytest.mark.f09
def test_f09_occluded_frames_strictly_suppress_stale_artwork_overlay(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """F9-2: Fully occluded middle frames suppress stale overlays with zero ghost artwork bleed."""
    import numpy as np

    occ_mp4 = demo_corpus_dir / "occlusion_recovery_640x480.mp4"
    meta = demo_corpus_manifest["occlusion_recovery_640x480"]
    selection = meta.get("default_selection")
    sel_args = _selection_to_process_args(selection)
    bbox0 = _bbox_tuple_from_selection(selection, 640, 480)
    out_mp4 = e2e_tmp_dir / "f09_occ_suppressed.mp4"

    _, report = run_cli_json(
        [
            "process",
            "--input",
            str(occ_mp4),
            "--output",
            str(out_mp4),
            "--pattern",
            "solid",
            "--colors",
            "#ff0055,#00ffcc",
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "normal",
            "--opacity",
            "0.95",
            *sel_args,
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, expect_output_video=True)

    pre_vis, occ_idxs, _ = _find_occlusion_phase_indices(occ_mp4, bbox0)
    # Sample interior occluded frames away from transition boundaries
    mid_occ = occ_idxs[len(occ_idxs) // 4 : max(1, (3 * len(occ_idxs)) // 4)] or occ_idxs
    sampled_occ = [mid_occ[0], mid_occ[len(mid_occ) // 2], mid_occ[-1]]
    wanted = sorted({pre_vis[0], *sampled_occ})

    raw_frames = decode_video_frames(occ_mp4, wanted)
    out_frames = decode_video_frames(out_mp4, wanted)

    pre_roi_delta, _ = compute_roi_and_bg_delta(raw_frames[pre_vis[0]], out_frames[pre_vis[0]], bbox0)
    assert pre_roi_delta >= 12.0, f"Expected strong pre-occlusion overlay delta >= 12.0, got {pre_roi_delta:.2f}"

    for f_idx in sampled_occ:
        full_diff = float(np.mean(np.abs(out_frames[f_idx].astype(np.float32) - raw_frames[f_idx].astype(np.float32))))
        assert full_diff <= 2.5, (
            f"Stale/frozen ghost overlay detected on occluded frame {f_idx}: "
            f"mean_abs_diff={full_diff:.2f} > 2.5"
        )


@pytest.mark.f09
def test_f09_post_occlusion_reentry_reacquires_and_restores_overlay(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """F9-3: Object re-entry in the final third automatically re-acquires lock and restores artwork overlay."""
    occ_mp4 = demo_corpus_dir / "occlusion_recovery_640x480.mp4"
    meta = demo_corpus_manifest["occlusion_recovery_640x480"]
    selection = meta.get("default_selection")
    sel_args = _selection_to_process_args(selection)
    bbox0 = _bbox_tuple_from_selection(selection, 640, 480)
    out_mp4 = e2e_tmp_dir / "f09_occ_reacquire.mp4"

    _, report = run_cli_json(
        [
            "process",
            "--input",
            str(occ_mp4),
            "--output",
            str(out_mp4),
            "--pattern",
            "stripes",
            "--colors",
            "#ff206e,#00f5d4",
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "normal",
            "--opacity",
            "0.95",
            *sel_args,
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, expect_output_video=True)
    assert report["recovered_count"] >= 1

    _, _, post_vis = _find_occlusion_phase_indices(occ_mp4, bbox0)
    # Sample multiple post-occlusion frames once re-acquisition settles
    tail_candidates = post_vis[max(0, len(post_vis) // 3) :] or post_vis
    sampled_post = sorted({tail_candidates[0], tail_candidates[len(tail_candidates) // 2], tail_candidates[-1]})

    raw_frames = decode_video_frames(occ_mp4, sampled_post)
    out_frames = decode_video_frames(out_mp4, sampled_post)

    # Expand margin slightly to accommodate smooth card motion on late frames
    post_roi_deltas: list[float] = []
    for f_idx in sampled_post:
        roi_delta, bg_delta = compute_roi_and_bg_delta(
            raw_frames[f_idx],
            out_frames[f_idx],
            bbox0,
            margin=56,
        )
        post_roi_deltas.append(roi_delta)
        assert bg_delta <= 3.5, f"Background corrupted on post-occlusion frame {f_idx}: bg_delta={bg_delta:.2f}"

    assert max(post_roi_deltas) >= 8.5, (
        f"Expected restored artwork delta > 8.5 after re-entry, got {post_roi_deltas}"
    )


@pytest.mark.f09
def test_f09_websocket_stream_emits_lost_or_recovering_then_tracking_states(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """F9-4: Streaming `occlusion_recovery_640x480` over WebSocket transitions tracking -> lost/recovering -> tracking."""
    occ_mp4 = demo_corpus_dir / "occlusion_recovery_640x480.mp4"
    meta = demo_corpus_manifest["occlusion_recovery_640x480"]
    selection = meta.get("default_selection")

    with spawn_live_server(storage_dir=e2e_tmp_dir / "srv_f09_ws") as client:
        up_resp = client.upload(
            "/api/videos/upload",
            filename="occlusion_recovery_640x480.mp4",
            content=occ_mp4.read_bytes(),
            content_type="video/mp4",
            expected_status=200,
            timeout=30.0,
        )
        video_id = up_resp["json"]["video_id"]

        ws_frames, complete_report = client.stream_video_ws(
            video_id,
            {
                "selection": selection,
                "design_recipe": {
                    "pattern": "checker",
                    "colors": ["#ff595e", "#1982c4"],
                },
                "placement": {
                    "wrap_mode": "planar",
                    "blend_mode": "normal",
                    "opacity": 0.90,
                },
            },
            frame_timeout=15.0,
            total_timeout=90.0,
        )

    assert len(ws_frames) >= 60, f"Expected full occlusion stream, got {len(ws_frames)} frames"
    states = [str(f.get("state")) for f in ws_frames]
    inliers = [int(f.get("inliers", 0)) for f in ws_frames]
    valid_states = {"tracking", "recovering", "lost"}
    assert set(states).issubset(valid_states), f"Unexpected tracking states: {set(states) - valid_states}"

    first_track_idx = next((i for i, s in enumerate(states) if s == "tracking"), None)
    assert first_track_idx is not None, f"Never entered 'tracking' state before occlusion: {states[:15]}"

    non_track_after = [i for i in range(first_track_idx + 1, len(states)) if states[i] in {"lost", "recovering"}]
    assert non_track_after, f"Never transitioned to 'lost'/'recovering' during occlusion: {states}"

    first_loss_idx = non_track_after[0]
    reacquired_indices = [i for i in range(first_loss_idx + 1, len(states)) if states[i] == "tracking"]
    assert reacquired_indices, (
        f"Never recovered to 'tracking' after loss at frame {first_loss_idx}: {states[first_loss_idx:]}"
    )

    pre_inliers = [inliers[i] for i in range(first_track_idx, first_loss_idx) if states[i] == "tracking"]
    occ_inliers = [inliers[i] for i in non_track_after]
    post_inliers = [inliers[i] for i in reacquired_indices]
    assert min(occ_inliers) < max(pre_inliers), (
        f"Expected inliers to drop during occlusion: pre={pre_inliers[:5]}, occ={occ_inliers[:5]}"
    )
    assert max(post_inliers) > 0, f"Expected positive inliers after recovery, got {post_inliers}"
    assert complete_report["lost_frame_count"] >= 1
    assert complete_report["recovered_count"] >= 1


@pytest.mark.f09
def test_f09_multiple_occlusion_cycles_recover_repeatedly(e2e_tmp_dir: Path) -> None:
    """F9-5: Two full occlusion wipes in one clip trigger `recovered_count >= 2` and restore artwork twice."""
    import numpy as np

    clip_path = e2e_tmp_dir / "two_wipes.mp4"
    clip_meta = make_occlusion_clip(
        clip_path,
        width=640,
        height=480,
        frames=54,
        fps=30,
        occlude_ranges=[(12, 19), (32, 39)],
        tmp_dir=e2e_tmp_dir,
    )
    bx, by, bw, bh = clip_meta["bbox_frame0"]
    out_mp4 = e2e_tmp_dir / "two_wipes_out.mp4"

    _, report = run_cli_json(
        [
            "process",
            "--input",
            str(clip_path),
            "--output",
            str(out_mp4),
            "--box",
            f"{bx},{by},{bw},{bh}",
            "--pattern",
            "stripes",
            "--colors",
            "#ff006e,#3a86ff",
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "normal",
            "--opacity",
            "0.95",
        ],
        timeout=60.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, expect_output_video=True)
    assert report["lost_frame_count"] >= 10, f"Expected >=10 lost frames across two 8-frame wipes, got {report}"
    assert report["recovered_count"] >= 2, f"Expected >=2 recoveries across two occlusion wipes, got {report}"

    sample_idxs = [5, 15, 26, 35, 48]
    raw_f = decode_video_frames(clip_path, sample_idxs)
    out_f = decode_video_frames(out_mp4, sample_idxs)

    for vis_idx in (5, 26, 48):
        roi_d, bg_d = compute_roi_and_bg_delta(raw_f[vis_idx], out_f[vis_idx], bbox_at_frame(clip_meta, vis_idx))
        assert roi_d >= 10.0, f"Expected visible overlay delta >= 10.0 at frame {vis_idx}, got {roi_d:.2f}"
        assert bg_d <= 3.5, f"Expected background preserved at frame {vis_idx}, got {bg_d:.2f}"

    for occ_idx in (15, 35):
        occ_diff = float(np.mean(np.abs(out_f[occ_idx].astype(np.float32) - raw_f[occ_idx].astype(np.float32))))
        assert occ_diff <= 2.5, f"Expected suppressed overlay on wipe frame {occ_idx}, got diff={occ_diff:.2f}"


@pytest.mark.f09
def test_f09_benchmark_occlusion_preset_passes_fps_and_recovery(e2e_tmp_dir: Path) -> None:
    """F9-6: `benchmark --preset occlusion --min-fps 15.0` exits 0 with >=15 FPS and verified recovery."""
    out_dir = e2e_tmp_dir / "bench_occ"
    report_path = e2e_tmp_dir / "bench_occ_report.json"

    _, report = run_cli_json(
        [
            "benchmark",
            "--preset",
            "occlusion",
            "--min-fps",
            "15.0",
            "--output-dir",
            str(out_dir),
            "--report",
            str(report_path),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, require_min_fps=15.0, expect_output_video=True)
    assert report["lost_frame_count"] >= 1, f"Expected occlusion preset to record lost frames: {report}"
    assert report["recovered_count"] >= 1, f"Expected occlusion preset to record recovery: {report}"
    assert report_path.is_file() and report_path.stat().st_size > 0


# ==============================================================================
# F10 — Multi-Mode Surface Wrapping & Luminance Blending (6 tests)
# ==============================================================================


@pytest.mark.f10
def test_f10_four_wrap_modes_produce_valid_distinct_surface_projections(e2e_tmp_dir: Path) -> None:
    """F10-1: `planar`, `cylinder`, `mesh_grid`, and `mesh_6dof` wrap modes all render valid surface projections."""
    import numpy as np

    clip_path = e2e_tmp_dir / "wrap_modes_src.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path,
        width=640,
        height=480,
        frames=16,
        fps=30,
        tmp_dir=e2e_tmp_dir,
    )
    bx, by, bw, bh = clip_meta["bbox_frame0"]
    raw_f4 = decode_video_frames(clip_path, [4])[4]
    bbox_f4 = bbox_at_frame(clip_meta, 4)

    mode_frames: dict[str, Any] = {}
    for wrap_mode in ("planar", "cylinder", "mesh_grid", "mesh_6dof"):
        out_mp4 = e2e_tmp_dir / f"wrap_{wrap_mode}.mp4"
        extra_args = ["--mesh", "plane", "--intrinsics", "560,560,320,240"] if wrap_mode == "mesh_6dof" else []
        _, report = run_cli_json(
            [
                "process",
                "--input",
                str(clip_path),
                "--output",
                str(out_mp4),
                "--box",
                f"{bx},{by},{bw},{bh}",
                "--pattern",
                "checker",
                "--colors",
                "#ff595e,#1982c4",
                "--wrap-mode",
                wrap_mode,
                "--blend-mode",
                "normal",
                "--opacity",
                "0.95",
                *extra_args,
            ],
            timeout=60.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_video_run_report_invariants(report, expect_output_video=True)
        assert report["wrap_mode"] == wrap_mode

        out_f4 = decode_video_frames(out_mp4, [4])[4]
        roi_d, bg_d = compute_roi_and_bg_delta(raw_f4, out_f4, bbox_f4)
        assert roi_d >= 10.0, f"Expected strong foreground delta for wrap_mode={wrap_mode}, got {roi_d:.2f}"
        assert bg_d <= 3.5, f"Background modified for wrap_mode={wrap_mode}: bg_delta={bg_d:.2f}"
        mode_frames[wrap_mode] = out_f4

    x4, y4, w4, h4 = bbox_f4
    planar_roi = mode_frames["planar"][y4 : y4 + h4, x4 : x4 + w4].astype(np.float32)
    cyl_roi = mode_frames["cylinder"][y4 : y4 + h4, x4 : x4 + w4].astype(np.float32)
    proj_diff = float(np.mean(np.abs(cyl_roi - planar_roi)))
    assert proj_diff >= 2.0, (
        f"Expected cylinder UV foreshortening to differ from planar projection inside ROI, got diff={proj_diff:.2f}"
    )


@pytest.mark.f10
def test_f10_three_blend_modes_luminance_normal_and_ink_behave_distinctly(e2e_tmp_dir: Path) -> None:
    """F10-2: `luminance` preserves shaded substrate ratio (>1.25x), `normal` flattens it, and `ink` darkens."""
    import numpy as np

    clip_path = e2e_tmp_dir / "shaded_card.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path,
        width=640,
        height=480,
        frames=14,
        fps=30,
        shaded_card=True,
        tmp_dir=e2e_tmp_dir,
    )
    bx, by, bw, bh = clip_meta["bbox_frame0"]
    f_idx = 3
    raw_f = decode_video_frames(clip_path, [f_idx])[f_idx]
    rx, ry, rw, rh = bbox_at_frame(clip_meta, f_idx)

    def _luma(rgb_patch: Any) -> float:
        arr = np.asarray(rgb_patch, dtype=np.float32)
        return float(np.mean(0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]))

    pad_x = max(14, rw // 10)
    pad_y = max(14, rh // 8)
    mid_x = rx + rw // 2
    left_slice = (slice(ry + pad_y, ry + rh - pad_y), slice(rx + pad_x, mid_x - 10))
    right_slice = (slice(ry + pad_y, ry + rh - pad_y), slice(mid_x + 10, rx + rw - pad_x))

    raw_left_l = _luma(raw_f[left_slice])
    raw_right_l = _luma(raw_f[right_slice])
    assert raw_left_l / max(1.0, raw_right_l) >= 1.6, (
        f"Shaded card fixture sanity failed: L_left={raw_left_l:.1f}, L_right={raw_right_l:.1f}"
    )

    outputs: dict[str, Any] = {}
    for blend_mode in ("luminance", "normal", "ink"):
        out_mp4 = e2e_tmp_dir / f"blend_{blend_mode}.mp4"
        _, report = run_cli_json(
            [
                "process",
                "--input",
                str(clip_path),
                "--output",
                str(out_mp4),
                "--box",
                f"{bx},{by},{bw},{bh}",
                "--pattern",
                "solid",
                "--colors",
                "#40a0ff,#40a0ff",
                "--wrap-mode",
                "planar",
                "--blend-mode",
                blend_mode,
                "--opacity",
                "0.95",
            ],
            timeout=60.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_video_run_report_invariants(report, expect_output_video=True)
        assert report["blend_mode"] == blend_mode
        outputs[blend_mode] = decode_video_frames(out_mp4, [f_idx])[f_idx]

    lum_left = _luma(outputs["luminance"][left_slice])
    lum_right = _luma(outputs["luminance"][right_slice])
    lum_ratio = lum_left / max(1.0, lum_right)
    assert lum_ratio > 1.25, f"Luminance blend failed to preserve left/right shading ratio: {lum_ratio:.3f}"

    norm_left = _luma(outputs["normal"][left_slice])
    norm_right = _luma(outputs["normal"][right_slice])
    norm_ratio = norm_left / max(1.0, norm_right)
    assert abs(norm_ratio - 1.0) < 0.18 and norm_ratio < lum_ratio - 0.15, (
        f"Normal blend @ 0.95 opacity should flatten shading relative to luminance: "
        f"norm_ratio={norm_ratio:.3f}, lum_ratio={lum_ratio:.3f}"
    )

    ink_left = _luma(outputs["ink"][left_slice])
    ink_right = _luma(outputs["ink"][right_slice])
    assert ink_left <= raw_left_l + 4.0 and ink_right <= raw_right_l + 4.0, (
        f"Ink multiply blend must not brighten above substrate: "
        f"ink=({ink_left:.1f}, {ink_right:.1f}) vs raw=({raw_left_l:.1f}, {raw_right_l:.1f})"
    )
    # Ink with #40a0ff tints neutral gray toward blue (B > R) and differs from both normal and luminance
    ink_patch = outputs["ink"][left_slice].astype(np.float32)
    assert float(np.mean(ink_patch[..., 2] - ink_patch[..., 0])) > 15.0, "Ink blend did not tint substrate toward #40a0ff"
    assert float(np.mean(np.abs(outputs["ink"][left_slice].astype(np.float32) - outputs["normal"][left_slice].astype(np.float32)))) > 8.0


@pytest.mark.f10
def test_f10_scale_transform_changes_mapped_artwork_footprint_monotonically(e2e_tmp_dir: Path) -> None:
    """F10-3: `--scale 0.5`, `1.0`, and `1.8` change mapped compact decal footprint strictly monotonically."""
    import numpy as np

    clip_path = e2e_tmp_dir / "scale_src.mp4"
    clip_meta = make_textured_tracking_clip(clip_path, frames=10, fps=30, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = clip_meta["bbox_frame0"]
    decal_png = _make_compact_disk_png(e2e_tmp_dir / "disk_decal.png", size=256, radius_px=34.0)

    f_idx = 2
    raw_f = decode_video_frames(clip_path, [f_idx])[f_idx]
    rx, ry, rw, rh = bbox_at_frame(clip_meta, f_idx)
    raw_roi = raw_f[ry : ry + rh, rx : rx + rw].astype(np.float32)

    changed_counts: list[int] = []
    for scale_val in (0.5, 1.0, 1.8):
        out_mp4 = e2e_tmp_dir / f"scale_{scale_val}.mp4"
        _, report = run_cli_json(
            [
                "process",
                "--input",
                str(clip_path),
                "--output",
                str(out_mp4),
                "--box",
                f"{bx},{by},{bw},{bh}",
                "--design-image",
                str(decal_png),
                "--wrap-mode",
                "planar",
                "--blend-mode",
                "normal",
                "--opacity",
                "1.0",
                "--scale",
                str(scale_val),
            ],
            timeout=45.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_video_run_report_invariants(report, expect_output_video=True)
        out_f = decode_video_frames(out_mp4, [f_idx])[f_idx]
        out_roi = out_f[ry : ry + rh, rx : rx + rw].astype(np.float32)
        per_px_diff = np.mean(np.abs(out_roi - raw_roi), axis=-1)
        changed_counts.append(int(np.count_nonzero(per_px_diff > 20.0)))

    c0, c1, c2 = changed_counts
    assert min(changed_counts) > 25, f"Decal produced too few changed pixels: {changed_counts}"
    strictly_increasing = c0 < c1 < c2
    strictly_decreasing = c0 > c1 > c2
    assert strictly_increasing or strictly_decreasing, (
        f"Expected strictly monotonic decal footprint across scales (0.5, 1.0, 1.8), got {changed_counts}"
    )
    assert max(c0, c2) / max(1.0, float(min(c0, c2))) >= 1.8, (
        f"Expected substantial footprint change across 3.6x scale span, got {changed_counts}"
    )


@pytest.mark.f10
def test_f10_rotation_deg_rotates_directional_stripes_orientation(e2e_tmp_dir: Path) -> None:
    """F10-4: `--rotation-deg 0` vs `--rotation-deg 90` orthogonally flips directional stripe gradient axes."""
    import numpy as np

    clip_path = e2e_tmp_dir / "rot_src.mp4"
    clip_meta = make_textured_tracking_clip(clip_path, frames=10, fps=30, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = clip_meta["bbox_frame0"]
    f_idx = 2
    rx, ry, rw, rh = bbox_at_frame(clip_meta, f_idx)

    grads: dict[int, tuple[float, float]] = {}
    for deg in (0, 90):
        out_mp4 = e2e_tmp_dir / f"rot_{deg}.mp4"
        _, report = run_cli_json(
            [
                "process",
                "--input",
                str(clip_path),
                "--output",
                str(out_mp4),
                "--box",
                f"{bx},{by},{bw},{bh}",
                "--pattern",
                "stripes",
                "--colors",
                "#ffffff,#000000",
                "--wrap-mode",
                "planar",
                "--blend-mode",
                "normal",
                "--opacity",
                "1.0",
                "--rotation-deg",
                str(deg),
            ],
            timeout=45.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_video_run_report_invariants(report, expect_output_video=True)
        out_f = decode_video_frames(out_mp4, [f_idx])[f_idx]
        # Sample interior of card away from border edges
        interior = out_f[ry + 24 : ry + rh - 24, rx + 24 : rx + rw - 24].astype(np.float32)
        gray = np.mean(interior, axis=-1)
        gx = float(np.mean(np.abs(gray[:, 1:] - gray[:, :-1])))
        gy = float(np.mean(np.abs(gray[1:, :] - gray[:-1, :])))
        grads[deg] = (gx, gy)

    gx0, gy0 = grads[0]
    gx90, gy90 = grads[90]
    flipped = (gx0 > 1.25 * gy0 and gy90 > 1.25 * gx90) or (gy0 > 1.25 * gx0 and gx90 > 1.25 * gy90)
    assert flipped, (
        f"Expected 90-degree stripe rotation to swap dominant spatial gradient axis: "
        f"deg=0 (gx={gx0:.2f}, gy={gy0:.2f}), deg=90 (gx={gx90:.2f}, gy={gy90:.2f})"
    )


@pytest.mark.f10
def test_f10_offset_x_and_offset_y_shift_artwork_centroid_on_surface(e2e_tmp_dir: Path) -> None:
    """F10-5: `--offset-x` and `--offset-y` shift a non-repeating decal's centroid across the tracked surface."""
    import numpy as np

    clip_path = e2e_tmp_dir / "offset_src.mp4"
    clip_meta = make_textured_tracking_clip(clip_path, frames=10, fps=30, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = clip_meta["bbox_frame0"]
    decal_png = _make_compact_disk_png(e2e_tmp_dir / "offset_disk.png", size=256, radius_px=28.0)

    f_idx = 2
    raw_f = decode_video_frames(clip_path, [f_idx])[f_idx]
    rx, ry, rw, rh = bbox_at_frame(clip_meta, f_idx)
    raw_roi = raw_f[ry : ry + rh, rx : rx + rw].astype(np.float32)

    configs = [
        ("neg_x", -0.20, 0.0),
        ("center", 0.0, 0.0),
        ("pos_xy", 0.20, 0.18),
    ]
    centroids: dict[str, tuple[float, float]] = {}
    for label, ox, oy in configs:
        out_mp4 = e2e_tmp_dir / f"offset_{label}.mp4"
        _, report = run_cli_json(
            [
                "process",
                "--input",
                str(clip_path),
                "--output",
                str(out_mp4),
                "--box",
                f"{bx},{by},{bw},{bh}",
                "--design-image",
                str(decal_png),
                "--wrap-mode",
                "planar",
                "--blend-mode",
                "normal",
                "--opacity",
                "1.0",
                "--offset-x",
                str(ox),
                "--offset-y",
                str(oy),
            ],
            timeout=45.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_video_run_report_invariants(report, expect_output_video=True)
        out_f = decode_video_frames(out_mp4, [f_idx])[f_idx]
        out_roi = out_f[ry : ry + rh, rx : rx + rw].astype(np.float32)
        weights = np.clip(np.mean(np.abs(out_roi - raw_roi), axis=-1) - 12.0, 0.0, None)
        total_w = float(np.sum(weights))
        assert total_w > 100.0, f"Decal not visible inside ROI for {label} ({ox}, {oy})"
        ys, xs = np.indices(weights.shape, dtype=np.float32)
        cx = float(np.sum(xs * weights) / total_w)
        cy = float(np.sum(ys * weights) / total_w)
        centroids[label] = (cx, cy)

    cx_neg, cy_neg = centroids["neg_x"]
    cx_cen, cy_cen = centroids["center"]
    cx_pos, cy_pos = centroids["pos_xy"]

    dx1 = cx_cen - cx_neg
    dx2 = cx_pos - cx_cen
    assert abs(dx1) >= 8.0 and abs(dx2) >= 8.0 and (dx1 * dx2 > 0.0), (
        f"Expected monotonic horizontal centroid shift across offset_x=(-0.2, 0.0, +0.2): {centroids}"
    )
    assert abs(cy_cen - cy_neg) <= 5.0, (
        f"Pure offset_x change should not shift vertical centroid significantly: {centroids}"
    )
    assert abs(cy_pos - cy_cen) >= 6.0, (
        f"Expected offset_y=0.18 to shift vertical centroid significantly: {centroids}"
    )


@pytest.mark.f10
def test_f10_opacity_scales_blend_strength_linearly(e2e_tmp_dir: Path) -> None:
    """F10-6: `--opacity 0.25`, `0.60`, and `1.0` strictly and proportionally increase foreground RGB delta."""
    clip_path = e2e_tmp_dir / "opacity_src.mp4"
    clip_meta = make_textured_tracking_clip(clip_path, frames=10, fps=30, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = clip_meta["bbox_frame0"]
    f_idx = 3
    raw_f = decode_video_frames(clip_path, [f_idx])[f_idx]
    bbox_f = bbox_at_frame(clip_meta, f_idx)

    deltas: dict[float, float] = {}
    for opacity in (0.25, 0.60, 1.0):
        out_mp4 = e2e_tmp_dir / f"op_{opacity}.mp4"
        _, report = run_cli_json(
            [
                "process",
                "--input",
                str(clip_path),
                "--output",
                str(out_mp4),
                "--box",
                f"{bx},{by},{bw},{bh}",
                "--pattern",
                "solid",
                "--colors",
                "#00ff66,#00ff66",
                "--wrap-mode",
                "planar",
                "--blend-mode",
                "normal",
                "--opacity",
                str(opacity),
            ],
            timeout=45.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_video_run_report_invariants(report, expect_output_video=True)
        out_f = decode_video_frames(out_mp4, [f_idx])[f_idx]
        roi_d, bg_d = compute_roi_and_bg_delta(raw_f, out_f, bbox_f)
        assert bg_d <= 3.5, f"Background altered at opacity={opacity}: bg_delta={bg_d:.2f}"
        deltas[opacity] = roi_d

    d25, d60, d100 = deltas[0.25], deltas[0.60], deltas[1.0]
    assert 5.0 < d25 < d60 < d100, f"Expected strict monotonic RGB delta 5 < d25 < d60 < d100, got {deltas}"
    assert 0.14 <= (d25 / d100) <= 0.44, f"Expected opacity=0.25 ratio near 0.25, got {d25 / d100:.3f} ({deltas})"
    assert 0.42 <= (d60 / d100) <= 0.78, f"Expected opacity=0.60 ratio near 0.60, got {d60 / d100:.3f} ({deltas})"


# ==============================================================================
# F11 — Streaming >=15 FPS Video CLI (`process`, `info`, `segment`) (6 tests)
# ==============================================================================


@pytest.mark.f11
def test_f11_info_subcommand_probes_mp4_metadata_accurately(
    demo_corpus_dir: Path,
    e2e_tmp_dir: Path,
) -> None:
    """F11-1: `visualizeit info --json` returns exact documented keys matching independent `ffprobe`."""
    planar_mp4 = demo_corpus_dir / "planar_card_640x480.mp4"
    ff_meta = probe_video(planar_mp4, tmp_dir=e2e_tmp_dir)

    _, info = run_cli_json(["info", "--input", str(planar_mp4)], timeout=30.0, tmp_dir=e2e_tmp_dir)
    required_keys = {"path", "width", "height", "fps", "frame_count", "duration_s", "codec"}
    missing = required_keys - set(info.keys())
    assert not missing, f"`visualizeit info` missing keys {missing}: {info}"

    assert Path(info["path"]).resolve() == planar_mp4.resolve()
    assert int(info["width"]) == ff_meta["width"] == 640
    assert int(info["height"]) == ff_meta["height"] == 480
    assert int(info["frame_count"]) == ff_meta["frame_count"]
    assert abs(float(info["fps"]) - float(ff_meta["fps"])) < 0.25
    assert abs(float(info["duration_s"]) - float(ff_meta["duration_s"])) < 0.25
    codec_lower = str(info["codec"]).lower()
    assert "264" in codec_lower or "avc" in codec_lower, f"Expected H.264/AVC codec string, got {info['codec']!r}"


@pytest.mark.f11
def test_f11_info_and_process_support_mp4_webm_mov_and_avi_containers(e2e_tmp_dir: Path) -> None:
    """F11-2: `info` and `process` support `.mp4`, `.webm`, `.mov`, and `.avi` containers with exact frame equality."""
    base_mp4 = e2e_tmp_dir / "src_24f.mp4"
    clip_meta = make_textured_tracking_clip(base_mp4, width=640, height=480, frames=24, fps=30, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = clip_meta["bbox_frame0"]

    container_paths: dict[str, Path] = {".mp4": base_mp4}
    for ext in (".webm", ".mov", ".avi"):
        dst = e2e_tmp_dir / f"src_24f{ext}"
        container_paths[ext] = transcode_clip(base_mp4, dst, tmp_dir=e2e_tmp_dir)

    for ext, in_path in container_paths.items():
        _, info = run_cli_json(["info", "--input", str(in_path)], timeout=30.0, tmp_dir=e2e_tmp_dir)
        assert int(info["width"]) == 640 and int(info["height"]) == 480, f"Bad dimensions for {ext}: {info}"
        assert int(info["frame_count"]) == 24, f"Expected 24 frames from `info` on {ext}, got {info}"

        out_mp4 = e2e_tmp_dir / f"out_from_{ext.lstrip('.')}.mp4"
        _, report = run_cli_json(
            [
                "process",
                "--input",
                str(in_path),
                "--output",
                str(out_mp4),
                "--box",
                f"{bx},{by},{bw},{bh}",
                "--pattern",
                "stripes",
                "--colors",
                "#ff595e,#1982c4",
                "--wrap-mode",
                "planar",
                "--blend-mode",
                "luminance",
            ],
            timeout=60.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_video_run_report_invariants(report, expect_output_video=True)
        assert (
            report["input_frame_count"]
            == report["processed_frame_count"]
            == report["rendered_frame_count"]
            == report["output_frame_count"]
            == 24
        ), f"Frame count mismatch processing {ext}: {report}"
        out_probe = probe_video(out_mp4, tmp_dir=e2e_tmp_dir)
        assert out_probe["codec_name"] == "h264"
        assert out_probe["frame_count"] == 24


@pytest.mark.f11
def test_f11_process_enforces_render_stride_one_and_four_way_frame_equality(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """F11-3: `process` enforces `render_stride == 1`, 4-way frame equality, and H.264 `yuv420p` output."""
    planar_mp4 = demo_corpus_dir / "planar_card_640x480.mp4"
    sel_args = _selection_to_process_args(demo_corpus_manifest["planar_card_640x480"].get("default_selection"))
    in_probe = probe_video(planar_mp4, tmp_dir=e2e_tmp_dir)

    out_mp4 = e2e_tmp_dir / "planar_processed.mp4"
    report_json = e2e_tmp_dir / "planar_report.json"

    _, report = run_cli_json(
        [
            "process",
            "--input",
            str(planar_mp4),
            "--output",
            str(out_mp4),
            "--report",
            str(report_json),
            "--pattern",
            "waves",
            "--colors",
            "#ff595e,#1982c4,#ffca3a",
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "luminance",
            *sel_args,
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, expect_output_video=True)
    assert report["render_stride"] == 1
    expected_n = in_probe["frame_count"]
    assert (
        report["input_frame_count"]
        == report["processed_frame_count"]
        == report["rendered_frame_count"]
        == report["output_frame_count"]
        == expected_n
    )

    out_probe = probe_video(out_mp4, tmp_dir=e2e_tmp_dir)
    assert out_probe["codec_name"] == "h264"
    assert out_probe["pix_fmt"] == "yuv420p"
    assert out_probe["width"] == 640 and out_probe["height"] == 480
    assert out_probe["frame_count"] == expected_n


@pytest.mark.f11
def test_f11_process_report_file_and_stdout_json_match_contract_schema(e2e_tmp_dir: Path) -> None:
    """F11-4: `--report <json_path>` and `--json` stdout parse identically and satisfy full `VideoRunReport` schema."""
    clip_path = e2e_tmp_dir / "schema_clip.mp4"
    clip_meta = make_textured_tracking_clip(clip_path, frames=20, fps=30, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = clip_meta["bbox_frame0"]
    out_mp4 = e2e_tmp_dir / "schema_out.mp4"
    report_path = e2e_tmp_dir / "schema_report.json"

    _, stdout_report = run_cli_json(
        [
            "process",
            "--input",
            str(clip_path),
            "--output",
            str(out_mp4),
            "--report",
            str(report_path),
            "--box",
            f"{bx},{by},{bw},{bh}",
            "--pattern",
            "grid",
            "--colors",
            "#1982c4,#ffca3a",
        ],
        timeout=60.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert report_path.is_file() and report_path.stat().st_size > 0
    file_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert_video_run_report_invariants(stdout_report, expect_output_video=True)
    assert_video_run_report_invariants(file_report, expect_output_video=True)
    assert file_report == stdout_report, (
        f"Mismatch between --report JSON file and --json stdout:\nFILE={file_report}\nSTDOUT={stdout_report}"
    )


@pytest.mark.f11
def test_f11_process_supports_both_native_cpp_and_opencv_raster_backends(e2e_tmp_dir: Path) -> None:
    """F11-5: `process` supports `--raster-backend native_cpp` and `opencv` with visual parity (`PSNR >= 25 dB`)."""
    clip_path = e2e_tmp_dir / "backends_src.mp4"
    clip_meta = make_textured_tracking_clip(clip_path, frames=18, fps=30, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = clip_meta["bbox_frame0"]
    f_idx = 5
    raw_f = decode_video_frames(clip_path, [f_idx])[f_idx]
    bbox_f = bbox_at_frame(clip_meta, f_idx)

    rendered_f5: dict[str, Any] = {}
    for backend in ("native_cpp", "opencv"):
        out_mp4 = e2e_tmp_dir / f"out_{backend}.mp4"
        _, report = run_cli_json(
            [
                "process",
                "--input",
                str(clip_path),
                "--output",
                str(out_mp4),
                "--box",
                f"{bx},{by},{bw},{bh}",
                "--pattern",
                "checker",
                "--colors",
                "#ff595e,#1982c4",
                "--wrap-mode",
                "planar",
                "--blend-mode",
                "luminance",
                "--opacity",
                "0.90",
                "--raster-backend",
                backend,
            ],
            timeout=60.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_video_run_report_invariants(report, expect_output_video=True)
        assert report["raster_backend"] == backend

        out_f = decode_video_frames(out_mp4, [f_idx])[f_idx]
        roi_d, bg_d = compute_roi_and_bg_delta(raw_f, out_f, bbox_f)
        assert roi_d >= 8.0, f"Backend {backend} failed to render foreground artwork: roi_delta={roi_d:.2f}"
        assert bg_d <= 3.5, f"Backend {backend} altered background: bg_delta={bg_d:.2f}"
        rendered_f5[backend] = out_f

    psnr = compute_psnr(rendered_f5["native_cpp"], rendered_f5["opencv"])
    assert psnr >= 25.0, f"Expected visual consistency (PSNR >= 25 dB) between native_cpp and opencv, got {psnr:.2f} dB"


@pytest.mark.f11
def test_f11_cli_entrypoint_works_via_python_m_and_console_script(e2e_tmp_dir: Path) -> None:
    """F11-6: Both `python -m visualizeit --help` and `.cache/venv/bin/visualizeit --help` list all 6 subcommands."""
    expected_subcommands = ("demo-assets", "info", "segment", "process", "benchmark", "serve")

    proc_mod = run_cli(["--help"], timeout=15.0, tmp_dir=e2e_tmp_dir)
    assert proc_mod.returncode == 0, f"`python -m visualizeit --help` failed: {proc_mod.stderr}"
    assert_no_python_traceback(proc_mod)
    for sub in expected_subcommands:
        assert sub in proc_mod.stdout, f"Subcommand {sub!r} missing from `python -m visualizeit --help`:\n{proc_mod.stdout}"

    console_script = CACHE_DIR / "venv" / "bin" / "visualizeit"
    if console_script.is_file():
        proc_bin = run_subproc([str(console_script), "--help"], timeout=15.0, tmp_dir=e2e_tmp_dir)
        assert proc_bin.returncode == 0, f"`visualizeit --help` console script failed: {proc_bin.stderr}"
        assert_no_python_traceback(proc_bin)
        for sub in expected_subcommands:
            assert sub in proc_bin.stdout, f"Subcommand {sub!r} missing from `visualizeit --help`:\n{proc_bin.stdout}"


# ==============================================================================
# F12 — Honest >=15 FPS Benchmark & Telemetry Gate (`benchmark`) (6 tests)
# ==============================================================================


@pytest.mark.f12
def test_f12_benchmark_quick_preset_passes_15fps_gate_with_exit_0(e2e_tmp_dir: Path) -> None:
    """F12-1: `benchmark --preset quick --min-fps 15.0` exits 0 with effective & steady-state FPS >= 15.0."""
    out_dir = e2e_tmp_dir / "bench_quick"
    report_json = e2e_tmp_dir / "bench_quick.json"

    _, report = run_cli_json(
        [
            "benchmark",
            "--preset",
            "quick",
            "--min-fps",
            "15.0",
            "--output-dir",
            str(out_dir),
            "--report",
            str(report_json),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, require_min_fps=15.0, expect_output_video=True)
    assert float(report["effective_fps"]) >= 15.0
    assert float(report["steady_state_fps"]) >= 15.0
    assert report["render_stride"] == 1
    assert report_json.is_file() and report_json.stat().st_size > 0


@pytest.mark.f12
def test_f12_benchmark_standard_and_cylinder_presets_report_complete_telemetry(e2e_tmp_dir: Path) -> None:
    """F12-2: `standard` and `cylinder` benchmark presets emit monotonic latency percentiles and all 5 stage timings."""
    for preset in ("standard", "cylinder"):
        out_dir = e2e_tmp_dir / f"bench_{preset}"
        _, report = run_cli_json(
            [
                "benchmark",
                "--preset",
                preset,
                "--min-fps",
                "15.0",
                "--output-dir",
                str(out_dir),
            ],
            timeout=90.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_video_run_report_invariants(report, require_min_fps=15.0, expect_output_video=True)
        fl = report["frame_latency_ms"]
        assert 0.0 < float(fl["mean"]) <= float(fl["p95"]) + 15.0
        assert 0.0 < float(fl["p50"]) <= float(fl["p95"]) <= float(fl["p99"]) <= float(fl["max"]) + 1e-5
        sl = report["stage_latency_mean_ms"]
        stage_sum = sum(float(sl[k]) for k in ("decode_ms", "prep_ms", "track_ms", "composite_ms", "encode_ms"))
        assert stage_sum > 0.0 and math.isfinite(stage_sum), f"Stage latency sum must be positive: {sl}"
        if preset == "cylinder":
            assert report["wrap_mode"] == "cylinder"


@pytest.mark.f12
def test_f12_benchmark_hd_1280x720_preset_sustains_at_least_15fps(e2e_tmp_dir: Path) -> None:
    """F12-3: `benchmark --preset hd --min-fps 15.0` processes 1280x720 video at >= 15.0 FPS with stride=1."""
    out_dir = e2e_tmp_dir / "bench_hd"
    _, report = run_cli_json(
        [
            "benchmark",
            "--preset",
            "hd",
            "--min-fps",
            "15.0",
            "--output-dir",
            str(out_dir),
        ],
        timeout=120.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, require_min_fps=15.0, expect_output_video=True)
    assert report["width"] == 1280 and report["height"] == 720
    assert float(report["effective_fps"]) >= 15.0
    assert report["input_frame_count"] == report["output_frame_count"]


@pytest.mark.f12
def test_f12_benchmark_wall_clock_honesty_cross_checked_against_external_timer(
    demo_corpus_dir: Path,
    e2e_tmp_dir: Path,
) -> None:
    """F12-4: `benchmark` reported `wall_time_seconds` and `effective_fps` agree with external `time.perf_counter()`."""
    planar_mp4 = demo_corpus_dir / "planar_card_640x480.mp4"
    out_dir = e2e_tmp_dir / "bench_honesty"

    t0 = time.perf_counter()
    _, report = run_cli_json(
        [
            "benchmark",
            "--input",
            str(planar_mp4),
            "--min-fps",
            "15.0",
            "--output-dir",
            str(out_dir),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    ext_wall_s = time.perf_counter() - t0

    assert_video_run_report_invariants(report, require_min_fps=15.0, expect_output_video=True)
    rep_wall = float(report["wall_time_seconds"])
    rep_fps = float(report["effective_fps"])
    n_proc = int(report["processed_frame_count"])

    assert abs(rep_fps - (n_proc / rep_wall)) < 0.5, (
        f"effective_fps={rep_fps:.2f} inconsistent with processed_frame_count/wall_time_seconds={n_proc / rep_wall:.2f}"
    )
    assert rep_wall <= ext_wall_s + 0.5, (
        f"Reported wall_time_seconds={rep_wall:.3f}s exceeds external subprocess lifetime {ext_wall_s:.3f}s"
    )
    # Guard against clock-spoofing (a real 180-frame 640x480 H.264 decode+track+encode takes > 0.15s on 2 vCPUs)
    assert rep_wall >= 0.08, f"Suspiciously tiny reported wall_time_seconds={rep_wall:.5f}s for {n_proc} frames"

    out_mp4 = Path(report["output_path"])
    assert out_mp4.is_file() and out_mp4.stat().st_size > 0
    assert probe_video(out_mp4, tmp_dir=e2e_tmp_dir)["frame_count"] == report["input_frame_count"]


@pytest.mark.f12
def test_f12_benchmark_custom_input_video_and_output_dir_writes_artifacts(
    demo_corpus_dir: Path,
    e2e_tmp_dir: Path,
) -> None:
    """F12-5: `benchmark --input <mp4> --output-dir <dir> --report <json>` writes both rendered MP4 and JSON report."""
    mug_mp4 = demo_corpus_dir / "cylinder_mug_640x480.mp4"
    out_dir = e2e_tmp_dir / "bench_artifacts"
    bench_json = out_dir / "bench.json"

    _, report = run_cli_json(
        [
            "benchmark",
            "--input",
            str(mug_mp4),
            "--min-fps",
            "15.0",
            "--output-dir",
            str(out_dir),
            "--report",
            str(bench_json),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, require_min_fps=15.0, expect_output_video=True)
    assert bench_json.is_file() and bench_json.stat().st_size > 0

    out_mp4 = Path(report["output_path"])
    assert out_mp4.is_file() and out_mp4.stat().st_size > 0
    assert out_dir.resolve() in out_mp4.resolve().parents
    probe = probe_video(out_mp4, tmp_dir=e2e_tmp_dir)
    assert probe["codec_name"] == "h264"
    assert probe["frame_count"] == report["input_frame_count"]


@pytest.mark.f12
def test_f12_benchmark_all_preset_aggregates_and_passes_all_scenarios(e2e_tmp_dir: Path) -> None:
    """F12-6: `benchmark --preset all --min-fps 15.0` runs multi-scenario suite and passes >=15 FPS on every scenario."""
    out_dir = e2e_tmp_dir / "bench_all"
    report_json = e2e_tmp_dir / "bench_all.json"

    _, payload = run_cli_json(
        [
            "benchmark",
            "--preset",
            "all",
            "--min-fps",
            "15.0",
            "--output-dir",
            str(out_dir),
            "--report",
            str(report_json),
        ],
        timeout=240.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert report_json.is_file() and report_json.stat().st_size > 0

    scenario_reports = _extract_benchmark_reports(payload)
    assert len(scenario_reports) >= 4, (
        f"Expected `benchmark --preset all` to include >=4 scenario VideoRunReports, got {len(scenario_reports)}: {payload}"
    )
    for idx, rep in enumerate(scenario_reports):
        assert_video_run_report_invariants(rep, require_min_fps=15.0, expect_output_video=True)
        assert rep["render_stride"] == 1, f"Scenario #{idx} violated render_stride == 1: {rep}"
