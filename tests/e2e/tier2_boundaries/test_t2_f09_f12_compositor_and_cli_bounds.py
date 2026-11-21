"""Tier 2 Boundary, Corner-Case & Negative Error-Handling E2E Tests for Features F9–F12 (24 tests).

Strictly opaque-box:
- Zero imports of `visualizeit.*` anywhere.
- Zero top-level third-party imports (besides `pytest`).
- Enforces `PROJECT.md §4` exit code contract (`2` = invalid input/usage error with clean `stderr` and zero
  Python tracebacks; `1` = runtime / benchmark `--min-fps` gate failure with valid `VideoRunReport` JSON).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.e2e._helpers import (
    CODEC_ARGS,
    assert_cli_usage_error,
    assert_no_python_traceback,
    assert_video_run_report_invariants,
    bbox_at_frame,
    compute_psnr,
    compute_roi_and_bg_delta,
    decode_video_frames,
    make_occlusion_clip,
    make_rgba_png,
    make_textured_tracking_clip,
    probe_video,
    run_cli,
    run_cli_json,
    run_subproc,
    spawn_live_server,
)

pytestmark = pytest.mark.tier2


def _make_flash_glitch_clip(
    out_path: Path,
    *,
    frames: int = 22,
    flash_frame: int = 10,
    tmp_dir: Path,
) -> dict[str, Any]:
    """Generate a 640x480 tracking clip with a 1-frame full-white (`#ffffff`) camera flash at `flash_frame`."""
    width, height, fps = 640, 480, 30
    bw, bh, x0, y0, dx, dy = 200, 150, 150, 120, 4, 2
    bg = (
        f"color=c=0x1e293b:s={width}x{height}:r={fps},"
        f"noise=all_seed=7:alls=8,trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/({fps}*TB)"
    )
    fg = (
        f"testsrc2=s={bw}x{bh}:r={fps},noise=all_seed=42:alls=25,"
        f"drawgrid=w=24:h=24:t=2:c=white@0.85,"
        f"trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/({fps}*TB)"
    )
    filt = (
        f"[0:v][1:v]overlay=x={x0}+{dx}*n:y={y0}+{dy}*n:eval=frame,"
        f"drawbox=x=0:y=0:w={width}:h={height}:c=white:t=fill:enable='eq(n,{flash_frame})',"
        "format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        bg,
        "-f",
        "lavfi",
        "-i",
        fg,
        "-filter_complex",
        filt,
        "-frames:v",
        str(frames),
        *CODEC_ARGS[".mp4"],
        str(out_path),
    ]
    run_subproc(cmd, timeout=25.0, check=True, tmp_dir=tmp_dir)
    return {
        "path": str(out_path),
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "x0": x0,
        "y0": y0,
        "box_w": bw,
        "box_h": bh,
        "dx": dx,
        "dy": dy,
        "bbox_frame0": (x0, y0, bw, bh),
        "flash_frame": flash_frame,
    }


def _make_gradual_occlusion_clip(out_path: Path, *, tmp_dir: Path) -> dict[str, Any]:
    """Generate a 32-frame 640x480 clip cycling 0% -> 50% -> 100% -> 50% -> 0% card occlusion."""
    width, height, fps, frames = 640, 480, 30, 32
    bw, bh, x0, y0, dx, dy = 220, 160, 170, 140, 2, 1
    bg = (
        f"color=c=0x1e293b:s={width}x{height}:r={fps},"
        f"noise=all_seed=11:alls=6,trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/({fps}*TB)"
    )
    fg = (
        f"testsrc2=s={bw}x{bh}:r={fps},noise=all_seed=42:alls=25,"
        f"drawgrid=w=26:h=26:t=2:c=white@0.85,"
        f"trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/({fps}*TB)"
    )
    # 8..13: left 50% covered; 14..21: 100% covered (+margin); 22..25: right 50% covered
    occ_left50 = f"drawbox=x={x0 - 16}:y={y0 - 16}:w={bw // 2 + 24}:h={bh + 48}:c=0x0f172a:t=fill:enable='between(n,8,13)'"
    occ_full = f"drawbox=x={x0 - 20}:y={y0 - 20}:w={bw + 110}:h={bh + 70}:c=0x0f172a:t=fill:enable='between(n,14,21)'"
    occ_right50 = (
        f"drawbox=x={x0 + bw // 2 + 30}:y={y0 - 10}:w={bw // 2 + 60}:h={bh + 60}:"
        f"c=0x0f172a:t=fill:enable='between(n,22,25)'"
    )
    filt = f"[0:v][1:v]overlay=x={x0}+{dx}*n:y={y0}+{dy}*n:eval=frame,{occ_left50},{occ_full},{occ_right50},format=yuv420p"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        bg,
        "-f",
        "lavfi",
        "-i",
        fg,
        "-filter_complex",
        filt,
        "-frames:v",
        str(frames),
        *CODEC_ARGS[".mp4"],
        str(out_path),
    ]
    run_subproc(cmd, timeout=25.0, check=True, tmp_dir=tmp_dir)
    return {
        "path": str(out_path),
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "x0": x0,
        "y0": y0,
        "box_w": bw,
        "box_h": bh,
        "dx": dx,
        "dy": dy,
        "bbox_frame0": (x0, y0, bw, bh),
    }


def _make_distractor_swap_clip(out_path: Path, *, tmp_dir: Path) -> dict[str, Any]:
    """Frames 0..9 show textured card A at top-left; frames 10..27 hide A and show unrelated distractor B at bottom-right."""
    width, height, fps, frames = 640, 480, 30, 28
    ax, ay, aw, ah = 64, 60, 200, 150
    bx, by, bw, bh = 384, 272, 180, 140
    bg = (
        f"color=c=0x1e293b:s={width}x{height}:r={fps},"
        f"trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/({fps}*TB)"
    )
    card_a = (
        f"testsrc2=s={aw}x{ah}:r={fps},noise=all_seed=99:alls=30,"
        f"drawgrid=w=20:h=20:t=2:c=white@0.9,"
        f"trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/({fps}*TB)"
    )
    # Distractor B: smooth low-frequency horizontal color bands with completely unrelated visual structure
    filt = (
        f"[0:v][1:v]overlay=x={ax}+2*n:y={ay}+n:enable='between(n,0,9)':eval=frame,"
        f"drawbox=x={bx}:y={by}:w={bw}:h={bh}:c=0x6b21a8:t=fill:enable='between(n,10,27)',"
        f"drawbox=x={bx}:y={by + 45}:w={bw}:h=45:c=0xf59e0b:t=fill:enable='between(n,10,27)',"
        "format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        bg,
        "-f",
        "lavfi",
        "-i",
        card_a,
        "-filter_complex",
        filt,
        "-frames:v",
        str(frames),
        *CODEC_ARGS[".mp4"],
        str(out_path),
    ]
    run_subproc(cmd, timeout=25.0, check=True, tmp_dir=tmp_dir)
    return {
        "card_a_bbox0": (ax, ay, aw, ah),
        "card_a_bbox4": (ax + 8, ay + 4, aw, ah),
        "distractor_b_bbox": (bx, by, bw, bh),
    }


def _make_black_white_substrate_clip(out_path: Path, *, tmp_dir: Path) -> dict[str, Any]:
    """Generate a 10-frame tracking clip whose card contains pure [0,0,0] black and [255,255,255] white interior patches."""
    width, height, fps, frames = 640, 480, 30, 10
    bw, bh, x0, y0, dx, dy = 260, 180, 180, 140, 2, 1
    bg = (
        f"color=c=0x1e293b:s={width}x{height}:r={fps},"
        f"noise=all_seed=7:alls=8,trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/({fps}*TB)"
    )
    # Rich outer border + grid for KLT/ORB tracking, plus pure-black left sub-patch and pure-white right sub-patch
    fg = (
        f"testsrc2=s={bw}x{bh}:r={fps},noise=all_seed=42:alls=24,"
        f"drawgrid=w=22:h=22:t=2:c=white@0.85,"
        f"drawbox=x=36:y=50:w=70:h=80:c=black:t=fill,"
        f"drawbox=x=154:y=50:w=70:h=80:c=white:t=fill,"
        f"trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/({fps}*TB)"
    )
    filt = f"[0:v][1:v]overlay=x={x0}+{dx}*n:y={y0}+{dy}*n:eval=frame,format=yuv420p"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        bg,
        "-f",
        "lavfi",
        "-i",
        fg,
        "-filter_complex",
        filt,
        "-frames:v",
        str(frames),
        *CODEC_ARGS[".mp4"],
        str(out_path),
    ]
    run_subproc(cmd, timeout=25.0, check=True, tmp_dir=tmp_dir)
    return {
        "x0": x0,
        "y0": y0,
        "box_w": bw,
        "box_h": bh,
        "dx": dx,
        "dy": dy,
        "bbox_frame0": (x0, y0, bw, bh),
        "black_rel": (44, 58, 54, 64),
        "white_rel": (162, 58, 54, 64),
    }


# ==============================================================================
# F9 — Occlusion Suppression & Re-Acquisition Boundaries (6 tests)
# ==============================================================================


@pytest.mark.f09
def test_f09_bound_permanent_object_exit_stays_suppressed_with_zero_recovery(e2e_tmp_dir: Path) -> None:
    """F9-B1: Permanent object exit after frame 7 stays suppressed (`recovered_count == 0`, zero ghost overlay)."""
    import numpy as np

    clip_path = e2e_tmp_dir / "perm_exit.mp4"
    meta = make_occlusion_clip(
        clip_path,
        width=640,
        height=480,
        frames=25,
        fps=30,
        occlude_ranges=[(8, 24)],
        tmp_dir=e2e_tmp_dir,
    )
    bx, by, bw, bh = meta["bbox_frame0"]
    out_mp4 = e2e_tmp_dir / "perm_exit_out.mp4"

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
            "#ff0055,#ffbe0b",
            "--blend-mode",
            "normal",
            "--opacity",
            "0.95",
        ],
        timeout=45.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, expect_output_video=True)
    assert report["lost_frame_count"] >= 14, f"Expected >=14 lost frames after permanent exit, got {report}"
    assert report["recovered_count"] == 0, f"Expected 0 recoveries on permanent exit, got {report}"

    wanted = [2, 12, 18, 24]
    raw_f = decode_video_frames(clip_path, wanted)
    out_f = decode_video_frames(out_mp4, wanted)

    roi_d2, _ = compute_roi_and_bg_delta(raw_f[2], out_f[2], bbox_at_frame(meta, 2))
    assert roi_d2 >= 10.0, f"Expected strong overlay before exit at frame 2, got {roi_d2:.2f}"

    for f_idx in (12, 18, 24):
        diff = float(np.mean(np.abs(out_f[f_idx].astype(np.float32) - raw_f[f_idx].astype(np.float32))))
        assert diff <= 2.5, f"Ghost overlay remained on permanently exited frame {f_idx}: diff={diff:.2f}"


@pytest.mark.f09
def test_f09_bound_single_frame_camera_flash_glitch_recovers_next_frame(e2e_tmp_dir: Path) -> None:
    """F9-B2: A 1-frame pure-white flash at frame 10 suppresses stale overlay and resumes tracking right after."""
    import numpy as np

    clip_path = e2e_tmp_dir / "flash_glitch.mp4"
    meta = _make_flash_glitch_clip(clip_path, frames=22, flash_frame=10, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = meta["bbox_frame0"]
    out_mp4 = e2e_tmp_dir / "flash_glitch_out.mp4"

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
            "#ff006e,#1982c4",
            "--blend-mode",
            "normal",
            "--opacity",
            "0.95",
        ],
        timeout=45.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, expect_output_video=True)

    wanted = [6, 10, 14, 19]
    raw_f = decode_video_frames(clip_path, wanted)
    out_f = decode_video_frames(out_mp4, wanted)

    flash_diff = float(np.mean(np.abs(out_f[10].astype(np.float32) - raw_f[10].astype(np.float32))))
    assert flash_diff <= 3.0, f"Stale overlay leaked onto white flash frame 10: diff={flash_diff:.2f}"

    for vis_idx in (6, 14, 19):
        roi_d, bg_d = compute_roi_and_bg_delta(raw_f[vis_idx], out_f[vis_idx], bbox_at_frame(meta, vis_idx))
        assert roi_d >= 10.0, f"Expected tracking overlay on visible frame {vis_idx}, got roi_delta={roi_d:.2f}"
        assert bg_d <= 3.5, f"Background corrupted on frame {vis_idx}: bg_delta={bg_d:.2f}"


@pytest.mark.f09
def test_f09_bound_partial_50pct_occlusion_vs_full_occlusion_transition(e2e_tmp_dir: Path) -> None:
    """F9-B3: Gradual 0% -> 50% -> 100% -> 50% -> 0% wipe transitions cleanly without polygon blowup."""
    import numpy as np

    clip_path = e2e_tmp_dir / "gradual_occ.mp4"
    meta = _make_gradual_occlusion_clip(clip_path, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = meta["bbox_frame0"]
    out_mp4 = e2e_tmp_dir / "gradual_occ_out.mp4"

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
            "#00f5d4,#ff006e",
            "--blend-mode",
            "normal",
            "--opacity",
            "0.92",
        ],
        timeout=45.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, expect_output_video=True)

    wanted = [4, 10, 17, 23, 29]
    raw_f = decode_video_frames(clip_path, wanted)
    out_f = decode_video_frames(out_mp4, wanted)

    # Full occlusion frame 17 must suppress overlay
    full_occ_diff = float(np.mean(np.abs(out_f[17].astype(np.float32) - raw_f[17].astype(np.float32))))
    assert full_occ_diff <= 2.5, f"Overlay not suppressed at 100% occlusion frame 17: diff={full_occ_diff:.2f}"

    # No full-screen degenerate homography explosion during 50% partial occlusion frames (10, 23)
    for trans_idx in (10, 23):
        _, bg_d = compute_roi_and_bg_delta(raw_f[trans_idx], out_f[trans_idx], bbox_at_frame(meta, trans_idx), margin=48)
        assert bg_d <= 4.0, (
            f"Degenerate homography polygon explosion outside card ROI at partial-occlusion frame {trans_idx}: "
            f"bg_delta={bg_d:.2f}"
        )

    # Pre-wipe (4) and post-wipe (29) must both render strong overlay
    for vis_idx in (4, 29):
        roi_d, bg_d = compute_roi_and_bg_delta(raw_f[vis_idx], out_f[vis_idx], bbox_at_frame(meta, vis_idx))
        assert roi_d >= 10.0, f"Expected overlay at visible frame {vis_idx}, got {roi_d:.2f}"
        assert bg_d <= 3.5


@pytest.mark.f09
def test_f09_bound_false_reacquisition_prevented_on_unrelated_distractor_pattern(e2e_tmp_dir: Path) -> None:
    """F9-B4: When target card A disappears and unrelated distractor B appears elsewhere, B is not falsely locked."""
    import numpy as np

    clip_path = e2e_tmp_dir / "distractor_swap.mp4"
    swap_meta = _make_distractor_swap_clip(clip_path, tmp_dir=e2e_tmp_dir)
    ax, ay, aw, ah = swap_meta["card_a_bbox0"]
    out_mp4 = e2e_tmp_dir / "distractor_swap_out.mp4"

    _, report = run_cli_json(
        [
            "process",
            "--input",
            str(clip_path),
            "--output",
            str(out_mp4),
            "--box",
            f"{ax},{ay},{aw},{ah}",
            "--pattern",
            "solid",
            "--colors",
            "#00ff88,#00ff88",
            "--blend-mode",
            "normal",
            "--opacity",
            "0.95",
        ],
        timeout=45.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, expect_output_video=True)

    wanted = [4, 16, 24]
    raw_f = decode_video_frames(clip_path, wanted)
    out_f = decode_video_frames(out_mp4, wanted)

    roi_a4, _ = compute_roi_and_bg_delta(raw_f[4], out_f[4], swap_meta["card_a_bbox4"])
    assert roi_a4 >= 12.0, f"Expected strong overlay on card A at frame 4, got {roi_a4:.2f}"

    bx, by, bw, bh = swap_meta["distractor_b_bbox"]
    for f_idx in (16, 24):
        dist_diff = float(
            np.mean(
                np.abs(
                    out_f[f_idx][by : by + bh, bx : bx + bw].astype(np.float32)
                    - raw_f[f_idx][by : by + bh, bx : bx + bw].astype(np.float32)
                )
            )
        )
        assert dist_diff <= 3.5, (
            f"False ORB re-acquisition snapped card A's artwork onto unrelated distractor B at frame {f_idx}: "
            f"distractor_diff={dist_diff:.2f}"
        )


@pytest.mark.f09
def test_f09_bound_occlusion_from_frame_1_immediately_after_onboarding_frame_0(e2e_tmp_dir: Path) -> None:
    """F9-B5: Occlusion starting on frame 1 (right after frame-0 onboarding) does not crash and recovers at frame 12+."""
    import numpy as np

    clip_path = e2e_tmp_dir / "occ_frame1.mp4"
    meta = make_occlusion_clip(
        clip_path,
        width=640,
        height=480,
        frames=22,
        fps=30,
        occlude_ranges=[(1, 11)],
        tmp_dir=e2e_tmp_dir,
    )
    bx, by, bw, bh = meta["bbox_frame0"]
    out_mp4 = e2e_tmp_dir / "occ_frame1_out.mp4"

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
            "#ff595e,#ffca3a",
            "--blend-mode",
            "normal",
            "--opacity",
            "0.95",
        ],
        timeout=45.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, expect_output_video=True)
    assert report["lost_frame_count"] >= 8, f"Expected >=8 lost frames during frames 1..11, got {report}"
    assert report["recovered_count"] >= 1, f"Expected recovery when card returns at frame 12, got {report}"

    raw_f = decode_video_frames(clip_path, [6, 17])
    out_f = decode_video_frames(out_mp4, [6, 17])

    occ6_diff = float(np.mean(np.abs(out_f[6].astype(np.float32) - raw_f[6].astype(np.float32))))
    assert occ6_diff <= 2.5, f"Expected suppressed overlay at occluded frame 6, got {occ6_diff:.2f}"

    roi_17, bg_17 = compute_roi_and_bg_delta(raw_f[17], out_f[17], bbox_at_frame(meta, 17))
    assert roi_17 >= 10.0, f"Expected restored overlay at frame 17 after immediate frame-1 loss, got {roi_17:.2f}"
    assert bg_17 <= 3.5


@pytest.mark.f09
def test_f09_bound_all_frames_occluded_after_frame_0_has_bounded_availability(e2e_tmp_dir: Path) -> None:
    """F9-B6: When only frame 0 of 20 is visible, `tracking_availability` stays in `(0.0, 0.15]` (honesty invariant)."""
    clip_path = e2e_tmp_dir / "only_f0_vis.mp4"
    meta = make_occlusion_clip(
        clip_path,
        width=640,
        height=480,
        frames=20,
        fps=30,
        occlude_ranges=[(1, 19)],
        tmp_dir=e2e_tmp_dir,
    )
    bx, by, bw, bh = meta["bbox_frame0"]
    out_mp4 = e2e_tmp_dir / "only_f0_vis_out.mp4"

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
            "--blend-mode",
            "normal",
            "--opacity",
            "0.90",
        ],
        timeout=45.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, expect_output_video=True)
    avail = float(report["tracking_availability"])
    assert 0.0 < avail <= 0.15, (
        f"Tracking availability honesty violated when 19/20 frames are occluded: "
        f"tracking_availability={avail:.3f}, tracked={report['tracked_frame_count']}, lost={report['lost_frame_count']}"
    )
    assert report["recovered_count"] == 0


# ==============================================================================
# F10 — Multi-Mode Surface Wrapping & Luminance Blending Boundaries (6 tests)
# ==============================================================================


@pytest.mark.f10
def test_f10_bound_opacity_exact_inclusive_0_0_and_1_0(e2e_tmp_dir: Path) -> None:
    """F10-B1: Exact boundary `--opacity 0.0` leaves scene visually unchanged while `--opacity 1.0` fully replaces."""
    clip_path = e2e_tmp_dir / "op_bounds.mp4"
    meta = make_textured_tracking_clip(clip_path, frames=10, fps=30, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = meta["bbox_frame0"]
    f_idx = 3
    raw_f = decode_video_frames(clip_path, [f_idx])[f_idx]
    bbox_f = bbox_at_frame(meta, f_idx)

    out_zero = e2e_tmp_dir / "op_0.mp4"
    _, rep_zero = run_cli_json(
        [
            "process",
            "--input",
            str(clip_path),
            "--output",
            str(out_zero),
            "--box",
            f"{bx},{by},{bw},{bh}",
            "--pattern",
            "solid",
            "--colors",
            "#ff00ff,#ff00ff",
            "--blend-mode",
            "normal",
            "--opacity",
            "0.0",
        ],
        timeout=45.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(rep_zero, expect_output_video=True)
    assert rep_zero["rendered_frame_count"] == rep_zero["input_frame_count"] == 10
    f_zero = decode_video_frames(out_zero, [f_idx])[f_idx]
    roi_zero, _ = compute_roi_and_bg_delta(raw_f, f_zero, bbox_f)
    assert roi_zero <= 3.5 and compute_psnr(raw_f, f_zero) >= 34.0, (
        f"opacity=0.0 should leave scene unchanged within H.264 quantization, got roi_delta={roi_zero:.2f}"
    )

    out_one = e2e_tmp_dir / "op_1.mp4"
    _, rep_one = run_cli_json(
        [
            "process",
            "--input",
            str(clip_path),
            "--output",
            str(out_one),
            "--box",
            f"{bx},{by},{bw},{bh}",
            "--pattern",
            "solid",
            "--colors",
            "#ff00ff,#ff00ff",
            "--blend-mode",
            "normal",
            "--opacity",
            "1.0",
        ],
        timeout=45.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(rep_one, expect_output_video=True)
    f_one = decode_video_frames(out_one, [f_idx])[f_idx]
    roi_one, bg_one = compute_roi_and_bg_delta(raw_f, f_one, bbox_f)
    assert roi_one >= 35.0, f"Expected full opaque replacement delta >= 35.0 at opacity=1.0, got {roi_one:.2f}"
    assert bg_one <= 3.5


@pytest.mark.f10
def test_f10_bound_opacity_out_of_range_minus_0_1_and_1_5_rejected(e2e_tmp_dir: Path) -> None:
    """F10-B2: Out-of-range `opacity` (`-0.1`, `1.5`) is rejected with CLI exit 2 and HTTP 400/422."""
    clip_path = e2e_tmp_dir / "op_oob.mp4"
    meta = make_textured_tracking_clip(clip_path, frames=8, fps=30, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = meta["bbox_frame0"]

    for bad_op in ("-0.1", "1.5"):
        proc = run_cli(
            [
                "process",
                "--input",
                str(clip_path),
                "--output",
                str(e2e_tmp_dir / f"bad_op_{bad_op}.mp4"),
                "--box",
                f"{bx},{by},{bw},{bh}",
                "--opacity",
                bad_op,
            ],
            timeout=30.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)

    with spawn_live_server(storage_dir=e2e_tmp_dir / "srv_op_oob") as client:
        up = client.upload(
            "/api/videos/upload",
            filename="op_oob.mp4",
            content=clip_path.read_bytes(),
            content_type="video/mp4",
            expected_status=200,
        )
        video_id = up["json"]["video_id"]
        for bad_op_num in (-0.1, 1.5):
            client.post_json(
                f"/api/videos/{video_id}/render",
                {
                    "selection": {"frame_index": 0, "bbox": [bx, by, bw, bh]},
                    "design_recipe": {"pattern": "stripes", "colors": ["#ff595e", "#1982c4"]},
                    "placement": {"wrap_mode": "planar", "blend_mode": "normal", "opacity": bad_op_num},
                },
                expected_status=(400, 422),
            )


@pytest.mark.f10
def test_f10_bound_scale_exact_inclusive_0_1_and_5_0(e2e_tmp_dir: Path) -> None:
    """F10-B3: Exact inclusive scale boundaries `--scale 0.1` and `--scale 5.0` both succeed cleanly."""
    clip_path = e2e_tmp_dir / "scale_bounds.mp4"
    meta = make_textured_tracking_clip(clip_path, frames=10, fps=30, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = meta["bbox_frame0"]

    for scale_val in ("0.1", "5.0"):
        out_mp4 = e2e_tmp_dir / f"scale_exact_{scale_val}.mp4"
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
                "#ff595e,#ffca3a",
                "--scale",
                scale_val,
            ],
            timeout=45.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_video_run_report_invariants(report, expect_output_video=True)
        assert probe_video(out_mp4, tmp_dir=e2e_tmp_dir)["frame_count"] == 10


@pytest.mark.f10
def test_f10_bound_scale_out_of_range_0_0_negative_and_6_0_rejected(e2e_tmp_dir: Path) -> None:
    """F10-B4: Out-of-range `scale` (`0.0`, `-1.0`, `6.0`) is rejected with CLI exit 2 and HTTP 400/422."""
    clip_path = e2e_tmp_dir / "scale_oob.mp4"
    meta = make_textured_tracking_clip(clip_path, frames=8, fps=30, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = meta["bbox_frame0"]

    for bad_scale in ("0.0", "-1.0", "6.0"):
        proc = run_cli(
            [
                "process",
                "--input",
                str(clip_path),
                "--output",
                str(e2e_tmp_dir / f"bad_scale_{bad_scale}.mp4"),
                "--box",
                f"{bx},{by},{bw},{bh}",
                "--scale",
                bad_scale,
            ],
            timeout=30.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)

    with spawn_live_server(storage_dir=e2e_tmp_dir / "srv_scale_oob") as client:
        up = client.upload(
            "/api/videos/upload",
            filename="scale_oob.mp4",
            content=clip_path.read_bytes(),
            content_type="video/mp4",
            expected_status=200,
        )
        video_id = up["json"]["video_id"]
        for bad_scale_num in (0.0, 6.0):
            client.post_json(
                f"/api/videos/{video_id}/render",
                {
                    "selection": {"frame_index": 0, "bbox": [bx, by, bw, bh]},
                    "design_recipe": {"pattern": "stripes", "colors": ["#ff595e", "#1982c4"]},
                    "placement": {"wrap_mode": "planar", "blend_mode": "luminance", "scale": bad_scale_num},
                },
                expected_status=(400, 422),
            )


@pytest.mark.f10
def test_f10_bound_extreme_rotation_angles_minus_720_and_plus_720_wrap_cleanly(e2e_tmp_dir: Path) -> None:
    """F10-B5: Multi-turn rotations (`-360`, `+720`) and `offset_x/y = ±1.0` render cleanly with modular equivalence."""
    clip_path = e2e_tmp_dir / "rot_extreme.mp4"
    meta = make_textured_tracking_clip(clip_path, frames=10, fps=30, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = meta["bbox_frame0"]

    out_m360 = e2e_tmp_dir / "rot_m360.mp4"
    out_p720 = e2e_tmp_dir / "rot_p720.mp4"
    out_off = e2e_tmp_dir / "rot_m720_off.mp4"

    for out_mp4, extra in (
        (out_m360, ["--rotation-deg", "-360"]),
        (out_p720, ["--rotation-deg", "720"]),
        (out_off, ["--rotation-deg", "-720", "--offset-x", "-1.0", "--offset-y", "1.0"]),
    ):
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
                "#ff595e,#1982c4",
                "--blend-mode",
                "normal",
                "--opacity",
                "0.95",
                *extra,
            ],
            timeout=45.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_video_run_report_invariants(report, expect_output_video=True)

    f_m360 = decode_video_frames(out_m360, [2])[2]
    f_p720 = decode_video_frames(out_p720, [2])[2]
    assert compute_psnr(f_m360, f_p720) >= 38.0, "Expected -360 deg and +720 deg full-turn rotations to match"


@pytest.mark.f10
def test_f10_bound_luminance_blend_on_pitch_black_0_and_saturated_white_255_substrate(e2e_tmp_dir: Path) -> None:
    """F10-B6: `luminance` blend over `[0,0,0]` black and `[255,255,255]` white substrate has zero NaN/uint8 overflow."""
    import numpy as np

    clip_path = e2e_tmp_dir / "bw_substrate.mp4"
    meta = _make_black_white_substrate_clip(clip_path, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = meta["bbox_frame0"]
    out_mp4 = e2e_tmp_dir / "bw_substrate_out.mp4"

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
            "#30a0ff,#30a0ff",
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "luminance",
            "--opacity",
            "1.0",
        ],
        timeout=45.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, expect_output_video=True)

    f_idx = 2
    out_f = decode_video_frames(out_mp4, [f_idx])[f_idx]
    assert out_f.dtype == np.uint8 and out_f.shape == (480, 640, 3)

    rx, ry, _, _ = bbox_at_frame(meta, f_idx)
    bk_x, bk_y, bk_w, bk_h = meta["black_rel"]
    wh_x, wh_y, wh_w, wh_h = meta["white_rel"]

    black_patch = out_f[ry + bk_y : ry + bk_y + bk_h, rx + bk_x : rx + bk_x + bk_w].astype(np.float32)
    white_patch = out_f[ry + wh_y : ry + wh_y + wh_h, rx + wh_x : rx + wh_x + wh_w].astype(np.float32)

    black_luma = float(np.mean(0.299 * black_patch[..., 0] + 0.587 * black_patch[..., 1] + 0.114 * black_patch[..., 2]))
    white_luma = float(np.mean(0.299 * white_patch[..., 0] + 0.587 * white_patch[..., 1] + 0.114 * white_patch[..., 2]))

    assert black_luma <= 30.0, f"Pitch-black [0,0,0] substrate overflowed or brightened under luminance: {black_luma:.1f}"
    assert white_luma >= 110.0, f"Saturated-white [255,255,255] substrate wrapped around under luminance: {white_luma:.1f}"


# ==============================================================================
# F11 — Streaming Video CLI Boundaries (6 tests)
# ==============================================================================


@pytest.mark.f11
def test_f11_bound_missing_nonexistent_input_video_exits_2_across_subcommands(e2e_tmp_dir: Path) -> None:
    """F11-B1: Nonexistent `--input` video exits with code 2 and zero traceback across `info`, `segment`, `process`."""
    missing = e2e_tmp_dir / "does_not_exist_999.mp4"
    out_mp4 = e2e_tmp_dir / "unused.mp4"

    for args in (
        ["info", "--input", str(missing)],
        ["segment", "--input", str(missing), "--point", "100,100"],
        ["process", "--input", str(missing), "--output", str(out_mp4)],
    ):
        proc = run_cli(args, timeout=20.0, tmp_dir=e2e_tmp_dir)
        assert_cli_usage_error(proc)


@pytest.mark.f11
def test_f11_bound_empty_zero_byte_mp4_file_exits_2(e2e_tmp_dir: Path) -> None:
    """F11-B2: 0-byte empty `.mp4` file exits with code 2 and clean `stderr` across `info`, `segment`, `process`."""
    empty_mp4 = e2e_tmp_dir / "empty.mp4"
    empty_mp4.write_bytes(b"")
    out_mp4 = e2e_tmp_dir / "unused_empty_out.mp4"

    for args in (
        ["info", "--input", str(empty_mp4)],
        ["segment", "--input", str(empty_mp4), "--point", "100,100"],
        ["process", "--input", str(empty_mp4), "--output", str(out_mp4)],
    ):
        proc = run_cli(args, timeout=20.0, tmp_dir=e2e_tmp_dir)
        assert_cli_usage_error(proc)


@pytest.mark.f11
def test_f11_bound_corrupt_truncated_mp4_header_and_garbage_bytes_exit_2(e2e_tmp_dir: Path) -> None:
    """F11-B3: Corrupt garbage bytes and truncated MP4 headers exit cleanly with code 2 and no hang/traceback."""
    garbage_mp4 = e2e_tmp_dir / "garbage.mp4"
    garbage_mp4.write_bytes(b"ftypisom_corrupt_garbage_bytes_0123456789")

    valid_mp4 = e2e_tmp_dir / "valid_ref.mp4"
    make_textured_tracking_clip(valid_mp4, frames=6, fps=30, tmp_dir=e2e_tmp_dir)
    truncated_mp4 = e2e_tmp_dir / "truncated_header.mp4"
    truncated_mp4.write_bytes(valid_mp4.read_bytes()[:256])

    for bad_mp4 in (garbage_mp4, truncated_mp4):
        proc_info = run_cli(["info", "--input", str(bad_mp4)], timeout=20.0, tmp_dir=e2e_tmp_dir)
        assert_cli_usage_error(proc_info)

        proc_run = run_cli(
            ["process", "--input", str(bad_mp4), "--output", str(e2e_tmp_dir / f"out_{bad_mp4.name}")],
            timeout=20.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc_run)


@pytest.mark.f11
def test_f11_bound_non_video_png_or_text_passed_as_video_input_exits_2(e2e_tmp_dir: Path) -> None:
    """F11-B4: Plain `.txt` and static `.png` files passed as `--input` video exit with code 2 and zero traceback."""
    txt_file = e2e_tmp_dir / "readme.txt"
    txt_file.write_text("plain text document, not a video container\n", encoding="utf-8")

    png_file = make_rgba_png(e2e_tmp_dir / "still_image.png", width=128, height=128, mode="emblem")

    for non_video in (txt_file, png_file):
        proc_info = run_cli(["info", "--input", str(non_video)], timeout=20.0, tmp_dir=e2e_tmp_dir)
        assert_cli_usage_error(proc_info)

        proc_run = run_cli(
            ["process", "--input", str(non_video), "--output", str(e2e_tmp_dir / f"out_{non_video.stem}.mp4")],
            timeout=20.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc_run)


@pytest.mark.f11
def test_f11_bound_invalid_wrap_mode_blend_mode_and_raster_backend_exit_2(e2e_tmp_dir: Path) -> None:
    """F11-B5: Unsupported `--wrap-mode`, `--blend-mode`, and `--raster-backend` values exit with code 2."""
    clip_path = e2e_tmp_dir / "enum_bounds.mp4"
    make_textured_tracking_clip(clip_path, frames=6, fps=30, tmp_dir=e2e_tmp_dir)

    for idx, bad_flags in enumerate(
        (
            ["--wrap-mode", "spherical"],
            ["--blend-mode", "dodge"],
            ["--raster-backend", "cuda"],
        )
    ):
        proc = run_cli(
            [
                "process",
                "--input",
                str(clip_path),
                "--output",
                str(e2e_tmp_dir / f"bad_enum_{idx}.mp4"),
                *bad_flags,
            ],
            timeout=20.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)


@pytest.mark.f11
def test_f11_bound_unwritable_or_directory_output_mp4_path_exits_cleanly(e2e_tmp_dir: Path) -> None:
    """F11-B6: Passing an existing directory or file-as-parent path to `--output` fails cleanly without traceback."""
    clip_path = e2e_tmp_dir / "unwritable_src.mp4"
    meta = make_textured_tracking_clip(clip_path, frames=6, fps=30, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = meta["bbox_frame0"]

    existing_dir = e2e_tmp_dir / "already_a_directory"
    existing_dir.mkdir(parents=True, exist_ok=True)

    regular_file = e2e_tmp_dir / "not_a_dir_file"
    regular_file.write_text("blocker", encoding="utf-8")
    blocked_child = regular_file / "out.mp4"

    for bad_out in (existing_dir, blocked_child):
        proc = run_cli(
            [
                "process",
                "--input",
                str(clip_path),
                "--output",
                str(bad_out),
                "--box",
                f"{bx},{by},{bw},{bh}",
            ],
            timeout=25.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert proc.returncode in (1, 2), (
            f"Expected non-zero exit code (1 or 2) for invalid --output={bad_out}, got {proc.returncode}"
        )
        combined = ((proc.stderr or "") + (proc.stdout or "")).strip()
        assert combined, f"Expected diagnostic message for invalid --output={bad_out}"
        assert_no_python_traceback(proc)


# ==============================================================================
# F12 — Benchmark & Telemetry Gate Boundaries (6 tests)
# ==============================================================================


@pytest.mark.f12
def test_f12_bound_impossible_min_fps_threshold_99999_exits_code_1_with_report(e2e_tmp_dir: Path) -> None:
    """F12-B1: `benchmark --preset quick --min-fps 99999.0` exits with code 1 (gate miss) AND writes valid report."""
    out_dir = e2e_tmp_dir / "bench_impossible"
    report_json = e2e_tmp_dir / "bench_impossible.json"

    proc = run_cli(
        [
            "benchmark",
            "--preset",
            "quick",
            "--min-fps",
            "99999.0",
            "--output-dir",
            str(out_dir),
            "--report",
            str(report_json),
            "--json",
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert proc.returncode == 1, (
        f"Expected benchmark threshold gate failure exit code 1 (NOT 0 or 2), got {proc.returncode}.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert_no_python_traceback(proc)

    assert proc.stdout.strip(), "Expected valid JSON report on stdout even when --min-fps gate fails"
    stdout_report = json.loads(proc.stdout.strip())
    assert_video_run_report_invariants(stdout_report, require_min_fps=None, expect_output_video=True)
    assert 0.0 < float(stdout_report["effective_fps"]) < 99999.0

    assert report_json.is_file() and report_json.stat().st_size > 0
    file_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert_video_run_report_invariants(file_report, require_min_fps=None, expect_output_video=True)
    assert float(file_report["effective_fps"]) < 99999.0


@pytest.mark.f12
def test_f12_bound_negative_or_zero_min_fps_exits_code_2(e2e_tmp_dir: Path) -> None:
    """F12-B2: Non-positive `--min-fps 0` and `--min-fps -5.0` are rejected with usage error exit code 2."""
    for bad_fps in ("0", "-5.0"):
        proc = run_cli(
            ["benchmark", "--preset", "quick", "--min-fps", bad_fps, "--output-dir", str(e2e_tmp_dir / f"b_{bad_fps}")],
            timeout=25.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)


@pytest.mark.f12
def test_f12_bound_unknown_benchmark_preset_exits_code_2(e2e_tmp_dir: Path) -> None:
    """F12-B3: Unknown benchmark preset (`--preset ultra_8k_nonexistent`, `--preset bogus`) exits with code 2."""
    for bad_preset in ("ultra_8k_nonexistent", "bogus"):
        proc = run_cli(
            ["benchmark", "--preset", bad_preset, "--output-dir", str(e2e_tmp_dir / f"p_{bad_preset}")],
            timeout=25.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)


@pytest.mark.f12
def test_f12_bound_corrupt_or_missing_benchmark_input_video_exits_code_2(e2e_tmp_dir: Path) -> None:
    """F12-B4: Missing or corrupt `--input` video in `benchmark` exits cleanly with code 2."""
    missing_mp4 = e2e_tmp_dir / "no_such_bench.mp4"
    corrupt_mp4 = e2e_tmp_dir / "corrupt_bench.mp4"
    corrupt_mp4.write_bytes(b"corrupt_not_an_mp4_container")

    for bad_in in (missing_mp4, corrupt_mp4):
        proc = run_cli(
            ["benchmark", "--input", str(bad_in), "--output-dir", str(e2e_tmp_dir / f"out_{bad_in.stem}")],
            timeout=25.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)


@pytest.mark.f12
def test_f12_bound_non_finite_nan_min_fps_rejected_with_exit_2(e2e_tmp_dir: Path) -> None:
    """F12-B5: Non-finite `--min-fps nan` / `inf` / `-inf` is rejected with exit code 2 (never silently passing)."""
    for non_finite in ("nan", "inf", "-inf"):
        proc = run_cli(
            [
                "benchmark",
                "--preset",
                "quick",
                "--min-fps",
                non_finite,
                "--output-dir",
                str(e2e_tmp_dir / f"nf_{non_finite}"),
            ],
            timeout=25.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)


@pytest.mark.f12
def test_f12_bound_2_frame_clip_benchmark_computes_finite_percentiles(e2e_tmp_dir: Path) -> None:
    """F12-B6: Benchmarking an ultra-short 2-frame MP4 computes finite latency percentiles and steady_state_fps > 0."""
    two_frame_mp4 = e2e_tmp_dir / "two_frames.mp4"
    make_textured_tracking_clip(two_frame_mp4, width=640, height=480, frames=2, fps=30, tmp_dir=e2e_tmp_dir)
    out_dir = e2e_tmp_dir / "bench_2f"

    _, report = run_cli_json(
        [
            "benchmark",
            "--input",
            str(two_frame_mp4),
            "--min-fps",
            "1.0",
            "--output-dir",
            str(out_dir),
        ],
        timeout=45.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report, require_min_fps=1.0, expect_output_video=True)
    assert (
        report["input_frame_count"]
        == report["processed_frame_count"]
        == report["rendered_frame_count"]
        == report["output_frame_count"]
        == 2
    )
    assert float(report["steady_state_fps"]) > 0.0
    out_mp4 = Path(report["output_path"])
    assert probe_video(out_mp4, tmp_dir=e2e_tmp_dir)["frame_count"] == 2
