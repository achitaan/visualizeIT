"""Tier 2 Boundary, Corner-Case & Negative Error-Handling E2E Tests — Slice B: F5–F8 (24 tests).

Strictly opaque-box:
- Zero imports of `visualizeit.*` anywhere.
- Zero top-level third-party imports (besides `pytest`); `numpy` and `cv2` are imported lazily
  inside functions so `pytest --collect-only` completes in <1s.
- All invalid CLI inputs assert exit code 2, non-empty diagnostic message, and zero Python stack traces.
- All runtime temporary files reside on XFS under `.cache/tmp` (never `/tmp`).
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.e2e._helpers import (
    CODEC_ARGS,
    StudioAPIClient,
    assert_cli_usage_error,
    assert_no_python_traceback,
    assert_video_run_report_invariants,
    assert_xfs_temp_path,
    build_xfs_env,
    compute_roi_and_bg_delta,
    decode_video_frames,
    make_ascii_ply,
    make_solid_clip,
    make_textured_tracking_clip,
    probe_video,
    run_cli,
    run_cli_json,
    run_subproc,
)

pytestmark = pytest.mark.tier2

EXPECTED_DEMO_CLIP_IDS: tuple[str, ...] = (
    "planar_card_640x480",
    "planar_hd_1280x720",
    "cylinder_mug_640x480",
    "occlusion_recovery_640x480",
    "mesh_box_640x480",
)


# ---------------------------------------------------------------------------
# Local opaque-box boundary helpers (stdlib at module load)
# ---------------------------------------------------------------------------


def _parse_bbox_xywh(raw_bbox: Any) -> tuple[float, float, float, float]:
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        return (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
    if isinstance(raw_bbox, dict):
        w_val = raw_bbox.get("w", raw_bbox.get("width"))
        h_val = raw_bbox.get("h", raw_bbox.get("height"))
        assert w_val is not None and h_val is not None, f"Invalid bbox dict: {raw_bbox!r}"
        return (float(raw_bbox["x"]), float(raw_bbox["y"]), float(w_val), float(h_val))
    raise AssertionError(f"Unsupported bbox structure: {raw_bbox!r}")


def _extract_intrinsics_from_meta(meta: dict[str, Any]) -> tuple[float, float, float, float]:
    if all(k in meta for k in ("fx", "fy", "cx", "cy")):
        return (float(meta["fx"]), float(meta["fy"]), float(meta["cx"]), float(meta["cy"]))
    for key in ("intrinsics", "camera_intrinsics", "camera", "calibration", "K", "cam_K", "cam_K_3x3"):
        sub = meta.get(key)
        if isinstance(sub, dict):
            if all(k in sub for k in ("fx", "fy", "cx", "cy")):
                return (float(sub["fx"]), float(sub["fy"]), float(sub["cx"]), float(sub["cy"]))
            for k_mat in ("K", "cam_K", "cam_K_3x3", "matrix"):
                if k_mat in sub:
                    sub = sub[k_mat]
                    break
        if isinstance(sub, (list, tuple)):
            if len(sub) == 4 and all(isinstance(x, (int, float)) for x in sub):
                return (float(sub[0]), float(sub[1]), float(sub[2]), float(sub[3]))
            if len(sub) == 3 and all(isinstance(r, (list, tuple)) and len(r) == 3 for r in sub):
                return (float(sub[0][0]), float(sub[1][1]), float(sub[0][2]), float(sub[1][2]))
            if len(sub) == 9 and all(isinstance(x, (int, float)) for x in sub):
                return (float(sub[0]), float(sub[4]), float(sub[2]), float(sub[5]))
    raise AssertionError(f"Unable to extract intrinsics from {sorted(meta.keys())}")


def _selection_cli_args_from_manifest(clip_meta: dict[str, Any]) -> list[str]:
    sel = clip_meta.get("default_selection") or {}
    args: list[str] = []
    pos = sel.get("positive_points") or []
    if pos:
        p0 = pos[0]
        px = int(round(float(p0["x"] if isinstance(p0, dict) else p0[0])))
        py = int(round(float(p0["y"] if isinstance(p0, dict) else p0[1])))
        args.extend(["--point", f"{px},{py}"])
    if sel.get("bbox") is not None:
        bx, by, bw, bh = _parse_bbox_xywh(sel["bbox"])
        args.extend(["--box", f"{int(round(bx))},{int(round(by))},{int(round(bw))},{int(round(bh))}"])
    return args


def _make_jump_cut_clip(out_path: Path, *, tmp_dir: Path) -> Path:
    """Synthesize a 20-frame 640x480 clip where a 200x150 textured card teleports (+220, +140) px at frame 10."""
    target = assert_xfs_temp_path(out_path)
    bg = "color=c=0x1e293b:s=640x480:r=30,noise=all_seed=11:alls=6,trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/(30*TB)"
    fg = (
        "testsrc2=s=200x150:r=30,noise=all_seed=42:alls=25,drawgrid=w=25:h=25:t=2:c=white@0.85,"
        "trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/(30*TB)"
    )
    filt = "[0:v][1:v]overlay=x='if(lt(n,10),80,300)':y='if(lt(n,10),70,210)':eval=frame,format=yuv420p"
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
        "20",
        *CODEC_ARGS[".mp4"],
        str(target),
    ]
    run_subproc(cmd, timeout=25.0, check=True, tmp_dir=tmp_dir)
    return target


def _make_right_border_slide_clip(out_path: Path, *, tmp_dir: Path) -> Path:
    """Synthesize a 24-frame 640x480 clip where >60% of a 220x160 card slides past x=639."""
    target = assert_xfs_temp_path(out_path)
    bg = "color=c=0x1e293b:s=640x480:r=30,noise=all_seed=17:alls=6,trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/(30*TB)"
    fg = (
        "testsrc2=s=220x160:r=30,noise=all_seed=42:alls=25,drawgrid=w=25:h=25:t=2:c=white@0.85,"
        "trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/(30*TB)"
    )
    # At n=0: x=370..590 (100% inside). At n=23: x=577..797 (>70% past x=639).
    filt = "[0:v][1:v]overlay=x='370+9*n':y=150:eval=frame,format=yuv420p"
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
        "24",
        *CODEC_ARGS[".mp4"],
        str(target),
    ]
    run_subproc(cmd, timeout=25.0, check=True, tmp_dir=tmp_dir)
    return target


def _make_rapid_zoom_clip(out_path: Path, *, tmp_dir: Path) -> tuple[Path, tuple[int, int, int, int]]:
    """Synthesize a 30-frame 640x480 clip scaling a textured planar target across 0.4x .. 2.2x."""
    import cv2
    import numpy as np

    target = assert_xfs_temp_path(out_path)
    w, h, n_frames = 640, 480, 30
    rng = np.random.default_rng(1234)

    # High-feature planar poster (base 200x150 at 1.0x)
    base_w, base_h = 200, 150
    card = rng.integers(35, 225, size=(base_h, base_w, 3), dtype=np.uint8)
    for gx in range(0, base_w, 20):
        cv2.line(card, (gx, 0), (gx, base_h - 1), (245, 245, 250), 2)
    for gy in range(0, base_h, 20):
        cv2.line(card, (0, gy), (base_w - 1, gy), (25, 30, 40), 2)
    cv2.rectangle(card, (2, 2), (base_w - 3, base_h - 3), (255, 210, 60), 4)
    cv2.circle(card, (base_w // 2, base_h // 2), 34, (40, 190, 245), -1)

    bg_base = np.full((h, w, 3), (32, 38, 48), dtype=np.uint8)

    # Frame 0 starts at 1.0x so onboarding box is crisp; zooms down to 0.4x by frame 10 and up to 2.2x by frame 29
    scales: list[float] = []
    for idx in range(n_frames):
        if idx <= 10:
            s = 1.0 - (1.0 - 0.40) * (idx / 10.0)
        else:
            s = 0.40 + (2.20 - 0.40) * ((idx - 10) / 19.0)
        scales.append(s)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        "30",
        "-i",
        "pipe:0",
        *CODEC_ARGS[".mp4"],
        str(target),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=build_xfs_env(tmp_dir=tmp_dir),
    )
    assert proc.stdin is not None
    try:
        for s in scales:
            frame = bg_base.copy()
            sw = max(16, int(round(base_w * s)))
            sh = max(12, int(round(base_h * s)))
            resized = cv2.resize(card, (sw, sh), interpolation=cv2.INTER_LINEAR)
            x0 = (w - sw) // 2
            y0 = (h - sh) // 2
            dx0, dy0 = max(0, x0), max(0, y0)
            dx1, dy1 = min(w, x0 + sw), min(h, y0 + sh)
            sx0, sy0 = dx0 - x0, dy0 - y0
            frame[dy0:dy1, dx0:dx1] = resized[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)]
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        stderr_out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        rc = proc.wait(timeout=20.0)
        assert rc == 0, f"ffmpeg failed encoding rapid zoom clip: {stderr_out}"
    finally:
        if proc.stderr is not None:
            proc.stderr.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)

    bbox0 = ((w - base_w) // 2, (h - base_h) // 2, base_w, base_h)
    return target, bbox0


# ===========================================================================
# F5 — Demo Corpus Boundaries (6 tests)
# ===========================================================================


@pytest.mark.f05
def test_f05_bound_demo_assets_idempotent_rerun_on_existing_directory(e2e_tmp_dir: Path) -> None:
    out_dir = e2e_tmp_dir / "existing_corpus"
    proc1, man1 = run_cli_json(["demo-assets", "--output-dir", str(out_dir)], timeout=90.0, tmp_dir=e2e_tmp_dir)
    proc2, man2 = run_cli_json(["demo-assets", "--output-dir", str(out_dir)], timeout=90.0, tmp_dir=e2e_tmp_dir)

    assert_no_python_traceback(proc1)
    assert_no_python_traceback(proc2)
    ids1 = [c["id"] for c in man1["clips"]]
    ids2 = [c["id"] for c in man2["clips"]]
    assert set(ids1) == set(ids2) == set(EXPECTED_DEMO_CLIP_IDS)


@pytest.mark.f05
def test_f05_bound_demo_assets_creates_deeply_nested_missing_parent_dirs(e2e_tmp_dir: Path) -> None:
    nested_dir = e2e_tmp_dir / "a" / "b" / "c" / "demos"
    assert not nested_dir.exists()

    proc, manifest = run_cli_json(
        ["demo-assets", "--output-dir", str(nested_dir)],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert nested_dir.is_dir()
    assert len(manifest["clips"]) == 5
    for cid in EXPECTED_DEMO_CLIP_IDS:
        assert (nested_dir / f"{cid}.mp4").is_file()
    assert (nested_dir / "mesh_box.ply").is_file()
    assert (nested_dir / "mesh_box_meta.json").is_file()


@pytest.mark.f05
def test_f05_bound_demo_assets_output_dir_pointing_to_regular_file_exits_2(e2e_tmp_dir: Path) -> None:
    regular_file = e2e_tmp_dir / "not_a_dir"
    regular_file.write_text("regular file blocking mkdir", encoding="utf-8")

    proc = run_cli(
        ["demo-assets", "--output-dir", str(regular_file), "--json"],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_cli_usage_error(proc)


@pytest.mark.f05
def test_f05_bound_demo_default_selections_lie_strictly_inside_frame_bounds(
    demo_corpus_manifest: dict[str, dict[str, Any]],
) -> None:
    assert set(demo_corpus_manifest.keys()) == set(EXPECTED_DEMO_CLIP_IDS)
    for cid, clip in demo_corpus_manifest.items():
        w = float(clip["width"])
        h = float(clip["height"])
        fc = int(clip["frame_count"])
        sel = clip["default_selection"]
        assert isinstance(sel, dict), f"{cid}: expected dict default_selection"

        f_idx = int(sel.get("frame_index", 0))
        assert 0 <= f_idx < fc, f"{cid}: default_selection.frame_index={f_idx} outside [0, {fc})"

        for pt_list_key in ("positive_points", "negative_points"):
            for pt in sel.get(pt_list_key) or []:
                px = float(pt["x"] if isinstance(pt, dict) else pt[0])
                py = float(pt["y"] if isinstance(pt, dict) else pt[1])
                assert math.isfinite(px) and math.isfinite(py)
                assert 0.0 <= px < w and 0.0 <= py < h, (
                    f"{cid}: {pt_list_key} point {(px, py)} outside frame ({w}x{h})"
                )

        assert sel.get("bbox") is not None, f"{cid}: expected default_selection.bbox"
        bx, by, bw, bh = _parse_bbox_xywh(sel["bbox"])
        assert bw > 0.0 and bh > 0.0
        assert 0.0 <= bx < w and 0.0 <= by < h, f"{cid}: bbox origin {(bx, by)} outside frame ({w}x{h})"
        assert bx + bw <= w and by + bh <= h, (
            f"{cid}: bbox {(bx, by, bw, bh)} exceeds frame dimensions ({w}x{h})"
        )


@pytest.mark.f05
def test_f05_bound_tiny_resolution_160x120_and_odd_dimension_mp4_probed_and_processed(
    e2e_tmp_dir: Path,
) -> None:
    # H.264 yuv420p requires 2-aligned dimensions: test tiny 160x120 and non-16-macroblock-aligned 322x242
    for w, h in ((160, 120), (322, 242)):
        src_mp4 = e2e_tmp_dir / f"clip_{w}x{h}.mp4"
        out_mp4 = e2e_tmp_dir / f"out_{w}x{h}.mp4"
        meta = make_textured_tracking_clip(src_mp4, width=w, height=h, frames=12, tmp_dir=e2e_tmp_dir)
        bx, by, bw, bh = meta["bbox_frame0"]

        _, info_json = run_cli_json(["info", "--input", str(src_mp4)], timeout=30.0, tmp_dir=e2e_tmp_dir)
        assert (int(info_json["width"]), int(info_json["height"])) == (w, h)
        assert int(info_json["frame_count"]) == 12

        proc, report = run_cli_json(
            [
                "process",
                "--input",
                str(src_mp4),
                "--output",
                str(out_mp4),
                "--box",
                f"{bx},{by},{bw},{bh}",
            ],
            timeout=60.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_no_python_traceback(proc)
        assert_video_run_report_invariants(report)
        assert (report["width"], report["height"]) == (w, h)
        assert report["output_frame_count"] == 12

        probed_out = probe_video(out_mp4, tmp_dir=e2e_tmp_dir)
        assert (probed_out["width"], probed_out["height"]) == (w, h)
        assert probed_out["frame_count"] == 12


@pytest.mark.f05
def test_f05_bound_ultra_short_2_frame_video_clip_processed_completely(e2e_tmp_dir: Path) -> None:
    src_mp4 = e2e_tmp_dir / "two_frame.mp4"
    out_mp4 = e2e_tmp_dir / "two_frame_out.mp4"
    meta = make_textured_tracking_clip(src_mp4, width=640, height=480, frames=2, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = meta["bbox_frame0"]

    _, info_json = run_cli_json(["info", "--input", str(src_mp4)], timeout=30.0, tmp_dir=e2e_tmp_dir)
    assert int(info_json["frame_count"]) == 2

    proc, report = run_cli_json(
        [
            "process",
            "--input",
            str(src_mp4),
            "--output",
            str(out_mp4),
            "--box",
            f"{bx},{by},{bw},{bh}",
        ],
        timeout=60.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)
    assert (
        report["input_frame_count"]
        == report["processed_frame_count"]
        == report["rendered_frame_count"]
        == report["output_frame_count"]
        == 2
    )
    assert report["render_stride"] == 1
    assert probe_video(out_mp4, tmp_dir=e2e_tmp_dir)["frame_count"] == 2


# ===========================================================================
# F6 — Segmentation Onboarding Boundaries (6 tests)
# ===========================================================================


@pytest.mark.f06
def test_f06_bound_negative_and_out_of_range_frame_index_exits_2_and_400(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    clip_path = e2e_tmp_dir / "seg_bounds_frame.mp4"
    make_textured_tracking_clip(clip_path, width=640, height=480, frames=10, tmp_dir=e2e_tmp_dir)

    for bad_frame in ("-1", "999999"):
        proc = run_cli(
            ["segment", "--input", str(clip_path), f"--frame={bad_frame}", "--point", "320,240", "--json"],
            timeout=30.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)

    up = live_studio_server.upload(
        "/api/videos/upload",
        filename="seg_bounds_frame.mp4",
        content=clip_path.read_bytes(),
        content_type="video/mp4",
        expected_status=200,
    )
    video_id = up["json"]["video_id"]

    for bad_idx in (-1, 999999):
        resp = live_studio_server.request(
            "POST",
            f"/api/videos/{video_id}/select",
            json_body={"frame_index": bad_idx, "positive_points": [[320, 240]], "negative_points": [], "bbox": None},
        )
        assert resp["status"] in (400, 404, 422), (
            f"Expected 400/404/422 for frame_index={bad_idx}, got {resp['status']}: {resp['body']!r}"
        )


@pytest.mark.f06
def test_f06_bound_out_of_bounds_and_negative_point_coordinates_rejected(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    clip_path = e2e_tmp_dir / "seg_bounds_pt.mp4"
    make_textured_tracking_clip(clip_path, width=640, height=480, frames=10, tmp_dir=e2e_tmp_dir)

    for bad_pt in ("-5,10", "6400,4800"):
        proc = run_cli(
            ["segment", "--input", str(clip_path), f"--point={bad_pt}", "--json"],
            timeout=30.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)

    up = live_studio_server.upload(
        "/api/videos/upload",
        filename="seg_bounds_pt.mp4",
        content=clip_path.read_bytes(),
        content_type="video/mp4",
        expected_status=200,
    )
    video_id = up["json"]["video_id"]

    for bad_xy in ([-10, 20], [6400, 4800]):
        resp = live_studio_server.request(
            "POST",
            f"/api/videos/{video_id}/select",
            json_body={"frame_index": 0, "positive_points": [bad_xy], "negative_points": [], "bbox": None},
        )
        assert resp["status"] in (400, 422), (
            f"Expected 400/422 for out-of-bounds positive_points={bad_xy}, got {resp['status']}: {resp['body']!r}"
        )


@pytest.mark.f06
def test_f06_bound_zero_or_negative_width_height_bbox_rejected(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    clip_path = e2e_tmp_dir / "seg_bounds_box.mp4"
    make_textured_tracking_clip(clip_path, width=640, height=480, frames=10, tmp_dir=e2e_tmp_dir)

    for bad_box in ("100,100,0,50", "100,100,-20,50"):
        proc = run_cli(
            ["segment", "--input", str(clip_path), f"--box={bad_box}", "--json"],
            timeout=30.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)

    up = live_studio_server.upload(
        "/api/videos/upload",
        filename="seg_bounds_box.mp4",
        content=clip_path.read_bytes(),
        content_type="video/mp4",
        expected_status=200,
    )
    video_id = up["json"]["video_id"]

    for bad_bbox in ([10, 10, 0, 0], [10, 10, -20, 50]):
        resp = live_studio_server.request(
            "POST",
            f"/api/videos/{video_id}/select",
            json_body={"frame_index": 0, "positive_points": [], "negative_points": [], "bbox": bad_bbox},
        )
        assert resp["status"] in (400, 422), (
            f"Expected 400/422 for degenerate bbox={bad_bbox}, got {resp['status']}: {resp['body']!r}"
        )


@pytest.mark.f06
def test_f06_bound_edge_of_frame_corner_pixels_0_0_and_w_minus_1_h_minus_1(
    e2e_tmp_dir: Path,
) -> None:
    clip_path = e2e_tmp_dir / "seg_corners.mp4"
    make_textured_tracking_clip(clip_path, width=640, height=480, frames=8, tmp_dir=e2e_tmp_dir)

    cases = (
        ["--point", "0,0"],
        ["--point", "639,479"],
        ["--box", "0,0,640,480"],
    )
    for flag_args in cases:
        proc = run_cli(
            ["segment", "--input", str(clip_path), *flag_args, "--json"],
            timeout=30.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_no_python_traceback(proc)
        assert proc.returncode in (0, 2), (
            f"Expected clean exit 0 or usage exit 2 for {flag_args}, got rc={proc.returncode}:\n{proc.stderr}"
        )
        if proc.returncode == 0:
            parsed = json.loads(proc.stdout.strip())
            assert (int(parsed["width"]), int(parsed["height"])) == (640, 480)
            cov = float(parsed["coverage_ratio"])
            assert math.isfinite(cov) and 0.0 <= cov <= 1.0
        else:
            assert_cli_usage_error(proc)


@pytest.mark.f06
def test_f06_bound_malformed_coordinate_syntax_non_numeric_exits_2(e2e_tmp_dir: Path) -> None:
    clip_path = e2e_tmp_dir / "seg_syntax.mp4"
    make_textured_tracking_clip(clip_path, width=640, height=480, frames=6, tmp_dir=e2e_tmp_dir)

    proc_pt = run_cli(
        ["segment", "--input", str(clip_path), "--point", "abc,def", "--json"],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_cli_usage_error(proc_pt)

    proc_box = run_cli(
        ["segment", "--input", str(clip_path), "--box", "10,20,30", "--json"],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_cli_usage_error(proc_box)


@pytest.mark.f06
def test_f06_bound_featureless_uniform_gray_video_segmentation_does_not_crash(
    e2e_tmp_dir: Path,
) -> None:
    gray_mp4 = make_solid_clip(
        e2e_tmp_dir / "uniform_gray.mp4",
        width=640,
        height=480,
        frames=10,
        color="0x808080",
        tmp_dir=e2e_tmp_dir,
    )
    proc = run_cli(
        ["segment", "--input", str(gray_mp4), "--box", "100,100,200,200", "--json"],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert proc.returncode in (0, 2), (
        f"Expected graceful exit 0 or diagnostic exit 2 on uniform gray clip, got {proc.returncode}:\n{proc.stderr}"
    )
    if proc.returncode == 0:
        data = json.loads(proc.stdout.strip())
        assert (int(data["width"]), int(data["height"])) == (640, 480)
        assert math.isfinite(float(data["coverage_ratio"]))
        assert isinstance(data["anchor_count"], int) and data["anchor_count"] >= 0
    else:
        assert_cli_usage_error(proc)


# ===========================================================================
# F7 — KLT + ORB Video Tracker Boundaries (6 tests)
# ===========================================================================


@pytest.mark.f07
def test_f07_bound_completely_black_zero_texture_video_reports_lost_cleanly(
    e2e_tmp_dir: Path,
) -> None:
    black_mp4 = make_solid_clip(
        e2e_tmp_dir / "solid_black.mp4",
        width=640,
        height=480,
        frames=15,
        color="0x000000",
        tmp_dir=e2e_tmp_dir,
    )
    out_mp4 = e2e_tmp_dir / "solid_black_out.mp4"

    proc = run_cli(
        ["process", "--input", str(black_mp4), "--output", str(out_mp4), "--box", "100,100,200,200", "--json"],
        timeout=60.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert proc.returncode in (0, 2), (
        f"Expected exit 0 (suppressed lost frames) or exit 2 (zero features), got {proc.returncode}:\n{proc.stderr}"
    )
    if proc.returncode == 0:
        report = json.loads(proc.stdout.strip())
        assert_video_run_report_invariants(report)
        assert int(report["tracked_frame_count"]) <= 1, (
            f"Pitch-black zero-texture video must not claim multi-frame tracking lock: {report}"
        )
    else:
        assert_cli_usage_error(proc)


@pytest.mark.f07
def test_f07_bound_single_frame_1_frame_video_clip_handled_without_div_zero(
    e2e_tmp_dir: Path,
) -> None:
    src_mp4 = e2e_tmp_dir / "single_frame.mp4"
    out_mp4 = e2e_tmp_dir / "single_frame_out.mp4"
    meta = make_textured_tracking_clip(src_mp4, width=640, height=480, frames=1, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = meta["bbox_frame0"]

    proc, report = run_cli_json(
        ["process", "--input", str(src_mp4), "--output", str(out_mp4), "--box", f"{bx},{by},{bw},{bh}"],
        timeout=60.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)
    assert (
        report["input_frame_count"]
        == report["processed_frame_count"]
        == report["rendered_frame_count"]
        == report["output_frame_count"]
        == 1
    )
    assert probe_video(out_mp4, tmp_dir=e2e_tmp_dir)["frame_count"] == 1


@pytest.mark.f07
def test_f07_bound_abrupt_180px_jump_cut_does_not_corrupt_homography(e2e_tmp_dir: Path) -> None:
    import numpy as np

    jump_mp4 = _make_jump_cut_clip(e2e_tmp_dir / "jump_cut.mp4", tmp_dir=e2e_tmp_dir)
    out_mp4 = e2e_tmp_dir / "jump_cut_out.mp4"

    proc, report = run_cli_json(
        [
            "process",
            "--input",
            str(jump_mp4),
            "--output",
            str(out_mp4),
            "--box",
            "80,70,200,150",
            "--wrap-mode",
            "planar",
        ],
        timeout=60.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)
    assert report["output_frame_count"] == 20

    raw_frames = decode_video_frames(jump_mp4)
    out_frames = decode_video_frames(out_mp4)
    assert len(out_frames) == 20
    for idx, rgb in out_frames.items():
        assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
        # Bottom-left background patch (0..60, 410..480) stays far from both jump positions: must not explode
        bg_diff = float(np.mean(np.abs(rgb[410:475, 5:60].astype(np.float32) - raw_frames[idx][410:475, 5:60].astype(np.float32))))
        assert bg_diff <= 5.0, f"Singular homography corrupted distant background at frame {idx} (diff={bg_diff:.2f})"


@pytest.mark.f07
def test_f07_bound_object_partially_sliding_across_right_frame_border(e2e_tmp_dir: Path) -> None:
    slide_mp4 = _make_right_border_slide_clip(e2e_tmp_dir / "border_slide.mp4", tmp_dir=e2e_tmp_dir)
    out_mp4 = e2e_tmp_dir / "border_slide_out.mp4"

    proc, report = run_cli_json(
        [
            "process",
            "--input",
            str(slide_mp4),
            "--output",
            str(out_mp4),
            "--box",
            "370,150,220,160",
            "--wrap-mode",
            "planar",
        ],
        timeout=60.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)

    probed = probe_video(out_mp4, tmp_dir=e2e_tmp_dir)
    assert (probed["width"], probed["height"]) == (640, 480)
    assert probed["frame_count"] == 24


@pytest.mark.f07
def test_f07_bound_invalid_tracking_height_and_nonpositive_threads_exit_2(
    e2e_tmp_dir: Path,
) -> None:
    src_mp4 = e2e_tmp_dir / "flags_src.mp4"
    out_mp4 = e2e_tmp_dir / "flags_out.mp4"
    make_textured_tracking_clip(src_mp4, width=640, height=480, frames=6, tmp_dir=e2e_tmp_dir)

    for bad_args in (
        ["--tracking-height", "999"],
        ["--threads", "0"],
        ["--threads=-2"],
    ):
        proc = run_cli(
            ["process", "--input", str(src_mp4), "--output", str(out_mp4), *bad_args, "--json"],
            timeout=30.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)


@pytest.mark.f07
def test_f07_bound_rapid_zoom_in_and_zoom_out_scale_extremes(e2e_tmp_dir: Path) -> None:
    import numpy as np

    zoom_mp4, (bx, by, bw, bh) = _make_rapid_zoom_clip(e2e_tmp_dir / "rapid_zoom.mp4", tmp_dir=e2e_tmp_dir)
    out_mp4 = e2e_tmp_dir / "rapid_zoom_out.mp4"

    proc, report = run_cli_json(
        [
            "process",
            "--input",
            str(zoom_mp4),
            "--output",
            str(out_mp4),
            "--wrap-mode",
            "planar",
            "--box",
            f"{bx},{by},{bw},{bh}",
        ],
        timeout=60.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)
    assert report["output_frame_count"] == 30
    assert int(report["tracked_frame_count"]) >= 1

    raw_frames = decode_video_frames(zoom_mp4, [0, 10, 29])
    out_frames = decode_video_frames(out_mp4, [0, 10, 29])
    for idx in (0, 10):
        # At 1.0x and 0.40x scale, top-left 40x40 corner is pure background and must not invert across frame
        corner_diff = float(
            np.mean(np.abs(out_frames[idx][:40, :40].astype(np.float32) - raw_frames[idx][:40, :40].astype(np.float32)))
        )
        assert corner_diff <= 5.0, f"Inverted quad artifact in background corner at frame {idx} (diff={corner_diff:.2f})"


# ===========================================================================
# F8 — Calibrated 6-DoF 3D Mesh Tracker Boundaries (6 tests)
# ===========================================================================


@pytest.mark.f08
def test_f08_bound_nonexistent_or_directory_ply_path_exits_2(e2e_tmp_dir: Path) -> None:
    src_mp4 = e2e_tmp_dir / "m6_src.mp4"
    out_mp4 = e2e_tmp_dir / "m6_out.mp4"
    make_textured_tracking_clip(src_mp4, width=640, height=480, frames=6, tmp_dir=e2e_tmp_dir)

    missing_ply = e2e_tmp_dir / "no_such_model.ply"
    ply_dir = e2e_tmp_dir / "directory_not_ply"
    ply_dir.mkdir(parents=True, exist_ok=True)

    for bad_mesh in (missing_ply, ply_dir):
        proc = run_cli(
            [
                "process",
                "--input",
                str(src_mp4),
                "--output",
                str(out_mp4),
                "--wrap-mode",
                "mesh_6dof",
                "--mesh",
                str(bad_mesh),
                "--intrinsics",
                "550,550,320,240",
                "--json",
            ],
            timeout=30.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)


@pytest.mark.f08
def test_f08_bound_corrupt_ply_bad_header_and_out_of_range_face_indices_exit_2(
    e2e_tmp_dir: Path,
) -> None:
    src_mp4 = e2e_tmp_dir / "corrupt_ply_src.mp4"
    out_mp4 = e2e_tmp_dir / "corrupt_ply_out.mp4"
    make_textured_tracking_clip(src_mp4, width=640, height=480, frames=6, tmp_dir=e2e_tmp_dir)

    bad_header_ply = make_ascii_ply(e2e_tmp_dir / "bad_header.ply", defect="bad_header")
    oob_index_ply = make_ascii_ply(e2e_tmp_dir / "oob_index.ply", defect="oob_index")

    for bad_ply in (bad_header_ply, oob_index_ply):
        proc = run_cli(
            [
                "process",
                "--input",
                str(src_mp4),
                "--output",
                str(out_mp4),
                "--wrap-mode",
                "mesh_6dof",
                "--mesh",
                str(bad_ply),
                "--intrinsics",
                "550,550,320,240",
                "--json",
            ],
            timeout=30.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)


@pytest.mark.f08
def test_f08_bound_ply_with_nan_or_inf_vertex_coordinates_exits_2(e2e_tmp_dir: Path) -> None:
    src_mp4 = e2e_tmp_dir / "nan_ply_src.mp4"
    out_mp4 = e2e_tmp_dir / "nan_ply_out.mp4"
    make_textured_tracking_clip(src_mp4, width=640, height=480, frames=6, tmp_dir=e2e_tmp_dir)

    nan_ply = make_ascii_ply(e2e_tmp_dir / "nan_vertex.ply", defect="nan_vertex")
    proc = run_cli(
        [
            "process",
            "--input",
            str(src_mp4),
            "--output",
            str(out_mp4),
            "--wrap-mode",
            "mesh_6dof",
            "--mesh",
            str(nan_ply),
            "--intrinsics",
            "550,550,320,240",
            "--json",
        ],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_cli_usage_error(proc)


@pytest.mark.f08
def test_f08_bound_invalid_intrinsics_zero_or_negative_focal_length_exits_2(
    e2e_tmp_dir: Path,
) -> None:
    src_mp4 = e2e_tmp_dir / "intr_src.mp4"
    out_mp4 = e2e_tmp_dir / "intr_out.mp4"
    make_textured_tracking_clip(src_mp4, width=640, height=480, frames=6, tmp_dir=e2e_tmp_dir)

    for bad_intr in ("0,500,320,240", "-500,500,320,240", "500,500,320"):
        proc = run_cli(
            [
                "process",
                "--input",
                str(src_mp4),
                "--output",
                str(out_mp4),
                "--wrap-mode",
                "mesh_6dof",
                "--mesh",
                "plane",
                f"--intrinsics={bad_intr}",
                "--json",
            ],
            timeout=30.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)


@pytest.mark.f08
def test_f08_bound_unknown_parametric_mesh_name_exits_2(e2e_tmp_dir: Path) -> None:
    src_mp4 = e2e_tmp_dir / "unk_mesh_src.mp4"
    out_mp4 = e2e_tmp_dir / "unk_mesh_out.mp4"
    make_textured_tracking_clip(src_mp4, width=640, height=480, frames=6, tmp_dir=e2e_tmp_dir)

    proc = run_cli(
        [
            "process",
            "--input",
            str(src_mp4),
            "--output",
            str(out_mp4),
            "--wrap-mode",
            "mesh_6dof",
            "--mesh",
            "dodecahedron_xyz",
            "--intrinsics",
            "550,550,320,240",
            "--json",
        ],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_cli_usage_error(proc)


@pytest.mark.f08
def test_f08_bound_mesh_6dof_on_occluded_blackout_frames_suppresses_pose(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    import numpy as np

    clip = demo_corpus_manifest["mesh_box_640x480"]
    ply_path = demo_corpus_dir / "mesh_box.ply"
    meta = json.loads((demo_corpus_dir / "mesh_box_meta.json").read_text(encoding="utf-8"))
    fx, fy, cx, cy = _extract_intrinsics_from_meta(meta)

    blackout_mp4 = e2e_tmp_dir / "mesh_box_blackout.mp4"
    run_subproc(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(clip["path"]),
            "-vf",
            "drawbox=x=0:y=0:w=640:h=480:c=black:t=fill:enable='between(n,14,23)'",
            "-frames:v",
            "45",
            *CODEC_ARGS[".mp4"],
            str(blackout_mp4),
        ],
        timeout=30.0,
        check=True,
        tmp_dir=e2e_tmp_dir,
    )

    out_mp4 = e2e_tmp_dir / "mesh_box_blackout_out.mp4"
    proc, report = run_cli_json(
        [
            "process",
            "--input",
            str(blackout_mp4),
            "--output",
            str(out_mp4),
            "--wrap-mode",
            "mesh_6dof",
            "--mesh",
            str(ply_path),
            "--intrinsics",
            f"{fx:g},{fy:g},{cx:g},{cy:g}",
            "--pattern",
            "solid",
            "--colors",
            "#ff006e,#00f5d4",
            "--blend-mode",
            "normal",
            "--opacity",
            "0.95",
            *_selection_cli_args_from_manifest(clip),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)
    assert int(report["lost_frame_count"]) >= 5, (
        f"Expected >= 5 lost frames during 10-frame blackout in mesh_6dof, got {report['lost_frame_count']}"
    )
    assert int(report["recovered_count"]) >= 1, (
        f"Expected mesh_6dof re-acquisition after blackout, got recovered_count={report['recovered_count']}"
    )

    sample_idxs = [16, 19, 21, 32, 38]
    raw_frames = decode_video_frames(blackout_mp4, sample_idxs)
    out_frames = decode_video_frames(out_mp4, sample_idxs)

    for occ_idx in (16, 19, 21):
        mad = float(np.mean(np.abs(out_frames[occ_idx].astype(np.float32) - raw_frames[occ_idx].astype(np.float32))))
        assert mad <= 2.5, (
            f"Stale 6-DoF pose overlay was NOT suppressed on blackout frame {occ_idx} (mean_abs_diff={mad:.2f})"
        )

    sel_bbox = _parse_bbox_xywh(clip["default_selection"]["bbox"])
    bbox_int = (int(round(sel_bbox[0])), int(round(sel_bbox[1])), int(round(sel_bbox[2])), int(round(sel_bbox[3])))
    post_deltas = [
        compute_roi_and_bg_delta(raw_frames[post_idx], out_frames[post_idx], bbox_int, margin=36)[0]
        for post_idx in (32, 38)
    ]
    assert max(post_deltas) >= 6.0, (
        f"Expected 6-DoF box overlay to recover after blackout, got ROI deltas {post_deltas}"
    )
