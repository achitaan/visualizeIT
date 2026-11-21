"""Tier 4 Real-World Video Pipeline Workloads & Honest >=15.0 FPS Gates (Scenarios 1, 2, 3, 4, 10).

Strictly opaque-box:
- Zero imports of `visualizeit.*` anywhere.
- Zero top-level third-party imports (besides `pytest`).
- Enforces `ORIGINAL_REQUEST.md` ("get it working at 15 fps at least for the video version") and
  `PROJECT.md` anti-cheating invariants (`render_stride == 1`, 4-way frame count equality, external
  `time.perf_counter()` wall-clock cross-check, genuine foreground RGB delta, preserved background,
  and strict occlusion ghost-overlay suppression).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import pytest

from tests.e2e._helpers import (
    assert_video_run_report_invariants,
    bbox_at_frame,
    compute_roi_and_bg_delta,
    decode_image_bytes,
    decode_video_frames,
    make_occlusion_clip,
    make_rgba_png,
    make_textured_tracking_clip,
    probe_video,
    ray_triangle_uv_oracle,
    run_cli_json,
    spawn_live_server,
)

pytestmark = pytest.mark.tier4


def _selection_to_segment_args(selection: dict[str, Any] | None) -> list[str]:
    if not selection:
        return ["--frame", "0"]
    args = ["--frame", str(int(selection.get("frame_index", 0)))]
    for pt in selection.get("positive_points") or []:
        args.extend(["--point", f"{float(pt[0]):g},{float(pt[1]):g}"])
    for pt in selection.get("negative_points") or []:
        args.extend(["--neg-point", f"{float(pt[0]):g},{float(pt[1]):g}"])
    bbox = selection.get("bbox")
    if bbox is not None and len(bbox) == 4:
        args.extend(["--box", ",".join(f"{float(v):g}" for v in bbox)])
    return args


def _selection_to_process_args(selection: dict[str, Any] | None) -> list[str]:
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


def _localize_textured_fg_bbox(
    raw_rgb: Any,
    fallback_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Opaque-box physical foreground bbox detector using high-frequency spatial gradients on raw frame."""
    import numpy as np

    arr = np.asarray(raw_rgb, dtype=np.float32)
    h, w = arr.shape[:2]
    gx = np.mean(np.abs(arr[:, 1:, :] - arr[:, :-1, :]), axis=-1)
    gy = np.mean(np.abs(arr[1:, :, :] - arr[:-1, :, :]), axis=-1)
    grad = gx[:-1, :] + gy[:, :-1]
    ys, xs = np.nonzero(grad > 20.0)
    if xs.size < 64:
        return fallback_bbox
    x1 = int(np.percentile(xs, 2))
    x2 = int(np.percentile(xs, 98))
    y1 = int(np.percentile(ys, 2))
    y2 = int(np.percentile(ys, 98))
    bw = max(24, min(w - x1, x2 - x1))
    bh = max(24, min(h - y1, y2 - y1))
    return (max(0, x1), max(0, y1), bw, bh)


def _assert_external_wall_clock_honesty(report: dict[str, Any], ext_wall_s: float) -> None:
    """Verify reported `wall_time_seconds` and `effective_fps` are honest against external `perf_counter()`."""
    rep_wall = float(report["wall_time_seconds"])
    rep_fps = float(report["effective_fps"])
    n_proc = int(report["processed_frame_count"])
    assert rep_wall > 0.05, f"Suspiciously small wall_time_seconds={rep_wall:.5f}s for {n_proc} frames"
    assert rep_wall <= ext_wall_s + 0.5, (
        f"Reported wall_time_seconds={rep_wall:.3f}s exceeds external subprocess time {ext_wall_s:.3f}s"
    )
    assert abs(rep_fps - (n_proc / rep_wall)) < 0.5, (
        f"effective_fps={rep_fps:.2f} does not match processed_frame_count/wall_time_seconds={n_proc / rep_wall:.2f}"
    )
    assert float(report["frame_latency_ms"]["mean"]) <= 66.67 + 1e-3, (
        f"Mean frame latency {report['frame_latency_ms']['mean']:.2f}ms exceeds 66.67ms (15 FPS budget)"
    )


def _parse_ascii_ply_mesh(ply_path: Path) -> tuple[Any, Any]:
    """Parse 3D vertex positions `(V, 3)` and triangle indices `(F, 3)` from an ASCII `.ply` file."""
    import numpy as np

    lines = ply_path.read_text(encoding="utf-8").splitlines()
    assert lines and lines[0].strip() == "ply", f"Invalid PLY header in {ply_path}"
    num_verts = 0
    num_faces = 0
    header_end = 0
    for idx, line in enumerate(lines):
        s = line.strip()
        if s.startswith("element vertex "):
            num_verts = int(s.split()[-1])
        elif s.startswith("element face "):
            num_faces = int(s.split()[-1])
        elif s == "end_header":
            header_end = idx + 1
            break
    assert header_end > 0 and num_verts > 0 and num_faces > 0, f"Incomplete PLY header in {ply_path}"

    verts = np.zeros((num_verts, 3), dtype=np.float32)
    for v_i in range(num_verts):
        parts = lines[header_end + v_i].split()
        verts[v_i, 0] = float(parts[0])
        verts[v_i, 1] = float(parts[1])
        verts[v_i, 2] = float(parts[2])

    tris: list[list[int]] = []
    face_start = header_end + num_verts
    for f_i in range(num_faces):
        parts = [int(tok) for tok in lines[face_start + f_i].split()]
        cnt = parts[0]
        assert cnt >= 3
        for k in range(1, cnt - 1):
            tris.append([parts[1], parts[1 + k], parts[2 + k]])
    return verts, np.asarray(tris, dtype=np.int32)


def _rodrigues_to_rot3x3(rvec: list[float]) -> Any:
    """Pure numpy Rodrigues rotation vector -> 3x3 rotation matrix."""
    import numpy as np

    rv = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = rv / theta
    kx, ky, kz = float(k[0]), float(k[1]), float(k[2])
    k_mat = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * k_mat + (1.0 - math.cos(theta)) * (k_mat @ k_mat)


def _dilate_bool_mask(mask: Any, radius_px: int) -> Any:
    """Dilate a 2D boolean mask by `radius_px` in Chebyshev norm using pure numpy shifts."""
    import numpy as np

    src = np.asarray(mask, dtype=bool)
    h, w = src.shape
    padded = np.pad(src, radius_px, mode="constant", constant_values=False)
    out = np.zeros_like(src)
    for dy in range(-radius_px, radius_px + 1):
        for dx in range(-radius_px, radius_px + 1):
            out |= padded[radius_px + dy : radius_px + dy + h, radius_px + dx : radius_px + dx + w]
    return out


# ==============================================================================
# Workload 1 — Full HD 1280x720 >=180-Frame Video Branding @ >=15 FPS
# ==============================================================================


@pytest.mark.f02
@pytest.mark.f03
@pytest.mark.f05
@pytest.mark.f06
@pytest.mark.f07
@pytest.mark.f10
@pytest.mark.f11
@pytest.mark.f12
def test_t4_01_workload_full_hd_1280x720_180frame_video_branding_15fps(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """Workload 1: Full HD 1280x720 (>=180 frames) `info -> segment -> process -> benchmark` @ >=15.0 FPS."""
    import numpy as np

    demo_hd_path = demo_corpus_dir / "planar_hd_1280x720.mp4"
    demo_probe = probe_video(demo_hd_path, tmp_dir=e2e_tmp_dir)

    synth_meta: dict[str, Any] | None = None
    if demo_probe["width"] == 1280 and demo_probe["height"] == 720 and demo_probe["frame_count"] >= 180:
        hd_clip = demo_hd_path
        selection = demo_corpus_manifest["planar_hd_1280x720"].get("default_selection")
    else:
        hd_clip = e2e_tmp_dir / "synth_hd_1280x720_180f.mp4"
        synth_meta = make_textured_tracking_clip(
            hd_clip,
            width=1280,
            height=720,
            frames=180,
            fps=30,
            tmp_dir=e2e_tmp_dir,
        )
        bx, by, bw, bh = synth_meta["bbox_frame0"]
        cx, cy = synth_meta["center_frame0"]
        selection = {
            "frame_index": 0,
            "positive_points": [[cx, cy]],
            "negative_points": [[40, 40], [1240, 680]],
            "bbox": [bx, by, bw, bh],
        }

    # Step 1: Probe via `visualizeit info --json`
    _, info_json = run_cli_json(["info", "--input", str(hd_clip)], timeout=30.0, tmp_dir=e2e_tmp_dir)
    assert int(info_json["width"]) == 1280 and int(info_json["height"]) == 720
    assert int(info_json["frame_count"]) >= 180
    assert float(info_json["fps"]) >= 29.5

    # Step 2: Segment target on frame 0 via `visualizeit segment --json`
    mask_png = e2e_tmp_dir / "t4_01_hd_mask.png"
    _, seg_json = run_cli_json(
        [
            "segment",
            "--input",
            str(hd_clip),
            "--output-mask",
            str(mask_png),
            *_selection_to_segment_args(selection),
        ],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert mask_png.is_file() and mask_png.stat().st_size > 0
    mask_arr, _ = decode_image_bytes(mask_png.read_bytes())
    assert mask_arr.shape[:2] == (720, 1280)
    assert 0.03 <= float(seg_json["coverage_ratio"]) <= 0.85
    assert int(seg_json["anchor_count"]) >= 8

    # Step 3: Process 100% of HD frames via `visualizeit process --report ... --json`
    out_mp4 = e2e_tmp_dir / "t4_01_hd_branded.mp4"
    proc_report_path = e2e_tmp_dir / "t4_01_hd_process_report.json"
    t0 = time.perf_counter()
    _, proc_report = run_cli_json(
        [
            "process",
            "--input",
            str(hd_clip),
            "--output",
            str(out_mp4),
            "--report",
            str(proc_report_path),
            "--prompt",
            'navy and gold stripes text "BRAND HD"',
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "luminance",
            "--opacity",
            "0.92",
            "--raster-backend",
            "native_cpp",
            *_selection_to_process_args(selection),
        ],
        timeout=120.0,
        tmp_dir=e2e_tmp_dir,
    )
    proc_ext_wall = time.perf_counter() - t0

    assert_video_run_report_invariants(proc_report, require_min_fps=15.0, expect_output_video=True)
    _assert_external_wall_clock_honesty(proc_report, proc_ext_wall)
    assert proc_report["render_stride"] == 1
    assert (
        proc_report["input_frame_count"]
        == proc_report["processed_frame_count"]
        == proc_report["rendered_frame_count"]
        == proc_report["output_frame_count"]
        >= 180
    )
    assert float(proc_report["tracking_availability"]) >= 0.90

    out_probe = probe_video(out_mp4, tmp_dir=e2e_tmp_dir)
    assert out_probe["width"] == 1280 and out_probe["height"] == 720
    assert out_probe["codec_name"] == "h264" and out_probe["pix_fmt"] == "yuv420p"
    assert out_probe["frame_count"] == proc_report["input_frame_count"]

    # Visual anti-cheating check across sampled 180-frame span
    sample_indices = [0, 45, 90, 150]
    raw_frames = decode_video_frames(hd_clip, sample_indices)
    out_frames = decode_video_frames(out_mp4, sample_indices)

    seg_bbox = tuple(int(round(v)) for v in seg_json["bbox"])
    for idx in sample_indices:
        bbox_t = bbox_at_frame(synth_meta, idx) if synth_meta is not None else _localize_textured_fg_bbox(raw_frames[idx], seg_bbox)  # type: ignore[arg-type]
        roi_delta, bg_delta = compute_roi_and_bg_delta(raw_frames[idx], out_frames[idx], bbox_t, margin=40)
        assert roi_delta >= 10.0, f"HD frame {idx} missing mapped artwork inside target ROI: roi_delta={roi_delta:.2f}"
        assert bg_delta <= 3.5, f"HD frame {idx} background modified outside target ROI: bg_delta={bg_delta:.2f}"

    inter_frame_motion = float(np.mean(np.abs(out_frames[45].astype(np.float32) - out_frames[0].astype(np.float32))))
    assert inter_frame_motion > 1.0, "Output video frames appear frozen/duplicated across HD sequence"

    # Step 4: Verify honest >=15.0 FPS benchmark gate on the exact HD clip
    bench_dir = e2e_tmp_dir / "t4_01_hd_bench"
    bench_report_path = e2e_tmp_dir / "t4_01_hd_bench_report.json"
    t1 = time.perf_counter()
    _, bench_report = run_cli_json(
        [
            "benchmark",
            "--input",
            str(hd_clip),
            "--min-fps",
            "15.0",
            "--output-dir",
            str(bench_dir),
            "--report",
            str(bench_report_path),
        ],
        timeout=120.0,
        tmp_dir=e2e_tmp_dir,
    )
    bench_ext_wall = time.perf_counter() - t1
    assert_video_run_report_invariants(bench_report, require_min_fps=15.0, expect_output_video=True)
    _assert_external_wall_clock_honesty(bench_report, bench_ext_wall)
    assert bench_report["input_frame_count"] == proc_report["input_frame_count"] >= 180


# ==============================================================================
# Workload 2 — Cylindrical Mug 360 Label Wrap + Custom Die-Cut RGBA + Luminance
# ==============================================================================


@pytest.mark.f04
@pytest.mark.f05
@pytest.mark.f06
@pytest.mark.f07
@pytest.mark.f10
@pytest.mark.f11
@pytest.mark.f12
def test_t4_02_workload_cylindrical_mug_360_label_wrap_luminance_15fps(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """Workload 2: 360° cylindrical mug wrap with die-cut RGBA label and luminance shading @ >=15.0 FPS."""
    import numpy as np

    mug_mp4 = demo_corpus_dir / "cylinder_mug_640x480.mp4"
    mug_meta = demo_corpus_manifest["cylinder_mug_640x480"]
    selection = mug_meta.get("default_selection")

    label_png = make_rgba_png(e2e_tmp_dir / "diecut_mug_label.png", width=512, height=256, mode="cutout_label")

    # Step 1: `info` & `segment`
    _, info_json = run_cli_json(["info", "--input", str(mug_mp4)], timeout=30.0, tmp_dir=e2e_tmp_dir)
    assert int(info_json["width"]) == 640 and int(info_json["height"]) == 480

    mask_png = e2e_tmp_dir / "t4_02_mug_mask.png"
    _, seg_json = run_cli_json(
        [
            "segment",
            "--input",
            str(mug_mp4),
            "--output-mask",
            str(mask_png),
            *_selection_to_segment_args(selection),
        ],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    mask_arr, _ = decode_image_bytes(mask_png.read_bytes())
    fg_mask_0 = (mask_arr[..., 0] if mask_arr.ndim == 3 else mask_arr) > 127
    assert int(np.count_nonzero(fg_mask_0)) >= 2000

    # Step 2: `process` with cylindrical wrap + custom die-cut RGBA artwork + luminance shading
    out_mp4 = e2e_tmp_dir / "t4_02_mug_wrapped.mp4"
    rep_json = e2e_tmp_dir / "t4_02_mug_report.json"
    t0 = time.perf_counter()
    _, proc_report = run_cli_json(
        [
            "process",
            "--input",
            str(mug_mp4),
            "--output",
            str(out_mp4),
            "--report",
            str(rep_json),
            "--design-image",
            str(label_png),
            "--wrap-mode",
            "cylinder",
            "--blend-mode",
            "luminance",
            "--scale",
            "1.05",
            "--opacity",
            "0.92",
            *_selection_to_process_args(selection),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    ext_wall = time.perf_counter() - t0
    assert_video_run_report_invariants(proc_report, require_min_fps=15.0, expect_output_video=True)
    _assert_external_wall_clock_honesty(proc_report, ext_wall)
    assert proc_report["wrap_mode"] == "cylinder"
    assert proc_report["blend_mode"] == "luminance"
    assert float(proc_report["tracking_availability"]) >= 0.85
    assert probe_video(out_mp4, tmp_dir=e2e_tmp_dir)["frame_count"] == proc_report["input_frame_count"]

    # Verify die-cut alpha=0 windows preserve raw mug surface while opaque label bands apply artwork
    sample_indices = [0, 35, 90]
    raw_frames = decode_video_frames(mug_mp4, sample_indices)
    out_frames = decode_video_frames(out_mp4, sample_indices)

    diff0 = np.mean(np.abs(out_frames[0].astype(np.float32) - raw_frames[0].astype(np.float32)), axis=-1)
    fg_diffs0 = diff0[fg_mask_0]
    opaque_frac = float(np.mean(fg_diffs0 > 12.0))
    cutout_frac = float(np.mean(fg_diffs0 < 4.5))
    assert opaque_frac >= 0.25, f"Expected >=25% of mug pixels covered by opaque label bands, got {opaque_frac:.3f}"
    assert cutout_frac >= 0.06, (
        f"Expected die-cut alpha=0 windows/margins to leave >=6% of mug pixels showing raw ceramic, got {cutout_frac:.3f}"
    )

    seg_bbox = tuple(int(round(v)) for v in seg_json["bbox"])
    for idx in sample_indices:
        bbox_t = _localize_textured_fg_bbox(raw_frames[idx], seg_bbox)  # type: ignore[arg-type]
        roi_d, bg_d = compute_roi_and_bg_delta(raw_frames[idx], out_frames[idx], bbox_t, margin=36)
        assert roi_d >= 9.0, f"Mug frame {idx} missing cylindrical label delta: {roi_d:.2f}"
        assert bg_d <= 3.5, f"Mug frame {idx} background modified: {bg_d:.2f}"

    # Quantitative shaded-surface proof: cylindrical `luminance` preserves highlight/shadow ratio (>1.25x)
    shaded_clip = e2e_tmp_dir / "cyl_shaded_probe.mp4"
    shaded_meta = make_textured_tracking_clip(
        shaded_clip,
        width=640,
        height=480,
        frames=10,
        fps=30,
        shaded_card=True,
        tmp_dir=e2e_tmp_dir,
    )
    sbx, sby, sbw, sbh = shaded_meta["bbox_frame0"]
    shaded_lum_mp4 = e2e_tmp_dir / "cyl_shaded_lum.mp4"
    run_cli_json(
        [
            "process",
            "--input",
            str(shaded_clip),
            "--output",
            str(shaded_lum_mp4),
            "--box",
            f"{sbx},{sby},{sbw},{sbh}",
            "--pattern",
            "solid",
            "--colors",
            "#40a0ff,#40a0ff",
            "--wrap-mode",
            "cylinder",
            "--blend-mode",
            "luminance",
            "--opacity",
            "0.95",
        ],
        timeout=45.0,
        tmp_dir=e2e_tmp_dir,
    )
    sf = decode_video_frames(shaded_lum_mp4, [2])[2].astype(np.float32)
    rx, ry, rw, rh = bbox_at_frame(shaded_meta, 2)
    mid_x = rx + rw // 2
    l_left = float(np.mean(0.299 * sf[ry + 20 : ry + rh - 20, rx + 20 : mid_x - 10, 0] + 0.587 * sf[ry + 20 : ry + rh - 20, rx + 20 : mid_x - 10, 1] + 0.114 * sf[ry + 20 : ry + rh - 20, rx + 20 : mid_x - 10, 2]))
    l_right = float(np.mean(0.299 * sf[ry + 20 : ry + rh - 20, mid_x + 10 : rx + rw - 20, 0] + 0.587 * sf[ry + 20 : ry + rh - 20, mid_x + 10 : rx + rw - 20, 1] + 0.114 * sf[ry + 20 : ry + rh - 20, mid_x + 10 : rx + rw - 20, 2]))
    assert (l_left / max(1.0, l_right)) > 1.25, (
        f"Cylindrical luminance blend failed to preserve left/right highlight-to-shadow ratio: {l_left / max(1.0, l_right):.3f}"
    )

    # Step 3: `benchmark --preset cylinder --min-fps 15.0`
    _, bench_report = run_cli_json(
        [
            "benchmark",
            "--preset",
            "cylinder",
            "--min-fps",
            "15.0",
            "--output-dir",
            str(e2e_tmp_dir / "t4_02_bench"),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(bench_report, require_min_fps=15.0, expect_output_video=True)


# ==============================================================================
# Workload 3 — Multi-Second Occlusion Suppression & Auto Re-Acquisition @ >=15 FPS
# ==============================================================================


@pytest.mark.f03
@pytest.mark.f05
@pytest.mark.f09
@pytest.mark.f10
@pytest.mark.f11
@pytest.mark.f12
def test_t4_03_workload_multi_second_occlusion_suppression_and_reacquisition_15fps(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """Workload 3: Multi-second object occlusion suppresses ghost overlays (`<=2.5`) and auto-reacquires @ >=15 FPS."""
    import numpy as np

    demo_occ = demo_corpus_dir / "occlusion_recovery_640x480.mp4"
    occ_probe = probe_video(demo_occ, tmp_dir=e2e_tmp_dir)

    synth_occ_meta: dict[str, Any] | None = None
    if occ_probe["frame_count"] >= 60:
        occ_clip = demo_occ
        selection = demo_corpus_manifest["occlusion_recovery_640x480"].get("default_selection")
    else:
        occ_clip = e2e_tmp_dir / "synth_occ_90f.mp4"
        synth_occ_meta = make_occlusion_clip(
            occ_clip,
            width=640,
            height=480,
            frames=90,
            fps=30,
            occlude_ranges=[(25, 60)],
            tmp_dir=e2e_tmp_dir,
        )
        bx, by, bw, bh = synth_occ_meta["bbox_frame0"]
        selection = {"frame_index": 0, "bbox": [bx, by, bw, bh]}

    _, info_json = run_cli_json(["info", "--input", str(occ_clip)], timeout=30.0, tmp_dir=e2e_tmp_dir)
    total_frames = int(info_json["frame_count"])
    assert total_frames >= 60

    mask_png = e2e_tmp_dir / "t4_03_occ_mask.png"
    _, seg_json = run_cli_json(
        [
            "segment",
            "--input",
            str(occ_clip),
            "--output-mask",
            str(mask_png),
            *_selection_to_segment_args(selection),
        ],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    seg_bbox = tuple(int(round(v)) for v in seg_json["bbox"])

    out_mp4 = e2e_tmp_dir / "t4_03_occ_processed.mp4"
    rep_json = e2e_tmp_dir / "t4_03_occ_report.json"
    t0 = time.perf_counter()
    _, proc_report = run_cli_json(
        [
            "process",
            "--input",
            str(occ_clip),
            "--output",
            str(out_mp4),
            "--report",
            str(rep_json),
            "--pattern",
            "checker",
            "--colors",
            "#ff006e,#ffbe0b",
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "normal",
            "--opacity",
            "0.95",
            *_selection_to_process_args(selection),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    ext_wall = time.perf_counter() - t0
    assert_video_run_report_invariants(proc_report, require_min_fps=15.0, expect_output_video=True)
    _assert_external_wall_clock_honesty(proc_report, ext_wall)
    assert proc_report["lost_frame_count"] >= 5, f"Expected >=5 occluded lost frames, got {proc_report}"
    assert proc_report["recovered_count"] >= 1, f"Expected >=1 post-occlusion recovery, got {proc_report}"
    assert probe_video(out_mp4, tmp_dir=e2e_tmp_dir)["frame_count"] == total_frames

    # Identify pre-occlusion, fully occluded middle, and post-occlusion frames from raw video physics
    step = max(1, total_frames // 36)
    probe_idxs = sorted(set(range(0, total_frames, step)) | {5, total_frames // 2, total_frames - 8})
    raw_scan = decode_video_frames(occ_clip, probe_idxs)
    sx, sy, sbw, sbh = seg_bbox
    stds = {
        idx: float(np.std(rgb[sy + sbh // 6 : sy + sbh - sbh // 6, sx + sbw // 6 : sx + sbw - sbw // 6].astype(np.float32)))
        for idx, rgb in raw_scan.items()
    }
    occ_frames = [idx for idx in probe_idxs if stds[idx] < 10.0]
    assert len(occ_frames) >= 3, f"Expected multi-frame full occlusion window, got stds={stds}"
    mid_occ = occ_frames[len(occ_frames) // 4 : max(1, (3 * len(occ_frames)) // 4)] or occ_frames
    sampled_occ = sorted({mid_occ[0], mid_occ[len(mid_occ) // 2], mid_occ[-1]})

    pre_candidates = [idx for idx in probe_idxs if idx < min(occ_frames) - step and stds[idx] >= 18.0]
    post_candidates = [idx for idx in probe_idxs if idx > max(occ_frames) + step and stds[idx] >= 18.0]
    sampled_pre = [pre_candidates[0], pre_candidates[len(pre_candidates) // 2]]
    post_tail = post_candidates[len(post_candidates) // 3 :] or post_candidates
    sampled_post = [post_tail[0], post_tail[-1]]

    all_sampled = sorted({*sampled_pre, *sampled_occ, *sampled_post})
    raw_f = decode_video_frames(occ_clip, all_sampled)
    out_f = decode_video_frames(out_mp4, all_sampled)

    # 1) Pre-occlusion: strong foreground overlay delta >= 12.0
    for idx in sampled_pre:
        bbox_t = bbox_at_frame(synth_occ_meta, idx) if synth_occ_meta is not None else _localize_textured_fg_bbox(raw_f[idx], seg_bbox)  # type: ignore[arg-type]
        roi_d, bg_d = compute_roi_and_bg_delta(raw_f[idx], out_f[idx], bbox_t, margin=36)
        assert roi_d >= 12.0, f"Pre-occlusion frame {idx} expected overlay delta >= 12.0, got {roi_d:.2f}"
        assert bg_d <= 3.5, f"Pre-occlusion frame {idx} background modified: {bg_d:.2f}"

    # 2) Occluded middle frames: strict overlay suppression (`mean_abs_diff <= 2.5`, zero ghost checker)
    for idx in sampled_occ:
        occ_diff = float(np.mean(np.abs(out_f[idx].astype(np.float32) - raw_f[idx].astype(np.float32))))
        assert occ_diff <= 2.5, (
            f"Frozen/stale ghost overlay leaked onto occluded frame {idx}: mean_abs_diff={occ_diff:.2f} > 2.5"
        )

    # 3) Post-occlusion re-entry: automatic descriptor re-acquisition restores overlay delta >= 10.0
    post_deltas: list[float] = []
    for idx in sampled_post:
        bbox_t = bbox_at_frame(synth_occ_meta, idx) if synth_occ_meta is not None else _localize_textured_fg_bbox(raw_f[idx], seg_bbox)  # type: ignore[arg-type]
        roi_d, bg_d = compute_roi_and_bg_delta(raw_f[idx], out_f[idx], bbox_t, margin=36)
        post_deltas.append(roi_d)
        assert bg_d <= 3.5, f"Post-occlusion frame {idx} background modified: {bg_d:.2f}"
    assert max(post_deltas) >= 10.0, f"Expected restored post-occlusion overlay delta >= 10.0, got {post_deltas}"

    # 4) Benchmark gate on occlusion clip @ >= 15.0 FPS
    _, bench_report = run_cli_json(
        [
            "benchmark",
            "--input",
            str(occ_clip),
            "--min-fps",
            "15.0",
            "--output-dir",
            str(e2e_tmp_dir / "t4_03_bench"),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(bench_report, require_min_fps=15.0, expect_output_video=True)


# ==============================================================================
# Workload 4 — Calibrated 6-DoF Rigid 3D Mesh Surface Design @ >=15 FPS
# ==============================================================================


@pytest.mark.f02
@pytest.mark.f05
@pytest.mark.f08
@pytest.mark.f10
@pytest.mark.f11
@pytest.mark.f12
def test_t4_04_workload_calibrated_6dof_rigid_3d_mesh_surface_design_15fps(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """Workload 4: Calibrated 6-DoF 3D PLY box tracking + native C++ z-buffer rasterization @ >=15.0 FPS."""
    import numpy as np

    box_mp4 = demo_corpus_dir / "mesh_box_640x480.mp4"
    ply_path = demo_corpus_dir / "mesh_box.ply"
    meta_json_path = demo_corpus_dir / "mesh_box_meta.json"
    assert ply_path.is_file() and meta_json_path.is_file(), f"Missing 3D mesh corpus files in {demo_corpus_dir}"

    box_meta = json.loads(meta_json_path.read_text(encoding="utf-8"))
    intrinsics = [float(v) for v in box_meta["intrinsics"]]
    assert len(intrinsics) == 4 and intrinsics[0] > 0 and intrinsics[1] > 0
    fx, fy, cx, cy = intrinsics
    intr_str = f"{fx:g},{fy:g},{cx:g},{cy:g}"

    selection = demo_corpus_manifest["mesh_box_640x480"].get("default_selection")

    # Step 1: `info` & `segment`
    _, info_json = run_cli_json(["info", "--input", str(box_mp4)], timeout=30.0, tmp_dir=e2e_tmp_dir)
    assert int(info_json["width"]) == 640 and int(info_json["height"]) == 480

    mask_png = e2e_tmp_dir / "t4_04_box_mask.png"
    run_cli_json(
        [
            "segment",
            "--input",
            str(box_mp4),
            "--output-mask",
            str(mask_png),
            *_selection_to_segment_args(selection),
        ],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert mask_png.is_file()

    # Step 2: `process` in `mesh_6dof` mode with native C++17 z-buffer rasterizer
    out_mp4 = e2e_tmp_dir / "t4_04_box_6dof.mp4"
    rep_json = e2e_tmp_dir / "t4_04_box_report.json"
    t0 = time.perf_counter()
    _, proc_report = run_cli_json(
        [
            "process",
            "--input",
            str(box_mp4),
            "--output",
            str(out_mp4),
            "--report",
            str(rep_json),
            "--wrap-mode",
            "mesh_6dof",
            "--mesh",
            str(ply_path),
            "--intrinsics",
            intr_str,
            "--raster-backend",
            "native_cpp",
            "--blend-mode",
            "luminance",
            "--pattern",
            "grid",
            "--colors",
            "#00f5d4,#ff006e,#ffbe0b",
            "--opacity",
            "0.92",
            *_selection_to_process_args(selection),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    ext_wall = time.perf_counter() - t0
    assert_video_run_report_invariants(proc_report, require_min_fps=15.0, expect_output_video=True)
    _assert_external_wall_clock_honesty(proc_report, ext_wall)
    assert proc_report["wrap_mode"] == "mesh_6dof"
    assert proc_report["raster_backend"] == "native_cpp"
    assert float(proc_report["tracking_availability"]) >= 0.85
    assert proc_report["reprojection_rmse_px"] is not None
    assert float(proc_report["reprojection_rmse_px"]) < 5.0, (
        f"6-DoF reprojection RMSE {proc_report['reprojection_rmse_px']:.2f}px exceeds 5.0px gate"
    )
    assert probe_video(out_mp4, tmp_dir=e2e_tmp_dir)["frame_count"] == proc_report["input_frame_count"]

    # Independent 3D ray-triangle silhouette oracle check across rotated frames
    verts_obj, tris = _parse_ascii_ply_mesh(ply_path)
    dummy_uvs = np.zeros((len(verts_obj), 2), dtype=np.float32)
    gt_poses = {int(p["frame_index"]): p for p in box_meta["ground_truth_poses"]}

    check_indices = [idx for idx in (0, 45, 90, 135) if idx in gt_poses]
    raw_frames = decode_video_frames(box_mp4, check_indices)
    out_frames = decode_video_frames(out_mp4, check_indices)

    for idx in check_indices:
        pose = gt_poses[idx]
        rot = (
            np.asarray(pose["R_3x3"], dtype=np.float64)
            if "R_3x3" in pose
            else _rodrigues_to_rot3x3(pose["rvec"])
        )
        tvec = np.asarray(pose["tvec"], dtype=np.float64).reshape(1, 3)
        verts_cam = (verts_obj.astype(np.float64) @ rot.T + tvec).astype(np.float32)

        oracle = ray_triangle_uv_oracle(
            verts_cam,
            tris,
            dummy_uvs,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            frame_w=640,
            frame_h=480,
        )
        gt_hit = oracle["hit_mask"]
        assert int(np.count_nonzero(gt_hit)) > 1000

        diff_map = np.mean(np.abs(out_frames[idx].astype(np.float32) - raw_frames[idx].astype(np.float32)), axis=-1)
        inside_mean = float(np.mean(diff_map[gt_hit]))
        assert inside_mean >= 8.0, f"Expected 3D surface design delta >= 8.0 inside GT box at frame {idx}, got {inside_mean:.2f}"

        dilated_gt = _dilate_bool_mask(gt_hit, radius_px=8)
        changed_px = diff_map > 12.0
        n_changed = int(np.count_nonzero(changed_px))
        assert n_changed > 250, f"Too few modified pixels ({n_changed}) on 3D box at frame {idx}"
        overlap_ratio = float(np.count_nonzero(changed_px & dilated_gt)) / float(n_changed)
        assert overlap_ratio >= 0.80, (
            f"6-DoF overlay drifted outside ground-truth 3D box silhouette at frame {idx}: "
            f"overlap_ratio={overlap_ratio:.3f} < 0.80"
        )

        far_bg = ~_dilate_bool_mask(gt_hit, radius_px=16)
        bg_mean = float(np.mean(diff_map[far_bg]))
        assert bg_mean <= 3.5, f"Background outside 3D box modified at frame {idx}: bg_mean={bg_mean:.2f}"

    # Step 3: `benchmark --preset mesh_6dof --min-fps 15.0`
    _, bench_report = run_cli_json(
        [
            "benchmark",
            "--preset",
            "mesh_6dof",
            "--min-fps",
            "15.0",
            "--output-dir",
            str(e2e_tmp_dir / "t4_04_bench"),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(bench_report, require_min_fps=15.0, expect_output_video=True)


# ==============================================================================
# Workload 10 — Sustained Planar 640x480 Studio, Range Media & Benchmark Gate
# ==============================================================================


@pytest.mark.f01
@pytest.mark.f12
@pytest.mark.f13
def test_t4_10_workload_sustained_planar_640x480_studio_and_benchmark_gate(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """Workload 10: Complete 640x480 planar CLI + `benchmark --preset standard` + HTTP/1.1 Range media verification."""
    planar_mp4 = demo_corpus_dir / "planar_card_640x480.mp4"
    selection = demo_corpus_manifest["planar_card_640x480"].get("default_selection")

    # 1) `info`
    _, info_json = run_cli_json(["info", "--input", str(planar_mp4)], timeout=30.0, tmp_dir=e2e_tmp_dir)
    assert int(info_json["width"]) == 640 and int(info_json["height"]) == 480

    # 2) `segment`
    mask_png = e2e_tmp_dir / "t4_10_planar_mask.png"
    _, seg_json = run_cli_json(
        [
            "segment",
            "--input",
            str(planar_mp4),
            "--output-mask",
            str(mask_png),
            *_selection_to_segment_args(selection),
        ],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert float(seg_json["init_ms"]) < 250.0
    seg_bbox = tuple(int(round(v)) for v in seg_json["bbox"])

    # 3) `process`
    out_mp4 = e2e_tmp_dir / "t4_10_planar_out.mp4"
    proc_json = e2e_tmp_dir / "t4_10_planar_report.json"
    t0 = time.perf_counter()
    _, proc_report = run_cli_json(
        [
            "process",
            "--input",
            str(planar_mp4),
            "--output",
            str(out_mp4),
            "--report",
            str(proc_json),
            "--prompt",
            'teal and coral waves text "STUDIO 640"',
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "luminance",
            "--opacity",
            "0.90",
            "--raster-backend",
            "native_cpp",
            *_selection_to_process_args(selection),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    ext_wall = time.perf_counter() - t0
    assert_video_run_report_invariants(proc_report, require_min_fps=15.0, expect_output_video=True)
    _assert_external_wall_clock_honesty(proc_report, ext_wall)

    # Verify stage latency breakdown consistency
    sl = proc_report["stage_latency_mean_ms"]
    stage_sum_ms = sum(float(sl[k]) for k in ("decode_ms", "prep_ms", "track_ms", "composite_ms", "encode_ms"))
    mean_frame_ms = float(proc_report["frame_latency_ms"]["mean"])
    assert 0.0 < stage_sum_ms <= mean_frame_ms * 1.6 + 5.0, (
        f"Stage latency sum ({stage_sum_ms:.2f}ms) inconsistent with mean frame latency ({mean_frame_ms:.2f}ms): {sl}"
    )

    # Visual foreground attachment & background preservation
    sample_idxs = [0, 30, 90, 150]
    raw_f = decode_video_frames(planar_mp4, sample_idxs)
    out_f = decode_video_frames(out_mp4, sample_idxs)
    for idx in sample_idxs:
        bbox_t = _localize_textured_fg_bbox(raw_f[idx], seg_bbox)  # type: ignore[arg-type]
        roi_d, bg_d = compute_roi_and_bg_delta(raw_f[idx], out_f[idx], bbox_t, margin=36)
        assert roi_d >= 10.0, f"Planar frame {idx} missing mapped artwork delta: {roi_d:.2f}"
        assert bg_d <= 3.5, f"Planar frame {idx} background modified: {bg_d:.2f}"

    # 4) `benchmark --preset standard --min-fps 15.0`
    t1 = time.perf_counter()
    _, bench_report = run_cli_json(
        [
            "benchmark",
            "--preset",
            "standard",
            "--min-fps",
            "15.0",
            "--output-dir",
            str(e2e_tmp_dir / "t4_10_bench"),
            "--report",
            str(e2e_tmp_dir / "t4_10_bench.json"),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    bench_ext_wall = time.perf_counter() - t1
    assert_video_run_report_invariants(bench_report, require_min_fps=15.0, expect_output_video=True)
    _assert_external_wall_clock_honesty(bench_report, bench_ext_wall)

    # 5) Live Studio Server HTTP/1.1 Keep-Alive + HTTP Range (206 Partial Content) media verification
    with spawn_live_server(storage_dir=e2e_tmp_dir / "srv_t4_10") as client:
        health = client.get_json("/api/health", expected_status=200)
        assert health.get("status") == "ok"

        raw_bytes = planar_mp4.read_bytes()
        up_resp = client.upload(
            "/api/videos/upload",
            filename="planar_card_640x480.mp4",
            content=raw_bytes,
            content_type="video/mp4",
            expected_status=200,
            timeout=30.0,
        )
        video_id = up_resp["json"]["video_id"]

        full_media = client.get(f"/api/videos/{video_id}/media", expected_status=200, timeout=30.0)
        assert "bytes" in full_media["headers"].get("accept-ranges", "").lower()
        range_media = client.get_range(f"/api/videos/{video_id}/media", 0, 4095, expected_status=206)
        assert len(range_media["body"]) == 4096
        assert range_media["body"] == full_media["body"][:4096]
        assert f"bytes 0-4095/{len(full_media['body'])}" in range_media["headers"].get("content-range", "")

        sel_resp = client.post_json(
            f"/api/videos/{video_id}/select",
            selection,
            expected_status=200,
            timeout=30.0,
        )
        render_resp = client.post_json(
            f"/api/videos/{video_id}/render",
            {
                "selection_id": sel_resp["selection_id"],
                "design_recipe": {"pattern": "waves", "colors": ["#287e80", "#ff6f61"], "text": "STUDIO 640"},
                "placement": {"wrap_mode": "planar", "blend_mode": "luminance", "opacity": 0.90},
            },
            expected_status=200,
            timeout=90.0,
        )
        assert render_resp.get("status") == "completed"
        assert_video_run_report_invariants(render_resp["report"], require_min_fps=15.0, expect_output_video=True)

        video_url = render_resp["video_url"]
        full_render_mp4 = client.get(video_url, expected_status=200, timeout=30.0)
        range_render_mp4 = client.get_range(video_url, 0, 2047, expected_status=206)
        assert len(range_render_mp4["body"]) == 2048
        assert range_render_mp4["body"] == full_render_mp4["body"][:2048]
