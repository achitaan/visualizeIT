"""Tier 3 Cross-Feature CLI, Native Engine & Pipeline Combination E2E Tests (T3-01 .. T3-10).

Strictly opaque-box:
- Zero imports of `visualizeit.*` anywhere.
- Zero top-level third-party imports (besides `pytest`); `numpy` and `PIL` are imported lazily
  inside functions so `pytest --collect-only` completes in <1s.
- Every test exercises genuine multi-feature interactions across CLI, C++17 native ABI,
  procedural/prompt/custom-RGBA designs, segmentation, 2D/cylindrical/grid/6-DoF tracking,
  occlusion recovery, cross-container transcode, and benchmark gates.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest

from tests.e2e._helpers import (
    REPO_ROOT,
    SETUP_ENV_SCRIPT,
    SO_PATH,
    VI_OK,
    assert_no_python_traceback,
    assert_video_run_report_invariants,
    assert_xfs_temp_path,
    call_pose_to_unity,
    call_rasterize_mesh_rgbd,
    compute_psnr,
    compute_roi_and_bg_delta,
    decode_image_bytes,
    decode_video_frames,
    get_venv_python,
    load_native_abi,
    make_planar_quad_mesh,
    make_rgba_png,
    probe_video,
    ray_triangle_uv_oracle,
    run_cli_json,
    run_subproc,
    transcode_clip,
)

pytestmark = pytest.mark.tier3


# ---------------------------------------------------------------------------
# Local opaque-box helpers (stdlib at module load)
# ---------------------------------------------------------------------------


def _parse_bbox_xywh(raw_bbox: Any) -> tuple[float, float, float, float]:
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        return (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
    if isinstance(raw_bbox, dict):
        w_val = raw_bbox.get("w", raw_bbox.get("width"))
        h_val = raw_bbox.get("h", raw_bbox.get("height"))
        assert w_val is not None and h_val is not None, f"Invalid bbox dict: {raw_bbox!r}"
        return (float(raw_bbox["x"]), float(raw_bbox["y"]), float(w_val), float(h_val))
    raise AssertionError(f"Unsupported bbox format: {raw_bbox!r}")


def _extract_selection_geometry(
    clip_meta: dict[str, Any],
) -> tuple[tuple[int, int], tuple[int, int, int, int], list[tuple[int, int]]]:
    w = int(clip_meta["width"])
    h = int(clip_meta["height"])
    sel = clip_meta.get("default_selection") or {}

    if sel.get("bbox") is not None:
        bx, by, bw, bh = _parse_bbox_xywh(sel["bbox"])
        bbox_int = (int(round(bx)), int(round(by)), int(round(bw)), int(round(bh)))
    else:
        bbox_int = (w // 4, h // 4, w // 2, h // 2)

    pos_pts = sel.get("positive_points") or []
    if pos_pts:
        p0 = pos_pts[0]
        pt_xy = (
            int(round(float(p0["x"] if isinstance(p0, dict) else p0[0]))),
            int(round(float(p0["y"] if isinstance(p0, dict) else p0[1]))),
        )
    else:
        pt_xy = (bbox_int[0] + bbox_int[2] // 2, bbox_int[1] + bbox_int[3] // 2)

    neg_pts: list[tuple[int, int]] = []
    for np_raw in sel.get("negative_points") or []:
        neg_pts.append(
            (
                int(round(float(np_raw["x"] if isinstance(np_raw, dict) else np_raw[0]))),
                int(round(float(np_raw["y"] if isinstance(np_raw, dict) else np_raw[1]))),
            )
        )
    return pt_xy, bbox_int, neg_pts


def _selection_cli_args(clip_meta: dict[str, Any]) -> list[str]:
    (px, py), (bx, by, bw, bh), _ = _extract_selection_geometry(clip_meta)
    return ["--point", f"{px},{py}", "--box", f"{bx},{by},{bw},{bh}"]


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
    raise AssertionError(f"Could not extract intrinsics from {sorted(meta.keys())}")


def _unwrap_single_benchmark_report(payload: dict[str, Any]) -> dict[str, Any]:
    if "render_stride" in payload and "effective_fps" in payload:
        return payload
    if isinstance(payload.get("report"), dict):
        return payload["report"]
    for list_key in ("scenarios", "results", "reports"):
        items = payload.get(list_key)
        if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
            item = items[0]
            return item["report"] if isinstance(item.get("report"), dict) else item
    return payload


def _parse_ply_for_oracle(ply_path: Path) -> tuple[Any, Any, Any]:
    import numpy as np

    lines = [ln.strip() for ln in ply_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    header_end = lines.index("end_header")
    num_verts = 0
    num_faces = 0
    vert_props: list[str] = []
    cur_elem: str | None = None
    for hl in lines[:header_end]:
        parts = hl.split()
        if parts[0] == "element" and len(parts) >= 3:
            cur_elem = parts[1]
            if cur_elem == "vertex":
                num_verts = int(parts[2])
            elif cur_elem == "face":
                num_faces = int(parts[2])
        elif parts[0] == "property" and cur_elem == "vertex":
            vert_props.append(parts[-1])

    u_key = "texture_u" if "texture_u" in vert_props else "u"
    v_key = "texture_v" if "texture_v" in vert_props else "v"
    ix, iy, iz = vert_props.index("x"), vert_props.index("y"), vert_props.index("z")
    iu, iv = vert_props.index(u_key), vert_props.index(v_key)

    body = lines[header_end + 1 :]
    verts = np.zeros((num_verts, 3), dtype=np.float32)
    uvs = np.zeros((num_verts, 2), dtype=np.float32)
    for i in range(num_verts):
        toks = body[i].split()
        verts[i] = [float(toks[ix]), float(toks[iy]), float(toks[iz])]
        uvs[i] = [float(toks[iu]), float(toks[iv])]
    if float(np.max(np.abs(verts))) > 10.0:
        verts *= 1e-3

    tris: list[list[int]] = []
    for i in range(num_faces):
        ftoks = [int(t) for t in body[num_verts + i].split()]
        cnt = ftoks[0]
        idxs = ftoks[1 : 1 + cnt]
        for k in range(1, cnt - 1):
            tris.append([idxs[0], idxs[k], idxs[k + 1]])
    return verts, np.asarray(tris, dtype=np.int32), uvs


def _make_vertical_bar_rgba_webp(out_path: Path, *, width: int = 512, height: int = 160) -> Path:
    """Create a 512x160 lossless RGBA WebP banner with uniform vertical bars for cylinder foreshortening tests."""
    import numpy as np
    from PIL import Image

    target = assert_xfs_temp_path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    ys, xs = np.ogrid[:height, :width]
    band = (ys >= 12) & (ys < height - 12)
    bar_even = ((xs // 32) % 2) == 0
    rgba[band & bar_even] = [0, 235, 205, 255]
    rgba[band & (~bar_even)] = [255, 75, 92, 255]
    try:
        Image.fromarray(rgba, mode="RGBA").save(target, format="WEBP", lossless=True)
    except Exception:
        tmp_png = target.with_name(".webp_banner_src.png")
        try:
            Image.fromarray(rgba, mode="RGBA").save(tmp_png, format="PNG")
            run_subproc(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(tmp_png),
                    "-c:v",
                    "libwebp",
                    "-lossless",
                    "1",
                    str(target),
                ],
                timeout=20.0,
                check=True,
                tmp_dir=target.parent,
            )
        finally:
            tmp_png.unlink(missing_ok=True)
    return target


# ===========================================================================
# Tier 3 Combinations: T3-01 .. T3-10 (10 tests)
# ===========================================================================


@pytest.mark.f01
@pytest.mark.f02
@pytest.mark.f08
@pytest.mark.f10
@pytest.mark.f11
def test_t3_01_setup_env_native_cpp_vs_opencv_parity_on_mesh_6dof(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """F1 × F2 × F8 × F10 × F11: Native C++17 rasterizer vs OpenCV backend parity on 6-DoF PLY box."""
    assert SETUP_ENV_SCRIPT.is_file(), f"Missing setup script: {SETUP_ENV_SCRIPT}"
    assert SO_PATH.is_file(), f"Missing compiled native shared library: {SO_PATH}"
    lib = load_native_abi(SO_PATH)
    assert int(lib.vi_pose_abi_version()) == 1

    clip = demo_corpus_manifest["mesh_box_640x480"]
    ply_path = demo_corpus_dir / "mesh_box.ply"
    meta = json.loads((demo_corpus_dir / "mesh_box_meta.json").read_text(encoding="utf-8"))
    fx, fy, cx, cy = _extract_intrinsics_from_meta(meta)

    common_args = [
        "--input",
        str(clip["path"]),
        "--wrap-mode",
        "mesh_6dof",
        "--mesh",
        str(ply_path),
        "--intrinsics",
        f"{fx:g},{fy:g},{cx:g},{cy:g}",
        "--pattern",
        "checker",
        "--colors",
        "#ff595e,#1982c4",
        "--blend-mode",
        "luminance",
        "--opacity",
        "0.90",
        *_selection_cli_args(clip),
    ]
    out_cpp = e2e_tmp_dir / "mesh_6dof_native_cpp.mp4"
    out_cv = e2e_tmp_dir / "mesh_6dof_opencv.mp4"

    proc_cpp, rep_cpp = run_cli_json(
        ["process", *common_args, "--raster-backend", "native_cpp", "--output", str(out_cpp)],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    proc_cv, rep_cv = run_cli_json(
        ["process", *common_args, "--raster-backend", "opencv", "--output", str(out_cv)],
        timeout=120.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc_cpp)
    assert_no_python_traceback(proc_cv)
    assert_video_run_report_invariants(rep_cpp)
    assert_video_run_report_invariants(rep_cv)

    assert rep_cpp["raster_backend"] == "native_cpp"
    assert rep_cv["raster_backend"] == "opencv"
    assert rep_cpp["output_frame_count"] == rep_cv["output_frame_count"]

    sample_idxs = [0, 20, 45]
    frames_cpp = decode_video_frames(out_cpp, sample_idxs)
    frames_cv = decode_video_frames(out_cv, sample_idxs)
    for idx in sample_idxs:
        psnr = compute_psnr(frames_cpp[idx], frames_cv[idx])
        assert psnr >= 26.0, f"Frame {idx} native_cpp vs opencv PSNR={psnr:.2f} dB < 26.0 dB"

    cpp_comp_ms = float(rep_cpp["stage_latency_mean_ms"]["composite_ms"])
    cv_comp_ms = float(rep_cv["stage_latency_mean_ms"]["composite_ms"])
    assert cpp_comp_ms < 15.0 and cpp_comp_ms <= cv_comp_ms + 0.5, (
        f"Expected native_cpp composite_ms ({cpp_comp_ms:.2f} ms) faster than opencv ({cv_comp_ms:.2f} ms)"
    )


@pytest.mark.f03
@pytest.mark.f05
@pytest.mark.f07
@pytest.mark.f10
@pytest.mark.f11
def test_t3_02_prompt_parsed_waves_with_planar_luminance_and_rotation(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """F3 × F5 × F7 × F10 × F11: Prompt-parsed waves + text mapped onto planar card with luminance & rotation."""
    import numpy as np

    clip = demo_corpus_manifest["planar_card_640x480"]
    out_mp4 = e2e_tmp_dir / "prompt_waves_planar.mp4"

    proc, report = run_cli_json(
        [
            "process",
            "--input",
            str(clip["path"]),
            "--output",
            str(out_mp4),
            "--prompt",
            'teal and orange waves text "SURF" scale 1.5',
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "luminance",
            "--rotation-deg",
            "35",
            "--opacity",
            "0.85",
            *_selection_cli_args(clip),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)
    assert report["wrap_mode"] == "planar"
    assert report["blend_mode"] == "luminance"
    assert float(report["tracking_availability"]) >= 0.90

    _, bbox0, _ = _extract_selection_geometry(clip)
    raw0 = decode_video_frames(Path(str(clip["path"])), [0])[0]
    out0 = decode_video_frames(out_mp4, [0])[0]
    roi_delta, bg_delta = compute_roi_and_bg_delta(raw0, out0, bbox0, margin=28)
    assert roi_delta >= 8.0, f"Expected strong prompt-generated wave artwork delta on card, got {roi_delta:.2f}"
    assert bg_delta <= 3.5, f"Expected background untouched outside card ROI, got {bg_delta:.2f}"

    bx, by, bw, bh = bbox0
    roi_luma = (
        0.299 * out0[by : by + bh, bx : bx + bw, 0].astype(np.float32)
        + 0.587 * out0[by : by + bh, bx : bx + bw, 1].astype(np.float32)
        + 0.114 * out0[by : by + bh, bx : bx + bw, 2].astype(np.float32)
    )
    assert float(np.std(roi_luma)) > 8.0, "Expected preserved shading + wave spatial variation inside card ROI"


@pytest.mark.f03
@pytest.mark.f05
@pytest.mark.f06
@pytest.mark.f07
@pytest.mark.f10
@pytest.mark.f11
def test_t3_03_procedural_checker_with_pos_neg_points_and_mesh_grid_ink(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """F3 × F5 × F6 × F7 × F10 × F11: Pos+neg point segmentation + procedural checker + mesh_grid + ink blend."""
    import numpy as np

    clip = demo_corpus_manifest["planar_card_640x480"]
    (cx, cy), bbox0, neg_pts = _extract_selection_geometry(clip)
    nx_bg, ny_bg = neg_pts[0] if neg_pts else (60, 60)

    # First inspect positive-only mask to locate a foreground edge point we can carve out with --neg-point
    mask_pos_path = e2e_tmp_dir / "t3_03_pos.png"
    _, seg_pos = run_cli_json(
        ["segment", "--input", str(clip["path"]), "--frame", "0", "--point", f"{cx},{cy}", "--output-mask", str(mask_pos_path)],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    m_pos_raw, _ = decode_image_bytes(mask_pos_path.read_bytes())
    m_pos = m_pos_raw[..., 0] if m_pos_raw.ndim == 3 else m_pos_raw
    fg_ys, fg_xs = np.nonzero(m_pos > 160)
    dists = np.hypot(fg_xs - cx, fg_ys - cy)
    edge_pick = int(np.argmax(dists))
    nx_edge, ny_edge = int(fg_xs[edge_pick]), int(fg_ys[edge_pick])

    mask_comb_path = e2e_tmp_dir / "t3_03_pos_neg.png"
    _, seg_comb = run_cli_json(
        [
            "segment",
            "--input",
            str(clip["path"]),
            "--frame",
            "0",
            "--point",
            f"{cx},{cy}",
            "--neg-point",
            f"{nx_bg},{ny_bg}",
            "--neg-point",
            f"{nx_edge},{ny_edge}",
            "--output-mask",
            str(mask_comb_path),
        ],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    m_comb_raw, _ = decode_image_bytes(mask_comb_path.read_bytes())
    m_comb = m_comb_raw[..., 0] if m_comb_raw.ndim == 3 else m_comb_raw
    assert int(m_comb[ny_bg, nx_bg]) == 0
    assert int(m_comb[ny_edge, nx_edge]) < int(m_pos[ny_edge, nx_edge]) or (
        int(seg_comb["foreground_pixels"]) < int(seg_pos["foreground_pixels"])
    )

    out_mp4 = e2e_tmp_dir / "t3_03_mesh_grid_ink.mp4"
    proc, report = run_cli_json(
        [
            "process",
            "--input",
            str(clip["path"]),
            "--output",
            str(out_mp4),
            "--point",
            f"{cx},{cy}",
            "--neg-point",
            f"{nx_bg},{ny_bg}",
            "--neg-point",
            f"{nx_edge},{ny_edge}",
            "--pattern",
            "checker",
            "--colors",
            "#111111,#ffca3a",
            "--wrap-mode",
            "mesh_grid",
            "--blend-mode",
            "ink",
            "--opacity",
            "0.92",
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)
    assert report["wrap_mode"] == "mesh_grid"
    assert report["blend_mode"] == "ink"
    assert float(report["tracking_availability"]) >= 0.85

    raw0 = decode_video_frames(Path(str(clip["path"])), [0])[0]
    out0 = decode_video_frames(out_mp4, [0])[0]

    # Negative-point background patch stays completely un-inked
    y0_n, y1_n = max(0, ny_bg - 8), min(480, ny_bg + 9)
    x0_n, x1_n = max(0, nx_bg - 8), min(640, nx_bg + 9)
    neg_patch_diff = float(np.mean(np.abs(out0[y0_n:y1_n, x0_n:x1_n].astype(np.float32) - raw0[y0_n:y1_n, x0_n:x1_n].astype(np.float32))))
    assert neg_patch_diff <= 3.0, f"Negative-point region was unexpectedly inked (diff={neg_patch_diff:.2f})"

    # Interior around positive point has strong ink modification
    center_diff = float(
        np.mean(np.abs(out0[cy - 30 : cy + 30, cx - 30 : cx + 30].astype(np.float32) - raw0[cy - 30 : cy + 30, cx - 30 : cx + 30].astype(np.float32)))
    )
    assert center_diff >= 6.0, f"Expected interior mesh_grid ink checker delta >= 6.0, got {center_diff:.2f}"


@pytest.mark.f02
@pytest.mark.f04
@pytest.mark.f05
@pytest.mark.f08
@pytest.mark.f10
@pytest.mark.f11
def test_t3_04_custom_rgba_decal_on_6dof_ply_box_with_native_zbuffer(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """F2 × F4 × F5 × F8 × F10 × F11: Custom RGBA PNG decal mapped onto 6-DoF PLY box via native C++ z-buffer."""
    import numpy as np

    logo_png = make_rgba_png(e2e_tmp_dir / "decal_logo.png", width=256, height=256, mode="emblem", alpha_value=255)
    clip = demo_corpus_manifest["mesh_box_640x480"]
    ply_path = demo_corpus_dir / "mesh_box.ply"
    meta = json.loads((demo_corpus_dir / "mesh_box_meta.json").read_text(encoding="utf-8"))
    fx, fy, cx, cy = _extract_intrinsics_from_meta(meta)
    out_mp4 = e2e_tmp_dir / "decal_6dof_box.mp4"

    proc, report = run_cli_json(
        [
            "process",
            "--input",
            str(clip["path"]),
            "--output",
            str(out_mp4),
            "--design-image",
            str(logo_png),
            "--wrap-mode",
            "mesh_6dof",
            "--mesh",
            str(ply_path),
            "--intrinsics",
            f"{fx:g},{fy:g},{cx:g},{cy:g}",
            "--raster-backend",
            "native_cpp",
            "--blend-mode",
            "luminance",
            "--scale",
            "0.8",
            "--opacity",
            "0.95",
            *_selection_cli_args(clip),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)
    assert report["raster_backend"] == "native_cpp"
    assert report["wrap_mode"] == "mesh_6dof"
    assert report["reprojection_rmse_px"] is not None and float(report["reprojection_rmse_px"]) < 5.0

    # Project ground-truth frame-0 3D box silhouette to verify opaque logo vs transparent border coexistence
    verts_obj, tris, uvs = _parse_ply_for_oracle(ply_path)
    poses = meta.get("ground_truth_poses") or meta.get("poses") or meta.get("frames")
    assert isinstance(poses, list) and poses
    p0 = poses[0]
    tvec = np.asarray(p0.get("tvec", p0.get("gt_tvec")), dtype=np.float64)
    if float(np.linalg.norm(tvec)) > 10.0:
        tvec *= 1e-3
    if "R_3x3" in p0 or "gt_R_3x3" in p0:
        r_mat = np.asarray(p0.get("R_3x3", p0.get("gt_R_3x3")), dtype=np.float64).reshape(3, 3)
    else:
        import cv2

        r_mat, _ = cv2.Rodrigues(np.asarray(p0.get("rvec", p0.get("gt_rvec")), dtype=np.float64).reshape(3, 1))
    verts_cam = (verts_obj.astype(np.float64) @ r_mat.T + tvec[None, :]).astype(np.float32)
    gt_hit = ray_triangle_uv_oracle(
        verts_cam, tris, uvs, fx=fx, fy=fy, cx=cx, cy=cy, frame_w=640, frame_h=480
    )["hit_mask"]

    raw0 = decode_video_frames(Path(str(clip["path"])), [0])[0]
    out0 = decode_video_frames(out_mp4, [0])[0]
    box_diffs = np.max(np.abs(out0.astype(np.float32) - raw0.astype(np.float32)), axis=-1)[gt_hit]

    opaque_decal_px = int(np.count_nonzero(box_diffs > 12.0))
    transparent_box_px = int(np.count_nonzero(box_diffs <= 4.0))
    assert opaque_decal_px >= 150, f"Expected opaque RGBA emblem pixels on 3D box, got {opaque_decal_px}"
    assert transparent_box_px >= 150, (
        f"Expected alpha=0 decal border regions on 3D box to preserve raw box texture, got {transparent_box_px}"
    )


@pytest.mark.f04
@pytest.mark.f05
@pytest.mark.f07
@pytest.mark.f10
@pytest.mark.f11
def test_t3_05_custom_webp_banner_on_cylindrical_mug_360_wrap_normal_blend(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """F4 × F5 × F7 × F10 × F11: Wide 512x160 RGBA WebP banner wrapped cylindrically onto mug."""
    import numpy as np

    webp_banner = _make_vertical_bar_rgba_webp(e2e_tmp_dir / "mug_banner.webp", width=512, height=160)
    clip = demo_corpus_manifest["cylinder_mug_640x480"]
    out_cyl = e2e_tmp_dir / "mug_webp_cylinder.mp4"
    out_planar = e2e_tmp_dir / "mug_webp_planar.mp4"

    common = [
        "--input",
        str(clip["path"]),
        "--design-image",
        str(webp_banner),
        "--blend-mode",
        "normal",
        "--offset-y",
        "-0.1",
        "--opacity",
        "0.95",
        *_selection_cli_args(clip),
    ]
    proc_cyl, rep_cyl = run_cli_json(
        ["process", *common, "--wrap-mode", "cylinder", "--output", str(out_cyl)],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    _, rep_planar = run_cli_json(
        ["process", *common, "--wrap-mode", "planar", "--output", str(out_planar)],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc_cyl)
    assert_video_run_report_invariants(rep_cyl)
    assert_video_run_report_invariants(rep_planar)
    assert rep_cyl["wrap_mode"] == "cylinder"
    assert rep_cyl["blend_mode"] == "normal"
    assert float(rep_cyl["tracking_availability"]) >= 0.85

    _, bbox0, _ = _extract_selection_geometry(clip)
    bx, by, bw, bh = bbox0
    raw0 = decode_video_frames(Path(str(clip["path"])), [0])[0]
    cyl0 = decode_video_frames(out_cyl, [0])[0]
    pla0 = decode_video_frames(out_planar, [0])[0]

    roi_delta, bg_delta = compute_roi_and_bg_delta(raw0, cyl0, bbox0, margin=28)
    assert roi_delta >= 6.0, f"Expected WebP banner rendered on mug body, got roi_delta={roi_delta:.2f}"
    assert bg_delta <= 3.5, f"Expected background outside mug untouched, got bg_delta={bg_delta:.2f}"

    # Verify genuine cylindrical curvature: cylinder wrap differs from flat planar warp inside mug ROI
    cyl_vs_planar_mad = float(
        np.mean(np.abs(cyl0[by : by + bh, bx : bx + bw].astype(np.float32) - pla0[by : by + bh, bx : bx + bw].astype(np.float32)))
    )
    assert cyl_vs_planar_mad > 1.5, (
        f"Cylindrical mug wrap indistinguishable from flat planar wrap (MAD={cyl_vs_planar_mad:.3f})"
    )


@pytest.mark.f05
@pytest.mark.f06
@pytest.mark.f09
@pytest.mark.f10
@pytest.mark.f11
def test_t3_06_box_plus_neg_point_selection_through_occlusion_recovery_ink(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """F5 × F6 × F9 × F10 × F11: Box + neg-point onboarding on occlusion clip with dots + ink suppression/recovery."""
    import numpy as np

    clip = demo_corpus_manifest["occlusion_recovery_640x480"]
    _, (bx, by, bw, bh), neg_pts = _extract_selection_geometry(clip)
    nx, ny = neg_pts[0] if neg_pts else (80, 75)
    out_mp4 = e2e_tmp_dir / "occlusion_dots_ink.mp4"

    proc, report = run_cli_json(
        [
            "process",
            "--input",
            str(clip["path"]),
            "--output",
            str(out_mp4),
            "--box",
            f"{bx},{by},{bw},{bh}",
            "--neg-point",
            f"{nx},{ny}",
            "--pattern",
            "dots",
            "--colors",
            "#101820,#fee715",
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "ink",
            "--opacity",
            "0.95",
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)
    assert int(report["lost_frame_count"]) >= 3, f"Expected >= 3 lost frames during occlusion, got {report['lost_frame_count']}"
    assert int(report["recovered_count"]) >= 1, f"Expected >= 1 recovery after occlusion, got {report['recovered_count']}"

    n_frames = int(report["input_frame_count"])
    early_idx = max(0, int(0.08 * n_frames))
    mid_idx = int(0.50 * n_frames)
    late_idx = min(n_frames - 1, int(0.84 * n_frames))

    raw_frames = decode_video_frames(Path(str(clip["path"])), [early_idx, mid_idx, late_idx])
    out_frames = decode_video_frames(out_mp4, [early_idx, mid_idx, late_idx])

    early_roi_delta, _ = compute_roi_and_bg_delta(raw_frames[early_idx], out_frames[early_idx], (bx, by, bw, bh))
    late_roi_delta, _ = compute_roi_and_bg_delta(raw_frames[late_idx], out_frames[late_idx], (bx, by, bw, bh))
    mid_full_mad = float(np.mean(np.abs(out_frames[mid_idx].astype(np.float32) - raw_frames[mid_idx].astype(np.float32))))

    assert early_roi_delta >= 4.0, f"Expected visible ink dots before occlusion, got {early_roi_delta:.2f}"
    assert mid_full_mad <= 2.5, f"Expected zero ink bleed during full occlusion frame {mid_idx}, got {mid_full_mad:.2f}"
    assert late_roi_delta >= 4.0, f"Expected automatic ink re-acquisition after occlusion, got {late_roi_delta:.2f}"


@pytest.mark.f05
@pytest.mark.f07
@pytest.mark.f10
@pytest.mark.f11
@pytest.mark.f12
def test_t3_07_hd_1280x720_tracking_height_360_grid_wrap_15fps_benchmark(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """F5 × F7 × F10 × F11 × F12: Full HD 1280x720 resize_first=360 + mesh_grid + luminance + >=15 FPS benchmark."""
    clip = demo_corpus_manifest["planar_hd_1280x720"]
    out_mp4 = e2e_tmp_dir / "hd_360_grid.mp4"

    proc_p, rep_p = run_cli_json(
        [
            "process",
            "--input",
            str(clip["path"]),
            "--output",
            str(out_mp4),
            "--tracking-height",
            "360",
            "--wrap-mode",
            "mesh_grid",
            "--blend-mode",
            "luminance",
            *_selection_cli_args(clip),
        ],
        timeout=120.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc_p)
    assert_video_run_report_invariants(rep_p, require_min_fps=15.0)
    assert (rep_p["width"], rep_p["height"]) == (1280, 720)
    assert rep_p["wrap_mode"] == "mesh_grid"
    assert rep_p["blend_mode"] == "luminance"

    probed = probe_video(out_mp4, tmp_dir=e2e_tmp_dir)
    assert (probed["width"], probed["height"]) == (1280, 720)
    assert probed["frame_count"] == int(rep_p["input_frame_count"])

    bench_dir = e2e_tmp_dir / "hd_bench_out"
    bench_report_file = e2e_tmp_dir / "hd_bench_report.json"
    proc_b, raw_bench = run_cli_json(
        [
            "benchmark",
            "--input",
            str(clip["path"]),
            "--min-fps",
            "15.0",
            "--output-dir",
            str(bench_dir),
            "--report",
            str(bench_report_file),
        ],
        timeout=120.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc_b)
    rep_b = _unwrap_single_benchmark_report(raw_bench)
    assert_video_run_report_invariants(rep_b, require_min_fps=15.0)
    assert (rep_b["width"], rep_b["height"]) == (1280, 720)


@pytest.mark.f02
@pytest.mark.f03
@pytest.mark.f05
@pytest.mark.f08
@pytest.mark.f10
@pytest.mark.f11
def test_t3_08_parametric_cylinder_mesh_6dof_with_custom_intrinsics_and_stripes(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """F2 × F3 × F5 × F8 × F10 × F11: Parametric cylinder 6-DoF + custom intrinsics + stripes + native luminance."""
    clip = demo_corpus_manifest["cylinder_mug_640x480"]
    out_mp4 = e2e_tmp_dir / "param_cyl_6dof_stripes.mp4"

    proc, report = run_cli_json(
        [
            "process",
            "--input",
            str(clip["path"]),
            "--output",
            str(out_mp4),
            "--wrap-mode",
            "mesh_6dof",
            "--mesh",
            "cylinder",
            "--intrinsics",
            "560,560,320,240",
            "--raster-backend",
            "native_cpp",
            "--pattern",
            "stripes",
            "--colors",
            "#e63946,#f1faee,#457b9d",
            "--blend-mode",
            "luminance",
            "--opacity",
            "0.92",
            *_selection_cli_args(clip),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)
    assert report["wrap_mode"] == "mesh_6dof"
    assert report["blend_mode"] == "luminance"
    assert report["raster_backend"] == "native_cpp"
    assert int(report["tracked_frame_count"]) > 0

    _, bbox0, _ = _extract_selection_geometry(clip)
    raw0 = decode_video_frames(Path(str(clip["path"])), [0])[0]
    out0 = decode_video_frames(out_mp4, [0])[0]
    roi_delta, bg_delta = compute_roi_and_bg_delta(raw0, out0, bbox0, margin=28)
    assert roi_delta >= 4.0, f"Expected parametric cylinder 6-DoF stripes on mug, got roi_delta={roi_delta:.2f}"
    assert bg_delta <= 3.5, f"Expected background outside mug preserved, got bg_delta={bg_delta:.2f}"


@pytest.mark.f05
@pytest.mark.f09
@pytest.mark.f11
@pytest.mark.f12
def test_t3_09_webm_transcoded_occlusion_clip_cli_process_and_benchmark_gate(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    """F5 × F9 × F11 × F12: VP8/WebM transcoded occlusion clip -> info -> process (H.264) -> >=15 FPS benchmark."""
    src_clip = demo_corpus_manifest["occlusion_recovery_640x480"]
    webm_path = transcode_clip(
        Path(str(src_clip["path"])),
        e2e_tmp_dir / "occlusion_transcoded.webm",
        tmp_dir=e2e_tmp_dir,
    )
    probed_webm = probe_video(webm_path, tmp_dir=e2e_tmp_dir)
    assert (probed_webm["width"], probed_webm["height"]) == (640, 480)

    proc_info, info_json = run_cli_json(["info", "--input", str(webm_path)], timeout=30.0, tmp_dir=e2e_tmp_dir)
    assert_no_python_traceback(proc_info)
    assert (int(info_json["width"]), int(info_json["height"])) == (640, 480)
    assert int(info_json["frame_count"]) == probed_webm["frame_count"]

    out_mp4 = e2e_tmp_dir / "webm_to_h264_out.mp4"
    proc_p, rep_p = run_cli_json(
        [
            "process",
            "--input",
            str(webm_path),
            "--output",
            str(out_mp4),
            *_selection_cli_args(src_clip),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc_p)
    assert_video_run_report_invariants(rep_p, require_min_fps=15.0)
    assert int(rep_p["lost_frame_count"]) >= 3
    assert int(rep_p["recovered_count"]) >= 1

    probed_out = probe_video(out_mp4, tmp_dir=e2e_tmp_dir)
    assert probed_out["codec_name"] == "h264"
    assert probed_out["frame_count"] == probed_webm["frame_count"]

    bench_dir = e2e_tmp_dir / "webm_bench_out"
    proc_b, raw_bench = run_cli_json(
        [
            "benchmark",
            "--input",
            str(webm_path),
            "--min-fps",
            "15.0",
            "--output-dir",
            str(bench_dir),
        ],
        timeout=120.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc_b)
    rep_b = _unwrap_single_benchmark_report(raw_bench)
    assert_video_run_report_invariants(rep_b, require_min_fps=15.0)


@pytest.mark.f01
@pytest.mark.f02
@pytest.mark.f12
def test_t3_10_prototype_bootstrap_native_abi_and_benchmark_coexistence(
    e2e_tmp_dir: Path,
) -> None:
    """F1 × F2 × F12: Linux `-S` prototype_bootstrap + dual C++ ABI symbols + >=15 FPS benchmark coexistence."""
    import numpy as np

    assert SETUP_ENV_SCRIPT.is_file()
    venv_py = get_venv_python()
    assert venv_py.is_file()

    # 1. Verify legacy pose ABI and new 3D mesh rasterizer ABI coexist in libvisualizeit_pose.so without conflict
    lib = load_native_abi(SO_PATH)
    assert int(lib.vi_pose_abi_version()) == 1

    pose_in = np.eye(4, dtype=np.float32)
    pose_in[0, 3], pose_in[1, 3], pose_in[2, 3] = 0.15, -0.25, 0.85
    rc_pose, pose_out = call_pose_to_unity(pose_in, lib=lib)
    assert rc_pose == VI_OK
    assert math.isclose(float(pose_out[1, 3]), 0.25, abs_tol=1e-5)

    verts, tris, uvs = make_planar_quad_mesh(width_m=0.36, height_m=0.26, z_m=1.0)
    tex = np.full((64, 64, 4), (240, 90, 50, 255), dtype=np.uint8)
    raster_res = call_rasterize_mesh_rgbd(
        verts,
        tris,
        uvs,
        tex,
        fx=520.0,
        fy=520.0,
        cx=320.0,
        cy=240.0,
        frame_w=640,
        frame_h=480,
        mask_out=True,
        lib=lib,
    )
    assert raster_res["rc"] > 0
    assert raster_res["mask"] is not None and int(np.count_nonzero(raster_res["mask"])) == raster_res["rc"]

    # 2. Verify prototype_bootstrap.py under Linux `-B -S` runs legacy unittest `tools.test_phone_static`
    bootstrap_py = REPO_ROOT / "tools" / "prototype_bootstrap.py"
    proc_unit = run_subproc(
        [str(venv_py), "-B", "-S", str(bootstrap_py), "unittest", "tools.test_phone_static"],
        timeout=60.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert proc_unit.returncode == 0, (
        f"prototype_bootstrap unittest tools.test_phone_static failed (rc={proc_unit.returncode}):\n"
        f"STDOUT:\n{proc_unit.stdout}\nSTDERR:\n{proc_unit.stderr}"
    )
    assert_no_python_traceback(proc_unit)

    # 3. Run benchmark gate (`--preset quick --min-fps 15.0`) and feed derived tracking rows to `tools.evaluate_tracking`
    bench_dir = e2e_tmp_dir / "coexist_bench"
    bench_json_path = e2e_tmp_dir / "coexist_bench.json"
    proc_bench, raw_bench = run_cli_json(
        [
            "benchmark",
            "--preset",
            "quick",
            "--min-fps",
            "15.0",
            "--output-dir",
            str(bench_dir),
            "--report",
            str(bench_json_path),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc_bench)
    rep = _unwrap_single_benchmark_report(raw_bench)
    assert_video_run_report_invariants(rep, require_min_fps=15.0)

    n_total = int(rep["input_frame_count"])
    n_tracked = int(rep["tracked_frame_count"])
    err_px = float(rep["reprojection_rmse_px"]) if rep["reprojection_rmse_px"] is not None else 1.25
    csv_path = e2e_tmp_dir / "bench_annotations.csv"
    eval_out_path = e2e_tmp_dir / "eval_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["frame", "point", "valid", "expected_x", "expected_y", "observed_x", "observed_y"],
        )
        writer.writeheader()
        for f_idx in range(n_total):
            is_valid = f_idx < n_tracked
            writer.writerow(
                {
                    "frame": str(f_idx),
                    "point": "center",
                    "valid": "1" if is_valid else "0",
                    "expected_x": "320.0",
                    "expected_y": "240.0",
                    "observed_x": f"{320.0 + (min(err_px, 4.0) if is_valid else 0.0):.3f}",
                    "observed_y": "240.0",
                }
            )

    proc_eval = run_subproc(
        [str(venv_py), "-B", "-S", str(bootstrap_py), "tools.evaluate_tracking", str(csv_path), "--output", str(eval_out_path)],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert proc_eval.returncode == 0, (
        f"prototype_bootstrap tools.evaluate_tracking failed (rc={proc_eval.returncode}):\n"
        f"STDOUT:\n{proc_eval.stdout}\nSTDERR:\n{proc_eval.stderr}"
    )
    assert_no_python_traceback(proc_eval)
    eval_summary = json.loads(eval_out_path.read_text(encoding="utf-8"))
    assert int(eval_summary["annotatedFrames"]) == n_total
