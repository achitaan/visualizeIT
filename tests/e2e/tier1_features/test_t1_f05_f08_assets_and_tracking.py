"""Tier 1 Feature Happy-Path E2E Tests — Slice B: Features F5, F6, F7, and F8 (24 tests).

Strictly opaque-box:
- Zero imports of `visualizeit.*` anywhere.
- Zero top-level third-party imports (besides `pytest`); `numpy`, `PIL`, and `cv2` are imported
  lazily inside test/helper functions so `pytest --collect-only` completes in <1s.
- All temporary artifacts reside under XFS `.cache/tmp` via `e2e_tmp_dir` (never `/tmp`).
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any

import pytest

from tests.e2e._helpers import (
    StudioAPIClient,
    assert_no_python_traceback,
    assert_video_run_report_invariants,
    compute_psnr,
    compute_roi_and_bg_delta,
    decode_image_bytes,
    decode_video_frames,
    make_textured_tracking_clip,
    probe_video,
    ray_triangle_uv_oracle,
    run_cli_json,
)

pytestmark = pytest.mark.tier1

EXPECTED_DEMO_CLIP_IDS: tuple[str, ...] = (
    "planar_card_640x480",
    "planar_hd_1280x720",
    "cylinder_mug_640x480",
    "occlusion_recovery_640x480",
    "mesh_box_640x480",
)

REQUIRED_SEGMENT_CLI_KEYS: tuple[str, ...] = (
    "frame_index",
    "width",
    "height",
    "foreground_pixels",
    "coverage_ratio",
    "bbox",
    "polygon",
    "anchor_count",
    "init_ms",
    "mask_path",
)

REQUIRED_SELECT_REST_KEYS: tuple[str, ...] = (
    "selection_id",
    "frame_index",
    "foreground_pixels",
    "coverage_ratio",
    "bbox",
    "polygon",
    "anchor_count",
    "init_ms",
    "mask_png_base64",
)


# ---------------------------------------------------------------------------
# Opaque-box schema & geometry helpers (stdlib at module load)
# ---------------------------------------------------------------------------


def _parse_bbox_xywh(raw_bbox: Any) -> tuple[float, float, float, float]:
    """Normalize a bbox formatted as `[x, y, w, h]` or `{x, y, w, h}` into floats."""
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        return (
            float(raw_bbox[0]),
            float(raw_bbox[1]),
            float(raw_bbox[2]),
            float(raw_bbox[3]),
        )
    if isinstance(raw_bbox, dict):
        w_val = raw_bbox.get("w", raw_bbox.get("width"))
        h_val = raw_bbox.get("h", raw_bbox.get("height"))
        assert w_val is not None and h_val is not None, f"Invalid bbox dict: {raw_bbox!r}"
        return (float(raw_bbox["x"]), float(raw_bbox["y"]), float(w_val), float(h_val))
    raise AssertionError(f"Unsupported bbox structure: {raw_bbox!r}")


def _extract_selection_geometry(
    clip_meta: dict[str, Any],
) -> tuple[tuple[int, int], tuple[int, int, int, int], list[tuple[int, int]]]:
    """Return `(pos_point_xy, bbox_xywh, neg_points_xy)` in integer pixel coordinates."""
    w = int(clip_meta["width"])
    h = int(clip_meta["height"])
    sel = clip_meta.get("default_selection") or {}
    assert isinstance(sel, dict), f"Expected dict default_selection, got {sel!r}"

    raw_bbox = sel.get("bbox")
    if raw_bbox is not None:
        bx, by, bw, bh = _parse_bbox_xywh(raw_bbox)
        bbox_int = (int(round(bx)), int(round(by)), int(round(bw)), int(round(bh)))
    else:
        bbox_int = (w // 4, h // 4, w // 2, h // 2)

    pos_pts = sel.get("positive_points") or []
    if pos_pts:
        p0 = pos_pts[0]
        if isinstance(p0, dict):
            pt_xy = (int(round(float(p0["x"]))), int(round(float(p0["y"]))))
        else:
            pt_xy = (int(round(float(p0[0]))), int(round(float(p0[1]))))
    else:
        pt_xy = (bbox_int[0] + bbox_int[2] // 2, bbox_int[1] + bbox_int[3] // 2)

    neg_pts: list[tuple[int, int]] = []
    for np_raw in sel.get("negative_points") or []:
        if isinstance(np_raw, dict):
            neg_pts.append((int(round(float(np_raw["x"]))), int(round(float(np_raw["y"])))))
        else:
            neg_pts.append((int(round(float(np_raw[0]))), int(round(float(np_raw[1])))))
    return pt_xy, bbox_int, neg_pts


def _selection_cli_args(clip_meta: dict[str, Any]) -> list[str]:
    """Build `--point` and `--box` CLI onboarding flags from `default_selection`."""
    pt_xy, bbox_int, _ = _extract_selection_geometry(clip_meta)
    bx, by, bw, bh = bbox_int
    return [
        "--point",
        f"{pt_xy[0]},{pt_xy[1]}",
        "--box",
        f"{bx},{by},{bw},{bh}",
    ]


def _extract_intrinsics_from_meta(meta: dict[str, Any]) -> tuple[float, float, float, float]:
    """Flexibly extract `(fx, fy, cx, cy)` from `mesh_box_meta.json`."""
    assert isinstance(meta, dict), f"Expected dict in mesh_box_meta.json, got {type(meta).__name__}"

    if all(k in meta for k in ("fx", "fy", "cx", "cy")):
        vals = (float(meta["fx"]), float(meta["fy"]), float(meta["cx"]), float(meta["cy"]))
        assert all(math.isfinite(v) and v > 0.0 for v in vals), f"Invalid top-level intrinsics: {vals}"
        return vals

    for key in ("intrinsics", "camera_intrinsics", "camera", "calibration", "K", "cam_K", "cam_K_3x3"):
        if key not in meta or meta[key] is None:
            continue
        sub = meta[key]
        if isinstance(sub, dict):
            if all(k in sub for k in ("fx", "fy", "cx", "cy")):
                vals = (float(sub["fx"]), float(sub["fy"]), float(sub["cx"]), float(sub["cy"]))
                assert all(math.isfinite(v) and v > 0.0 for v in vals), f"Invalid nested intrinsics: {vals}"
                return vals
            for k_mat in ("K", "cam_K", "cam_K_3x3", "matrix"):
                if k_mat in sub:
                    sub = sub[k_mat]
                    break
        if isinstance(sub, (list, tuple)):
            if len(sub) == 4 and all(isinstance(x, (int, float)) for x in sub):
                vals = (float(sub[0]), float(sub[1]), float(sub[2]), float(sub[3]))
                assert all(math.isfinite(v) and v > 0.0 for v in vals), f"Invalid 4-tuple intrinsics: {vals}"
                return vals
            if len(sub) == 3 and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in sub):
                vals = (float(sub[0][0]), float(sub[1][1]), float(sub[0][2]), float(sub[1][2]))
                assert all(math.isfinite(v) and v > 0.0 for v in vals), f"Invalid 3x3 K matrix: {vals}"
                return vals
            if len(sub) == 9 and all(isinstance(x, (int, float)) for x in sub):
                vals = (float(sub[0]), float(sub[4]), float(sub[2]), float(sub[5]))
                assert all(math.isfinite(v) and v > 0.0 for v in vals), f"Invalid flat 9-element K: {vals}"
                return vals

    raise AssertionError(f"Could not extract camera intrinsics (fx, fy, cx, cy) from: {sorted(meta.keys())}")


def _format_intrinsics_arg(intr: tuple[float, float, float, float]) -> str:
    return f"{intr[0]:g},{intr[1]:g},{intr[2]:g},{intr[3]:g}"


def _parse_ascii_ply(ply_path: Path) -> dict[str, Any]:
    """Parse an ASCII `.ply` file and validate header, finite 3D vertices, UVs, and triangle faces."""
    import numpy as np

    assert ply_path.is_file() and ply_path.stat().st_size > 0, f"Missing or empty PLY: {ply_path}"
    lines = [ln.strip() for ln in ply_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines and lines[0] == "ply", f"PLY file must start with 'ply', got {lines[:2]!r}"
    assert any(ln.startswith("format ascii") for ln in lines[:6]), f"Expected ASCII PLY format: {lines[:6]!r}"
    assert "end_header" in lines, "PLY missing 'end_header'"

    header_end = lines.index("end_header")
    header_lines = lines[:header_end]
    body_lines = lines[header_end + 1 :]

    num_verts = 0
    num_faces = 0
    vert_props: list[str] = []
    current_element: str | None = None
    for hl in header_lines:
        parts = hl.split()
        if parts[0] == "element" and len(parts) >= 3:
            current_element = parts[1]
            if current_element == "vertex":
                num_verts = int(parts[2])
            elif current_element == "face":
                num_faces = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            vert_props.append(parts[-1])

    assert num_verts >= 8, f"Expected >= 8 box vertices in {ply_path}, got {num_verts}"
    assert num_faces >= 12, f"Expected >= 12 triangle faces in {ply_path}, got {num_faces}"
    for req_xyz in ("x", "y", "z"):
        assert req_xyz in vert_props, f"PLY missing vertex property {req_xyz!r}: {vert_props}"

    u_name = "texture_u" if "texture_u" in vert_props else ("u" if "u" in vert_props else ("s" if "s" in vert_props else None))
    v_name = "texture_v" if "texture_v" in vert_props else ("v" if "v" in vert_props else ("t" if "t" in vert_props else None))
    assert u_name is not None and v_name is not None, f"PLY missing UV properties: {vert_props}"

    ix, iy, iz = vert_props.index("x"), vert_props.index("y"), vert_props.index("z")
    iu, iv = vert_props.index(u_name), vert_props.index(v_name)

    assert len(body_lines) >= num_verts + num_faces, (
        f"PLY body truncated: {len(body_lines)} lines < {num_verts} vertices + {num_faces} faces"
    )
    verts = np.zeros((num_verts, 3), dtype=np.float64)
    uvs = np.zeros((num_verts, 2), dtype=np.float64)
    for vi in range(num_verts):
        tokens = body_lines[vi].split()
        assert len(tokens) >= len(vert_props), f"Vertex {vi} has too few columns: {tokens}"
        verts[vi, 0] = float(tokens[ix])
        verts[vi, 1] = float(tokens[iy])
        verts[vi, 2] = float(tokens[iz])
        uvs[vi, 0] = float(tokens[iu])
        uvs[vi, 1] = float(tokens[iv])

    assert np.all(np.isfinite(verts)), "Non-finite XYZ coordinates in mesh_box.ply"
    assert np.all(np.isfinite(uvs)), "Non-finite UV coordinates in mesh_box.ply"
    assert float(np.min(uvs)) >= -1e-4 and float(np.max(uvs)) <= 1.0 + 1e-4, (
        f"UV coordinates out of [0, 1] range: min={np.min(uvs)}, max={np.max(uvs)}"
    )

    tris: list[list[int]] = []
    for fi in range(num_faces):
        ftoks = [int(t) for t in body_lines[num_verts + fi].split()]
        assert len(ftoks) >= 4 and ftoks[0] >= 3, f"Invalid face entry at {fi}: {ftoks}"
        cnt = ftoks[0]
        idxs = ftoks[1 : 1 + cnt]
        for idx_v in idxs:
            assert 0 <= idx_v < num_verts, f"Out-of-bounds face vertex index {idx_v} (num_verts={num_verts})"
        for k in range(1, cnt - 1):
            tris.append([idxs[0], idxs[k], idxs[k + 1]])

    return {
        "num_vertices": num_verts,
        "num_faces": num_faces,
        "vertices": verts.astype(np.float32),
        "uvs": uvs.astype(np.float32),
        "triangles": np.asarray(tris, dtype=np.int32),
    }


def _extract_gt_poses_list(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-frame ground-truth pose list from `mesh_box_meta.json`."""
    for key in ("ground_truth_poses", "poses", "frames", "trajectory"):
        val = meta.get(key)
        if isinstance(val, list) and val:
            return val
    raise AssertionError(f"mesh_box_meta.json missing per-frame poses list; keys={sorted(meta.keys())}")


def _pose_entry_to_rt(entry: dict[str, Any]) -> tuple[Any, Any]:
    """Convert a ground-truth pose dict into `(R_3x3 float64, tvec_3 float64)`."""
    import numpy as np

    t_raw = entry.get("tvec", entry.get("gt_tvec", entry.get("cam_t_m2c", entry.get("t"))))
    assert isinstance(t_raw, (list, tuple)) and len(t_raw) == 3, f"Missing 3D tvec in pose entry: {entry!r}"
    tvec = np.asarray([float(x) for x in t_raw], dtype=np.float64)
    assert np.all(np.isfinite(tvec)) and tvec[2] > 0.0, f"Invalid camera-space tvec: {tvec}"

    r_mat_raw = entry.get("R_3x3", entry.get("gt_R_3x3", entry.get("R", entry.get("cam_R_m2c"))))
    if r_mat_raw is not None:
        r_mat = np.asarray(r_mat_raw, dtype=np.float64).reshape(3, 3)
        assert np.all(np.isfinite(r_mat)), f"Non-finite R_3x3: {r_mat}"
        return r_mat, tvec

    rvec_raw = entry.get("rvec", entry.get("gt_rvec"))
    assert isinstance(rvec_raw, (list, tuple)) and len(rvec_raw) == 3, f"Pose missing R_3x3/rvec: {entry!r}"
    import cv2

    rvec = np.asarray([float(x) for x in rvec_raw], dtype=np.float64).reshape(3, 1)
    r_mat, _ = cv2.Rodrigues(rvec)
    return r_mat, tvec


def _assert_segment_cli_schema(
    data: dict[str, Any],
    *,
    expected_w: int,
    expected_h: int,
    expected_frame: int,
) -> None:
    missing = [k for k in REQUIRED_SEGMENT_CLI_KEYS if k not in data]
    assert not missing, f"segment JSON missing keys {missing}; got {sorted(data.keys())}"
    assert int(data["frame_index"]) == expected_frame
    assert int(data["width"]) == expected_w
    assert int(data["height"]) == expected_h
    assert isinstance(data["foreground_pixels"], int) and data["foreground_pixels"] > 0
    cov = float(data["coverage_ratio"])
    assert math.isfinite(cov) and 0.0 < cov <= 1.0
    bx, by, bw, bh = _parse_bbox_xywh(data["bbox"])
    assert bw > 0.0 and bh > 0.0 and 0.0 <= bx < expected_w and 0.0 <= by < expected_h
    poly = data["polygon"]
    assert isinstance(poly, list) and len(poly) >= 3, f"Expected polygon with >=3 vertices, got {poly!r}"
    assert isinstance(data["anchor_count"], int) and data["anchor_count"] >= 0
    init_ms = float(data["init_ms"])
    assert math.isfinite(init_ms) and init_ms > 0.0


def _spatial_gradient_energy(rgb: Any) -> float:
    import numpy as np

    arr = np.asarray(rgb, dtype=np.float32)
    gx = float(np.mean(np.abs(arr[:, 1:, :] - arr[:, :-1, :])))
    gy = float(np.mean(np.abs(arr[1:, :, :] - arr[:-1, :, :])))
    return gx + gy


def _locate_textured_object_on_frame(rgb: Any) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """Locate `(center_xy, bbox_xywh)` of the textured foreground object purely from RGB gradients."""
    import numpy as np

    arr = np.asarray(rgb, dtype=np.float32)
    h, w = arr.shape[0], arr.shape[1]
    grad = np.zeros((h, w), dtype=np.float32)
    grad[:, :-1] += np.mean(np.abs(arr[:, 1:, :] - arr[:, :-1, :]), axis=-1)
    grad[:-1, :] += np.mean(np.abs(arr[1:, :, :] - arr[:-1, :, :]), axis=-1)

    # Suppress 16px outer frame border and keep strong interior texture gradients
    grad[:16, :] = 0.0
    grad[-16:, :] = 0.0
    grad[:, :16] = 0.0
    grad[:, -16:] = 0.0

    thresh = max(12.0, float(np.percentile(grad, 90)))
    ys, xs = np.nonzero(grad >= thresh)
    assert len(xs) >= 50, "Failed to find textured object pixels on frame"
    cx = int(round(float(np.mean(xs))))
    cy = int(round(float(np.mean(ys))))
    x0 = int(round(float(np.percentile(xs, 2))))
    x1 = int(round(float(np.percentile(xs, 98))))
    y0 = int(round(float(np.percentile(ys, 2))))
    y1 = int(round(float(np.percentile(ys, 98))))
    pad = 12
    bx0 = max(0, x0 - pad)
    by0 = max(0, y0 - pad)
    bx1 = min(w, max(bx0 + 32, x1 + pad))
    by1 = min(h, max(by0 + 32, y1 + pad))
    return (cx, cy), (bx0, by0, bx1 - bx0, by1 - by0)


def _unwrap_single_benchmark_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the `VideoRunReport` dict from a single-preset `benchmark --json` payload."""
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


# ===========================================================================
# F5 — Realistic Multi-Scenario Demo Video & 3D Corpus (6 tests)
# ===========================================================================


@pytest.mark.f05
def test_f05_demo_assets_cli_generates_all_five_clips_and_json_manifest(e2e_tmp_dir: Path) -> None:
    out_dir = e2e_tmp_dir / "demo_corpus"
    proc, payload = run_cli_json(
        ["demo-assets", "--output-dir", str(out_dir)],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)

    clips = payload.get("clips")
    assert isinstance(clips, list) and len(clips) == 5, f"Expected 5 demo clips, got: {payload!r}"

    by_id = {str(c["id"]): c for c in clips if isinstance(c, dict) and "id" in c}
    assert set(by_id.keys()) == set(EXPECTED_DEMO_CLIP_IDS), (
        f"Expected clip IDs {EXPECTED_DEMO_CLIP_IDS}, got {sorted(by_id.keys())}"
    )

    required_clip_keys = {
        "id",
        "path",
        "width",
        "height",
        "fps",
        "frame_count",
        "default_selection",
        "recommended_wrap",
    }
    for cid in EXPECTED_DEMO_CLIP_IDS:
        entry = by_id[cid]
        missing = required_clip_keys - set(entry.keys())
        assert not missing, f"Clip {cid} manifest entry missing keys: {missing}"
        clip_path = Path(str(entry["path"]))
        assert clip_path.is_file() and clip_path.stat().st_size > 10_000, (
            f"Demo clip {cid} missing or too small on disk: {clip_path}"
        )

    assert (out_dir / "mesh_box.ply").is_file(), "Expected mesh_box.ply alongside demo clips"
    assert (out_dir / "mesh_box_meta.json").is_file(), "Expected mesh_box_meta.json alongside demo clips"


@pytest.mark.f05
def test_f05_demo_clips_have_valid_h264_streams_dimensions_and_30fps(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    assert demo_corpus_dir.is_dir()
    for cid in EXPECTED_DEMO_CLIP_IDS:
        assert cid in demo_corpus_manifest, f"Missing {cid} in demo_corpus_manifest"
        clip_info = demo_corpus_manifest[cid]
        probed = probe_video(Path(str(clip_info["path"])), tmp_dir=e2e_tmp_dir)

        assert probed["codec_name"] == "h264", f"{cid}: expected h264 codec, got {probed['codec_name']!r}"
        assert probed["fps"] >= 29.9, f"{cid}: expected >= 29.9 FPS, got {probed['fps']}"
        expected_dims = (1280, 720) if cid == "planar_hd_1280x720" else (640, 480)
        assert (probed["width"], probed["height"]) == expected_dims, (
            f"{cid}: expected dimensions {expected_dims}, got {(probed['width'], probed['height'])}"
        )
        assert probed["frame_count"] >= 60, f"{cid}: expected >= 60 frames, got {probed['frame_count']}"
        assert probed["frame_count"] == int(clip_info["frame_count"]), (
            f"{cid}: ffprobe frame_count ({probed['frame_count']}) != manifest ({clip_info['frame_count']})"
        )


@pytest.mark.f05
def test_f05_demo_clips_contain_genuine_inter_frame_motion_and_texture(
    demo_corpus_manifest: dict[str, dict[str, Any]],
) -> None:
    import numpy as np

    sample_indices = [0, 15, 30, 45]
    for cid in ("planar_card_640x480", "cylinder_mug_640x480"):
        clip_info = demo_corpus_manifest[cid]
        frames = decode_video_frames(Path(str(clip_info["path"])), sample_indices)
        assert set(frames.keys()) == set(sample_indices), f"{cid}: failed decoding sample frames"

        f0 = frames[0].astype(np.float32)
        for t_idx in (15, 30, 45):
            ft = frames[t_idx].astype(np.float32)
            mad = float(np.mean(np.abs(ft - f0)))
            assert mad > 2.0, f"{cid}: static/frozen video between frame 0 and {t_idx} (MAD={mad:.3f})"

        _, (bx, by, bw, bh), _ = _extract_selection_geometry(clip_info)
        roi_0 = frames[0][by : by + bh, bx : bx + bw]
        grad_energy = _spatial_gradient_energy(roi_0)
        assert grad_energy > 6.0, (
            f"{cid}: target ROI lacks trackable spatial texture (grad_energy={grad_energy:.2f})"
        )


@pytest.mark.f05
def test_f05_occlusion_demo_clip_has_distinct_occluded_and_reappeared_phases(
    demo_corpus_manifest: dict[str, dict[str, Any]],
) -> None:
    import numpy as np

    clip_info = demo_corpus_manifest["occlusion_recovery_640x480"]
    n_frames = int(clip_info["frame_count"])
    early_idx = max(0, int(0.08 * n_frames))
    # Scan a handful of middle candidates in [0.35..0.65] to verify a genuine occluded phase exists
    mid_candidates = sorted({int(frac * n_frames) for frac in (0.40, 0.48, 0.50, 0.55, 0.60)})
    late_idx = min(n_frames - 1, int(0.84 * n_frames))

    wanted = sorted({early_idx, late_idx, *mid_candidates})
    frames = decode_video_frames(Path(str(clip_info["path"])), wanted)

    _, (bx, by, bw, bh), _ = _extract_selection_geometry(clip_info)
    roi_early = frames[early_idx][by : by + bh, bx : bx + bw]
    roi_late = frames[late_idx][by : by + bh, bx : bx + bw]
    early_grad = _spatial_gradient_energy(roi_early)
    late_grad = _spatial_gradient_energy(roi_late)
    assert early_grad > 6.0, f"Expected textured target in early phase (frame {early_idx}), got {early_grad:.2f}"
    assert late_grad > 6.0, f"Expected textured target to reappear in late phase (frame {late_idx}), got {late_grad:.2f}"

    best_mid_idx = min(
        mid_candidates,
        key=lambda idx: _spatial_gradient_energy(frames[idx][by : by + bh, bx : bx + bw]),
    )
    roi_mid = frames[best_mid_idx][by : by + bh, bx : bx + bw]
    mid_grad = _spatial_gradient_energy(roi_mid)

    mad_early_mid = float(np.mean(np.abs(roi_mid.astype(np.float32) - roi_early.astype(np.float32))))
    mad_late_mid = float(np.mean(np.abs(roi_late.astype(np.float32) - roi_mid.astype(np.float32))))
    assert mid_grad < 0.55 * early_grad, (
        f"Expected middle occlusion phase to suppress target texture: mid_grad={mid_grad:.2f} vs early={early_grad:.2f}"
    )
    assert mad_early_mid > 12.0 and mad_late_mid > 12.0, (
        f"Occlusion clip lacks distinct visual transition: mad_early_mid={mad_early_mid:.2f}, mad_late_mid={mad_late_mid:.2f}"
    )


@pytest.mark.f05
def test_f05_mesh_box_ply_and_ground_truth_meta_json_valid(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
) -> None:
    import numpy as np

    ply_path = demo_corpus_dir / "mesh_box.ply"
    meta_path = demo_corpus_dir / "mesh_box_meta.json"

    ply_data = _parse_ascii_ply(ply_path)
    verts = ply_data["vertices"]
    extents = np.ptp(verts, axis=0)
    assert np.all(extents > 1e-3), f"Expected non-degenerate 3D box extents, got {extents}"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fx, fy, cx, cy = _extract_intrinsics_from_meta(meta)
    assert fx > 100.0 and fy > 100.0
    assert 0.0 < cx < 640.0 and 0.0 < cy < 480.0

    poses = _extract_gt_poses_list(meta)
    expected_frames = int(demo_corpus_manifest["mesh_box_640x480"]["frame_count"])
    assert len(poses) == expected_frames, (
        f"Expected {expected_frames} GT poses in mesh_box_meta.json, got {len(poses)}"
    )
    for sample_i in (0, len(poses) // 2, len(poses) - 1):
        r_mat, tvec = _pose_entry_to_rt(poses[sample_i])
        det_r = float(np.linalg.det(r_mat))
        assert abs(det_r - 1.0) < 1e-2, f"Non-orthonormal rotation matrix at frame {sample_i}: det={det_r}"
        assert tvec[2] > 0.05, f"Expected box in front of camera (tz > 0.05m), got {tvec}"


@pytest.mark.f05
def test_f05_demo_assets_generation_is_deterministic_across_runs(e2e_tmp_dir: Path) -> None:
    dir_a = e2e_tmp_dir / "run_a"
    dir_b = e2e_tmp_dir / "run_b"

    _, man_a = run_cli_json(["demo-assets", "--output-dir", str(dir_a)], timeout=90.0, tmp_dir=e2e_tmp_dir)
    _, man_b = run_cli_json(["demo-assets", "--output-dir", str(dir_b)], timeout=90.0, tmp_dir=e2e_tmp_dir)

    clips_a = {c["id"]: c for c in man_a["clips"]}
    clips_b = {c["id"]: c for c in man_b["clips"]}
    assert set(clips_a.keys()) == set(clips_b.keys()) == set(EXPECTED_DEMO_CLIP_IDS)

    for cid in EXPECTED_DEMO_CLIP_IDS:
        ca, cb = clips_a[cid], clips_b[cid]
        for field in ("width", "height", "fps", "frame_count", "default_selection", "recommended_wrap"):
            assert ca[field] == cb[field], f"{cid} field {field!r} differed across runs: {ca[field]!r} vs {cb[field]!r}"

        f0_a = decode_video_frames(Path(str(ca["path"])), [0])[0]
        f0_b = decode_video_frames(Path(str(cb["path"])), [0])[0]
        psnr = compute_psnr(f0_a, f0_b)
        assert psnr > 40.0, f"{cid} frame-0 not deterministic across runs (PSNR={psnr:.2f} dB)"

    ply_a = _parse_ascii_ply(dir_a / "mesh_box.ply")
    ply_b = _parse_ascii_ply(dir_b / "mesh_box.ply")
    assert ply_a["num_vertices"] == ply_b["num_vertices"]
    assert ply_a["num_faces"] == ply_b["num_faces"]
    assert (dir_a / "mesh_box.ply").read_bytes() == (dir_b / "mesh_box.ply").read_bytes()


# ===========================================================================
# F6 — Fast Point & Box Object Segmentation Onboarding (6 tests)
# ===========================================================================


@pytest.mark.f06
def test_f06_positive_point_segmentation_extracts_foreground_mask(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    clip = demo_corpus_manifest["planar_card_640x480"]
    (cx, cy), _, _ = _extract_selection_geometry(clip)
    mask_path = e2e_tmp_dir / "pos_point_mask.png"

    proc, data = run_cli_json(
        [
            "segment",
            "--input",
            str(clip["path"]),
            "--frame",
            "0",
            "--point",
            f"{cx},{cy}",
            "--output-mask",
            str(mask_path),
        ],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    _assert_segment_cli_schema(data, expected_w=640, expected_h=480, expected_frame=0)

    cov = float(data["coverage_ratio"])
    assert 0.03 <= cov <= 0.85, f"Expected foreground coverage_ratio in [0.03, 0.85], got {cov:.4f}"
    assert int(data["anchor_count"]) >= 8, f"Expected >= 8 interior anchors, got {data['anchor_count']}"
    assert Path(str(data["mask_path"])).resolve() == mask_path.resolve()
    assert mask_path.is_file()

    mask_arr, _ = decode_image_bytes(mask_path.read_bytes())
    mask_2d = mask_arr[..., 0] if mask_arr.ndim == 3 else mask_arr
    assert mask_2d.shape == (480, 640), f"Expected (480, 640) mask PNG, got {mask_2d.shape}"
    assert int(mask_2d[cy, cx]) > 0, f"Expected positive foreground mask intensity at clicked point ({cx}, {cy})"


@pytest.mark.f06
def test_f06_bounding_box_segmentation_constrains_mask_to_box(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    import numpy as np

    clip = demo_corpus_manifest["planar_card_640x480"]
    _, (bx, by, bw, bh), _ = _extract_selection_geometry(clip)
    mask_path = e2e_tmp_dir / "bbox_mask.png"

    proc, data = run_cli_json(
        [
            "segment",
            "--input",
            str(clip["path"]),
            "--frame",
            "0",
            "--box",
            f"{bx},{by},{bw},{bh}",
            "--output-mask",
            str(mask_path),
        ],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    _assert_segment_cli_schema(data, expected_w=640, expected_h=480, expected_frame=0)

    rx, ry, rw, rh = _parse_bbox_xywh(data["bbox"])
    pad = 16.0
    assert rx >= bx - pad and ry >= by - pad, f"Returned bbox {(rx, ry, rw, rh)} escaped top-left of {(bx, by, bw, bh)}"
    assert (rx + rw) <= (bx + bw + pad) and (ry + rh) <= (by + bh + pad), (
        f"Returned bbox {(rx, ry, rw, rh)} escaped bottom-right of {(bx, by, bw, bh)}"
    )

    mask_arr, _ = decode_image_bytes(mask_path.read_bytes())
    mask_2d = mask_arr[..., 0] if mask_arr.ndim == 3 else mask_arr
    outside = np.ones((480, 640), dtype=bool)
    margin = 16
    outside[max(0, by - margin) : min(480, by + bh + margin), max(0, bx - margin) : min(640, bx + bw + margin)] = False
    assert int(np.max(mask_2d[outside])) == 0, (
        "Pixels >16px outside --box must be strictly 0 in segmentation output mask"
    )


@pytest.mark.f06
def test_f06_positive_plus_negative_points_carve_out_excluded_region(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    import numpy as np

    clip = demo_corpus_manifest["planar_card_640x480"]
    (cx, cy), _, _ = _extract_selection_geometry(clip)
    mask1_path = e2e_tmp_dir / "mask_pos_only.png"
    mask2_path = e2e_tmp_dir / "mask_pos_and_neg.png"

    _, data1 = run_cli_json(
        [
            "segment",
            "--input",
            str(clip["path"]),
            "--frame",
            "0",
            "--point",
            f"{cx},{cy}",
            "--output-mask",
            str(mask1_path),
        ],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    _assert_segment_cli_schema(data1, expected_w=640, expected_h=480, expected_frame=0)
    m1_raw, _ = decode_image_bytes(mask1_path.read_bytes())
    m1 = m1_raw[..., 0] if m1_raw.ndim == 3 else m1_raw

    # Pick a strong foreground pixel in run 1 that sits well away (>=40px) from the positive point
    fg_ys, fg_xs = np.nonzero(m1 > 160)
    dists = np.hypot(fg_xs - cx, fg_ys - cy)
    far_candidates = np.nonzero(dists >= 40.0)[0]
    assert len(far_candidates) > 0, "Expected foreground region wide enough to place a distinct negative point"
    pick = int(far_candidates[np.argmax(dists[far_candidates])])
    nx, ny = int(fg_xs[pick]), int(fg_ys[pick])

    _, data2 = run_cli_json(
        [
            "segment",
            "--input",
            str(clip["path"]),
            "--frame",
            "0",
            "--point",
            f"{cx},{cy}",
            "--neg-point",
            f"{nx},{ny}",
            "--output-mask",
            str(mask2_path),
        ],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    _assert_segment_cli_schema(data2, expected_w=640, expected_h=480, expected_frame=0)
    m2_raw, _ = decode_image_bytes(mask2_path.read_bytes())
    m2 = m2_raw[..., 0] if m2_raw.ndim == 3 else m2_raw

    assert int(m2[ny, nx]) < int(m1[ny, nx]) or int(data2["foreground_pixels"]) < int(data1["foreground_pixels"]), (
        f"Expected --neg-point ({nx},{ny}) to carve out foreground: "
        f"mask1[ny,nx]={int(m1[ny, nx])} -> mask2[ny,nx]={int(m2[ny, nx])}, "
        f"fg1={data1['foreground_pixels']} -> fg2={data2['foreground_pixels']}"
    )


@pytest.mark.f06
def test_f06_non_zero_frame_index_segmentation_tracks_moved_object(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    clip = demo_corpus_manifest["planar_card_640x480"]
    clip_path = Path(str(clip["path"]))
    decoded = decode_video_frames(clip_path, [0, 30])
    (cx0, cy0), _ = _locate_textured_object_on_frame(decoded[0])
    (cx30, cy30), _ = _locate_textured_object_on_frame(decoded[30])
    assert math.hypot(cx30 - cx0, cy30 - cy0) >= 8.0, "Expected card center on frame 30 to have moved vs frame 0"

    _, d0 = run_cli_json(
        ["segment", "--input", str(clip_path), "--frame", "0", "--point", f"{cx0},{cy0}"],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    _, d30 = run_cli_json(
        ["segment", "--input", str(clip_path), "--frame", "30", "--point", f"{cx30},{cy30}"],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    _assert_segment_cli_schema(d0, expected_w=640, expected_h=480, expected_frame=0)
    _assert_segment_cli_schema(d30, expected_w=640, expected_h=480, expected_frame=30)

    bx0, by0, bw0, bh0 = _parse_bbox_xywh(d0["bbox"])
    bx30, by30, bw30, bh30 = _parse_bbox_xywh(d30["bbox"])
    seg_c0 = (bx0 + 0.5 * bw0, by0 + 0.5 * bh0)
    seg_c30 = (bx30 + 0.5 * bw30, by30 + 0.5 * bh30)

    shift = math.hypot(seg_c30[0] - seg_c0[0], seg_c30[1] - seg_c0[1])
    err_30 = math.hypot(seg_c30[0] - cx30, seg_c30[1] - cy30)
    assert shift >= 6.0, f"Expected segmented bbox center to shift on frame 30 vs frame 0 (shift={shift:.2f}px)"
    assert err_30 <= 70.0, f"Frame-30 segmented bbox center {seg_c30} far from visual object center {(cx30, cy30)}"


@pytest.mark.f06
def test_f06_segmentation_onboarding_latency_within_cpu_budget(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    clip = demo_corpus_manifest["planar_card_640x480"]
    (cx, cy), (bx, by, bw, bh), _ = _extract_selection_geometry(clip)
    mask_path = e2e_tmp_dir / "latency_mask.png"

    _, data = run_cli_json(
        [
            "segment",
            "--input",
            str(clip["path"]),
            "--frame",
            "0",
            "--point",
            f"{cx},{cy}",
            "--box",
            f"{bx},{by},{bw},{bh}",
            "--output-mask",
            str(mask_path),
        ],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    _assert_segment_cli_schema(data, expected_w=640, expected_h=480, expected_frame=0)

    init_ms = float(data["init_ms"])
    assert 0.0 < init_ms < 250.0, f"Expected GrabCut onboarding init_ms < 250.0 ms on CPU, got {init_ms:.2f} ms"

    polygon = data["polygon"]
    assert isinstance(polygon, list) and len(polygon) >= 3
    for pt in polygon:
        assert isinstance(pt, (list, tuple)) and len(pt) == 2
        px, py = float(pt[0]), float(pt[1])
        assert math.isfinite(px) and math.isfinite(py)
        assert -1.0 <= px <= 641.0 and -1.0 <= py <= 481.0


@pytest.mark.f06
def test_f06_rest_api_select_returns_mask_base64_and_selection_id(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    clip_path = e2e_tmp_dir / "select_upload.mp4"
    clip_meta = make_textured_tracking_clip(clip_path, width=640, height=480, frames=15, tmp_dir=e2e_tmp_dir)
    up_resp = live_studio_server.upload(
        "/api/videos/upload",
        filename="select_upload.mp4",
        content=clip_path.read_bytes(),
        content_type="video/mp4",
        expected_status=200,
    )
    video_id = up_resp["json"]["video_id"]

    cx, cy = clip_meta["center_frame0"]
    bx, by, bw, bh = clip_meta["bbox_frame0"]
    sel_json = live_studio_server.post_json(
        f"/api/videos/{video_id}/select",
        {
            "frame_index": 0,
            "positive_points": [[cx, cy]],
            "negative_points": [],
            "bbox": [bx, by, bw, bh],
        },
        expected_status=200,
    )

    missing = [k for k in REQUIRED_SELECT_REST_KEYS if k not in sel_json]
    assert not missing, f"POST /api/videos/{{id}}/select missing keys {missing}: {sorted(sel_json.keys())}"
    assert isinstance(sel_json["selection_id"], str) and sel_json["selection_id"].strip()
    assert int(sel_json["frame_index"]) == 0
    assert int(sel_json["foreground_pixels"]) > 0

    b64_str = str(sel_json["mask_png_base64"])
    if "," in b64_str and b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    png_bytes = base64.b64decode(b64_str)
    mask_arr, _ = decode_image_bytes(png_bytes)
    mask_2d = mask_arr[..., 0] if mask_arr.ndim == 3 else mask_arr
    assert mask_2d.shape == (480, 640), f"Expected (480, 640) decoded REST mask, got {mask_2d.shape}"
    assert int(mask_2d[cy, cx]) > 0, "Expected non-zero foreground in REST mask at selected center"


# ===========================================================================
# F7 — Real-Time KLT + ORB Homography & Grid Video Tracker (6 tests)
# ===========================================================================


@pytest.mark.f07
def test_f07_planar_card_tracking_sustains_high_availability_and_inliers(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    clip = demo_corpus_manifest["planar_card_640x480"]
    out_mp4 = e2e_tmp_dir / "planar_tracked.mp4"

    proc, report = run_cli_json(
        [
            "process",
            "--input",
            str(clip["path"]),
            "--output",
            str(out_mp4),
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "normal",
            "--pattern",
            "stripes",
            *_selection_cli_args(clip),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)

    assert float(report["tracking_availability"]) >= 0.90, (
        f"Expected planar tracking_availability >= 0.90, got {report['tracking_availability']}"
    )
    assert float(report["inlier_count_median"]) >= 12.0, (
        f"Expected inlier_count_median >= 12, got {report['inlier_count_median']}"
    )
    assert int(report["lost_frame_count"]) == 0, (
        f"Expected 0 lost frames on planar_card_640x480, got {report['lost_frame_count']}"
    )

    probed = probe_video(out_mp4, tmp_dir=e2e_tmp_dir)
    assert probed["frame_count"] == int(report["input_frame_count"])

    _, bbox0, _ = _extract_selection_geometry(clip)
    raw0 = decode_video_frames(Path(str(clip["path"])), [0])[0]
    out0 = decode_video_frames(out_mp4, [0])[0]
    roi_delta, bg_delta = compute_roi_and_bg_delta(raw0, out0, bbox0, margin=28)
    assert roi_delta >= 8.0, f"Expected strong foreground overlay delta on planar card, got {roi_delta:.2f}"
    assert bg_delta <= 3.5, f"Expected background preserved outside card ROI, got bg_delta={bg_delta:.2f}"


@pytest.mark.f07
def test_f07_hd_1280x720_tracking_with_resize_first_360_and_480(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    clip = demo_corpus_manifest["planar_hd_1280x720"]
    sel_args = _selection_cli_args(clip)

    for th in (360, 480):
        out_mp4 = e2e_tmp_dir / f"hd_track_{th}.mp4"
        proc, report = run_cli_json(
            [
                "process",
                "--input",
                str(clip["path"]),
                "--output",
                str(out_mp4),
                "--wrap-mode",
                "planar",
                "--tracking-height",
                str(th),
                *sel_args,
            ],
            timeout=120.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_no_python_traceback(proc)
        assert_video_run_report_invariants(report)
        assert (report["width"], report["height"]) == (1280, 720)
        assert float(report["tracking_availability"]) >= 0.90, (
            f"HD tracking_height={th} availability={report['tracking_availability']} < 0.90"
        )
        track_ms = float(report["stage_latency_mean_ms"]["track_ms"])
        assert track_ms <= 25.0, f"HD tracking_height={th} track_ms={track_ms:.2f} > 25.0 ms budget"


@pytest.mark.f07
def test_f07_cylindrical_mug_tracking_maintains_lock_across_pan_and_yaw(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    clip = demo_corpus_manifest["cylinder_mug_640x480"]
    out_mp4 = e2e_tmp_dir / "cylinder_tracked.mp4"

    proc, report = run_cli_json(
        [
            "process",
            "--input",
            str(clip["path"]),
            "--output",
            str(out_mp4),
            "--wrap-mode",
            "cylinder",
            *_selection_cli_args(clip),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)
    assert report["wrap_mode"] == "cylinder"
    assert float(report["tracking_availability"]) >= 0.85
    assert int(report["tracked_frame_count"]) >= int(0.85 * int(report["input_frame_count"]))
    assert probe_video(out_mp4, tmp_dir=e2e_tmp_dir)["frame_count"] == int(report["input_frame_count"])


@pytest.mark.f07
def test_f07_deformable_mesh_grid_mode_tracks_surface_smoothly(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    import numpy as np

    clip = demo_corpus_manifest["planar_card_640x480"]
    out_mp4 = e2e_tmp_dir / "mesh_grid_tracked.mp4"

    proc, report = run_cli_json(
        [
            "process",
            "--input",
            str(clip["path"]),
            "--output",
            str(out_mp4),
            "--wrap-mode",
            "mesh_grid",
            "--blend-mode",
            "normal",
            "--pattern",
            "checker",
            "--opacity",
            "0.95",
            *_selection_cli_args(clip),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)
    assert report["wrap_mode"] == "mesh_grid"
    assert float(report["tracking_availability"]) >= 0.90

    check_indices = list(range(0, 14))
    raw_frames = decode_video_frames(Path(str(clip["path"])), check_indices)
    out_frames = decode_video_frames(out_mp4, check_indices)

    centroids: list[tuple[float, float]] = []
    for idx in check_indices:
        diff = np.max(np.abs(out_frames[idx].astype(np.float32) - raw_frames[idx].astype(np.float32)), axis=-1)
        ys, xs = np.nonzero(diff > 15.0)
        assert len(xs) >= 100, f"Expected rendered mesh_grid overlay on frame {idx}, got {len(xs)} changed pixels"
        centroids.append((float(np.mean(xs)), float(np.mean(ys))))

    max_step_px = max(
        math.hypot(centroids[i][0] - centroids[i - 1][0], centroids[i][1] - centroids[i - 1][1])
        for i in range(1, len(centroids))
    )
    assert max_step_px < 35.0, (
        f"Detected sudden inter-frame overlay teleport spike in mesh_grid mode (max_step={max_step_px:.2f}px)"
    )


@pytest.mark.f07
def test_f07_tracker_is_causal_and_deterministic_on_identical_clip(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    clip = demo_corpus_manifest["planar_card_640x480"]
    common_args = [
        "--input",
        str(clip["path"]),
        "--wrap-mode",
        "planar",
        "--threads",
        "1",
        "--pattern",
        "stripes",
        *_selection_cli_args(clip),
    ]
    out1 = e2e_tmp_dir / "det_run1.mp4"
    out2 = e2e_tmp_dir / "det_run2.mp4"

    _, rep1 = run_cli_json(["process", *common_args, "--output", str(out1)], timeout=90.0, tmp_dir=e2e_tmp_dir)
    _, rep2 = run_cli_json(["process", *common_args, "--output", str(out2)], timeout=90.0, tmp_dir=e2e_tmp_dir)

    assert_video_run_report_invariants(rep1)
    assert_video_run_report_invariants(rep2)
    assert rep1["tracked_frame_count"] == rep2["tracked_frame_count"]
    assert abs(float(rep1["inlier_count_median"]) - float(rep2["inlier_count_median"])) <= 2.0


@pytest.mark.f07
def test_f07_select_frame_mid_video_tracks_remaining_or_full_clip_cleanly(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    clip = demo_corpus_manifest["planar_card_640x480"]
    clip_path = Path(str(clip["path"]))
    f10 = decode_video_frames(clip_path, [10])[10]
    (cx10, cy10), (bx10, by10, bw10, bh10) = _locate_textured_object_on_frame(f10)

    out_mp4 = e2e_tmp_dir / "select_frame_10.mp4"
    proc, report = run_cli_json(
        [
            "process",
            "--input",
            str(clip_path),
            "--output",
            str(out_mp4),
            "--select-frame",
            "10",
            "--point",
            f"{cx10},{cy10}",
            "--box",
            f"{bx10},{by10},{bw10},{bh10}",
            "--wrap-mode",
            "planar",
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)

    probed = probe_video(out_mp4, tmp_dir=e2e_tmp_dir)
    assert probed["frame_count"] == int(report["input_frame_count"])
    min_expected_tracked = int(0.80 * max(1, int(report["input_frame_count"]) - 10))
    assert int(report["tracked_frame_count"]) >= min_expected_tracked, (
        f"Expected >= {min_expected_tracked} tracked frames when selecting at frame 10, got {report['tracked_frame_count']}"
    )


# ===========================================================================
# F8 — Calibrated 6-DoF 3D Mesh Fast Tracker (6 tests)
# ===========================================================================


@pytest.mark.f08
def test_f08_mesh_6dof_tracks_demo_ply_box_with_low_reprojection_rmse(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    clip = demo_corpus_manifest["mesh_box_640x480"]
    ply_path = demo_corpus_dir / "mesh_box.ply"
    meta = json.loads((demo_corpus_dir / "mesh_box_meta.json").read_text(encoding="utf-8"))
    intr = _extract_intrinsics_from_meta(meta)
    out_mp4 = e2e_tmp_dir / "mesh_6dof_ply.mp4"

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
            str(ply_path),
            "--intrinsics",
            _format_intrinsics_arg(intr),
            *_selection_cli_args(clip),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)

    assert report["wrap_mode"] == "mesh_6dof"
    assert float(report["tracking_availability"]) >= 0.85, (
        f"Expected mesh_6dof tracking_availability >= 0.85, got {report['tracking_availability']}"
    )
    rmse = report["reprojection_rmse_px"]
    assert rmse is not None, "Expected non-None reprojection_rmse_px in mesh_6dof mode"
    assert 0.0 <= float(rmse) < 5.0, f"Expected reprojection_rmse_px < 5.0 px, got {rmse}"


@pytest.mark.f08
def test_f08_parametric_plane_shape_supported_in_mesh_6dof(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    clip = demo_corpus_manifest["planar_card_640x480"]
    out_mp4 = e2e_tmp_dir / "mesh_6dof_plane.mp4"

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
            "plane",
            "--intrinsics",
            "550,550,320,240",
            *_selection_cli_args(clip),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)
    assert report["wrap_mode"] == "mesh_6dof"
    assert int(report["tracked_frame_count"]) > 0
    assert probe_video(out_mp4, tmp_dir=e2e_tmp_dir)["codec_name"] == "h264"


@pytest.mark.f08
def test_f08_parametric_cylinder_shape_supported_in_mesh_6dof(
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    clip = demo_corpus_manifest["cylinder_mug_640x480"]
    out_mp4 = e2e_tmp_dir / "mesh_6dof_cylinder.mp4"

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
            "550,550,320,240",
            "--blend-mode",
            "normal",
            "--pattern",
            "stripes",
            "--opacity",
            "0.95",
            *_selection_cli_args(clip),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)
    assert report["wrap_mode"] == "mesh_6dof"
    assert int(report["tracked_frame_count"]) > 0

    _, bbox0, _ = _extract_selection_geometry(clip)
    raw0 = decode_video_frames(Path(str(clip["path"])), [0])[0]
    out0 = decode_video_frames(out_mp4, [0])[0]
    roi_delta, _ = compute_roi_and_bg_delta(raw0, out0, bbox0, margin=24)
    assert roi_delta >= 4.0, f"Expected visible parametric cylinder 6-DoF overlay on mug, got {roi_delta:.2f}"


@pytest.mark.f08
def test_f08_parametric_box_and_curved_panel_shapes_supported(e2e_tmp_dir: Path) -> None:
    src_mp4 = e2e_tmp_dir / "short_src.mp4"
    clip_meta = make_textured_tracking_clip(src_mp4, width=640, height=480, frames=24, tmp_dir=e2e_tmp_dir)
    bx, by, bw, bh = clip_meta["bbox_frame0"]
    cx, cy = clip_meta["center_frame0"]

    for shape_name in ("box", "curved_panel"):
        out_mp4 = e2e_tmp_dir / f"mesh_6dof_{shape_name}.mp4"
        proc, report = run_cli_json(
            [
                "process",
                "--input",
                str(src_mp4),
                "--output",
                str(out_mp4),
                "--wrap-mode",
                "mesh_6dof",
                "--mesh",
                shape_name,
                "--intrinsics",
                "550,550,320,240",
                "--point",
                f"{cx},{cy}",
                "--box",
                f"{bx},{by},{bw},{bh}",
            ],
            timeout=60.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_no_python_traceback(proc)
        assert_video_run_report_invariants(report)
        assert report["wrap_mode"] == "mesh_6dof"
        probed = probe_video(out_mp4, tmp_dir=e2e_tmp_dir)
        assert probed["codec_name"] == "h264"
        assert probed["frame_count"] == 24 == int(report["rendered_frame_count"])


@pytest.mark.f08
def test_f08_mesh_6dof_sustains_at_least_15_fps_end_to_end(e2e_tmp_dir: Path) -> None:
    bench_out = e2e_tmp_dir / "bench_mesh_6dof"
    report_path = e2e_tmp_dir / "bench_mesh_6dof_report.json"

    proc, payload = run_cli_json(
        [
            "benchmark",
            "--preset",
            "mesh_6dof",
            "--min-fps",
            "15.0",
            "--output-dir",
            str(bench_out),
            "--report",
            str(report_path),
        ],
        timeout=120.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    report = _unwrap_single_benchmark_report(payload)
    assert_video_run_report_invariants(report, require_min_fps=15.0)
    assert report["wrap_mode"] == "mesh_6dof"
    assert float(report["stage_latency_mean_ms"]["track_ms"]) < 30.0, (
        f"Expected mesh_6dof track_ms < 30.0 ms, got {report['stage_latency_mean_ms']['track_ms']:.2f} ms"
    )
    assert report_path.is_file()


@pytest.mark.f08
def test_f08_mesh_6dof_overlay_attaches_to_moving_3d_box_roi(
    demo_corpus_dir: Path,
    demo_corpus_manifest: dict[str, dict[str, Any]],
    e2e_tmp_dir: Path,
) -> None:
    import numpy as np

    clip = demo_corpus_manifest["mesh_box_640x480"]
    ply_path = demo_corpus_dir / "mesh_box.ply"
    ply_data = _parse_ascii_ply(ply_path)
    meta = json.loads((demo_corpus_dir / "mesh_box_meta.json").read_text(encoding="utf-8"))
    fx, fy, cx, cy = _extract_intrinsics_from_meta(meta)
    gt_poses = _extract_gt_poses_list(meta)

    out_mp4 = e2e_tmp_dir / "mesh_6dof_attached.mp4"
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
            str(ply_path),
            "--intrinsics",
            _format_intrinsics_arg((fx, fy, cx, cy)),
            "--pattern",
            "checker",
            "--colors",
            "#ff006e,#00f5d4",
            "--blend-mode",
            "normal",
            "--opacity",
            "0.95",
            *_selection_cli_args(clip),
        ],
        timeout=90.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc)
    assert_video_run_report_invariants(report)

    sample_frames = [0, 25, 50]
    raw_frames = decode_video_frames(Path(str(clip["path"])), sample_frames)
    out_frames = decode_video_frames(out_mp4, sample_frames)

    verts_obj = ply_data["vertices"].astype(np.float64)
    # Handle PLY files stored in millimeters (like legacy bench/prototype_fixture.py) vs meters
    if float(np.max(np.abs(verts_obj))) > 10.0:
        verts_obj = verts_obj * 1e-3

    for f_idx in sample_frames:
        r_mat, tvec = _pose_entry_to_rt(gt_poses[f_idx])
        tvec_m = tvec * 1e-3 if float(np.linalg.norm(tvec)) > 10.0 else tvec
        verts_cam = (verts_obj @ r_mat.T + tvec_m[None, :]).astype(np.float32)

        oracle = ray_triangle_uv_oracle(
            verts_cam,
            ply_data["triangles"],
            ply_data["uvs"],
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            frame_w=640,
            frame_h=480,
        )
        gt_mask = oracle["hit_mask"]
        assert int(np.count_nonzero(gt_mask)) >= 500, f"Empty GT projected 3D box mask at frame {f_idx}"

        # Dilate GT silhouette by 8px to tolerate H.264 4:2:0 chroma block bleed and <5px PnP pose tolerance
        from scipy.ndimage import binary_dilation  # lazy import inside function

        gt_region = binary_dilation(gt_mask, iterations=8)

        diff = np.max(np.abs(out_frames[f_idx].astype(np.float32) - raw_frames[f_idx].astype(np.float32)), axis=-1)
        changed = diff > 18.0
        changed_count = int(np.count_nonzero(changed))
        assert changed_count >= 400, (
            f"Expected >= 400 modified overlay pixels on 3D box at frame {f_idx}, got {changed_count}"
        )
        overlap_ratio = float(np.count_nonzero(changed & gt_region)) / float(changed_count)
        assert overlap_ratio >= 0.80, (
            f"Frame {f_idx}: only {overlap_ratio:.2%} of mesh_6dof overlay pixels fell inside GT 3D box silhouette"
        )
