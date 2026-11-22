#!/usr/bin/env python3
"""Rigorous Dev visual-quality evaluation & controlled ablation harness.

Key design guarantees:
  - Uses VideoPipelineSession directly to capture exact per-frame `rendered_mask` (eliminating
    the dark-ink luminance-blend artifact in erosion/leakage scoring) AND verifies decoded visibility.
  - Reports all-frame metrics with missing AND incorrect overlays in the denominators:
      * f0_init_ok (wrong-object initialization detector on frame 0)
      * attached_visible_rate (requires rendered mask + aligned landmarks/centroid + low leakage + coverage)
      * false_confident_rate (rendering when misaligned, wrong-object, or heavily leaking)
      * wrong_object_rate (centroid outside GT box or >50% paint outside object)
      * fg_leakage_rate, bg_leakage_rate, target_erosion_rate, false_occlusion_rate, temporal_flicker
      * landmark_err_720p_med, landmark_err_720p_p95 (exact GT on synthetic & GT SE(3) rigid motion on YCB)
      * correct_recovery_count, false_recovery_count, ambiguous/recovering/lost counts
  - Strictly separates `default_auto` vs `user_selection`, supports `--code-root` for exact
    apples-to-apples old-vs-new comparisons under identical inputs, and keeps Holdout sealed
    (unless explicitly requested with `--include-holdout-regression-only` for read-only verification).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Human-verified frame-0 selections for Dev diagnostic clips (click + tight box + optional negative points).
# Used ONLY when --selection-mode=user_selection (explicitly separated from automatic/default init).
DEV_USER_SELECTIONS: dict[str, dict[str, Any]] = {
    "show3d_mug_60fps": {
        "frame_index": 0,
        "positive_points": [[314.0, 596.0], [326.0, 588.0]],
        "negative_points": [[280.0, 580.0], [340.0, 615.0], [370.0, 595.0], [315.0, 648.0]],
        "bbox": [296.0, 566.0, 56.0, 74.0],
    },
    "show3d_ranch_bottle_60fps": {
        "frame_index": 0,
        "positive_points": [[260.0, 325.0], [272.0, 350.0]],
        "negative_points": [[230.0, 330.0], [305.0, 330.0], [285.0, 395.0]],
        "bbox": [238.0, 285.0, 56.0, 102.0],
    },
    "show3d_keyboard_60fps": {
        "frame_index": 0,
        "positive_points": [[312.0, 545.0], [300.0, 590.0]],
        "negative_points": [[245.0, 540.0], [365.0, 540.0], [290.0, 665.0]],
        "bbox": [266.0, 448.0, 96.0, 212.0],
    },
    "ycb_s52_window1": {
        "frame_index": 0,
        "positive_points": [[335.0, 185.0], [320.0, 255.0]],
        "negative_points": [[412.0, 275.0], [405.0, 335.0], [295.0, 345.0], [230.0, 180.0]],
        "bbox": [250.0, 82.0, 195.0, 255.0],
    },
    "ycb_s59_window1": {
        "frame_index": 0,
        "positive_points": [[165.0, 185.0], [185.0, 105.0]],
        "negative_points": [[265.0, 145.0], [290.0, 210.0], [240.0, 305.0]],
        "bbox": [118.0, 44.0, 188.0, 265.0],
    },
    "ycb_s50_window1": {
        "frame_index": 0,
        "positive_points": [[362.0, 315.0], [358.0, 385.0]],
        "negative_points": [[305.0, 320.0], [420.0, 320.0]],
        "bbox": [319.0, 202.0, 92.0, 253.0],
    },
    "hope_dyn_obj01_720p": {
        "frame_index": 0,
        "positive_points": [[616.0, 495.0], [608.0, 465.0]],
        "negative_points": [[525.0, 495.0], [705.0, 495.0]],
        "bbox": [553.0, 404.0, 127.0, 182.0],
    },
    "hope_dyn_obj02_720p": {
        "frame_index": 0,
        "positive_points": [[588.0, 446.0], [588.0, 385.0]],
        "negative_points": [[505.0, 446.0], [670.0, 446.0]],
        "bbox": [530.0, 312.0, 115.0, 268.0],
    },
}

# Diagnostic-only sparse keyframe GT boxes for Dev SHOW3D clips (evaluation ONLY; NEVER enters inference).
SHOW3D_DEV_KEYFRAME_BOXES: dict[str, dict[int, tuple[float, float, float, float]]] = {
    "show3d_mug_60fps": {
        0: (296.0, 566.0, 58.0, 76.0),
        30: (298.0, 424.0, 58.0, 72.0),
        60: (296.0, 444.0, 58.0, 70.0),
        90: (308.0, 448.0, 56.0, 62.0),
        120: (304.0, 456.0, 54.0, 62.0),
        150: (308.0, 452.0, 56.0, 66.0),
        179: (308.0, 434.0, 58.0, 66.0),
    },
    "show3d_ranch_bottle_60fps": {
        0: (238.0, 285.0, 58.0, 102.0),
        30: (246.0, 268.0, 64.0, 108.0),
        60: (258.0, 260.0, 56.0, 114.0),
        90: (254.0, 270.0, 58.0, 112.0),
        120: (252.0, 264.0, 68.0, 114.0),
        150: (250.0, 268.0, 60.0, 108.0),
        179: (248.0, 268.0, 56.0, 108.0),
    },
    "show3d_keyboard_60fps": {
        0: (266.0, 448.0, 96.0, 212.0),
        30: (282.0, 426.0, 76.0, 214.0),
        60: (282.0, 426.0, 78.0, 200.0),
        90: (274.0, 450.0, 96.0, 174.0),
        120: (272.0, 466.0, 98.0, 160.0),
        150: (274.0, 514.0, 70.0, 84.0),
        179: (206.0, 502.0, 128.0, 74.0),
    },
}


def _load_diagnostic_clips(*, include_holdout_controls: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    base_dir = REPO_ROOT / ".cache" / "benchmark-dataset"
    manifest = json.loads((base_dir / "manifest_v2.json").read_text())
    gt_bundle = json.loads((base_dir / "ground_truth_eval_v2.json").read_text())
    diag_manifest = json.loads((REPO_ROOT / ".cache" / "vq-diagnostics" / "diagnostics_manifest.json").read_text())

    real_ids = [
        "show3d_mug_60fps",
        "show3d_ranch_bottle_60fps",
        "show3d_keyboard_60fps",
        "ycb_s52_window1",
        "ycb_s50_window1",
        "ycb_s59_window1",
        "hope_dyn_obj01_720p",
        "hope_dyn_obj02_720p",
    ]
    by_id = {c["id"]: c for c in manifest["clips"]}
    clips: list[dict[str, Any]] = []
    for cid in real_ids:
        c = dict(by_id[cid])
        assert c["split"] == "dev", f"Refusing to run dev diagnostics on non-dev clip {cid} ({c['split']})"
        c["source_kind"] = "real"
        c["user_selection"] = DEV_USER_SELECTIONS[cid]
        clips.append(c)

    if include_holdout_controls:
        # Read-only regression control check (NEVER tune on these clips!)
        for cid in ("ycb_s49_window1",):
            c = dict(by_id[cid])
            c["source_kind"] = "real"
            c["user_selection"] = c["default_selection"]
            c["holdout_regression_only"] = True
            clips.append(c)

    for c in diag_manifest["clips"]:
        cc = dict(c)
        cc["source_kind"] = "syn_diag"
        cc["default_selection"] = cc["user_selection"]
        clips.append(cc)

    return clips, gt_bundle, base_dir


def _run_session_on_video(
    *,
    session: Any,
    raw_mp4: Path,
    sel: Any,
    strip_idxs: set[int],
    save_video_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[int, np.ndarray, np.ndarray]], float, tuple[int, int, int, int]]:
    """Run causal VideoPipelineSession across all frames, capturing exact rendered masks & states."""
    cap = cv2.VideoCapture(str(raw_mp4))
    fps_src = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames_records: list[dict[str, Any]] = []
    strip_frames: list[tuple[int, np.ndarray, np.ndarray]] = []
    writer = None

    t0 = time.perf_counter()
    f_idx = 0
    ref_bbox = (0, 0, 0, 0)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            ts = float(f_idx) / max(fps_src, 1.0)
            if f_idx == 0:
                res = session.initialize(frame, sel, timestamp_s=ts)
                ref_bbox = tuple(int(v) for v in session.tracker.ref_bbox)
                if save_video_path is not None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(save_video_path), fourcc, fps_src, (w, h))
            else:
                res = session.process_frame(frame, f_idx, ts)

            tr = res.track_result
            rmask = (res.rendered_mask > 0) if res.rendered_mask is not None else None
            # Verify visual output actually changed where rendered (on non-black target pixels)
            vis_diff = np.max(np.abs(res.frame_bgr.astype(np.int16) - frame.astype(np.int16)), axis=2) > 6
            frames_records.append({
                "frame_index": f_idx,
                "state": tr.state,
                "confidence_state": getattr(tr, "confidence_state", tr.state.upper()),
                "inliers": int(tr.inliers),
                "confidence": float(tr.confidence),
                "homography_3x3": tr.homography_3x3.copy() if tr.homography_3x3 is not None else None,
                "pose_rvec": tr.pose_rvec.copy() if tr.pose_rvec is not None else None,
                "pose_tvec": tr.pose_tvec.copy() if tr.pose_tvec is not None else None,
                "rendered_mask": rmask,
                "vis_diff": vis_diff,
            })
            if f_idx in strip_idxs:
                strip_frames.append((f_idx, frame.copy(), res.frame_bgr.copy()))
            if writer is not None:
                writer.write(res.frame_bgr)
            f_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = max(1e-6, time.perf_counter() - t0)
    eff_fps = float(f_idx) / elapsed
    return frames_records, strip_frames, eff_fps, ref_bbox


def _save_strip_from_frames(strip_frames: list[tuple[int, np.ndarray, np.ndarray]], out_jpg: Path) -> None:
    cols = []
    for k, fr, fo in strip_frames:
        h, w = fr.shape[:2]
        tw = int(round(w * 240.0 / float(h)))
        r_s = cv2.resize(fr, (tw, 240))
        o_s = cv2.resize(fo, (tw, 240))
        cv2.putText(r_s, f"f={k}", (6, 20), 0, 0.55, (0, 255, 255), 2)
        cols.append(np.vstack([r_s, o_s]))
    if cols:
        cv2.imwrite(str(out_jpg), np.hstack(cols))


def _score_syn_diag(
    clip: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    gt = np.load(clip["gt_npz"])
    kind = str(gt["kind"])
    target_vis = gt["target_visible"] > 0
    target_full = gt["target_full"] > 0
    occluder = gt["occluder"] > 0
    n_gt, h, w = target_vis.shape
    scale_720 = 720.0 / float(h)

    fg_occ_painted = 0
    fg_occ_total = 0
    bg_leak_vals: list[float] = []
    erosion_vals: list[float] = []
    false_occ_vals: list[float] = []
    flicker_vals: list[float] = []
    landmark_errs: list[float] = []
    attached = 0
    false_conf = 0
    wrong_obj = 0
    eligible = 0
    prev_vis_on_target: np.ndarray | None = None
    prev_painted: np.ndarray | None = None

    cyl_anchors_uv = [(0.50, 0.50), (0.42, 0.35), (0.58, 0.35), (0.42, 0.65), (0.58, 0.65)]
    cyl_ref_xy: list[tuple[float, float]] = []
    if kind == "cylinder":
        uv0 = gt["surface_uv"][0].astype(np.float32)
        for au, av in cyl_anchors_uv:
            d = np.hypot(uv0[..., 0] - au, uv0[..., 1] - av)
            d = np.where(uv0[..., 0] < 0, 999.0, d)
            iy, ix = np.unravel_index(int(np.argmin(d)), d.shape)
            cyl_ref_xy.append((float(ix), float(iy)))

    if kind == "planar":
        cw, ch = int(gt["card_size"][0]), int(gt["card_size"][1])
        card_corners = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32)
        H0 = gt["H_card_to_img"][0]
        ref_quad_gt = cv2.perspectiveTransform(card_corners.reshape(1, 4, 2), H0).reshape(4, 2)

    f0_init_ok = False
    n = min(len(records), n_gt)
    for i in range(n):
        row = records[i]
        state = row["state"]
        rmask = row["rendered_mask"]
        painted = rmask if rmask is not None else np.zeros((h, w), dtype=bool)
        n_painted = int(np.count_nonzero(painted))

        tv = target_vis[i]
        tf = target_full[i]
        occ_on_target = tf & occluder[i]
        n_tv = int(np.count_nonzero(tv))
        if n_tv < 200:
            continue
        eligible += 1

        if i == 0 and n_painted >= 100:
            iou0 = float(np.count_nonzero(painted & tv)) / float(max(int(np.count_nonzero(painted | tv)), 1))
            f0_init_ok = iou0 >= 0.50

        n_occ = int(np.count_nonzero(occ_on_target))
        if n_occ > 50:
            fg_occ_total += n_occ
            fg_occ_painted += int(np.count_nonzero(painted & occ_on_target))

        if n_painted > 20:
            tf_dil = cv2.dilate(tf.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
            bg_leak = float(np.count_nonzero(painted & ~tf_dil)) / float(n_painted)
            bg_leak_vals.append(bg_leak)
        else:
            bg_leak = 1.0

        tv_er = cv2.erode(tv.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        n_tv_er = int(np.count_nonzero(tv_er))
        cov = 0.0
        if state == "tracking" and n_tv_er > 100:
            cov = float(np.count_nonzero(painted & tv_er)) / float(n_tv_er)
            erosion = 1.0 - cov
            erosion_vals.append(erosion)
            if n_occ == 0:
                false_occ_vals.append(erosion)
            if prev_vis_on_target is not None and prev_painted is not None:
                both = tv_er & prev_vis_on_target
                if int(np.count_nonzero(both)) > 100:
                    fl = float(np.count_nonzero((painted != prev_painted) & both)) / float(np.count_nonzero(both))
                    flicker_vals.append(fl)
            prev_vis_on_target = tv_er
            prev_painted = painted
        else:
            prev_vis_on_target = None
            prev_painted = None

        err_px = None
        if state == "tracking":
            H_est = row["homography_3x3"]
            if kind == "planar" and H_est is not None:
                H_gt = gt["H_card_to_img"][i]
                q_gt = cv2.perspectiveTransform(card_corners.reshape(1, 4, 2), H_gt).reshape(4, 2)
                q_est = cv2.perspectiveTransform(ref_quad_gt.reshape(1, 4, 2), H_est).reshape(4, 2)
                err_px = float(np.median(np.linalg.norm(q_est - q_gt, axis=1))) * scale_720
            elif kind == "cylinder":
                uv_i = gt["surface_uv"][i].astype(np.float32)
                errs_a = []
                if row["pose_rvec"] is not None and row["pose_tvec"] is not None:
                    K = gt["K"].astype(np.float64)
                    R_cyl = float(gt["radius"])
                    H_cyl = float(gt["height"])
                    rvec = np.asarray(row["pose_rvec"], dtype=np.float64)
                    tvec = np.asarray(row["pose_tvec"], dtype=np.float64)
                    for au, av in cyl_anchors_uv:
                        d = np.hypot(uv_i[..., 0] - au, uv_i[..., 1] - av)
                        d = np.where(uv_i[..., 0] < 0, 999.0, d)
                        if float(d.min()) > 0.025:
                            continue
                        iy, ix = np.unravel_index(int(np.argmin(d)), d.shape)
                        phi = (au - 0.5) * 2.0 * math.pi
                        p3 = np.array([[R_cyl * math.sin(phi), (av - 0.5) * H_cyl, -R_cyl * math.cos(phi)]], np.float64)
                        proj, _ = cv2.projectPoints(p3, rvec, tvec, K, None)
                        errs_a.append(math.hypot(float(proj[0, 0, 0]) - ix, float(proj[0, 0, 1]) - iy) * scale_720)
                elif H_est is not None:
                    pts0 = np.asarray(cyl_ref_xy, dtype=np.float32).reshape(-1, 1, 2)
                    pts_est = cv2.perspectiveTransform(pts0, H_est).reshape(-1, 2)
                    for idx_a, (au, av) in enumerate(cyl_anchors_uv):
                        d = np.hypot(uv_i[..., 0] - au, uv_i[..., 1] - av)
                        d = np.where(uv_i[..., 0] < 0, 999.0, d)
                        if float(d.min()) > 0.025:
                            continue
                        iy, ix = np.unravel_index(int(np.argmin(d)), d.shape)
                        errs_a.append(math.hypot(float(pts_est[idx_a, 0]) - ix, float(pts_est[idx_a, 1]) - iy) * scale_720)
                if errs_a:
                    err_px = float(np.median(errs_a))

        if err_px is not None:
            landmark_errs.append(err_px)

        occ_bleed_f = float(np.count_nonzero(painted & occ_on_target)) / float(n_occ) if n_occ > 50 else 0.0
        if state == "tracking":
            vis_ok = int(np.count_nonzero(row["vis_diff"] & painted)) >= 50
            is_att = (
                vis_ok
                and cov >= 0.55
                and bg_leak <= 0.18
                and occ_bleed_f <= 0.15
                and (err_px is None or err_px <= 10.0)
            )
            if is_att:
                attached += 1
            if bg_leak > 0.40 or occ_bleed_f > 0.35 or (err_px is not None and err_px > 18.0):
                false_conf += 1
            if bg_leak > 0.50:
                wrong_obj += 1

    return {
        "f0_init_ok": bool(f0_init_ok),
        "eligible_frames": eligible,
        "attached_visible_rate": round(attached / max(eligible, 1), 4),
        "false_confident_rate": round(false_conf / max(eligible, 1), 4),
        "wrong_object_rate": round(wrong_obj / max(eligible, 1), 4),
        "fg_leakage_rate": round(fg_occ_painted / max(fg_occ_total, 1), 4) if fg_occ_total > 0 else "n/a",
        "bg_leakage_rate": round(float(np.mean(bg_leak_vals)), 4) if bg_leak_vals else "unscored",
        "target_erosion_rate": round(float(np.mean(erosion_vals)), 4) if erosion_vals else "unscored",
        "false_occlusion_rate": round(float(np.mean(false_occ_vals)), 4) if false_occ_vals else "unscored",
        "temporal_flicker": round(float(np.mean(flicker_vals)), 4) if flicker_vals else "unscored",
        "landmark_err_720p_med": round(float(np.median(landmark_errs)), 2) if landmark_errs else "unscored",
        "landmark_err_720p_p95": round(float(np.percentile(landmark_errs, 95)), 2) if landmark_errs else "unscored",
    }


def _score_real_clip(
    clip: dict[str, Any],
    gt_bundle: dict[str, Any],
    base_dir: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    cid = clip["id"]
    cat = clip["category"]
    w, h = int(clip["width"]), int(clip["height"])
    scale_720 = 720.0 / float(h)

    gt_frames = []
    cam_K = None
    if cat == "real_hope_720p":
        gt_frames = (gt_bundle.get("hope_720p") or {}).get(cid, {}).get("frames") or []
    elif cat == "real_ycb_video":
        ycb_entry = (gt_bundle.get("ycb") or {}).get(cid, {})
        gt_frames = ycb_entry.get("frames") or []
        if ycb_entry.get("cam_K"):
            cam_K = np.asarray(ycb_entry["cam_K"], dtype=np.float64).reshape(3, 3)

    # For YCB: precompute 5 canonical 3D interior surface landmarks on frame 0 via exact GT cam_K + depth t0[2]
    ycb_ref_2d: np.ndarray | None = None
    ycb_ref_3d_obj: np.ndarray | None = None
    if cat == "real_ycb_video" and cam_K is not None and gt_frames:
        f0 = gt_frames[0]
        bx0, by0, bw0, bh0 = [float(v) for v in f0["bbox_visib"]]
        R0 = np.asarray(f0["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        t0 = np.asarray(f0["cam_t_m2c"], dtype=np.float64).reshape(3)
        # 5 interior points inside frame-0 visible object bbox lifted at z = t0[2]
        ycb_ref_2d = np.array([
            [bx0 + 0.50 * bw0, by0 + 0.50 * bh0],
            [bx0 + 0.35 * bw0, by0 + 0.35 * bh0],
            [bx0 + 0.65 * bw0, by0 + 0.35 * bh0],
            [bx0 + 0.35 * bw0, by0 + 0.65 * bh0],
            [bx0 + 0.65 * bw0, by0 + 0.65 * bh0],
        ], dtype=np.float64)
        fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
        z0 = float(t0[2])
        p_cam0 = np.column_stack([(ycb_ref_2d[:, 0] - cx) * z0 / fx, (ycb_ref_2d[:, 1] - cy) * z0 / fy, np.full(5, z0)])
        ycb_ref_3d_obj = (p_cam0 - t0.reshape(1, 3)) @ R0

    eligible = 0
    attached = 0
    false_conf = 0
    wrong_obj = 0
    f0_init_ok = False
    fg_leak_vals: list[float] = []
    bg_leak_vals: list[float] = []
    erosion_vals: list[float] = []
    flicker_vals: list[float] = []
    landmark_errs: list[float] = []
    correct_rec = 0
    false_rec = 0

    prev_state: str | None = None
    had_tracked = False
    prev_vis: np.ndarray | None = None
    prev_painted: np.ndarray | None = None

    s3d_boxes = SHOW3D_DEV_KEYFRAME_BOXES.get(cid, {}) if cat == "real_show3d_60fps" else {}

    for i, row in enumerate(records):
        state = row["state"]
        rmask = row["rendered_mask"]
        painted = rmask if rmask is not None else np.zeros((h, w), dtype=bool)
        n_painted = int(np.count_nonzero(painted))
        gt_f = gt_frames[i] if i < len(gt_frames) else None

        is_recovery_frame = (state == "tracking" and had_tracked and prev_state in ("lost", "recovering"))
        if state == "tracking":
            had_tracked = True
        prev_state = state

        if cat == "real_hope_720p" and gt_f is not None:
            m_rel = gt_f.get("mask_visib_relpath")
            m_px = int(gt_f.get("mask_visib_pixels") or 0)
            if not m_rel or m_px < 400:
                continue
            m_img = cv2.imread(str(base_dir / m_rel), cv2.IMREAD_GRAYSCALE)
            if m_img is None:
                continue
            vis = m_img > 0
            vis_dil = cv2.dilate(m_img, np.ones((5, 5), np.uint8), iterations=1) > 0
            vis_er = cv2.erode(m_img, np.ones((5, 5), np.uint8), iterations=1) > 0
            eligible += 1
            if i == 0 and n_painted >= 50:
                iou0 = float(np.count_nonzero(painted & vis)) / float(max(int(np.count_nonzero(painted | vis)), 1))
                f0_init_ok = iou0 >= 0.45

            if state == "tracking" and n_painted >= 40:
                leak = float(np.count_nonzero(painted & ~vis_dil)) / float(n_painted)
                fg_leak_vals.append(leak)
                bg_leak_vals.append(leak)
                n_ver = max(int(np.count_nonzero(vis_er)), 1)
                cov = float(np.count_nonzero(painted & vis_er)) / float(n_ver)
                erosion_vals.append(1.0 - cov)
                if prev_vis is not None and prev_painted is not None:
                    both = vis_er & prev_vis
                    if int(np.count_nonzero(both)) > 100:
                        flicker_vals.append(float(np.count_nonzero((painted != prev_painted) & both)) / float(np.count_nonzero(both)))
                prev_vis = vis_er
                prev_painted = painted

                ys_c, xs_c = np.nonzero(painted)
                bx, by, bw, bh = gt_f["bbox_visib"]
                in_box = (bx <= float(xs_c.mean()) <= bx + bw) and (by <= float(ys_c.mean()) <= by + bh)
                vis_ok = int(np.count_nonzero(row["vis_diff"] & painted)) >= 40
                is_att = vis_ok and in_box and leak <= 0.15 and cov >= 0.45
                if is_att:
                    attached += 1
                if not in_box or leak > 0.35:
                    false_conf += 1
                if not in_box or leak > 0.50:
                    wrong_obj += 1
                if is_recovery_frame:
                    if in_box and leak <= 0.25:
                        correct_rec += 1
                    else:
                        false_rec += 1
            else:
                prev_vis = None
                prev_painted = None

        elif cat == "real_ycb_video" and gt_f is not None:
            visib_f = float(gt_f.get("visib_fract") or 0.0)
            bbox = gt_f.get("bbox_visib")
            if visib_f < 0.25 or not bbox or bbox[2] < 16 or bbox[3] < 16:
                continue
            eligible += 1
            bx, by, bw, bh = [float(v) for v in bbox]
            box_m = np.zeros((h, w), dtype=bool)
            box_m[max(0, int(by) - 3):min(h, int(by + bh) + 4), max(0, int(bx) - 3):min(w, int(bx + bw) + 4)] = True

            if i == 0 and n_painted >= 40:
                ys_c, xs_c = np.nonzero(painted)
                in_b0 = (bx <= float(xs_c.mean()) <= bx + bw) and (by <= float(ys_c.mean()) <= by + bh)
                out0 = float(np.count_nonzero(painted & ~box_m)) / float(n_painted)
                f0_init_ok = in_b0 and out0 <= 0.20

            if state == "tracking" and n_painted >= 30:
                out_leak = float(np.count_nonzero(painted & ~box_m)) / float(n_painted)
                bg_leak_vals.append(out_leak)
                ys_c, xs_c = np.nonzero(painted)
                in_box = (bx <= float(xs_c.mean()) <= bx + bw) and (by <= float(ys_c.mean()) <= by + bh)

                # Exact GT SE(3) rigid landmark reprojection error
                err_px = None
                if cam_K is not None and ycb_ref_2d is not None and ycb_ref_3d_obj is not None:
                    Rt = np.asarray(gt_f["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
                    tt = np.asarray(gt_f["cam_t_m2c"], dtype=np.float64).reshape(3)
                    p_cam_t = ycb_ref_3d_obj @ Rt.T + tt.reshape(1, 3)
                    gt_2d = np.column_stack([
                        p_cam_t[:, 0] * cam_K[0, 0] / p_cam_t[:, 2] + cam_K[0, 2],
                        p_cam_t[:, 1] * cam_K[1, 1] / p_cam_t[:, 2] + cam_K[1, 2],
                    ])
                    if row["homography_3x3"] is not None:
                        est_2d = cv2.perspectiveTransform(
                            ycb_ref_2d.astype(np.float32).reshape(-1, 1, 2), row["homography_3x3"]
                        ).reshape(-1, 2)
                        err_px = float(np.median(np.linalg.norm(est_2d - gt_2d, axis=1))) * scale_720
                        landmark_errs.append(err_px)
                    elif row["pose_rvec"] is not None and row["pose_tvec"] is not None and records[0]["pose_rvec"] is not None and records[0]["pose_tvec"] is not None:
                        r0_e = np.asarray(records[0]["pose_rvec"], dtype=np.float64).reshape(3)
                        t0_e = np.asarray(records[0]["pose_tvec"], dtype=np.float64).reshape(3)
                        R0_e, _ = cv2.Rodrigues(r0_e)
                        z0_e = float(t0_e[2])
                        p_cam0_e = np.column_stack([
                            (ycb_ref_2d[:, 0] - cam_K[0, 2]) * z0_e / cam_K[0, 0],
                            (ycb_ref_2d[:, 1] - cam_K[1, 2]) * z0_e / cam_K[1, 1],
                            np.full(len(ycb_ref_2d), z0_e),
                        ])
                        p_obj_e = (p_cam0_e - t0_e.reshape(1, 3)) @ R0_e
                        rt_e = np.asarray(row["pose_rvec"], dtype=np.float64).reshape(3)
                        tt_e = np.asarray(row["pose_tvec"], dtype=np.float64).reshape(3)
                        est_2d, _ = cv2.projectPoints(p_obj_e, rt_e, tt_e, cam_K, None)
                        est_2d = est_2d.reshape(-1, 2)
                        err_px = float(np.median(np.linalg.norm(est_2d - gt_2d, axis=1))) * scale_720
                        landmark_errs.append(err_px)

                vis_ok = int(np.count_nonzero(row["vis_diff"] & painted)) >= 30
                is_att = vis_ok and in_box and out_leak <= 0.25 and (err_px is None or err_px <= 10.0)
                if is_att:
                    attached += 1
                if not in_box or out_leak > 0.40 or (err_px is not None and err_px > 18.0):
                    false_conf += 1
                if not in_box or out_leak > 0.50:
                    wrong_obj += 1

        elif cat == "real_show3d_60fps" and i in s3d_boxes:
            eligible += 1
            bx, by, bw, bh = s3d_boxes[i]
            box_m = np.zeros((h, w), dtype=bool)
            box_m[max(0, int(by) - 4):min(h, int(by + bh) + 5), max(0, int(bx) - 4):min(w, int(bx + bw) + 5)] = True
            if i == 0 and n_painted >= 30:
                ys_c, xs_c = np.nonzero(painted)
                in_b0 = (bx <= float(xs_c.mean()) <= bx + bw) and (by <= float(ys_c.mean()) <= by + bh)
                out0 = float(np.count_nonzero(painted & ~box_m)) / float(n_painted)
                f0_init_ok = in_b0 and out0 <= 0.35

            if state == "tracking" and n_painted >= 20:
                out_leak = float(np.count_nonzero(painted & ~box_m)) / float(n_painted)
                bg_leak_vals.append(out_leak)
                ys_c, xs_c = np.nonzero(painted)
                in_box = (bx <= float(xs_c.mean()) <= bx + bw) and (by <= float(ys_c.mean()) <= by + bh)
                if in_box and out_leak <= 0.35:
                    attached += 1
                if not in_box or out_leak > 0.50:
                    false_conf += 1
                    wrong_obj += 1

    return {
        "f0_init_ok": bool(f0_init_ok),
        "eligible_frames": eligible,
        "attached_visible_rate": round(attached / max(eligible, 1), 4),
        "false_confident_rate": round(false_conf / max(eligible, 1), 4),
        "wrong_object_rate": round(wrong_obj / max(eligible, 1), 4),
        "fg_leakage_rate": round(float(np.mean(fg_leak_vals)), 4) if fg_leak_vals else "unscored",
        "bg_leakage_rate": round(float(np.mean(bg_leak_vals)), 4) if bg_leak_vals else "unscored",
        "target_erosion_rate": round(float(np.mean(erosion_vals)), 4) if erosion_vals else "unscored",
        "temporal_flicker": round(float(np.mean(flicker_vals)), 4) if flicker_vals else "unscored",
        "landmark_err_720p_med": round(float(np.median(landmark_errs)), 2) if landmark_errs else "unscored",
        "landmark_err_720p_p95": round(float(np.percentile(landmark_errs, 95)), 2) if landmark_errs else "unscored",
        "correct_recoveries": correct_rec,
        "false_recoveries": false_rec,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--selection-mode", choices=("default_auto", "user_selection"), default="default_auto")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--wrap-override", default=None)
    ap.add_argument("--code-root", type=Path, default=REPO_ROOT, help="Repo root to import visualizeit from (for old-vs-new audit)")
    ap.add_argument("--include-holdout-regression-only", action="store_true", help="Read-only regression check on sealed YCB sugar_box")
    ap.add_argument("--save-strips", action="store_true")
    ap.add_argument("--save-videos", action="store_true")
    ap.add_argument("--occlusion-mode", default=None, help="Optional compositor occlusion_mode override if supported")
    args = ap.parse_args()

    code_root = args.code_root.resolve()
    sys.path.insert(0, str(code_root))
    from visualizeit.contracts import DesignSpec, PlacementSpec, SelectionSpec
    from visualizeit.pipeline import VideoPipelineSession
    from visualizeit.segmentation import segment_frame
    from visualizeit.tracker import resolve_mesh

    clips, gt_bundle, base_dir = _load_diagnostic_clips(include_holdout_controls=args.include_holdout_regression_only)
    if args.only:
        clips = [c for c in clips if c["id"] in set(args.only)]

    out_dir = REPO_ROOT / "artifacts" / "visual-quality" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_dir = REPO_ROOT / ".cache" / "tmp" / "vq_runs" / args.tag
    if args.save_videos:
        vid_dir.mkdir(parents=True, exist_ok=True)

    design = DesignSpec(
        source="image",
        image_path=str(REPO_ROOT / ".cache" / "vq-diagnostics" / "design_asym.png"),
    )

    results: list[dict[str, Any]] = []
    print(f"[{args.tag}] code={code_root.name} mode={args.selection_mode} clips={len(clips)}", flush=True)

    for c in clips:
        cid = c["id"]
        raw_mp4 = Path(c["path"])
        wrap_mode = args.wrap_override or c["recommended_wrap"]
        sel_dict = c["user_selection"] if args.selection_mode == "user_selection" else c["default_selection"]
        sel = SelectionSpec.from_dict(sel_dict)
        placement = PlacementSpec(wrap_mode=wrap_mode, blend_mode="luminance", opacity=0.88)

        mesh = None
        intr = None
        if wrap_mode == "mesh_6dof":
            mesh_spec = "cylinder" if c["recommended_wrap"] == "cylinder" else "box"
            mesh = resolve_mesh(mesh_spec)
            if "intrinsics" in c:
                intr = tuple(c["intrinsics"])
            elif c.get("category") == "real_hope_720p":
                K_list = gt_bundle["hope_720p"][cid].get("cam_K_720p")
                if K_list and len(K_list) == 9:
                    intr = (float(K_list[0]), float(K_list[4]), float(K_list[2]), float(K_list[5]))
            elif c.get("category") == "real_ycb_video":
                K_list = gt_bundle["ycb"][cid].get("cam_K")
                if K_list and len(K_list) == 9:
                    intr = (float(K_list[0]), float(K_list[4]), float(K_list[2]), float(K_list[5]))

        # Inspect frame-0 segmentation
        cap0 = cv2.VideoCapture(str(raw_mp4))
        ok0, f0 = cap0.read()
        cap0.release()
        seg0 = segment_frame(f0, sel, sharp_onboarding=True)
        init_meta: dict[str, Any] = {
            "fg_px": seg0.foreground_pixels,
            "bbox": list(seg0.bbox),
            "init_ms": round(seg0.init_ms, 1),
            "init_valid": getattr(seg0, "init_valid", True),
            "quality_score": round(float(getattr(seg0, "quality_score", 1.0)), 3),
        }
        if c["source_kind"] == "syn_diag":
            gt0 = np.load(c["gt_npz"])["target_visible"][0] > 0
            m0 = seg0.mask > 0
            init_meta["f0_iou"] = round(int(np.count_nonzero(m0 & gt0)) / max(int(np.count_nonzero(m0 | gt0)), 1), 4)
        elif c.get("category") == "real_hope_720p":
            gt_f0 = gt_bundle["hope_720p"][cid]["frames"][0]
            m_gt = cv2.imread(str(base_dir / gt_f0["mask_visib_relpath"]), cv2.IMREAD_GRAYSCALE) > 0
            m0 = seg0.mask > 0
            init_meta["f0_iou"] = round(int(np.count_nonzero(m0 & m_gt)) / max(int(np.count_nonzero(m0 | m_gt)), 1), 4)

        sess_kwargs: dict[str, Any] = dict(
            design=design,
            placement=placement,
            mesh=mesh,
            intrinsics=intr,
            raster_backend="native_cpp",
        )
        if args.occlusion_mode is not None:
            import inspect
            if "occlusion_mode" in inspect.signature(VideoPipelineSession.__init__).parameters:
                sess_kwargs["occlusion_mode"] = args.occlusion_mode

        session = VideoPipelineSession(**sess_kwargs)

        n_tot = int(c["frame_count"])
        strip_idxs = {0, n_tot // 5, (2 * n_tot) // 5, (3 * n_tot) // 5, (4 * n_tot) // 5, n_tot - 1} if args.save_strips else set()
        vid_out = (vid_dir / f"{cid}.mp4") if args.save_videos else None

        records, strip_frames, eff_fps, _ = _run_session_on_video(
            session=session,
            raw_mp4=raw_mp4,
            sel=sel,
            strip_idxs=strip_idxs,
            save_video_path=vid_out,
        )

        if args.save_strips and strip_frames:
            _save_strip_from_frames(strip_frames, out_dir / f"strip_{cid}.jpg")

        if c["source_kind"] == "syn_diag":
            score = _score_syn_diag(c, records)
        else:
            score = _score_real_clip(c, gt_bundle, base_dir, records)

        n_frames = max(len(records), 1)
        tracked = sum(1 for r in records if r["state"] == "tracking")
        ambig = sum(1 for r in records if r["confidence_state"] == "AMBIGUOUS")
        lost = sum(1 for r in records if r["state"] == "lost")
        rec_ev = 0
        had_t = False
        prev_s = None
        for r in records:
            if r["state"] == "tracking":
                if had_t and prev_s in ("lost", "recovering"):
                    rec_ev += 1
                had_t = True
            prev_s = r["state"]

        rec = {
            "id": cid,
            "holdout_regression_only": bool(c.get("holdout_regression_only", False)),
            "wrap_mode": wrap_mode,
            "fps": round(eff_fps, 1),
            "avail": round(tracked / n_frames, 4),
            "lost": lost,
            "ambiguous": ambig,
            "recovered": rec_ev,
            "init": init_meta,
            "score": score,
        }
        results.append(rec)
        print(
            f"  {cid:28s} wrap={wrap_mode:8s} f0_ok={str(score['f0_init_ok']):5s} "
            f"att={score['attached_visible_rate']*100:5.1f}% fc={score['false_confident_rate']*100:4.1f}% "
            f"wo={score['wrong_object_rate']*100:4.1f}% fg={str(score.get('fg_leakage_rate')):7s} "
            f"ero={str(score.get('target_erosion_rate')):7s} lm={str(score.get('landmark_err_720p_med')):6s}/{str(score.get('landmark_err_720p_p95')):6s} "
            f"rec={rec_ev:2d} fps={eff_fps:5.1f}",
            flush=True,
        )

    summary = {"tag": args.tag, "code_root": str(code_root), "selection_mode": args.selection_mode, "clips": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
