#!/usr/bin/env python3
"""Rigorous visual-quality evaluation & controlled ablation harness (Dev-only; Holdout stays sealed).

Evaluates:
  - Real Dev clips: show3d_mug_60fps, show3d_ranch_bottle_60fps, show3d_keyboard_60fps,
                    ycb_s52_window1 (pitcher), ycb_s50_window1 (mustard positive control),
                    ycb_s59_window1 (cracker box), hope_dyn_obj01_720p (soup can), hope_dyn_obj02_720p (BBQ bottle)
  - Synthetic Dev GT clips: vqdev_occluder_planar, vqdev_occluder_planar_mono, vqdev_spin_cylinder

Strictly separates:
  - 'default_auto': existing benchmark default_selection
  - 'user_selection': human-provided click/box on frame 0 (never leaks reference poses)
  - 'diag_ref_mask' / 'diag_ref_pose': diagnostic-only controls (never used in causal production)
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualizeit.contracts import DesignSpec, PlacementSpec, SelectionSpec, TrackFrameResult
from visualizeit.pipeline import VideoPipeline, VideoPipelineSession
from visualizeit.segmentation import segment_frame
from visualizeit.tracker import build_parametric_mesh

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


def _load_diagnostic_clips() -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
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
        assert c["split"] == "dev", f"Refusing to run diagnostics on non-dev clip {cid} ({c['split']})"
        c["source_kind"] = "real"
        c["user_selection"] = DEV_USER_SELECTIONS[cid]
        clips.append(c)

    for c in diag_manifest["clips"]:
        cc = dict(c)
        cc["source_kind"] = "syn_diag"
        cc["default_selection"] = cc["user_selection"]
        clips.append(cc)

    return clips, gt_bundle, base_dir


def _score_syn_diag(
    clip: dict[str, Any],
    raw_mp4: Path,
    out_mp4: Path,
    trace: list[dict[str, Any]],
    ref_bbox: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    gt = np.load(clip["gt_npz"])
    kind = str(gt["kind"])
    target_vis = gt["target_visible"] > 0
    target_full = gt["target_full"] > 0
    occluder = gt["occluder"] > 0
    n_gt, h, w = target_vis.shape
    scale_720 = 720.0 / float(h)

    cap_r = cv2.VideoCapture(str(raw_mp4))
    cap_o = cv2.VideoCapture(str(out_mp4))

    fg_occ_painted = 0
    fg_occ_total = 0
    bg_leak_vals: list[float] = []
    erosion_vals: list[float] = []
    false_occ_vals: list[float] = []
    flicker_vals: list[float] = []
    landmark_errs: list[float] = []
    attached = 0
    false_conf = 0
    eligible = 0
    prev_vis_on_target: np.ndarray | None = None

    ref_quad_trk: np.ndarray | None = None
    if ref_bbox is not None:
        bx, by, bw, bh = ref_bbox
        ref_quad_trk = np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]], np.float32)

    # For spinning cylinder, track 5 canonical surface anchors in UV space on frame 0
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

    n = min(len(trace), n_gt)
    try:
        for i in range(n):
            ok_r, raw_f = cap_r.read()
            ok_o, out_f = cap_o.read()
            if not ok_r or not ok_o:
                break
            row = trace[i]
            state = row["state"]
            diff = np.max(np.abs(out_f.astype(np.float32) - raw_f.astype(np.float32)), axis=2)
            painted = diff > 14.0
            n_painted = int(np.count_nonzero(painted))

            tv = target_vis[i]
            tf = target_full[i]
            occ_on_target = tf & occluder[i]
            n_tv = int(np.count_nonzero(tv))
            if n_tv < 200:
                continue
            eligible += 1

            # Foreground occluder bleed-through (pixels of occluder in front of target that got painted)
            n_occ = int(np.count_nonzero(occ_on_target))
            if n_occ > 50:
                fg_occ_total += n_occ
                fg_occ_painted += int(np.count_nonzero(painted & occ_on_target))

            # Background leakage outside target_full
            if n_painted > 20:
                tf_dil = cv2.dilate(tf.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
                bg_leak = float(np.count_nonzero(painted & ~tf_dil)) / float(n_painted)
                bg_leak_vals.append(bg_leak)
            else:
                bg_leak = 1.0

            # Target erosion (visible target pixels NOT painted when tracking)
            tv_er = cv2.erode(tv.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
            n_tv_er = int(np.count_nonzero(tv_er))
            if state == "tracking" and n_tv_er > 100:
                cov = float(np.count_nonzero(painted & tv_er)) / float(n_tv_er)
                erosion = 1.0 - cov
                erosion_vals.append(erosion)
                if n_occ == 0:
                    false_occ_vals.append(erosion)
                # Temporal flicker on stably visible target interior
                if prev_vis_on_target is not None:
                    both = tv_er & prev_vis_on_target
                    if int(np.count_nonzero(both)) > 100:
                        fl = float(np.count_nonzero((painted != prev_painted) & both)) / float(np.count_nonzero(both))
                        flicker_vals.append(fl)
                prev_vis_on_target = tv_er
                prev_painted = painted
            else:
                prev_vis_on_target = None

            # Recover H_est from warped_quad if available
            H_est: np.ndarray | None = None
            if row.get("homography_3x3") is not None:
                H_est = np.asarray(row["homography_3x3"], dtype=np.float64)
            elif row.get("warped_quad") is not None and ref_quad_trk is not None:
                wq = np.asarray(row["warped_quad"], dtype=np.float32).reshape(4, 2)
                H_est = cv2.getPerspectiveTransform(ref_quad_trk, wq).astype(np.float64)

            # Landmark error
            err_px = None
            if state == "tracking":
                if kind == "planar" and H_est is not None:
                    H_gt = gt["H_card_to_img"][i]
                    q_gt = cv2.perspectiveTransform(card_corners.reshape(1, 4, 2), H_gt).reshape(4, 2)
                    q_est = cv2.perspectiveTransform(ref_quad_gt.reshape(1, 4, 2), H_est).reshape(4, 2)
                    err_px = float(np.median(np.linalg.norm(q_est - q_gt, axis=1))) * scale_720
                elif kind == "cylinder":
                    uv_i = gt["surface_uv"][i].astype(np.float32)
                    errs_a = []
                    if row.get("pose_rvec") is not None and row.get("pose_tvec") is not None:
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
                            px, py = float(proj[0, 0, 0]), float(proj[0, 0, 1])
                            errs_a.append(math.hypot(px - ix, py - iy) * scale_720)
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

            occ_bleed_frame = (
                float(np.count_nonzero(painted & occ_on_target)) / float(n_occ)
                if n_occ > 50 else 0.0
            )
            if state == "tracking":
                is_att = (
                    n_painted >= 100
                    and bg_leak <= 0.20
                    and occ_bleed_frame <= 0.30
                    and (err_px is None or err_px <= 10.0)
                )
                if is_att:
                    attached += 1
                if bg_leak > 0.40 or (err_px is not None and err_px > 20.0):
                    false_conf += 1
    finally:
        cap_r.release()
        cap_o.release()

    return {
        "eligible_frames": eligible,
        "attached_visible_rate": round(attached / max(eligible, 1), 4),
        "false_confident_rate": round(false_conf / max(eligible, 1), 4),
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
    raw_mp4: Path,
    out_mp4: Path,
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    cid = clip["id"]
    cat = clip["category"]
    w, h = int(clip["width"]), int(clip["height"])
    scale_720 = 720.0 / float(h)

    gt_frames = []
    if cat == "real_hope_720p":
        gt_frames = (gt_bundle.get("hope_720p") or {}).get(cid, {}).get("frames") or []
    elif cat == "real_ycb_video":
        gt_frames = (gt_bundle.get("ycb") or {}).get(cid, {}).get("frames") or []

    cap_r = cv2.VideoCapture(str(raw_mp4))
    cap_o = cv2.VideoCapture(str(out_mp4))

    eligible = 0
    attached = 0
    false_conf = 0
    leak_vals: list[float] = []
    erosion_vals: list[float] = []
    centroid_errs: list[float] = []
    tracked_frames = sum(1 for r in trace if r["state"] == "tracking")

    try:
        for i, row in enumerate(trace):
            ok_r, raw_f = cap_r.read()
            ok_o, out_f = cap_o.read()
            if not ok_r or not ok_o:
                break
            state = row["state"]
            diff = np.max(np.abs(out_f.astype(np.float32) - raw_f.astype(np.float32)), axis=2)
            painted = diff > 14.0
            n_painted = int(np.count_nonzero(painted))
            gt_f = gt_frames[i] if i < len(gt_frames) else None

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
                eligible += 1
                if state == "tracking" and n_painted >= 40:
                    leak = float(np.count_nonzero(painted & ~vis_dil)) / float(n_painted)
                    leak_vals.append(leak)
                    erosion_vals.append(1.0 - float(np.count_nonzero(painted & vis)) / float(max(int(np.count_nonzero(vis)), 1)))
                    ys_c, xs_c = np.nonzero(painted)
                    ys_v, xs_v = np.nonzero(vis)
                    c_err = math.hypot(float(xs_c.mean() - xs_v.mean()), float(ys_c.mean() - ys_v.mean())) * scale_720
                    centroid_errs.append(c_err)
                    bx, by, bw, bh = gt_f["bbox_visib"]
                    in_box = (bx <= xs_c.mean() <= bx + bw) and (by <= ys_c.mean() <= by + bh)
                    if leak <= 0.30 and in_box:
                        attached += 1
                    elif leak > 0.50 or not in_box:
                        false_conf += 1

            elif cat == "real_ycb_video" and gt_f is not None:
                visib_f = float(gt_f.get("visib_fract") or 0.0)
                bbox = gt_f.get("bbox_visib")
                if visib_f < 0.25 or not bbox or bbox[2] < 16 or bbox[3] < 16:
                    continue
                eligible += 1
                bx, by, bw, bh = [float(v) for v in bbox]
                if state == "tracking" and n_painted >= 30:
                    box_m = np.zeros((h, w), dtype=bool)
                    box_m[max(0, int(by) - 3):min(h, int(by + bh) + 4), max(0, int(bx) - 3):min(w, int(bx + bw) + 4)] = True
                    out_leak = float(np.count_nonzero(painted & ~box_m)) / float(n_painted)
                    leak_vals.append(out_leak)
                    ys_c, xs_c = np.nonzero(painted)
                    c_err = math.hypot(float(xs_c.mean()) - (bx + bw * 0.5), float(ys_c.mean()) - (by + bh * 0.5)) * scale_720
                    centroid_errs.append(c_err)
                    in_box = (bx <= xs_c.mean() <= bx + bw) and (by <= ys_c.mean() <= by + bh)
                    if in_box and out_leak <= 0.30:
                        attached += 1
                    elif not in_box or out_leak > 0.50:
                        false_conf += 1
    finally:
        cap_r.release()
        cap_o.release()

    if cat == "real_show3d_60fps":
        return {
            "eligible_frames": len(trace),
            "tracked_rate": round(tracked_frames / max(len(trace), 1), 4),
            "attached_visible_rate": "unscored",
            "false_confident_rate": "unscored",
            "fg_leakage_rate": "unscored",
            "bg_leakage_rate": "unscored",
            "target_erosion_rate": "unscored",
            "landmark_err_720p_med": "unscored",
            "landmark_err_720p_p95": "unscored",
        }

    return {
        "eligible_frames": eligible,
        "tracked_rate": round(tracked_frames / max(len(trace), 1), 4),
        "attached_visible_rate": round(attached / max(eligible, 1), 4),
        "false_confident_rate": round(false_conf / max(eligible, 1), 4),
        "fg_leakage_rate": round(float(np.mean(leak_vals)), 4) if (cat == "real_hope_720p" and leak_vals) else "unscored",
        "bg_leakage_rate": round(float(np.mean(leak_vals)), 4) if leak_vals else "unscored",
        "target_erosion_rate": round(float(np.mean(erosion_vals)), 4) if erosion_vals else "unscored",
        "centroid_err_720p_med": round(float(np.median(centroid_errs)), 2) if centroid_errs else "unscored",
        "landmark_err_720p_med": "unscored",
        "landmark_err_720p_p95": "unscored",
    }


def _save_strip(raw_mp4: Path, out_mp4: Path, out_jpg: Path, idxs: list[int]) -> None:
    cap_r = cv2.VideoCapture(str(raw_mp4))
    cap_o = cv2.VideoCapture(str(out_mp4))
    n_tot = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
    want = sorted(set(min(max(0, i), max(0, n_tot - 1)) for i in idxs))
    cols = []
    for k in range(max(want) + 1):
        ok_r, fr = cap_r.read()
        ok_o, fo = cap_o.read()
        if not ok_r or not ok_o:
            break
        if k in want:
            h, w = fr.shape[:2]
            tw = int(round(w * 240.0 / float(h)))
            r_s = cv2.resize(fr, (tw, 240))
            o_s = cv2.resize(fo, (tw, 240))
            cv2.putText(r_s, f"f={k}", (6, 20), 0, 0.55, (0, 255, 255), 2)
            cols.append(np.vstack([r_s, o_s]))
    cap_r.release()
    cap_o.release()
    if cols:
        cv2.imwrite(str(out_jpg), np.hstack(cols))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--selection-mode", choices=("default_auto", "user_selection"), default="default_auto")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--wrap-override", default=None)
    ap.add_argument("--save-strips", action="store_true")
    args = ap.parse_args()

    clips, gt_bundle, base_dir = _load_diagnostic_clips()
    if args.only:
        clips = [c for c in clips if c["id"] in set(args.only)]

    out_dir = REPO_ROOT / "artifacts" / "visual-quality" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_dir = REPO_ROOT / ".cache" / "tmp" / "vq_runs" / args.tag
    vid_dir.mkdir(parents=True, exist_ok=True)

    design = DesignSpec(
        source="image",
        image_path=str(REPO_ROOT / ".cache" / "vq-diagnostics" / "design_asym.png"),
    )
    pipeline = VideoPipeline()
    results: list[dict[str, Any]] = []

    print(f"[{args.tag}] mode={args.selection_mode} clips={len(clips)}", flush=True)
    for c in clips:
        cid = c["id"]
        raw_mp4 = Path(c["path"])
        out_mp4 = vid_dir / f"{cid}.mp4"
        wrap_mode = args.wrap_override or c["recommended_wrap"]
        sel_dict = c["user_selection"] if args.selection_mode == "user_selection" else c["default_selection"]
        sel = SelectionSpec.from_dict(sel_dict)
        placement = PlacementSpec(wrap_mode=wrap_mode, blend_mode="luminance", opacity=0.88)

        mesh_spec = None
        intr = None
        if wrap_mode == "mesh_6dof":
            mesh_spec = "cylinder" if c["recommended_wrap"] == "cylinder" else "box"
            if "intrinsics" in c:
                intr = tuple(c["intrinsics"])
            elif c.get("category") == "real_hope_720p":
                K_list = gt_bundle["hope_720p"][cid].get("cam_K_720p")
                if K_list and len(K_list) == 9:
                    intr = (float(K_list[0]), float(K_list[4]), float(K_list[2]), float(K_list[5]))

        # Measure frame-0 segmentation quality
        cap0 = cv2.VideoCapture(str(raw_mp4))
        ok0, f0 = cap0.read()
        cap0.release()
        seg0 = segment_frame(f0, sel, sharp_onboarding=True)
        init_meta: dict[str, Any] = {
            "fg_px": seg0.foreground_pixels,
            "bbox": list(seg0.bbox),
            "init_ms": round(seg0.init_ms, 1),
        }
        if c["source_kind"] == "syn_diag":
            gt0 = np.load(c["gt_npz"])["target_visible"][0] > 0
            m0 = seg0.mask > 0
            inter = int(np.count_nonzero(m0 & gt0))
            union = int(np.count_nonzero(m0 | gt0))
            init_meta["f0_iou"] = round(inter / max(union, 1), 4)
        elif c.get("category") == "real_hope_720p":
            gt_f0 = gt_bundle["hope_720p"][cid]["frames"][0]
            m_gt = cv2.imread(str(base_dir / gt_f0["mask_visib_relpath"]), cv2.IMREAD_GRAYSCALE) > 0
            m0 = seg0.mask > 0
            init_meta["f0_iou"] = round(int(np.count_nonzero(m0 & m_gt)) / max(int(np.count_nonzero(m0 | m_gt)), 1), 4)

        trace: list[dict[str, Any]] = []
        rep = pipeline.process_video(
            input_path=raw_mp4,
            output_path=out_mp4,
            design=design,
            placement=placement,
            selection=sel,
            mesh_spec=mesh_spec,
            intrinsics=intr,
            raster_backend="native_cpp",
            trace_records=trace,
        )

        if c["source_kind"] == "syn_diag":
            score = _score_syn_diag(c, raw_mp4, out_mp4, trace, ref_bbox=seg0.bbox)
        else:
            score = _score_real_clip(c, gt_bundle, base_dir, raw_mp4, out_mp4, trace)

        if args.save_strips:
            n = int(c["frame_count"])
            idxs = [0, n // 5, (2 * n) // 5, (3 * n) // 5, (4 * n) // 5, n - 1]
            _save_strip(raw_mp4, out_mp4, out_dir / f"strip_{cid}.jpg", idxs)

        rec = {
            "id": cid,
            "wrap_mode": wrap_mode,
            "fps": round(rep.effective_fps, 1),
            "avail": round(rep.tracking_availability, 4),
            "lost": rep.lost_frame_count,
            "recovered": rep.recovered_count,
            "init": init_meta,
            "score": score,
        }
        results.append(rec)
        print(
            f"  {cid:28s} wrap={wrap_mode:8s} fps={rec['fps']:5.1f} avail={rec['avail']*100:5.1f}% "
            f"rec={rec['recovered']:2d} att={str(score.get('attached_visible_rate')):8s} "
            f"fg_leak={str(score.get('fg_leakage_rate')):8s} bg_leak={str(score.get('bg_leakage_rate')):8s} "
            f"erosion={str(score.get('target_erosion_rate')):8s} lm_med={str(score.get('landmark_err_720p_med')):8s}",
            flush=True,
        )

    summary = {"tag": args.tag, "selection_mode": args.selection_mode, "clips": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
