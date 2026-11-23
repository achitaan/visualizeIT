#!/usr/bin/env python3
"""Comprehensive recorded-video evaluator for visualizeIT (`iter-000`, `iter-001`, ...).

Enforces:
  - Leakage-free Dev vs Holdout split (`manifest_v2.json`) — never opens holdout unless `--split holdout|all` is explicitly requested.
  - Full open-to-flush wall timing (`processed_fps`), 4-way frame equality (`ffprobe`), resolution breakouts (480p / 720p / 1080p), and repeated-run reporting.
  - Shared-core 30 Hz live-replay paced benchmark (`p50`/`p95` arrival-to-ready ms, deadline misses, max queue depth).
  - "Artwork the user sees" metrics: attached visible frame fraction (lost & wrong frames in denominator),
    720p-eq landmark error (where GT exists; explicit `unscored` otherwise), real hand/background overlay
    leakage on HOPE Dynamic `mask_visib` & synthetic quads, occluded-frame ghosting MAD, and frames to first
    correct reacquisition.
  - Worst-case Dev MP4 & side-by-side PNG strip exports + `--compare` iteration diff table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualizeit.contracts import DesignSpec, PlacementSpec, SelectionSpec
from visualizeit.pipeline import VideoPipeline, probe_video_info, replay_live_30hz


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_info() -> dict[str, str]:
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
        return {"branch": branch, "commit": commit, "dirty_working_tree": str(dirty)}
    except Exception:
        return {"branch": "unknown", "commit": "unknown", "dirty_working_tree": "unknown"}


def _resolution_bucket(width: int, height: int) -> str:
    h = min(width, height)
    if h >= 1000:
        return "1080p"
    if h >= 700:
        return "720p"
    if h >= 440:
        return "480p"
    return "360p"


def _score_clip_artwork(
    *,
    clip: dict[str, Any],
    gt_bundle: dict[str, Any],
    base_dir: Path,
    raw_mp4: Path,
    out_mp4: Path,
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score the artwork the user actually sees from decoded raw & output frames + trace."""
    cid = clip["id"]
    cat = clip["category"]
    w, h = int(clip["width"]), int(clip["height"])
    scale_720 = 720.0 / float(h)

    cap_raw = cv2.VideoCapture(str(raw_mp4))
    cap_out = cv2.VideoCapture(str(out_mp4))

    eligible_visible = 0
    attached_visible = 0
    landmark_errs_720: list[float] = []
    leakage_vals: list[float] = []
    bbox_outside_leakage_vals: list[float] = []
    centroid_offsets_720: list[float] = []
    occluded_mads: list[float] = []

    # Reacquisition tracking across verified full occlusions
    in_occlusion = False
    waiting_reacq = False
    reacq_wait_frames = 0
    reacq_delays: list[int | None] = []

    # Resolve clip GT frames
    gt_frames: list[dict[str, Any] | None] = []
    if cat == "real_hope_720p":
        gt_frames = (gt_bundle.get("hope_720p") or {}).get(cid, {}).get("frames") or []
    elif cat == "real_ycb_video":
        gt_frames = (gt_bundle.get("ycb") or {}).get(cid, {}).get("frames") or []
    elif cat.startswith("synthetic_"):
        gt_frames = (gt_bundle.get("synthetic") or {}).get(cid, {}).get("frames") or []

    n_trace = len(trace)
    try:
        for i in range(n_trace):
            ok_r, raw_f = cap_raw.read()
            ok_o, out_f = cap_out.read()
            if not ok_r or not ok_o or raw_f is None or out_f is None:
                break

            row = trace[i]
            state = row["state"]
            gt_f = gt_frames[i] if i < len(gt_frames) else None
            diff_abs = np.abs(out_f.astype(np.float32) - raw_f.astype(np.float32))
            diff_max = np.max(diff_abs, axis=2)
            diff_mean = float(np.mean(diff_abs))
            changed = diff_max > 16.0
            n_changed = int(np.count_nonzero(changed))

            if cat.startswith("synthetic_") and gt_f is not None:
                is_occ = bool(gt_f.get("occluded", False))
                if is_occ:
                    occluded_mads.append(diff_mean)
                    if not in_occlusion:
                        in_occlusion = True
                        waiting_reacq = False
                    continue

                if in_occlusion:
                    in_occlusion = False
                    waiting_reacq = True
                    reacq_wait_frames = 0

                eligible_visible += 1
                quad_gt_raw = gt_f.get("quad")
                is_attached = False
                if quad_gt_raw is not None:
                    q_gt = np.asarray(quad_gt_raw, dtype=np.float64)
                    # Mask of GT quad (dilated 2px for raster/H.264 edge tolerance)
                    gt_poly_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(gt_poly_mask, np.round(q_gt).astype(np.int32), 255)
                    gt_poly_dil = cv2.dilate(gt_poly_mask, np.ones((5, 5), np.uint8), iterations=1) > 0

                    if state == "tracking" and row.get("warped_quad") is not None:
                        q_est = np.asarray(row["warped_quad"], dtype=np.float64)
                        corner_err = float(np.median(np.linalg.norm(q_est - q_gt, axis=1))) * scale_720
                        landmark_errs_720.append(corner_err)
                        if n_changed > 20:
                            leak = float(np.count_nonzero(changed & ~gt_poly_dil)) / float(n_changed)
                            leakage_vals.append(leak)
                        else:
                            leak = 1.0
                        if corner_err <= 10.0 and leak <= 0.25 and n_changed >= 100:
                            is_attached = True
                else:
                    # Synthetic cylinder or 3D mesh (no 4-corner quad_gt)
                    if state == "tracking" and n_changed >= 100:
                        is_attached = True

                if is_attached:
                    attached_visible += 1
                    if waiting_reacq:
                        reacq_delays.append(reacq_wait_frames)
                        waiting_reacq = False
                elif waiting_reacq:
                    reacq_wait_frames += 1

            elif cat == "real_hope_720p" and gt_f is not None:
                m_rel = gt_f.get("mask_visib_relpath")
                m_px = int(gt_f.get("mask_visib_pixels") or 0)
                if m_rel is None or m_px < 400:
                    continue
                m_img = cv2.imread(str(base_dir / m_rel), cv2.IMREAD_GRAYSCALE)
                if m_img is None:
                    continue
                vis_mask = m_img > 0
                vis_dil = cv2.dilate(m_img, np.ones((5, 5), np.uint8), iterations=1) > 0
                eligible_visible += 1

                if state == "tracking" and n_changed >= 40:
                    leak = float(np.count_nonzero(changed & ~vis_dil)) / float(n_changed)
                    leakage_vals.append(leak)
                    # Check centroid inside visible bbox
                    ys_c, xs_c = np.nonzero(changed)
                    cx_ren, cy_ren = float(xs_c.mean()), float(ys_c.mean())
                    ys_v, xs_v = np.nonzero(vis_mask)
                    cx_gt, cy_gt = float(xs_v.mean()), float(ys_v.mean())
                    c_off = math.hypot(cx_ren - cx_gt, cy_ren - cy_gt) * scale_720
                    centroid_offsets_720.append(c_off)
                    bx, by, bw, bh = gt_f["bbox_visib"]
                    in_box = (bx <= cx_ren <= bx + bw) and (by <= cy_ren <= by + bh)
                    if leak <= 0.35 and in_box:
                        attached_visible += 1

            elif cat == "real_ycb_video" and gt_f is not None:
                visib_f = float(gt_f.get("visib_fract") or 0.0)
                bbox = gt_f.get("bbox_visib")
                if visib_f < 0.25 or not bbox or bbox[2] < 16 or bbox[3] < 16:
                    continue
                eligible_visible += 1
                bx, by, bw, bh = [float(v) for v in bbox]
                if state == "tracking" and n_changed >= 30:
                    box_mask = np.zeros((h, w), dtype=bool)
                    x0, y0 = max(0, int(bx) - 3), max(0, int(by) - 3)
                    x1, y1 = min(w, int(bx + bw) + 4), min(h, int(by + bh) + 4)
                    box_mask[y0:y1, x0:x1] = True
                    out_leak = float(np.count_nonzero(changed & ~box_mask)) / float(n_changed)
                    bbox_outside_leakage_vals.append(out_leak)
                    ys_c, xs_c = np.nonzero(changed)
                    cx_ren, cy_ren = float(xs_c.mean()), float(ys_c.mean())
                    cx_gt, cy_gt = bx + bw * 0.5, by + bh * 0.5
                    centroid_offsets_720.append(math.hypot(cx_ren - cx_gt, cy_ren - cy_gt) * scale_720)
                    in_box = (bx <= cx_ren <= bx + bw) and (by <= cy_ren <= by + bh)
                    if in_box and out_leak <= 0.40:
                        attached_visible += 1
    finally:
        cap_raw.release()
        cap_out.release()

    if waiting_reacq:
        reacq_delays.append(None)  # Never reacquired after occlusion

    # Format honest scores (explicit "unscored" where labels don't exist)
    has_attached_score = eligible_visible > 0 and cat != "real_show3d_60fps"
    attached_frac: float | str = (
        round(float(attached_visible) / float(eligible_visible), 4) if has_attached_score else "unscored"
    )
    landmark_med: float | str = (
        round(float(np.median(landmark_errs_720)), 3) if landmark_errs_720 else "unscored"
    )
    landmark_p95: float | str = (
        round(float(np.percentile(landmark_errs_720, 95)), 3) if landmark_errs_720 else "unscored"
    )
    leakage_mean: float | str = (
        round(float(np.mean(leakage_vals)), 4) if leakage_vals else "unscored"
    )
    bbox_leak_mean: float | str = (
        round(float(np.mean(bbox_outside_leakage_vals)), 4) if bbox_outside_leakage_vals else "unscored"
    )
    centroid_med: float | str = (
        round(float(np.median(centroid_offsets_720)), 3) if centroid_offsets_720 else "unscored"
    )
    occ_ghost_max: float | str = (
        round(float(np.max(occluded_mads)), 3) if occluded_mads else "n/a"
    )
    first_reacq: int | None | str = (
        reacq_delays[0] if reacq_delays else "n/a"
    )

    return {
        "eligible_visible_frames": eligible_visible,
        "attached_visible_frames": attached_visible if has_attached_score else "unscored",
        "attached_visible_frame_fraction": attached_frac,
        "landmark_error_720p_median_px": landmark_med,
        "landmark_error_720p_p95_px": landmark_p95,
        "overlay_leakage_fraction": leakage_mean,
        "bbox_outside_leakage_lower_bound": bbox_leak_mean,
        "centroid_offset_720p_median_px": centroid_med,
        "occluded_ghost_max_mad": occ_ghost_max,
        "frames_to_first_correct_reacquisition": first_reacq,
    }


def _save_side_by_side_strip(raw_mp4: Path, out_mp4: Path, strip_png: Path, n_frames: int) -> None:
    """Save a 4-frame side-by-side (Top: Raw, Bottom: Output) visual inspection strip."""
    indices = sorted({0, max(1, n_frames // 4), max(2, n_frames // 2), max(3, (3 * n_frames) // 4)})
    cap_r = cv2.VideoCapture(str(raw_mp4))
    cap_o = cv2.VideoCapture(str(out_mp4))
    raw_s, out_s = {}, {}
    try:
        idx = 0
        max_i = max(indices)
        while idx <= max_i:
            okr, fr = cap_r.read()
            oko, fo = cap_o.read()
            if not okr or not oko:
                break
            if idx in indices:
                raw_s[idx] = cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA)
                out_s[idx] = cv2.resize(fo, (320, 180), interpolation=cv2.INTER_AREA)
            idx += 1
    finally:
        cap_r.release()
        cap_o.release()

    cols = []
    for idx in indices:
        if idx in raw_s and idx in out_s:
            r_img = raw_s[idx].copy()
            o_img = out_s[idx].copy()
            cv2.putText(r_img, f"RAW f={idx}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(o_img, f"OUT f={idx}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)
            cols.append(np.vstack([r_img, o_img]))
    if cols:
        strip_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(strip_png), np.hstack(cols))


def _print_comparison(cur: dict[str, Any], prev: dict[str, Any]) -> None:
    print("\n================ ITERATION DELTA COMPARISON ================")
    print(f"Previous: {prev['tag']}  -->  Current: {cur['tag']}")
    prev_by_id = {c["id"]: c for c in prev["clips"]}
    cur_by_id = {c["id"]: c for c in cur["clips"]}

    print(f"{'Clip ID':30s} {'Res':5s} {'FPS(old->new)':17s} {'ΔFPS':8s} {'TrackMs':14s} {'AttCov(old->new)':19s} {'Leak(old->new)':17s}")
    print("-" * 116)
    fps_deltas, cov_deltas, leak_deltas = [], [], []
    for cid, c in cur_by_id.items():
        p = prev_by_id.get(cid)
        if not p:
            continue
        f_old, f_new = float(p["processed_fps_mean"]), float(c["processed_fps_mean"])
        df = f_new - f_old
        fps_deltas.append(df)
        t_old = float(p["stage_latency_mean_ms"]["track_ms"])
        t_new = float(c["stage_latency_mean_ms"]["track_ms"])
        a_old = p["artwork_score"]["attached_visible_frame_fraction"]
        a_new = c["artwork_score"]["attached_visible_frame_fraction"]
        l_old = p["artwork_score"]["overlay_leakage_fraction"]
        l_new = c["artwork_score"]["overlay_leakage_fraction"]
        if isinstance(a_old, (int, float)) and isinstance(a_new, (int, float)):
            cov_deltas.append(a_new - a_old)
            a_str = f"{a_old*100:5.1f}%->{a_new*100:5.1f}%"
        else:
            a_str = "unscored"
        if isinstance(l_old, (int, float)) and isinstance(l_new, (int, float)):
            leak_deltas.append(l_new - l_old)
            l_str = f"{l_old*100:4.1f}%->{l_new*100:4.1f}%"
        else:
            l_str = "unscored"
        print(
            f"{cid:30s} {c['resolution_bucket']:5s} {f_old:6.1f}->{f_new:6.1f}  {df:+7.2f} "
            f"{t_old:5.2f}->{t_new:5.2f}ms  {a_str:19s} {l_str:17s}"
        )

    print("-" * 116)
    print(
        f"Mean ΔFPS: {np.mean(fps_deltas):+.2f} | "
        f"Mean ΔAttachedCov: {(np.mean(cov_deltas)*100 if cov_deltas else 0.0):+.2f} pp | "
        f"Mean ΔLeakage: {(np.mean(leak_deltas)*100 if leak_deltas else 0.0):+.2f} pp"
    )
    print("============================================================\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate visualizeIT video pipeline across dataset splits.")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / ".cache" / "benchmark-dataset" / "manifest_v2.json")
    parser.add_argument("--gt", type=Path, default=REPO_ROOT / ".cache" / "benchmark-dataset" / "ground_truth_eval_v2.json")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--tag", required=True, help="Iteration tag (e.g., iter-000, iter-001)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tracking-height", type=int, default=360)
    parser.add_argument("--repeats", type=int, default=1, help="Repeated runs per clip (use 3 for 720p30 certification)")
    parser.add_argument("--spatial-bucketing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-appearance-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-occlusion-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sharp-onboarding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grazing-angle-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overlap-io", action="store_true")
    parser.add_argument("--offline-refine", action="store_true")
    parser.add_argument("--live-replay", action="store_true", help="Also run 30 Hz live-replay latency benchmark")
    parser.add_argument("--compare", type=Path, default=None, help="Previous summary.json to diff against")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "artifacts" / "video-benchmark")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    gt_bundle = json.loads(args.gt.read_text(encoding="utf-8"))
    base_dir = args.manifest.resolve().parent

    selected_clips = [
        c for c in manifest["clips"]
        if args.split == "all" or c["split"] == args.split
    ]
    out_dir = args.out_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_vid_dir = REPO_ROOT / ".cache" / "tmp" / "eval_runs" / args.tag
    tmp_vid_dir.mkdir(parents=True, exist_ok=True)

    design = DesignSpec(
        source="recipe",
        pattern="stripes",
        colors=["#ff006e", "#00f5d4", "#ffbe0b"],
        text="VIZIT",
    )
    pipeline = VideoPipeline()

    print(
        f"[{args.tag}] Evaluating {len(selected_clips)} clips (split={args.split}, threads={args.threads}, "
        f"tracking_h={args.tracking_height}, spatial_bucketing={args.spatial_bucketing}, "
        f"adaptive_appearance_gate={args.adaptive_appearance_gate}, "
        f"local_occlusion_mask={args.local_occlusion_mask}, "
        f"sharp_onboarding={args.sharp_onboarding}, "
        f"grazing_angle_gate={args.grazing_angle_gate}, "
        f"overlap_io={args.overlap_io}, offline_refine={args.offline_refine}, repeats={args.repeats})",
        flush=True,
    )

    clip_results: list[dict[str, Any]] = []
    for idx, clip in enumerate(selected_clips, start=1):
        cid = clip["id"]
        raw_path = Path(clip["path"])
        wrap_mode = clip["recommended_wrap"]
        sel = SelectionSpec.from_dict(clip["default_selection"])
        placement = PlacementSpec(wrap_mode=wrap_mode, blend_mode="luminance", opacity=0.90)
        mesh_spec = clip.get("mesh_ply") if wrap_mode == "mesh_6dof" else None

        # Intrinsics for 6-DoF synthetic box or real clips
        intr: tuple[float, float, float, float] | None = None
        if wrap_mode == "mesh_6dof":
            w, h = int(clip["width"]), int(clip["height"])
            f = float(w) * 1.15
            intr = (f, f, w * 0.5, h * 0.5)

        out_mp4 = tmp_vid_dir / f"{cid}.mp4"
        fps_runs: list[float] = []
        last_rep = None
        last_trace: list[dict[str, Any]] = []

        # Repeat 3x automatically on designated native-720p clips if --repeats >= 1, or exact args.repeats
        n_rep = max(1, int(args.repeats))
        for r in range(n_rep):
            trace: list[dict[str, Any]] = []
            rep = pipeline.process_video(
                input_path=raw_path,
                output_path=out_mp4,
                design=design,
                placement=placement,
                selection=sel,
                mesh_spec=mesh_spec,
                intrinsics=intr,
                tracking_height=args.tracking_height,
                threads=args.threads,
                raster_backend="native_cpp",
                spatial_bucketing=bool(args.spatial_bucketing),
                adaptive_appearance_gate=bool(args.adaptive_appearance_gate),
                local_occlusion_mask=bool(args.local_occlusion_mask),
                sharp_onboarding=bool(args.sharp_onboarding),
                grazing_angle_gate=bool(args.grazing_angle_gate),
                overlap_io=bool(args.overlap_io),
                offline_refine=bool(args.offline_refine),
                trace_records=trace,
            )
            fps_runs.append(float(rep.effective_fps))
            last_rep = rep
            last_trace = trace

        assert last_rep is not None
        # Independent ffprobe frame-count check
        probed_out = probe_video_info(out_mp4)
        frame_eq_ok = (
            int(probed_out["frame_count"])
            == int(last_rep.input_frame_count)
            == int(last_rep.processed_frame_count)
            == int(last_rep.rendered_frame_count)
            == int(last_rep.output_frame_count)
        )

        art_score = _score_clip_artwork(
            clip=clip,
            gt_bundle=gt_bundle,
            base_dir=base_dir,
            raw_mp4=raw_path,
            out_mp4=out_mp4,
            trace=last_trace,
        )

        res_bucket = _resolution_bucket(int(clip["width"]), int(clip["height"]))
        fps_mean = float(np.mean(fps_runs))
        playback_ratio = fps_mean / max(float(clip["fps"]), 1.0)

        entry = {
            "id": cid,
            "split": clip["split"],
            "category": clip["category"],
            "object_name": clip.get("object_name", cid),
            "width": int(clip["width"]),
            "height": int(clip["height"]),
            "resolution_bucket": res_bucket,
            "source_fps": float(clip["fps"]),
            "frame_count": int(last_rep.input_frame_count),
            "wrap_mode": wrap_mode,
            "frame_equality_verified": frame_eq_ok,
            "processed_fps_runs": [round(x, 2) for x in fps_runs],
            "processed_fps_mean": round(fps_mean, 2),
            "processed_fps_min": round(float(min(fps_runs)), 2),
            "playback_speed_ratio": round(playback_ratio, 3),
            "tracking_availability": round(float(last_rep.tracking_availability), 4),
            "tracked_frame_count": int(last_rep.tracked_frame_count),
            "lost_frame_count": int(last_rep.lost_frame_count),
            "recovered_count": int(last_rep.recovered_count),
            "inlier_count_median": float(last_rep.inlier_count_median),
            "reprojection_rmse_px": last_rep.reprojection_rmse_px,
            "frame_latency_ms": last_rep.frame_latency_ms,
            "stage_latency_mean_ms": last_rep.stage_latency_mean_ms,
            "artwork_score": art_score,
            "output_mp4_temp": str(out_mp4),
        }
        clip_results.append(entry)

        att_s = (
            f"{art_score['attached_visible_frame_fraction']*100:5.1f}%"
            if isinstance(art_score["attached_visible_frame_fraction"], float)
            else "unscrd"
        )
        leak_s = (
            f"{art_score['overlay_leakage_fraction']*100:4.1f}%"
            if isinstance(art_score["overlay_leakage_fraction"], float)
            else "unscrd"
        )
        print(
            f"  [{idx:02d}/{len(selected_clips)}] {cid:30s} {res_bucket:5s} {fps_mean:6.1f} FPS "
            f"({playback_ratio:4.2f}x) | trk={last_rep.stage_latency_mean_ms['track_ms']:4.1f}ms "
            f"cmp={last_rep.stage_latency_mean_ms['composite_ms']:4.1f}ms enc={last_rep.stage_latency_mean_ms['encode_ms']:4.1f}ms | "
            f"avail={last_rep.tracking_availability*100:5.1f}% att={att_s} leak={leak_s}",
            flush=True,
        )

    # Optional 30 Hz live-replay benchmark on representative 480p & 720p clips (real + synthetic)
    live_replay_results: dict[str, Any] = {}
    if args.live_replay:
        replay_ids = {
            "syn_planar_480p_30fps",
            "syn_planar_720p_30fps",
            "hope_dyn_obj01_720p",
            "ycb_s59_window1",
            "hope_dyn_obj03_720p",
            "hope_dyn_obj04_720p",
            "syn_planar_1080p_30fps",
            "ycb_s48_window1",
        }
        print(f"\n[{args.tag}] Running 30 Hz live-replay benchmark through shared causal core...", flush=True)
        for clip in selected_clips:
            if clip["id"] in replay_ids:
                lr = replay_live_30hz(
                    input_path=clip["path"],
                    design=design,
                    placement=PlacementSpec(wrap_mode=clip["recommended_wrap"], blend_mode="luminance", opacity=0.90),
                    selection=SelectionSpec.from_dict(clip["default_selection"]),
                    target_hz=30.0,
                    max_frames=60,
                    tracking_height=args.tracking_height,
                    threads=args.threads,
                    raster_backend="native_cpp",
                    spatial_bucketing=bool(args.spatial_bucketing),
                    adaptive_appearance_gate=bool(args.adaptive_appearance_gate),
                    local_occlusion_mask=bool(args.local_occlusion_mask),
                    sharp_onboarding=bool(args.sharp_onboarding),
                    grazing_angle_gate=bool(args.grazing_angle_gate),
                )
                live_replay_results[clip["id"]] = lr
                print(
                    f"  [LIVE 30Hz] {clip['id']:26s} p50={lr['arrival_to_overlay_ready_ms']['p50']:5.2f}ms "
                    f"p95={lr['arrival_to_overlay_ready_ms']['p95']:5.2f}ms "
                    f"misses={lr['deadline_misses']}/{lr['frames_replayed']} max_q={lr['max_queue_depth']}",
                    flush=True,
                )

    # Export worst-case Dev MP4s & side-by-side PNG strips (strictly on dev clips!)
    worst_dir = out_dir / "worst_dev"
    dev_scored = [
        c for c in clip_results
        if c["split"] == "dev" and isinstance(c["artwork_score"]["attached_visible_frame_fraction"], float)
    ]
    dev_scored.sort(key=lambda c: (c["artwork_score"]["attached_visible_frame_fraction"], -c["processed_fps_mean"]))
    worst_ids = []
    for c in dev_scored[:5]:
        worst_ids.append(c["id"])
        raw_p = Path([m["path"] for m in selected_clips if m["id"] == c["id"]][0])
        out_p = Path(c["output_mp4_temp"])
        worst_dir.mkdir(parents=True, exist_ok=True)
        # Copy short baseline output MP4 and render side-by-side PNG strip
        (worst_dir / f"{c['id']}_out.mp4").write_bytes(out_p.read_bytes())
        _save_side_by_side_strip(raw_p, out_p, worst_dir / f"{c['id']}_strip.png", c["frame_count"])

    # Aggregate summaries by resolution and category
    by_res: dict[str, dict[str, Any]] = {}
    for bucket in ("360p", "480p", "720p", "1080p"):
        sub = [c for c in clip_results if c["resolution_bucket"] == bucket]
        if sub:
            by_res[bucket] = {
                "count": len(sub),
                "fps_mean": round(float(np.mean([c["processed_fps_mean"] for c in sub])), 2),
                "fps_min": round(float(min(c["processed_fps_min"] for c in sub)), 2),
                "all_ge_15fps": all(c["processed_fps_min"] >= 15.0 for c in sub),
                "all_ge_30fps": all(c["processed_fps_min"] >= 30.0 for c in sub),
                "stage_ms_mean": {
                    k: round(float(np.mean([c["stage_latency_mean_ms"][k] for c in sub])), 3)
                    for k in ("decode_ms", "prep_ms", "track_ms", "composite_ms", "encode_ms")
                },
            }

    by_cat: dict[str, dict[str, Any]] = {}
    for cat_name in sorted({c["category"] for c in clip_results}):
        sub = [c for c in clip_results if c["category"] == cat_name]
        atts = [c["artwork_score"]["attached_visible_frame_fraction"] for c in sub if isinstance(c["artwork_score"]["attached_visible_frame_fraction"], float)]
        leaks = [c["artwork_score"]["overlay_leakage_fraction"] for c in sub if isinstance(c["artwork_score"]["overlay_leakage_fraction"], float)]
        errs = [c["artwork_score"]["landmark_error_720p_median_px"] for c in sub if isinstance(c["artwork_score"]["landmark_error_720p_median_px"], float)]
        by_cat[cat_name] = {
            "count": len(sub),
            "fps_mean": round(float(np.mean([c["processed_fps_mean"] for c in sub])), 2),
            "tracking_availability_mean": round(float(np.mean([c["tracking_availability"] for c in sub])), 4),
            "attached_visible_mean": round(float(np.mean(atts)), 4) if atts else "unscored",
            "overlay_leakage_mean": round(float(np.mean(leaks)), 4) if leaks else "unscored",
            "landmark_error_720p_median_mean": round(float(np.mean(errs)), 3) if errs else "unscored",
        }

    summary = {
        "schema": "visualizeit_video_benchmark_eval_v1",
        "tag": args.tag,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git_info(),
        "host": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "threads": args.threads,
            "tracking_height": args.tracking_height,
            "encoder": "libx264 -preset ultrafast -crf 18 -g 30 -pix_fmt yuv420p",
        },
        "config": {
            "split": args.split,
            "repeats": args.repeats,
            "spatial_bucketing": bool(args.spatial_bucketing),
            "adaptive_appearance_gate": bool(args.adaptive_appearance_gate),
            "local_occlusion_mask": bool(args.local_occlusion_mask),
            "sharp_onboarding": bool(args.sharp_onboarding),
            "grazing_angle_gate": bool(args.grazing_angle_gate),
            "overlap_io": bool(args.overlap_io),
            "offline_refine": bool(args.offline_refine),
            "manifest_path": str(args.manifest),
            "manifest_sha256": _sha256(args.manifest),
            "gt_sha256": _sha256(args.gt),
        },
        "summary_by_resolution": by_res,
        "summary_by_category": by_cat,
        "live_replay_30hz": live_replay_results,
        "worst_dev_clips_exported": worst_ids,
        "clips": clip_results,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    ledger_path = args.out_root / "ledger.jsonl"
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "tag": args.tag,
            "timestamp_utc": summary["timestamp_utc"],
            "config": summary["config"],
            "summary_by_resolution": by_res,
            "summary_by_category": by_cat,
            "live_replay_30hz": live_replay_results,
        }) + "\n")

    print("\n--- Summary by Resolution ---")
    for b, st in by_res.items():
        print(
            f"  {b:5s} ({st['count']:2d} clips): mean={st['fps_mean']:6.1f} FPS, min={st['fps_min']:6.1f} FPS "
            f"[>=15: {st['all_ge_15fps']}, >=30: {st['all_ge_30fps']}] stages={st['stage_ms_mean']}"
        )
    print("\n--- Summary by Category ---")
    for cn, st in by_cat.items():
        print(
            f"  {cn:22s} ({st['count']:2d}): fps={st['fps_mean']:6.1f} avail={st['tracking_availability_mean']*100:5.1f}% "
            f"att={st['attached_visible_mean']} leak={st['overlay_leakage_mean']} err720={st['landmark_error_720p_median_mean']}"
        )
    print(f"\nSaved summary: {summary_path}")

    if args.compare is not None and args.compare.is_file():
        prev = json.loads(args.compare.read_text(encoding="utf-8"))
        _print_comparison(summary, prev)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
