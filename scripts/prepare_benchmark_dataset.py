#!/usr/bin/env python3
"""Prepare a comprehensive 45-video benchmark dataset for visualizeIT.

Creates `.cache/benchmark-dataset/` with:
  1. 16 real-world YCB-Video RGB sequences (640x480 @ 30 FPS, 90 frames each)
     across all 12 BOP YCB-Video test scenes (000048..000059) with ground-truth
     bboxes and 6D poses extracted via HTTP range requests.
  2. 4 real-world SHOW3D 60 FPS handheld inspection sequences (640x800 @ 60 FPS,
     180 frames each) streamed from Hugging Face.
  3. 25 parametric synthetic stress-test clips covering planar, cylindrical, and
     3D mesh surfaces across resolutions (360p..1080p), frame rates (30/60 FPS),
     motion speeds, perspective tilts, jitter, lighting/shadows/glare, and
     short/medium/long/partial occlusions, with exact analytical ground truth.

All assets are indexed in `.cache/benchmark-dataset/manifest.json`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.remote_zip import RemoteZip
from visualizeit.demo_assets import (
    _FFmpegH264Writer,
    _build_box_mesh_and_atlas,
    _build_cylinder_mesh,
    _build_feature_rich_poster,
    _build_studio_background,
    _composite_warped_poster_roi,
    _crop_breathing_bg,
    _project_planar_quad,
    _rot_x,
    _rot_y,
    _rot_z,
    _write_ascii_ply,
)
from visualizeit.native_raster import rasterize_mesh_rgbd

YCBV_REVISION = "5c2c4aa229800355648cd268040aa814f8dc94f0"
YCBV_TEST_ZIP_URL = f"https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/{YCBV_REVISION}/ycbv_test_all.zip"

SHOW3D_REVISION = "720104c5793a55aab352ca933f1f73bde8582d3c"
SHOW3D_BASE = f"https://huggingface.co/datasets/facebook/show3d-dataset/resolve/{SHOW3D_REVISION}/scenes"

YCB_OBJECT_NAMES: dict[int, str] = {
    1: "master_chef_can",
    2: "cracker_box",
    3: "sugar_box",
    4: "tomato_soup_can",
    5: "mustard_bottle",
    6: "tuna_fish_can",
    7: "pudding_box",
    8: "gelatin_box",
    9: "potted_meat_can",
    10: "banana",
    11: "pitcher_base",
    12: "bleach_cleanser",
    13: "bowl",
    14: "mug",
    15: "power_drill",
    16: "wood_block",
    17: "scissors",
    18: "large_marker",
    19: "large_clamp",
    20: "extra_large_clamp",
    21: "foam_brick",
}

CYLINDER_OBJ_IDS = {1, 4, 5, 6, 11, 12, 13, 14, 18}
PLANAR_OBJ_IDS = {2, 3, 7, 8, 9, 16, 21}

# 16 diverse real YCB-Video clips across all 12 test scenes (000048..000059)
YCB_CLIP_SPECS: list[tuple[str, str, int, int]] = [
    # (clip_id, scene_id, start_frame, num_frames)
    ("ycb_s48_window1", "000048", 1, 90),
    ("ycb_s48_window2", "000048", 1400, 90),
    ("ycb_s49_window1", "000049", 1, 90),
    ("ycb_s49_window2", "000049", 900, 90),
    ("ycb_s50_window1", "000050", 1, 90),
    ("ycb_s51_window1", "000051", 1, 90),
    ("ycb_s52_window1", "000052", 1, 90),
    ("ycb_s53_window1", "000053", 1, 90),
    ("ycb_s54_window1", "000054", 1, 90),
    ("ycb_s54_window2", "000054", 900, 90),
    ("ycb_s55_window1", "000055", 1, 90),
    ("ycb_s56_window1", "000056", 1, 90),
    ("ycb_s57_window1", "000057", 1, 90),
    ("ycb_s58_window1", "000058", 1, 90),
    ("ycb_s59_window1", "000059", 1, 90),
    ("ycb_s59_window2", "000059", 400, 90),
]

# 4 real SHOW3D 60 FPS handheld inspection videos
SHOW3D_CLIP_SPECS: list[tuple[str, str, str, str]] = [
    ("show3d_keyboard_60fps", "ASC023/keyboard_inspecting_8039", "rig0.mp4", "planar"),
    ("show3d_mug_60fps", "ASC023/mug_inspecting_817d", "rig0.mp4", "cylinder"),
    ("show3d_ranch_bottle_60fps", "LWA828/ranch_inspecting_ca23", "rig0.mp4", "cylinder"),
    ("show3d_bbq_bottle_60fps", "AAN828/bbq_inspecting_74a3", "rig0.mp4", "cylinder"),
]


def _default_selection_from_bbox(
    bbox: list[float], width: int, height: int
) -> dict[str, Any]:
    x, y, w, h = [float(v) for v in bbox]
    x = max(0.0, min(float(width - 4), x))
    y = max(0.0, min(float(height - 4), y))
    w = max(4.0, min(float(width) - x, w))
    h = max(4.0, min(float(height) - y, h))
    cx, cy = x + w * 0.5, y + h * 0.5
    pos = [
        [round(cx, 1), round(cy, 1)],
        [round(x + w * 0.3, 1), round(y + h * 0.35, 1)],
        [round(x + w * 0.7, 1), round(y + h * 0.65, 1)],
    ]
    neg_x1 = max(8.0, x - 35.0) if x > 45.0 else min(float(width - 8), x + w + 35.0)
    neg_y1 = max(8.0, y - 35.0) if y > 45.0 else min(float(height - 8), y + h + 35.0)
    neg_x2 = min(float(width - 8), x + w + 35.0) if (x + w) < (width - 45.0) else max(8.0, x - 35.0)
    neg_y2 = min(float(height - 8), y + h + 35.0) if (y + h) < (height - 45.0) else max(8.0, y - 35.0)
    return {
        "frame_index": 0,
        "positive_points": pos,
        "negative_points": [
            [round(neg_x1, 1), round(neg_y1, 1)],
            [round(neg_x2, 1), round(neg_y2, 1)],
        ],
        "bbox": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
    }


def _acquire_ycb_clips(out_dir: Path, max_workers: int = 8) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    real_dir = out_dir / "real_ycb"
    real_dir.mkdir(parents=True, exist_ok=True)
    gt_out: dict[str, Any] = {}
    clips_meta: list[dict[str, Any]] = []

    print(f"[YCB] Opening remote zip index ({YCBV_REVISION[:10]})...", flush=True)
    rz = RemoteZip(YCBV_TEST_ZIP_URL)

    # Cache scene_gt and scene_gt_info per scene
    scene_gt_cache: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    needed_scenes = sorted({s for _, s, _, _ in YCB_CLIP_SPECS})

    def _load_scene_meta(scene_id: str) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
        gt = json.loads(rz.read(f"test/{scene_id}/scene_gt.json").decode("utf-8"))
        gt_info = json.loads(rz.read(f"test/{scene_id}/scene_gt_info.json").decode("utf-8"))
        cam = json.loads(rz.read(f"test/{scene_id}/scene_camera.json").decode("utf-8"))
        return scene_id, gt, gt_info, cam

    with ThreadPoolExecutor(max_workers=min(6, len(needed_scenes))) as pool:
        for sid, gt, gt_info, cam in pool.map(_load_scene_meta, needed_scenes):
            scene_gt_cache[sid] = (gt, gt_info, cam)

    for idx, (clip_id, scene_id, start_frame, num_frames) in enumerate(YCB_CLIP_SPECS, start=1):
        mp4_path = real_dir / f"{clip_id}.mp4"
        gt_map, gt_info_map, cam_map = scene_gt_cache[scene_id]

        # Pick the most prominent visible object on start_frame
        f0_key = str(start_frame)
        objs_f0 = gt_map.get(f0_key, [])
        infos_f0 = gt_info_map.get(f0_key, [])
        best_inst_idx = 0
        best_score = -1.0
        for i, (obj, info) in enumerate(zip(objs_f0, infos_f0)):
            visib = float(info.get("visib_fract", 0.0))
            px_visib = float(info.get("px_count_visib", 0.0))
            bx, by, bw, bh = info.get("bbox_visib", [0, 0, 0, 0])
            if bw < 24 or bh < 24:
                continue
            score = visib * px_visib
            if score > best_score:
                best_score = score
                best_inst_idx = i

        target_obj = objs_f0[best_inst_idx]
        target_info = infos_f0[best_inst_idx]
        target_obj_id = int(target_obj["obj_id"])
        target_obj_name = YCB_OBJECT_NAMES.get(target_obj_id, f"obj_{target_obj_id}")
        bbox0 = [float(v) for v in target_info.get("bbox_visib", [220, 160, 200, 160])]
        wrap_mode = "cylinder" if target_obj_id in CYLINDER_OBJ_IDS else "planar"

        if not mp4_path.exists():
            t0 = time.perf_counter()
            members = [f"test/{scene_id}/rgb/{f:06d}.png" for f in range(start_frame, start_frame + num_frames)]
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                raw_frames = list(pool.map(rz.read, members))
            with _FFmpegH264Writer(mp4_path, 640, 480, 30.0) as writer:
                for raw in raw_frames:
                    arr = np.frombuffer(raw, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is None or bgr.shape[:2] != (480, 640):
                        raise RuntimeError(f"Failed to decode YCB frame in {clip_id}")
                    writer.write(bgr)
            dt = time.perf_counter() - t0
            print(
                f"  [{idx:02d}/{len(YCB_CLIP_SPECS)}] {clip_id} ({target_obj_name}, scene {scene_id}) "
                f"-> {mp4_path.stat().st_size / 1024:.0f} KiB in {dt:.1f}s",
                flush=True,
            )
        else:
            print(f"  [{idx:02d}/{len(YCB_CLIP_SPECS)}] {clip_id} (cached)", flush=True)

        # Collect per-frame ground truth for the chosen object
        gt_frames = []
        for offset in range(num_frames):
            fk = str(start_frame + offset)
            f_objs = gt_map.get(fk, [])
            f_infos = gt_info_map.get(fk, [])
            entry = None
            for o, inf in zip(f_objs, f_infos):
                if int(o["obj_id"]) == target_obj_id:
                    entry = {
                        "frame_index": offset,
                        "source_frame_id": start_frame + offset,
                        "bbox_visib": inf.get("bbox_visib"),
                        "visib_fract": inf.get("visib_fract"),
                        "cam_R_m2c": o.get("cam_R_m2c"),
                        "cam_t_m2c": o.get("cam_t_m2c"),
                    }
                    break
            gt_frames.append(entry)

        gt_out[clip_id] = {
            "source": "bop_ycbv",
            "scene_id": scene_id,
            "target_obj_id": target_obj_id,
            "target_obj_name": target_obj_name,
            "cam_K": cam_map.get(f0_key, {}).get("cam_K"),
            "frames": gt_frames,
        }

        clips_meta.append(
            {
                "id": clip_id,
                "category": "real_ycb_video",
                "source_dataset": "BOP YCB-Video",
                "object_name": target_obj_name,
                "scene_id": scene_id,
                "path": str(mp4_path),
                "width": 640,
                "height": 480,
                "fps": 30.0,
                "frame_count": num_frames,
                "recommended_wrap": wrap_mode,
                "default_selection": _default_selection_from_bbox(bbox0, 640, 480),
            }
        )

    return clips_meta, gt_out


def _acquire_show3d_clips(out_dir: Path) -> list[dict[str, Any]]:
    s3d_dir = out_dir / "real_show3d"
    s3d_dir.mkdir(parents=True, exist_ok=True)
    clips_meta: list[dict[str, Any]] = []

    print("[SHOW3D] Streaming 4 real 60 FPS handheld clips via FFmpeg...", flush=True)
    for idx, (clip_id, rel_path, filename, wrap_mode) in enumerate(SHOW3D_CLIP_SPECS, start=1):
        mp4_path = s3d_dir / f"{clip_id}.mp4"
        url = f"{SHOW3D_BASE}/{rel_path}/{filename}"
        if not mp4_path.exists():
            t0 = time.perf_counter()
            tmp_path = s3d_dir / f".pending-{clip_id}.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                url,
                "-frames:v",
                "180",
                "-vf",
                "scale=640:800",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp_path),
            ]
            subprocess.run(cmd, check=True, timeout=180)
            tmp_path.replace(mp4_path)
            dt = time.perf_counter() - t0
            print(
                f"  [{idx}/{len(SHOW3D_CLIP_SPECS)}] {clip_id} -> {mp4_path.stat().st_size / 1024:.0f} KiB in {dt:.1f}s",
                flush=True,
            )
        else:
            print(f"  [{idx}/{len(SHOW3D_CLIP_SPECS)}] {clip_id} (cached)", flush=True)

        # Center-region default selection for handheld inspection
        bbox0 = [170.0, 230.0, 300.0, 340.0]
        clips_meta.append(
            {
                "id": clip_id,
                "category": "real_show3d_60fps",
                "source_dataset": "Facebook SHOW3D",
                "object_name": clip_id.replace("show3d_", "").replace("_60fps", ""),
                "scene_id": rel_path,
                "path": str(mp4_path),
                "width": 640,
                "height": 800,
                "fps": 60.0,
                "frame_count": 180,
                "recommended_wrap": wrap_mode,
                "default_selection": _default_selection_from_bbox(bbox0, 640, 800),
            }
        )

    return clips_meta


def _apply_lighting_effect(
    frame_bgr: np.ndarray,
    t: float,
    effect: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if effect == "none":
        return frame_bgr
    out = frame_bgr.astype(np.float32)
    h, w = out.shape[:2]
    if effect == "low_light":
        out *= 0.42
        noise = rng.normal(0.0, 7.5, size=out.shape).astype(np.float32)
        out += noise
    elif effect == "moving_shadow":
        cx = w * (0.15 + 0.70 * t)
        xs = np.arange(w, dtype=np.float32)[None, :]
        band = np.exp(-0.5 * ((xs - cx) / (w * 0.14)) ** 2)
        shadow = 1.0 - 0.58 * band
        out *= shadow[:, :, None]
    elif effect == "specular_glare":
        cx = w * (0.20 + 0.60 * math.sin(2.0 * math.pi * t) * 0.5 + 0.30)
        cy = h * (0.35 + 0.30 * math.cos(2.0 * math.pi * t))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        glare = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * (min(w, h) * 0.14) ** 2))
        out += glare[:, :, None] * 135.0
    elif effect == "low_contrast":
        mean = out.mean(axis=(0, 1), keepdims=True)
        out = mean + (out - mean) * 0.45
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _draw_occluder(
    frame_bgr: np.ndarray,
    frame_idx: int,
    width: int,
    height: int,
    occ_mode: str,
) -> bool:
    """Draw a textured occluder; returns True if target is fully occluded."""
    if occ_mode == "none":
        return False
    if occ_mode == "short_5f":
        active = 35 <= frame_idx <= 39
        if not active:
            return False
        cv2.rectangle(frame_bgr, (int(width * 0.12), int(height * 0.10)), (int(width * 0.88), int(height * 0.90)), (44, 48, 56), -1)
        return True
    if occ_mode == "medium_15f":
        active = 35 <= frame_idx <= 49
        if not active:
            return False
        cv2.rectangle(frame_bgr, (int(width * 0.10), int(height * 0.08)), (int(width * 0.90), int(height * 0.92)), (38, 44, 52), -1)
        return True
    if occ_mode == "long_30f":
        active = 30 <= frame_idx <= 59
        if not active:
            return False
        cv2.rectangle(frame_bgr, (int(width * 0.08), int(height * 0.06)), (int(width * 0.92), int(height * 0.94)), (32, 38, 46), -1)
        return True
    if occ_mode == "partial_hand":
        # Moving hand/bar covering ~38% of center area from frame 20..75
        if 20 <= frame_idx <= 75:
            p = (frame_idx - 20) / 55.0
            x0 = int(width * (0.28 + 0.18 * math.sin(2.0 * math.pi * p)))
            x1 = x0 + int(width * 0.16)
            cv2.rectangle(frame_bgr, (x0, int(height * 0.15)), (x1, int(height * 0.88)), (118, 148, 185), -1)
            cv2.rectangle(frame_bgr, (x0, int(height * 0.15)), (x1, int(height * 0.88)), (70, 90, 120), 2)
        return False
    return False


def _generate_synthetic_matrix(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    syn_dir = out_dir / "synthetic_stress"
    syn_dir.mkdir(parents=True, exist_ok=True)
    clips_meta: list[dict[str, Any]] = []
    gt_out: dict[str, Any] = {}

    # Shared 3D box mesh PLY for mesh_6dof clips
    box_verts, box_norms, box_uvs, box_vcols, box_faces, box_tex_rgba, _ = _build_box_mesh_and_atlas()
    ply_path = syn_dir / "stress_mesh_box.ply"
    if not ply_path.exists():
        _write_ascii_ply(ply_path, box_verts, box_norms, box_uvs, box_vcols, box_faces)

    # 25 parametric synthetic scenarios
    specs: list[dict[str, Any]] = [
        # Planar (9)
        {"id": "syn_planar_360p_30fps", "kind": "planar", "w": 640, "h": 360, "fps": 30.0, "n": 90, "motion": "normal", "effect": "none", "occ": "none", "seed": 101},
        {"id": "syn_planar_480p_30fps", "kind": "planar", "w": 640, "h": 480, "fps": 30.0, "n": 120, "motion": "normal", "effect": "none", "occ": "none", "seed": 102},
        {"id": "syn_planar_720p_30fps", "kind": "planar", "w": 1280, "h": 720, "fps": 30.0, "n": 120, "motion": "normal", "effect": "none", "occ": "none", "seed": 103},
        {"id": "syn_planar_1080p_30fps", "kind": "planar", "w": 1920, "h": 1080, "fps": 30.0, "n": 90, "motion": "normal", "effect": "none", "occ": "none", "seed": 104},
        {"id": "syn_planar_480p_60fps", "kind": "planar", "w": 640, "h": 480, "fps": 60.0, "n": 150, "motion": "normal", "effect": "none", "occ": "none", "seed": 105},
        {"id": "syn_planar_fast_motion", "kind": "planar", "w": 640, "h": 480, "fps": 30.0, "n": 120, "motion": "fast", "effect": "none", "occ": "none", "seed": 106},
        {"id": "syn_planar_steep_perspective", "kind": "planar", "w": 640, "h": 480, "fps": 30.0, "n": 120, "motion": "steep", "effect": "none", "occ": "none", "seed": 107},
        {"id": "syn_planar_scale_zoom", "kind": "planar", "w": 640, "h": 480, "fps": 30.0, "n": 120, "motion": "zoom", "effect": "none", "occ": "none", "seed": 108},
        {"id": "syn_planar_handheld_jitter", "kind": "planar", "w": 640, "h": 480, "fps": 30.0, "n": 120, "motion": "jitter", "effect": "none", "occ": "none", "seed": 109},
        # Cylinder (5)
        {"id": "syn_cylinder_mug_480p", "kind": "cylinder", "w": 640, "h": 480, "fps": 30.0, "n": 120, "motion": "normal", "effect": "none", "occ": "none", "seed": 201},
        {"id": "syn_cylinder_bottle_720p", "kind": "cylinder", "w": 1280, "h": 720, "fps": 30.0, "n": 90, "motion": "normal", "effect": "none", "occ": "none", "seed": 202},
        {"id": "syn_cylinder_fast_spin", "kind": "cylinder", "w": 640, "h": 480, "fps": 30.0, "n": 120, "motion": "fast", "effect": "none", "occ": "none", "seed": 203},
        {"id": "syn_cylinder_tilted_axis", "kind": "cylinder", "w": 640, "h": 480, "fps": 30.0, "n": 120, "motion": "steep", "effect": "none", "occ": "none", "seed": 204},
        {"id": "syn_cylinder_can_lowlight", "kind": "cylinder", "w": 640, "h": 480, "fps": 30.0, "n": 90, "motion": "normal", "effect": "low_light", "occ": "none", "seed": 205},
        # Lighting & Appearance Stress (4)
        {"id": "syn_lighting_low_light", "kind": "planar", "w": 640, "h": 480, "fps": 30.0, "n": 90, "motion": "normal", "effect": "low_light", "occ": "none", "seed": 301},
        {"id": "syn_lighting_moving_shadow", "kind": "planar", "w": 640, "h": 480, "fps": 30.0, "n": 90, "motion": "normal", "effect": "moving_shadow", "occ": "none", "seed": 302},
        {"id": "syn_lighting_specular_glare", "kind": "planar", "w": 640, "h": 480, "fps": 30.0, "n": 90, "motion": "normal", "effect": "specular_glare", "occ": "none", "seed": 303},
        {"id": "syn_appearance_low_contrast", "kind": "planar", "w": 640, "h": 480, "fps": 30.0, "n": 90, "motion": "normal", "effect": "low_contrast", "occ": "none", "seed": 304},
        # Occlusion & Recovery (4)
        {"id": "syn_occlusion_short_5f", "kind": "planar", "w": 640, "h": 480, "fps": 30.0, "n": 120, "motion": "normal", "effect": "none", "occ": "short_5f", "seed": 401},
        {"id": "syn_occlusion_medium_15f", "kind": "planar", "w": 640, "h": 480, "fps": 30.0, "n": 120, "motion": "normal", "effect": "none", "occ": "medium_15f", "seed": 402},
        {"id": "syn_occlusion_long_30f", "kind": "planar", "w": 640, "h": 480, "fps": 30.0, "n": 120, "motion": "normal", "effect": "none", "occ": "long_30f", "seed": 403},
        {"id": "syn_occlusion_partial_hand", "kind": "planar", "w": 640, "h": 480, "fps": 30.0, "n": 120, "motion": "normal", "effect": "none", "occ": "partial_hand", "seed": 404},
        # 3D Mesh 6-DoF (3)
        {"id": "syn_mesh_box_480p", "kind": "mesh", "w": 640, "h": 480, "fps": 30.0, "n": 120, "motion": "normal", "effect": "none", "occ": "none", "seed": 501},
        {"id": "syn_mesh_box_720p", "kind": "mesh", "w": 1280, "h": 720, "fps": 30.0, "n": 90, "motion": "normal", "effect": "none", "occ": "none", "seed": 502},
        {"id": "syn_mesh_box_occlusion", "kind": "mesh", "w": 640, "h": 480, "fps": 30.0, "n": 120, "motion": "normal", "effect": "none", "occ": "medium_15f", "seed": 503},
    ]

    print(f"[SYNTHETIC] Generating {len(specs)} parametric stress-test clips...", flush=True)
    cyl_verts, cyl_norms, cyl_uvs, cyl_faces = _build_cylinder_mesh()
    cyl_poster = _build_feature_rich_poster(512, 256, 777, "CURVED WRAP", "CYLINDER BENCH", (40, 140, 220))
    cyl_tex_rgba = cv2.cvtColor(cyl_poster, cv2.COLOR_BGR2RGBA)

    for idx, sp in enumerate(specs, start=1):
        cid = sp["id"]
        w, h, fps, n = sp["w"], sp["h"], sp["fps"], sp["n"]
        kind, motion, effect, occ, seed = sp["kind"], sp["motion"], sp["effect"], sp["occ"], sp["seed"]
        mp4_path = syn_dir / f"{cid}.mp4"
        rng = np.random.default_rng(seed)

        t0 = time.perf_counter()
        padded_bg = _build_studio_background(w, h, seed)
        poster = _build_feature_rich_poster(
            512, 384, seed + 17, cid.upper()[:18], f"{w}x{h} {motion.upper()}", (50 + (seed * 37) % 170, 70, 210)
        )
        cx, cy = w * 0.5, h * 0.5
        focal = float(w) * 1.15
        half_w, half_h = w * 0.21, h * 0.22

        gt_frames = []
        bbox0: list[float] | None = None

        if not mp4_path.exists():
            with _FFmpegH264Writer(mp4_path, w, h, fps) as writer:
                for f in range(n):
                    t = f / max(1.0, float(n - 1))
                    frame = _crop_breathing_bg(padded_bg, w, h, f)

                    if kind == "planar":
                        if motion == "fast":
                            dx = w * 0.16 * math.sin(4.0 * math.pi * t)
                            dy = h * 0.12 * math.cos(3.0 * math.pi * t)
                            yaw = 26.0 * math.sin(4.0 * math.pi * t)
                            pitch = 18.0 * math.cos(3.0 * math.pi * t)
                            roll = 20.0 * math.sin(2.0 * math.pi * t)
                            z_scale = 1.0 + 0.12 * math.sin(3.0 * math.pi * t)
                        elif motion == "steep":
                            dx = w * 0.06 * math.sin(2.0 * math.pi * t)
                            dy = h * 0.05 * math.cos(2.0 * math.pi * t)
                            yaw = 52.0 * math.sin(2.0 * math.pi * t)
                            pitch = 34.0 * math.cos(2.0 * math.pi * t)
                            roll = 12.0 * math.sin(2.0 * math.pi * t)
                            z_scale = 1.0
                        elif motion == "zoom":
                            dx = w * 0.05 * math.sin(2.0 * math.pi * t)
                            dy = h * 0.04 * math.cos(2.0 * math.pi * t)
                            yaw = 12.0 * math.sin(2.0 * math.pi * t)
                            pitch = 8.0 * math.cos(2.0 * math.pi * t)
                            roll = 6.0 * math.sin(2.0 * math.pi * t)
                            z_scale = 0.55 + 0.95 * (0.5 - 0.5 * math.cos(2.0 * math.pi * t))
                        elif motion == "jitter":
                            dx = w * 0.06 * math.sin(2.0 * math.pi * t) + float(rng.normal(0.0, w * 0.012))
                            dy = h * 0.05 * math.cos(2.0 * math.pi * t) + float(rng.normal(0.0, h * 0.012))
                            yaw = 14.0 * math.sin(2.0 * math.pi * t) + float(rng.normal(0.0, 2.5))
                            pitch = 10.0 * math.cos(2.0 * math.pi * t) + float(rng.normal(0.0, 2.0))
                            roll = 8.0 * math.sin(2.0 * math.pi * t) + float(rng.normal(0.0, 2.0))
                            z_scale = 1.0
                        else:
                            dx = w * 0.10 * math.sin(2.0 * math.pi * t)
                            dy = h * 0.08 * math.sin(4.0 * math.pi * t) * 0.6
                            yaw = 18.0 * math.sin(2.0 * math.pi * t)
                            pitch = 12.0 * math.cos(2.0 * math.pi * t)
                            roll = 10.0 * math.sin(2.0 * math.pi * t)
                            z_scale = 1.0 + 0.08 * math.sin(2.0 * math.pi * t)

                        if f == 0:
                            dx = dy = yaw = pitch = roll = 0.0
                            z_scale = 1.0

                        quad = _project_planar_quad(half_w, half_h, cx, cy, focal, dx, dy, yaw, pitch, roll, z_scale)
                        _composite_warped_poster_roi(frame, poster, quad, 0.88, 1.06)
                        xs, ys = quad[:, 0], quad[:, 1]
                        bx, by = float(xs.min()), float(ys.min())
                        bw, bh = float(xs.max() - bx), float(ys.max() - by)
                        if f == 0:
                            bbox0 = [bx, by, bw, bh]
                        occluded = _draw_occluder(frame, f, w, h, occ)
                        frame = _apply_lighting_effect(frame, t, effect, rng)
                        gt_frames.append({"frame_index": f, "quad": quad.round(2).tolist(), "occluded": occluded})

                    else:
                        # 3D Cylinder or 3D Box via native C++ AVX-512 rasterizer
                        verts, norms, uvs, faces, tex_rgba = (
                            (cyl_verts, cyl_norms, cyl_uvs, cyl_faces, cyl_tex_rgba)
                            if kind == "cylinder"
                            else (box_verts, box_norms, box_uvs, box_faces, box_tex_rgba)
                        )
                        if f == 0:
                            rx, ry, rz_a, tx, ty, tz = 0.18, 0.25, 0.0, 0.0, 0.0, 0.56
                        elif motion == "fast":
                            rx = 0.25 * math.sin(4.0 * math.pi * t)
                            ry = 2.4 * math.pi * t
                            rz_a = 0.20 * math.cos(3.0 * math.pi * t)
                            tx = 0.06 * math.sin(3.0 * math.pi * t)
                            ty = 0.04 * math.cos(2.0 * math.pi * t)
                            tz = 0.56 + 0.06 * math.sin(2.0 * math.pi * t)
                        elif motion == "steep":
                            rx = 0.65 * math.sin(2.0 * math.pi * t)
                            ry = 1.2 * math.pi * t
                            rz_a = 0.45 * math.cos(2.0 * math.pi * t)
                            tx = 0.04 * math.sin(2.0 * math.pi * t)
                            ty = 0.03 * math.cos(2.0 * math.pi * t)
                            tz = 0.58
                        else:
                            rx = 0.22 * math.sin(2.0 * math.pi * t)
                            ry = 0.25 + 1.4 * math.pi * t
                            rz_a = 0.12 * math.sin(2.0 * math.pi * t)
                            tx = 0.05 * math.sin(2.0 * math.pi * t)
                            ty = 0.03 * math.cos(2.0 * math.pi * t)
                            tz = 0.56 + 0.04 * math.sin(2.0 * math.pi * t)

                        R = _rot_z(rz_a) @ _rot_y(ry) @ _rot_x(rx)
                        tvec = np.array([tx, ty, tz], dtype=np.float32)
                        verts_cam = (verts.astype(np.float32) @ R.astype(np.float32).T) + tvec[None, :]
                        frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        res = rasterize_mesh_rgbd(
                            verts_cam,
                            faces.astype(np.int32),
                            uvs.astype(np.float32),
                            tex_rgba,
                            (focal, focal, cx, cy),
                            frame_rgb,
                            opacity=1.0,
                            blend_mode="normal",
                        )
                        frame = cv2.cvtColor(res.frame_rgb, cv2.COLOR_RGB2BGR)
                        mask_u8 = res.mask
                        if f == 0:
                            ys, xs = np.nonzero(mask_u8)
                            if len(xs) > 0:
                                bbox0 = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())]
                            else:
                                bbox0 = [w * 0.3, h * 0.25, w * 0.4, h * 0.5]
                        occluded = _draw_occluder(frame, f, w, h, occ)
                        frame = _apply_lighting_effect(frame, t, effect, rng)
                        gt_frames.append({"frame_index": f, "tvec": tvec.round(4).tolist(), "occluded": occluded})

                    writer.write(frame)
            dt = time.perf_counter() - t0
            print(
                f"  [{idx:02d}/{len(specs)}] {cid} ({w}x{h}@{int(fps)}fps) "
                f"-> {mp4_path.stat().st_size / 1024:.0f} KiB in {dt:.1f}s",
                flush=True,
            )
        else:
            bbox0 = [w * 0.29, h * 0.28, w * 0.42, h * 0.44]
            print(f"  [{idx:02d}/{len(specs)}] {cid} (cached)", flush=True)

        assert bbox0 is not None
        wrap_mode = "planar" if kind == "planar" else ("cylinder" if kind == "cylinder" else "mesh_6dof")
        entry: dict[str, Any] = {
            "id": cid,
            "category": f"synthetic_{kind}",
            "source_dataset": "visualizeIT Parametric Stress Suite",
            "object_name": cid,
            "motion": motion,
            "effect": effect,
            "occlusion": occ,
            "path": str(mp4_path),
            "width": w,
            "height": h,
            "fps": fps,
            "frame_count": n,
            "recommended_wrap": wrap_mode,
            "default_selection": _default_selection_from_bbox(bbox0, w, h),
        }
        if kind == "mesh":
            entry["mesh_ply"] = str(ply_path)
        clips_meta.append(entry)
        gt_out[cid] = {"source": "synthetic_analytical", "frames": gt_frames}

    return clips_meta, gt_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 45-video comprehensive benchmark dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".cache" / "benchmark-dataset",
        help="Destination directory (default: .cache/benchmark-dataset)",
    )
    args = parser.parse_args()
    out_dir: Path = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    ycb_clips, ycb_gt = _acquire_ycb_clips(out_dir)
    s3d_clips = _acquire_show3d_clips(out_dir)
    syn_clips, syn_gt = _generate_synthetic_matrix(out_dir)

    all_clips = ycb_clips + s3d_clips + syn_clips
    manifest = {
        "schema": "visualizeit_benchmark_dataset_v1",
        "total_clips": len(all_clips),
        "counts_by_category": {
            "real_ycb_video": len(ycb_clips),
            "real_show3d_60fps": len(s3d_clips),
            "synthetic_stress": len(syn_clips),
        },
        "clips": all_clips,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "ground_truth_eval.json").write_text(
        json.dumps({"evaluation_only": True, "ycb": ycb_gt, "synthetic": syn_gt}, indent=2) + "\n",
        encoding="utf-8",
    )

    total_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    elapsed = time.perf_counter() - t_start
    print(
        f"\nDONE: {len(all_clips)} benchmark videos ready in {out_dir} "
        f"({total_bytes / (1024 * 1024):.1f} MiB total, {elapsed:.1f}s elapsed).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
