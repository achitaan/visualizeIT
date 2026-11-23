#!/usr/bin/env python3
"""Upgrade `.cache/benchmark-dataset/` with:
  1. Leakage-free scene-grouped Dev / Holdout split metadata (preserving all 45 original clips intact).
  2. 4 real native-720p color handheld recordings from NVIDIA/BOP HOPE Dynamic
     (`hope_onboarding_dynamic.zip`, exact 16:9 1920x1080 -> 1280x720 @ 30 FPS, 180-210 frames each)
     complete with real `mask_visib` masks (excluding human hands) and 6D poses for honest
     artwork leakage and attachment scoring.
  3. Exact `source_frame_ids`, `timestamps_s`, `available_labels`, and `prior_repo_inspected`
     records written to `.cache/benchmark-dataset/manifest_v2.json` and
     `.cache/benchmark-dataset/ground_truth_eval_v2.json` (preserving original v1 files unchanged).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.remote_zip import RemoteZip
from visualizeit.demo_assets import _FFmpegH264Writer

HOPE_REVISION = "ddd0a26ca3460085e93b648748160fb25e4b5566"
HOPE_DYNAMIC_ZIP_URL = (
    f"https://huggingface.co/datasets/bop-benchmark/hope/resolve/{HOPE_REVISION}/hope_onboarding_dynamic.zip"
)

# 4 real handheld color recordings (1920x1080 -> 1280x720 exact 16:9):
#   - 2 in dev (obj_000001 rotation+hand, obj_000002 rotation+hand+view-exit segment)
#   - 2 in holdout (obj_000003, obj_000004) — never opened during dev tuning
HOPE_DYN_SPECS: list[dict[str, Any]] = [
    {
        "id": "hope_dyn_obj01_720p",
        "scene": "obj_000001",
        "obj_name": "alphabet_soup_can",
        "wrap_mode": "cylinder",
        "split": "dev",
        "start_frame": 0,
        "num_frames": 200,
        "notes": "Real handheld 720p color rotation with hand crossing/holding can and real mask_visib.",
    },
    {
        "id": "hope_dyn_obj02_720p",
        "scene": "obj_000002",
        "obj_name": "bbq_sauce_bottle",
        "wrap_mode": "cylinder",
        "split": "dev",
        "start_frame": 0,
        "num_frames": 200,
        "notes": "Real handheld 720p color bottle rotation with hand grip and real mask_visib.",
    },
    {
        "id": "hope_dyn_obj03_720p",
        "scene": "obj_000003",
        "obj_name": "butter_box",
        "wrap_mode": "planar",
        "split": "holdout",
        "start_frame": 0,
        "num_frames": 200,
        "notes": "Unseen real handheld 720p color planar box rotation with hand and real mask_visib.",
    },
    {
        "id": "hope_dyn_obj04_720p",
        "scene": "obj_000004",
        "obj_name": "cherries_can",
        "wrap_mode": "cylinder",
        "split": "holdout",
        "start_frame": 0,
        "num_frames": 200,
        "notes": "Unseen real handheld 720p color can rotation with hand and real mask_visib.",
    },
]

# Fixed scene-grouped Dev / Holdout assignments (zero scene leakage across splits):
# - YCB scenes 48, 52, 55, 59 were inspected in prior repo experiments -> strictly DEV.
# - YCB scenes 48, 49, 54, 59 each have 2 windows; BOTH windows stay on the same side!
YCB_SCENE_SPLIT: dict[str, tuple[str, bool]] = {
    # scene_id -> (split, prior_repo_inspected)
    "000048": ("dev", True),
    "000050": ("dev", False),
    "000052": ("dev", True),
    "000055": ("dev", True),
    "000059": ("dev", True),
    "000049": ("holdout", False),
    "000051": ("holdout", False),
    "000053": ("holdout", False),
    "000054": ("holdout", False),
    "000056": ("holdout", False),
    "000057": ("holdout", False),
    "000058": ("holdout", False),
}

SHOW3D_SPLIT: dict[str, tuple[str, bool]] = {
    "show3d_keyboard_60fps": ("dev", True),
    "show3d_mug_60fps": ("dev", True),
    "show3d_ranch_bottle_60fps": ("dev", True),
    "show3d_bbq_bottle_60fps": ("holdout", False),
}

SYN_DEV_IDS: set[str] = {
    "syn_planar_360p_30fps",
    "syn_planar_480p_30fps",
    "syn_planar_720p_30fps",
    "syn_planar_fast_motion",
    "syn_planar_steep_perspective",
    "syn_planar_scale_zoom",
    "syn_cylinder_mug_480p",
    "syn_cylinder_bottle_720p",
    "syn_cylinder_tilted_axis",
    "syn_lighting_low_light",
    "syn_lighting_moving_shadow",
    "syn_occlusion_short_5f",
    "syn_occlusion_medium_15f",
    "syn_mesh_box_480p",
    "syn_mesh_box_720p",
}


def _selection_from_mask(mask_u8: np.ndarray) -> dict[str, Any]:
    h, w = mask_u8.shape[:2]
    ys, xs = np.nonzero(mask_u8 > 0)
    if len(xs) < 16:
        bx, by, bw, bh = w * 0.3, h * 0.25, w * 0.4, h * 0.5
    else:
        bx, by = float(xs.min()), float(ys.min())
        bw, bh = float(xs.max() - bx + 1.0), float(ys.max() - by + 1.0)
    cx, cy = bx + bw * 0.5, by + bh * 0.5
    # Pick 3 positive points strictly inside the mask
    pos = []
    for fx, fy in ((0.5, 0.5), (0.38, 0.40), (0.62, 0.60)):
        px = int(np.clip(round(bx + bw * fx), 0, w - 1))
        py = int(np.clip(round(by + bh * fy), 0, h - 1))
        if mask_u8[py, px] > 0:
            pos.append([float(px), float(py)])
    if not pos and len(xs) > 0:
        mid = len(xs) // 2
        pos.append([float(xs[mid]), float(ys[mid])])
    neg = [
        [round(max(12.0, bx - 48.0), 1), round(max(12.0, by - 48.0), 1)],
        [round(min(float(w - 12), bx + bw + 48.0), 1), round(min(float(h - 12), by + bh + 48.0), 1)],
    ]
    return {
        "frame_index": 0,
        "positive_points": pos,
        "negative_points": neg,
        "bbox": [round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)],
    }


def _acquire_hope_dynamic_720p(
    base_dir: Path,
    max_workers: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hope_dir = base_dir / "real_hope_720p"
    hope_dir.mkdir(parents=True, exist_ok=True)
    print(f"[HOPE-720p] Opening remote zip index ({HOPE_REVISION[:10]})...", flush=True)
    rz = RemoteZip(HOPE_DYNAMIC_ZIP_URL)

    clips_meta: list[dict[str, Any]] = []
    gt_out: dict[str, Any] = {}

    for idx, sp in enumerate(HOPE_DYN_SPECS, start=1):
        cid = sp["id"]
        scene = sp["scene"]
        start_f = int(sp["start_frame"])
        n_frames = int(sp["num_frames"])
        mp4_path = hope_dir / f"{cid}.mp4"
        masks_dir = hope_dir / f"{cid}_mask_visib"
        masks_dir.mkdir(parents=True, exist_ok=True)

        gt_raw = json.loads(rz.read(f"onboarding_dynamic/{scene}/scene_gt.json").decode("utf-8"))
        cam_raw = json.loads(rz.read(f"onboarding_dynamic/{scene}/scene_camera.json").decode("utf-8"))

        # Find contiguous existing RGB frame indices
        avail_rgb = sorted(
            int(Path(k).stem)
            for k in rz.entries
            if k.startswith(f"onboarding_dynamic/{scene}/rgb/") and k.endswith(".jpg")
        )
        chosen_ids = avail_rgb[start_f : start_f + n_frames]
        actual_n = len(chosen_ids)

        # Find matching mask_visib members (sampled every 5 frames + frame 0 for compact evaluation, or all frames!)
        # Mask PNGs are tiny (~3-8 KB each), so caching all 200 masks takes ~1 MB per clip!
        if not mp4_path.exists() or len(list(masks_dir.glob("*.png"))) < actual_n:
            t0 = time.perf_counter()
            rgb_members = [f"onboarding_dynamic/{scene}/rgb/{fid:06d}.jpg" for fid in chosen_ids]
            mask_members = [f"onboarding_dynamic/{scene}/mask_visib/{fid:06d}_000000.png" for fid in chosen_ids]

            def _fetch_opt(member: str) -> bytes | None:
                if member not in rz.entries:
                    return None
                return rz.read(member)

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                raw_rgbs = list(pool.map(rz.read, rgb_members))
                raw_masks = list(pool.map(_fetch_opt, mask_members))

            first_mask_720: np.ndarray | None = None
            mask_paths_rel: list[str | None] = []
            with _FFmpegH264Writer(mp4_path, 1280, 720, 30.0) as writer:
                for local_i, (fid, raw_rgb, raw_mask) in enumerate(zip(chosen_ids, raw_rgbs, raw_masks)):
                    bgr_1080 = cv2.imdecode(np.frombuffer(raw_rgb, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if bgr_1080 is None:
                        raise RuntimeError(f"Failed to decode HOPE RGB {scene}/{fid}")
                    bgr_720 = cv2.resize(bgr_1080, (1280, 720), interpolation=cv2.INTER_AREA)
                    writer.write(bgr_720)

                    m_path = masks_dir / f"{local_i:06d}.png"
                    if raw_mask is not None:
                        m_1080 = cv2.imdecode(np.frombuffer(raw_mask, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                        if m_1080 is not None:
                            m_720 = cv2.resize(m_1080, (1280, 720), interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(str(m_path), m_720)
                            if local_i == 0:
                                first_mask_720 = m_720
                            mask_paths_rel.append(str(m_path.relative_to(base_dir)))
                            continue
                    mask_paths_rel.append(None)

            dt = time.perf_counter() - t0
            print(
                f"  [{idx}/{len(HOPE_DYN_SPECS)}] {cid} ({sp['obj_name']}, {sp['split']}, 1280x720@{actual_n}f) "
                f"-> {mp4_path.stat().st_size / 1024:.0f} KiB in {dt:.1f}s",
                flush=True,
            )
        else:
            first_mask_720 = cv2.imread(str(masks_dir / "000000.png"), cv2.IMREAD_GRAYSCALE)
            mask_paths_rel = [
                str((masks_dir / f"{i:06d}.png").relative_to(base_dir))
                if (masks_dir / f"{i:06d}.png").is_file()
                else None
                for i in range(actual_n)
            ]
            print(f"  [{idx}/{len(HOPE_DYN_SPECS)}] {cid} (cached)", flush=True)

        assert first_mask_720 is not None
        sel0 = _selection_from_mask(first_mask_720)

        # Scale intrinsics from 1920x1080 -> 1280x720 (exact factor 2/3)
        scale_xy = 1280.0 / 1920.0
        cam0 = cam_raw.get(str(chosen_ids[0]), {})
        k_1080 = cam0.get("cam_K")
        k_720 = None
        if isinstance(k_1080, list) and len(k_1080) == 9:
            k_720 = [
                k_1080[0] * scale_xy, 0.0, k_1080[2] * scale_xy,
                0.0, k_1080[4] * scale_xy, k_1080[5] * scale_xy,
                0.0, 0.0, 1.0,
            ]

        gt_frames = []
        for local_i, fid in enumerate(chosen_ids):
            objs = gt_raw.get(str(fid), [])
            pose = objs[0] if objs else {}
            m_rel = mask_paths_rel[local_i]
            m_bbox = None
            m_area = 0
            if m_rel is not None:
                m_img = cv2.imread(str(base_dir / m_rel), cv2.IMREAD_GRAYSCALE)
                if m_img is not None:
                    ys, xs = np.nonzero(m_img > 0)
                    m_area = int(len(xs))
                    if m_area > 0:
                        m_bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
            gt_frames.append(
                {
                    "frame_index": local_i,
                    "source_frame_id": fid,
                    "timestamp_s": round(local_i / 30.0, 6),
                    "mask_visib_relpath": m_rel,
                    "mask_visib_pixels": m_area,
                    "bbox_visib": m_bbox,
                    "cam_R_m2c": pose.get("cam_R_m2c"),
                    "cam_t_m2c": pose.get("cam_t_m2c"),
                }
            )

        gt_out[cid] = {
            "source": "bop_hope_dynamic",
            "scene_id": scene,
            "object_name": sp["obj_name"],
            "cam_K_720p": k_720,
            "frames": gt_frames,
        }

        clips_meta.append(
            {
                "id": cid,
                "split": sp["split"],
                "split_group": f"hope_dynamic:{scene}",
                "prior_repo_inspected": False,
                "category": "real_hope_720p",
                "source_dataset": "NVIDIA/BOP HOPE Dynamic (CC BY-NC-SA 4.0)",
                "object_name": sp["obj_name"],
                "scene_id": scene,
                "path": str(mp4_path),
                "width": 1280,
                "height": 720,
                "fps": 30.0,
                "frame_count": actual_n,
                "source_frame_ids": chosen_ids,
                "timestamps_s": [round(i / 30.0, 6) for i in range(actual_n)],
                "available_labels": ["mask_visib", "bbox_visib", "pose_6d", "cam_K"],
                "recommended_wrap": sp["wrap_mode"],
                "default_selection": sel0,
                "notes": sp["notes"],
            }
        )

    return clips_meta, gt_out


def main() -> int:
    base_dir = REPO_ROOT / ".cache" / "benchmark-dataset"
    v1_manifest_path = base_dir / "manifest.json"
    v1_gt_path = base_dir / "ground_truth_eval.json"
    if not v1_manifest_path.is_file() or not v1_gt_path.is_file():
        raise FileNotFoundError("Run scripts/prepare_benchmark_dataset.py first to build v1 dataset.")

    v1_manifest = json.loads(v1_manifest_path.read_text(encoding="utf-8"))
    v1_gt = json.loads(v1_gt_path.read_text(encoding="utf-8"))

    hope_clips, hope_gt = _acquire_hope_dynamic_720p(base_dir)

    upgraded_clips: list[dict[str, Any]] = []
    for c in v1_manifest["clips"]:
        entry = dict(c)
        cid = entry["id"]
        cat = entry["category"]
        n_frames = int(entry["frame_count"])
        fps = float(entry["fps"])

        if cat == "real_ycb_video":
            sid = str(entry["scene_id"])
            split, inspected = YCB_SCENE_SPLIT[sid]
            ygt = v1_gt["ycb"].get(cid, {})
            f_ids = [
                int(fr["source_frame_id"]) if fr and "source_frame_id" in fr else i
                for i, fr in enumerate(ygt.get("frames") or [])
            ]
            entry["split"] = split
            entry["split_group"] = f"ycb_scene:{sid}"
            entry["prior_repo_inspected"] = inspected
            entry["source_frame_ids"] = f_ids if len(f_ids) == n_frames else list(range(n_frames))
            entry["timestamps_s"] = [round(i / fps, 6) for i in range(n_frames)]
            entry["available_labels"] = ["bbox_visib", "visib_fract", "pose_6d", "cam_K"]
        elif cat == "real_show3d_60fps":
            split, inspected = SHOW3D_SPLIT[cid]
            entry["split"] = split
            entry["split_group"] = f"show3d_recording:{entry['scene_id']}"
            entry["prior_repo_inspected"] = inspected
            entry["source_frame_ids"] = list(range(n_frames))
            entry["timestamps_s"] = [round(i / fps, 6) for i in range(n_frames)]
            entry["available_labels"] = []  # Unscored accuracy (throughput & stability telemetry only)
        else:
            split = "dev" if cid in SYN_DEV_IDS else "holdout"
            entry["split"] = split
            entry["split_group"] = f"synthetic_family:{cid}"
            entry["prior_repo_inspected"] = False
            entry["source_frame_ids"] = list(range(n_frames))
            entry["timestamps_s"] = [round(i / fps, 6) for i in range(n_frames)]
            entry["available_labels"] = (
                ["quad_gt", "occlusion_gt"] if entry["recommended_wrap"] == "planar" else ["tvec_gt", "occlusion_gt"]
            )
        upgraded_clips.append(entry)

    all_clips = upgraded_clips + hope_clips

    # Verify strict zero-leakage invariant: no split_group appears in both dev and holdout!
    group_splits: dict[str, set[str]] = {}
    for c in all_clips:
        group_splits.setdefault(c["split_group"], set()).add(c["split"])
    leaks = {g: sorted(s) for g, s in group_splits.items() if len(s) > 1}
    if leaks:
        raise RuntimeError(f"Split leakage detected across groups: {leaks}")

    dev_count = sum(1 for c in all_clips if c["split"] == "dev")
    holdout_count = sum(1 for c in all_clips if c["split"] == "holdout")

    manifest_v2 = {
        "schema": "visualizeit_benchmark_dataset_v2",
        "v1_manifest_sha256": hashlib.sha256(v1_manifest_path.read_bytes()).hexdigest(),
        "total_clips": len(all_clips),
        "split_counts": {"dev": dev_count, "holdout": holdout_count},
        "counts_by_category": {
            "real_ycb_video": sum(1 for c in all_clips if c["category"] == "real_ycb_video"),
            "real_show3d_60fps": sum(1 for c in all_clips if c["category"] == "real_show3d_60fps"),
            "real_hope_720p": len(hope_clips),
            "synthetic_stress": sum(1 for c in all_clips if c["category"].startswith("synthetic_")),
        },
        "zero_scene_leakage_verified": True,
        "clips": all_clips,
    }

    gt_v2 = {
        "evaluation_only": True,
        "v1_ground_truth_sha256": hashlib.sha256(v1_gt_path.read_bytes()).hexdigest(),
        "ycb": v1_gt["ycb"],
        "hope_720p": hope_gt,
        "synthetic": v1_gt["synthetic"],
    }

    m2_path = base_dir / "manifest_v2.json"
    gt2_path = base_dir / "ground_truth_eval_v2.json"
    m2_path.write_text(json.dumps(manifest_v2, indent=2) + "\n", encoding="utf-8")
    gt2_path.write_text(json.dumps(gt_v2, indent=2) + "\n", encoding="utf-8")

    print(
        f"\nDONE: manifest_v2.json written ({len(all_clips)} total clips: {dev_count} dev, {holdout_count} holdout; "
        f"zero scene leakage verified).\n"
        f"  manifest_v2.json SHA256: {hashlib.sha256(m2_path.read_bytes()).hexdigest()}\n"
        f"  ground_truth_eval_v2.json SHA256: {hashlib.sha256(gt2_path.read_bytes()).hexdigest()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
