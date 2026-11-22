"""Fast interactive point/box foreground segmentation onboarding (< 80 ms CPU)."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from .contracts import ContractValidationError, SelectionSpec


@dataclass(slots=True)
class SegmentationResult:
    frame_index: int
    width: int
    height: int
    mask: np.ndarray  # HxW uint8 in {0, 255}
    alpha: np.ndarray  # HxW float32 in [0.0, 1.0]
    foreground_pixels: int
    coverage_ratio: float
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    polygon: list[list[float]]
    anchors: np.ndarray  # Nx2 float32
    anchor_count: int
    init_ms: float
    mask_path: str | None = None
    quality_score: float = 1.0
    init_valid: bool = True
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "width": int(self.width),
            "height": int(self.height),
            "foreground_pixels": int(self.foreground_pixels),
            "coverage_ratio": float(self.coverage_ratio),
            "bbox": [int(v) for v in self.bbox],
            "polygon": [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in self.polygon],
            "anchor_count": int(self.anchor_count),
            "init_ms": round(float(self.init_ms), 3),
            "mask_path": self.mask_path,
            "quality_score": round(float(self.quality_score), 4),
            "init_valid": bool(self.init_valid),
            "rejection_reason": self.rejection_reason,
        }


def _extract_polygon(mask_u8: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    eps = max(1.0, 0.012 * peri)
    approx = cv2.approxPolyDP(largest, eps, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        x, y, w, h = cv2.boundingRect(largest)
        return [
            [float(x), float(y)],
            [float(x + w), float(y)],
            [float(x + w), float(y + h)],
            [float(x), float(y + h)],
        ]
    return [[float(p[0]), float(p[1])] for p in pts]


def _extract_interior_anchors(
    gray: np.ndarray,
    mask_u8: np.ndarray,
    *,
    max_anchors: int = 240,
) -> np.ndarray:
    h, w = gray.shape[:2]
    fg_px = int(np.count_nonzero(mask_u8))
    if fg_px < 16:
        return np.zeros((0, 2), dtype=np.float32)

    # Erode slightly so anchors sit inside the physical surface rather than on background edges
    ksize = 5 if fg_px > 2000 else 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    inner = cv2.erode(mask_u8, kernel, iterations=1)
    if int(np.count_nonzero(inner)) < max(32, fg_px // 5):
        inner = mask_u8

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_anchors,
        qualityLevel=0.01,
        minDistance=5.0,
        mask=inner,
        blockSize=5,
    )
    pts_list: list[np.ndarray] = []
    if corners is not None and len(corners) > 0:
        pts_list.append(corners.reshape(-1, 2).astype(np.float32))

    # If texture is weak, also sample interior grid points so we always have enough anchors
    total = sum(len(a) for a in pts_list)
    if total < 16:
        ys, xs = np.nonzero(inner)
        if len(xs) > 0:
            step = max(1, len(xs) // 48)
            grid = np.column_stack([xs[::step], ys[::step]]).astype(np.float32)
            pts_list.append(grid[:48])

    if not pts_list:
        return np.zeros((0, 2), dtype=np.float32)

    all_pts = np.vstack(pts_list)
    # Clamp strictly within image
    all_pts[:, 0] = np.clip(all_pts[:, 0], 0.0, float(w - 1))
    all_pts[:, 1] = np.clip(all_pts[:, 1], 0.0, float(h - 1))
    return all_pts[:max_anchors].astype(np.float32)


def _keep_selected_components(
    mask_u8: np.ndarray,
    pos_pts: list[tuple[float, float]],
    neg_pts: list[tuple[float, float]],
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8

    h, w = mask_u8.shape[:2]
    chosen_labels: set[int] = set()
    for px, py in pos_pts:
        ix = int(np.clip(round(px), 0, w - 1))
        iy = int(np.clip(round(py), 0, h - 1))
        lbl = int(labels[iy, ix])
        if lbl > 0:
            chosen_labels.add(lbl)

    if not chosen_labels:
        # Pick largest non-background component that does not contain a negative point
        neg_labels: set[int] = set()
        for nx, ny in neg_pts:
            ix = int(np.clip(round(nx), 0, w - 1))
            iy = int(np.clip(round(ny), 0, h - 1))
            lbl = int(labels[iy, ix])
            if lbl > 0:
                neg_labels.add(lbl)
        best_lbl = 0
        best_area = -1
        for lbl in range(1, num_labels):
            if lbl in neg_labels:
                continue
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_lbl = lbl
        if best_lbl == 0:
            # All components touched a negative point; fall back to largest component
            best_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        chosen_labels.add(best_lbl)

    out = np.zeros_like(mask_u8)
    for lbl in chosen_labels:
        out[labels == lbl] = 255
    return out


def _estimate_point_roi(
    frame_bgr: np.ndarray,
    pos_pts: list[tuple[float, float]],
    neg_pts: list[tuple[float, float]],
) -> tuple[int, int, int, int]:
    """Estimate an initial bounding rect for point-only GrabCut using color/gradient cues."""
    h, w = frame_bgr.shape[:2]
    xs = [p[0] for p in pos_pts]
    ys = [p[1] for p in pos_pts]
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    # Compute local distance to negative points if available
    if neg_pts:
        dists = [math.hypot(nx - cx, ny - cy) for nx, ny in neg_pts]
        half_r = max(24.0, 0.72 * min(dists))
        half_w = min(half_r, w * 0.42)
        half_h = min(half_r, h * 0.42)
    else:
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        half_w = max(w * 0.24, spread_x * 0.9 + w * 0.12)
        half_h = max(h * 0.24, spread_y * 0.9 + h * 0.12)

    x0 = int(max(1, math.floor(cx - half_w)))
    y0 = int(max(1, math.floor(cy - half_h)))
    x1 = int(min(w - 1, math.ceil(cx + half_w)))
    y1 = int(min(h - 1, math.ceil(cy + half_h)))
    bw = max(8, x1 - x0)
    bh = max(8, y1 - y0)
    if x0 + bw >= w:
        x0 = max(1, w - 1 - bw)
    if y0 + bh >= h:
        y0 = max(1, h - 1 - bh)
    return (x0, y0, min(bw, w - 1 - x0), min(bh, h - 1 - y0))


def segment_frame(
    frame_bgr: np.ndarray,
    selection: SelectionSpec,
    *,
    output_mask_path: str | os.PathLike[str] | None = None,
    max_work_dim: int = 240,
    grabcut_iters: int = 2,
    sharp_onboarding: bool = False,
) -> SegmentationResult:
    """Segment the target object on a single frame using fast downscaled GrabCut."""
    t0 = time.perf_counter()

    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ContractValidationError("frame_bgr must be an HxWx3 uint8 BGR numpy array")
    if frame_bgr.dtype != np.uint8:
        raise ContractValidationError(f"frame_bgr must be uint8, got {frame_bgr.dtype}")

    h, w = frame_bgr.shape[:2]
    if h < 8 or w < 8:
        raise ContractValidationError(f"Frame too small for segmentation: {w}x{h}")

    selection.validate_for_frame(w, h)
    pos_pts = list(selection.positive_points)
    neg_pts = list(selection.negative_points)
    explicit_box = selection.bbox

    # Optional ROI-cropped GrabCut: focuses the pixel budget around explicit_box for 2x-4x sharper silhouettes
    crop_x0, crop_y0, crop_x1, crop_y1 = 0, 0, w, h
    if sharp_onboarding and explicit_box is not None:
        bx, by, bw, bh = explicit_box
        pad = max(48.0, 0.60 * max(bw, bh))
        all_pts = pos_pts + neg_pts
        min_x = min([bx - pad] + [p[0] - 16.0 for p in all_pts])
        min_y = min([by - pad] + [p[1] - 16.0 for p in all_pts])
        max_x = max([bx + bw + pad] + [p[0] + 16.0 for p in all_pts])
        max_y = max([by + bh + pad] + [p[1] + 16.0 for p in all_pts])
        crop_x0 = int(max(0, math.floor(min_x)))
        crop_y0 = int(max(0, math.floor(min_y)))
        crop_x1 = int(min(w, math.ceil(max_x)))
        crop_y1 = int(min(h, math.ceil(max_y)))
        if (crop_x1 - crop_x0) < 16 or (crop_y1 - crop_y0) < 16:
            crop_x0, crop_y0, crop_x1, crop_y1 = 0, 0, w, h

    work_bgr = frame_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
    ch, cw = work_bgr.shape[:2]

    # Downscale for < 60 ms GrabCut execution on 2 vCPUs (area-capped when ROI-cropped)
    scale = min(1.0, float(max_work_dim) / float(max(cw, ch)))
    if sharp_onboarding:
        max_pixels = int(round(float(max_work_dim) * float(max_work_dim) * 0.62))
        scale = min(scale, math.sqrt(float(max_pixels) / float(max(cw * ch, 1))))
    sw = max(16, int(round(cw * scale)))
    sh = max(16, int(round(ch * scale)))
    sx = float(sw) / float(cw)
    sy = float(sh) / float(ch)

    small_bgr = cv2.resize(work_bgr, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1.0 else work_bgr
    gc_mask = np.full((sh, sw), cv2.GC_BGD, dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    if explicit_box is not None:
        bx, by, bw, bh = explicit_box
        lx = bx - float(crop_x0)
        ly = by - float(crop_y0)
        rx = int(np.clip(math.floor(lx * sx), 0, sw - 2))
        ry = int(np.clip(math.floor(ly * sy), 0, sh - 2))
        if sharp_onboarding:
            rx2 = int(np.clip(math.ceil((lx + bw) * sx), rx + 2, sw))
            ry2 = int(np.clip(math.ceil((ly + bh) * sy), ry + 2, sh))
            rw = rx2 - rx
            rh = ry2 - ry
        else:
            rw = int(np.clip(math.ceil(bw * sx), 2, sw - rx))
            rh = int(np.clip(math.ceil(bh * sy), 2, sh - ry))
        # Mark interior of box as probable foreground
        gc_mask[ry : ry + rh, rx : rx + rw] = cv2.GC_PR_FGD
        # Shrink center slightly as stronger foreground cue if no positive points given
        if not pos_pts and rw >= 8 and rh >= 8:
            mx = max(1, rw // 5)
            my = max(1, rh // 5)
            gc_mask[ry + my : ry + rh - my, rx + mx : rx + rw - mx] = cv2.GC_PR_FGD
        elif sharp_onboarding and pos_pts and rw >= 12 and rh >= 12:
            # Suppress clearly dissimilar adjacent distractors inside the box (e.g., yellow bottle inside blue pitcher box)
            small_lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            chroma_std = float(np.std(small_lab[ry : ry + rh, rx : rx + rw, 1:3]))
            is_color = chroma_std > 4.0
            seed_feats = []
            for px, py in pos_pts:
                sx_i = int(np.clip(round((px - crop_x0) * sx), 0, sw - 1))
                sy_i = int(np.clip(round((py - crop_y0) * sy), 0, sh - 1))
                patch = small_lab[max(0, sy_i - 2) : min(sh, sy_i + 3), max(0, sx_i - 2) : min(sw, sx_i + 3)]
                if patch.size > 0:
                    seed_feats.append(patch.reshape(-1, 3).mean(axis=0))
            if seed_feats:
                roi_lab = small_lab[ry : ry + rh, rx : rx + rw]
                min_dist = np.full((rh, rw), 1e6, dtype=np.float32)
                for sf in seed_feats:
                    if is_color:
                        # Weight chroma (a, b) higher than L so shading on the same object is tolerated
                        d = np.sqrt(
                            0.35 * (roi_lab[:, :, 0] - sf[0]) ** 2
                            + 1.6 * (roi_lab[:, :, 1] - sf[1]) ** 2
                            + 1.6 * (roi_lab[:, :, 2] - sf[2]) ** 2
                        )
                    else:
                        d = np.abs(roi_lab[:, :, 0] - sf[0])
                    min_dist = np.minimum(min_dist, d)
                thresh = 52.0 if is_color else 58.0
                dissimilar = min_dist > thresh
                # Never demote > 75% of the box (preserves multi-colored textured objects like printed boxes/cans)
                if float(np.mean(dissimilar)) < 0.75:
                    sub_gc = gc_mask[ry : ry + rh, rx : rx + rw]
                    sub_gc[dissimilar] = cv2.GC_PR_BGD
    else:
        rx, ry, rw, rh = _estimate_point_roi(
            small_bgr,
            [((x - crop_x0) * sx, (y - crop_y0) * sy) for x, y in pos_pts],
            [((x - crop_x0) * sx, (y - crop_y0) * sy) for x, y in neg_pts],
        )
        gc_mask[ry : ry + rh, rx : rx + rw] = cv2.GC_PR_FGD

    # Apply explicit positive/negative point strokes
    if sharp_onboarding and explicit_box is not None:
        pos_r = max(3, int(round(min(rw, rh) * 0.08)))
    else:
        pos_r = max(2, int(round(min(sw, sh) * 0.025)))
    neg_r = max(3, int(round(min(sw, sh) * 0.035)))
    for px, py in pos_pts:
        cv2.circle(
            gc_mask,
            (int(round((px - crop_x0) * sx)), int(round((py - crop_y0) * sy))),
            pos_r,
            int(cv2.GC_FGD),
            -1,
        )
    for nx, ny in neg_pts:
        cv2.circle(
            gc_mask,
            (int(round((nx - crop_x0) * sx)), int(round((ny - crop_y0) * sy))),
            neg_r,
            int(cv2.GC_BGD),
            -1,
        )

    has_fg_cue = bool(np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)))
    has_bg_cue = bool(np.any((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD)))

    effective_iters = max(grabcut_iters, 3) if sharp_onboarding else grabcut_iters
    small_mask = np.zeros((sh, sw), dtype=np.uint8)
    if has_fg_cue and has_bg_cue:
        try:
            cv2.setRNGSeed(17)
            cv2.grabCut(small_bgr, gc_mask, None, bgd_model, fgd_model, effective_iters, cv2.GC_INIT_WITH_MASK)
            small_mask = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
        except cv2.error:
            small_mask = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
    else:
        small_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

    # Fallback if GrabCut collapsed on low-contrast or uniform surface
    if int(np.count_nonzero(small_mask)) < 16:
        small_mask = np.zeros((sh, sw), dtype=np.uint8)
        if explicit_box is not None:
            bx, by, bw, bh = explicit_box
            lx = bx - float(crop_x0)
            ly = by - float(crop_y0)
            rx = int(np.clip(math.floor(lx * sx), 0, sw - 1))
            ry = int(np.clip(math.floor(ly * sy), 0, sh - 1))
            rx2 = int(np.clip(math.ceil((lx + bw) * sx), rx + 1, sw))
            ry2 = int(np.clip(math.ceil((ly + bh) * sy), ry + 1, sh))
            small_mask[ry:ry2, rx:rx2] = 255
        elif pos_pts:
            # Color-similarity fallback in LAB space around positive points
            lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            samples = [
                lab[
                    int(np.clip(round((py - crop_y0) * sy), 0, sh - 1)),
                    int(np.clip(round((px - crop_x0) * sx), 0, sw - 1)),
                ]
                for px, py in pos_pts
            ]
            mean_lab = np.mean(samples, axis=0)
            dist = np.linalg.norm(lab - mean_lab[None, None, :], axis=-1)
            small_mask = np.where(dist < 32.0, 255, 0).astype(np.uint8)
            if int(np.count_nonzero(small_mask)) < 16:
                for px, py in pos_pts:
                    cv2.circle(
                        small_mask,
                        (int(round((px - crop_x0) * sx)), int(round((py - crop_y0) * sy))),
                        max(8, min(sw, sh) // 6),
                        255,
                        -1,
                    )

    # Upsample crop back into full native resolution
    if scale < 1.0:
        crop_mask = cv2.resize(small_mask, (cw, ch), interpolation=cv2.INTER_LINEAR)
        crop_mask = np.where(crop_mask >= 127, 255, 0).astype(np.uint8)
    else:
        crop_mask = small_mask

    if crop_x0 == 0 and crop_y0 == 0 and crop_x1 == w and crop_y1 == h:
        mask_u8 = crop_mask
    else:
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        mask_u8[crop_y0:crop_y1, crop_x0:crop_x1] = crop_mask

    # Morphological cleanup (remove speckles, fill small interior holes)
    k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k_morph)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Enforce explicit bounding box hard constraint (never leak outside user box)
    if explicit_box is not None:
        bx, by, bw, bh = explicit_box
        x0 = int(np.clip(math.floor(bx), 0, w - 1))
        y0 = int(np.clip(math.floor(by), 0, h - 1))
        x1 = int(np.clip(math.ceil(bx + bw), x0 + 1, w))
        y1 = int(np.clip(math.ceil(by + bh), y0 + 1, h))
        box_gate = np.zeros((h, w), dtype=np.uint8)
        box_gate[y0:y1, x0:x1] = 255
        mask_u8 = cv2.bitwise_and(mask_u8, box_gate)
        if int(np.count_nonzero(mask_u8)) < 16:
            mask_u8 = box_gate

    # Enforce negative point exclusions
    neg_excl_r = max(4, int(round(min(w, h) * 0.02)))
    for nx, ny in neg_pts:
        cv2.circle(mask_u8, (int(round(nx)), int(round(ny))), neg_excl_r, 0, -1)

    # Track how many positive points GrabCut actually segmented BEFORE rescue circles
    pos_hits_pre = 0
    for px, py in pos_pts:
        ix = int(np.clip(round(px), 0, w - 1))
        iy = int(np.clip(round(py), 0, h - 1))
        if mask_u8[iy, ix] > 0:
            pos_hits_pre += 1
        else:
            cv2.circle(mask_u8, (ix, iy), max(4, int(round(min(w, h) * 0.015))), 255, -1)

    mask_u8 = _keep_selected_components(mask_u8, pos_pts, neg_pts)

    # Re-enforce negative point exclusion and explicit box after component filtering
    for nx, ny in neg_pts:
        cv2.circle(mask_u8, (int(round(nx)), int(round(ny))), max(2, neg_excl_r // 2), 0, -1)
    if explicit_box is not None:
        bx, by, bw, bh = explicit_box
        x0 = int(np.clip(math.floor(bx), 0, w - 1))
        y0 = int(np.clip(math.floor(by), 0, h - 1))
        x1 = int(np.clip(math.ceil(bx + bw), x0 + 1, w))
        y1 = int(np.clip(math.ceil(by + bh), y0 + 1, h))
        box_gate = np.zeros((h, w), dtype=np.uint8)
        box_gate[y0:y1, x0:x1] = 255
        mask_u8 = cv2.bitwise_and(mask_u8, box_gate)

    # Soft edge-feathered alpha matte (1-2 px transition strictly inside mask)
    dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    alpha = np.clip(dist_in / 2.0, 0.0, 1.0).astype(np.float32)

    fg_pixels = int(np.count_nonzero(mask_u8))
    coverage = float(fg_pixels) / float(w * h)
    ys, xs = np.nonzero(mask_u8)
    if len(xs) > 0:
        bx_out = int(xs.min())
        by_out = int(ys.min())
        bw_out = int(xs.max() - bx_out + 1)
        bh_out = int(ys.max() - by_out + 1)
        bbox_out = (bx_out, by_out, bw_out, bh_out)
    else:
        bbox_out = (0, 0, 0, 0)

    polygon = _extract_polygon(mask_u8)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    anchors = _extract_interior_anchors(gray, mask_u8)

    # Initialization consistency verification
    init_valid = True
    rejection_reason: str | None = None
    quality_score = 1.0
    if fg_pixels < 16:
        init_valid = False
        rejection_reason = "empty_mask"
        quality_score = 0.0
    elif coverage > 0.92:
        init_valid = False
        rejection_reason = "full_frame_spill"
        quality_score = 0.05
    else:
        fill_ratio = float(fg_pixels) / float(max(bbox_out[2] * bbox_out[3], 1))
        pos_hit_ratio = (float(pos_hits_pre) / float(len(pos_pts))) if pos_pts else 1.0
        # Boundary contrast between mask interior and immediate 8px exterior ring
        dil = cv2.dilate(mask_u8, np.ones((17, 17), np.uint8), iterations=1)
        ring = (dil > 0) & (mask_u8 == 0)
        contrast = 15.0
        if np.any(ring):
            fg_mean = frame_bgr[mask_u8 > 0].astype(np.float32).mean(axis=0)
            bg_mean = frame_bgr[ring].astype(np.float32).mean(axis=0)
            contrast = float(np.linalg.norm(fg_mean - bg_mean))
        c_score = float(np.clip(contrast / 25.0, 0.0, 1.0))
        f_score = float(np.clip(fill_ratio / 0.35, 0.0, 1.0))
        quality_score = float(np.clip(0.45 * pos_hit_ratio + 0.35 * c_score + 0.20 * f_score, 0.0, 1.0))
        if pos_pts and pos_hits_pre == 0 and fill_ratio < 0.15:
            init_valid = False
            rejection_reason = "positive_seed_mismatch"
        elif contrast < 2.5 and fill_ratio < 0.12:
            init_valid = False
            rejection_reason = "low_contrast_fragmented"

    saved_path: str | None = None
    if output_mask_path is not None:
        out_p = Path(os.path.abspath(os.fspath(output_mask_path)))
        posix = out_p.as_posix()
        if posix == "/tmp" or posix.startswith("/tmp/"):
            raise ContractValidationError(f"Refusing to write mask under /tmp: {out_p}")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = out_p.parent / f".pending-{os.getpid()}-{out_p.name}"
        ok, buf = cv2.imencode(".png", mask_u8)
        if not ok:
            raise RuntimeError(f"Failed to encode mask PNG for {out_p}")
        tmp_p.write_bytes(buf.tobytes())
        tmp_p.replace(out_p)
        saved_path = str(out_p)

    init_ms = (time.perf_counter() - t0) * 1000.0
    return SegmentationResult(
        frame_index=int(selection.frame_index),
        width=w,
        height=h,
        mask=mask_u8,
        alpha=alpha,
        foreground_pixels=fg_pixels,
        coverage_ratio=coverage,
        bbox=bbox_out,
        polygon=polygon,
        anchors=anchors,
        anchor_count=int(len(anchors)),
        init_ms=init_ms,
        mask_path=saved_path,
        quality_score=quality_score,
        init_valid=init_valid,
        rejection_reason=rejection_reason,
    )
