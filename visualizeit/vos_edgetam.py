"""Causal streaming neural Video Object Segmentation backed by Meta EdgeTAM (SAM 2 family)."""
from __future__ import annotations

from collections import OrderedDict
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np

from .contracts import ContractValidationError, SelectionSpec
from .segmentation import SegmentationResult, _extract_interior_anchors, _extract_polygon

REPO_ROOT = Path(__file__).resolve().parent.parent

_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_PREDICTOR_LOCK = threading.Lock()
_PREDICTOR_CACHE: dict[tuple[str, int], Any] = {}


def _resolve_edgetam_paths() -> tuple[Path, Path]:
    root = Path(os.environ.get("VISUALIZEIT_EDGETAM_DIR", str(REPO_ROOT / ".cache" / "models" / "EdgeTAM")))
    ckpt = Path(os.environ.get("VISUALIZEIT_EDGETAM_CKPT", str(root / "checkpoints" / "edgetam.pt")))
    return root, ckpt


def is_edgetam_available() -> bool:
    """Return True if PyTorch, EdgeTAM repo, and edgetam.pt weights are available locally."""
    root, ckpt = _resolve_edgetam_paths()
    if not (root / "sam2").is_dir() or not ckpt.is_file() or ckpt.stat().st_size < 1_000_000:
        return False
    try:
        import torch  # noqa: F401
        import hydra  # noqa: F401
        import timm  # noqa: F401
    except ImportError:
        return False
    return True


def _get_edgetam_predictor(device: str = "cpu", image_size: int = 512) -> Any:
    """Lazy-load and cache the singleton fused EdgeTAM SAM2VideoPredictor."""
    img_sz = 1024 if int(image_size) >= 768 else 512
    cache_key = (device, img_sz)
    with _PREDICTOR_LOCK:
        cached = _PREDICTOR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        root, ckpt = _resolve_edgetam_paths()
        if not (root / "sam2").is_dir() or not ckpt.is_file():
            raise RuntimeError(
                f"EdgeTAM model files not found at {root} / {ckpt}. "
                "Install EdgeTAM into .cache/models/EdgeTAM or set VISUALIZEIT_EDGETAM_DIR."
            )

        import torch

        if device == "cpu":
            n_threads = max(1, min(4, os.cpu_count() or 2))
            try:
                torch.set_num_threads(n_threads)
            except RuntimeError:
                pass

        root_str = str(root.resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        from hydra.core.global_hydra import GlobalHydra
        from hydra import initialize_config_dir

        cfg_dir = str((root / "sam2" / "configs").resolve())
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=cfg_dir, version_base="1.2")

        from sam2.build_sam import build_sam2_video_predictor

        overrides: list[str] = []
        if img_sz != 1024:
            q = img_sz // 16
            overrides = [
                f"++model.image_size={img_sz}",
                f"++model.memory_attention.layer.self_attention.feat_sizes=[{q},{q}]",
                f"++model.memory_attention.layer.cross_attention.q_sizes=[{q},{q}]",
            ]

        predictor = build_sam2_video_predictor(
            config_file="edgetam.yaml",
            ckpt_path=str(ckpt.resolve()),
            device=device,
            hydra_overrides_extra=overrides,
        )
        predictor.eval()
        # Algebraically fuse all RepViT multi-branch Conv2d + BatchNorm2d blocks (exact, ~1.5-2x speedup)
        try:
            from timm.utils.model import reparameterize_model

            predictor.image_encoder.trunk = reparameterize_model(predictor.image_encoder.trunk)
        except Exception:
            pass
        if device == "cpu":
            try:
                predictor.image_encoder = predictor.image_encoder.to(memory_format=torch.channels_last)
                orig_enc_fwd = predictor.image_encoder.forward

                def _cl_forward(sample: Any) -> Any:
                    if hasattr(sample, "ndim") and sample.ndim == 4:
                        sample = sample.to(memory_format=torch.channels_last)
                    return orig_enc_fwd(sample)

                predictor.image_encoder.forward = _cl_forward
            except Exception:
                pass
            try:
                predictor.memory_encoder = predictor.memory_encoder.to(memory_format=torch.channels_last)
            except Exception:
                pass

        _PREDICTOR_CACHE[cache_key] = predictor
        return predictor


def _fast_guided_filter(
    guide_gray_u8: np.ndarray,
    mask_f32: np.ndarray,
    *,
    radius: int = 3,
    eps: float = 0.008,
) -> np.ndarray:
    """Fast O(1) subsampled guided filter snapping soft masks to crisp image edges."""
    h, w = guide_gray_u8.shape[:2]
    I_full = guide_gray_u8.astype(np.float32) * (1.0 / 255.0)
    if h >= 48 and w >= 48:
        sh, sw = max(16, h // 2), max(16, w // 2)
        I = cv2.resize(I_full, (sw, sh), interpolation=cv2.INTER_AREA)
        p = cv2.resize(mask_f32.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
        r_s = max(1, radius // 2)
        k = (2 * r_s + 1, 2 * r_s + 1)
        mean_I = cv2.boxFilter(I, cv2.CV_32F, k)
        mean_p = cv2.boxFilter(p, cv2.CV_32F, k)
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, k)
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(I * I, cv2.CV_32F, k)
        var_I = np.maximum(mean_II - mean_I * mean_I, 0.0)
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = cv2.resize(cv2.boxFilter(a, cv2.CV_32F, k), (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(cv2.boxFilter(b, cv2.CV_32F, k), (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(mean_a * I_full + mean_b, 0.0, 1.0)
    p = mask_f32.astype(np.float32)
    k = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(I_full, cv2.CV_32F, k)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, k)
    mean_Ip = cv2.boxFilter(I_full * p, cv2.CV_32F, k)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I_full * I_full, cv2.CV_32F, k)
    var_I = np.maximum(mean_II - mean_I * mean_I, 0.0)
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, k)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, k)
    return np.clip(mean_a * I_full + mean_b, 0.0, 1.0)


def _clean_vos_mask(mask_u8: np.ndarray, *, max_pinhole_frac: float = 0.004) -> np.ndarray:
    """Remove tiny specks and fill strictly tiny interior pinholes without closing genuine handle gaps."""
    fg_px = int(np.count_nonzero(mask_u8))
    if fg_px < 16:
        return np.zeros_like(mask_u8)

    # Keep components >= 4% of total foreground area (preserves objects split in two by a crossing bar or hand!)
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 127).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask_u8)
    if n_cc > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = int(areas.max())
        min_keep = max(24, int(0.04 * max_area))
        for lbl in range(1, n_cc):
            if int(stats[lbl, cv2.CC_STAT_AREA]) >= min_keep:
                out[labels == lbl] = 255
    else:
        out = (mask_u8 > 127).astype(np.uint8) * 255

    # Fill ONLY tiny strictly enclosed interior pinholes (e.g., < 0.4% of fg area) so handle holes stay open!
    inv = cv2.bitwise_not(out)
    n_h, h_labels, h_stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if n_h > 2:
        border_lbls = set(np.unique(h_labels[0, :]).tolist())
        border_lbls.update(np.unique(h_labels[-1, :]).tolist())
        border_lbls.update(np.unique(h_labels[:, 0]).tolist())
        border_lbls.update(np.unique(h_labels[:, -1]).tolist())
        max_hole = max(16, int(max_pinhole_frac * fg_px))
        for lbl in range(1, n_h):
            if lbl not in border_lbls and int(h_stats[lbl, cv2.CC_STAT_AREA]) <= max_hole:
                out[h_labels == lbl] = 255

    return out


class EdgeTAMSession:
    """Causal streaming Video Object Segmentation using stabilized-crop EdgeTAM + soft-prob DIS flow."""

    def __init__(
        self,
        *,
        vos_stride: int = 3,
        device: str = "cpu",
        memory_window: int = 5,
        image_size: int = 512,
        adaptive_stride: bool = True,
        enable_roi_crop: bool = True,
    ) -> None:
        self.vos_stride = max(1, int(vos_stride))
        self.device = device
        self.memory_window = max(3, int(memory_window))
        self.image_size = 1024 if int(image_size) >= 768 else 512
        self.adaptive_stride = bool(adaptive_stride)
        self.enable_roi_crop = bool(enable_roi_crop)
        self._predictor = _get_edgetam_predictor(device, image_size=self.image_size)
        self._state: dict[str, Any] | None = None
        self._vos_step_idx: int = 0
        self._frames_since_vos: int = 0
        self._frame_w: int = 0
        self._frame_h: int = 0
        # Stabilized object-centered crop state (gives ~1024-equivalent object detail at 5x faster 512 speed!)
        self._use_crop: bool = False
        self._crop_size: int = 0
        self._crop_cx: float = 0.0
        self._crop_cy: float = 0.0
        self._last_mask: np.ndarray | None = None
        self._kf_prob_f32: np.ndarray | None = None
        self._prev_gray_u8: np.ndarray | None = None
        self._kf_pts: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self._live_pts: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self._kf_init_pts_count: int = 0
        self._kf_edge_score: float = 0.0
        self._initialized: bool = False

    def _frame_to_tensor(self, frame_bgr: np.ndarray) -> Any:
        import torch

        img_sz = int(self._predictor.image_size)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (img_sz, img_sz), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) * (1.0 / 255.0)
        arr = (arr - _IMG_MEAN[None, None, :]) / _IMG_STD[None, None, :]
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
        return torch.from_numpy(arr)

    def _extract_crop(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
        if not self._use_crop:
            return frame_bgr, 0, 0
        s = self._crop_size
        x0 = int(round(np.clip(self._crop_cx - 0.5 * s, 0.0, float(self._frame_w - s))))
        y0 = int(round(np.clip(self._crop_cy - 0.5 * s, 0.0, float(self._frame_h - s))))
        return frame_bgr[y0 : y0 + s, x0 : x0 + s], x0, y0

    def _update_crop_center(self, full_mask_u8: np.ndarray) -> None:
        if not self._use_crop:
            return
        ys, xs = np.nonzero(full_mask_u8 > 127)
        if len(xs) < 32:
            return
        half_s = 0.5 * float(self._crop_size)
        ncx = float(xs.mean())
        ncy = float(ys.mean())
        self._crop_cx = float(np.clip(0.65 * self._crop_cx + 0.35 * ncx, half_s, float(self._frame_w) - half_s))
        self._crop_cy = float(np.clip(0.65 * self._crop_cy + 0.35 * ncy, half_s, float(self._frame_h) - half_s))

    @staticmethod
    def _boundary_edge_score(gmag_roi: np.ndarray, prob_roi: np.ndarray) -> float:
        """Mean image gradient magnitude along the 2px silhouette contour ring of prob_roi."""
        fg = (prob_roi >= 0.5).astype(np.uint8) * 255
        if int(np.count_nonzero(fg)) < 32:
            return 0.0
        k3 = np.ones((3, 3), np.uint8)
        ring = cv2.dilate(fg, k3, iterations=1) > cv2.erode(fg, k3, iterations=1)
        if not np.any(ring):
            return 0.0
        return float(gmag_roi[ring].mean())

    def _record_kf_state(self, frame_bgr: np.ndarray, mask_u8: np.ndarray, prob_full: np.ndarray) -> None:
        """Cache keyframe probability map, live grayscale, interior KLT anchors, and baseline edge energy."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._kf_prob_f32 = prob_full
        self._prev_gray_u8 = gray
        self._kf_pts = np.zeros((0, 2), dtype=np.float32)
        self._live_pts = np.zeros((0, 2), dtype=np.float32)
        self._kf_init_pts_count = 0
        self._kf_edge_score = 0.0

        ys, xs = np.nonzero(mask_u8 > 127)
        if len(xs) < 48:
            return
        h, w = self._frame_h, self._frame_w
        pad = 8
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(w, int(xs.max()) + pad + 1)
        y1 = min(h, int(ys.max()) + pad + 1)
        sub_g = gray[y0:y1, x0:x1]
        sub_m = cv2.erode(mask_u8[y0:y1, x0:x1], np.ones((5, 5), np.uint8), iterations=1)
        if int(np.count_nonzero(sub_m)) < 32:
            sub_m = mask_u8[y0:y1, x0:x1]
        corners = cv2.goodFeaturesToTrack(
            sub_g,
            maxCorners=64,
            qualityLevel=0.015,
            minDistance=5.0,
            mask=sub_m,
            blockSize=5,
        )
        if corners is not None and len(corners) >= 6:
            pts = corners.reshape(-1, 2).astype(np.float32)
            pts[:, 0] += float(x0)
            pts[:, 1] += float(y0)
            self._kf_pts = pts.copy()
            self._live_pts = pts.copy()
            self._kf_init_pts_count = int(len(pts))
        gx = cv2.Sobel(sub_g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sub_g, cv2.CV_32F, 0, 1, ksize=3)
        self._kf_edge_score = self._boundary_edge_score(np.hypot(gx, gy), prob_full[y0:y1, x0:x1])

    def _build_initial_state(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        import torch

        h, w = frame_bgr.shape[:2]
        compute_device = self._predictor.device
        state: dict[str, Any] = {
            "images": {0: self._frame_to_tensor(frame_bgr)},
            "num_frames": 100000,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": int(h),
            "video_width": int(w),
            "device": compute_device,
            "storage_device": compute_device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {
                "cond_frame_outputs": set(),
                "non_cond_frame_outputs": set(),
            },
            "tracking_has_started": False,
            "frames_already_tracked": {},
        }
        with torch.inference_mode():
            self._predictor._get_image_feature(state, frame_idx=0, batch_size=1)
        return state

    def initialize(
        self,
        frame_bgr: np.ndarray,
        selection: SelectionSpec,
    ) -> SegmentationResult:
        """Prompt EdgeTAM on frame 0 with user bbox/points and prime temporal memory."""
        import torch

        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ContractValidationError("frame_bgr must be an HxWx3 uint8 BGR image")
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        self._frame_h, self._frame_w = h, w
        self._vos_step_idx = 0

        b0, g0, r0 = frame_bgr[::4, ::4, 0], frame_bgr[::4, ::4, 1], frame_bgr[::4, ::4, 2]
        chroma0 = float(
            np.mean(np.abs(r0.astype(np.int16) - g0.astype(np.int16)) + np.abs(g0.astype(np.int16) - b0.astype(np.int16)))
        )
        is_gray = chroma0 < 6.0

        # Use stabilized object-centered square crop into fused 512x512 whenever the target bbox is compact
        # (< 92% of max dim): zooms small objects (like 56px monochrome mugs/bottles) to ~2x higher effective
        # resolution than full-frame 1024 while running 5x faster! Only fall back to 1024 full-frame for huge
        # grayscale targets when ROI crop is disabled or cannot zoom in.
        self._use_crop = False
        if self.enable_roi_crop and self.image_size < 768 and selection.bbox is not None:
            bx_s, by_s, bw_s, bh_s = selection.bbox
            cand_s = int(np.clip(round(max(float(bw_s), float(bh_s)) * 2.6), 288, min(h, w)))
            if cand_s < int(0.92 * max(w, h)) and cand_s >= 128:
                self._use_crop = True
                self._crop_size = cand_s
                half_s = 0.5 * float(cand_s)
                self._crop_cx = float(np.clip(bx_s + 0.5 * bw_s, half_s, float(w) - half_s))
                self._crop_cy = float(np.clip(by_s + 0.5 * bh_s, half_s, float(h) - half_s))
        if is_gray and (not self._use_crop) and self.image_size < 768:
            self.image_size = 1024
            self.vos_stride = max(self.vos_stride * 2, 7)
            self._predictor = _get_edgetam_predictor(self.device, image_size=1024)

        work_frame, cx0, cy0 = self._extract_crop(frame_bgr)
        wh, ww = work_frame.shape[:2]
        state = self._build_initial_state(work_frame)
        self._state = state

        pts_list: list[list[float]] = []
        lbl_list: list[int] = []
        for x, y in selection.positive_points:
            pts_list.append([float(np.clip(x - cx0, 0.0, ww - 1.0)), float(np.clip(y - cy0, 0.0, wh - 1.0))])
            lbl_list.append(1)
        for x, y in selection.negative_points:
            pts_list.append([float(np.clip(x - cx0, 0.0, ww - 1.0)), float(np.clip(y - cy0, 0.0, wh - 1.0))])
            lbl_list.append(0)

        box_arr: np.ndarray | None = None
        if selection.bbox is not None:
            bx, by, bw, bh = selection.bbox
            x0 = float(np.clip(bx - cx0, 0.0, ww - 2.0))
            y0 = float(np.clip(by - cy0, 0.0, wh - 2.0))
            x1 = float(np.clip(bx + bw - cx0, x0 + 2.0, ww - 1.0))
            y1 = float(np.clip(by + bh - cy0, y0 + 2.0, wh - 1.0))
            box_arr = np.array([x0, y0, x1, y1], dtype=np.float32)

        pts_arr = np.array(pts_list, dtype=np.float32) if pts_list else None
        lbl_arr = np.array(lbl_list, dtype=np.int32) if lbl_list else None
        if pts_arr is None and box_arr is None:
            pts_arr = np.array([[ww * 0.5, wh * 0.5]], dtype=np.float32)
            lbl_arr = np.array([1], dtype=np.int32)

        # On grayscale imagery (no color cues to separate skin from object), a loose axis-aligned bbox
        # around a tilted handheld object can pull gripping hands into frame-0 memory. Check if points-only
        # already produces a clean compact object while box+points inflates > 2x onto attached hands.
        use_box = box_arr
        if (
            is_gray
            and box_arr is not None
            and pts_arr is not None
            and len(selection.positive_points) >= 1
            and len(selection.negative_points) >= 2
        ):
            with torch.inference_mode():
                _, _, v_pts = self._predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=pts_arr,
                    labels=lbl_arr,
                    box=None,
                    clear_old_points=True,
                    normalize_coords=True,
                )
                pts_fg = int(np.count_nonzero(v_pts[0, 0].detach().cpu().numpy() > 0.0))
                self._predictor.reset_state(state)
                _, _, v_box = self._predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=pts_arr,
                    labels=lbl_arr,
                    box=box_arr,
                    clear_old_points=True,
                    normalize_coords=True,
                )
                box_fg = int(np.count_nonzero(v_box[0, 0].detach().cpu().numpy() > 0.0))
                self._predictor.reset_state(state)
            bbox_area = max(1.0, float((box_arr[2] - box_arr[0]) * (box_arr[3] - box_arr[1])))
            if pts_fg >= 400 and (float(pts_fg) / bbox_area) >= 0.12 and float(box_fg) > 2.0 * float(pts_fg):
                use_box = None

        with torch.inference_mode():
            _, _, video_res_masks = self._predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=pts_arr,
                labels=lbl_arr,
                box=use_box,
                clear_old_points=True,
                normalize_coords=True,
            )
            self._predictor.propagate_in_video_preflight(state)

        logits_crop = video_res_masks[0, 0].detach().cpu().numpy()
        prob_crop = (1.0 / (1.0 + np.exp(-np.clip(logits_crop, -12.0, 12.0)))).astype(np.float32)
        if self._use_crop:
            prob_full = np.zeros((h, w), dtype=np.float32)
            prob_full[cy0 : cy0 + wh, cx0 : cx0 + ww] = prob_crop
        else:
            prob_full = prob_crop

        raw_u8 = (prob_full > 0.5).astype(np.uint8) * 255
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # On monochrome frames (zero color cues), if 512-crop's coarse stride-16 features merged a
        # contrasting gripping hand into the target (mask median luminance differs >= 40 from positive clicks),
        # cleanly re-initialize at fused native 1024x1024 full-frame so hand and target stay cleanly separated!
        if (
            is_gray
            and self._use_crop
            and len(selection.positive_points) >= 1
            and int(np.count_nonzero(raw_u8)) >= 200
        ):
            pos_ls = [float(gray[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))]) for x, y in selection.positive_points]
            pos_med_L = float(np.median(pos_ls))
            m_vals = gray[raw_u8 > 127]
            p10_m, p50_m, p90_m = np.percentile(m_vals, [10.0, 50.0, 90.0])
            bimodal_merged = (float(p90_m - p50_m) >= 45.0) and (float(p50_m - p10_m) >= 42.0)
            if bimodal_merged or abs(float(p50_m) - pos_med_L) >= 40.0:
                self._use_crop = False
                self.enable_roi_crop = False
                self.image_size = 1024
                self.vos_stride = max(self.vos_stride * 2, 7)
                self._predictor = _get_edgetam_predictor(self.device, image_size=1024)
                return self.initialize(frame_bgr, selection)

        mask_u8 = _clean_vos_mask(raw_u8)

        fg_px = int(np.count_nonzero(mask_u8))
        init_valid = fg_px >= 64
        rejection_reason: str | None = None if init_valid else "EdgeTAM returned empty/tiny initial mask"
        if init_valid:
            alpha = _fast_guided_filter(gray, prob_full, radius=3, eps=0.008)
            alpha = np.where(mask_u8 > 0, np.clip(alpha, 0.15, 1.0), np.clip(alpha, 0.0, 0.35)).astype(np.float32)
            ys, xs = np.nonzero(mask_u8)
            x0_i, x1_i = int(xs.min()), int(xs.max())
            y0_i, y1_i = int(ys.min()), int(ys.max())
            bbox = (x0_i, y0_i, max(1, x1_i - x0_i + 1), max(1, y1_i - y0_i + 1))
            polygon = _extract_polygon(mask_u8)
            anchors = _extract_interior_anchors(gray, mask_u8)
            quality_score = float(np.clip(0.85 + 0.15 * min(1.0, len(anchors) / 60.0), 0.0, 1.0))
        else:
            alpha = np.zeros((h, w), dtype=np.float32)
            bbox = (0, 0, 1, 1)
            polygon = []
            anchors = np.zeros((0, 2), dtype=np.float32)
            quality_score = 0.0

        self._last_mask = mask_u8
        self._record_kf_state(frame_bgr, mask_u8, prob_full)
        self._update_crop_center(mask_u8)
        self._initialized = True
        init_ms = (time.perf_counter() - t0) * 1000.0

        return SegmentationResult(
            frame_index=int(selection.frame_index),
            width=w,
            height=h,
            mask=mask_u8,
            alpha=alpha,
            foreground_pixels=fg_px,
            coverage_ratio=float(fg_px) / float(max(w * h, 1)),
            bbox=bbox,
            polygon=polygon,
            anchors=anchors,
            anchor_count=int(len(anchors)),
            init_ms=init_ms,
            quality_score=quality_score,
            init_valid=init_valid,
            rejection_reason=rejection_reason,
        )

    def _run_edgetam_step(self, frame_bgr: np.ndarray, frame_index: int = 0) -> np.ndarray:
        import torch

        assert self._state is not None
        self._vos_step_idx += 1
        s = self._vos_step_idx
        state = self._state
        work_frame, cx0, cy0 = self._extract_crop(frame_bgr)
        wh, ww = work_frame.shape[:2]
        state["images"][s] = self._frame_to_tensor(work_frame)
        if state["num_frames"] <= s + 1:
            state["num_frames"] = s + 1000

        with torch.inference_mode():
            current_out, pred_masks = self._predictor._run_single_frame_inference(
                inference_state=state,
                output_dict=state["output_dict"],
                frame_idx=s,
                batch_size=1,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            state["output_dict"]["non_cond_frame_outputs"][s] = current_out
            self._predictor._add_output_per_object(state, s, current_out, "non_cond_frame_outputs")
            state["frames_already_tracked"][s] = {"reverse": False}
            _, video_res_masks = self._predictor._get_orig_video_res_output(state, pred_masks)

        # Prune old non-conditioning memory & frame tensors to keep RAM O(1) and cross-attn lean
        cutoff = s - self.memory_window
        if cutoff > 0:
            for k in [k for k in list(state["images"].keys()) if k != 0 and k < cutoff]:
                state["images"].pop(k, None)
            nco = state["output_dict"]["non_cond_frame_outputs"]
            for k in [k for k in list(nco.keys()) if k < cutoff]:
                nco.pop(k, None)
            for obj_dict in state["output_dict_per_obj"].values():
                obj_nco = obj_dict["non_cond_frame_outputs"]
                for k in [k for k in list(obj_nco.keys()) if k < cutoff]:
                    obj_nco.pop(k, None)
            for k in [k for k in list(state["frames_already_tracked"].keys()) if k != 0 and k < cutoff]:
                state["frames_already_tracked"].pop(k, None)

        logits_crop = video_res_masks[0, 0].detach().cpu().numpy()
        prob_crop = (1.0 / (1.0 + np.exp(-np.clip(logits_crop, -12.0, 12.0)))).astype(np.float32)
        if self._use_crop:
            prob_full = np.zeros((self._frame_h, self._frame_w), dtype=np.float32)
            prob_full[cy0 : cy0 + wh, cx0 : cx0 + ww] = prob_crop
        else:
            prob_full = prob_crop

        raw_u8 = (prob_full > 0.5).astype(np.uint8) * 255
        if int(np.count_nonzero(raw_u8)) >= 64:
            # Snap soft EdgeTAM probabilities to full-resolution image luminance edges inside tight padded ROI
            ys_r, xs_r = np.nonzero(raw_u8)
            pad_e = 12
            ex0 = max(0, int(xs_r.min()) - pad_e)
            ey0 = max(0, int(ys_r.min()) - pad_e)
            ex1 = min(self._frame_w, int(xs_r.max()) + pad_e + 1)
            ey1 = min(self._frame_h, int(ys_r.max()) + pad_e + 1)
            gray_roi = cv2.cvtColor(frame_bgr[ey0:ey1, ex0:ex1], cv2.COLOR_BGR2GRAY)
            snapped = _fast_guided_filter(gray_roi, prob_full[ey0:ey1, ex0:ex1], radius=2, eps=0.004)
            raw_u8 = np.zeros_like(raw_u8)
            raw_u8[ey0:ey1, ex0:ex1] = (snapped >= 0.48).astype(np.uint8) * 255
        mask_u8 = _clean_vos_mask(raw_u8)
        self._last_mask = mask_u8
        self._record_kf_state(frame_bgr, mask_u8, prob_full)
        self._update_crop_center(mask_u8)
        self._frames_since_vos = 0
        return mask_u8

    def _propagate_mask_live_warp(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        """Fast (~1.2ms) live sparse-KLT 4-DoF similarity warp verified by silhouette boundary gradient.

        Unlike blind velocity extrapolation (which overshoots on handheld direction reversals like
        ycb_s52 f=51 and misses in-plane tilts like hope_obj02 f=85) and unlike dense optical flow
        (which drags walls off spinning cylinders), this tracks exact live translation + rotation +
        scale via sparse FB-KLT while verifying that the warped silhouette contour actually improves
        boundary edge energy over identity (rejecting in-place spinning-texture drift automatically).
        """
        kf_prob = self._kf_prob_f32
        prev_mask = self._last_mask
        prev_gray = self._prev_gray_u8
        if kf_prob is None or prev_mask is None or prev_gray is None or int(np.count_nonzero(prev_mask)) < 32:
            return np.zeros((self._frame_h, self._frame_w), dtype=np.uint8), True

        h, w = self._frame_h, self._frame_w
        cur_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1) Track sparse keyframe interior corners from prev_gray -> cur_gray via FB-gated KLT
        M_sim: np.ndarray | None = None
        motion_px = 0.0
        rot_deg = 0.0
        klt_lost = False
        if len(self._live_pts) >= 6:
            p0 = self._live_pts.reshape(-1, 1, 2).astype(np.float32)
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                cur_gray,
                p0,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
            )
            p0_b, st2, _ = cv2.calcOpticalFlowPyrLK(
                cur_gray,
                prev_gray,
                p1,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
            )
            if p1 is not None and p0_b is not None and st1 is not None and st2 is not None:
                fb = np.linalg.norm(p0.reshape(-1, 2) - p0_b.reshape(-1, 2), axis=1)
                good = (
                    (st1.ravel() == 1)
                    & (st2.ravel() == 1)
                    & (fb < 1.5)
                    & (p1[:, 0, 0] >= 0.0)
                    & (p1[:, 0, 0] < float(w))
                    & (p1[:, 0, 1] >= 0.0)
                    & (p1[:, 0, 1] < float(h))
                )
                self._kf_pts = self._kf_pts[good]
                self._live_pts = p1.reshape(-1, 2)[good].astype(np.float32)
            else:
                self._kf_pts = np.zeros((0, 2), dtype=np.float32)
                self._live_pts = np.zeros((0, 2), dtype=np.float32)

        if self._kf_init_pts_count >= 16 and len(self._live_pts) < 6:
            klt_lost = True

        if len(self._live_pts) >= 6:
            M_cand, inl = cv2.estimateAffinePartial2D(
                self._kf_pts,
                self._live_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.5,
            )
            if M_cand is not None and inl is not None and int(np.count_nonzero(inl)) >= 5:
                a, b = float(M_cand[0, 0]), float(M_cand[0, 1])
                scale_est = math.hypot(a, b)
                rot_deg = abs(math.degrees(math.atan2(float(M_cand[1, 0]), a)))
                disp = self._live_pts[inl.ravel().astype(bool)] - self._kf_pts[inl.ravel().astype(bool)]
                motion_px = float(np.median(np.linalg.norm(disp, axis=1))) if len(disp) else 0.0
                if 0.88 <= scale_est <= 1.14 and rot_deg <= 16.0 and motion_px <= 32.0:
                    M_sim = M_cand.astype(np.float32)

        # 2) Build padded ROI around keyframe/previous mask
        ys_k, xs_k = np.nonzero(kf_prob >= 0.5)
        if len(xs_k) < 32:
            ys_k, xs_k = np.nonzero(prev_mask)
        pad_nat = max(20, int(math.ceil(motion_px) + 16))
        x0 = max(0, int(xs_k.min()) - pad_nat)
        y0 = max(0, int(ys_k.min()) - pad_nat)
        x1 = min(w, int(xs_k.max()) + pad_nat + 1)
        y1 = min(h, int(ys_k.max()) + pad_nat + 1)
        roi_w, roi_h = x1 - x0, y1 - y0

        prob_roi = np.ascontiguousarray(kf_prob[y0:y1, x0:x1], dtype=np.float32)
        cur_roi_gray = cur_gray[y0:y1, x0:x1]
        gx = cv2.Sobel(cur_roi_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(cur_roi_gray, cv2.CV_32F, 0, 1, ksize=3)
        gmag_roi = np.hypot(gx, gy)

        # 3) Silhouette-edge verification: accept KLT similarity warp ONLY when it preserves or
        # improves contour gradient alignment over identity (automatically rejects spin-texture drag!)
        chosen_prob = prob_roi
        score_id = self._boundary_edge_score(gmag_roi, prob_roi)
        best_edge_score = score_id
        if M_sim is not None and (motion_px >= 0.45 or rot_deg >= 0.4):
            M_roi = M_sim.copy()
            M_roi[0, 2] = float(M_sim[0, 0] * x0 + M_sim[0, 1] * y0 + M_sim[0, 2] - x0)
            M_roi[1, 2] = float(M_sim[1, 0] * x0 + M_sim[1, 1] * y0 + M_sim[1, 2] - y0)
            warped_prob = cv2.warpAffine(
                prob_roi,
                M_roi,
                (roi_w, roi_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            score_warp = self._boundary_edge_score(gmag_roi, warped_prob)
            if score_warp >= score_id * 0.96:
                chosen_prob = warped_prob
                best_edge_score = score_warp

        soft_roi = _fast_guided_filter(cur_roi_gray, chosen_prob, radius=2, eps=0.004)
        snapped_roi = (soft_roi >= 0.48).astype(np.uint8) * 255
        # Remove 1-2px hair ticks along intermediate edges without shrinking smooth curves
        snapped_roi = cv2.morphologyEx(
            snapped_roi,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        out_mask = np.zeros((h, w), dtype=np.uint8)
        out_mask[y0:y1, x0:x1] = snapped_roi
        out_mask = _clean_vos_mask(out_mask)
        self._last_mask = out_mask
        self._prev_gray_u8 = cur_gray
        self._update_crop_center(out_mask)
        self._frames_since_vos += 1

        edge_dropped = (self._kf_edge_score > 18.0) and (best_edge_score < 0.72 * self._kf_edge_score)
        need_early_kf = (
            (motion_px >= 9.0)
            or (rot_deg >= 6.5)
            or klt_lost
            or edge_dropped
        )
        return out_mask, need_early_kf

    def step(self, frame_bgr: np.ndarray, frame_index: int) -> np.ndarray:
        """Causal per-frame segmentation step returning HxW uint8 mask in {0, 255}."""
        if not self._initialized or self._state is None:
            raise RuntimeError("EdgeTAMSession.initialize() must be called before step()")
        if frame_index == 0 and self._last_mask is not None:
            return self._last_mask
        if (self._last_mask is None) or (int(np.count_nonzero(self._last_mask)) < 64):
            return self._run_edgetam_step(frame_bgr, frame_index)
        if not self.adaptive_stride:
            if frame_index % self.vos_stride == 0:
                return self._run_edgetam_step(frame_bgr, frame_index)
            mask_u8, _ = self._propagate_mask_live_warp(frame_bgr)
            return mask_u8
        if self._frames_since_vos >= max(1, self.vos_stride - 1):
            return self._run_edgetam_step(frame_bgr, frame_index)
        mask_u8, need_early_kf = self._propagate_mask_live_warp(frame_bgr)
        if need_early_kf:
            return self._run_edgetam_step(frame_bgr, frame_index)
        return mask_u8
