"""Causal streaming neural Video Object Segmentation backed by Meta EdgeTAM (SAM 2 family)."""
from __future__ import annotations

from collections import OrderedDict
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
_PREDICTOR_CACHE: dict[str, Any] = {}


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


def _get_edgetam_predictor(device: str = "cpu") -> Any:
    """Lazy-load and cache the singleton EdgeTAM SAM2VideoPredictor."""
    with _PREDICTOR_LOCK:
        cached = _PREDICTOR_CACHE.get(device)
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

        predictor = build_sam2_video_predictor(
            config_file="edgetam.yaml",
            ckpt_path=str(ckpt.resolve()),
            device=device,
        )
        predictor.eval()
        _PREDICTOR_CACHE[device] = predictor
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
    """Causal streaming Video Object Segmentation session using EdgeTAM + optical-flow interpolation."""

    def __init__(
        self,
        *,
        vos_stride: int = 3,
        device: str = "cpu",
        memory_window: int = 12,
    ) -> None:
        self.vos_stride = max(1, int(vos_stride))
        self.device = device
        self.memory_window = max(6, int(memory_window))
        self._predictor = _get_edgetam_predictor(device)
        self._state: dict[str, Any] | None = None
        self._vos_step_idx: int = 0
        self._frame_w: int = 0
        self._frame_h: int = 0
        self._last_mask: np.ndarray | None = None
        self._prev_gray_flow: np.ndarray | None = None
        self._flow_scale: float = 1.0
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

    def _to_flow_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        max_d = max(h, w)
        if max_d > 360:
            self._flow_scale = 360.0 / float(max_d)
            sw = max(32, int(round(w * self._flow_scale)))
            sh = max(24, int(round(h * self._flow_scale)))
            return cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        self._flow_scale = 1.0
        return gray

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

        state = self._build_initial_state(frame_bgr)
        self._state = state

        pts_list: list[list[float]] = []
        lbl_list: list[int] = []
        for x, y in selection.positive_points:
            pts_list.append([float(np.clip(x, 0.0, w - 1.0)), float(np.clip(y, 0.0, h - 1.0))])
            lbl_list.append(1)
        for x, y in selection.negative_points:
            pts_list.append([float(np.clip(x, 0.0, w - 1.0)), float(np.clip(y, 0.0, h - 1.0))])
            lbl_list.append(0)

        box_arr: np.ndarray | None = None
        if selection.bbox is not None:
            bx, by, bw, bh = selection.bbox
            x0 = float(np.clip(bx, 0.0, w - 2.0))
            y0 = float(np.clip(by, 0.0, h - 2.0))
            x1 = float(np.clip(bx + bw, x0 + 2.0, w - 1.0))
            y1 = float(np.clip(by + bh, y0 + 2.0, h - 1.0))
            box_arr = np.array([x0, y0, x1, y1], dtype=np.float32)

        pts_arr = np.array(pts_list, dtype=np.float32) if pts_list else None
        lbl_arr = np.array(lbl_list, dtype=np.int32) if lbl_list else None
        if pts_arr is None and box_arr is None:
            pts_arr = np.array([[w * 0.5, h * 0.5]], dtype=np.float32)
            lbl_arr = np.array([1], dtype=np.int32)

        # On grayscale imagery (no color cues to separate skin from object), a loose axis-aligned bbox
        # around a tilted handheld object can pull gripping hands into frame-0 memory. Check if points-only
        # already produces a clean compact object while box+points inflates > 2x onto attached hands.
        use_box = box_arr
        if box_arr is not None and pts_arr is not None and len(selection.positive_points) >= 1 and len(selection.negative_points) >= 2:
            b0, g0, r0 = frame_bgr[::4, ::4, 0], frame_bgr[::4, ::4, 1], frame_bgr[::4, ::4, 2]
            chroma0 = float(np.mean(np.abs(r0.astype(np.int16) - g0.astype(np.int16)) + np.abs(g0.astype(np.int16) - b0.astype(np.int16))))
            if chroma0 < 6.0:
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

        logits = video_res_masks[0, 0].detach().cpu().numpy()
        raw_u8 = (logits > 0.0).astype(np.uint8) * 255
        mask_u8 = _clean_vos_mask(raw_u8)

        fg_px = int(np.count_nonzero(mask_u8))
        init_valid = fg_px >= 64
        rejection_reason: str | None = None if init_valid else "EdgeTAM returned empty/tiny initial mask"

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if init_valid:
            prob_f32 = (1.0 / (1.0 + np.exp(-np.clip(logits, -12.0, 12.0)))).astype(np.float32)
            alpha = _fast_guided_filter(gray, prob_f32, radius=3, eps=0.008)
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
        self._prev_gray_flow = self._to_flow_gray(frame_bgr)
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

    def _run_edgetam_step(self, frame_bgr: np.ndarray) -> np.ndarray:
        import torch

        assert self._state is not None
        self._vos_step_idx += 1
        s = self._vos_step_idx
        state = self._state
        state["images"][s] = self._frame_to_tensor(frame_bgr)
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

        # Prune old non-conditioning memory & frame tensors to keep RAM O(1)
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

        logits = video_res_masks[0, 0].detach().cpu().numpy()
        raw_u8 = (logits > 0.0).astype(np.uint8) * 255
        mask_u8 = _clean_vos_mask(raw_u8)
        self._last_mask = mask_u8
        self._prev_gray_flow = self._to_flow_gray(frame_bgr)
        return mask_u8

    def _warp_mask_optical_flow(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Fast ROI-cropped backward Farneback optical flow + guided-filter edge snap for intermediate frames."""
        cur_gray_flow = self._to_flow_gray(frame_bgr)
        prev_mask = self._last_mask
        prev_gray_flow = self._prev_gray_flow
        if prev_mask is None or prev_gray_flow is None or int(np.count_nonzero(prev_mask)) < 32:
            self._prev_gray_flow = cur_gray_flow
            return np.zeros((self._frame_h, self._frame_w), dtype=np.uint8)

        h, w = self._frame_h, self._frame_w
        fh, fw = cur_gray_flow.shape[:2]
        ys, xs = np.nonzero(prev_mask)
        pad_nat = max(24, int(0.08 * max(h, w)))
        x0 = max(0, int(xs.min()) - pad_nat)
        y0 = max(0, int(ys.min()) - pad_nat)
        x1 = min(w, int(xs.max()) + pad_nat + 1)
        y1 = min(h, int(ys.max()) + pad_nat + 1)

        sx = float(fw) / float(w)
        sy = float(fh) / float(h)
        fx0 = max(0, int(np.floor(x0 * sx)))
        fy0 = max(0, int(np.floor(y0 * sy)))
        fx1 = min(fw, max(fx0 + 16, int(np.ceil(x1 * sx))))
        fy1 = min(fh, max(fy0 + 16, int(np.ceil(y1 * sy))))

        # Backward optical flow (cur -> prev) yields hole-free remap coordinates!
        flow_roi = cv2.calcOpticalFlowFarneback(
            cur_gray_flow[fy0:fy1, fx0:fx1],
            prev_gray_flow[fy0:fy1, fx0:fx1],
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=13,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )

        roi_w, roi_h = x1 - x0, y1 - y0
        flow_up = cv2.resize(flow_roi, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
        flow_up[:, :, 0] *= (1.0 / max(sx, 1e-4))
        flow_up[:, :, 1] *= (1.0 / max(sy, 1e-4))

        grid_x, grid_y = np.meshgrid(
            np.arange(roi_w, dtype=np.float32),
            np.arange(roi_h, dtype=np.float32),
        )
        map_x = grid_x + flow_up[:, :, 0]
        map_y = grid_y + flow_up[:, :, 1]

        prev_roi = prev_mask[y0:y1, x0:x1]
        warped_roi = cv2.remap(
            prev_roi,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Snap warped boundary crisply to current frame edges via Fast Guided Filter
        roi_gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        soft_roi = _fast_guided_filter(roi_gray, warped_roi.astype(np.float32) * (1.0 / 255.0), radius=3, eps=0.008)
        snapped_roi = (soft_roi >= 0.48).astype(np.uint8) * 255

        out_mask = np.zeros((h, w), dtype=np.uint8)
        out_mask[y0:y1, x0:x1] = snapped_roi
        out_mask = _clean_vos_mask(out_mask)
        self._last_mask = out_mask
        self._prev_gray_flow = cur_gray_flow
        return out_mask

    def step(self, frame_bgr: np.ndarray, frame_index: int) -> np.ndarray:
        """Causal per-frame segmentation step returning HxW uint8 mask in {0, 255}."""
        if not self._initialized or self._state is None:
            raise RuntimeError("EdgeTAMSession.initialize() must be called before step()")
        if frame_index == 0 and self._last_mask is not None:
            return self._last_mask
        if (frame_index % self.vos_stride == 0) or (self._last_mask is None) or (int(np.count_nonzero(self._last_mask)) < 64):
            return self._run_edgetam_step(frame_bgr)
        return self._warp_mask_optical_flow(frame_bgr)
