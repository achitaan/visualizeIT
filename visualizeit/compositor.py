"""Multi-mode surface compositor (planar, cylinder, mesh_grid, mesh_6dof) via native C++17 AVX-512 rasterizer."""
from __future__ import annotations

import math
import time
from typing import Literal

import cv2
import numpy as np

from .contracts import ContractValidationError, PlacementSpec, TrackFrameResult
from .native_raster import RasterizeResult, is_native_raster_available, rasterize_mesh_rgbd
from .tracker import MeshGeometry


def _prepare_placed_texture(
    artwork_rgba: np.ndarray,
    placement: PlacementSpec,
    surface_mask_crop: np.ndarray | None = None,
) -> np.ndarray:
    """Apply placement (scale, rotation_deg, offset_x, offset_y, tile_repeat) and surface mask alpha once."""
    if artwork_rgba.ndim != 3 or artwork_rgba.shape[2] != 4 or artwork_rgba.dtype != np.uint8:
        raise ContractValidationError("artwork_rgba must be an HxWx4 uint8 RGBA array")

    tex_h, tex_w = artwork_rgba.shape[:2]
    # Build inverse affine transform mapping output surface pixel (x, y) -> source artwork pixel (sx, sy)
    # Output center shifted by (offset_x * tex_w, offset_y * tex_h)
    cx = tex_w * 0.5
    cy = tex_h * 0.5
    ox = placement.offset_x * tex_w
    oy = placement.offset_y * tex_h

    # cv2.warpAffine with WARP_INVERSE_MAP=False takes forward src -> dst matrix:
    # dst = R(rot) * scale * (src - center) + center + offset
    rad = math.radians(placement.rotation_deg)
    cos_a = math.cos(rad) * placement.scale
    sin_a = math.sin(rad) * placement.scale
    tx = (cx + ox) - (cos_a * cx - sin_a * cy)
    ty = (cy + oy) - (sin_a * cx + cos_a * cy)
    M = np.array([[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], dtype=np.float32)

    border_mode = cv2.BORDER_WRAP if placement.tile_repeat else cv2.BORDER_CONSTANT
    placed = cv2.warpAffine(
        artwork_rgba,
        M,
        (tex_w, tex_h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=(0, 0, 0, 0),
    )

    if surface_mask_crop is not None and surface_mask_crop.size > 0:
        mask_resized = cv2.resize(surface_mask_crop, (tex_w, tex_h), interpolation=cv2.INTER_LINEAR)
        # Preserve interior mask feathering strictly inside segmented object
        mask_f = mask_resized.astype(np.float32) * (1.0 / 255.0)
        alpha_new = np.clip(placed[:, :, 3].astype(np.float32) * mask_f, 0.0, 255.0).astype(np.uint8)
        placed = placed.copy()
        placed[:, :, 3] = alpha_new

    return np.ascontiguousarray(placed, dtype=np.uint8)


def _skin_prob_bgr(bgr: np.ndarray) -> np.ndarray:
    """Fast YCrCb + RGB skin likelihood in [0, 1] (returns zeros on monochrome footage)."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    y = ycrcb[:, :, 0]
    if float(np.std(cr)) < 2.0 and float(np.std(cb)) < 2.0:
        return np.zeros(bgr.shape[:2], dtype=np.float32)
    d2 = ((cr - 152.0) / 17.0) ** 2 + ((cb - 104.0) / 17.0) ** 2
    p = np.exp(-0.5 * d2)
    p *= np.clip((y - 35.0) / 30.0, 0.0, 1.0)
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    valid_rgb = (r > g * 0.95) & (r > b * 1.05) & ((r - np.minimum(g, b)) > 12.0)
    return np.where(valid_rgb, p, 0.0).astype(np.float32)


def _guided_filter_alpha(
    guide_gray_u8: np.ndarray,
    mask_f32: np.ndarray,
    *,
    radius: int = 4,
    eps: float = 0.012,
) -> np.ndarray:
    """Fast O(1) subsampled guided filter (He & Sun 2015) snapping soft masks to full-res image edges."""
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


def _compute_oriented_cylinder_frame(
    ref_mask: np.ndarray | None,
    ref_bbox: tuple[int, int, int, int],
) -> tuple[tuple[float, float], float, float, float] | None:
    """Return (center_xy, width, height, angle_rad) if ref_mask is clearly elongated and tilted (>= 8 deg)."""
    if ref_mask is None or int(np.count_nonzero(ref_mask)) < 120:
        return None
    ys, xs = np.nonzero(ref_mask > 127)
    if len(xs) < 120:
        return None
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    (cx, cy), (rw, rh), angle_deg = cv2.minAreaRect(pts)
    if rw < 6.0 or rh < 6.0:
        return None
    # Orient so major axis (height) points roughly upward-downward (closest to vertical)
    if rw > rh:
        eff_w, eff_h = float(rh), float(rw)
        axis_deg = float(angle_deg) + 90.0
    else:
        eff_w, eff_h = float(rw), float(rh)
        axis_deg = float(angle_deg)
    # Normalize axis_deg into [-90, 90] (0 = vertical upright cylinder)
    while axis_deg > 90.0:
        axis_deg -= 180.0
    while axis_deg < -90.0:
        axis_deg += 180.0
    aspect = eff_h / max(eff_w, 1.0)
    if aspect < 1.45 or abs(axis_deg) < 8.0 or abs(axis_deg) > 78.0:
        return None
    return (float(cx), float(cy)), eff_w, eff_h, math.radians(axis_deg)


def _fill_enclosed_holes_bool(mask_b: np.ndarray) -> np.ndarray:
    """Fill strictly enclosed interior holes in a 2D boolean mask without expanding exterior silhouette."""
    u8 = np.where(mask_b, np.uint8(255), np.uint8(0))
    inv = cv2.bitwise_not(u8)
    n_cc, labels, _, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if n_cc <= 2:
        return mask_b
    border_lbls = set(np.unique(labels[0, :]).tolist())
    border_lbls.update(np.unique(labels[-1, :]).tolist())
    border_lbls.update(np.unique(labels[:, 0]).tolist())
    border_lbls.update(np.unique(labels[:, -1]).tolist())
    out = mask_b.copy()
    for lbl in range(1, n_cc):
        if lbl not in border_lbls:
            out[labels == lbl] = True
    return out


class SurfaceCompositor:
    """Composites 2D RGBA artwork onto tracked surfaces using native C++17 AVX-512 or OpenCV backend."""

    def __init__(
        self,
        artwork_rgba: np.ndarray,
        placement: PlacementSpec,
        *,
        ref_bbox: tuple[int, int, int, int],
        ref_mask: np.ndarray | None = None,
        ref_frame_bgr: np.ndarray | None = None,
        mesh: MeshGeometry | None = None,
        intrinsics: tuple[float, float, float, float] | None = None,
        init_pose: tuple[np.ndarray, np.ndarray] | None = None,
        raster_backend: Literal["auto", "native_cpp", "opencv"] = "auto",
        local_occlusion_mask: bool = True,
        occlusion_mode: Literal["full", "ref_only", "ref_plus_boundary", "ref_plus_flow"] = "full",
    ) -> None:
        self.placement = placement
        self.ref_bbox = tuple(int(v) for v in ref_bbox)
        self.mesh = mesh
        self.intrinsics = intrinsics
        self.local_occlusion_mask = bool(local_occlusion_mask)
        self.occlusion_mode = occlusion_mode
        self.last_occluder_mask_img: np.ndarray | None = None

        if raster_backend == "auto":
            self.backend: Literal["native_cpp", "opencv"] = (
                "native_cpp" if is_native_raster_available(auto_build=False) else "opencv"
            )
        elif raster_backend in ("native_cpp", "opencv"):
            self.backend = raster_backend
        else:
            raise ContractValidationError(f"Invalid raster_backend: {raster_backend!r}")

        # Optional oriented cylinder frame for tilted elongated objects (e.g. handheld bottles)
        self._cyl_oriented: tuple[tuple[float, float], float, float, float] | None = None
        if placement.wrap_mode == "cylinder" and ref_mask is not None:
            self._cyl_oriented = _compute_oriented_cylinder_frame(ref_mask, self.ref_bbox)

        # Crop/align surface mask to ref_bbox (or oriented cylinder box) for 2D alpha clipping
        mask_crop: np.ndarray | None = None
        mask_crop_for_tex: np.ndarray | None = None
        if ref_mask is not None and placement.wrap_mode != "mesh_6dof":
            if placement.wrap_mode == "cylinder" and self._cyl_oriented is not None:
                (ocx, ocy), ow, oh, oang = self._cyl_oriented
                ca, sa = math.cos(oang), math.sin(oang)
                hw, hh = 0.5 * ow, 0.5 * oh
                src_quad = np.array([
                    [ocx - hw * ca + hh * sa, ocy - hw * sa - hh * ca],
                    [ocx + hw * ca + hh * sa, ocy + hw * sa - hh * ca],
                    [ocx + hw * ca - hh * sa, ocy + hw * sa + hh * ca],
                    [ocx - hw * ca - hh * sa, ocy - hw * sa + hh * ca],
                ], dtype=np.float32)
                mw, mh = max(16, int(round(ow))), max(16, int(round(oh)))
                dst_quad = np.array([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]], dtype=np.float32)
                M_o = cv2.getPerspectiveTransform(src_quad, dst_quad)
                crop = cv2.warpPerspective(ref_mask, M_o, (mw, mh), flags=cv2.INTER_LINEAR, borderValue=0)
                if float(np.count_nonzero(crop > 127)) / float(crop.size) >= 0.20:
                    mask_crop = crop
            else:
                bx, by, bw, bh = self.ref_bbox
                h, w = ref_mask.shape[:2]
                x0, y0 = max(0, bx), max(0, by)
                x1, y1 = min(w, bx + bw), min(h, by + bh)
                if x1 > x0 and y1 > y0:
                    crop = ref_mask[y0:y1, x0:x1]
                    if float(np.count_nonzero(crop)) / float(crop.size) >= 0.20:
                        mask_crop = crop

            mask_crop_for_tex = mask_crop
            # In cylinder mode, vertices map u -> x_frac = 0.5 + 0.5*sin(theta)/sin(max_arc).
            # Remap mask_crop through the exact same sin(theta) mapping so tapered bottle shoulders
            # NEVER bulge outward into empty air!
            if placement.wrap_mode == "cylinder" and mask_crop is not None:
                mh_c, mw_c = mask_crop.shape[:2]
                half_arc = math.radians(
                    min(82.0, max(35.0, 74.0 * math.sqrt(max(0.15, placement.cylinder_radius_frac) / 0.5)))
                )
                sin_max = math.sin(half_arc)
                u_g = np.linspace(0.0, 1.0, mw_c, dtype=np.float32)
                thetas = (u_g * 2.0 - 1.0) * half_arc
                x_fracs = 0.5 + 0.5 * (np.sin(thetas) / sin_max)
                mx_c = np.clip(x_fracs * float(mw_c - 1), 0.0, float(mw_c - 1)).astype(np.float32)
                my_c = np.arange(mh_c, dtype=np.float32)
                cmap_x, cmap_y = np.meshgrid(mx_c, my_c)
                mask_crop_for_tex = cv2.remap(mask_crop, cmap_x, cmap_y, cv2.INTER_LINEAR, borderValue=0)

        # For mesh_6dof we keep full texture (placement applied via texture transform; mesh geometry defines silhouette)
        self._artwork_rgba = artwork_rgba
        self._placed_rgba = _prepare_placed_texture(
            artwork_rgba,
            placement,
            surface_mask_crop=mask_crop_for_tex if placement.wrap_mode != "mesh_6dof" else None,
        )
        self._placed_rgba_unclipped: np.ndarray | None = None

        # Prebuild canonical local 2D grids for planar, cylinder, and mesh_grid
        self._planar_uvs, self._planar_tris = self._build_grid_topology(2, 2)
        self._cyl_ref_xy, self._cyl_uvs, self._cyl_tris = self._build_cylinder_topology(
            cols=24, rows=4, radius_frac=placement.cylinder_radius_frac
        )
        self._grid_uvs, self._grid_tris = self._build_grid_topology(4, 4)

        # Optional per-pixel illumination-invariant + boundary/hand + flow local occlusion mask state
        self._occ_enabled = False
        self._occ_size = 96
        self._occ_pad = 14
        self._occ_full = self._occ_size + 2 * self._occ_pad  # 124x124 with exterior ring
        self._occ_H_ref_to_canon: np.ndarray | None = None
        self._occ_ref_ab: np.ndarray | None = None
        self._occ_ref_ab_min: np.ndarray | None = None
        self._occ_ref_ab_max: np.ndarray | None = None
        self._occ_ref_L: np.ndarray | None = None
        self._occ_ref_L_min: np.ndarray | None = None
        self._occ_ref_L_max: np.ndarray | None = None
        self._occ_ref_mean: np.ndarray | None = None
        self._occ_ref_std: np.ndarray | None = None
        self._occ_ref_skin: np.ndarray | None = None
        self._occ_is_color: bool = True
        self._occ_target_mask: np.ndarray | None = None
        self._occ_prev_L_half: np.ndarray | None = None
        self._occ_prev_vis: np.ndarray | None = None
        self._occ_cyl_map_x: np.ndarray | None = None
        self._occ_cyl_map_y: np.ndarray | None = None
        self._work_rgba: np.ndarray | None = None
        self._base_alpha_u8: np.ndarray | None = None
        self._last_2d_occ_u8: np.ndarray | None = None
        self._last_2d_ref_match_u8: np.ndarray | None = None
        self._last_2d_H_crop2img: np.ndarray | None = None

        # 6-DoF 3D-projected UV reference + image-space exterior hand/occluder state
        self._occ_6dof_enabled = False
        self._occ_6dof_is_color = True
        self._occ_6dof_ref_skin_frac = 0.0
        self._occ_6dof_ref_L_lo = 0.0
        self._occ_6dof_ref_L_hi = 255.0
        self._occ_6dof_ref_L_med = 128.0
        self._occ_6dof_target_lab_lut: np.ndarray | None = None
        self._occ_6dof_uv_pts3d: np.ndarray | None = None
        self._occ_6dof_uv_norms: np.ndarray | None = None
        self._occ_6dof_ref_uv_L: np.ndarray | None = None
        self._occ_6dof_ref_uv_L_min: np.ndarray | None = None
        self._occ_6dof_ref_uv_L_max: np.ndarray | None = None
        self._occ_6dof_ref_uv_ab: np.ndarray | None = None
        self._occ_6dof_ref_uv_ab_min: np.ndarray | None = None
        self._occ_6dof_ref_uv_ab_max: np.ndarray | None = None
        self._occ_6dof_ref_uv_valid: np.ndarray | None = None
        self._occ_6dof_prev_vis_uv: np.ndarray | None = None

        if self.local_occlusion_mask and ref_frame_bgr is not None:
            if placement.wrap_mode != "mesh_6dof":
                self._init_local_occlusion_reference(ref_frame_bgr, ref_mask=mask_crop)
            else:
                self._init_6dof_occlusion_reference(ref_frame_bgr, ref_mask=ref_mask, init_pose=init_pose)

        self._mono_bright_fg_L_lo: float | None = None
        if ref_frame_bgr is not None and ref_mask is not None and int(np.count_nonzero(ref_mask)) >= 200:
            b0, g0, r0 = ref_frame_bgr[::4, ::4, 0], ref_frame_bgr[::4, ::4, 1], ref_frame_bgr[::4, ::4, 2]
            chroma0 = float(np.mean(np.abs(r0.astype(np.int16) - g0.astype(np.int16)) + np.abs(g0.astype(np.int16) - b0.astype(np.int16))))
            if chroma0 < 6.0:
                gray0 = cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2GRAY)
                fg_vals = gray0[ref_mask > 127]
                if len(fg_vals) >= 200:
                    p15, p50 = np.percentile(fg_vals, [15.0, 50.0])
                    if float(p50) >= 158.0:
                        self._mono_bright_fg_L_lo = max(96.0, min(float(p15) + 2.0, float(p50) - 44.0))

        # Reusable scratch buffers (allocated lazily on first frame)
        self._buf_shape: tuple[int, int] | None = None
        self._depth_buf: np.ndarray | None = None
        self._mask_buf: np.ndarray | None = None

    def _init_local_occlusion_reference(
        self,
        ref_frame_bgr: np.ndarray,
        *,
        ref_mask: np.ndarray | None = None,
    ) -> None:
        """Precompute 124x124 canonical (96x96 target + 14px exterior ring) reference stats for multi-cue occlusion."""
        bx, by, bw, bh = self.ref_bbox
        if bw < 8 or bh < 8:
            return
        sz = self._occ_size
        pad = self._occ_pad
        full = self._occ_full
        if self.placement.wrap_mode == "cylinder" and self._cyl_oriented is not None:
            (ocx, ocy), ow, oh, oang = self._cyl_oriented
            ca, sa = math.cos(oang), math.sin(oang)
            hw, hh = 0.5 * ow, 0.5 * oh
            ref_quad = np.array([
                [ocx - hw * ca + hh * sa, ocy - hw * sa - hh * ca],
                [ocx + hw * ca + hh * sa, ocy + hw * sa - hh * ca],
                [ocx + hw * ca - hh * sa, ocy + hw * sa + hh * ca],
                [ocx - hw * ca - hh * sa, ocy - hw * sa + hh * ca],
            ], dtype=np.float32)
        else:
            ref_quad = np.array(
                [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]],
                dtype=np.float32,
            )
        p = float(pad)
        s = float(sz)
        canon_quad = np.array([[p, p], [p + s - 1.0, p], [p + s - 1.0, p + s - 1.0], [p, p + s - 1.0]], dtype=np.float32)
        H_r2c = cv2.getPerspectiveTransform(ref_quad, canon_quad).astype(np.float64)
        ref_full = cv2.warpPerspective(
            ref_frame_bgr, H_r2c, (full, full), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        ref_lab = cv2.cvtColor(ref_full, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_L = ref_lab[:, :, 0]
        ref_ab = ref_lab[:, :, 1:3]

        tmask = np.zeros((full, full), dtype=bool)
        if ref_mask is not None and ref_mask.size > 0:
            tmask[pad:-pad, pad:-pad] = cv2.resize(ref_mask, (sz, sz), interpolation=cv2.INTER_LINEAR) > 127
        if int(np.count_nonzero(tmask)) < 64:
            tmask[pad:-pad, pad:-pad] = True

        k3 = np.ones((3, 3), np.uint8)
        ref_L_min = cv2.erode(ref_L, k3)
        ref_L_max = cv2.dilate(ref_L, k3)
        ref_ab_min = cv2.erode(ref_ab, k3)
        ref_ab_max = cv2.dilate(ref_ab, k3)

        # Dense 7x7 sliding-window luminance stats for fast per-pixel NCC
        win = (7, 7)
        ref_mean = cv2.boxFilter(ref_L, cv2.CV_32F, win)
        ref_sq = cv2.boxFilter(ref_L * ref_L, cv2.CV_32F, win)
        ref_std = np.sqrt(np.maximum(ref_sq - ref_mean * ref_mean, 0.0))

        self._occ_H_ref_to_canon = H_r2c
        self._occ_ref_ab = np.ascontiguousarray(ref_ab, dtype=np.float32)
        self._occ_ref_ab_min = np.ascontiguousarray(ref_ab_min, dtype=np.float32)
        self._occ_ref_ab_max = np.ascontiguousarray(ref_ab_max, dtype=np.float32)
        self._occ_ref_L = np.ascontiguousarray(ref_L, dtype=np.float32)
        self._occ_ref_L_min = np.ascontiguousarray(ref_L_min, dtype=np.float32)
        self._occ_ref_L_max = np.ascontiguousarray(ref_L_max, dtype=np.float32)
        self._occ_ref_mean = np.ascontiguousarray(ref_mean, dtype=np.float32)
        self._occ_ref_std = np.ascontiguousarray(ref_std, dtype=np.float32)
        self._occ_ref_skin = _skin_prob_bgr(ref_full)
        ref_f32 = ref_full.astype(np.float32)
        self._occ_ref_chrom = np.ascontiguousarray(
            ref_f32 / (ref_f32.sum(axis=2, keepdims=True) + 12.0), dtype=np.float32
        )
        self._occ_is_color = float(np.std(ref_ab[pad:-pad, pad:-pad])) > 2.5
        self._occ_target_mask = tmask
        self._occ_prev_L_half = None
        self._occ_prev_vis = None
        self._work_rgba = np.ascontiguousarray(self._placed_rgba.copy(), dtype=np.uint8)
        self._base_alpha_u8 = np.ascontiguousarray(self._placed_rgba[:, :, 3], dtype=np.uint8)

        if self.placement.wrap_mode == "cylinder":
            half_arc = math.radians(
                min(82.0, max(35.0, 74.0 * math.sqrt(max(0.15, self.placement.cylinder_radius_frac) / 0.5)))
            )
            sin_max = math.sin(half_arc)
            u_grid = np.linspace(0.0, 1.0, sz, dtype=np.float32)
            thetas = (u_grid * 2.0 - 1.0) * half_arc
            x_fracs = 0.5 + 0.5 * (np.sin(thetas) / sin_max)
            xs = np.clip(x_fracs * float(sz - 1), 0.0, float(sz - 1)).astype(np.float32)
            ys = np.arange(sz, dtype=np.float32)
            self._occ_cyl_map_x, self._occ_cyl_map_y = np.meshgrid(xs, ys)

        self._occ_enabled = True

    def _init_6dof_occlusion_reference(
        self,
        ref_frame_bgr: np.ndarray,
        *,
        ref_mask: np.ndarray | None = None,
        init_pose: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Initialize 3D-projected UV reference texture, target color LUT, and shape-mask clipping for mesh_6dof."""
        h, w = ref_frame_bgr.shape[:2]
        ref_lab_u8 = cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2LAB)
        ref_lab = ref_lab_u8.astype(np.float32)
        self._occ_6dof_is_color = float(np.std(ref_lab[:, :, 1:3])) > 2.5
        ref_skin = _skin_prob_bgr(ref_frame_bgr)
        if ref_mask is not None and int(np.count_nonzero(ref_mask)) >= 64:
            m_bool = ref_mask > 127
            self._occ_6dof_ref_skin_frac = float(np.mean(ref_skin[m_bool] > 0.45))
            ref_gray = cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2GRAY)
            l_vals = ref_gray[m_bool].astype(np.float32)
            self._occ_6dof_ref_L_lo = float(np.percentile(l_vals, 5))
            self._occ_6dof_ref_L_hi = float(np.percentile(l_vals, 95))
            self._occ_6dof_ref_L_med = float(np.median(l_vals))
            # Build quantized 3D (L, a, b) color histogram of the target object (bins: L/24, a/10, b/10)
            # requiring >= 1.5% occupancy per bin so anti-aliased edge pixels never whitelist background/skin.
            # Secondary object label colors (e.g. orange soup band, red/white paper labels) are protected
            # via exact per-pixel 3D UV reference matching inside _compute_6dof_exterior_hand_mask!
            counts = np.zeros((12, 27, 27), dtype=np.int32)
            pts_lab = ref_lab_u8[m_bool]
            l_idx = np.clip(pts_lab[:, 0] // 24, 0, 11)
            a_idx = np.clip(pts_lab[:, 1] // 10, 0, 26)
            b_idx = np.clip(pts_lab[:, 2] // 10, 0, 26)
            np.add.at(counts, (l_idx, a_idx, b_idx), 1)
            min_bin_cnt = max(20, int(0.015 * len(pts_lab)))
            lut = counts >= min_bin_cnt
            # Dilate +/-1 bin in luminance L only to absorb shading changes without spilling into skin chroma
            lut_dil = lut.copy()
            lut_dil[1:, :, :] |= lut[:-1, :, :]
            lut_dil[:-1, :, :] |= lut[1:, :, :]
            self._occ_6dof_target_lab_lut = lut_dil
        else:
            self._occ_6dof_ref_skin_frac = 0.0
            self._occ_6dof_target_lab_lut = None

        self._work_rgba = np.ascontiguousarray(self._placed_rgba.copy(), dtype=np.uint8)
        self._base_alpha_u8 = np.ascontiguousarray(self._placed_rgba[:, :, 3], dtype=np.uint8)
        self._occ_6dof_prev_vis_uv = None
        self._occ_6dof_cyl_h: float = 0.14
        self._occ_6dof_y_min: float = -0.07
        self._last_6dof_gain: float = 1.0

        # If mesh is a parametric cylinder and we have init_pose, bake visible frame-0 surface into 96x64 UV reference
        if (
            self.mesh is not None
            and self.mesh.source_kind == "parametric"
            and self.mesh.name == "cylinder"
            and init_pose is not None
        ):
            rvec0, tvec0 = init_pose
            R0, _ = cv2.Rodrigues(np.asarray(rvec0, dtype=np.float64).reshape(3))
            t0 = np.asarray(tvec0, dtype=np.float64).reshape(3)
            if self.intrinsics is not None:
                fx, fy, cx, cy = self.intrinsics
            else:
                f = float(max(w, h)) * 0.9
                fx, fy, cx, cy = f, f, w * 0.5, h * 0.5
            verts = self.mesh.vertices.astype(np.float64)
            v_radii = np.hypot(verts[:, 0], verts[:, 2])
            cyl_h = float(np.ptp(verts[:, 1]))
            y_min_v = float(verts[:, 1].min())
            self._occ_6dof_cyl_h = max(cyl_h, 1e-4)
            self._occ_6dof_y_min = y_min_v
            uw, vh = 96, 64
            us = np.linspace(0.0, 1.0, uw, dtype=np.float64)
            vs = np.linspace(0.0, 1.0, vh, dtype=np.float64)
            # Extract per-row axisymmetric radius r(v) from mesh vertices (handles both straight & tapered lathe bottles)
            ys_unique = np.unique(np.round(verts[:, 1], 5))
            if len(ys_unique) >= 2 and cyl_h > 1e-6:
                r_by_y = np.array([
                    float(np.median(v_radii[np.abs(verts[:, 1] - yv) < 1e-4])) for yv in ys_unique
                ], dtype=np.float64)
                v_nodes = (ys_unique - y_min_v) / cyl_h
                r_vs = np.interp(vs, v_nodes, r_by_y)
            else:
                r_vs = np.full(vh, float(np.max(v_radii)), dtype=np.float64)
            uu, vv = np.meshgrid(us, vs)
            rr = r_vs[:, None] * np.ones((1, uw), dtype=np.float64)
            theta = (uu - 0.5) * 2.0 * math.pi
            nx = np.sin(theta)
            nz = -np.cos(theta)
            px = rr * nx
            py = (vv - 0.5) * cyl_h
            pz = rr * nz
            pts3d = np.stack([px, py, pz], axis=-1).reshape(-1, 3)
            norms = np.stack([nx, np.zeros_like(nx), nz], axis=-1).reshape(-1, 3)

            pc = pts3d @ R0.T + t0.reshape(1, 3)
            nc = norms @ R0.T
            view = -pc / np.maximum(np.linalg.norm(pc, axis=1, keepdims=True), 1e-6)
            front_2d = ((np.sum(nc * view, axis=1) > 0.25) & (pc[:, 2] > 0.04)).reshape(vh, uw)
            mx = (pc[:, 0] / np.maximum(pc[:, 2], 1e-4) * fx + cx).reshape(vh, uw).astype(np.float32)
            my = (pc[:, 1] / np.maximum(pc[:, 2], 1e-4) * fy + cy).reshape(vh, uw).astype(np.float32)
            baked_bgr = cv2.remap(ref_frame_bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            baked_lab = cv2.cvtColor(baked_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

            in_bounds = (mx >= 1) & (mx < w - 1) & (my >= 1) & (my < h - 1) & front_2d
            if ref_mask is not None:
                m_uv = cv2.remap(ref_mask, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > 127
                # Fill any enclosed interior holes before clipping shape so labels/barcodes never become permanent holes
                m_uv_filled = _fill_enclosed_holes_bool(m_uv)
                in_bounds &= m_uv_filled
                front_observed = (mx >= 1) & (mx < w - 1) & (my >= 1) & (my < h - 1) & front_2d
                n_front_obs = int(np.count_nonzero(front_observed))
                out_frac = float(np.count_nonzero(front_observed & (~m_uv_filled))) / float(max(n_front_obs, 1))
                if out_frac >= 0.06:
                    uv_shape = np.ones((vh, uw), dtype=bool)
                    uv_shape[front_observed] = m_uv_filled[front_observed]
                    uv_shape_u8 = cv2.dilate(uv_shape.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=1)
                    tex_h, tex_w = self._base_alpha_u8.shape
                    shape_tex = cv2.resize(uv_shape_u8, (tex_w, tex_h), interpolation=cv2.INTER_LINEAR)
                    self._base_alpha_u8 = cv2.multiply(self._base_alpha_u8, shape_tex, scale=1.0 / 255.0)
                    self._work_rgba[:, :, 3] = self._base_alpha_u8

            # 5x5 Birchfield-Tomasi spatial tolerance bounds in UV space to prevent rotational edge-shift erosion
            # on high-contrast printed text/logos during 360-deg spin.
            k5 = np.ones((5, 5), np.uint8)
            ref_L = np.ascontiguousarray(baked_lab[:, :, 0], dtype=np.float32)
            ref_ab = np.ascontiguousarray(baked_lab[:, :, 1:3], dtype=np.float32)
            self._occ_6dof_uv_pts3d = pts3d
            self._occ_6dof_uv_norms = norms
            self._occ_6dof_ref_uv_L = ref_L
            self._occ_6dof_ref_uv_L_min = np.ascontiguousarray(cv2.erode(ref_L, k5), dtype=np.float32)
            self._occ_6dof_ref_uv_L_max = np.ascontiguousarray(cv2.dilate(ref_L, k5), dtype=np.float32)
            self._occ_6dof_ref_uv_ab = ref_ab
            self._occ_6dof_ref_uv_ab_min = np.ascontiguousarray(cv2.erode(ref_ab, k5), dtype=np.float32)
            self._occ_6dof_ref_uv_ab_max = np.ascontiguousarray(cv2.dilate(ref_ab, k5), dtype=np.float32)
            self._occ_6dof_ref_uv_valid = in_bounds

        self._occ_6dof_enabled = True

    def _apply_local_occlusion_mask(
        self, frame_bgr: np.ndarray, H: np.ndarray, *, export_ref_match: bool = False
    ) -> np.ndarray:
        """Compute multi-cue 124x124 visibility mask (ref+dense NCC, exterior-connected hand/fg seg, half-scale flow)."""
        if (
            not self._occ_enabled
            or self._occ_H_ref_to_canon is None
            or self._occ_ref_ab is None
            or self._occ_ref_ab_min is None
            or self._occ_ref_ab_max is None
            or self._occ_ref_L is None
            or self._occ_ref_L_min is None
            or self._occ_ref_L_max is None
            or self._occ_ref_mean is None
            or self._occ_ref_std is None
            or self._occ_ref_skin is None
            or self._occ_target_mask is None
            or self._work_rgba is None
            or self._base_alpha_u8 is None
        ):
            return self._placed_rgba

        try:
            H_inv = np.linalg.inv(H.astype(np.float64))
        except np.linalg.LinAlgError:
            return self._placed_rgba

        full = self._occ_full
        pad = self._occ_pad
        M_cur_to_canon = self._occ_H_ref_to_canon @ H_inv
        cur_full = cv2.warpPerspective(
            frame_bgr, M_cur_to_canon, (full, full), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        cur_lab = cv2.cvtColor(cur_full, cv2.COLOR_BGR2LAB).astype(np.float32)
        cur_L = cur_lab[:, :, 0]
        cur_ab = cur_lab[:, :, 1:3]

        # Robust global luminance gain normalization using least-changed 50% of valid target pixels
        tmask = self._occ_target_mask
        rv = self._occ_ref_L[tmask]
        cv_ = cur_L[tmask]
        raw_d = np.abs(cv_ - rv)
        inl = raw_d <= max(float(np.median(raw_d)), 10.0)
        if int(np.count_nonzero(inl)) >= 48:
            gain = float(np.clip(float(rv[inl].mean()) / max(float(cv_[inl].mean()), 10.0), 0.65, 1.55))
        else:
            gain = 1.0
        cur_L_norm = np.clip(cur_L * gain, 0.0, 255.0)

        # Signal 1: Edge-envelope tolerant L & chroma residuals + dense 7x7 sliding-window NCC
        dL_env = np.maximum(0.0, np.maximum(self._occ_ref_L_min - cur_L_norm, cur_L_norm - self._occ_ref_L_max))
        dL_abs = np.abs(cur_L_norm - self._occ_ref_L)
        is_edge_L = (self._occ_ref_L_max - self._occ_ref_L_min) > 16.0
        dL_s = cv2.GaussianBlur(np.where(is_edge_L, dL_env, dL_abs), (3, 3), 0.0)

        dab_env_v = np.maximum(0.0, np.maximum(self._occ_ref_ab_min - cur_ab, cur_ab - self._occ_ref_ab_max))
        dab_env = np.sqrt(dab_env_v[:, :, 0] ** 2 + dab_env_v[:, :, 1] ** 2)
        dab_abs_v = cur_ab - self._occ_ref_ab
        dab_abs = np.sqrt(dab_abs_v[:, :, 0] ** 2 + dab_abs_v[:, :, 1] ** 2)
        is_edge_ab = np.max(self._occ_ref_ab_max - self._occ_ref_ab_min, axis=2) > 10.0
        dab_s = cv2.GaussianBlur(np.where(is_edge_ab, dab_env, dab_abs), (3, 3), 0.0)

        win = (7, 7)
        cur_mean = cv2.boxFilter(cur_L_norm, cv2.CV_32F, win)
        cur_sq = cv2.boxFilter(cur_L_norm * cur_L_norm, cv2.CV_32F, win)
        cur_std = np.sqrt(np.maximum(cur_sq - cur_mean * cur_mean, 0.0))
        cov = cv2.boxFilter(self._occ_ref_L * cur_L_norm, cv2.CV_32F, win) - self._occ_ref_mean * cur_mean
        denom = self._occ_ref_std * cur_std
        has_tex = self._occ_ref_std > 11.0
        ncc = np.where(has_tex & (denom > 1e-3), cov / np.maximum(denom, 1e-3), 1.0)
        tex_destroyed = (self._occ_ref_std > 13.0) & (cur_std < 0.30 * self._occ_ref_std)
        ncc = np.where(tex_destroyed, 0.0, ncc)

        if self.placement.wrap_mode == "cylinder":
            occ = (
                (dab_s > 26.0)
                | ((dab_s > 14.0) & (ncc < 0.42))
                | (has_tex & (dL_s > 22.0) & (ncc < 0.30))
                | (tex_destroyed & (dL_s > 14.0))
                | ((~has_tex) & (dL_s > 44.0))
            )
        else:
            occ = (
                (dab_s > 22.0)
                | ((dab_s > 11.0) & (ncc < 0.48))
                | (has_tex & (dL_s > 11.0) & (ncc < 0.44))
                | (has_tex & (ncc < 0.16) & (dL_s > 6.0))
                | (tex_destroyed & (dL_s > 6.0))
                | ((~has_tex) & (dL_s > 22.0))
            )

        # Signal 2: Explicit skin-excess hand detector + exterior-bridging smooth-region propagation
        if self.occlusion_mode in ("ref_plus_boundary", "full"):
            if self._occ_is_color:
                cur_skin = _skin_prob_bgr(cur_full)
                skin_excess = np.maximum(0.0, cur_skin - self._occ_ref_skin)
                skin_thr = 0.24 if export_ref_match else 0.32
                occ |= (skin_excess > skin_thr) & ((dab_s > 6.5) | (dL_s > 7.0) | (ncc < 0.58))

            gx = cv2.Sobel(cur_L_norm, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(cur_L_norm, cv2.CV_32F, 0, 1, ksize=3)
            cur_grad = np.hypot(gx, gy)
            if self._occ_is_color:
                ga = np.hypot(
                    cv2.Sobel(cur_ab[:, :, 0], cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(cur_ab[:, :, 0], cv2.CV_32F, 0, 1, ksize=3),
                )
                gb = np.hypot(
                    cv2.Sobel(cur_ab[:, :, 1], cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(cur_ab[:, :, 1], cv2.CV_32F, 0, 1, ksize=3),
                )
                cur_grad = np.maximum(cur_grad, 2.5 * np.hypot(ga, gb))

            seed_mask = (occ & tmask & (cur_grad < 22.0)).astype(np.uint8)
            t_u8 = tmask.astype(np.uint8) * 255
            border_in = (t_u8 > 0) & (cv2.erode(t_u8, np.ones((5, 5), np.uint8)) == 0)
            dl_bridge = 14.0 if self._occ_is_color else 10.0
            border_bridge = border_in & (cur_grad < 18.0) & ((dL_s > dl_bridge) | (dab_s > 13.0) | tex_destroyed | (ncc < 0.30))
            seed_mask[border_bridge] = 1

            # Require low gradient AND non-matching appearance (or exterior) so beige hands and bars
            # flood cleanly while NEVER leaking into unoccluded matching card pixels (e.g. split card f=104)!
            if self.placement.wrap_mode == "cylinder":
                smooth_changed = (~tmask) | (dL_s > 14.0) | (dab_s > 14.0) | (ncc < 0.45) | tex_destroyed
            else:
                smooth_changed = (~tmask) | (dL_s > 9.5) | (dab_s > 8.5) | (has_tex & (ncc < 0.48)) | tex_destroyed
            smooth = ((cur_grad < 22.0) & smooth_changed).astype(np.uint8)
            skin_seed = np.zeros_like(seed_mask)
            if self._occ_is_color:
                skin_seed = ((skin_excess > 0.25) & tmask).astype(np.uint8)
            n_cc, cc_labels, _, _ = cv2.connectedComponentsWithStats(smooth, connectivity=4)
            if n_cc > 1:
                flat_lbl = cc_labels.ravel()
                ext_cnt = np.bincount(flat_lbl, weights=(~tmask).ravel().astype(np.float32), minlength=n_cc)
                in_cnt = np.bincount(flat_lbl, weights=tmask.ravel().astype(np.float32), minlength=n_cc)
                seed_cnt = np.bincount(flat_lbl, weights=seed_mask.ravel().astype(np.float32), minlength=n_cc)
                skin_cnt = np.bincount(flat_lbl, weights=skin_seed.ravel().astype(np.float32), minlength=n_cc)
                is_fg_cc = np.zeros(n_cc, dtype=bool)
                min_seed_frac = 0.14 if self._occ_is_color else 0.08
                min_seed_cnt = 12 if self._occ_is_color else 8
                for lbl in range(1, n_cc):
                    if ext_cnt[lbl] >= 10 and in_cnt[lbl] >= 18:
                        seed_frac = seed_cnt[lbl] / max(in_cnt[lbl], 1.0)
                        if (seed_frac >= min_seed_frac and seed_cnt[lbl] >= min_seed_cnt) or (
                            seed_cnt[lbl] >= 8 and (seed_frac >= 0.06) and (skin_cnt[lbl] >= 4 or ext_cnt[lbl] >= 40)
                        ):
                            is_fg_cc[lbl] = True
                if np.any(is_fg_cc):
                    fg_grown = is_fg_cc[cc_labels]
                    # Never let exterior flood-fill eat pixels that strongly match textured reference surface!
                    strong_ref_tex = has_tex & (ncc > 0.72) & (dL_s < 9.0) & (dab_s < 8.5) & (~tex_destroyed)
                    fg_grown = fg_grown & (~strong_ref_tex)
                    fg_grown_u8 = cv2.dilate(
                        fg_grown.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    )
                    occ |= (fg_grown_u8 > 0) & (~strong_ref_tex)

        # Signal 3: Fast half-scale (62x62) canonical optical flow residual
        if self.occlusion_mode in ("ref_plus_flow", "full"):
            half_sz = full // 2
            cur_L_half = cv2.resize(cur_L_norm.astype(np.uint8), (half_sz, half_sz), interpolation=cv2.INTER_AREA)
            if self._occ_prev_L_half is not None and self._occ_prev_L_half.shape == cur_L_half.shape:
                flow = cv2.calcOpticalFlowFarneback(
                    self._occ_prev_L_half, cur_L_half, None, 0.5, 2, 9, 1, 5, 1.1, 0
                )
                mag_half = np.hypot(flow[:, :, 0], flow[:, :, 1])
                mag = cv2.resize(mag_half, (full, full), interpolation=cv2.INTER_LINEAR)
                if self.placement.wrap_mode == "cylinder":
                    occ |= (mag > 1.8) & ((dL_s > 20.0) | (dab_s > 18.0))
                else:
                    occ |= (mag > 1.0) & ((dL_s > 8.0) | (dab_s > 8.0) | (ncc < 0.58))
            self._occ_prev_L_half = cur_L_half

        # Luminance-only shadow protection: identical local texture (ncc >= 0.78) + invariant RGB chromaticity is a shadow, never a foreign occluder
        if self._occ_is_color and getattr(self, "_occ_ref_chrom", None) is not None:
            cur_f32 = cur_full.astype(np.float32)
            cur_chrom = cur_f32 / (cur_f32.sum(axis=2, keepdims=True) + 12.0)
            chrom_diff = np.max(np.abs(cur_chrom - self._occ_ref_chrom), axis=2)
            shadow_ncc = np.where((self._occ_ref_std > 4.0) & (denom > 1e-3), cov / np.maximum(denom, 1e-3), 0.0)
            is_pure_shadow = (self._occ_ref_std > 4.0) & (shadow_ncc >= 0.78) & (chrom_diff < 0.06) & (cur_L < self._occ_ref_L + 10.0)
            occ &= ~is_pure_shadow

        # Suppress tiny isolated interior noise specks that do not touch the silhouette boundary
        occ_in = occ & tmask
        t_er = cv2.erode(tmask.astype(np.uint8) * 255, np.ones((5, 5), np.uint8)) > 0
        touch_bnd = tmask & (~t_er)
        n_oc, oc_lbl, oc_stats, _ = cv2.connectedComponentsWithStats(occ_in.astype(np.uint8), connectivity=8)
        if n_oc > 1:
            strong_ev = (dab_s > 26.0) | tex_destroyed | ((~self._occ_is_color) & (dL_s > 26.0))
            flat_o = oc_lbl.ravel()
            bnd_hits = np.bincount(flat_o, weights=touch_bnd.ravel().astype(np.float32), minlength=n_oc)
            strong_hits = np.bincount(flat_o, weights=strong_ev.ravel().astype(np.float32), minlength=n_oc)
            keep_o = np.zeros(n_oc, dtype=bool)
            for lbl in range(1, n_oc):
                area = int(oc_stats[lbl, cv2.CC_STAT_AREA])
                if bnd_hits[lbl] >= 2 and area >= 12:
                    keep_o[lbl] = True
                elif area >= 36 and strong_hits[lbl] >= 18:
                    keep_o[lbl] = True
            occ_clean = keep_o[oc_lbl]
        else:
            occ_clean = occ_in

        occ_crop = occ_clean[pad:-pad, pad:-pad]
        occ_u8 = np.where(occ_crop, np.uint8(255), np.uint8(0))
        occ_u8 = cv2.morphologyEx(occ_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        close_k = (3, 3) if export_ref_match else (5, 5)
        occ_u8 = cv2.morphologyEx(occ_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_k))

        vis_cur = (255 - occ_u8).astype(np.float32)

        # Asymmetric zero-lag temporal hysteresis: instant zero on occlusion, crisp recovery on exit
        if self._occ_prev_vis is not None and self._occ_prev_vis.shape == vis_cur.shape:
            vis_f = np.minimum(vis_cur, 0.85 * vis_cur + 0.15 * self._occ_prev_vis)
        else:
            vis_f = vis_cur
        self._occ_prev_vis = vis_f
        vis_u8 = np.clip(vis_f, 0.0, 255.0).astype(np.uint8)
        vis_u8[vis_u8 < 32] = 0
        vis_u8[vis_u8 >= 190] = 255
        occ_soft_u8 = (255 - vis_u8).astype(np.uint8)

        # Export textured unoccluded reference-match map only when called from neural VOS mode
        # (zero extra work on the classical GrabCut/ORB path!)
        if export_ref_match:
            ref_match_tex = (
                tmask
                & (~occ_clean)
                & has_tex
                & (ncc > 0.68)
                & (dL_s < 8.5)
                & (dab_s < 8.0)
                & (~tex_destroyed)
            )
            if self._occ_is_color and self.occlusion_mode in ("ref_plus_boundary", "full"):
                ref_match_tex &= skin_excess < 0.14
            # If reference-diff confirms the planar surface is unoccluded (e.g. bar/hand absent or just exited),
            # restore 100% of the crisp homography card rectangle (immune to post-occlusion VOS memory erosion!).
            occ_frac_card = float(np.count_nonzero(occ_clean & tmask)) / float(max(int(np.count_nonzero(tmask)), 1))
            if occ_frac_card < 0.025 and float(np.median(dL_s[tmask])) < 6.5 and float(np.median(dab_s[tmask])) < 6.5:
                self._last_2d_ref_match_u8 = np.full((self._occ_size, self._occ_size), 255, dtype=np.uint8)
            else:
                tex_dil = (
                    cv2.dilate(ref_match_tex.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                    > 0
                )
                ref_match_full = ref_match_tex | (
                    tex_dil & tmask & (~occ_clean) & (dL_s < 6.5) & (dab_s < 6.0) & (~tex_destroyed)
                )
                if self._occ_is_color and self.occlusion_mode in ("ref_plus_boundary", "full"):
                    ref_match_full &= skin_excess < 0.14
                ref_match_crop = ref_match_full[pad:-pad, pad:-pad]
                self._last_2d_ref_match_u8 = np.where(ref_match_crop, np.uint8(255), np.uint8(0))
        else:
            self._last_2d_ref_match_u8 = None

        # Store native-space warp matrix and export occluder mask for tracker + guided-filter compositing
        self._last_2d_occ_u8 = occ_soft_u8 if int(np.count_nonzero(occ_soft_u8 > 16)) >= 12 else None
        self._last_2d_H_crop2img = None
        try:
            H_c2img = H.astype(np.float64) @ np.linalg.inv(self._occ_H_ref_to_canon)
            T_pad = np.array([[1.0, 0.0, float(pad)], [0.0, 1.0, float(pad)], [0.0, 0.0, 1.0]], dtype=np.float64)
            H_crop2img = H_c2img @ T_pad
            self._last_2d_H_crop2img = H_crop2img
            if self._last_2d_occ_u8 is not None and int(np.count_nonzero(occ_u8)) >= 64:
                fh, fw = frame_bgr.shape[:2]
                self.last_occluder_mask_img = cv2.warpPerspective(
                    occ_u8, H_crop2img, (fw, fh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
        except np.linalg.LinAlgError:
            self._last_2d_occ_u8 = None
            self._last_2d_ref_match_u8 = None
            self._last_2d_H_crop2img = None
            self.last_occluder_mask_img = None

        # Return full-res clean texture for rasterization; native-space guided-filter occlusion
        # refinement is applied in composite() for crisp, non-staircased edges!
        return self._placed_rgba

    def _is_in_6dof_target_color(self, bgr_img: np.ndarray) -> np.ndarray:
        """Return bool mask of pixels whose (L, a, b) color falls in the dilated frame-0 target color LUT."""
        if self._occ_6dof_target_lab_lut is None:
            return np.zeros(bgr_img.shape[:2], dtype=bool)
        lab_u8 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l_idx = np.clip(lab_u8[:, :, 0] // 24, 0, 11)
        a_idx = np.clip(lab_u8[:, :, 1] // 10, 0, 26)
        b_idx = np.clip(lab_u8[:, :, 2] // 10, 0, 26)
        return self._occ_6dof_target_lab_lut[l_idx, a_idx, b_idx]

    def _apply_6dof_uv_occlusion(
        self,
        frame_bgr: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        intr: tuple[float, float, float, float],
    ) -> np.ndarray:
        """3D-projected UV reference-appearance occlusion mask for mesh_6dof with 5x5 Birchfield-Tomasi & boundary gating."""
        if (
            not self._occ_6dof_enabled
            or self._occ_6dof_uv_pts3d is None
            or self._occ_6dof_uv_norms is None
            or self._occ_6dof_ref_uv_L is None
            or self._occ_6dof_ref_uv_L_min is None
            or self._occ_6dof_ref_uv_L_max is None
            or self._occ_6dof_ref_uv_ab is None
            or self._occ_6dof_ref_uv_ab_min is None
            or self._occ_6dof_ref_uv_ab_max is None
            or self._occ_6dof_ref_uv_valid is None
            or self._work_rgba is None
            or self._base_alpha_u8 is None
        ):
            return self._placed_rgba

        h, w = frame_bgr.shape[:2]
        fx, fy, cx, cy = intr
        R, _ = cv2.Rodrigues(rvec.astype(np.float64).reshape(3))
        tv = tvec.astype(np.float64).reshape(1, 3)
        vh, uw = self._occ_6dof_ref_uv_L.shape

        pc = self._occ_6dof_uv_pts3d @ R.T + tv
        nc = self._occ_6dof_uv_norms @ R.T
        view = -pc / np.maximum(np.linalg.norm(pc, axis=1, keepdims=True), 1e-6)
        front = ((np.sum(nc * view, axis=1) > 0.22) & (pc[:, 2] > 0.04)).reshape(vh, uw)

        mx = (pc[:, 0] / np.maximum(pc[:, 2], 1e-4) * fx + cx).reshape(vh, uw).astype(np.float32)
        my = (pc[:, 1] / np.maximum(pc[:, 2], 1e-4) * fy + cy).reshape(vh, uw).astype(np.float32)
        cur_uv_bgr = cv2.remap(frame_bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cur_lab = cv2.cvtColor(cur_uv_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        cur_L = cur_lab[:, :, 0]
        cur_ab = cur_lab[:, :, 1:3]

        # Compare ONLY on UV columns that were observed in the verified reference bank (never erode newly revealed backside)
        overlap = front & self._occ_6dof_ref_uv_valid & (mx >= 1) & (mx < w - 1) & (my >= 1) & (my < h - 1)
        occ_uv = np.zeros((vh, uw), dtype=bool)
        if int(np.count_nonzero(overlap)) >= 48:
            rv = self._occ_6dof_ref_uv_L[overlap]
            cv_ = cur_L[overlap]
            raw_d = np.abs(cv_ - rv)
            inl = raw_d <= max(float(np.median(raw_d)), 12.0)
            gain = float(np.clip(float(rv[inl].mean()) / max(float(cv_[inl].mean()), 10.0), 0.65, 1.55)) if int(np.count_nonzero(inl)) >= 24 else 1.0
            self._last_6dof_gain = gain
            cur_L_n = np.clip(cur_L * gain, 0.0, 255.0)
            # 5x5 Birchfield-Tomasi distance absorbs 1-2 texel rotational pose shifts without eroding sharp labels/text
            dL_bt = np.maximum(0.0, np.maximum(self._occ_6dof_ref_uv_L_min - cur_L_n, cur_L_n - self._occ_6dof_ref_uv_L_max))
            dab_bt_v = np.maximum(0.0, np.maximum(self._occ_6dof_ref_uv_ab_min - cur_ab, cur_ab - self._occ_6dof_ref_uv_ab_max))
            dL = cv2.GaussianBlur(dL_bt, (3, 3), 0.0)
            dab = cv2.GaussianBlur(np.hypot(dab_bt_v[:, :, 0], dab_bt_v[:, :, 1]), (3, 3), 0.0)
            raw_occ = overlap & ((dab > 34.0) | ((dL > 32.0) & (dab > 16.0)) | ((~self._occ_6dof_is_color) & (dL > 36.0)))

            # Real foreground occluders (hands, fingers, bars, neighboring bottles) enter from the silhouette
            # boundary. Filtering out isolated interior patches eliminates popping Swiss-cheese holes over
            # shifted printed text (e.g. AKAC, TANGY BBQ) and specular highlights!
            if np.any(raw_occ):
                front_u8 = front.astype(np.uint8) * 255
                front_er = cv2.erode(front_u8, np.ones((5, 5), np.uint8)) > 0
                front_bnd = front & (~front_er)
                # Top/bottom rows of UV cylinder are also physical silhouette boundaries
                front_bnd[0:2, :] = True
                front_bnd[-2:, :] = True
                n_oc, oc_lbl, oc_stats, _ = cv2.connectedComponentsWithStats(raw_occ.astype(np.uint8), connectivity=8)
                if n_oc > 1:
                    flat_o = oc_lbl.ravel()
                    bnd_hits = np.bincount(flat_o, weights=front_bnd.ravel().astype(np.float32), minlength=n_oc)
                    keep_o = np.zeros(n_oc, dtype=bool)
                    for lbl in range(1, n_oc):
                        area = int(oc_stats[lbl, cv2.CC_STAT_AREA])
                        if bnd_hits[lbl] >= 2 and area >= 28:
                            keep_o[lbl] = True
                    occ_uv = keep_o[oc_lbl]

        occ_u8 = np.where(occ_uv, np.uint8(255), np.uint8(0))
        occ_u8 = cv2.morphologyEx(occ_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        occ_u8 = cv2.morphologyEx(occ_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        vis_cur = (255 - occ_u8).astype(np.float32)
        if self._occ_6dof_prev_vis_uv is not None and self._occ_6dof_prev_vis_uv.shape == vis_cur.shape:
            vis_f = np.minimum(vis_cur, 0.72 * vis_cur + 0.28 * self._occ_6dof_prev_vis_uv)
        else:
            vis_f = vis_cur
        self._occ_6dof_prev_vis_uv = vis_f
        vis_u8 = np.clip(vis_f, 0.0, 255.0).astype(np.uint8)
        vis_u8[vis_u8 < 32] = 0

        tex_h, tex_w = self._base_alpha_u8.shape
        vis_tex = cv2.resize(vis_u8, (tex_w, tex_h), interpolation=cv2.INTER_LINEAR)
        self._work_rgba[:, :, 3] = cv2.multiply(self._base_alpha_u8, vis_tex, scale=1.0 / 255.0)
        return self._work_rgba

    def _compute_6dof_exterior_hand_mask(
        self,
        frame_bgr: np.ndarray,
        sil_mask: np.ndarray,
        *,
        depth_buf: np.ndarray | None = None,
        rvec: np.ndarray | None = None,
        tvec: np.ndarray | None = None,
        intr: tuple[float, float, float, float] | None = None,
    ) -> np.ndarray | None:
        """Detect exterior-connected hands/occluders & non-target rim overhang with guided-filter edge snapping.

        Returns HxW uint8 soft alpha mask (0..255) for smooth edge-aligned compositing.
        """
        if not self._occ_6dof_enabled:
            return None
        if self.occlusion_mode not in ("ref_plus_boundary", "full"):
            return None
        if self._occ_6dof_is_color and self._occ_6dof_ref_skin_frac >= 0.45:
            return None
        ys, xs = np.nonzero(sil_mask)
        if len(xs) < 64:
            return None
        h, w = frame_bgr.shape[:2]
        pad = 24
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(w, int(xs.max()) + pad + 1)
        y1 = min(h, int(ys.max()) + pad + 1)

        roi_bgr_full = frame_bgr[y0:y1, x0:x1]
        roi_bgr = roi_bgr_full[::2, ::2]
        roi_sil = sil_mask[y0:y1:2, x0:x1:2] > 0
        n_sil_px = int(np.count_nonzero(roi_sil))
        if n_sil_px < 16:
            return None
        ext = ~roi_sil
        occ_roi = np.zeros(roi_sil.shape, dtype=bool)
        hand_only_roi = np.zeros(roi_sil.shape, dtype=bool)
        sil_u8_roi = roi_sil.astype(np.uint8) * 255
        sil_core = cv2.erode(sil_u8_roi, np.ones((3, 3), np.uint8), iterations=1) > 0

        # Exact per-pixel 3D depth -> cylinder UV reference-match protection:
        # Any pixel inside the silhouette whose current Lab color STILL MATCHES its verified frame-0
        # baked UV appearance (e.g. red/white paper label on blue pitcher or orange bottom band on soup can)
        # is unequivocally the object itself and must NEVER be masked out as a hand or background!
        uv_matches_ref = np.zeros(roi_sil.shape, dtype=bool)
        if (
            depth_buf is not None
            and rvec is not None
            and tvec is not None
            and intr is not None
            and self._occ_6dof_ref_uv_valid is not None
            and self._occ_6dof_ref_uv_L_min is not None
            and self._occ_6dof_ref_uv_L_max is not None
            and self._occ_6dof_ref_uv_ab_min is not None
            and self._occ_6dof_ref_uv_ab_max is not None
        ):
            roi_z = depth_buf[y0:y1:2, x0:x1:2]
            valid_z = roi_sil & np.isfinite(roi_z) & (roi_z > 0.03) & (roi_z < 100.0)
            if np.any(valid_z):
                ry_i, rx_i = np.nonzero(valid_z)
                fx, fy, cx, cy = intr
                px_img = (x0 + 2 * rx_i).astype(np.float64)
                py_img = (y0 + 2 * ry_i).astype(np.float64)
                z_v = roi_z[ry_i, rx_i].astype(np.float64)
                p_cam = np.column_stack([
                    (px_img - cx) * z_v / fx,
                    (py_img - cy) * z_v / fy,
                    z_v,
                ])
                R_m, _ = cv2.Rodrigues(rvec.astype(np.float64).reshape(3))
                t_m = tvec.astype(np.float64).reshape(1, 3)
                p_obj = (p_cam - t_m) @ R_m
                theta = np.arctan2(p_obj[:, 0], -p_obj[:, 2])
                u_f = theta / (2.0 * math.pi) + 0.5
                v_f = (p_obj[:, 1] - self._occ_6dof_y_min) / self._occ_6dof_cyl_h
                vh, uw = self._occ_6dof_ref_uv_valid.shape
                iu = np.clip(np.round(u_f * (uw - 1)).astype(np.int32), 0, uw - 1)
                iv = np.clip(np.round(v_f * (vh - 1)).astype(np.int32), 0, vh - 1)
                in_ref = self._occ_6dof_ref_uv_valid[iv, iu]
                if np.any(in_ref):
                    roi_lab_f = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                    cL = np.clip(roi_lab_f[ry_i, rx_i, 0] * float(getattr(self, "_last_6dof_gain", 1.0)), 0.0, 255.0)
                    cab = roi_lab_f[ry_i, rx_i, 1:3]
                    dL_p = np.maximum(
                        0.0,
                        np.maximum(self._occ_6dof_ref_uv_L_min[iv, iu] - cL, cL - self._occ_6dof_ref_uv_L_max[iv, iu]),
                    )
                    dab_v = np.maximum(
                        0.0,
                        np.maximum(self._occ_6dof_ref_uv_ab_min[iv, iu] - cab, cab - self._occ_6dof_ref_uv_ab_max[iv, iu]),
                    )
                    dab_p = np.hypot(dab_v[:, 0], dab_v[:, 1])
                    matched = in_ref & (dL_p < 22.0) & (dab_p < 18.0)
                    uv_matches_ref[ry_i[matched], rx_i[matched]] = True

        if self._occ_6dof_is_color:
            in_target_color = self._is_in_6dof_target_color(roi_bgr) | uv_matches_ref

            # 1. Exterior-connected gripping hands/fingers that penetrate inside the silhouette core
            skin = _skin_prob_bgr(roi_bgr)
            skin_cand = (skin > 0.32) & (~(roi_sil & in_target_color))
            skin_bin = cv2.morphologyEx(skin_cand.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            n_cc, labels, _, _ = cv2.connectedComponentsWithStats(skin_bin, connectivity=8)
            if n_cc > 1:
                flat = labels.ravel()
                ext_cnt = np.bincount(flat, weights=ext.ravel().astype(np.float32), minlength=n_cc)
                in_cnt = np.bincount(flat, weights=roi_sil.ravel().astype(np.float32), minlength=n_cc)
                core_cnt = np.bincount(flat, weights=sil_core.ravel().astype(np.float32), minlength=n_cc)
                keep = np.zeros(n_cc, dtype=bool)
                for lbl in range(1, n_cc):
                    if ext_cnt[lbl] >= 9 and in_cnt[lbl] >= 6 and core_cnt[lbl] >= 2:
                        keep[lbl] = True
                if np.any(keep):
                    hand_core = keep[labels].astype(np.uint8) * 255
                    hand_dil = cv2.dilate(hand_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
                    hand_halo = (cv2.dilate(hand_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0) & (skin > 0.18) & (~in_target_color)
                    hand_in_sil = (hand_dil | hand_halo) & roi_sil & (~uv_matches_ref)
                    if int(np.count_nonzero(hand_in_sil)) <= int(0.60 * max(n_sil_px, 1)):
                        occ_roi |= hand_in_sil
                        hand_only_roi |= hand_in_sil

            # 2. Exterior-connected non-target background / foreground-neighbor trim (e.g., adjacent yellow mustard
            # bottle overlapping blue pitcher or dark desk visible through handle air-gap).
            if self._occ_6dof_target_lab_lut is not None:
                sil_w_roi = max(1, int(xs.max() - xs.min() + 1))
                rim_k = min(31, max(7, int(round(0.24 * sil_w_roi)) | 1))
                k_rim = cv2.getStructuringElement(cv2.MORPH_RECT, (rim_k, rim_k))
                interior = cv2.erode(sil_u8_roi, k_rim) > 0
                ext_ring = (cv2.dilate(sil_u8_roi, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))) > 0) & ext
                if int(np.count_nonzero(ext_ring)) >= 10:
                    roi_lab_u8 = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
                    bg_pts = roi_lab_u8[ext_ring]
                    bg_cnt = np.zeros((12, 23, 23), dtype=np.int32)
                    np.add.at(
                        bg_cnt,
                        (
                            np.clip(bg_pts[:, 0] // 24, 0, 11),
                            np.clip(bg_pts[:, 1] // 12, 0, 22),
                            np.clip(bg_pts[:, 2] // 12, 0, 22),
                        ),
                        1,
                    )
                    bg_lut = bg_cnt >= max(4, int(0.025 * len(bg_pts)))
                    in_bg_color = bg_lut[
                        np.clip(roi_lab_u8[:, :, 0] // 24, 0, 11),
                        np.clip(roi_lab_u8[:, :, 1] // 12, 0, 22),
                        np.clip(roi_lab_u8[:, :, 2] // 12, 0, 22),
                    ]
                    allowed_trim = ext | (roi_sil & (~interior))
                    non_target_rim = allowed_trim & (~in_target_color) & (ext | in_bg_color)
                    n_bg, bg_lbl, _, _ = cv2.connectedComponentsWithStats(non_target_rim.astype(np.uint8), connectivity=4)
                    if n_bg > 1:
                        flat_b = bg_lbl.ravel()
                        b_ext = np.bincount(flat_b, weights=ext.ravel().astype(np.float32), minlength=n_bg)
                        b_in = np.bincount(flat_b, weights=(roi_sil & (~interior)).ravel().astype(np.float32), minlength=n_bg)
                        keep_bg = np.zeros(n_bg, dtype=bool)
                        for lbl in range(1, n_bg):
                            if b_ext[lbl] >= 9 and b_in[lbl] >= 3:
                                keep_bg[lbl] = True
                        if np.any(keep_bg):
                            rim_occ = keep_bg[bg_lbl] & roi_sil
                            rim_occ = cv2.dilate(rim_occ.astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0
                            occ_roi |= rim_occ & roi_sil & (~uv_matches_ref)
        else:
            # Grayscale/monochrome exterior-connected hand/occluder trim (e.g. darker fingers gripping a bright mug)
            roi_gray_s = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            l_lo = self._occ_6dof_ref_L_lo - 10.0
            l_hi = self._occ_6dof_ref_L_hi + 10.0
            # Only apply when target object has a reasonably coherent intensity band (not full 0..255 span)
            if (self._occ_6dof_ref_L_hi - self._occ_6dof_ref_L_lo) < 110.0:
                non_target_L = (roi_gray_s < l_lo) | (roi_gray_s > l_hi)
                cand = cv2.morphologyEx(non_target_L.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                n_cc, labels, _, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
                if n_cc > 1:
                    flat = labels.ravel()
                    ext_cnt = np.bincount(flat, weights=ext.ravel().astype(np.float32), minlength=n_cc)
                    in_cnt = np.bincount(flat, weights=roi_sil.ravel().astype(np.float32), minlength=n_cc)
                    keep = np.zeros(n_cc, dtype=bool)
                    for lbl in range(1, n_cc):
                        if ext_cnt[lbl] >= 12 and in_cnt[lbl] >= 6:
                            keep[lbl] = True
                    if np.any(keep):
                        mono_occ = keep[labels] & roi_sil
                        if int(np.count_nonzero(mono_occ)) <= int(0.70 * max(n_sil_px, 1)):
                            occ_roi |= mono_occ
                            hand_only_roi |= mono_occ

        roi_w, roi_h = x1 - x0, y1 - y0
        self._last_6dof_hand_only_mask: np.ndarray | None = None
        if np.any(hand_only_roi):
            fh_only = np.zeros((h, w), dtype=np.uint8)
            fh_only[y0:y1, x0:x1] = cv2.resize(
                np.where(hand_only_roi, np.uint8(255), np.uint8(0)),
                (roi_w, roi_h),
                interpolation=cv2.INTER_NEAREST,
            )
            self._last_6dof_hand_only_mask = fh_only

        if not np.any(occ_roi):
            return None

        # Mark exterior pixels outside roi_sil as 1.0 BEFORE guided filtering so the filter never averages
        # down to 0.5 along the outer mesh border (eliminating floating 1-px white outline fringes!),
        # then clip back to full-res silhouette.
        occ_for_guide = (occ_roi | (~roi_sil)).astype(np.float32)
        occ_up = cv2.resize(
            occ_for_guide,
            (roi_w, roi_h),
            interpolation=cv2.INTER_LINEAR,
        )
        roi_gray_full = cv2.cvtColor(roi_bgr_full, cv2.COLOR_BGR2GRAY)
        occ_alpha = _guided_filter_alpha(roi_gray_full, occ_up, radius=4, eps=0.012)
        occ_alpha = np.clip((occ_alpha - 0.18) / 0.64, 0.0, 1.0)
        # Ensure pixels strictly outside sil_mask stay 0 in the returned mask
        roi_sil_full = sil_mask[y0:y1, x0:x1] > 0
        occ_alpha = np.where(roi_sil_full, occ_alpha, 0.0)
        full_occ = np.zeros((h, w), dtype=np.uint8)
        full_occ[y0:y1, x0:x1] = np.clip(occ_alpha * 255.0, 0.0, 255.0).astype(np.uint8)
        self._last_6dof_occ_roi = (x0, y0, x1, y1)
        return full_occ

    @staticmethod
    def _build_grid_topology(cols: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
        us = np.linspace(0.0, 1.0, cols + 1, dtype=np.float32)
        vs = np.linspace(0.0, 1.0, rows + 1, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)
        uvs = np.column_stack([uu.ravel(), vv.ravel()]).astype(np.float32)
        tris: list[list[int]] = []
        stride = cols + 1
        for r in range(rows):
            for c in range(cols):
                a = r * stride + c
                b = a + 1
                d = a + stride
                e = d + 1
                tris.append([a, b, e])
                tris.append([a, e, d])
        return np.ascontiguousarray(uvs, dtype=np.float32), np.ascontiguousarray(tris, dtype=np.int32)

    def _build_cylinder_topology(
        self,
        cols: int,
        rows: int,
        radius_frac: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build cylindrical foreshortening mesh in reference bbox (or oriented cylinder) coordinates."""
        bx, by, bw, bh = self.ref_bbox
        # Clamp angular sweep based on cylinder_radius_frac
        half_arc = math.radians(min(82.0, max(35.0, 74.0 * math.sqrt(max(0.15, radius_frac) / 0.5))))
        sin_max = math.sin(half_arc)

        thetas = np.linspace(-half_arc, half_arc, cols + 1, dtype=np.float64)
        x_fracs = 0.5 + 0.5 * (np.sin(thetas) / sin_max)
        u_vals = np.linspace(0.0, 1.0, cols + 1, dtype=np.float32)
        v_vals = np.linspace(0.0, 1.0, rows + 1, dtype=np.float32)

        ref_xy: list[list[float]] = []
        uvs: list[list[float]] = []
        if self._cyl_oriented is not None:
            (ocx, ocy), ow, oh, oang = self._cyl_oriented
            ca, sa = math.cos(oang), math.sin(oang)
            bow = 0.035 * float(oh) * min(1.5, max(0.3, radius_frac / 0.5))
            for v in v_vals:
                v_centered = (float(v) - 0.5) * 2.0
                for i, xf in enumerate(x_fracs):
                    cos_t = math.cos(float(thetas[i]))
                    lx = (float(xf) - 0.5) * float(ow)
                    ly = (float(v) - 0.5) * float(oh) + bow * (cos_t - math.cos(half_arc)) * (0.4 + 0.6 * abs(v_centered))
                    px = ocx + lx * ca - ly * sa
                    py = ocy + lx * sa + ly * ca
                    ref_xy.append([px, py])
                    uvs.append([float(u_vals[i]), float(v)])
        else:
            bow = 0.035 * float(bh) * min(1.5, max(0.3, radius_frac / 0.5))
            for v in v_vals:
                v_centered = (float(v) - 0.5) * 2.0  # -1 at top, +1 at bottom
                for i, xf in enumerate(x_fracs):
                    cos_t = math.cos(float(thetas[i]))
                    px = float(bx) + float(xf) * float(bw)
                    py = float(by) + float(v) * float(bh) + bow * (cos_t - math.cos(half_arc)) * (0.4 + 0.6 * abs(v_centered))
                    ref_xy.append([px, py])
                    uvs.append([float(u_vals[i]), float(v)])

        tris: list[list[int]] = []
        stride = cols + 1
        for r in range(rows):
            for c in range(cols):
                a = r * stride + c
                b = a + 1
                d = a + stride
                e = d + 1
                tris.append([a, b, e])
                tris.append([a, e, d])

        return (
            np.ascontiguousarray(ref_xy, dtype=np.float32),
            np.ascontiguousarray(uvs, dtype=np.float32),
            np.ascontiguousarray(tris, dtype=np.int32),
        )

    def _ensure_buffers(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        if self._buf_shape != (h, w) or self._depth_buf is None or self._mask_buf is None:
            self._buf_shape = (h, w)
            self._depth_buf = np.full((h, w), np.inf, dtype=np.float32)
            self._mask_buf = np.zeros((h, w), dtype=np.uint8)
        else:
            self._depth_buf.fill(np.inf)
            self._mask_buf.fill(0)
        return self._depth_buf, self._mask_buf

    def _project_homography_to_cam_verts(
        self,
        ref_xy: np.ndarray,
        H: np.ndarray,
    ) -> np.ndarray:
        """Lift 2D homography-warped vertices to projective camera coords (X, Y, W) with K = I.

        In vi_rasterize_mesh_rgbd with (fx=1, fy=1, cx=0, cy=0), vertex (X, Y, W) projects to (X/W, Y/W)
        with exact perspective-correct barycentric UV interpolation!
        """
        ones = np.ones((len(ref_xy), 1), dtype=np.float64)
        pts_h = np.hstack([ref_xy.astype(np.float64), ones])  # (N, 3)
        proj = pts_h @ H.astype(np.float64).T  # (N, 3) = [X, Y, W]
        # Ensure positive W (normalize sign of H if needed)
        if float(np.median(proj[:, 2])) < 0.0:
            proj = -proj
        # Clamp tiny W safely above near-plane
        proj[:, 2] = np.maximum(proj[:, 2], 1e-3)
        return np.ascontiguousarray(proj, dtype=np.float32)

    def composite(
        self,
        frame_bgr: np.ndarray,
        track_result: TrackFrameResult,
        *,
        external_fg_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, float]:
        """Composite artwork onto `frame_bgr` if `track_result.state == 'tracking'`.

        Strictly suppresses overlay (returns untouched frame) when state != 'tracking'.
        When `external_fg_mask` (HxW uint8 foreground mask from neural VOS) is provided,
        bypasses heuristic occlusion carving and intersects the rasterized artwork cleanly
        with an edge-snapped Fast Guided Filter matte of the live VOS foreground mask.
        Returns `(out_bgr, rendered_mask_or_None, composite_ms)`.
        """
        t0 = time.perf_counter()
        if (
            track_result.state != "tracking"
            or track_result.confidence_state in ("AMBIGUOUS", "LOST", "RECOVERING")
            or self.placement.opacity <= 0.0
        ):
            self._occ_prev_vis = None
            self._occ_prev_L_half = None
            self._occ_6dof_prev_vis_uv = None
            self._last_2d_occ_u8 = None
            self._last_2d_H_crop2img = None
            self.last_occluder_mask_img = None
            return frame_bgr, None, (time.perf_counter() - t0) * 1000.0

        h, w = frame_bgr.shape[:2]
        use_vos = external_fg_mask is not None and external_fg_mask.shape == (h, w)
        if use_vos and int(np.count_nonzero(external_fg_mask)) < 32:
            self.last_occluder_mask_img = None
            return frame_bgr, None, (time.perf_counter() - t0) * 1000.0

        depth_buf, mask_buf = self._ensure_buffers(h, w)
        mode = self.placement.wrap_mode
        self.last_occluder_mask_img = None
        self._last_2d_occ_u8 = None
        self._last_2d_ref_match_u8 = None
        self._last_2d_H_crop2img = None

        if mode == "mesh_6dof":
            if (
                self.mesh is None
                or track_result.pose_rvec is None
                or track_result.pose_tvec is None
            ):
                return frame_bgr, None, (time.perf_counter() - t0) * 1000.0
            R, _ = cv2.Rodrigues(track_result.pose_rvec.astype(np.float64))
            tvec = track_result.pose_tvec.astype(np.float64).reshape(1, 3)
            verts_local = self.mesh.vertices.astype(np.float64)
            # In neural VOS mode, pad parametric cylinder render envelope slightly so EdgeTAM's exact
            # pixel silhouette is the sole outer cutter (eliminates rigid blunt clipping on flared spouts/bases/limbs).
            if use_vos and self.mesh.source_kind == "parametric" and self.mesh.name == "cylinder":
                verts_local = verts_local.copy()
                verts_local[:, 0] *= 1.16
                verts_local[:, 1] *= 1.04
                verts_local[:, 2] *= 1.16
            verts_cam = np.ascontiguousarray(
                (verts_local @ R.T + tvec),
                dtype=np.float32,
            )
            if self.intrinsics is not None:
                fx, fy, cx, cy = self.intrinsics
            else:
                f = float(max(w, h)) * 0.9
                fx, fy, cx, cy = f, f, w * 0.5, h * 0.5
            triangles = self.mesh.triangles
            # Backface culling for closed/curved 3D meshes: keep grazing limb triangles (-0.02)
            # while still preventing inside back-wall triangles from bleeding across top rims/holes.
            if self.mesh.name != "plane" and len(triangles) > 0 and len(self.mesh.normals) == len(self.mesh.vertices):
                norms_cam = self.mesh.normals.astype(np.float64) @ R.T
                t_norms = norms_cam[triangles].mean(axis=1)
                t_cent = verts_cam[triangles].astype(np.float64).mean(axis=1)
                c_len = np.maximum(np.linalg.norm(t_cent, axis=1), 1e-6)
                n_len = np.maximum(np.linalg.norm(t_norms, axis=1), 1e-6)
                facing = np.sum(t_norms * (-t_cent), axis=1) / (c_len * n_len)
                front_tris = triangles[facing > -0.02]
                if len(front_tris) > 0:
                    triangles = np.ascontiguousarray(front_tris, dtype=np.int32)
            uvs = self.mesh.uvs
            intr = (fx, fy, cx, cy)
            wrap_repeat = self.placement.tile_repeat
            if use_vos:
                tex_rgba = self._placed_rgba
            else:
                tex_rgba = (
                    self._apply_6dof_uv_occlusion(frame_bgr, track_result.pose_rvec, track_result.pose_tvec, intr)
                    if self._occ_6dof_enabled
                    else self._placed_rgba
                )
        else:
            if track_result.homography_3x3 is None:
                return frame_bgr, None, (time.perf_counter() - t0) * 1000.0
            H = track_result.homography_3x3
            bx, by, bw, bh = self.ref_bbox
            if mode == "planar":
                ref_xy = np.column_stack([
                    bx + self._planar_uvs[:, 0] * bw,
                    by + self._planar_uvs[:, 1] * bh,
                ]).astype(np.float32)
                verts_cam = self._project_homography_to_cam_verts(ref_xy, H)
                triangles = self._planar_tris
                uvs = self._planar_uvs
            elif mode == "cylinder":
                cyl_xy = self._cyl_ref_xy
                if use_vos:
                    # Pad 2D cylinder mesh slightly so live VOS silhouette is the sole outer cutter
                    # (eliminates narrow unpainted crescents along curved bottle flanks).
                    c_mid = cyl_xy.mean(axis=0, keepdims=True)
                    cyl_xy = (c_mid + (cyl_xy - c_mid) * np.array([[1.08, 1.05]], dtype=np.float32)).astype(np.float32)
                verts_cam = self._project_homography_to_cam_verts(cyl_xy, H)
                triangles = self._cyl_tris
                uvs = self._cyl_uvs
            elif mode == "mesh_grid":
                if track_result.grid_dst_pts is not None and len(track_result.grid_dst_pts) == len(self._grid_uvs):
                    dst_xy = track_result.grid_dst_pts.astype(np.float64)
                    ref_xy = track_result.grid_src_pts.astype(np.float64) if track_result.grid_src_pts is not None else np.column_stack([bx + self._grid_uvs[:, 0] * bw, by + self._grid_uvs[:, 1] * bh])
                    ones = np.ones((len(ref_xy), 1), dtype=np.float64)
                    w_vals = (np.hstack([ref_xy, ones]) @ H.astype(np.float64).T)[:, 2]
                    if float(np.median(w_vals)) < 0.0:
                        w_vals = -w_vals
                    w_vals = np.maximum(w_vals, 1e-3)
                    u_c = self._grid_uvs[:, 0] - 0.5
                    v_c = self._grid_uvs[:, 1] - 0.5
                    r2 = u_c * u_c + v_c * v_c
                    uvs = np.column_stack([
                        np.clip(0.5 + u_c * (1.0 - 0.14 * r2), 0.0, 1.0),
                        np.clip(0.5 + v_c * (1.0 - 0.14 * r2), 0.0, 1.0),
                    ]).astype(np.float32)
                    verts_cam = np.ascontiguousarray(
                        np.column_stack([dst_xy[:, 0] * w_vals, dst_xy[:, 1] * w_vals, w_vals]),
                        dtype=np.float32,
                    )
                else:
                    ref_xy = np.column_stack([
                        bx + self._grid_uvs[:, 0] * bw,
                        by + self._grid_uvs[:, 1] * bh,
                    ]).astype(np.float32)
                    verts_cam = self._project_homography_to_cam_verts(ref_xy, H)
                    uvs = self._grid_uvs
                triangles = self._grid_tris
            else:
                raise ContractValidationError(f"Unsupported wrap_mode: {mode!r}")

            intr = (1.0, 1.0, 0.0, 0.0)
            wrap_repeat = False  # Placement/repeat already baked into _placed_rgba
            if use_vos:
                if self._placed_rgba_unclipped is None:
                    self._placed_rgba_unclipped = _prepare_placed_texture(
                        self._artwork_rgba,
                        self.placement,
                        surface_mask_crop=None,
                    )
                tex_rgba = self._placed_rgba_unclipped
                # On color planar surfaces, also run fast local texture-destruction check to catch
                # same-color foreground bars that split the card and can cause VOS bridging.
                if mode == "planar" and self._occ_enabled and self._occ_is_color:
                    self._apply_local_occlusion_mask(frame_bgr, H, export_ref_match=True)
            else:
                tex_rgba = self._apply_local_occlusion_mask(frame_bgr, H) if self._occ_enabled else self._placed_rgba

        frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res: RasterizeResult = rasterize_mesh_rgbd(
            verts_cam,
            triangles,
            uvs,
            tex_rgba,
            intr,
            frame_rgb,
            depth_buf=depth_buf,
            mask_out=mask_buf,
            wrap_repeat=wrap_repeat,
            opacity=self.placement.opacity,
            blend_mode=self.placement.blend_mode,
            backend=self.backend,
        )
        out_bgr = cv2.cvtColor(res.frame_rgb, cv2.COLOR_RGB2BGR)
        out_mask = res.mask

        if use_vos and external_fg_mask is not None:
            ys_m, xs_m = np.nonzero(out_mask > 0)
            if len(xs_m) < 24:
                return frame_bgr, None, (time.perf_counter() - t0) * 1000.0
            pad_r = 10
            x0 = max(0, int(xs_m.min()) - pad_r)
            y0 = max(0, int(ys_m.min()) - pad_r)
            x1 = min(w, int(xs_m.max()) + pad_r + 1)
            y1 = min(h, int(ys_m.max()) + pad_r + 1)
            vos_roi = external_fg_mask[y0:y1, x0:x1]
            if not np.any(vos_roi > 0):
                return frame_bgr, None, (time.perf_counter() - t0) * 1000.0
            if mode == "cylinder":
                # Fill strictly enclosed interior specular slits (up to 3.5% of fg area) inside solid cylinders
                # (eliminates vertical shiny-plastic holes like ycb_s50 f=25..44 without affecting exterior cutouts!)
                fg_r_cnt = int(np.count_nonzero(vos_roi > 127))
                if fg_r_cnt >= 128:
                    inv_r = ((vos_roi <= 127).astype(np.uint8)) * 255
                    n_hr, hr_lbl, hr_stats, _ = cv2.connectedComponentsWithStats(inv_r, connectivity=4)
                    if n_hr > 2:
                        bnd_r = set(np.unique(hr_lbl[0, :]).tolist())
                        bnd_r.update(np.unique(hr_lbl[-1, :]).tolist())
                        bnd_r.update(np.unique(hr_lbl[:, 0]).tolist())
                        bnd_r.update(np.unique(hr_lbl[:, -1]).tolist())
                        max_slit = max(32, int(0.035 * fg_r_cnt))
                        filled_r = vos_roi.copy()
                        changed_r = False
                        for hl in range(1, n_hr):
                            if hl not in bnd_r and int(hr_stats[hl, cv2.CC_STAT_AREA]) <= max_slit:
                                filled_r[hr_lbl == hl] = 255
                                changed_r = True
                        if changed_r:
                            vos_roi = filled_r
            roi_gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            vos_alpha = _guided_filter_alpha(
                roi_gray,
                vos_roi.astype(np.float32) * (1.0 / 255.0),
                radius=2,
                eps=0.004,
            )
            vos_alpha = np.clip((vos_alpha - 0.16) / 0.68, 0.0, 1.0)
            if (
                mode == "planar"
                and self._occ_is_color
                and self._last_2d_H_crop2img is not None
            ):
                roi_w, roi_h = x1 - x0, y1 - y0
                T_roi = np.array([[1.0, 0.0, float(-x0)], [0.0, 1.0, float(-y0)], [0.0, 0.0, 1.0]], dtype=np.float64)
                H_to_roi = T_roi @ self._last_2d_H_crop2img
                # Restore strictly unoccluded reference-matching planar pixels (recovers inter-finger gaps & split islands!)
                if self._last_2d_ref_match_u8 is not None:
                    ref_roi_u8 = cv2.warpPerspective(
                        self._last_2d_ref_match_u8,
                        H_to_roi,
                        (roi_w, roi_h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    if np.any(ref_roi_u8 > 127):
                        ref_alpha = _guided_filter_alpha(
                            roi_gray,
                            ref_roi_u8.astype(np.float32) * (1.0 / 255.0),
                            radius=2,
                            eps=0.004,
                        )
                        ref_alpha = np.clip((ref_alpha - 0.25) / 0.55, 0.0, 1.0)
                        vos_alpha = np.maximum(vos_alpha, ref_alpha)
                # Subtract local reference-diff occluders (cleanly cuts out same-color crossing bars!)
                if self._last_2d_occ_u8 is not None:
                    occ_roi_u8 = cv2.warpPerspective(
                        self._last_2d_occ_u8,
                        H_to_roi,
                        (roi_w, roi_h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                        borderValue=0,
                    )
                    if np.any(occ_roi_u8 > 16):
                        occ_alpha = _guided_filter_alpha(
                            roi_gray,
                            occ_roi_u8.astype(np.float32) * (1.0 / 255.0),
                            radius=2,
                            eps=0.004,
                        )
                        occ_alpha = np.clip((occ_alpha - 0.16) / 0.68, 0.0, 1.0)
                        vos_alpha = vos_alpha * (1.0 - occ_alpha)
            if self._mono_bright_fg_L_lo is not None and mode in ("cylinder", "mesh_6dof"):
                # On bright uniform monochrome objects (e.g. white plastic bottle/mug), carve out
                # exterior-connected darker fingers/thumbs crossing the front surface.
                dark_cand = (roi_gray.astype(np.float32) < self._mono_bright_fg_L_lo).astype(np.uint8)
                n_dc, dc_lbl, dc_stats, _ = cv2.connectedComponentsWithStats(dark_cand, connectivity=4)
                if n_dc > 1:
                    ext_vos = vos_roi <= 127
                    flat_dc = dc_lbl.ravel()
                    ext_hits = np.bincount(flat_dc, weights=ext_vos.ravel().astype(np.float32), minlength=n_dc)
                    in_hits = np.bincount(flat_dc, weights=(~ext_vos).ravel().astype(np.float32), minlength=n_dc)
                    fg_dark = np.zeros(n_dc, dtype=bool)
                    for lbl in range(1, n_dc):
                        if ext_hits[lbl] >= 8 and in_hits[lbl] >= 16:
                            fg_dark[lbl] = True
                    if np.any(fg_dark):
                        dark_occ_f = fg_dark[dc_lbl].astype(np.float32)
                        dark_alpha = _guided_filter_alpha(roi_gray, dark_occ_f, radius=2, eps=0.005)
                        dark_alpha = np.clip((dark_alpha - 0.20) / 0.60, 0.0, 1.0)
                        vos_alpha = vos_alpha * (1.0 - dark_alpha)
            a3 = vos_alpha[:, :, None]
            roi_out = out_bgr[y0:y1, x0:x1].astype(np.float32)
            roi_src = frame_bgr[y0:y1, x0:x1].astype(np.float32)
            out_bgr[y0:y1, x0:x1] = np.clip(roi_out * a3 + roi_src * (1.0 - a3), 0.0, 255.0).astype(np.uint8)
            out_mask = out_mask.copy()
            roi_m = out_mask[y0:y1, x0:x1]
            roi_m[vos_alpha < 0.45] = 0
            # Close 1px UV-seam cracks and remove tiny stray specks (< 2.5% of main component,
            # or < 15% on monochrome cylinders where bright knuckles can form small islands below a gripping thumb)
            if int(np.count_nonzero(roi_m)) >= 64:
                n_cc, cc_lbl, cc_stats, _ = cv2.connectedComponentsWithStats((roi_m > 0).astype(np.uint8), connectivity=8)
                if n_cc > 2:
                    max_a = int(cc_stats[1:, cv2.CC_STAT_AREA].max())
                    keep_frac = 0.22 if (self._mono_bright_fg_L_lo is not None and mode in ("cylinder", "mesh_6dof")) else 0.025
                    min_keep = max(24, int(keep_frac * max_a))
                    for lbl in range(1, n_cc):
                        if int(cc_stats[lbl, cv2.CC_STAT_AREA]) < min_keep:
                            roi_m[cc_lbl == lbl] = 0
            out_mask[y0:y1, x0:x1] = roi_m
            # Also suppress RGB overlay on pruned speck pixels so no faint specks remain
            pruned_b = (roi_m == 0) & (vos_alpha >= 0.45)
            if np.any(pruned_b):
                out_bgr[y0:y1, x0:x1][pruned_b] = frame_bgr[y0:y1, x0:x1][pruned_b]
            if int(np.count_nonzero(out_mask)) < 24:
                return frame_bgr, None, (time.perf_counter() - t0) * 1000.0
            comp_ms = (time.perf_counter() - t0) * 1000.0
            return out_bgr, out_mask, comp_ms

        if mode == "mesh_6dof" and self._occ_6dof_enabled:
            sil_u8 = (res.depth_buf < 1e20).astype(np.uint8) * 255
            hand_mask = self._compute_6dof_exterior_hand_mask(
                frame_bgr,
                sil_u8,
                depth_buf=res.depth_buf,
                rvec=track_result.pose_rvec,
                tvec=track_result.pose_tvec,
                intr=intr,
            )
            if hand_mask is not None:
                x0, y0, x1, y1 = getattr(self, "_last_6dof_occ_roi", (0, 0, w, h))
                hm_roi = hand_mask[y0:y1, x0:x1]
                nz = hm_roi > 0
                if np.any(nz):
                    a = hm_roi[nz].astype(np.float32)[:, None] * (1.0 / 255.0)
                    roi_out = out_bgr[y0:y1, x0:x1]
                    roi_src = frame_bgr[y0:y1, x0:x1]
                    roi_out[nz] = np.clip(
                        roi_out[nz].astype(np.float32) * (1.0 - a) + roi_src[nz].astype(np.float32) * a,
                        0.0,
                        255.0,
                    ).astype(np.uint8)
                    out_mask = out_mask.copy()
                    out_mask[y0:y1, x0:x1][hm_roi >= 115] = 0
                self.last_occluder_mask_img = getattr(self, "_last_6dof_hand_only_mask", None)
            # Sanity gate: suppress shredded floating scraps (< 18% of projected 3D silhouette survived occlusion)
            n_sil = int(np.count_nonzero(sil_u8))
            n_ren = int(np.count_nonzero(out_mask))
            if n_sil >= 400 and n_ren < int(0.18 * n_sil):
                return frame_bgr, None, (time.perf_counter() - t0) * 1000.0
        elif mode == "planar" and self._last_2d_occ_u8 is not None and self._last_2d_H_crop2img is not None:
            ys_m, xs_m = np.nonzero(out_mask > 0)
            n_pre = len(xs_m)
            if n_pre >= 32:
                pad_r = 12
                x0 = max(0, int(xs_m.min()) - pad_r)
                y0 = max(0, int(ys_m.min()) - pad_r)
                x1 = min(w, int(xs_m.max()) + pad_r + 1)
                y1 = min(h, int(ys_m.max()) + pad_r + 1)
                roi_w, roi_h = x1 - x0, y1 - y0
                T_roi = np.array([[1.0, 0.0, float(-x0)], [0.0, 1.0, float(-y0)], [0.0, 0.0, 1.0]], dtype=np.float64)
                H_to_roi = T_roi @ self._last_2d_H_crop2img
                occ_roi_u8 = cv2.warpPerspective(
                    self._last_2d_occ_u8, H_to_roi, (roi_w, roi_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE, borderValue=0
                )
                if np.any(occ_roi_u8 > 16):
                    roi_gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                    occ_alpha = _guided_filter_alpha(roi_gray, occ_roi_u8.astype(np.float32) * (1.0 / 255.0), radius=4, eps=0.01)
                    occ_alpha = np.clip((occ_alpha - 0.16) / 0.68, 0.0, 1.0)
                    a3 = occ_alpha[:, :, None]
                    roi_out = out_bgr[y0:y1, x0:x1].astype(np.float32)
                    roi_src = frame_bgr[y0:y1, x0:x1].astype(np.float32)
                    out_bgr[y0:y1, x0:x1] = np.clip(roi_out * (1.0 - a3) + roi_src * a3, 0.0, 255.0).astype(np.uint8)
                    out_mask = out_mask.copy()
                    out_mask[y0:y1, x0:x1][occ_alpha >= 0.45] = 0
        comp_ms = (time.perf_counter() - t0) * 1000.0
        return out_bgr, out_mask, comp_ms
