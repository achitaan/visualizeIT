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
        raster_backend: Literal["auto", "native_cpp", "opencv"] = "auto",
        local_occlusion_mask: bool = True,
    ) -> None:
        self.placement = placement
        self.ref_bbox = tuple(int(v) for v in ref_bbox)
        self.mesh = mesh
        self.intrinsics = intrinsics
        self.local_occlusion_mask = bool(local_occlusion_mask)

        if raster_backend == "auto":
            self.backend: Literal["native_cpp", "opencv"] = (
                "native_cpp" if is_native_raster_available(auto_build=False) else "opencv"
            )
        elif raster_backend in ("native_cpp", "opencv"):
            self.backend = raster_backend
        else:
            raise ContractValidationError(f"Invalid raster_backend: {raster_backend!r}")

        # Crop surface mask to ref_bbox for planar/cylinder/mesh_grid alpha clipping
        mask_crop: np.ndarray | None = None
        if ref_mask is not None and placement.wrap_mode != "mesh_6dof":
            bx, by, bw, bh = self.ref_bbox
            h, w = ref_mask.shape[:2]
            x0, y0 = max(0, bx), max(0, by)
            x1, y1 = min(w, bx + bw), min(h, by + bh)
            if x1 > x0 and y1 > y0:
                crop = ref_mask[y0:y1, x0:x1]
                # Only apply silhouette clipping if mask is non-trivial (>= 20% of bbox)
                if float(np.count_nonzero(crop)) / float(crop.size) >= 0.20:
                    mask_crop = crop

        # For mesh_6dof we keep full texture (placement applied via texture transform; mesh geometry defines silhouette)
        self._placed_rgba = _prepare_placed_texture(
            artwork_rgba,
            placement,
            surface_mask_crop=mask_crop if placement.wrap_mode != "mesh_6dof" else None,
        )

        # Prebuild canonical local 2D grids for planar, cylinder, and mesh_grid
        self._planar_uvs, self._planar_tris = self._build_grid_topology(2, 2)
        self._cyl_ref_xy, self._cyl_uvs, self._cyl_tris = self._build_cylinder_topology(
            cols=24, rows=4, radius_frac=placement.cylinder_radius_frac
        )
        self._grid_uvs, self._grid_tris = self._build_grid_topology(4, 4)

        # Optional per-cell illumination-invariant local occlusion mask state (48x48 canonical patch)
        self._occ_enabled = False
        self._occ_H_ref_to_canon: np.ndarray | None = None
        self._occ_ref_ab: np.ndarray | None = None
        self._occ_ref_L_zm: np.ndarray | None = None
        self._occ_ref_L_norm: np.ndarray | None = None
        self._occ_cyl_map_x: np.ndarray | None = None
        self._occ_cyl_map_y: np.ndarray | None = None
        self._work_rgba: np.ndarray | None = None
        self._base_alpha_u8: np.ndarray | None = None
        if self.local_occlusion_mask and ref_frame_bgr is not None and placement.wrap_mode != "mesh_6dof":
            self._init_local_occlusion_reference(ref_frame_bgr)

        # Reusable scratch buffers (allocated lazily on first frame)
        self._buf_shape: tuple[int, int] | None = None
        self._depth_buf: np.ndarray | None = None
        self._mask_buf: np.ndarray | None = None

    def _init_local_occlusion_reference(self, ref_frame_bgr: np.ndarray) -> None:
        """Precompute 48x48 canonical Lab chroma & 6x6 block L-channel stats for fast hand/bg occlusion masking."""
        bx, by, bw, bh = self.ref_bbox
        if bw < 8 or bh < 8:
            return
        ref_quad = np.array(
            [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]],
            dtype=np.float32,
        )
        canon_quad = np.array([[0.0, 0.0], [47.0, 0.0], [47.0, 47.0], [0.0, 47.0]], dtype=np.float32)
        H_r2c = cv2.getPerspectiveTransform(ref_quad, canon_quad)
        ref_48 = cv2.warpPerspective(
            ref_frame_bgr, H_r2c, (48, 48), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        ref_lab = cv2.cvtColor(ref_48, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_L = ref_lab[:, :, 0]
        blocks = ref_L.reshape(6, 8, 6, 8).transpose(0, 2, 1, 3).reshape(36, 64)
        zm = blocks - blocks.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(zm, axis=1)

        self._occ_H_ref_to_canon = H_r2c.astype(np.float64)
        self._occ_ref_ab = np.ascontiguousarray(ref_lab[:, :, 1:3], dtype=np.float32)
        self._occ_ref_L_zm = np.ascontiguousarray(zm, dtype=np.float32)
        self._occ_ref_L_norm = np.ascontiguousarray(norms, dtype=np.float32)
        self._work_rgba = np.ascontiguousarray(self._placed_rgba.copy(), dtype=np.uint8)
        self._base_alpha_u8 = np.ascontiguousarray(self._placed_rgba[:, :, 3], dtype=np.uint8)

        if self.placement.wrap_mode == "cylinder":
            half_arc = math.radians(
                min(82.0, max(35.0, 74.0 * math.sqrt(max(0.15, self.placement.cylinder_radius_frac) / 0.5)))
            )
            sin_max = math.sin(half_arc)
            u_grid = np.linspace(0.0, 1.0, 48, dtype=np.float32)
            thetas = (u_grid * 2.0 - 1.0) * half_arc
            x_fracs = 0.5 + 0.5 * (np.sin(thetas) / sin_max)
            xs = np.clip(x_fracs * 47.0, 0.0, 47.0).astype(np.float32)
            ys = np.arange(48, dtype=np.float32)
            self._occ_cyl_map_x, self._occ_cyl_map_y = np.meshgrid(xs, ys)

        self._occ_enabled = True

    def _apply_local_occlusion_mask(self, frame_bgr: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Compute illumination-invariant 48x48 visibility mask and modulate artwork alpha (~0.25 ms)."""
        if (
            not self._occ_enabled
            or self._occ_H_ref_to_canon is None
            or self._occ_ref_ab is None
            or self._occ_ref_L_zm is None
            or self._occ_ref_L_norm is None
            or self._work_rgba is None
            or self._base_alpha_u8 is None
        ):
            return self._placed_rgba

        try:
            H_inv = np.linalg.inv(H.astype(np.float64))
        except np.linalg.LinAlgError:
            return self._placed_rgba

        M_cur_to_canon = self._occ_H_ref_to_canon @ H_inv
        cur_48 = cv2.warpPerspective(
            frame_bgr, M_cur_to_canon, (48, 48), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        cur_lab = cv2.cvtColor(cur_48, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Illumination-invariant chroma distance (a, b channels)
        diff_ab = cur_lab[:, :, 1:3] - self._occ_ref_ab
        dab = np.sqrt(diff_ab[:, :, 0] * diff_ab[:, :, 0] + diff_ab[:, :, 1] * diff_ab[:, :, 1])
        dab_s = cv2.GaussianBlur(dab, (5, 5), 0.0)

        # Contrast-invariant 6x6 blockwise L-channel normalized cross-correlation
        cur_L = cur_lab[:, :, 0]
        cur_blocks = cur_L.reshape(6, 8, 6, 8).transpose(0, 2, 1, 3).reshape(36, 64)
        cur_zm = cur_blocks - cur_blocks.mean(axis=1, keepdims=True)
        cur_norms = np.linalg.norm(cur_zm, axis=1)
        denom = self._occ_ref_L_norm * cur_norms
        # On textureless reference blocks (ref_norm < 12), rely on chroma shift only rather than noisy L NCC
        valid_tex = (self._occ_ref_L_norm >= 12.0) & (denom > 1e-4)
        corr_36 = np.ones(36, dtype=np.float32)
        if np.any(valid_tex):
            dots = np.sum(self._occ_ref_L_zm[valid_tex] * cur_zm[valid_tex], axis=1)
            corr_36[valid_tex] = dots / denom[valid_tex]
        corr_map = cv2.resize(corr_36.reshape(6, 6), (48, 48), interpolation=cv2.INTER_LINEAR)

        # Pixel is occluded if (moderate chroma shift AND structural mismatch) OR (strong chroma shift)
        occluded = ((dab_s > 14.0) & (corr_map < 0.45)) | (dab_s > 28.0)
        vis_48 = np.where(occluded, np.uint8(0), np.uint8(255))
        vis_48 = cv2.GaussianBlur(vis_48, (5, 5), 0.0)

        if self._occ_cyl_map_x is not None and self._occ_cyl_map_y is not None:
            vis_48 = cv2.remap(vis_48, self._occ_cyl_map_x, self._occ_cyl_map_y, cv2.INTER_LINEAR)

        tex_h, tex_w = self._base_alpha_u8.shape
        vis_tex = cv2.resize(vis_48, (tex_w, tex_h), interpolation=cv2.INTER_LINEAR)
        self._work_rgba[:, :, 3] = cv2.multiply(self._base_alpha_u8, vis_tex, scale=1.0 / 255.0)
        return self._work_rgba

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
        """Build cylindrical foreshortening mesh in reference bbox coordinates.

        Maps angular arc theta in [-max_angle, +max_angle] onto x_frac = 0.5 + 0.5 * sin(theta)/sin(max_angle),
        while texture u advances linearly with arc-length theta — creating authentic curved foreshortening
        around mugs, bottles, and cans.
        """
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
        # Subtle vertical curvature bow along top/bottom of cylinder silhouette
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
    ) -> tuple[np.ndarray, np.ndarray | None, float]:
        """Composite artwork onto `frame_bgr` if `track_result.state == 'tracking'`.

        Strictly suppresses overlay (returns untouched frame) when state != 'tracking'.
        Returns `(out_bgr, rendered_mask_or_None, composite_ms)`.
        """
        t0 = time.perf_counter()
        if track_result.state != "tracking" or self.placement.opacity <= 0.0:
            return frame_bgr, None, (time.perf_counter() - t0) * 1000.0

        h, w = frame_bgr.shape[:2]
        depth_buf, mask_buf = self._ensure_buffers(h, w)
        mode = self.placement.wrap_mode

        if mode == "mesh_6dof":
            if (
                self.mesh is None
                or track_result.pose_rvec is None
                or track_result.pose_tvec is None
            ):
                return frame_bgr, None, (time.perf_counter() - t0) * 1000.0
            R, _ = cv2.Rodrigues(track_result.pose_rvec.astype(np.float64))
            tvec = track_result.pose_tvec.astype(np.float64).reshape(1, 3)
            verts_cam = np.ascontiguousarray(
                (self.mesh.vertices.astype(np.float64) @ R.T + tvec),
                dtype=np.float32,
            )
            if self.intrinsics is not None:
                fx, fy, cx, cy = self.intrinsics
            else:
                f = float(max(w, h)) * 0.9
                fx, fy, cx, cy = f, f, w * 0.5, h * 0.5
            triangles = self.mesh.triangles
            uvs = self.mesh.uvs
            intr = (fx, fy, cx, cy)
            wrap_repeat = self.placement.tile_repeat
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
                verts_cam = self._project_homography_to_cam_verts(self._cyl_ref_xy, H)
                triangles = self._cyl_tris
                uvs = self._cyl_uvs
            elif mode == "mesh_grid":
                if track_result.grid_dst_pts is not None and len(track_result.grid_dst_pts) == len(self._grid_uvs):
                    # Subtle pillow-grid wave modulation + tracked grid points
                    dst_xy = track_result.grid_dst_pts.astype(np.float64)
                    # Lift grid via homography W for perspective correctness while honoring grid_dst_pts
                    ref_xy = track_result.grid_src_pts.astype(np.float64) if track_result.grid_src_pts is not None else np.column_stack([bx + self._grid_uvs[:, 0] * bw, by + self._grid_uvs[:, 1] * bh])
                    ones = np.ones((len(ref_xy), 1), dtype=np.float64)
                    w_vals = (np.hstack([ref_xy, ones]) @ H.astype(np.float64).T)[:, 2]
                    if float(np.median(w_vals)) < 0.0:
                        w_vals = -w_vals
                    w_vals = np.maximum(w_vals, 1e-3)
                    # Add gentle smooth radial UV dome warp distinct from flat planar
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
            tex_rgba = self._apply_local_occlusion_mask(frame_bgr, H) if self._occ_enabled else self._placed_rgba

        if mode == "mesh_6dof":
            tex_rgba = self._placed_rgba

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
        comp_ms = (time.perf_counter() - t0) * 1000.0
        return out_bgr, res.mask, comp_ms
