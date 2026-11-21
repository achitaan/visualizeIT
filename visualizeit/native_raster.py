"""Python ctypes binding and fast vectorized OpenCV/NumPy fallback for libvisualizeit_pose.so."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import threading
from typing import Literal, Sequence

import cv2
import numpy as np


REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_BUILD_DIR: Path = REPO_ROOT / ".cache" / "native-build"
DEFAULT_LIB_PATH: Path = DEFAULT_BUILD_DIR / "libvisualizeit_pose.so"

BLEND_MODE_CODES: dict[str, int] = {
    "normal": 0,
    "luminance": 1,
    "luminance_shading": 1,
    "ink": 2,
}

_LOAD_LOCK = threading.Lock()
_CACHED_LIB: ctypes.CDLL | None = None
_CACHED_LIB_PATH: Path | None = None


@dataclass(slots=True)
class RasterizeResult:
    rendered_pixels: int
    frame_rgb: np.ndarray
    depth_buf: np.ndarray
    mask: np.ndarray
    backend_used: Literal["native_cpp", "opencv"]


def get_native_lib_path() -> Path:
    override = os.environ.get("VISUALIZEIT_NATIVE_LIB")
    if override:
        return Path(override)
    return DEFAULT_LIB_PATH


def _configure_cdll_signatures(lib: ctypes.CDLL) -> ctypes.CDLL:
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int32_p = ctypes.POINTER(ctypes.c_int32)
    c_uint8_p = ctypes.POINTER(ctypes.c_uint8)

    lib.vi_cv_gltf_pose_to_unity.argtypes = [c_float_p, c_float_p]
    lib.vi_cv_gltf_pose_to_unity.restype = ctypes.c_int

    lib.vi_pose_abi_version.argtypes = []
    lib.vi_pose_abi_version.restype = ctypes.c_uint

    lib.vi_rasterize_mesh_rgbd.argtypes = [
        c_float_p,       # vertices_cam
        ctypes.c_int,    # num_vertices
        c_int32_p,       # triangles
        ctypes.c_int,    # num_triangles
        c_float_p,       # vertex_uvs
        c_uint8_p,       # texture_rgba
        ctypes.c_int,    # tex_w
        ctypes.c_int,    # tex_h
        ctypes.c_int,    # wrap_repeat
        ctypes.c_float,  # fx
        ctypes.c_float,  # fy
        ctypes.c_float,  # cx
        ctypes.c_float,  # cy
        ctypes.c_float,  # opacity
        ctypes.c_int,    # blend_mode
        c_uint8_p,       # frame_rgb_inout
        c_float_p,       # depth_buf_inout
        c_uint8_p,       # mask_out
        ctypes.c_int,    # frame_w
        ctypes.c_int,    # frame_h
    ]
    lib.vi_rasterize_mesh_rgbd.restype = ctypes.c_int
    return lib


def load_native_library(*, auto_build: bool = True) -> ctypes.CDLL | None:
    """Load libvisualizeit_pose.so via ctypes, optionally building via CMake if missing."""
    global _CACHED_LIB, _CACHED_LIB_PATH
    lib_path = get_native_lib_path()

    with _LOAD_LOCK:
        if _CACHED_LIB is not None and _CACHED_LIB_PATH == lib_path and lib_path.is_file():
            return _CACHED_LIB

        if not lib_path.is_file() and auto_build and lib_path == DEFAULT_LIB_PATH:
            tmp_dir = REPO_ROOT / ".cache" / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            DEFAULT_BUILD_DIR.mkdir(parents=True, exist_ok=True)
            env = {
                **os.environ,
                "TMPDIR": str(tmp_dir),
                "TEMP": str(tmp_dir),
                "TMP": str(tmp_dir),
            }
            try:
                subprocess.run(
                    [
                        "cmake",
                        "-S",
                        str(REPO_ROOT / "native"),
                        "-B",
                        str(DEFAULT_BUILD_DIR),
                        "-DCMAKE_BUILD_TYPE=Release",
                        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={DEFAULT_BUILD_DIR}",
                    ],
                    check=True,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                subprocess.run(
                    [
                        "cmake",
                        "--build",
                        str(DEFAULT_BUILD_DIR),
                        "--target",
                        "visualizeit_pose",
                        "--parallel",
                        "2",
                    ],
                    check=True,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except (OSError, subprocess.CalledProcessError):
                return None

        if not lib_path.is_file():
            return None

        try:
            loaded = _configure_cdll_signatures(ctypes.CDLL(str(lib_path)))
        except OSError:
            return None

        _CACHED_LIB = loaded
        _CACHED_LIB_PATH = lib_path
        return loaded


def is_native_raster_available(*, auto_build: bool = True) -> bool:
    """Return True if the compiled C++17 native rasterizer shared library is available."""
    return load_native_library(auto_build=auto_build) is not None


def cv_gltf_pose_to_unity(pose_4x4: np.ndarray) -> np.ndarray:
    """Convert an OpenCV/glTF 4x4 rigid pose matrix to Unity coordinates via native ABI."""
    arr = np.asarray(pose_4x4, dtype=np.float32)
    if arr.shape != (4, 4) or not np.all(np.isfinite(arr)):
        raise ValueError("pose_4x4 must be a finite (4, 4) matrix")
    in_buf = np.ascontiguousarray(arr, dtype=np.float32)
    out_buf = np.empty((4, 4), dtype=np.float32)

    lib = load_native_library(auto_build=True)
    if lib is None:
        raise RuntimeError("Native library libvisualizeit_pose.so is unavailable")

    c_float_p = ctypes.POINTER(ctypes.c_float)
    rc = lib.vi_cv_gltf_pose_to_unity(
        in_buf.ctypes.data_as(c_float_p),
        out_buf.ctypes.data_as(c_float_p),
    )
    if rc != 0:
        raise ValueError(f"vi_cv_gltf_pose_to_unity rejected non-rigid/invalid pose (status={rc})")
    return out_buf


def _resolve_blend_mode(blend_mode: int | str | np.integer) -> int:
    if isinstance(blend_mode, (bool, np.bool_)):
        raise ValueError("blend_mode cannot be a bool")
    if isinstance(blend_mode, (int, np.integer)):
        code = int(blend_mode)
        if code not in (0, 1, 2):
            raise ValueError(f"Invalid integer blend_mode {blend_mode}; expected 0, 1, or 2")
        return code
    if isinstance(blend_mode, str) and blend_mode in BLEND_MODE_CODES:
        return BLEND_MODE_CODES[blend_mode]
    raise ValueError(f"Unsupported blend_mode {blend_mode!r}")


def _clip_poly_near(
    poly: list[tuple[float, float, float, float, float]],
    near_z: float,
) -> list[tuple[float, float, float, float, float]]:
    n = len(poly)
    if n < 3:
        return []
    out: list[tuple[float, float, float, float, float]] = []
    px, py, pz, pu, pv = poly[-1]
    d_prev = pz - near_z
    in_prev = d_prev >= 0.0
    for cx_v, cy_v, cz_v, cu_v, cv_v in poly:
        d_curr = cz_v - near_z
        in_curr = d_curr >= 0.0
        if in_prev != in_curr:
            denom = d_prev - d_curr
            t = 0.0
            if denom != 0.0:
                raw_t = d_prev / denom
                if math.isfinite(raw_t):
                    t = max(0.0, min(1.0, raw_t))
            out.append(
                (
                    px + t * (cx_v - px),
                    py + t * (cy_v - py),
                    near_z,
                    pu + t * (cu_v - pu),
                    pv + t * (cv_v - pv),
                )
            )
        if in_curr:
            out.append((cx_v, cy_v, cz_v, cu_v, cv_v))
        px, py, pz, pu, pv = cx_v, cy_v, cz_v, cu_v, cv_v
        d_prev = d_curr
        in_prev = in_curr
    return out


def _clip_poly_homogeneous(
    poly: list[tuple[float, float, float, float, float]],
    nx: float,
    ny: float,
    nz: float,
    near_z: float,
) -> list[tuple[float, float, float, float, float]]:
    n = len(poly)
    if n < 3:
        return []
    out: list[tuple[float, float, float, float, float]] = []
    px, py, pz, pu, pv = poly[-1]
    d_prev = nx * px + ny * py + nz * pz
    in_prev = d_prev >= 0.0
    for cx_v, cy_v, cz_v, cu_v, cv_v in poly:
        d_curr = nx * cx_v + ny * cy_v + nz * cz_v
        in_curr = d_curr >= 0.0
        if in_prev != in_curr:
            denom = d_prev - d_curr
            t = 0.0
            if denom != 0.0:
                raw_t = d_prev / denom
                if math.isfinite(raw_t):
                    t = max(0.0, min(1.0, raw_t))
            out.append(
                (
                    px + t * (cx_v - px),
                    py + t * (cy_v - py),
                    max(near_z, pz + t * (cz_v - pz)),
                    pu + t * (cu_v - pu),
                    pv + t * (cv_v - pv),
                )
            )
        if in_curr:
            out.append((cx_v, cy_v, cz_v, cu_v, cv_v))
        px, py, pz, pu, pv = cx_v, cy_v, cz_v, cu_v, cv_v
        d_prev = d_curr
        in_prev = in_curr
    return out


def _clip_triangle_frustum_5plane(
    tri_xyz: np.ndarray,
    tri_uv: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    frame_w: int,
    frame_h: int,
    near_z: float = 1e-4,
    guard_margin_px: float = 2.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    poly: list[tuple[float, float, float, float, float]] = [
        (
            float(tri_xyz[k, 0]),
            float(tri_xyz[k, 1]),
            float(tri_xyz[k, 2]),
            float(tri_uv[k, 0]),
            float(tri_uv[k, 1]),
        )
        for k in range(3)
    ]
    poly = _clip_poly_near(poly, near_z)
    if len(poly) < 3:
        return []
    poly = _clip_poly_homogeneous(poly, fx, 0.0, cx + guard_margin_px, near_z)
    if len(poly) < 3:
        return []
    poly = _clip_poly_homogeneous(poly, -fx, 0.0, float(frame_w) + guard_margin_px - cx, near_z)
    if len(poly) < 3:
        return []
    poly = _clip_poly_homogeneous(poly, 0.0, fy, cy + guard_margin_px, near_z)
    if len(poly) < 3:
        return []
    poly = _clip_poly_homogeneous(poly, 0.0, -fy, float(frame_h) + guard_margin_px - cy, near_z)
    if len(poly) < 3:
        return []

    v0_xyz = np.array(poly[0][:3], dtype=np.float64)
    v0_uv = np.array(poly[0][3:], dtype=np.float64)
    res: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, len(poly) - 1):
        vi_xyz = np.array(poly[i][:3], dtype=np.float64)
        vi_uv = np.array(poly[i][3:], dtype=np.float64)
        vj_xyz = np.array(poly[i + 1][:3], dtype=np.float64)
        vj_uv = np.array(poly[i + 1][3:], dtype=np.float64)
        res.append(
            (
                np.stack([v0_xyz, vi_xyz, vj_xyz], axis=0),
                np.stack([v0_uv, vi_uv, vj_uv], axis=0),
            )
        )
    return res


def _rasterize_mesh_opencv(
    verts: np.ndarray,
    tris: np.ndarray,
    uvs: np.ndarray,
    tex_rgba: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    wrap_code: int,
    opacity: float,
    blend_code: int,
    frame_rgb: np.ndarray,
    depth_buf: np.ndarray,
    mask_out: np.ndarray | None,
) -> int:
    frame_h, frame_w = frame_rgb.shape[:2]
    tex_h, tex_w = tex_rgba.shape[:2]
    near_z = 1e-4
    guard_margin_px = 2.0

    tri_xyz = verts[tris]   # (T, 3, 3)
    tri_uv = uvs[tris]      # (T, 3, 2)
    z_vals = tri_xyz[:, :, 2]

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        q_est = (1.0 / np.maximum(z_vals, near_z)).astype(np.float32)
        uv_q_ok = np.all(np.isfinite(tri_uv[:, :, 0] * q_est) & np.isfinite(tri_uv[:, :, 1] * q_est), axis=1)

    all_behind = np.all(z_vals < near_z, axis=1) | (~uv_q_ok)
    all_front = np.all(z_vals >= near_z, axis=1) & uv_q_ok

    fast_mask = np.zeros(len(tris), dtype=bool)
    if np.any(all_front):
        front_xyz = tri_xyz[all_front]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            qf = 1.0 / front_xyz[:, :, 2]
            pxf = fx * front_xyz[:, :, 0] * qf + cx
            pyf = fy * front_xyz[:, :, 1] * qf + cy
        in_fast = np.all(
            (pxf >= -guard_margin_px)
            & (pxf <= float(frame_w) + guard_margin_px)
            & (pyf >= -guard_margin_px)
            & (pyf <= float(frame_h) + guard_margin_px),
            axis=1,
        )
        fast_mask[all_front] = in_fast

    clip_indices = np.flatnonzero(~all_behind & ~fast_mask)

    xyz_parts: list[np.ndarray] = []
    uv_parts: list[np.ndarray] = []
    if np.any(fast_mask):
        xyz_parts.append(tri_xyz[fast_mask].astype(np.float64, copy=False))
        uv_parts.append(tri_uv[fast_mask].astype(np.float64, copy=False))

    for idx in clip_indices:
        for c_xyz, c_uv in _clip_triangle_frustum_5plane(
            tri_xyz[idx], tri_uv[idx], fx, fy, cx, cy, frame_w, frame_h, near_z, guard_margin_px
        ):
            xyz_parts.append(c_xyz[None, :, :])
            uv_parts.append(c_uv[None, :, :])

    if not xyz_parts:
        return 0

    all_xyz64 = np.concatenate(xyz_parts, axis=0)
    all_uv = np.concatenate(uv_parts, axis=0).astype(np.float32, copy=False)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        q64 = 1.0 / all_xyz64[:, :, 2]
        px = (fx * all_xyz64[:, :, 0] * q64 + cx).astype(np.float32)
        py = (fy * all_xyz64[:, :, 1] * q64 + cy).astype(np.float32)
        q = q64.astype(np.float32)

    valid_guard = (
        np.all(np.isfinite(px), axis=1)
        & np.all(np.isfinite(py), axis=1)
        & np.all(np.isfinite(q), axis=1)
        & np.all(np.abs(px) <= 1e6, axis=1)
        & np.all(np.abs(py) <= 1e6, axis=1)
    )
    if not np.any(valid_guard):
        return 0

    px = px[valid_guard].copy()
    py = py[valid_guard].copy()
    q = q[valid_guard].copy()
    all_uv = all_uv[valid_guard].copy()

    with np.errstate(over="ignore", invalid="ignore"):
        area = (px[:, 1] - px[:, 0]) * (py[:, 2] - py[:, 0]) - (py[:, 1] - py[:, 0]) * (px[:, 2] - px[:, 0])
    non_degen = np.isfinite(area) & (np.abs(area) >= 1e-8)
    if not np.any(non_degen):
        return 0

    px = px[non_degen]
    py = py[non_degen]
    q = q[non_degen]
    all_uv = all_uv[non_degen]
    area = area[non_degen]

    flip = area < 0.0
    if np.any(flip):
        px[flip, 1], px[flip, 2] = px[flip, 2].copy(), px[flip, 1].copy()
        py[flip, 1], py[flip, 2] = py[flip, 2].copy(), py[flip, 1].copy()
        q[flip, 1], q[flip, 2] = q[flip, 2].copy(), q[flip, 1].copy()
        all_uv[flip, 1], all_uv[flip, 2] = all_uv[flip, 2].copy(), all_uv[flip, 1].copy()
        area[flip] = -area[flip]

    bx0_arr = np.maximum(0, np.floor(np.min(px, axis=1) - 0.5).astype(np.int32))
    bx1_arr = np.minimum(frame_w - 1, np.ceil(np.max(px, axis=1) - 0.5).astype(np.int32))
    by0_arr = np.maximum(0, np.floor(np.min(py, axis=1) - 0.5).astype(np.int32))
    by1_arr = np.minimum(frame_h - 1, np.ceil(np.max(py, axis=1) - 0.5).astype(np.int32))

    in_screen = (bx0_arr <= bx1_arr) & (by0_arr <= by1_arr)
    if not np.any(in_screen):
        return 0

    px = px[in_screen]
    py = py[in_screen]
    q = q[in_screen]
    all_uv = all_uv[in_screen]
    area = area[in_screen]
    bx0_arr = bx0_arr[in_screen]
    bx1_arr = bx1_arr[in_screen]
    by0_arr = by0_arr[in_screen]
    by1_arr = by1_arr[in_screen]

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inv_area = 1.0 / area
        dl0_dx = (py[:, 1] - py[:, 2]) * inv_area
        dl1_dx = (py[:, 2] - py[:, 0]) * inv_area
        dl2_dx = -dl0_dx - dl1_dx
        uq = all_uv[:, :, 0] * q
        vq = all_uv[:, :, 1] * q
        dq_dx = dl0_dx * q[:, 0] + dl1_dx * q[:, 1] + dl2_dx * q[:, 2]
        duq_dx = dl0_dx * uq[:, 0] + dl1_dx * uq[:, 1] + dl2_dx * uq[:, 2]
        dvq_dx = dl0_dx * vq[:, 0] + dl1_dx * vq[:, 1] + dl2_dx * vq[:, 2]

    valid_uv_grad = (
        np.all(np.isfinite(uq), axis=1)
        & np.all(np.isfinite(vq), axis=1)
        & np.isfinite(inv_area)
        & np.isfinite(dq_dx)
        & np.isfinite(duq_dx)
        & np.isfinite(dvq_dx)
    )
    if not np.any(valid_uv_grad):
        return 0

    px = px[valid_uv_grad]
    py = py[valid_uv_grad]
    q = q[valid_uv_grad]
    uq = uq[valid_uv_grad]
    vq = vq[valid_uv_grad]
    area = area[valid_uv_grad]
    inv_area = inv_area[valid_uv_grad]
    bx0_arr = bx0_arr[valid_uv_grad]
    bx1_arr = bx1_arr[valid_uv_grad]
    by0_arr = by0_arr[valid_uv_grad]
    by1_arr = by1_arr[valid_uv_grad]

    roi_x0 = int(np.min(bx0_arr))
    roi_x1 = int(np.max(bx1_arr))
    roi_y0 = int(np.min(by0_arr))
    roi_y1 = int(np.max(by1_arr))
    roi_w = roi_x1 - roi_x0 + 1
    roi_h = roi_y1 - roi_y0 + 1

    hit_roi = np.zeros((roi_h, roi_w), dtype=bool)
    u_roi = np.zeros((roi_h, roi_w), dtype=np.float32)
    v_roi = np.zeros((roi_h, roi_w), dtype=np.float32)

    xs_all = np.arange(frame_w, dtype=np.float32) + 0.5
    ys_all = np.arange(frame_h, dtype=np.float32) + 0.5

    for idx in range(len(area)):
        bx0 = int(bx0_arr[idx])
        bx1 = int(bx1_arr[idx])
        by0 = int(by0_arr[idx])
        by1 = int(by1_arr[idx])

        sx = xs_all[bx0 : bx1 + 1][None, :]
        sy = ys_all[by0 : by1 + 1][:, None]

        x0, x1, x2 = float(px[idx, 0]), float(px[idx, 1]), float(px[idx, 2])
        y0, y1, y2 = float(py[idx, 0]), float(py[idx, 1]), float(py[idx, 2])
        eps0 = 1e-4 * max(abs(x2 - x1), abs(y2 - y1))
        eps1 = 1e-4 * max(abs(x0 - x2), abs(y0 - y2))
        eps2 = 1e-4 * max(abs(x1 - x0), abs(y1 - y0))

        e0 = (x2 - x1) * (sy - y1) - (y2 - y1) * (sx - x1)
        e1 = (x0 - x2) * (sy - y2) - (y0 - y2) * (sx - x2)
        e2 = (x1 - x0) * (sy - y0) - (y1 - y0) * (sx - x0)

        inside = (e0 >= -eps0) & (e1 >= -eps1) & (e2 >= -eps2)
        if not np.any(inside):
            continue

        ia = float(inv_area[idx])
        l0 = e0 * ia
        l1 = e1 * ia
        l2 = 1.0 - l0 - l1

        q_pix = l0 * q[idx, 0] + l1 * q[idx, 1] + l2 * q[idx, 2]
        valid_q = inside & (q_pix > 1e-12)
        if not np.any(valid_q):
            continue

        z_pix = np.where(valid_q, 1.0 / np.maximum(q_pix, 1e-12), np.inf).astype(np.float32)
        d_slice = depth_buf[by0 : by1 + 1, bx0 : bx1 + 1]
        vis = valid_q & (z_pix < d_slice)
        if not np.any(vis):
            continue

        d_slice[vis] = z_pix[vis]
        with np.errstate(over="ignore", invalid="ignore"):
            u_pix = (l0 * uq[idx, 0] + l1 * uq[idx, 1] + l2 * uq[idx, 2]) * z_pix
            v_pix = (l0 * vq[idx, 0] + l1 * vq[idx, 1] + l2 * vq[idx, 2]) * z_pix

        ry0, ry1 = by0 - roi_y0, by1 - roi_y0 + 1
        rx0, rx1 = bx0 - roi_x0, bx1 - roi_x0 + 1
        hit_roi[ry0:ry1, rx0:rx1][vis] = True
        u_roi[ry0:ry1, rx0:rx1][vis] = u_pix[vis]
        v_roi[ry0:ry1, rx0:rx1][vis] = v_pix[vis]

    if opacity <= (0.5 / 255.0) or not np.any(hit_roi):
        return 0

    finite_uv = hit_roi & np.isfinite(u_roi) & np.isfinite(v_roi)
    if not np.any(finite_uv):
        return 0

    u_safe = np.where(finite_uv, u_roi, 0.0)
    v_safe = np.where(finite_uv, v_roi, 0.0)

    if wrap_code == 1:
        active = finite_uv
        with np.errstate(over="ignore", invalid="ignore"):
            us = u_safe - np.floor(u_safe)
            vs = v_safe - np.floor(v_safe)
        active = active & np.isfinite(us) & np.isfinite(vs)
        if not np.any(active):
            return 0
        us = np.where(active, np.where(us >= 1.0, us - 1.0, np.maximum(us, 0.0)), 0.0)
        vs = np.where(active, np.where(vs >= 1.0, vs - 1.0, np.maximum(vs, 0.0)), 0.0)
    else:
        in_decal = (
            finite_uv
            & (u_safe >= -1e-4)
            & (u_safe <= 1.0001)
            & (v_safe >= -1e-4)
            & (v_safe <= 1.0001)
        )
        active = in_decal
        if not np.any(active):
            return 0
        us = np.clip(u_safe, 0.0, 1.0)
        vs = np.clip(v_safe, 0.0, 1.0)

    map_x = (us * float(tex_w - 1)).astype(np.float32)
    map_y = (vs * float(tex_h - 1)).astype(np.float32)
    sampled = cv2.remap(
        tex_rgba,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)

    alpha = (sampled[:, :, 3] * (1.0 / 255.0)) * float(opacity)
    comp_mask = active & (alpha > (0.5 / 255.0))
    rendered_count = int(np.count_nonzero(comp_mask))
    if rendered_count == 0:
        return 0

    dst_roi = frame_rgb[roi_y0 : roi_y1 + 1, roi_x0 : roi_x1 + 1]
    dst_f = dst_roi.astype(np.float32)
    src_rgb = sampled[:, :, :3]

    if blend_code == 0:
        blended = src_rgb
    elif blend_code == 1:
        hit_rgb = dst_f[hit_roi]
        lum_hit = (77.0 * hit_rgb[:, 0] + 150.0 * hit_rgb[:, 1] + 29.0 * hit_rgb[:, 2]) / 256.0
        l_mean = float(np.mean(lum_hit))
        l_base = max(48.0, min(220.0, 0.65 * l_mean + 0.35 * 160.0))
        l_px = (77.0 * dst_f[:, :, 0] + 150.0 * dst_f[:, :, 1] + 29.0 * dst_f[:, :, 2]) / 256.0
        ratio = np.clip(l_px / l_base, 0.0, 1.65)[:, :, None]
        gain_lo = 0.05 + 0.95 * ratio
        hi = np.maximum(0.0, ratio - 1.0)
        gain_hi = 1.0 + 0.50 * hi
        sheen = 0.20 * hi
        blended = np.where(
            ratio <= 1.0,
            src_rgb * gain_lo,
            src_rgb * gain_hi + sheen * (255.0 - src_rgb),
        )
    else:
        blended = dst_f * src_rgb * (0.92 / 255.0)

    a3 = alpha[:, :, None]
    composited = np.clip(a3 * blended + (1.0 - a3) * dst_f + 0.5, 0.0, 255.0).astype(np.uint8)
    dst_roi[comp_mask] = composited[comp_mask]
    if mask_out is not None:
        mask_out[roi_y0 : roi_y1 + 1, roi_x0 : roi_x1 + 1][comp_mask] = 255
    return rendered_count


def rasterize_mesh_rgbd(
    vertices_cam: np.ndarray,
    triangles: np.ndarray,
    vertex_uvs: np.ndarray,
    texture_rgba: np.ndarray,
    intrinsics: Sequence[float] | np.ndarray,
    frame_rgb: np.ndarray,
    *,
    depth_buf: np.ndarray | None = None,
    mask_out: np.ndarray | None = None,
    wrap_repeat: bool | int = False,
    opacity: float = 1.0,
    blend_mode: int | str = "luminance",
    backend: Literal["auto", "native_cpp", "opencv"] = "auto",
) -> RasterizeResult:
    """Rasterize and composite a 3D UV-mapped triangle mesh onto an RGB frame."""
    if backend not in ("auto", "native_cpp", "opencv"):
        raise ValueError(f"Invalid raster backend {backend!r}; expected 'auto', 'native_cpp', or 'opencv'")

    verts_arr = np.asarray(vertices_cam)
    if verts_arr.ndim != 2 or verts_arr.shape[1] != 3 or verts_arr.shape[0] < 1:
        raise ValueError(f"vertices_cam must have shape (V, 3) with V >= 1, got {verts_arr.shape}")
    verts_c = np.ascontiguousarray(verts_arr, dtype=np.float32)
    if not np.all(np.isfinite(verts_c)):
        raise ValueError("vertices_cam contains NaN or Inf values")

    num_v = int(verts_c.shape[0])

    tris_arr = np.asarray(triangles)
    if tris_arr.ndim != 2 or tris_arr.shape[1] != 3 or tris_arr.shape[0] < 1:
        raise ValueError(f"triangles must have shape (T, 3) with T >= 1, got {tris_arr.shape}")
    if tris_arr.dtype.kind not in ("i", "u"):
        raise ValueError(f"triangles must have integer dtype, got {tris_arr.dtype}")
    if int(np.min(tris_arr)) < 0 or int(np.max(tris_arr)) >= num_v:
        raise ValueError(f"triangle vertex index out of range [0, {num_v})")
    tris_c = np.ascontiguousarray(tris_arr, dtype=np.int32)
    num_t = int(tris_c.shape[0])

    uv_arr = np.asarray(vertex_uvs)
    if uv_arr.shape != (num_v, 2):
        raise ValueError(f"vertex_uvs must have shape ({num_v}, 2), got {uv_arr.shape}")
    uvs_c = np.ascontiguousarray(uv_arr, dtype=np.float32)
    if not np.all(np.isfinite(uvs_c)):
        raise ValueError("vertex_uvs contains NaN or Inf values")

    if not isinstance(texture_rgba, np.ndarray) or texture_rgba.dtype != np.uint8:
        raise ValueError("texture_rgba must be a uint8 numpy ndarray")
    if texture_rgba.ndim != 3 or texture_rgba.shape[2] != 4 or texture_rgba.shape[0] < 1 or texture_rgba.shape[1] < 1:
        raise ValueError(f"texture_rgba must have shape (H, W, 4), got {texture_rgba.shape}")
    tex_c = np.ascontiguousarray(texture_rgba, dtype=np.uint8)
    tex_h, tex_w = int(tex_c.shape[0]), int(tex_c.shape[1])

    intr_raw = np.asarray(intrinsics)
    if np.issubdtype(intr_raw.dtype, np.bool_) or intr_raw.dtype.kind not in ("i", "u", "f"):
        raise ValueError("intrinsics must be 4 finite numbers (fx, fy, cx, cy)")
    intr_arr = intr_raw.astype(np.float64, copy=False).ravel()
    if intr_arr.size != 4 or not np.all(np.isfinite(intr_arr)):
        raise ValueError("intrinsics must be 4 finite numbers (fx, fy, cx, cy)")
    fx, fy, cx, cy = float(intr_arr[0]), float(intr_arr[1]), float(intr_arr[2]), float(intr_arr[3])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"fx and fy must be > 0, got fx={fx}, fy={fy}")

    if isinstance(opacity, (bool, np.bool_)) or not isinstance(opacity, (int, float, np.integer, np.floating)):
        raise ValueError("opacity must be a finite float in [0.0, 1.0]")
    opacity_f = float(opacity)
    if not math.isfinite(opacity_f) or opacity_f < 0.0 or opacity_f > 1.0:
        raise ValueError(f"opacity must be in [0.0, 1.0], got {opacity_f}")

    if isinstance(wrap_repeat, (bool, np.bool_)):
        wrap_code = 1 if bool(wrap_repeat) else 0
    elif isinstance(wrap_repeat, (int, np.integer)) and int(wrap_repeat) in (0, 1):
        wrap_code = int(wrap_repeat)
    else:
        raise ValueError(f"wrap_repeat must be bool or 0/1, got {wrap_repeat!r}")

    blend_code = _resolve_blend_mode(blend_mode)

    if not isinstance(frame_rgb, np.ndarray) or frame_rgb.dtype != np.uint8:
        raise ValueError("frame_rgb must be a uint8 numpy ndarray")
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3 or frame_rgb.shape[0] < 1 or frame_rgb.shape[1] < 1:
        raise ValueError(f"frame_rgb must have shape (H, W, 3), got {frame_rgb.shape}")
    if not frame_rgb.flags["WRITEABLE"]:
        raise ValueError("frame_rgb must be writeable")
    frame_h, frame_w = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])

    if depth_buf is not None:
        if not isinstance(depth_buf, np.ndarray) or depth_buf.dtype != np.float32 or depth_buf.shape != (frame_h, frame_w):
            raise ValueError(f"depth_buf must be float32 of shape ({frame_h}, {frame_w})")
        if not depth_buf.flags["WRITEABLE"]:
            raise ValueError("depth_buf must be writeable")

    if mask_out is not None:
        if not isinstance(mask_out, np.ndarray) or mask_out.dtype != np.uint8 or mask_out.shape != (frame_h, frame_w):
            raise ValueError(f"mask_out must be uint8 of shape ({frame_h}, {frame_w})")
        if not mask_out.flags["WRITEABLE"]:
            raise ValueError("mask_out must be writeable")

    frame_c = frame_rgb if frame_rgb.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame_rgb)

    if depth_buf is None:
        depth_c = np.full((frame_h, frame_w), np.inf, dtype=np.float32)
        depth_out = depth_c
    else:
        depth_c = depth_buf if depth_buf.flags["C_CONTIGUOUS"] else np.ascontiguousarray(depth_buf)
        depth_out = depth_buf

    if mask_out is None:
        mask_c = np.zeros((frame_h, frame_w), dtype=np.uint8)
        mask_ret = mask_c
    else:
        mask_c = mask_out if mask_out.flags["C_CONTIGUOUS"] else np.ascontiguousarray(mask_out)
        mask_ret = mask_out

    use_native = False
    lib: ctypes.CDLL | None = None
    if backend in ("auto", "native_cpp"):
        lib = load_native_library(auto_build=True)
        if lib is not None:
            use_native = True
        elif backend == "native_cpp":
            raise RuntimeError("Native C++ rasterizer library is unavailable")

    if use_native and lib is not None:
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_int32_p = ctypes.POINTER(ctypes.c_int32)
        c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
        rc = int(
            lib.vi_rasterize_mesh_rgbd(
                verts_c.ctypes.data_as(c_float_p),
                num_v,
                tris_c.ctypes.data_as(c_int32_p),
                num_t,
                uvs_c.ctypes.data_as(c_float_p),
                tex_c.ctypes.data_as(c_uint8_p),
                tex_w,
                tex_h,
                wrap_code,
                fx,
                fy,
                cx,
                cy,
                opacity_f,
                blend_code,
                frame_c.ctypes.data_as(c_uint8_p),
                depth_c.ctypes.data_as(c_float_p),
                mask_c.ctypes.data_as(c_uint8_p),
                frame_w,
                frame_h,
            )
        )
        if rc < 0:
            raise ValueError(f"vi_rasterize_mesh_rgbd failed with error code {rc}")
        rendered_px = rc
        used_backend: Literal["native_cpp", "opencv"] = "native_cpp"
    else:
        rendered_px = _rasterize_mesh_opencv(
            verts_c,
            tris_c,
            uvs_c,
            tex_c,
            fx,
            fy,
            cx,
            cy,
            wrap_code,
            opacity_f,
            blend_code,
            frame_c,
            depth_c,
            mask_c,
        )
        used_backend = "opencv"

    if frame_c is not frame_rgb:
        frame_rgb[:] = frame_c
    if depth_c is not depth_out:
        depth_out[:] = depth_c
    if mask_c is not mask_ret:
        mask_ret[:] = mask_c

    return RasterizeResult(
        rendered_pixels=rendered_px,
        frame_rgb=frame_rgb,
        depth_buf=depth_out,
        mask=mask_ret,
        backend_used=used_backend,
    )
