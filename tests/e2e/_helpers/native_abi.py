"""Opaque-box ctypes bindings and independent 3D ray-triangle oracle for `libvisualizeit_pose.so`.

Strictly stdlib-only at module import time; `numpy` is imported lazily inside helper calls.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Any

from .paths_and_proc import SO_PATH

VI_OK: int = 0
VI_INVALID_ARGUMENT: int = 1

BLEND_NORMAL: int = 0
BLEND_LUMINANCE: int = 1
BLEND_INK: int = 2

REQUIRED_NATIVE_SYMBOLS = (
    "vi_pose_abi_version",
    "vi_cv_gltf_pose_to_unity",
    "vi_rasterize_mesh_rgbd",
)


def load_native_abi(so_path: Path | None = None) -> ctypes.CDLL:
    """Load `libvisualizeit_pose.so` and configure exact `extern "C"` argtypes/restype."""
    target = (so_path or SO_PATH).resolve()
    assert target.is_file(), f"Missing native shared library: {target}"
    try:
        lib = ctypes.CDLL(str(target))
    except OSError as exc:
        raise AssertionError(f"Failed to load native shared library {target}: {exc}") from exc

    for sym in REQUIRED_NATIVE_SYMBOLS:
        if not hasattr(lib, sym):
            raise AssertionError(f"Missing required extern 'C' symbol {sym!r} in {target}")

    # pose.h ABI
    lib.vi_pose_abi_version.argtypes = []
    lib.vi_pose_abi_version.restype = ctypes.c_uint

    lib.vi_cv_gltf_pose_to_unity.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # const float input[16] (nullable for negative tests)
        ctypes.POINTER(ctypes.c_float),  # float output[16] (nullable for negative tests)
    ]
    lib.vi_cv_gltf_pose_to_unity.restype = ctypes.c_int

    # raster.h ABI
    lib.vi_rasterize_mesh_rgbd.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # vertices_cam (V * 3)
        ctypes.c_int,                     # num_vertices
        ctypes.POINTER(ctypes.c_int32),   # triangles (T * 3)
        ctypes.c_int,                     # num_triangles
        ctypes.POINTER(ctypes.c_float),   # vertex_uvs (V * 2)
        ctypes.POINTER(ctypes.c_uint8),   # texture_rgba (tex_h * tex_w * 4)
        ctypes.c_int,                     # tex_w
        ctypes.c_int,                     # tex_h
        ctypes.c_int,                     # wrap_repeat (1 = repeat, 0 = clamp/decal)
        ctypes.c_float,                   # fx
        ctypes.c_float,                   # fy
        ctypes.c_float,                   # cx
        ctypes.c_float,                   # cy
        ctypes.c_float,                   # opacity
        ctypes.c_int,                     # blend_mode (0=normal, 1=luminance, 2=ink)
        ctypes.POINTER(ctypes.c_uint8),   # frame_rgb_inout (frame_h * frame_w * 3)
        ctypes.POINTER(ctypes.c_float),   # depth_buf_inout (frame_h * frame_w)
        ctypes.POINTER(ctypes.c_uint8),   # mask_out (frame_h * frame_w) or NULL
        ctypes.c_int,                     # frame_w
        ctypes.c_int,                     # frame_h
    ]
    lib.vi_rasterize_mesh_rgbd.restype = ctypes.c_int
    return lib


def call_pose_to_unity(
    mat4: Any,
    *,
    lib: ctypes.CDLL | None = None,
    in_place: bool = False,
) -> tuple[int, Any]:
    """Invoke `vi_cv_gltf_pose_to_unity` on a 4x4 row-major float32 matrix and return `(rc, out_4x4)`."""
    import numpy as np

    cdll = lib or load_native_abi()
    arr_in = np.ascontiguousarray(mat4, dtype=np.float32).reshape(4, 4)
    if in_place:
        arr_out = arr_in.copy()
        ptr = arr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = int(cdll.vi_cv_gltf_pose_to_unity(ptr, ptr))
    else:
        arr_out = np.zeros((4, 4), dtype=np.float32)
        in_ptr = arr_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = arr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = int(cdll.vi_cv_gltf_pose_to_unity(in_ptr, out_ptr))
    return rc, arr_out


def call_rasterize_mesh_rgbd(
    vertices_cam: Any,
    triangles: Any,
    vertex_uvs: Any,
    texture_rgba: Any,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    frame_rgb_inout: Any | None = None,
    depth_buf_inout: Any | None = None,
    mask_out: Any | bool | None = None,
    wrap_repeat: int = 0,
    opacity: float = 1.0,
    blend_mode: int = BLEND_NORMAL,
    frame_w: int | None = None,
    frame_h: int | None = None,
    lib: ctypes.CDLL | None = None,
) -> dict[str, Any]:
    """High-level numpy wrapper around `vi_rasterize_mesh_rgbd`.

    - Mutates `frame_rgb_inout`, `depth_buf_inout`, and `mask_out` in-place if passed as
      C-contiguous arrays of exact matching dtype (`uint8`, `float32`, `uint8`).
    - When `mask_out=None`, passes a C `NULL` pointer to `vi_rasterize_mesh_rgbd`.
    - When `mask_out=True`, allocates a fresh `(frame_h, frame_w)` `uint8` mask buffer.
    - Returns `{"rc": int, "frame_rgb": ndarray, "depth_buf": ndarray, "mask": ndarray | None}`.
    """
    import numpy as np

    cdll = lib or load_native_abi()

    verts = np.ascontiguousarray(vertices_cam, dtype=np.float32).reshape(-1, 3)
    tris = np.ascontiguousarray(triangles, dtype=np.int32).reshape(-1, 3)
    uvs = np.ascontiguousarray(vertex_uvs, dtype=np.float32).reshape(-1, 2)
    tex = np.ascontiguousarray(texture_rgba, dtype=np.uint8)
    assert tex.ndim == 3 and tex.shape[2] == 4, f"texture_rgba must have shape (H, W, 4), got {tex.shape}"
    tex_h, tex_w = int(tex.shape[0]), int(tex.shape[1])

    if frame_rgb_inout is not None:
        if (
            isinstance(frame_rgb_inout, np.ndarray)
            and frame_rgb_inout.dtype == np.uint8
            and frame_rgb_inout.flags["C_CONTIGUOUS"]
        ):
            frame_buf = frame_rgb_inout
        else:
            frame_buf = np.ascontiguousarray(frame_rgb_inout, dtype=np.uint8)
        assert frame_buf.ndim == 3 and frame_buf.shape[2] == 3, (
            f"frame_rgb_inout must have shape (H, W, 3), got {frame_buf.shape}"
        )
        fh, fw = int(frame_buf.shape[0]), int(frame_buf.shape[1])
    else:
        fw = int(frame_w if frame_w is not None else 640)
        fh = int(frame_h if frame_h is not None else 480)
        frame_buf = np.zeros((fh, fw, 3), dtype=np.uint8)

    eff_w = int(frame_w if frame_w is not None else fw)
    eff_h = int(frame_h if frame_h is not None else fh)

    if depth_buf_inout is not None:
        if (
            isinstance(depth_buf_inout, np.ndarray)
            and depth_buf_inout.dtype == np.float32
            and depth_buf_inout.flags["C_CONTIGUOUS"]
        ):
            depth_buf = depth_buf_inout
        else:
            depth_buf = np.ascontiguousarray(depth_buf_inout, dtype=np.float32)
    else:
        depth_buf = np.full((fh, fw), np.inf, dtype=np.float32)

    mask_buf: Any | None
    mask_ptr: Any
    if mask_out is None or mask_out is False:
        mask_buf = None
        mask_ptr = None
    elif mask_out is True:
        mask_buf = np.zeros((fh, fw), dtype=np.uint8)
        mask_ptr = mask_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    else:
        if (
            isinstance(mask_out, np.ndarray)
            and mask_out.dtype == np.uint8
            and mask_out.flags["C_CONTIGUOUS"]
        ):
            mask_buf = mask_out
        else:
            mask_buf = np.ascontiguousarray(mask_out, dtype=np.uint8)
        mask_ptr = mask_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    rc = int(
        cdll.vi_rasterize_mesh_rgbd(
            verts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(verts.shape[0]),
            tris.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            int(tris.shape[0]),
            uvs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            tex.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            tex_w,
            tex_h,
            int(wrap_repeat),
            float(fx),
            float(fy),
            float(cx),
            float(cy),
            float(opacity),
            int(blend_mode),
            frame_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            depth_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mask_ptr,
            eff_w,
            eff_h,
        )
    )
    return {
        "rc": rc,
        "frame_rgb": frame_buf,
        "depth_buf": depth_buf,
        "mask": mask_buf,
    }


def make_planar_quad_mesh(
    *,
    width_m: float = 0.4,
    height_m: float = 0.3,
    z_m: float = 1.0,
    tilt_rad: float = 0.0,
    center_xy: tuple[float, float] = (0.0, 0.0),
    uv_scale: float = 1.0,
) -> tuple[Any, Any, Any]:
    """Create a 2-triangle planar quad in OpenCV camera coordinates (+X right, +Y down, +Z forward).

    `tilt_rad` rotates the quad around the camera Y-axis so left/right edges sit at distinct depths
    `z_m +/- (width_m/2)*sin(tilt_rad)` — ideal for testing perspective-correct `1/z` UV interpolation.
    Triangles have outward normals facing the camera (`-Z`).
    """
    import numpy as np

    hw = 0.5 * float(width_m)
    hh = 0.5 * float(height_m)
    cx_m, cy_m = float(center_xy[0]), float(center_xy[1])
    cos_t = math.cos(float(tilt_rad))
    sin_t = math.sin(float(tilt_rad))

    local_xy = [
        (-hw, -hh),  # v0: top-left
        (+hw, -hh),  # v1: top-right
        (+hw, +hh),  # v2: bottom-right
        (-hw, +hh),  # v3: bottom-left
    ]
    verts = np.array(
        [
            [cx_m + lx * cos_t, cy_m + ly, float(z_m) + lx * sin_t]
            for lx, ly in local_xy
        ],
        dtype=np.float32,
    )
    tris = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    s = float(uv_scale)
    uvs = np.array(
        [
            [0.0, 0.0],
            [s, 0.0],
            [s, s],
            [0.0, s],
        ],
        dtype=np.float32,
    )
    return verts, tris, uvs


def make_uv_cylinder_mesh(
    *,
    radius_m: float = 0.12,
    height_m: float = 0.28,
    z_center_m: float = 0.85,
    num_sectors: int = 32,
    num_stacks: int = 4,
    arc_rad: float = math.pi,
) -> tuple[Any, Any, Any]:
    """Create a UV-mapped cylindrical surface mesh centered at `(0, 0, z_center_m)` with axis along Y."""
    import numpy as np

    assert num_sectors >= 3 and num_stacks >= 1
    verts_list: list[list[float]] = []
    uvs_list: list[list[float]] = []
    tris_list: list[list[int]] = []

    half_h = 0.5 * float(height_m)
    start_theta = -0.5 * float(arc_rad)
    for row in range(num_stacks + 1):
        v_coord = row / float(num_stacks)
        y_pos = -half_h + v_coord * float(height_m)
        for col in range(num_sectors + 1):
            u_coord = col / float(num_sectors)
            theta = start_theta + u_coord * float(arc_rad)
            x_pos = float(radius_m) * math.sin(theta)
            z_pos = float(z_center_m) - float(radius_m) * math.cos(theta)
            verts_list.append([x_pos, y_pos, z_pos])
            uvs_list.append([u_coord, v_coord])

    stride = num_sectors + 1
    for row in range(num_stacks):
        for col in range(num_sectors):
            tl = row * stride + col
            tr = tl + 1
            bl = (row + 1) * stride + col
            br = bl + 1
            tris_list.append([tl, br, tr])
            tris_list.append([tl, bl, br])

    return (
        np.asarray(verts_list, dtype=np.float32),
        np.asarray(tris_list, dtype=np.int32),
        np.asarray(uvs_list, dtype=np.float32),
    )


def ray_triangle_uv_oracle(
    vertices_cam: Any,
    triangles: Any,
    vertex_uvs: Any,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    frame_w: int,
    frame_h: int,
    texture_rgba: Any | None = None,
    wrap_repeat: int = 0,
    opacity: float = 1.0,
    background_rgb: Any | None = None,
    pixel_offset: float = 0.5,
) -> dict[str, Any]:
    """Independent 3D Moller-Trumbore ray-triangle oracle with perspective-correct `1/z` UV interpolation.

    Intersecting pinhole camera rays `O + z * ((px + pixel_offset - cx)/fx, (py + pixel_offset - cy)/fy, 1)`
    directly against 3D camera-space triangles yields exact 3D barycentric weights `(w0, w1, w2)` that are
    algebraically identical to screen-space `1/z` perspective-correct barycentric weights:
    `w_i = (lambda_i / z_i) / sum_j(lambda_j / z_j)`.

    Returns:
      - `hit_mask`: `(frame_h, frame_w)` bool ndarray
      - `depth`: `(frame_h, frame_w)` float32 ndarray (`+inf` where unhit)
      - `uv`: `(frame_h, frame_w, 2)` float32 ndarray (`NaN` where unhit)
      - `rgb`: `(frame_h, frame_w, 3)` uint8 ndarray if `texture_rgba` is provided, else `None`
    """
    import numpy as np

    verts = np.asarray(vertices_cam, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    uvs = np.asarray(vertex_uvs, dtype=np.float64).reshape(-1, 2)

    xs = (np.arange(frame_w, dtype=np.float64) + float(pixel_offset) - float(cx)) / float(fx)
    ys = (np.arange(frame_h, dtype=np.float64) + float(pixel_offset) - float(cy)) / float(fy)
    grid_x, grid_y = np.meshgrid(xs, ys)
    dirs = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)  # (H, W, 3), D.z == 1.0

    best_z = np.full((frame_h, frame_w), np.inf, dtype=np.float64)
    best_uv = np.full((frame_h, frame_w, 2), np.nan, dtype=np.float64)
    hit_mask = np.zeros((frame_h, frame_w), dtype=bool)
    eps = 1e-9

    for tri in tris:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        if v0[2] <= 1e-6 and v1[2] <= 1e-6 and v2[2] <= 1e-6:
            continue
        e1 = v1 - v0
        e2 = v2 - v0
        # Ray origin O = (0, 0, 0), so T_vec = O - v0 = -v0
        pvec = np.cross(dirs, e2)  # (H, W, 3)
        det = np.einsum("ijk,k->ij", pvec, e1)  # (H, W)
        valid_det = np.abs(det) > eps
        inv_det = np.where(valid_det, 1.0 / np.where(valid_det, det, 1.0), 0.0)

        tvec = -v0  # (3,)
        u_bary = np.einsum("k,ijk->ij", tvec, pvec) * inv_det
        qvec = np.cross(tvec, e1)  # (3,)
        v_bary = np.einsum("ijk,k->ij", dirs, qvec) * inv_det
        t_ray = float(np.dot(e2, qvec)) * inv_det  # Because D.z == 1, t_ray is camera-space depth Z!

        w0 = 1.0 - u_bary - v_bary
        inside = (
            valid_det
            & (w0 >= -1e-6)
            & (u_bary >= -1e-6)
            & (v_bary >= -1e-6)
            & (t_ray > 1e-5)
            & (t_ray < best_z)
        )
        if not np.any(inside):
            continue

        uv_hit = (
            w0[..., None] * uvs[i0]
            + u_bary[..., None] * uvs[i1]
            + v_bary[..., None] * uvs[i2]
        )
        best_z = np.where(inside, t_ray, best_z)
        best_uv = np.where(inside[..., None], uv_hit, best_uv)
        hit_mask |= inside

    out_rgb: Any | None = None
    if texture_rgba is not None:
        tex = np.asarray(texture_rgba, dtype=np.float64)
        th, tw = tex.shape[0], tex.shape[1]
        if background_rgb is not None:
            bg = np.asarray(background_rgb, dtype=np.float64).copy()
            if bg.ndim == 1:
                bg = np.broadcast_to(bg.reshape(1, 1, 3), (frame_h, frame_w, 3)).copy()
        else:
            bg = np.zeros((frame_h, frame_w, 3), dtype=np.float64)

        sample_mask = hit_mask.copy()
        u_coord = np.where(sample_mask, best_uv[..., 0], 0.0)
        v_coord = np.where(sample_mask, best_uv[..., 1], 0.0)
        if wrap_repeat:
            u_coord = u_coord - np.floor(u_coord)
            v_coord = v_coord - np.floor(v_coord)
        else:
            in_decal = (u_coord >= 0.0) & (u_coord <= 1.0) & (v_coord >= 0.0) & (v_coord <= 1.0)
            sample_mask &= in_decal
            u_coord = np.clip(u_coord, 0.0, 1.0)
            v_coord = np.clip(v_coord, 0.0, 1.0)

        fx_tex = np.clip(u_coord * (tw - 1), 0.0, max(0, tw - 1))
        fy_tex = np.clip(v_coord * (th - 1), 0.0, max(0, th - 1))
        x0 = np.floor(fx_tex).astype(np.int32)
        y0 = np.floor(fy_tex).astype(np.int32)
        x1 = np.minimum(x0 + 1, tw - 1)
        y1 = np.minimum(y0 + 1, th - 1)
        wx = (fx_tex - x0)[..., None]
        wy = (fy_tex - y0)[..., None]

        c00 = tex[y0, x0]
        c10 = tex[y0, x1]
        c01 = tex[y1, x0]
        c11 = tex[y1, x1]
        rgba = (
            (1.0 - wx) * (1.0 - wy) * c00
            + wx * (1.0 - wy) * c10
            + (1.0 - wx) * wy * c01
            + wx * wy * c11
        )
        eff_alpha = np.where(sample_mask, (rgba[..., 3] / 255.0) * float(opacity), 0.0)[..., None]
        comp = eff_alpha * rgba[..., :3] + (1.0 - eff_alpha) * bg
        out_rgb = np.clip(np.rint(comp), 0.0, 255.0).astype(np.uint8)

    return {
        "hit_mask": hit_mask,
        "depth": best_z.astype(np.float32),
        "uv": best_uv.astype(np.float32),
        "rgb": out_rgb,
    }
