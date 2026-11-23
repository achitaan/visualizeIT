"""Deterministic multi-scenario demo video (.mp4) & 3D (.ply) corpus generator."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import cv2
import numpy as np

from .contracts import ContractValidationError, DemoClipInfo, SelectionSpec
from .native_raster import rasterize_mesh_rgbd


REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_DEMO_DIR: Path = REPO_ROOT / ".cache" / "demo-videos"

DEMO_CLIP_IDS: tuple[str, ...] = (
    "planar_card_640x480",
    "planar_hd_1280x720",
    "cylinder_mug_640x480",
    "occlusion_recovery_640x480",
    "mesh_box_640x480",
)

DEMO_FPS: float = 30.0
DEMO_FRAME_COUNT: int = 180


def _validate_output_dir(output_dir: str | os.PathLike[str]) -> Path:
    if output_dir is None:
        raise ValueError("output_dir must be a non-empty path")
    try:
        raw_fspath = os.fspath(output_dir)
    except TypeError as exc:
        raise ValueError("output_dir must be a string or PathLike") from exc
    if isinstance(raw_fspath, bytes):
        raw_str = raw_fspath.decode("utf-8", errors="surrogateescape")
    elif isinstance(raw_fspath, str):
        raw_str = raw_fspath
    else:
        raise ValueError("output_dir must be a string or PathLike")
    if not raw_str.strip():
        raise ValueError("output_dir must be a non-empty path")

    out = Path(os.path.abspath(raw_str))
    posix = out.as_posix()
    if posix == "/tmp" or posix.startswith("/tmp/"):
        raise ValueError(f"Refusing to place demo assets under /tmp: {out}")
    if ".agents/teamwork" in posix:
        raise ValueError(f"Refusing to place demo assets inside .agents/teamwork: {out}")
    if out.is_symlink() and not out.is_dir():
        raise ValueError(f"output_dir symlink does not point to a directory: {out}")
    for ancestor in (out, *out.parents):
        if ancestor.exists():
            if not ancestor.is_dir():
                raise ValueError(f"output_dir exists and is not a directory: {ancestor}")
            break
    return out


class _FFmpegH264Writer:
    """Streaming raw bgr24 -> libx264 yuv420p MP4 writer with atomic file replacement."""

    def __init__(self, target_path: Path, width: int, height: int, fps: float = DEMO_FPS) -> None:
        self.target_path = target_path
        self.temp_path = target_path.parent / f".pending-{os.getpid()}-{target_path.name}"
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._proc: subprocess.Popen[bytes] | None = None
        self._frames_written = 0

    def __enter__(self) -> _FFmpegH264Writer:
        tmp_dir = REPO_ROOT / ".cache" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "TMPDIR": str(tmp_dir),
            "TEMP": str(tmp_dir),
            "TMP": str(tmp_dir),
        }
        fps_str = str(int(self.fps)) if self.fps.is_integer() else f"{self.fps:.4f}"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            fps_str,
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-g",
            "15",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.temp_path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        return self

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("FFmpeg writer is not open")
        if frame_bgr.shape != (self.height, self.width, 3) or frame_bgr.dtype != np.uint8:
            raise ValueError(
                f"Expected ({self.height}, {self.width}, 3) uint8 BGR frame, got {frame_bgr.shape} {frame_bgr.dtype}"
            )
        buf = frame_bgr if frame_bgr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame_bgr)
        self._proc.stdin.write(memoryview(buf))
        self._frames_written += 1

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._proc is None:
            return
        stderr_bytes = b""
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            if self._proc.stderr is not None:
                stderr_bytes = self._proc.stderr.read()
            rc = self._proc.wait()
        finally:
            self._proc = None

        if exc_type is not None:
            self.temp_path.unlink(missing_ok=True)
            return

        if rc != 0:
            self.temp_path.unlink(missing_ok=True)
            err_msg = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed encoding {self.target_path.name} (rc={rc}): {err_msg}")

        os.replace(self.temp_path, self.target_path)


def _rot_x(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _build_studio_background(width: int, height: int, seed: int, pad: int = 12) -> np.ndarray:
    """Create a smooth, warm indoor studio background (padded for subtle camera breathing)."""
    rng = np.random.default_rng(seed)
    ph = height + 2 * pad
    pw = width + 2 * pad

    yy = np.linspace(0.0, 1.0, ph, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, pw, dtype=np.float32)[None, :]

    # Upper studio wall (soft warm plaster) transitioning to warm light-oak tabletop.
    horizon = 0.56
    table_t = np.clip((yy - (horizon - 0.06)) / 0.12, 0.0, 1.0)

    wall_b = 204.0 - 14.0 * yy
    wall_g = 212.0 - 12.0 * yy
    wall_r = 218.0 - 10.0 * yy

    wood_wave = np.sin(yy * 55.0 + 1.8 * np.sin(xx * 6.0)) * 4.0
    desk_b = 148.0 - 16.0 * (yy - horizon) + wood_wave * 0.6
    desk_g = 174.0 - 14.0 * (yy - horizon) + wood_wave * 0.8
    desk_r = 198.0 - 10.0 * (yy - horizon) + wood_wave * 1.0

    b = (1.0 - table_t) * wall_b + table_t * desk_b
    g = (1.0 - table_t) * wall_g + table_t * desk_g
    r = (1.0 - table_t) * wall_r + table_t * desk_r

    canvas = np.stack([b, g, r], axis=-1).astype(np.float32)

    # Soft out-of-focus background ambient panels so scene looks natural without sharp background corners
    for _ in range(4):
        rx0 = int(rng.integers(0, pw - pw // 4))
        ry0 = int(rng.integers(0, int(ph * 0.45)))
        rx1 = min(pw, rx0 + int(rng.integers(pw // 6, pw // 3)))
        ry1 = min(int(ph * 0.52), ry0 + int(rng.integers(ph // 8, ph // 4)))
        delta = rng.uniform(-10.0, 10.0, size=(1, 1, 3)).astype(np.float32)
        canvas[ry0:ry1, rx0:rx1] += delta

    # Soft radial studio vignette
    vignette = 1.0 - 0.14 * ((xx - 0.5) ** 2 + (yy - 0.48) ** 2) * 2.2
    canvas *= vignette[:, :, None]

    blurred = cv2.GaussianBlur(
        np.clip(canvas, 0.0, 255.0).astype(np.uint8),
        (31, 31),
        9.0,
    )
    return blurred


def _crop_breathing_bg(padded_bg: np.ndarray, width: int, height: int, frame_idx: int, pad: int = 12) -> np.ndarray:
    u = 2.0 * math.pi * float(frame_idx) / float(DEMO_FRAME_COUNT)
    max_ox = float(max(0, padded_bg.shape[1] - width))
    max_oy = float(max(0, padded_bg.shape[0] - height))
    ox = min(max_ox, max(0.0, float(pad) + 4.5 * math.sin(u)))
    oy = min(max_oy, max(0.0, float(pad) + 3.5 * math.cos(u)))
    center = (ox + (float(width) - 1.0) * 0.5, oy + (float(height) - 1.0) * 0.5)
    return cv2.getRectSubPix(padded_bg, (width, height), center)


def _build_feature_rich_poster(
    tex_w: int,
    tex_h: int,
    seed: int,
    title: str,
    subtitle: str,
    accent_bgr: tuple[int, int, int] = (54, 48, 210),
) -> np.ndarray:
    """Build an asymmetric, high-contrast BGR poster rich in distinct multi-scale ORB features."""
    rng = np.random.default_rng(seed)
    img = np.full((tex_h, tex_w, 3), (38, 46, 62), dtype=np.uint8)

    # Bold dark outer frame and inner bright trim for sharp GrabCut boundary contrast
    border = max(8, min(tex_w, tex_h) // 28)
    cv2.rectangle(img, (0, 0), (tex_w - 1, tex_h - 1), (22, 24, 30), thickness=border * 2)
    cv2.rectangle(
        img,
        (border, border),
        (tex_w - 1 - border, tex_h - 1 - border),
        (235, 240, 245),
        thickness=max(3, border // 2),
    )

    palette_bgr = [
        accent_bgr,
        (135, 152, 42),   # teal
        (80, 52, 29),     # deep navy
        (95, 162, 244),   # warm gold
        (110, 72, 185),   # crimson-plum
        (228, 234, 238),  # crisp off-white
    ]

    # Asymmetric color-block grid + nested geometric fiducials
    gx_n, gy_n = 6, 4
    ix0, iy0 = border * 2, border * 2
    iw = tex_w - 4 * border
    ih = tex_h - 4 * border
    cw = max(12, iw // gx_n)
    ch = max(12, ih // gy_n)

    for gy in range(gy_n):
        for gx in range(gx_n):
            x0 = ix0 + gx * cw
            y0 = iy0 + gy * ch
            x1 = ix0 + (gx + 1) * cw if gx + 1 < gx_n else ix0 + iw
            y1 = iy0 + (gy + 1) * ch if gy + 1 < gy_n else iy0 + ih
            base_c = palette_bgr[int(rng.integers(0, len(palette_bgr)))]
            fg_c = (20, 24, 30) if sum(base_c) > 380 else (242, 244, 246)
            cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), base_c, thickness=-1)
            cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (24, 28, 34), thickness=2)

            mx = (x0 + x1) // 2
            my = (y0 + y1) // 2
            rad = max(5, min(cw, ch) // 3)
            motif = int((gx * 3 + gy * 5 + seed) % 5)
            if motif == 0:
                cv2.circle(img, (mx, my), rad, fg_c, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(img, (mx + rad // 3, my - rad // 4), max(2, rad // 2), base_c, thickness=-1, lineType=cv2.LINE_AA)
            elif motif == 1:
                pts = np.array(
                    [[mx, y0 + 6], [x1 - 6, y1 - 6], [x0 + 6, y1 - 6]],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(img, pts, fg_c, lineType=cv2.LINE_AA)
            elif motif == 2:
                # 3x3 mini-checker for crisp FAST/ORB corners
                sub_w = max(3, (x1 - x0 - 12) // 3)
                sub_h = max(3, (y1 - y0 - 12) // 3)
                for sy in range(3):
                    for sx in range(3):
                        if (sx + sy) % 2 == 0:
                            sx0 = x0 + 6 + sx * sub_w
                            sy0 = y0 + 6 + sy * sub_h
                            cv2.rectangle(img, (sx0, sy0), (sx0 + sub_w, sy0 + sub_h), fg_c, thickness=-1)
            elif motif == 3:
                cv2.line(img, (x0 + 6, y0 + 6), (x1 - 6, y1 - 6), fg_c, thickness=3, lineType=cv2.LINE_AA)
                cv2.line(img, (x0 + 6, y1 - 6), (x1 - 6, y0 + 6), fg_c, thickness=3, lineType=cv2.LINE_AA)
                cv2.rectangle(img, (mx - rad // 2, my - rad // 2), (mx + rad // 2, my + rad // 2), base_c, thickness=-1)
            else:
                pts = np.array(
                    [[mx, y0 + 5], [x1 - 5, my], [mx, y1 - 5], [x0 + 5, my]],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(img, pts, fg_c, lineType=cv2.LINE_AA)
                cv2.circle(img, (mx, my), max(2, rad // 3), base_c, thickness=-1)

    # 4 distinct asymmetric corner fiducials so 0/90/180/270 orientation is completely unique
    f_sz = max(18, min(tex_w, tex_h) // 8)
    corners = [
        (ix0 + 4, iy0 + 4, (35, 35, 225), 1),
        (tex_w - ix0 - f_sz - 4, iy0 + 4, (40, 175, 55), 2),
        (tex_w - ix0 - f_sz - 4, tex_h - iy0 - f_sz - 4, (215, 115, 35), 3),
        (ix0 + 4, tex_h - iy0 - f_sz - 4, (175, 45, 185), 4),
    ]
    for cx0, cy0, col, rings in corners:
        cv2.rectangle(img, (cx0, cy0), (cx0 + f_sz, cy0 + f_sz), (18, 20, 24), thickness=-1)
        cv2.rectangle(img, (cx0 + 2, cy0 + 2), (cx0 + f_sz - 2, cy0 + f_sz - 2), col, thickness=2)
        mcx = cx0 + f_sz // 2
        mcy = cy0 + f_sz // 2
        for r_i in range(1, rings + 1):
            rr = max(2, (f_sz // 2 - 4) * r_i // rings)
            cv2.circle(img, (mcx, mcy), rr, (245, 245, 245), thickness=2, lineType=cv2.LINE_AA)

    # Central asymmetric badge + typography
    bw = int(tex_w * 0.58)
    bh = int(tex_h * 0.28)
    bx0 = (tex_w - bw) // 2
    by0 = (tex_h - bh) // 2
    cv2.rectangle(img, (bx0, by0), (bx0 + bw, by0 + bh), (24, 28, 38), thickness=-1)
    cv2.rectangle(img, (bx0 + 3, by0 + 3), (bx0 + bw - 3, by0 + bh - 3), (95, 180, 245), thickness=2)

    font_scale = max(0.55, min(tex_w, tex_h) / 360.0)
    thick = max(2, int(round(font_scale * 2.2)))
    cv2.putText(
        img,
        title,
        (bx0 + 14, by0 + int(bh * 0.52)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (248, 248, 248),
        thick,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        subtitle,
        (bx0 + 16, by0 + int(bh * 0.84)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.58,
        (110, 205, 250),
        max(1, thick - 1),
        cv2.LINE_AA,
    )
    return img


def _project_planar_quad(
    half_w: float,
    half_h: float,
    cx: float,
    cy: float,
    focal: float,
    dx: float,
    dy: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    z_scale: float,
) -> np.ndarray:
    corners_3d = np.array(
        [
            [-half_w, -half_h, 0.0],
            [half_w, -half_h, 0.0],
            [half_w, half_h, 0.0],
            [-half_w, half_h, 0.0],
        ],
        dtype=np.float64,
    )
    rot = _rot_z(roll_deg) @ _rot_x(pitch_deg) @ _rot_y(yaw_deg)
    rotated = corners_3d @ rot.T
    z_depth = focal * z_scale + rotated[:, 2]
    px = (rotated[:, 0] * focal / z_depth) + cx + dx
    py = (rotated[:, 1] * focal / z_depth) + cy + dy
    return np.stack([px, py], axis=-1).astype(np.float32)


def _composite_warped_poster_roi(
    frame_bgr: np.ndarray,
    poster_bgr: np.ndarray,
    dst_quad: np.ndarray,
    shade_lo: float,
    shade_hi: float,
) -> np.ndarray:
    """Fast ROI-only perspective warp with soft drop shadow and 2D directional shading."""
    h, w = frame_bgr.shape[:2]
    ph, pw = poster_bgr.shape[:2]

    # Soft drop shadow
    shadow_quad = (dst_quad + np.array([[8.0, 10.0]], dtype=np.float32)).astype(np.int32)
    sx0 = max(0, int(np.min(shadow_quad[:, 0])) - 14)
    sy0 = max(0, int(np.min(shadow_quad[:, 1])) - 14)
    sx1 = min(w, int(np.max(shadow_quad[:, 0])) + 15)
    sy1 = min(h, int(np.max(shadow_quad[:, 1])) + 15)
    if sx1 > sx0 and sy1 > sy0:
        s_mask = np.zeros((sy1 - sy0, sx1 - sx0), dtype=np.uint8)
        cv2.fillConvexPoly(s_mask, shadow_quad - np.array([[sx0, sy0]], dtype=np.int32), 95, lineType=cv2.LINE_AA)
        s_blur = cv2.GaussianBlur(s_mask, (15, 15), 5.0).astype(np.float32) * (1.0 / 255.0)
        roi_bg = frame_bgr[sy0:sy1, sx0:sx1].astype(np.float32)
        frame_bgr[sy0:sy1, sx0:sx1] = np.clip(roi_bg * (1.0 - s_blur[:, :, None]), 0.0, 255.0).astype(np.uint8)

    # Spatial lighting ramp across poster
    rx = np.linspace(shade_lo, shade_hi, pw, dtype=np.float32)[None, :]
    ry = np.linspace(1.04, 0.94, ph, dtype=np.float32)[:, None]
    lit_poster = np.clip(poster_bgr.astype(np.float32) * (rx * ry)[:, :, None], 0.0, 255.0).astype(np.uint8)

    bx0 = max(0, int(math.floor(float(np.min(dst_quad[:, 0])))) - 2)
    by0 = max(0, int(math.floor(float(np.min(dst_quad[:, 1])))) - 2)
    bx1 = min(w, int(math.ceil(float(np.max(dst_quad[:, 0])))) + 3)
    by1 = min(h, int(math.ceil(float(np.max(dst_quad[:, 1])))) + 3)

    mask_full = np.zeros((h, w), dtype=np.uint8)
    if bx1 <= bx0 or by1 <= by0:
        return mask_full

    src_quad = np.array(
        [[0.0, 0.0], [pw - 1.0, 0.0], [pw - 1.0, ph - 1.0], [0.0, ph - 1.0]],
        dtype=np.float32,
    )
    local_quad = dst_quad - np.array([[float(bx0), float(by0)]], dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(src_quad, local_quad)

    rw = bx1 - bx0
    rh = by1 - by0
    warped = cv2.warpPerspective(
        lit_poster,
        h_mat,
        (rw, rh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    alpha_poly = np.zeros((rh, rw), dtype=np.uint8)
    cv2.fillConvexPoly(
        alpha_poly,
        np.rint(local_quad * 16.0).astype(np.int32),
        255,
        lineType=cv2.LINE_AA,
        shift=4,
    )

    a_f = (alpha_poly.astype(np.float32) * (1.0 / 255.0))[:, :, None]
    dst_slice = frame_bgr[by0:by1, bx0:bx1].astype(np.float32)
    frame_bgr[by0:by1, bx0:bx1] = np.clip(
        a_f * warped.astype(np.float32) + (1.0 - a_f) * dst_slice + 0.5,
        0.0,
        255.0,
    ).astype(np.uint8)
    mask_full[by0:by1, bx0:bx1] = np.where(alpha_poly >= 128, 255, 0).astype(np.uint8)
    return mask_full


def _verify_frame0_selection(mask_0: np.ndarray, sel: SelectionSpec, clip_id: str) -> None:
    h, w = mask_0.shape[:2]
    sel.validate_for_frame(w, h, frame_count=DEMO_FRAME_COUNT)
    eroded = cv2.erode(mask_0, np.ones((7, 7), np.uint8), iterations=1)
    dilated = cv2.dilate(mask_0, np.ones((9, 9), np.uint8), iterations=1)

    for px, py in sel.positive_points:
        ix, iy = int(round(px)), int(round(py))
        if eroded[iy, ix] == 0:
            raise RuntimeError(f"[{clip_id}] positive_point ({px}, {py}) not safely inside foreground mask")
    for nx, ny in sel.negative_points:
        ix, iy = int(round(nx)), int(round(ny))
        if dilated[iy, ix] != 0:
            raise RuntimeError(f"[{clip_id}] negative_point ({nx}, {ny}) overlaps foreground mask")

    ys, xs = np.nonzero(mask_0 > 0)
    if xs.size == 0 or sel.bbox is None:
        raise RuntimeError(f"[{clip_id}] empty frame-0 foreground mask or missing bbox")
    bx, by, bw, bh = sel.bbox
    if not (bx <= float(np.min(xs)) and by <= float(np.min(ys)) and bx + bw >= float(np.max(xs)) and by + bh >= float(np.max(ys))):
        raise RuntimeError(f"[{clip_id}] default_selection.bbox {sel.bbox} does not enclose frame-0 object")


def _generate_planar_card(out_dir: Path) -> tuple[DemoClipInfo, dict[str, Any]]:
    clip_id = "planar_card_640x480"
    w, h = 640, 480
    mp4_path = out_dir / f"{clip_id}.mp4"
    bg_pad = _build_studio_background(w, h, seed=101)
    poster = _build_feature_rich_poster(520, 380, seed=202, title="STUDIO CARD", subtitle="PLANAR 8-DOF TRACK")

    quad_0: np.ndarray | None = None
    mask_0: np.ndarray | None = None
    frames_gt: list[dict[str, Any]] = []

    with _FFmpegH264Writer(mp4_path, w, h, DEMO_FPS) as writer:
        for t in range(DEMO_FRAME_COUNT):
            u = 2.0 * math.pi * float(t) / float(DEMO_FRAME_COUNT)
            dx = 52.0 * math.sin(u)
            dy = 34.0 * math.sin(2.0 * u)
            roll = 16.0 * math.sin(u)
            yaw = 18.0 * math.sin(u)
            pitch = 13.0 * math.sin(2.0 * u)
            z_scale = 1.0 - 0.14 * math.sin(u)

            quad = _project_planar_quad(130.0, 95.0, 320.0, 240.0, 620.0, dx, dy, yaw, pitch, roll, z_scale)
            if quad_0 is None:
                quad_0 = quad.copy()

            shade_lo = 0.84 - 0.10 * math.sin(u)
            shade_hi = 1.08 + 0.08 * math.sin(u)

            frame = _crop_breathing_bg(bg_pad, w, h, t)
            mask = _composite_warped_poster_roi(frame, poster, quad, shade_lo, shade_hi)
            if t == 0:
                mask_0 = mask
            writer.write(frame)

            h_0_to_t = cv2.getPerspectiveTransform(quad_0, quad)
            frames_gt.append(
                {
                    "frame_index": t,
                    "timestamp_s": round(t / DEMO_FPS, 6),
                    "visible": True,
                    "visibility_fraction": 1.0,
                    "gt_quad_xy": quad.tolist(),
                    "gt_homography_3x3_from_frame0": h_0_to_t.tolist(),
                }
            )

    selection = SelectionSpec(
        frame_index=0,
        positive_points=[(320.0, 240.0), (245.0, 190.0), (395.0, 290.0)],
        negative_points=[(70.0, 70.0), (570.0, 410.0)],
        bbox=(180.0, 135.0, 280.0, 210.0),
    )
    assert mask_0 is not None
    _verify_frame0_selection(mask_0, selection, clip_id)

    info = DemoClipInfo(
        id=clip_id,
        path=str(mp4_path),
        width=w,
        height=h,
        fps=DEMO_FPS,
        frame_count=DEMO_FRAME_COUNT,
        default_selection=selection,
        recommended_wrap="planar",
    )
    return info, {"clip_id": clip_id, "frames": frames_gt}


def _generate_planar_hd(out_dir: Path) -> tuple[DemoClipInfo, dict[str, Any]]:
    clip_id = "planar_hd_1280x720"
    w, h = 1280, 720
    mp4_path = out_dir / f"{clip_id}.mp4"
    bg_pad = _build_studio_background(w, h, seed=303)
    poster = _build_feature_rich_poster(
        780,
        510,
        seed=404,
        title="GALLERY HD",
        subtitle="1280X720 SURFACE",
        accent_bgr=(44, 92, 224),
    )

    quad_0: np.ndarray | None = None
    mask_0: np.ndarray | None = None
    frames_gt: list[dict[str, Any]] = []

    with _FFmpegH264Writer(mp4_path, w, h, DEMO_FPS) as writer:
        for t in range(DEMO_FRAME_COUNT):
            u = 2.0 * math.pi * float(t) / float(DEMO_FRAME_COUNT)
            dx = 110.0 * math.sin(u)
            dy = 55.0 * math.sin(2.0 * u)
            roll = -14.0 * math.sin(u)
            yaw = 20.0 * math.sin(u)
            pitch = -12.0 * math.sin(2.0 * u)
            z_scale = 1.0 + 0.14 * math.sin(u)

            quad = _project_planar_quad(260.0, 170.0, 640.0, 360.0, 1050.0, dx, dy, yaw, pitch, roll, z_scale)
            if quad_0 is None:
                quad_0 = quad.copy()

            shade_lo = 0.82 + 0.10 * math.sin(u)
            shade_hi = 1.10 - 0.08 * math.sin(u)

            frame = _crop_breathing_bg(bg_pad, w, h, t)
            mask = _composite_warped_poster_roi(frame, poster, quad, shade_lo, shade_hi)
            if t == 0:
                mask_0 = mask
            writer.write(frame)

            h_0_to_t = cv2.getPerspectiveTransform(quad_0, quad)
            frames_gt.append(
                {
                    "frame_index": t,
                    "timestamp_s": round(t / DEMO_FPS, 6),
                    "visible": True,
                    "visibility_fraction": 1.0,
                    "gt_quad_xy": quad.tolist(),
                    "gt_homography_3x3_from_frame0": h_0_to_t.tolist(),
                }
            )

    selection = SelectionSpec(
        frame_index=0,
        positive_points=[(640.0, 360.0), (470.0, 260.0), (810.0, 460.0)],
        negative_points=[(140.0, 110.0), (1140.0, 610.0)],
        bbox=(368.0, 178.0, 544.0, 364.0),
    )
    assert mask_0 is not None
    _verify_frame0_selection(mask_0, selection, clip_id)

    info = DemoClipInfo(
        id=clip_id,
        path=str(mp4_path),
        width=w,
        height=h,
        fps=DEMO_FPS,
        frame_count=DEMO_FRAME_COUNT,
        default_selection=selection,
        recommended_wrap="planar",
    )
    return info, {"clip_id": clip_id, "frames": frames_gt}


def _build_cylinder_mesh(
    radius: float = 0.074,
    height: float = 0.172,
    n_theta: int = 48,
    n_y: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build closed cylinder mesh (X right, Y down, Z forward; front face at -Z -> u=0.5)."""
    verts: list[list[float]] = []
    norms: list[list[float]] = []
    uvs: list[list[float]] = []
    tris: list[list[int]] = []

    half_h = height * 0.5
    # Side grid: phi in [-pi, +pi], where phi=0 points toward camera (-Z) -> u=0.5
    for iy in range(n_y + 1):
        vf = float(iy) / float(n_y)
        y = -half_h + vf * height
        # Map side body to v in [0.04, 0.96]
        uv_v = 0.04 + vf * 0.92
        for it in range(n_theta + 1):
            uf = float(it) / float(n_theta)
            phi = (uf - 0.5) * 2.0 * math.pi
            nx = math.sin(phi)
            nz = -math.cos(phi)
            verts.append([radius * nx, y, radius * nz])
            norms.append([nx, 0.0, nz])
            uvs.append([uf, uv_v])

    stride = n_theta + 1
    for iy in range(n_y):
        for it in range(n_theta):
            i00 = iy * stride + it
            i10 = i00 + 1
            i01 = i00 + stride
            i11 = i01 + 1
            tris.append([i00, i01, i11])
            tris.append([i00, i11, i10])

    # Top cap (y = -half_h, uv near top rim) and bottom cap (y = +half_h, uv near bottom rim)
    for cap_y, cap_ny, cap_uv_v, flip_w in [(-half_h, -1.0, 0.015, False), (half_h, 1.0, 0.985, True)]:
        c_idx = len(verts)
        verts.append([0.0, cap_y, 0.0])
        norms.append([0.0, cap_ny, 0.0])
        uvs.append([0.5, cap_uv_v])
        ring_start = len(verts)
        for it in range(n_theta):
            uf = float(it) / float(n_theta)
            phi = (uf - 0.5) * 2.0 * math.pi
            verts.append([radius * math.sin(phi), cap_y, -radius * math.cos(phi)])
            norms.append([0.0, cap_ny, 0.0])
            uvs.append([uf, cap_uv_v])
        for it in range(n_theta):
            a = ring_start + it
            b = ring_start + ((it + 1) % n_theta)
            if flip_w:
                tris.append([c_idx, b, a])
            else:
                tris.append([c_idx, a, b])

    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(norms, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
        np.asarray(tris, dtype=np.int32),
    )


def _generate_cylinder_mug(out_dir: Path) -> tuple[DemoClipInfo, dict[str, Any]]:
    clip_id = "cylinder_mug_640x480"
    w, h = 640, 480
    mp4_path = out_dir / f"{clip_id}.mp4"
    bg_pad = _build_studio_background(w, h, seed=505)

    # 1024x512 wraparound mug graphic texture (convert BGR -> RGBA for native_raster)
    mug_bgr = _build_feature_rich_poster(
        1024,
        512,
        seed=606,
        title="CERAMIC MUG",
        subtitle="360 CYLINDER WRAP",
        accent_bgr=(140, 62, 34),
    )
    mug_rgba = cv2.cvtColor(mug_bgr, cv2.COLOR_BGR2RGBA)

    verts_obj, _, uvs_obj, tris = _build_cylinder_mesh()
    intrinsics = (560.0, 560.0, 320.0, 240.0)

    mask_0: np.ndarray | None = None
    frames_gt: list[dict[str, Any]] = []

    with _FFmpegH264Writer(mp4_path, w, h, DEMO_FPS) as writer:
        for t in range(DEMO_FRAME_COUNT):
            u = 2.0 * math.pi * float(t) / float(DEMO_FRAME_COUNT)
            yaw = 24.0 * math.sin(u)
            pitch = 7.0 * math.sin(2.0 * u)
            roll = 8.0 * math.sin(u)
            tx = 0.065 * math.sin(u)
            ty = 0.028 * math.sin(2.0 * u)
            tz = 0.40 - 0.035 * math.sin(u)

            rot = _rot_z(roll) @ _rot_x(pitch) @ _rot_y(yaw)
            tvec = np.array([tx, ty, tz], dtype=np.float64)
            rvec, _ = cv2.Rodrigues(rot)

            verts_cam = (verts_obj.astype(np.float64) @ rot.T + tvec[None, :]).astype(np.float32)

            bg_bgr = _crop_breathing_bg(bg_pad, w, h, t)
            frame_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
            res = rasterize_mesh_rgbd(
                verts_cam,
                tris,
                uvs_obj,
                mug_rgba,
                intrinsics,
                frame_rgb,
                wrap_repeat=True,
                opacity=1.0,
                blend_mode="normal",
                backend="auto",
            )
            mask = res.mask
            if t == 0:
                mask_0 = mask.copy()

            # Apply 3D cylindrical directional shading & soft ceramic sheen across rendered mug pixels
            ys, xs = np.nonzero(mask > 0)
            if xs.size > 0:
                x0, x1 = int(np.min(xs)), int(np.max(xs))
                span_w = max(1.0, float(x1 - x0))
                rel = (xs.astype(np.float32) - float(x0)) / span_w * 2.0 - 1.0
                # Off-center directional light + soft specular strip
                cyl_cos = np.sqrt(np.clip(1.0 - rel * rel, 0.05, 1.0))
                light_dir_bias = -0.32 * rel + 0.94 * cyl_cos
                spec = 22.0 * np.exp(-((rel + 0.28) / 0.16) ** 2)
                gain = np.clip(0.58 + 0.48 * np.maximum(0.0, light_dir_bias), 0.52, 1.12)
                px_rgb = frame_rgb[ys, xs].astype(np.float32)
                frame_rgb[ys, xs] = np.clip(px_rgb * gain[:, None] + spec[:, None], 0.0, 255.0).astype(np.uint8)

            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            frames_gt.append(
                {
                    "frame_index": t,
                    "timestamp_s": round(t / DEMO_FPS, 6),
                    "visible": True,
                    "visibility_fraction": 1.0,
                    "gt_rvec": rvec.ravel().tolist(),
                    "gt_tvec": tvec.tolist(),
                    "gt_R_3x3": rot.tolist(),
                }
            )

    assert mask_0 is not None
    ys0, xs0 = np.nonzero(mask_0 > 0)
    bx0 = max(0.0, float(int(np.min(xs0)) - 8))
    by0 = max(0.0, float(int(np.min(ys0)) - 8))
    bx1 = min(float(w), float(int(np.max(xs0)) + 9))
    by1 = min(float(h), float(int(np.max(ys0)) + 9))

    selection = SelectionSpec(
        frame_index=0,
        positive_points=[(320.0, 240.0), (285.0, 190.0), (355.0, 290.0)],
        negative_points=[(90.0, 240.0), (550.0, 240.0)],
        bbox=(bx0, by0, bx1 - bx0, by1 - by0),
    )
    _verify_frame0_selection(mask_0, selection, clip_id)

    info = DemoClipInfo(
        id=clip_id,
        path=str(mp4_path),
        width=w,
        height=h,
        fps=DEMO_FPS,
        frame_count=DEMO_FRAME_COUNT,
        default_selection=selection,
        recommended_wrap="cylinder",
    )
    return info, {"clip_id": clip_id, "intrinsics": list(intrinsics), "frames": frames_gt}


def _generate_occlusion_recovery(out_dir: Path) -> tuple[DemoClipInfo, dict[str, Any]]:
    clip_id = "occlusion_recovery_640x480"
    w, h = 640, 480
    mp4_path = out_dir / f"{clip_id}.mp4"
    bg_pad = _build_studio_background(w, h, seed=707)
    poster = _build_feature_rich_poster(
        500,
        360,
        seed=808,
        title="RE-ACQUIRE",
        subtitle="OCCLUSION TEST",
        accent_bgr=(48, 138, 52),
    )

    quad_0: np.ndarray | None = None
    mask_0: np.ndarray | None = None
    frames_gt: list[dict[str, Any]] = []

    with _FFmpegH264Writer(mp4_path, w, h, DEMO_FPS) as writer:
        for t in range(DEMO_FRAME_COUNT):
            u = 2.0 * math.pi * float(t) / float(DEMO_FRAME_COUNT)
            dx = 36.0 * math.sin(u)
            dy = 22.0 * math.sin(2.0 * u)
            roll = 10.0 * math.sin(u)
            yaw = 12.0 * math.sin(u)
            pitch = 8.0 * math.sin(2.0 * u)

            quad = _project_planar_quad(125.0, 90.0, 320.0, 240.0, 620.0, dx, dy, yaw, pitch, roll, 1.0)
            if quad_0 is None:
                quad_0 = quad.copy()

            frame = _crop_breathing_bg(bg_pad, w, h, t)
            card_mask = _composite_warped_poster_roi(frame, poster, quad, 0.88, 1.08)
            if t == 0:
                mask_0 = card_mask.copy()

            total_card_px = max(1, int(np.count_nonzero(card_mask)))
            occ_mask = np.zeros((h, w), dtype=np.uint8)

            # Exact schedule:
            # 0..69:   100% visible (68..69 occluder enters far right margin with 0 card overlap)
            # 70..109: 0% visible (100% occluded by smooth featureless dark slate foreground panel)
            # 110..179: 100% visible (110..111 occluder clears far left margin with 0 card overlap)
            card_max_x = int(math.ceil(float(np.max(quad[:, 0]))))
            card_min_x = int(math.floor(float(np.min(quad[:, 0]))))

            occ_rect: tuple[int, int, int, int] | None = None
            if t in (68, 69):
                ox0 = max(card_max_x + 26, 565 if t == 68 else 515)
                if ox0 < w - 4:
                    occ_rect = (ox0, 26, w, 454)
            elif 70 <= t <= 109:
                # Continuous monotonic right-to-left sweep (-2 px/frame) covering card + drop shadow 100%
                step = t - 70
                progress = float(step) / 39.0
                ox0 = max(0, min(135 - 2 * step, card_min_x - 24))
                ox1 = min(w, max(ox0 + 460, card_max_x + 28))
                oy0 = 24 + int(round(6.0 * math.sin(math.pi * progress)))
                oy1 = min(h, oy0 + 428)
                occ_rect = (ox0, oy0, ox1, oy1)
            elif t in (110, 111):
                ox1 = min(card_min_x - 22, 125 if t == 110 else 68)
                if ox1 > 4:
                    occ_rect = (0, 26, ox1, 454)

            if occ_rect is not None:
                ox0, oy0, ox1, oy1 = occ_rect
                occ_mask[oy0:oy1, ox0:ox1] = 255
                # Smooth C^inf low-frequency studio lamp sheen on featureless dark slate panel
                # (zero corners / 0 ORB keypoints inside occluder, continuous temporal motion every frame)
                xg = np.arange(ox1 - ox0, dtype=np.float32)[None, :]
                yg = np.arange(oy1 - oy0, dtype=np.float32)[:, None]
                phase = 0.14 * float(t - 68)
                wave1 = 16.0 * np.cos((xg * 0.022) + (yg * 0.008) - phase)
                wave2 = 9.0 * np.sin((yg * 0.019) - (xg * 0.006) + 0.85 * phase)
                panel_b = np.clip(58.0 + wave1 + 0.6 * wave2, 24.0, 96.0)
                panel_g = np.clip(52.0 + 0.92 * wave1 + 0.7 * wave2, 20.0, 90.0)
                panel_r = np.clip(46.0 + 0.82 * wave1 + 0.8 * wave2, 18.0, 84.0)
                panel = np.stack([panel_b, panel_g, panel_r], axis=-1).astype(np.uint8)
                frame[oy0:oy1, ox0:ox1] = panel

            rem_px = int(np.count_nonzero((card_mask > 0) & (occ_mask == 0)))
            vis_frac = round(float(rem_px) / float(total_card_px), 6)
            if t <= 69 or t >= 110:
                assert vis_frac == 1.0, f"Unexpected occlusion on visible frame {t}: {vis_frac}"
            else:
                assert vis_frac == 0.0, f"Card not 100% occluded on frame {t}: {vis_frac}"

            writer.write(frame)

            h_0_to_t = cv2.getPerspectiveTransform(quad_0, quad)
            frames_gt.append(
                {
                    "frame_index": t,
                    "timestamp_s": round(t / DEMO_FPS, 6),
                    "visible": vis_frac > 0.0,
                    "visibility_fraction": vis_frac,
                    "gt_quad_xy": quad.tolist(),
                    "gt_homography_3x3_from_frame0": h_0_to_t.tolist(),
                }
            )

    selection = SelectionSpec(
        frame_index=0,
        positive_points=[(320.0, 240.0), (245.0, 195.0), (395.0, 285.0)],
        negative_points=[(80.0, 75.0), (560.0, 405.0)],
        bbox=(185.0, 140.0, 270.0, 200.0),
    )
    assert mask_0 is not None
    _verify_frame0_selection(mask_0, selection, clip_id)

    info = DemoClipInfo(
        id=clip_id,
        path=str(mp4_path),
        width=w,
        height=h,
        fps=DEMO_FPS,
        frame_count=DEMO_FRAME_COUNT,
        default_selection=selection,
        recommended_wrap="planar",
    )
    return info, {"clip_id": clip_id, "frames": frames_gt}


def _build_box_mesh_and_atlas(
    hx: float = 0.095,
    hy: float = 0.072,
    hz: float = 0.062,
    subdiv: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build subdivided 6-face outward-wound 3D box (metres) + 3x2 high-feature RGBA atlas."""
    # Each tuple: (face_name, outward_normal, top_left_3d, right_vec_3d, down_vec_3d, accent_bgr)
    face_defs = [
        (
            "FRONT",
            np.array([0.0, 0.0, -1.0]),
            np.array([-hx, -hy, -hz]),
            np.array([2.0 * hx, 0.0, 0.0]),
            np.array([0.0, 2.0 * hy, 0.0]),
            (52, 58, 212),
        ),
        (
            "RIGHT",
            np.array([1.0, 0.0, 0.0]),
            np.array([hx, -hy, -hz]),
            np.array([0.0, 0.0, 2.0 * hz]),
            np.array([0.0, 2.0 * hy, 0.0]),
            (142, 152, 38),
        ),
        (
            "TOP",
            np.array([0.0, -1.0, 0.0]),
            np.array([-hx, -hy, hz]),
            np.array([2.0 * hx, 0.0, 0.0]),
            np.array([0.0, 0.0, -2.0 * hz]),
            (72, 148, 228),
        ),
        (
            "LEFT",
            np.array([-1.0, 0.0, 0.0]),
            np.array([-hx, -hy, hz]),
            np.array([0.0, 0.0, -2.0 * hz]),
            np.array([0.0, 2.0 * hy, 0.0]),
            (156, 68, 142),
        ),
        (
            "BOTTOM",
            np.array([0.0, 1.0, 0.0]),
            np.array([-hx, hy, -hz]),
            np.array([2.0 * hx, 0.0, 0.0]),
            np.array([0.0, 0.0, 2.0 * hz]),
            (88, 128, 64),
        ),
        (
            "BACK",
            np.array([0.0, 0.0, 1.0]),
            np.array([hx, -hy, hz]),
            np.array([-2.0 * hx, 0.0, 0.0]),
            np.array([0.0, 2.0 * hy, 0.0]),
            (118, 78, 44),
        ),
    ]

    tile_w, tile_h = 384, 384
    atlas_bgr = np.zeros((2 * tile_h, 3 * tile_w, 3), dtype=np.uint8)

    verts: list[np.ndarray] = []
    norms: list[np.ndarray] = []
    uvs: list[list[float]] = []
    vcols: list[list[int]] = []
    tris: list[list[int]] = []
    face_normals = np.zeros((6, 3), dtype=np.float64)

    stride = subdiv + 1
    for f_idx, (fname, normal, p_tl, vec_u, vec_v, accent) in enumerate(face_defs):
        face_normals[f_idx] = normal
        cx_cell = f_idx % 3
        cy_cell = f_idx // 3
        tile = _build_feature_rich_poster(
            tile_w,
            tile_h,
            seed=910 + f_idx * 37,
            title=f"BOX {fname}",
            subtitle=f"FACE {f_idx + 1} 6DOF",
            accent_bgr=accent,
        )
        atlas_bgr[cy_cell * tile_h : (cy_cell + 1) * tile_h, cx_cell * tile_w : (cx_cell + 1) * tile_w] = tile

        base_v = len(verts)
        for iy in range(stride):
            fv = float(iy) / float(subdiv)
            uv_v = (float(cy_cell) + 0.02 + 0.96 * fv) / 2.0
            for ix in range(stride):
                fu = float(ix) / float(subdiv)
                uv_u = (float(cx_cell) + 0.02 + 0.96 * fu) / 3.0
                pos = p_tl + fu * vec_u + fv * vec_v
                verts.append(pos)
                norms.append(normal)
                uvs.append([uv_u, uv_v])

                tx = min(3 * tile_w - 1, max(0, int(round(uv_u * (3 * tile_w - 1)))))
                ty = min(2 * tile_h - 1, max(0, int(round(uv_v * (2 * tile_h - 1)))))
                b_c, g_c, r_c = atlas_bgr[ty, tx]
                vcols.append([int(r_c), int(g_c), int(b_c)])

        for iy in range(subdiv):
            for ix in range(subdiv):
                i00 = base_v + iy * stride + ix
                i10 = i00 + 1
                i01 = i00 + stride
                i11 = i01 + 1
                # (p01 - p00) x (p11 - p00) = vec_v x vec_u = +outward_normal
                tris.append([i00, i01, i11])
                tris.append([i00, i11, i10])

    atlas_rgba = cv2.cvtColor(atlas_bgr, cv2.COLOR_BGR2RGBA)
    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(norms, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
        np.asarray(vcols, dtype=np.uint8),
        np.asarray(tris, dtype=np.int32),
        atlas_rgba,
        face_normals,
    )


def _write_ascii_ply(
    ply_path: Path,
    verts: np.ndarray,
    norms: np.ndarray,
    uvs: np.ndarray,
    vcols_rgb: np.ndarray,
    tris: np.ndarray,
) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        "comment visualizeIT metric 3D box mesh (units: meters, OpenCV object coordinates)",
        f"element vertex {len(verts)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float texture_u",
        "property float texture_v",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(tris)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for i in range(len(verts)):
        vx, vy, vz = verts[i]
        nx, ny, nz = norms[i]
        tu, tv = uvs[i]
        rc, gc, bc = vcols_rgb[i]
        lines.append(
            f"{vx:.6f} {vy:.6f} {vz:.6f} {nx:.5f} {ny:.5f} {nz:.5f} "
            f"{tu:.6f} {tv:.6f} {int(rc)} {int(gc)} {int(bc)}"
        )
    for f0, f1, f2 in tris:
        lines.append(f"3 {int(f0)} {int(f1)} {int(f2)}")
    tmp_ply = ply_path.parent / f".pending-{os.getpid()}-{ply_path.name}"
    tmp_ply.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp_ply, ply_path)


def _generate_mesh_box(out_dir: Path) -> tuple[DemoClipInfo, dict[str, Any]]:
    clip_id = "mesh_box_640x480"
    w, h = 640, 480
    mp4_path = out_dir / f"{clip_id}.mp4"
    ply_path = out_dir / "mesh_box.ply"
    meta_path = out_dir / "mesh_box_meta.json"

    bg_pad = _build_studio_background(w, h, seed=1010)
    verts_obj, norms_obj, uvs_obj, vcols_rgb, tris, base_atlas_rgba, face_normals = _build_box_mesh_and_atlas()
    _write_ascii_ply(ply_path, verts_obj, norms_obj, uvs_obj, vcols_rgb, tris)

    intrinsics = (560.0, 560.0, 320.0, 240.0)
    cam_k = [[560.0, 0.0, 320.0], [0.0, 560.0, 240.0], [0.0, 0.0, 1.0]]
    light_cam = np.array([-0.42, -0.52, -0.74], dtype=np.float64)
    light_cam /= np.linalg.norm(light_cam)

    tile_h = base_atlas_rgba.shape[0] // 2
    tile_w = base_atlas_rgba.shape[1] // 3
    base_rgb_tiles = base_atlas_rgba[:, :, :3].reshape(2, tile_h, 3, tile_w, 3).astype(np.float32)

    mask_0: np.ndarray | None = None
    poses_meta: list[dict[str, Any]] = []
    frames_gt: list[dict[str, Any]] = []

    with _FFmpegH264Writer(mp4_path, w, h, DEMO_FPS) as writer:
        for t in range(DEMO_FRAME_COUNT):
            u = 2.0 * math.pi * float(t) / float(DEMO_FRAME_COUNT)
            # Frame 0: yaw=+24, pitch=+18 exposes Front (-Z), Right (+X), and Top (-Y) simultaneously.
            yaw = 24.0 + 15.0 * math.sin(u)
            pitch = 18.0 + 9.0 * math.sin(2.0 * u)
            roll = 10.0 * math.sin(u)
            tx = 0.058 * math.sin(u)
            ty = 0.032 * math.sin(2.0 * u)
            tz = 0.41 - 0.038 * math.sin(u)

            rot = _rot_z(roll) @ _rot_x(pitch) @ _rot_y(yaw)
            tvec = np.array([tx, ty, tz], dtype=np.float64)
            rvec, _ = cv2.Rodrigues(rot)

            # Per-face 3D Lambertian normal illumination
            n_cam = face_normals @ rot.T
            dots = np.maximum(0.0, n_cam @ light_cam)
            face_gains = np.clip(0.64 + 0.42 * dots, 0.60, 1.12).astype(np.float32).reshape(2, 1, 3, 1, 1)

            lit_atlas = np.empty_like(base_atlas_rgba)
            lit_atlas[:, :, :3] = np.clip(base_rgb_tiles * face_gains + 0.5, 0.0, 255.0).astype(np.uint8).reshape(
                2 * tile_h, 3 * tile_w, 3
            )
            lit_atlas[:, :, 3] = 255

            verts_cam = (verts_obj.astype(np.float64) @ rot.T + tvec[None, :]).astype(np.float32)
            bg_bgr = _crop_breathing_bg(bg_pad, w, h, t)
            frame_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
            res = rasterize_mesh_rgbd(
                verts_cam,
                tris,
                uvs_obj,
                lit_atlas,
                intrinsics,
                frame_rgb,
                wrap_repeat=False,
                opacity=1.0,
                blend_mode="normal",
                backend="auto",
            )
            if t == 0:
                mask_0 = res.mask.copy()

            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            pose_entry = {
                "frame_index": t,
                "timestamp_s": round(t / DEMO_FPS, 6),
                "rvec": rvec.ravel().tolist(),
                "tvec": tvec.tolist(),
                "R_3x3": rot.tolist(),
            }
            poses_meta.append(pose_entry)
            frames_gt.append(
                {
                    "frame_index": t,
                    "timestamp_s": round(t / DEMO_FPS, 6),
                    "visible": True,
                    "visibility_fraction": 1.0,
                    "gt_rvec": pose_entry["rvec"],
                    "gt_tvec": pose_entry["tvec"],
                    "gt_R_3x3": pose_entry["R_3x3"],
                }
            )

    assert mask_0 is not None
    ys0, xs0 = np.nonzero(mask_0 > 0)
    bx0 = max(0.0, float(int(np.min(xs0)) - 8))
    by0 = max(0.0, float(int(np.min(ys0)) - 8))
    bx1 = min(float(w), float(int(np.max(xs0)) + 9))
    by1 = min(float(h), float(int(np.max(ys0)) + 9))

    selection = SelectionSpec(
        frame_index=0,
        positive_points=[(320.0, 240.0), (275.0, 210.0), (360.0, 270.0)],
        negative_points=[(80.0, 75.0), (560.0, 405.0)],
        bbox=(bx0, by0, bx1 - bx0, by1 - by0),
    )
    _verify_frame0_selection(mask_0, selection, clip_id)

    meta_payload = {
        "clip_id": clip_id,
        "mesh_file": "mesh_box.ply",
        "vertex_units": "meters",
        "box_half_extents_m": [0.095, 0.072, 0.062],
        "intrinsics": list(intrinsics),
        "cam_K_3x3": cam_k,
        "onboarding_frame_0": poses_meta[0],
        "evaluation_only_notice": "ground_truth_poses is strictly for external test evaluation, never runtime tracking.",
        "ground_truth_poses": poses_meta,
    }
    tmp_meta = out_dir / f".pending-{os.getpid()}-mesh_box_meta.json"
    tmp_meta.write_text(json.dumps(meta_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp_meta, meta_path)

    info = DemoClipInfo(
        id=clip_id,
        path=str(mp4_path),
        width=w,
        height=h,
        fps=DEMO_FPS,
        frame_count=DEMO_FRAME_COUNT,
        default_selection=selection,
        recommended_wrap="mesh_6dof",
    )
    return info, {
        "clip_id": clip_id,
        "mesh_file": str(ply_path),
        "intrinsics": list(intrinsics),
        "frames": frames_gt,
    }


def _corpus_is_ready(out_dir: Path) -> bool:
    required_files = [
        out_dir / "demo_manifest.json",
        out_dir / "ground_truth_eval.json",
        out_dir / "mesh_box.ply",
        out_dir / "mesh_box_meta.json",
        *(out_dir / f"{cid}.mp4" for cid in DEMO_CLIP_IDS),
    ]
    for fpath in required_files:
        if not fpath.is_file():
            return False
    for cid in DEMO_CLIP_IDS:
        mp4 = out_dir / f"{cid}.mp4"
        if mp4.stat().st_size < 10_000:
            return False
    try:
        clips = load_demo_manifest(out_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    if [c.id for c in clips] != list(DEMO_CLIP_IDS):
        return False
    return all(Path(c.path).is_file() for c in clips)


def load_demo_manifest(output_dir: str | os.PathLike[str] = DEFAULT_DEMO_DIR) -> list[DemoClipInfo]:
    """Load validated DemoClipInfo list from demo_manifest.json in output_dir."""
    out = _validate_output_dir(output_dir)
    manifest_path = out / "demo_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Demo manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("clips"), list):
        raise ContractValidationError(f"Invalid demo_manifest.json structure in {manifest_path}")
    return [DemoClipInfo.from_dict(item) for item in payload["clips"]]


def generate_demo_corpus(
    output_dir: str | os.PathLike[str] = DEFAULT_DEMO_DIR,
    *,
    overwrite: bool = False,
) -> list[DemoClipInfo]:
    """Deterministically generate all 5 H.264 MP4 demo clips, 3D PLY box, and manifests."""
    out = _validate_output_dir(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not overwrite and _corpus_is_ready(out):
        return load_demo_manifest(out)

    generators = (
        _generate_planar_card,
        _generate_planar_hd,
        _generate_cylinder_mug,
        _generate_occlusion_recovery,
        _generate_mesh_box,
    )

    clips: list[DemoClipInfo] = []
    gt_eval: dict[str, Any] = {
        "evaluation_only": True,
        "fps": DEMO_FPS,
        "frame_count": DEMO_FRAME_COUNT,
        "clips": {},
    }

    for gen_fn in generators:
        clip_info, clip_gt = gen_fn(out)
        clips.append(clip_info)
        gt_eval["clips"][clip_info.id] = clip_gt

    manifest_payload = {"clips": [c.to_dict() for c in clips]}
    tmp_manifest = out / f".pending-{os.getpid()}-demo_manifest.json"
    tmp_manifest.write_text(json.dumps(manifest_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, out / "demo_manifest.json")

    tmp_gt = out / f".pending-{os.getpid()}-ground_truth_eval.json"
    tmp_gt.write_text(json.dumps(gt_eval, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp_gt, out / "ground_truth_eval.json")

    return clips


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m visualizeit.demo_assets",
        description="Generate deterministic H.264 MP4 & 3D PLY demo corpus for visualizeIT.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_DEMO_DIR),
        help="Target directory for generated demo videos and manifests (default: .cache/demo-videos).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate demo corpus even if valid cached files already exist.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON manifest to stdout.",
    )

    try:
        args = parser.parse_args(argv)
        clips = generate_demo_corpus(args.output_dir, overwrite=bool(args.force))
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"error: failed generating demo corpus: {exc}", file=sys.stderr)
        return 1

    payload = {"clips": [c.to_dict() for c in clips]}
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        for c in clips:
            print(f"[demo-assets] {c.id}: {c.path} ({c.width}x{c.height} @ {c.fps:.1f} FPS, {c.frame_count} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
