"""Synthetic video/image/PLY fixture generators, ffprobe/frame decoders, and pixel delta oracles.

Strictly stdlib-only at module import time; `numpy`, `PIL`, and `cv2` are imported lazily.
All temporary/intermediate files use XFS `.cache/tmp` (never `/tmp`).
"""

from __future__ import annotations

from fractions import Fraction
import io
import json
import math
from pathlib import Path
from typing import Any

from .native_abi import ray_triangle_uv_oracle
from .paths_and_proc import assert_xfs_temp_path, build_xfs_env, run_subproc

CODEC_ARGS: dict[str, list[str]] = {
    ".mp4": ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p"],
    ".mov": ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p"],
    ".webm": ["-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "8", "-b:v", "1500k", "-pix_fmt", "yuv420p"],
    ".avi": ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"],
    ".y4m": ["-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe"],
}


def _compute_card_motion_geometry(
    width: int,
    height: int,
    frames: int,
) -> tuple[int, int, int, int, int, int, int]:
    """Compute `(bw, bh, x0, y0, dx, dy, grid_w)` keeping the moving card inside the frame."""
    assert width >= 32 and height >= 32, f"Dimensions must be >= 32x32, got {width}x{height}"
    max_step = max(1, frames - 1)

    if (width, height) == (64, 64):
        bw, bh = 28, 28
        x0, y0 = 10, 14
        dx = 1 if (x0 + max_step + bw <= width - 4) else 0
        dy = 0
        gw = 7
        return bw, bh, x0, y0, dx, dy, gw

    if (width, height) == (1280, 720):
        bw, bh, gw = 360, 260, 36
        pref_x0, pref_y0, pref_dx, pref_dy = 320, 200, 6, 3
    else:
        bw = max(24, min(width // 2, (width * 5 // 16) & ~1))
        bh = max(24, min(height // 2, (height * 5 // 16) & ~1))
        gw = max(6, bw // 8)
        pref_x0, pref_y0 = (width // 4) & ~1, (height // 4) & ~1
        pref_dx = max(1, width // 160)
        pref_dy = max(1, height // 240)

    margin = 12
    if (
        pref_x0 + pref_dx * max_step + bw <= width - margin
        and pref_y0 + pref_dy * max_step + bh <= height - margin
    ):
        return bw, bh, pref_x0, pref_y0, pref_dx, pref_dy, gw

    x0 = max(margin, width // 12) & ~1
    y0 = max(margin, height // 12) & ~1
    avail_x = max(0, width - margin - bw - x0)
    avail_y = max(0, height - margin - bh - y0)
    dx = min(pref_dx, avail_x // max_step)
    dy = min(pref_dy, avail_y // max_step)
    return bw, bh, x0, y0, dx, dy, gw


def _build_frozen_bg_fg_lavfi(
    *,
    width: int,
    height: int,
    bw: int,
    bh: int,
    gw: int,
    fps: int,
    shaded_card: bool,
) -> tuple[str, str]:
    bg = (
        f"color=c=0x1e293b:s={width}x{height}:r={fps},"
        f"noise=all_seed=7:alls=8,trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/({fps}*TB)"
    )
    if shaded_card:
        half_w = bw // 2
        right_w = bw - half_w
        fg = (
            f"color=c=0xdcdcdc:s={bw}x{bh}:r={fps},"
            f"drawbox=x={half_w}:y=0:w={right_w}:h={bh}:c=0x696969:t=fill,"
            f"noise=all_seed=42:alls=22,"
            f"drawgrid=w={gw}:h={gw}:t=2:c=0x888888@0.55,"
            f"drawbox=x=0:y=0:w={bw}:h={bh}:c=0xf0f0f0@0.80:t=2,"
            f"trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/({fps}*TB)"
        )
    else:
        fg = (
            f"testsrc2=s={bw}x{bh}:r={fps},noise=all_seed=42:alls=25,"
            f"drawgrid=w={gw}:h={gw}:t=2:c=white@0.85,"
            f"trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/({fps}*TB)"
        )
    return bg, fg


def bbox_at_frame(clip_meta: dict[str, Any], frame_index: int) -> tuple[int, int, int, int]:
    """Return exact ground-truth `(x, y, w, h)` bounding box at `frame_index` for linear clips."""
    x = int(clip_meta["x0"]) + int(clip_meta["dx"]) * int(frame_index)
    y = int(clip_meta["y0"]) + int(clip_meta["dy"]) * int(frame_index)
    return (x, y, int(clip_meta["box_w"]), int(clip_meta["box_h"]))


def make_textured_tracking_clip(
    out_path: Path,
    *,
    width: int = 640,
    height: int = 480,
    frames: int = 30,
    fps: int = 30,
    shaded_card: bool = False,
    tmp_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate a deterministic video clip with a rigidly translating textured target in <0.35s.

    Supports `.mp4`, `.webm`, `.mov`, and `.avi`.
    When `shaded_card=True`, modulates left half brighter (`L~215`) and right half darker (`L~105`)
    while keeping crisp interior corners for KLT/ORB tracking.
    """
    target = assert_xfs_temp_path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ext = target.suffix.lower()
    if ext not in CODEC_ARGS or ext == ".y4m":
        raise ValueError(f"Unsupported video container extension {ext!r}; expected .mp4/.webm/.mov/.avi")

    bw, bh, x0, y0, dx, dy, gw = _compute_card_motion_geometry(width, height, frames)
    bg, fg = _build_frozen_bg_fg_lavfi(
        width=width,
        height=height,
        bw=bw,
        bh=bh,
        gw=gw,
        fps=fps,
        shaded_card=shaded_card,
    )
    filt = f"[0:v][1:v]overlay=x={x0}+{dx}*n:y={y0}+{dy}*n:eval=frame,format=yuv420p"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        bg,
        "-f",
        "lavfi",
        "-i",
        fg,
        "-filter_complex",
        filt,
        "-frames:v",
        str(frames),
        *CODEC_ARGS[ext],
        str(target),
    ]
    run_subproc(cmd, timeout=30.0, check=True, tmp_dir=tmp_dir or target.parent)
    return {
        "path": str(target),
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "x0": x0,
        "y0": y0,
        "box_w": bw,
        "box_h": bh,
        "dx": dx,
        "dy": dy,
        "center_frame0": (x0 + bw // 2, y0 + bh // 2),
        "bbox_frame0": (x0, y0, bw, bh),
        "shaded_card": shaded_card,
    }


def make_occlusion_clip(
    out_path: Path,
    *,
    width: int = 640,
    height: int = 480,
    frames: int = 60,
    fps: int = 30,
    occlude_ranges: list[tuple[int, int]] | None = None,
    tmp_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate a tracking clip where an opaque featureless occluder hides 100% of the card during `occlude_ranges`.

    `occlude_ranges` uses inclusive `(start_frame, end_frame)` indices (default `[(20, 38)]`).
    """
    target = assert_xfs_temp_path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ext = target.suffix.lower()
    if ext not in CODEC_ARGS:
        raise ValueError(f"Unsupported extension: {ext!r}")

    ranges = list(occlude_ranges) if occlude_ranges is not None else [(20, min(38, max(20, frames - 12)))]
    for s_idx, e_idx in ranges:
        assert 0 <= s_idx <= e_idx < frames, f"Invalid occlude range {(s_idx, e_idx)} for frames={frames}"

    bw, bh, x0, y0, dx, dy, gw = _compute_card_motion_geometry(width, height, frames)
    bg, fg = _build_frozen_bg_fg_lavfi(
        width=width,
        height=height,
        bw=bw,
        bh=bh,
        gw=gw,
        fps=fps,
        shaded_card=False,
    )
    pad = 20
    drawboxes: list[str] = []
    for s_idx, e_idx in ranges:
        xs_occ = [x0 + dx * s_idx, x0 + dx * e_idx]
        ys_occ = [y0 + dy * s_idx, y0 + dy * e_idx]
        ux1 = max(0, min(xs_occ) - pad)
        uy1 = max(0, min(ys_occ) - pad)
        ux2 = min(width, max(xs_occ) + bw + pad)
        uy2 = min(height, max(ys_occ) + bh + pad)
        drawboxes.append(
            f"drawbox=x={ux1}:y={uy1}:w={ux2 - ux1}:h={uy2 - uy1}:"
            f"c=0x0f172a:t=fill:enable='between(n,{s_idx},{e_idx})'"
        )
    occ_chain = ",".join(drawboxes)
    filt = f"[0:v][1:v]overlay=x={x0}+{dx}*n:y={y0}+{dy}*n:eval=frame,{occ_chain},format=yuv420p"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        bg,
        "-f",
        "lavfi",
        "-i",
        fg,
        "-filter_complex",
        filt,
        "-frames:v",
        str(frames),
        *CODEC_ARGS[ext],
        str(target),
    ]
    run_subproc(cmd, timeout=30.0, check=True, tmp_dir=tmp_dir or target.parent)
    occluded_set = {f for s_idx, e_idx in ranges for f in range(s_idx, e_idx + 1)}
    first_occ = min(occluded_set) if occluded_set else frames
    last_occ = max(occluded_set) if occluded_set else -1
    return {
        "path": str(target),
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "x0": x0,
        "y0": y0,
        "box_w": bw,
        "box_h": bh,
        "dx": dx,
        "dy": dy,
        "center_frame0": (x0 + bw // 2, y0 + bh // 2),
        "bbox_frame0": (x0, y0, bw, bh),
        "occlude_ranges": ranges,
        "visible_frames_before": [f for f in range(frames) if f < first_occ],
        "occluded_frames": sorted(occluded_set),
        "visible_frames_after": [f for f in range(frames) if f > last_occ],
        "occluder_rgb": (15, 23, 42),
    }


def make_solid_clip(
    out_path: Path,
    *,
    width: int = 640,
    height: int = 480,
    frames: int = 15,
    fps: int = 30,
    color: str = "0x000000",
    tmp_dir: Path | None = None,
) -> Path:
    """Generate a uniform solid-color video clip (useful for zero-texture / boundary tests)."""
    target = assert_xfs_temp_path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ext = target.suffix.lower()
    if ext not in CODEC_ARGS:
        raise ValueError(f"Unsupported extension: {ext!r}")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={width}x{height}:r={fps}",
        "-frames:v",
        str(frames),
        *CODEC_ARGS[ext],
        str(target),
    ]
    run_subproc(cmd, timeout=20.0, check=True, tmp_dir=tmp_dir or target.parent)
    return target


def make_y4m_camera_clip(
    out_path: Path,
    *,
    width: int = 640,
    height: int = 480,
    frames: int = 60,
    fps: int = 30,
    occlude_ranges: list[tuple[int, int]] | None = None,
    tmp_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate a `.y4m` `yuv420p` camera clip for Chromium `--use-file-for-fake-video-capture`.

    Uses a smooth periodic `1 - cos(2*PI*n/frames)` trajectory starting at `(x0, y0)` on frame 0
    so Chromium's seamless `.y4m` EOF loop has zero velocity discontinuity at loop boundaries.
    """
    target = assert_xfs_temp_path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    assert target.suffix.lower() == ".y4m", f"Expected .y4m path, got {target}"

    bw = max(24, (width * 5 // 16) & ~1)
    bh = max(24, (height * 5 // 16) & ~1)
    gw = max(6, bw // 8)
    x0 = (width // 5) & ~1
    y0 = (height // 5) & ~1
    span_x = max(16, min(width - bw - x0 - 16, width // 5))
    span_y = max(12, min(height - bh - y0 - 16, height // 6))

    bg, fg = _build_frozen_bg_fg_lavfi(
        width=width,
        height=height,
        bw=bw,
        bh=bh,
        gw=gw,
        fps=fps,
        shaded_card=False,
    )
    period = max(1, frames)
    x_expr = f"trunc({x0}+{span_x}*(1-cos(2*PI*n/{period}))/2)"
    y_expr = f"trunc({y0}+{span_y}*(1-cos(2*PI*n/{period}))/2)"

    inputs = ["-f", "lavfi", "-i", bg, "-f", "lavfi", "-i", fg]
    if occlude_ranges:
        enable_expr = "+".join(f"between(n,{s_idx},{e_idx})" for s_idx, e_idx in occlude_ranges)
        filt = (
            f"[0:v][1:v]overlay=x='{x_expr}':y='{y_expr}':eval=frame,"
            f"drawbox=x=0:y=0:w={width}:h={height}:c=0x0f172a:t=fill:enable='{enable_expr}',format=yuv420p"
        )
    else:
        filt = f"[0:v][1:v]overlay=x='{x_expr}':y='{y_expr}':eval=frame,format=yuv420p"

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *inputs,
        "-filter_complex",
        filt,
        "-frames:v",
        str(frames),
        *CODEC_ARGS[".y4m"],
        str(target),
    ]
    run_subproc(cmd, timeout=30.0, check=True, tmp_dir=tmp_dir or target.parent)
    return {
        "path": str(target),
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "x0": x0,
        "y0": y0,
        "box_w": bw,
        "box_h": bh,
        "span_x": span_x,
        "span_y": span_y,
        "center_frame0": (x0 + bw // 2, y0 + bh // 2),
        "bbox_frame0": (x0, y0, bw, bh),
        "occlude_ranges": list(occlude_ranges or []),
    }


def transcode_clip(
    src_path: Path,
    dst_path: Path,
    *,
    tmp_dir: Path | None = None,
) -> Path:
    """Transcode an existing clip into `.mp4`, `.webm`, `.mov`, `.avi`, or `.y4m`."""
    target = assert_xfs_temp_path(dst_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ext = target.suffix.lower()
    if ext not in CODEC_ARGS:
        raise ValueError(f"Unsupported target extension: {ext!r}")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src_path),
        *CODEC_ARGS[ext],
        str(target),
    ]
    run_subproc(cmd, timeout=45.0, check=True, tmp_dir=tmp_dir or target.parent)
    return target


def probe_video(path: Path, *, tmp_dir: Path | None = None) -> dict[str, Any]:
    """Inspect video metadata and exact decoded frame count via `ffprobe -count_frames`."""
    assert path.is_file() and path.stat().st_size > 0, f"Video file missing or empty: {path}"
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames,duration:format=format_name,duration,size",
        "-of",
        "json",
        str(path),
    ]
    proc = run_subproc(cmd, timeout=30.0, check=True, tmp_dir=tmp_dir)
    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    assert streams, f"ffprobe found zero video streams in {path}"
    s = streams[0]
    fmt = data.get("format") or {}
    dur_str = s.get("duration") or fmt.get("duration")
    nb_read = int(s["nb_read_frames"])
    fps_val = float(Fraction(s["r_frame_rate"]))
    return {
        "path": str(path),
        "codec_name": str(s["codec_name"]),
        "width": int(s["width"]),
        "height": int(s["height"]),
        "pix_fmt": s.get("pix_fmt"),
        "fps": fps_val,
        "nb_read_frames": nb_read,
        "frame_count": nb_read,
        "duration_s": float(dur_str) if dur_str is not None else (nb_read / fps_val if fps_val > 0 else None),
        "format_name": str(fmt.get("format_name", "")),
        "size_bytes": int(fmt["size"]) if fmt.get("size") is not None else path.stat().st_size,
    }


def decode_video_frames(
    path: Path,
    frame_indices: list[int] | None = None,
    *,
    max_frames: int | None = None,
) -> dict[int, Any]:
    """Decode video frames into `{frame_idx: RGB uint8 ndarray (H, W, 3)}`.

    Uses `cv2.VideoCapture` when OpenCV is installed, with an automatic pure-`ffmpeg`
    `rawvideo rgb24` pipe fallback so frame verification also works on bare stdlib + `numpy`.
    """
    import subprocess
    import numpy as np

    assert path.is_file(), f"Video file does not exist: {path}"
    wanted = set(frame_indices) if frame_indices is not None else None
    stop_after = max(wanted) if (wanted is not None and len(wanted) > 0) else None
    if max_frames is not None:
        stop_after = min(stop_after, max_frames - 1) if stop_after is not None else (max_frames - 1)

    if wanted is not None and len(wanted) == 0:
        return {}

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        cv2 = None

    out: dict[int, Any] = {}
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), f"cv2.VideoCapture failed to open {path}"
        try:
            idx = 0
            while True:
                if stop_after is not None and idx > stop_after:
                    break
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                if wanted is None or idx in wanted:
                    out[idx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                idx += 1
        finally:
            cap.release()
        return out

    meta = probe_video(path)
    w, h = int(meta["width"]), int(meta["height"])
    frame_bytes = h * w * 3
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=build_xfs_env(),
    )
    assert proc.stdout is not None
    try:
        idx = 0
        while True:
            if stop_after is not None and idx > stop_after:
                break
            buf = b""
            while len(buf) < frame_bytes:
                chunk = proc.stdout.read(frame_bytes - len(buf))
                if not chunk:
                    break
                buf += chunk
            if len(buf) < frame_bytes:
                break
            if wanted is None or idx in wanted:
                out[idx] = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3)).copy()
            idx += 1
    finally:
        proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
    return out


decode_frames = decode_video_frames


def decode_image_bytes(data: bytes) -> tuple[Any, str]:
    """Decode PNG/JPEG/WebP bytes via Pillow and return `(ndarray, mode)`."""
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        img.load()
        mode = img.mode
        arr = np.asarray(img).copy()
    return arr, mode


def _build_rgba_artwork_array(
    *,
    width: int = 256,
    height: int = 256,
    mode: str = "emblem",
    alpha_value: int = 255,
) -> Any:
    import numpy as np

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    ys, xs = np.ogrid[:height, :width]
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    min_dim = float(min(width, height))
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    if mode == "emblem":
        r_outer = 0.40 * min_dim
        inside = dist <= r_outer
        ring1 = dist <= 0.24 * min_dim
        ring2 = dist <= 0.12 * min_dim
        rgba[inside] = [255, 72, 88, int(alpha_value)]
        rgba[ring1] = [24, 210, 235, int(alpha_value)]
        rgba[ring2] = [255, 214, 42, int(alpha_value)]
        cross = inside & ((np.abs(xs - cx) <= max(2.0, 0.03 * min_dim)) | (np.abs(ys - cy) <= max(2.0, 0.03 * min_dim)))
        rgba[cross] = [250, 250, 255, int(alpha_value)]
    elif mode == "ring":
        r_in = 0.18 * min_dim
        r_out = 0.43 * min_dim
        band = (dist >= r_in) & (dist <= r_out)
        quad = ((xs >= cx) ^ (ys >= cy)) & band
        rgba[band] = [255, 90, 54, int(alpha_value)]
        rgba[quad] = [40, 215, 160, int(alpha_value)]
    elif mode == "zero_alpha":
        rgba[..., 0] = 255
        rgba[..., 1] = 32
        rgba[..., 2] = 180
        rgba[..., 3] = 0
    elif mode == "semi":
        stripe = ((xs // max(4, width // 12)) + (ys // max(4, height // 12))) % 2 == 0
        rgba[..., :3] = [36, 198, 255]
        rgba[stripe, :3] = [255, 108, 64]
        rgba[..., 3] = int(alpha_value)
    elif mode == "cutout_label":
        margin_x = max(6, width // 14)
        margin_y = max(6, height // 10)
        body = (
            (xs >= margin_x)
            & (xs < width - margin_x)
            & (ys >= margin_y)
            & (ys < height - margin_y)
        )
        stripe = ((ys // max(6, height // 10)) % 2) == 0
        rgba[body] = [255, 84, 72, int(alpha_value)]
        rgba[body & stripe] = [32, 196, 232, int(alpha_value)]
        # Die-cut transparent windows inside the label so underlying surface shows through
        hole_left = (np.abs(xs - width * 0.30) <= width * 0.09) & (np.abs(ys - cy) <= height * 0.20)
        hole_right = np.sqrt((xs - width * 0.70) ** 2 + (ys - cy) ** 2) <= 0.15 * min_dim
        rgba[hole_left | hole_right] = [0, 0, 0, 0]
    else:
        raise ValueError(
            f"Unknown RGBA mode {mode!r}; expected 'emblem', 'ring', 'zero_alpha', 'semi', or 'cutout_label'"
        )
    return rgba


def make_rgba_png(
    out_path: Path,
    *,
    width: int = 256,
    height: int = 256,
    mode: str = "emblem",
    alpha_value: int = 255,
) -> Path:
    """Create a synthetic RGBA PNG with exact alpha topology (`emblem`, `ring`, `zero_alpha`, `semi`, `cutout_label`)."""
    from PIL import Image

    target = assert_xfs_temp_path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rgba = _build_rgba_artwork_array(width=width, height=height, mode=mode, alpha_value=alpha_value)
    Image.fromarray(rgba, mode="RGBA").save(target, format="PNG")
    return target


def make_rgb_jpeg(
    out_path: Path,
    *,
    width: int = 320,
    height: int = 240,
) -> Path:
    """Create a vibrant 3-channel RGB JPEG artwork file."""
    import numpy as np
    from PIL import Image

    target = assert_xfs_temp_path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    ys, xs = np.ogrid[:height, :width]
    checker = ((xs // max(8, width // 10)) + (ys // max(8, height // 8))) % 2 == 0
    rgb[...] = [24, 152, 224]
    rgb[checker] = [255, 168, 34]
    border = (xs < 8) | (xs >= width - 8) | (ys < 8) | (ys >= height - 8)
    rgb[border] = [245, 245, 250]
    Image.fromarray(rgb, mode="RGB").save(target, format="JPEG", quality=92)
    return target


def make_rgba_webp(
    out_path: Path,
    *,
    width: int = 256,
    height: int = 256,
    mode: str = "emblem",
) -> Path:
    """Create a lossless RGBA WebP artwork file (with automatic `ffmpeg libwebp` fallback)."""
    from PIL import Image

    target = assert_xfs_temp_path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rgba = _build_rgba_artwork_array(width=width, height=height, mode=mode, alpha_value=255)
    try:
        Image.fromarray(rgba, mode="RGBA").save(target, format="WEBP", lossless=True)
        return target
    except Exception:
        tmp_png = target.with_name(f".{target.stem}_webp_src.png")
        try:
            Image.fromarray(rgba, mode="RGBA").save(tmp_png, format="PNG")
            run_subproc(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(tmp_png),
                    "-c:v",
                    "libwebp",
                    "-lossless",
                    "1",
                    str(target),
                ],
                timeout=20.0,
                check=True,
                tmp_dir=target.parent,
            )
        finally:
            tmp_png.unlink(missing_ok=True)
        return target


def make_ascii_ply(
    out_path: Path,
    *,
    valid: bool = True,
    defect: str | None = None,
    half_extents: tuple[float, float, float] = (0.10, 0.08, 0.06),
) -> Path:
    """Create a valid UV-mapped ASCII `.ply` box mesh or a specific defective `.ply` boundary fixture.

    Supported `defect` values: `None`, `"bad_header"`, `"oob_index"`, `"nan_vertex"`, `"empty"`, `"zero_faces"`.
    """
    target = assert_xfs_temp_path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    eff_defect = defect
    if not valid and eff_defect is None:
        eff_defect = "bad_header"

    if eff_defect == "empty":
        target.write_bytes(b"")
        return target
    if eff_defect == "bad_header":
        target.write_text("NOT_A_PLY_FILE\nformat invalid 9.9\nend_header\n0 0 0\n", encoding="ascii")
        return target
    if eff_defect == "zero_faces":
        target.write_text(
            "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\n"
            "element face 0\nproperty list uchar int vertex_indices\nend_header\n",
            encoding="ascii",
        )
        return target

    hx, hy, hz = float(half_extents[0]), float(half_extents[1]), float(half_extents[2])
    # 6 faces x 4 vertices = 24 UV-mapped vertices, 12 outward-facing triangles
    face_quads = [
        # Front (-Z facing camera in object coords or +Z)
        [(-hx, -hy, -hz, 0.0, 0.0), (+hx, -hy, -hz, 1.0, 0.0), (+hx, +hy, -hz, 1.0, 1.0), (-hx, +hy, -hz, 0.0, 1.0)],
        # Back (+Z)
        [(+hx, -hy, +hz, 0.0, 0.0), (-hx, -hy, +hz, 1.0, 0.0), (-hx, +hy, +hz, 1.0, 1.0), (+hx, +hy, +hz, 0.0, 1.0)],
        # Left (-X)
        [(-hx, -hy, +hz, 0.0, 0.0), (-hx, -hy, -hz, 1.0, 0.0), (-hx, +hy, -hz, 1.0, 1.0), (-hx, +hy, +hz, 0.0, 1.0)],
        # Right (+X)
        [(+hx, -hy, -hz, 0.0, 0.0), (+hx, -hy, +hz, 1.0, 0.0), (+hx, +hy, +hz, 1.0, 1.0), (+hx, +hy, -hz, 0.0, 1.0)],
        # Top (-Y)
        [(-hx, -hy, +hz, 0.0, 0.0), (+hx, -hy, +hz, 1.0, 0.0), (+hx, -hy, -hz, 1.0, 1.0), (-hx, -hy, -hz, 0.0, 1.0)],
        # Bottom (+Y)
        [(-hx, +hy, -hz, 0.0, 0.0), (+hx, +hy, -hz, 1.0, 0.0), (+hx, +hy, +hz, 1.0, 1.0), (-hx, +hy, +hz, 0.0, 1.0)],
    ]
    verts: list[str] = []
    faces: list[str] = []
    for f_idx, quad in enumerate(face_quads):
        base = f_idx * 4
        for vx, vy, vz, vu, vv in quad:
            verts.append(f"{vx:.6f} {vy:.6f} {vz:.6f} {vu:.6f} {vv:.6f}")
        faces.append(f"3 {base} {base + 2} {base + 1}")
        faces.append(f"3 {base} {base + 3} {base + 2}")

    if eff_defect == "nan_vertex":
        verts[0] = "nan inf -nan 0.0 0.0"
    elif eff_defect == "oob_index":
        faces[0] = "3 0 1 999"
    elif eff_defect is not None:
        raise ValueError(f"Unknown PLY defect {eff_defect!r}")

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float u\n"
        "property float v\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    target.write_text(header + "\n".join(verts) + "\n" + "\n".join(faces) + "\n", encoding="ascii")
    return target


def compute_roi_and_bg_delta(
    raw_rgb: Any,
    out_rgb: Any,
    bbox: tuple[int, int, int, int],
    margin: int = 24,
) -> tuple[float, float]:
    """Compute `(roi_mean_abs_diff, bg_mean_abs_diff)` between raw and composited RGB frames."""
    import numpy as np

    a = np.asarray(raw_rgb, dtype=np.float32)
    b = np.asarray(out_rgb, dtype=np.float32)
    assert a.shape == b.shape and a.ndim == 3 and a.shape[2] == 3, (
        f"Expected matching (H, W, 3) RGB frames, got {a.shape} vs {b.shape}"
    )
    h, w = int(a.shape[0]), int(a.shape[1])
    x, y, bw, bh = (int(round(v)) for v in bbox)
    rx1 = max(0, min(w - 1, x))
    ry1 = max(0, min(h - 1, y))
    rx2 = max(rx1 + 1, min(w, x + bw))
    ry2 = max(ry1 + 1, min(h, y + bh))

    diff = np.abs(b - a)
    roi_mean = float(diff[ry1:ry2, rx1:rx2].mean())

    bx1 = max(0, x - int(margin))
    by1 = max(0, y - int(margin))
    bx2 = min(w, x + bw + int(margin))
    by2 = min(h, y + bh + int(margin))
    bg_mask = np.ones((h, w), dtype=bool)
    bg_mask[by1:by2, bx1:bx2] = False
    bg_mean = float(diff[bg_mask].mean()) if np.any(bg_mask) else 0.0
    return roi_mean, bg_mean


compute_roi_outside_delta = compute_roi_and_bg_delta


def compute_psnr(img_a: Any, img_b: Any) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between two uint8 images (returns 100.0 dB if identical)."""
    import numpy as np

    a = np.asarray(img_a, dtype=np.float64)
    b = np.asarray(img_b, dtype=np.float64)
    assert a.shape == b.shape, f"Shape mismatch in compute_psnr: {a.shape} vs {b.shape}"
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * math.log10((255.0 * 255.0) / mse))


__all__ = [
    "CODEC_ARGS",
    "bbox_at_frame",
    "compute_psnr",
    "compute_roi_and_bg_delta",
    "compute_roi_outside_delta",
    "decode_frames",
    "decode_image_bytes",
    "decode_video_frames",
    "make_ascii_ply",
    "make_occlusion_clip",
    "make_rgb_jpeg",
    "make_rgba_png",
    "make_rgba_webp",
    "make_solid_clip",
    "make_textured_tracking_clip",
    "make_y4m_camera_clip",
    "probe_video",
    "ray_triangle_uv_oracle",
    "transcode_clip",
]
