"""Streaming causal >=15-30 FPS video pipeline, shared per-frame live core, and 30 Hz replay harness."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any, Literal

import cv2
import numpy as np

from .compositor import SurfaceCompositor
from .contracts import (
    ContractValidationError,
    DesignSpec,
    PlacementSpec,
    SelectionSpec,
    TrackFrameResult,
    VideoRunReport,
)
from .designs import load_design_image, render_design
from .segmentation import SegmentationResult, segment_frame
from .tracker import MeshGeometry, VideoSurfaceTracker, resolve_mesh

REPO_ROOT = Path(__file__).resolve().parent.parent


def _validate_safe_output_path(path: str | os.PathLike[str], label: str = "output") -> Path:
    raw = os.fspath(path)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="surrogateescape")
    if not str(raw).strip():
        raise ContractValidationError(f"{label} path must be non-empty")
    out = Path(os.path.abspath(raw))
    posix = out.as_posix()
    if posix == "/tmp" or posix.startswith("/tmp/"):
        raise ContractValidationError(f"Refusing to write {label} under /tmp: {out}")
    if ".agents/teamwork" in posix:
        raise ContractValidationError(f"Refusing to write {label} under .agents/teamwork: {out}")
    if out.exists() and out.is_dir():
        raise ContractValidationError(f"{label} path points to an existing directory: {out}")
    return out


_STATIC_IMAGE_CODECS = frozenset({"png", "mjpeg", "webp", "bmp", "gif", "tiff", "ppm", "pgm"})
_STATIC_IMAGE_FORMATS = frozenset({"image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe", "tty"})


def probe_video_info(input_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Probe a video file and return {path, width, height, fps, frame_count, duration_s, codec}."""
    p = Path(os.path.abspath(os.fspath(input_path)))
    if not p.is_file() or p.stat().st_size == 0:
        raise ContractValidationError(f"Input video file is missing or empty: {p}")

    width = height = frame_count = 0
    fps = 0.0
    codec = "unknown"
    duration_s = 0.0
    format_name = ""

    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "format=format_name:stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames,nb_frames,duration",
                "-of",
                "json",
                str(p),
            ],
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            raise ContractValidationError(f"Input is not a valid video container: {p}")
        info = json.loads(proc.stdout)
        format_name = str((info.get("format") or {}).get("format_name") or "").lower()
        streams = info.get("streams") or []
        if not streams:
            raise ContractValidationError(f"Input contains no video streams: {p}")
        st = streams[0]
        width = int(st.get("width") or 0)
        height = int(st.get("height") or 0)
        codec = str(st.get("codec_name") or "unknown").lower()
        fmt_parts = set(format_name.split(","))
        if codec in _STATIC_IMAGE_CODECS or (fmt_parts & _STATIC_IMAGE_FORMATS):
            raise ContractValidationError(f"Input appears to be a static image/non-video ({codec}/{format_name}): {p}")
        nb = st.get("nb_read_frames") or st.get("nb_frames")
        if nb and str(nb).isdigit():
            frame_count = int(nb)
        for rate_key in ("avg_frame_rate", "r_frame_rate"):
            rate_s = str(st.get(rate_key) or "")
            if "/" in rate_s:
                num, den = rate_s.split("/", 1)
                if float(den) > 0 and float(num) > 0:
                    fps = float(num) / float(den)
                    break
        if st.get("duration") not in (None, "N/A"):
            try:
                duration_s = float(st["duration"])
            except ValueError:
                pass
    except ContractValidationError:
        raise
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as exc:
        raise ContractValidationError(f"Unable to probe input video {p}: {exc}") from exc

    # Cross-verify decodability via OpenCV
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ContractValidationError(f"Unable to open video stream: {p}")
    try:
        ok, first = cap.read()
        if not ok or first is None or first.ndim != 3:
            raise ContractValidationError(f"Video file contains no decodable frames: {p}")
        if width <= 0 or height <= 0:
            height, width = first.shape[:2]
        if fps <= 0.0 or not math.isfinite(fps):
            cv_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            fps = cv_fps if (cv_fps > 0.0 and math.isfinite(cv_fps)) else 30.0
        if frame_count <= 0:
            # Count frames explicitly so frame_count is exact
            cnt = 1
            while True:
                ret = cap.grab()
                if not ret:
                    break
                cnt += 1
            frame_count = cnt
    finally:
        cap.release()

    if duration_s <= 0.0 and fps > 0.0:
        duration_s = float(frame_count) / float(fps)

    return {
        "path": str(p),
        "width": int(width),
        "height": int(height),
        "fps": round(float(fps), 4),
        "frame_count": int(frame_count),
        "duration_s": round(float(duration_s), 4),
        "codec": codec,
    }


def _resolve_artwork_rgba(design: DesignSpec) -> np.ndarray:
    if design.source == "image":
        if not design.image_path:
            raise ContractValidationError("DesignSpec.image_path required when source='image'")
        return load_design_image(design.image_path)
    return render_design(design)


class _StreamingFFmpegEncoder:
    """Streaming BGR24 -> libx264 MP4 encoder with explicit thread control and atomic replace."""

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        *,
        threads: int = 2,
        crf: int = 16,
        preset: str = "ultrafast",
    ) -> None:
        self.output_path = output_path
        self.temp_path = output_path.parent / f".pending-{os.getpid()}-{output_path.name}"
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.threads = max(1, int(threads))
        self.crf = int(crf)
        self.preset = str(preset)
        self._use_i420 = (self.width % 2 == 0) and (self.height % 2 == 0)
        self._proc: subprocess.Popen[bytes] | None = None
        self.frames_written = 0

    def __enter__(self) -> _StreamingFFmpegEncoder:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = REPO_ROOT / ".cache" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "TMPDIR": str(tmp_dir), "TEMP": str(tmp_dir), "TMP": str(tmp_dir)}
        fps_str = f"{self.fps:.4f}".rstrip("0").rstrip(".") if not self.fps.is_integer() else str(int(self.fps))
        in_pix_fmt = "yuv420p" if self._use_i420 else "bgr24"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            in_pix_fmt,
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
            self.preset,
            "-threads",
            str(self.threads),
            "-crf",
            str(self.crf),
            "-g",
            "30",
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

    def write(self, frame_bgr: np.ndarray) -> float:
        t0 = time.perf_counter()
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Encoder not started")
        if frame_bgr.shape != (self.height, self.width, 3) or frame_bgr.dtype != np.uint8:
            raise ValueError(f"Unexpected frame shape {frame_bgr.shape}/{frame_bgr.dtype}")
        if self._use_i420:
            buf = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420))
        else:
            buf = np.ascontiguousarray(frame_bgr)
        self._proc.stdin.write(memoryview(buf))
        self.frames_written += 1
        return (time.perf_counter() - t0) * 1000.0

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._proc is None:
            return
        stderr_data = b""
        try:
            if self._proc.stdin is not None:
                try:
                    self._proc.stdin.flush()
                except OSError:
                    pass
                self._proc.stdin.close()
            if self._proc.stderr is not None:
                stderr_data = self._proc.stderr.read()
                self._proc.stderr.close()
            rc = self._proc.wait(timeout=60.0)
        except Exception:
            self._proc.kill()
            self._proc.wait()
            self.temp_path.unlink(missing_ok=True)
            raise

        if exc_type is not None:
            self.temp_path.unlink(missing_ok=True)
            return
        if rc != 0 or not self.temp_path.is_file() or self.temp_path.stat().st_size == 0:
            self.temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg encode failed (rc={rc}): {stderr_data.decode('utf-8', 'replace')}")
        self.temp_path.replace(self.output_path)


@dataclass(slots=True)
class FrameProcessOutput:
    frame_bgr: np.ndarray
    track_result: TrackFrameResult
    rendered_mask: np.ndarray | None
    prep_ms: float
    track_ms: float
    composite_ms: float


class VideoPipelineSession:
    """Shared causal per-frame engine used by both recorded file processing and 30 Hz live replay."""

    def __init__(
        self,
        *,
        design: DesignSpec,
        placement: PlacementSpec,
        tracking_height: int = 360,
        raster_backend: Literal["auto", "native_cpp", "opencv"] = "auto",
        mesh: MeshGeometry | None = None,
        intrinsics: tuple[float, float, float, float] | None = None,
        init_pose: tuple[np.ndarray, np.ndarray] | None = None,
        spatial_bucketing: bool = True,
        adaptive_appearance_gate: bool = True,
        local_occlusion_mask: bool = True,
        sharp_onboarding: bool = True,
        grazing_angle_gate: bool = False,
        max_klt_points: int = 180,
    ) -> None:
        self.design = design
        self.placement = placement
        self.artwork_rgba = _resolve_artwork_rgba(design)
        self.mesh = mesh if mesh is not None else (resolve_mesh(None) if placement.wrap_mode == "mesh_6dof" else None)
        self.intrinsics = intrinsics
        self.raster_backend = raster_backend
        self.local_occlusion_mask = bool(local_occlusion_mask)
        self.sharp_onboarding = bool(sharp_onboarding)

        self.tracker = VideoSurfaceTracker(
            wrap_mode=placement.wrap_mode,
            tracking_height=tracking_height,
            max_klt_points=max_klt_points,
            spatial_bucketing=spatial_bucketing,
            adaptive_appearance_gate=adaptive_appearance_gate,
            grazing_angle_gate=grazing_angle_gate,
            mesh=self.mesh,
            intrinsics=intrinsics,
            init_pose=init_pose,
        )
        self.compositor: SurfaceCompositor | None = None
        self.segmentation: SegmentationResult | None = None
        self._initialized = False

    def initialize(
        self,
        frame_bgr: np.ndarray,
        selection: SelectionSpec,
        timestamp_s: float = 0.0,
    ) -> FrameProcessOutput:
        """One-time frame segmentation + tracker onboarding + first-frame composite."""
        t_prep0 = time.perf_counter()
        seg = segment_frame(frame_bgr, selection, sharp_onboarding=self.sharp_onboarding)
        self.segmentation = seg
        prep_ms = (time.perf_counter() - t_prep0) * 1000.0

        track_res = self.tracker.initialize(frame_bgr, seg)
        track_res.timestamp_s = float(timestamp_s)

        self.compositor = SurfaceCompositor(
            self.artwork_rgba,
            self.placement,
            ref_bbox=self.tracker.ref_bbox,
            ref_mask=self.tracker.ref_mask,
            ref_frame_bgr=frame_bgr,
            mesh=self.tracker.mesh,
            intrinsics=self.intrinsics,
            raster_backend=self.raster_backend,
            local_occlusion_mask=self.local_occlusion_mask,
        )
        self._initialized = True

        out_bgr, mask, comp_ms = self.compositor.composite(frame_bgr, track_res)
        return FrameProcessOutput(
            frame_bgr=out_bgr,
            track_result=track_res,
            rendered_mask=mask,
            prep_ms=prep_ms,
            track_ms=track_res.track_ms,
            composite_ms=comp_ms,
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_s: float,
    ) -> FrameProcessOutput:
        """Causal per-frame step shared identically between recorded files and live camera/30Hz replay."""
        if not self._initialized or self.compositor is None:
            raise RuntimeError("VideoPipelineSession.initialize() must be called before process_frame()")
        t_prep0 = time.perf_counter()
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ContractValidationError("frame_bgr must be HxWx3 uint8")
        prep_ms = (time.perf_counter() - t_prep0) * 1000.0

        track_res = self.tracker.track_frame(frame_bgr, frame_index, timestamp_s)
        out_bgr, mask, comp_ms = self.compositor.composite(frame_bgr, track_res)
        return FrameProcessOutput(
            frame_bgr=out_bgr,
            track_result=track_res,
            rendered_mask=mask,
            prep_ms=prep_ms,
            track_ms=track_res.track_ms,
            composite_ms=comp_ms,
        )


def _smooth_homographies_contiguous(results: list[TrackFrameResult], alpha: float = 0.35) -> list[TrackFrameResult]:
    """Zero-phase forward-backward smoothing strictly within contiguous tracking runs.

    NEVER alters or bridges lost/recovering frames (full occlusion suppression is strictly preserved).
    """
    n = len(results)
    if n < 3:
        return results

    # Identify contiguous runs where state == "tracking" and homography_3x3 is not None
    i = 0
    while i < n:
        if results[i].state != "tracking" or results[i].homography_3x3 is None:
            i += 1
            continue
        j = i
        while j < n and results[j].state == "tracking" and results[j].homography_3x3 is not None:
            j += 1
        run_len = j - i
        if run_len >= 3:
            mats = [results[k].homography_3x3.astype(np.float64) for k in range(i, j)]  # type: ignore[union-attr]
            # Normalize H[2,2] = 1 for consistent interpolation
            norm_mats = [m / (m[2, 2] if abs(m[2, 2]) > 1e-9 else 1.0) for m in mats]
            # Forward EMA
            fwd = [norm_mats[0].copy()]
            for k in range(1, run_len):
                fwd.append((1.0 - alpha) * norm_mats[k] + alpha * fwd[-1])
            # Backward EMA (zero-phase)
            bwd = [fwd[-1].copy()]
            for k in range(run_len - 2, -1, -1):
                bwd.append((1.0 - alpha) * fwd[k] + alpha * bwd[-1])
            bwd.reverse()
            for idx_offset, smoothed in enumerate(bwd):
                if abs(smoothed[2, 2]) > 1e-6 and np.all(np.isfinite(smoothed)):
                    results[i + idx_offset].homography_3x3 = smoothed / smoothed[2, 2]
        i = j
    return results


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def _default_selection_for_frame(width: int, height: int, frame_index: int = 0) -> SelectionSpec:
    bx = width * 0.25
    by = height * 0.25
    bw = width * 0.50
    bh = height * 0.50
    return SelectionSpec(
        frame_index=frame_index,
        positive_points=[(width * 0.5, height * 0.5)],
        negative_points=[],
        bbox=(bx, by, bw, bh),
    )


class VideoPipeline:
    """End-to-end streaming recorded-video pipeline (`render_stride=1`, open-to-flush wall timing)."""

    def process_video(
        self,
        *,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str] | None,
        design: DesignSpec,
        placement: PlacementSpec,
        selection: SelectionSpec | None = None,
        mesh_spec: str | Path | MeshGeometry | None = None,
        intrinsics: tuple[float, float, float, float] | None = None,
        init_pose: tuple[np.ndarray, np.ndarray] | None = None,
        tracking_height: int = 360,
        threads: int = 2,
        raster_backend: Literal["auto", "native_cpp", "opencv"] = "auto",
        spatial_bucketing: bool = True,
        adaptive_appearance_gate: bool = True,
        local_occlusion_mask: bool = True,
        sharp_onboarding: bool = True,
        grazing_angle_gate: bool = False,
        overlap_io: bool = False,
        offline_refine: bool = False,
        report_path: str | os.PathLike[str] | None = None,
        trace_records: list[dict[str, Any]] | None = None,
    ) -> VideoRunReport:
        """Process 100% of frames of `input_path` and optionally write H.264 MP4 to `output_path`."""
        in_path = Path(os.path.abspath(os.fspath(input_path)))
        if not in_path.is_file() or in_path.stat().st_size == 0:
            raise ContractValidationError(f"Input video not found or empty: {in_path}")

        out_path = _validate_safe_output_path(output_path, "output") if output_path is not None else None
        rep_path = _validate_safe_output_path(report_path, "report") if report_path is not None else None

        threads = max(1, int(threads))
        cv2.setNumThreads(threads)

        mesh = resolve_mesh(mesh_spec) if placement.wrap_mode == "mesh_6dof" else None

        # Pre-check metadata for validation before starting wall timer
        probe = probe_video_info(in_path)
        width, height = int(probe["width"]), int(probe["height"])
        source_fps = float(probe["fps"])
        expected_frames = int(probe["frame_count"])

        sel = selection if selection is not None else _default_selection_for_frame(width, height, frame_index=0)
        sel.validate_for_frame(width, height, frame_count=expected_frames)
        select_idx = int(sel.frame_index)

        session = VideoPipelineSession(
            design=design,
            placement=placement,
            tracking_height=tracking_height,
            raster_backend=raster_backend,
            mesh=mesh,
            intrinsics=intrinsics,
            init_pose=init_pose,
            spatial_bucketing=spatial_bucketing,
            adaptive_appearance_gate=adaptive_appearance_gate,
            local_occlusion_mask=local_occlusion_mask,
            sharp_onboarding=sharp_onboarding,
            grazing_angle_gate=grazing_angle_gate,
        )

        # ---- FULL WALL CLOCK: input open through final encoder flush ----
        t_wall_start = time.perf_counter()

        decode_times: list[float] = []
        prep_times: list[float] = []
        track_times: list[float] = []
        comp_times: list[float] = []
        encode_times: list[float] = []
        frame_totals: list[float] = []
        inlier_counts: list[int] = []

        tracked_count = 0
        lost_count = 0
        recovered_count = 0
        had_tracked_before = False
        prev_state: str | None = None
        processed_count = 0
        rendered_count = 0
        output_count = 0
        backend_used: Literal["native_cpp", "opencv"] = (
            "native_cpp" if raster_backend == "native_cpp" else ("opencv" if raster_backend == "opencv" else "native_cpp")
        )

        if offline_refine:
            # Opt-in 2-pass offline refinement (benchmark separately from causal default!)
            report = self._process_offline_refine(
                in_path=in_path,
                out_path=out_path,
                width=width,
                height=height,
                source_fps=source_fps,
                sel=sel,
                session=session,
                threads=threads,
                t_wall_start=t_wall_start,
                trace_records=trace_records,
            )
            if rep_path is not None:
                rep_path.parent.mkdir(parents=True, exist_ok=True)
                rep_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
            return report

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise ContractValidationError(f"Failed to open input video: {in_path}")

        encoder_ctx = (
            _StreamingFFmpegEncoder(out_path, width, height, source_fps, threads=threads)
            if out_path is not None
            else None
        )

        # Optional bounded 3-stage overlap (decode thread -> main process -> encode thread)
        if overlap_io and encoder_ctx is not None:
            decode_q: queue.Queue[tuple[int, np.ndarray, float] | None] = queue.Queue(maxsize=4)
            encode_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=4)
            worker_exc: list[BaseException] = []

            def _decode_worker() -> None:
                idx = 0
                try:
                    while True:
                        td0 = time.perf_counter()
                        ok, frm = cap.read()
                        td_ms = (time.perf_counter() - td0) * 1000.0
                        if not ok or frm is None:
                            break
                        decode_q.put((idx, frm, td_ms))
                        idx += 1
                except BaseException as exc:
                    worker_exc.append(exc)
                finally:
                    decode_q.put(None)

            def _encode_worker(enc: _StreamingFFmpegEncoder) -> None:
                try:
                    while True:
                        item = encode_q.get()
                        if item is None:
                            break
                        enc_ms = enc.write(item)
                        encode_times.append(enc_ms)
                except BaseException as exc:
                    worker_exc.append(exc)

            try:
                with encoder_ctx as enc:
                    t_dec = threading.Thread(target=_decode_worker, daemon=True)
                    t_enc = threading.Thread(target=_encode_worker, args=(enc,), daemon=True)
                    t_dec.start()
                    t_enc.start()

                    while True:
                        item = decode_q.get()
                        if item is None:
                            break
                        f_idx, frame_bgr, dec_ms = item
                        tf0 = time.perf_counter()
                        ts = float(f_idx) / source_fps
                        decode_times.append(dec_ms)

                        if f_idx < select_idx:
                            tr = TrackFrameResult(frame_index=f_idx, timestamp_s=ts, state="lost", inliers=0, confidence=0.0, track_ms=0.0)
                            out_bgr = frame_bgr
                            prep_ms = comp_ms = 0.0
                            rendered_mask = None
                        elif f_idx == select_idx:
                            res_f = session.initialize(frame_bgr, sel, timestamp_s=ts)
                            tr, out_bgr, rendered_mask = res_f.track_result, res_f.frame_bgr, res_f.rendered_mask
                            prep_ms, comp_ms = res_f.prep_ms, res_f.composite_ms
                            if session.compositor is not None:
                                backend_used = session.compositor.backend
                        else:
                            res_f = session.process_frame(frame_bgr, f_idx, ts)
                            tr, out_bgr, rendered_mask = res_f.track_result, res_f.frame_bgr, res_f.rendered_mask
                            prep_ms, comp_ms = res_f.prep_ms, res_f.composite_ms

                        prep_times.append(prep_ms)
                        track_times.append(tr.track_ms)
                        comp_times.append(comp_ms)
                        inlier_counts.append(tr.inliers)

                        if tr.state == "tracking":
                            tracked_count += 1
                            if had_tracked_before and prev_state in ("lost", "recovering"):
                                recovered_count += 1
                            had_tracked_before = True
                        elif tr.state == "lost":
                            lost_count += 1
                        prev_state = tr.state

                        if trace_records is not None:
                            self._append_trace(trace_records, f_idx, ts, tr, session, rendered_mask, dec_ms, prep_ms, comp_ms, 0.0)

                        encode_q.put(out_bgr)
                        processed_count += 1
                        rendered_count += 1
                        frame_totals.append((time.perf_counter() - tf0) * 1000.0 + dec_ms)

                    encode_q.put(None)
                    t_dec.join()
                    t_enc.join()
                    output_count = enc.frames_written
                    if worker_exc:
                        raise worker_exc[0]
            finally:
                cap.release()
        else:
            # Strict single-thread sequential baseline
            try:
                if encoder_ctx is not None:
                    encoder_ctx.__enter__()
                f_idx = 0
                while True:
                    tf0 = time.perf_counter()
                    td0 = time.perf_counter()
                    ok, frame_bgr = cap.read()
                    dec_ms = (time.perf_counter() - td0) * 1000.0
                    if not ok or frame_bgr is None:
                        break
                    ts = float(f_idx) / source_fps
                    decode_times.append(dec_ms)

                    if f_idx < select_idx:
                        tr = TrackFrameResult(frame_index=f_idx, timestamp_s=ts, state="lost", inliers=0, confidence=0.0, track_ms=0.0)
                        out_bgr = frame_bgr
                        prep_ms = comp_ms = 0.0
                        rendered_mask = None
                    elif f_idx == select_idx:
                        res_f = session.initialize(frame_bgr, sel, timestamp_s=ts)
                        tr, out_bgr, rendered_mask = res_f.track_result, res_f.frame_bgr, res_f.rendered_mask
                        prep_ms, comp_ms = res_f.prep_ms, res_f.composite_ms
                        if session.compositor is not None:
                            backend_used = session.compositor.backend
                    else:
                        res_f = session.process_frame(frame_bgr, f_idx, ts)
                        tr, out_bgr, rendered_mask = res_f.track_result, res_f.frame_bgr, res_f.rendered_mask
                        prep_ms, comp_ms = res_f.prep_ms, res_f.composite_ms

                    enc_ms = encoder_ctx.write(out_bgr) if encoder_ctx is not None else 0.0
                    encode_times.append(enc_ms)
                    prep_times.append(prep_ms)
                    track_times.append(tr.track_ms)
                    comp_times.append(comp_ms)
                    inlier_counts.append(tr.inliers)

                    if tr.state == "tracking":
                        tracked_count += 1
                        if had_tracked_before and prev_state in ("lost", "recovering"):
                            recovered_count += 1
                        had_tracked_before = True
                    elif tr.state == "lost":
                        lost_count += 1
                    prev_state = tr.state

                    if trace_records is not None:
                        self._append_trace(trace_records, f_idx, ts, tr, session, rendered_mask, dec_ms, prep_ms, comp_ms, enc_ms)

                    processed_count += 1
                    rendered_count += 1
                    frame_totals.append((time.perf_counter() - tf0) * 1000.0)
                    f_idx += 1

                if encoder_ctx is not None:
                    output_count = encoder_ctx.frames_written
                    encoder_ctx.__exit__(None, None, None)
            except BaseException as exc:
                if encoder_ctx is not None:
                    encoder_ctx.__exit__(type(exc), exc, exc.__traceback__)
                raise
            finally:
                cap.release()

        wall_time_s = max(1e-6, time.perf_counter() - t_wall_start)
        effective_fps = float(processed_count) / wall_time_s
        steady_slice = frame_totals[3:] if len(frame_totals) > 6 else frame_totals
        steady_mean_ms = float(np.mean(steady_slice)) if steady_slice else 0.0
        steady_state_fps = (1000.0 / steady_mean_ms) if steady_mean_ms > 1e-6 else effective_fps

        report = VideoRunReport(
            input_path=str(in_path),
            output_path=str(out_path) if out_path is not None else None,
            width=width,
            height=height,
            source_fps=source_fps,
            wrap_mode=placement.wrap_mode,
            blend_mode=placement.blend_mode,
            raster_backend=backend_used,
            render_stride=1,
            input_frame_count=processed_count,
            processed_frame_count=processed_count,
            rendered_frame_count=rendered_count,
            output_frame_count=output_count if out_path is not None else processed_count,
            tracked_frame_count=tracked_count,
            lost_frame_count=lost_count,
            recovered_count=recovered_count,
            tracking_availability=float(tracked_count) / float(max(processed_count, 1)),
            wall_time_seconds=round(wall_time_s, 6),
            effective_fps=round(effective_fps, 4),
            steady_state_fps=round(steady_state_fps, 4),
            frame_latency_ms=_percentiles(frame_totals),
            stage_latency_mean_ms={
                "decode_ms": round(float(np.mean(decode_times)) if decode_times else 0.0, 4),
                "prep_ms": round(float(np.mean(prep_times)) if prep_times else 0.0, 4),
                "track_ms": round(float(np.mean(track_times)) if track_times else 0.0, 4),
                "composite_ms": round(float(np.mean(comp_times)) if comp_times else 0.0, 4),
                "encode_ms": round(float(np.mean(encode_times)) if encode_times else 0.0, 4),
            },
            inlier_count_median=round(float(np.median(inlier_counts)) if inlier_counts else 0.0, 2),
            reprojection_rmse_px=round(session.tracker.reprojection_rmse_px, 4) if session.tracker.reprojection_rmse_px is not None else None,
        )

        if rep_path is not None:
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            rep_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

        return report

    def _process_offline_refine(
        self,
        *,
        in_path: Path,
        out_path: Path | None,
        width: int,
        height: int,
        source_fps: float,
        sel: SelectionSpec,
        session: VideoPipelineSession,
        threads: int,
        t_wall_start: float,
        trace_records: list[dict[str, Any]] | None,
    ) -> VideoRunReport:
        """Opt-in 2-pass offline mode: bidirectional track from select_frame + zero-phase smoothing.

        Preserves strict overlay suppression across all lost/occluded frames!
        """
        cap = cv2.VideoCapture(str(in_path))
        frames: list[np.ndarray] = []
        decode_times: list[float] = []
        try:
            while True:
                td0 = time.perf_counter()
                ok, frm = cap.read()
                if not ok or frm is None:
                    break
                decode_times.append((time.perf_counter() - td0) * 1000.0)
                frames.append(frm)
        finally:
            cap.release()

        n = len(frames)
        select_idx = int(sel.frame_index)
        track_results: list[TrackFrameResult | None] = [None] * n
        prep_times: list[float] = [0.0] * n

        # Initialize on select_idx
        t_p0 = time.perf_counter()
        seg = segment_frame(frames[select_idx], sel, sharp_onboarding=session.sharp_onboarding)
        prep_times[select_idx] = (time.perf_counter() - t_p0) * 1000.0
        tr0 = session.tracker.initialize(frames[select_idx], seg)
        tr0.timestamp_s = float(select_idx) / source_fps
        track_results[select_idx] = tr0

        compositor = SurfaceCompositor(
            session.artwork_rgba,
            session.placement,
            ref_bbox=session.tracker.ref_bbox,
            ref_mask=session.tracker.ref_mask,
            ref_frame_bgr=frames[select_idx],
            mesh=session.tracker.mesh,
            intrinsics=session.intrinsics,
            raster_backend=session.raster_backend,
            local_occlusion_mask=session.local_occlusion_mask,
        )
        session.compositor = compositor

        # Forward pass (select_idx + 1 .. n - 1)
        for i in range(select_idx + 1, n):
            track_results[i] = session.tracker.track_frame(frames[i], i, float(i) / source_fps)

        # Backward pass (select_idx - 1 .. 0) if select_idx > 0
        if select_idx > 0:
            back_tracker = VideoSurfaceTracker(
                wrap_mode=session.placement.wrap_mode,
                tracking_height=session.tracker.tracking_height,
                max_klt_points=session.tracker.max_klt_points,
                spatial_bucketing=session.tracker.spatial_bucketing,
                adaptive_appearance_gate=session.tracker.adaptive_appearance_gate,
                grazing_angle_gate=session.tracker.grazing_angle_gate,
                mesh=session.tracker.mesh,
                intrinsics=session.intrinsics,
                init_pose=session.tracker.init_pose_control,
            )
            back_tracker.initialize(frames[select_idx], seg)
            for i in range(select_idx - 1, -1, -1):
                track_results[i] = back_tracker.track_frame(frames[i], i, float(i) / source_fps)

        final_tracks: list[TrackFrameResult] = [t for t in track_results if t is not None]
        # Zero-phase smoothing ONLY inside contiguous tracking runs (occluded/lost stay suppressed!)
        final_tracks = _smooth_homographies_contiguous(final_tracks, alpha=0.35)

        comp_times: list[float] = []
        encode_times: list[float] = []
        frame_totals: list[float] = []
        inlier_counts: list[int] = []
        tracked_count = lost_count = recovered_count = 0
        had_tracked_before = False
        prev_state: str | None = None

        encoder_ctx = (
            _StreamingFFmpegEncoder(out_path, width, height, source_fps, threads=threads)
            if out_path is not None
            else None
        )
        output_count = 0
        if encoder_ctx is not None:
            encoder_ctx.__enter__()
        try:
            for i in range(n):
                tr = final_tracks[i]
                out_bgr, mask, c_ms = compositor.composite(frames[i], tr)
                e_ms = encoder_ctx.write(out_bgr) if encoder_ctx is not None else 0.0
                comp_times.append(c_ms)
                encode_times.append(e_ms)
                inlier_counts.append(tr.inliers)
                frame_totals.append(decode_times[i] + prep_times[i] + tr.track_ms + c_ms + e_ms)
                if tr.state == "tracking":
                    tracked_count += 1
                    if had_tracked_before and prev_state in ("lost", "recovering"):
                        recovered_count += 1
                    had_tracked_before = True
                elif tr.state == "lost":
                    lost_count += 1
                prev_state = tr.state
                if trace_records is not None:
                    self._append_trace(trace_records, i, tr.timestamp_s, tr, session, mask, decode_times[i], prep_times[i], c_ms, e_ms)
            if encoder_ctx is not None:
                output_count = encoder_ctx.frames_written
                encoder_ctx.__exit__(None, None, None)
        except BaseException as exc:
            if encoder_ctx is not None:
                encoder_ctx.__exit__(type(exc), exc, exc.__traceback__)
            raise

        wall_time_s = max(1e-6, time.perf_counter() - t_wall_start)
        effective_fps = float(n) / wall_time_s
        steady_slice = frame_totals[3:] if len(frame_totals) > 6 else frame_totals
        steady_mean_ms = float(np.mean(steady_slice)) if steady_slice else 0.0
        steady_state_fps = (1000.0 / steady_mean_ms) if steady_mean_ms > 1e-6 else effective_fps

        return VideoRunReport(
            input_path=str(in_path),
            output_path=str(out_path) if out_path is not None else None,
            width=width,
            height=height,
            source_fps=source_fps,
            wrap_mode=session.placement.wrap_mode,
            blend_mode=session.placement.blend_mode,
            raster_backend=compositor.backend,
            render_stride=1,
            input_frame_count=n,
            processed_frame_count=n,
            rendered_frame_count=n,
            output_frame_count=output_count if out_path is not None else n,
            tracked_frame_count=tracked_count,
            lost_frame_count=lost_count,
            recovered_count=recovered_count,
            tracking_availability=float(tracked_count) / float(max(n, 1)),
            wall_time_seconds=round(wall_time_s, 6),
            effective_fps=round(effective_fps, 4),
            steady_state_fps=round(steady_state_fps, 4),
            frame_latency_ms=_percentiles(frame_totals),
            stage_latency_mean_ms={
                "decode_ms": round(float(np.mean(decode_times)) if decode_times else 0.0, 4),
                "prep_ms": round(float(np.mean(prep_times)) if prep_times else 0.0, 4),
                "track_ms": round(float(np.mean([t.track_ms for t in final_tracks])) if final_tracks else 0.0, 4),
                "composite_ms": round(float(np.mean(comp_times)) if comp_times else 0.0, 4),
                "encode_ms": round(float(np.mean(encode_times)) if encode_times else 0.0, 4),
            },
            inlier_count_median=round(float(np.median(inlier_counts)) if inlier_counts else 0.0, 2),
            reprojection_rmse_px=round(session.tracker.reprojection_rmse_px, 4) if session.tracker.reprojection_rmse_px is not None else None,
        )

    @staticmethod
    def _append_trace(
        trace_records: list[dict[str, Any]],
        f_idx: int,
        ts: float,
        tr: TrackFrameResult,
        session: VideoPipelineSession,
        rendered_mask: np.ndarray | None,
        dec_ms: float,
        prep_ms: float,
        comp_ms: float,
        enc_ms: float,
    ) -> None:
        warped_quad = None
        if tr.state == "tracking" and tr.homography_3x3 is not None:
            q = cv2.perspectiveTransform(session.tracker.ref_quad.reshape(1, 4, 2), tr.homography_3x3).reshape(4, 2)
            warped_quad = q.round(3).tolist()
        rec: dict[str, Any] = {
            "frame_index": f_idx,
            "timestamp_s": round(ts, 6),
            "state": tr.state,
            "inliers": tr.inliers,
            "confidence": round(tr.confidence, 4),
            "warped_quad": warped_quad,
            "pose_rvec": tr.pose_rvec.round(5).tolist() if tr.pose_rvec is not None else None,
            "pose_tvec": tr.pose_tvec.round(5).tolist() if tr.pose_tvec is not None else None,
            "rendered_pixels": int(np.count_nonzero(rendered_mask)) if rendered_mask is not None else 0,
            "decode_ms": round(dec_ms, 3),
            "prep_ms": round(prep_ms, 3),
            "track_ms": round(tr.track_ms, 3),
            "composite_ms": round(comp_ms, 3),
            "encode_ms": round(enc_ms, 3),
        }
        trace_records.append(rec)


def replay_live_30hz(
    *,
    input_path: str | os.PathLike[str],
    design: DesignSpec,
    placement: PlacementSpec,
    selection: SelectionSpec,
    target_hz: float = 30.0,
    max_frames: int = 90,
    tracking_height: int = 360,
    threads: int = 2,
    raster_backend: Literal["auto", "native_cpp", "opencv"] = "auto",
    spatial_bucketing: bool = True,
    adaptive_appearance_gate: bool = True,
    local_occlusion_mask: bool = True,
    sharp_onboarding: bool = True,
    grazing_angle_gate: bool = False,
    mesh_spec: str | Path | MeshGeometry | None = None,
    intrinsics: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Replay pre-decoded timestamped frames at exact `target_hz` (30 Hz) through the shared causal core.

    Measures real arrival-to-overlay-ready latency (p50/p95/max), deadline misses (> 1000/target_hz ms),
    and pending queue buildup.
    """
    cv2.setNumThreads(max(1, int(threads)))
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ContractValidationError(f"Unable to open video for live replay: {input_path}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frm = cap.read()
            if not ok or frm is None:
                break
            frames.append(frm)
    finally:
        cap.release()

    if len(frames) < 2:
        raise ContractValidationError("Need at least 2 frames for 30 Hz live replay")

    mesh = resolve_mesh(mesh_spec) if placement.wrap_mode == "mesh_6dof" else None
    session = VideoPipelineSession(
        design=design,
        placement=placement,
        tracking_height=tracking_height,
        raster_backend=raster_backend,
        mesh=mesh,
        intrinsics=intrinsics,
        spatial_bucketing=spatial_bucketing,
        adaptive_appearance_gate=adaptive_appearance_gate,
        local_occlusion_mask=local_occlusion_mask,
        sharp_onboarding=sharp_onboarding,
        grazing_angle_gate=grazing_angle_gate,
    )

    # Frame 0 onboarding occurs once on user tap before live tracking stream begins
    init_out = session.initialize(frames[0], selection, timestamp_s=0.0)

    period_s = 1.0 / float(target_hz)
    deadline_ms = period_s * 1000.0
    t_stream_start = time.perf_counter()

    arrival_to_ready_ms: list[float] = []
    compute_only_ms: list[float] = []
    deadline_misses = 0
    max_queue_depth = 0

    for i in range(1, len(frames)):
        arrival_t = t_stream_start + float(i - 1) * period_s
        now = time.perf_counter()
        if now < arrival_t:
            time.sleep(arrival_t - now)
            queue_depth = 0
        else:
            # How many newer frames have already arrived while we were busy?
            queue_depth = int(math.floor((now - arrival_t) / period_s))
        if queue_depth > max_queue_depth:
            max_queue_depth = queue_depth

        t_proc0 = time.perf_counter()
        _ = session.process_frame(frames[i], i, float(i) * period_s)
        t_ready = time.perf_counter()

        comp_ms = (t_ready - t_proc0) * 1000.0
        e2e_ms = (t_ready - arrival_t) * 1000.0
        compute_only_ms.append(comp_ms)
        arrival_to_ready_ms.append(e2e_ms)
        if e2e_ms > deadline_ms:
            deadline_misses += 1

    n_replayed = len(arrival_to_ready_ms)
    return {
        "target_hz": float(target_hz),
        "deadline_ms": round(deadline_ms, 3),
        "frames_replayed": n_replayed,
        "init_ms": round(init_out.prep_ms + init_out.track_ms + init_out.composite_ms, 3),
        "arrival_to_overlay_ready_ms": _percentiles(arrival_to_ready_ms),
        "compute_only_ms": _percentiles(compute_only_ms),
        "deadline_misses": int(deadline_misses),
        "deadline_miss_rate": round(float(deadline_misses) / float(max(n_replayed, 1)), 4),
        "max_queue_depth": int(max_queue_depth),
        "kept_up_live": bool(deadline_misses == 0 and max_queue_depth == 0),
    }
