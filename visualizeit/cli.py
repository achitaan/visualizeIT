"""Command-line interface for visualizeIT (`python -m visualizeit` / `visualizeit`)."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import cv2
import numpy as np

from .contracts import (
    ContractValidationError,
    DesignSpec,
    PlacementSpec,
    SelectionSpec,
    VALID_BLEND_MODES,
    VALID_PATTERNS,
    VALID_WRAP_MODES,
)
from .demo_assets import DEFAULT_DEMO_DIR, generate_demo_corpus, load_demo_manifest
from .designs import parse_design_prompt
from .pipeline import VideoPipeline, probe_video_info
from .segmentation import segment_frame


class _CleanArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        print(f"error: {message}", file=sys.stderr)
        raise SystemExit(2)


def _parse_point(raw: str) -> tuple[float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ContractValidationError(f"Point must have format X,Y; got {raw!r}")
    try:
        x, y = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ContractValidationError(f"Point coordinates must be numeric: {raw!r}") from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ContractValidationError(f"Point coordinates must be finite: {raw!r}")
    return (x, y)


def _parse_box(raw: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ContractValidationError(f"Box must have format X,Y,W,H; got {raw!r}")
    try:
        vals = [float(p) for p in parts]
    except ValueError as exc:
        raise ContractValidationError(f"Box values must be numeric: {raw!r}") from exc
    if not all(math.isfinite(v) for v in vals):
        raise ContractValidationError(f"Box values must be finite: {raw!r}")
    return (vals[0], vals[1], vals[2], vals[3])


def _parse_intrinsics(raw: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ContractValidationError(f"Intrinsics must have format fx,fy,cx,cy; got {raw!r}")
    try:
        fx, fy, cx, cy = [float(p) for p in parts]
    except ValueError as exc:
        raise ContractValidationError(f"Intrinsics must be numeric: {raw!r}") from exc
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy)):
        raise ContractValidationError(f"Intrinsics must be finite: {raw!r}")
    if fx <= 0.0 or fy <= 0.0:
        raise ContractValidationError(f"Focal lengths fx, fy must be > 0, got ({fx}, {fy})")
    return (fx, fy, cx, cy)


def _parse_init_pose(raw: str) -> tuple[np.ndarray, np.ndarray]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 6:
        raise ContractValidationError(f"init-pose must have format rx,ry,rz,tx,ty,tz; got {raw!r}")
    try:
        vals = [float(p) for p in parts]
    except ValueError as exc:
        raise ContractValidationError(f"init-pose values must be numeric: {raw!r}") from exc
    if not all(math.isfinite(v) for v in vals) or vals[5] <= 0.03:
        raise ContractValidationError(f"init-pose must be finite with tz > 0.03 m: {raw!r}")
    return np.array(vals[:3], dtype=np.float64), np.array(vals[3:], dtype=np.float64)


def _build_design_spec(args: argparse.Namespace) -> DesignSpec:
    if args.design_image:
        return DesignSpec(source="image", image_path=str(args.design_image), colors=[])

    if args.prompt:
        spec = parse_design_prompt(args.prompt)
        # Allow explicit overrides on top of prompt
        pattern = args.pattern if args.pattern is not None else spec.pattern
        colors = (
            [c.strip() for c in args.colors.split(",") if c.strip()]
            if args.colors is not None
            else list(spec.colors)
        )
        text = args.text if args.text is not None else spec.text
        return DesignSpec(
            source="recipe",
            pattern=pattern,
            colors=colors,
            text=text,
            prompt=args.prompt,
            width=spec.width,
            height=spec.height,
            seed=spec.seed,
        )

    pattern = args.pattern or "stripes"
    colors = (
        [c.strip() for c in args.colors.split(",") if c.strip()]
        if args.colors is not None
        else ["#ff595e", "#1982c4", "#ffca3a"]
    )
    return DesignSpec(
        source="recipe",
        pattern=pattern,
        colors=colors,
        text=args.text or "",
        prompt="",
    )


def _build_selection_spec(
    frame_index: int,
    points: list[str] | None,
    neg_points: list[str] | None,
    box: str | None,
) -> SelectionSpec | None:
    if not points and not neg_points and box is None:
        return None
    pos = [_parse_point(p) for p in (points or [])]
    neg = [_parse_point(p) for p in (neg_points or [])]
    bbox = _parse_box(box) if box is not None else None
    return SelectionSpec(
        frame_index=frame_index,
        positive_points=pos,
        negative_points=neg,
        bbox=bbox,
    )


def _cmd_demo_assets(args: argparse.Namespace) -> int:
    clips = generate_demo_corpus(args.output_dir)
    payload = {"clips": [c.to_dict() for c in clips]}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Generated {len(clips)} demo clips in {args.output_dir}")
        for c in clips:
            print(f"  - {c.id}: {c.width}x{c.height} @ {c.fps:.1f} FPS ({c.frame_count} frames) [{c.recommended_wrap}]")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    info = probe_video_info(args.input)
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(
            f"{info['path']}: {info['width']}x{info['height']} @ {info['fps']} FPS, "
            f"{info['frame_count']} frames ({info['duration_s']}s), codec={info['codec']}"
        )
    return 0


def _cmd_segment(args: argparse.Namespace) -> int:
    info = probe_video_info(args.input)
    frame_idx = int(args.frame)
    total_frames = int(info["frame_count"])
    if frame_idx < 0 or frame_idx >= total_frames:
        raise ContractValidationError(
            f"--frame {frame_idx} out of range for video with {total_frames} frames"
        )

    sel = _build_selection_spec(frame_idx, args.point, args.neg_point, args.box)
    if sel is None:
        raise ContractValidationError("segment requires at least one --point or --box selection")

    cap = cv2.VideoCapture(str(info["path"]))
    if not cap.isOpened():
        raise ContractValidationError(f"Unable to open video: {info['path']}")
    frame_bgr: np.ndarray | None = None
    try:
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frm = cap.read()
        if not ok or frm is None:
            # Sequential fallback seek
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
            for i in range(frame_idx + 1):
                ok, frm = cap.read()
                if not ok:
                    break
        if ok and frm is not None:
            frame_bgr = frm
    finally:
        cap.release()

    if frame_bgr is None:
        raise ContractValidationError(f"Failed to decode frame {frame_idx} from {info['path']}")

    res = segment_frame(frame_bgr, sel, output_mask_path=args.output_mask)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(
            f"Segmented frame {res.frame_index} ({res.width}x{res.height}): "
            f"{res.foreground_pixels} px ({res.coverage_ratio * 100:.1f}%), "
            f"bbox={res.bbox}, anchors={res.anchor_count} in {res.init_ms:.1f} ms"
        )
    return 0


def _cmd_process(args: argparse.Namespace) -> int:
    if args.threads < 1:
        raise ContractValidationError(f"--threads must be >= 1, got {args.threads}")
    if args.tracking_height not in (360, 480):
        raise ContractValidationError(
            f"--tracking-height must be 360 or 480, got {args.tracking_height}"
        )
    if args.select_frame < 0:
        raise ContractValidationError(f"--select-frame must be >= 0, got {args.select_frame}")

    design = _build_design_spec(args)
    placement = PlacementSpec(
        wrap_mode=args.wrap_mode,
        blend_mode=args.blend_mode,
        opacity=args.opacity,
        scale=args.scale,
        rotation_deg=args.rotation_deg,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        cylinder_radius_frac=args.cylinder_radius_frac,
        tile_repeat=bool(args.tile_repeat),
    )
    sel = _build_selection_spec(args.select_frame, args.point, args.neg_point, args.box)
    intrinsics = _parse_intrinsics(args.intrinsics) if args.intrinsics else None
    init_pose = _parse_init_pose(args.init_pose) if args.init_pose else None

    pipeline = VideoPipeline()
    report = pipeline.process_video(
        input_path=args.input,
        output_path=args.output,
        design=design,
        placement=placement,
        selection=sel,
        mesh_spec=args.mesh,
        intrinsics=intrinsics,
        init_pose=init_pose,
        tracking_height=args.tracking_height,
        threads=args.threads,
        raster_backend=args.raster_backend,
        spatial_bucketing=bool(args.spatial_bucketing),
        adaptive_appearance_gate=bool(args.adaptive_appearance_gate),
        local_occlusion_mask=bool(args.local_occlusion_mask),
        sharp_onboarding=bool(args.sharp_onboarding),
        grazing_angle_gate=bool(args.grazing_angle_gate),
        overlap_io=bool(args.overlap_io),
        offline_refine=bool(args.offline_refine),
        report_path=args.report,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(
            f"Processed {report.processed_frame_count} frames ({report.width}x{report.height}) "
            f"in {report.wall_time_seconds:.2f}s -> {report.effective_fps:.2f} FPS "
            f"(tracking={report.tracking_availability * 100:.1f}%, lost={report.lost_frame_count}, "
            f"recovered={report.recovered_count})"
        )
    return 0


def _ensure_demo_manifest() -> dict[str, Any]:
    if not (DEFAULT_DEMO_DIR / "demo_manifest.json").is_file():
        generate_demo_corpus(DEFAULT_DEMO_DIR)
    clips = load_demo_manifest(DEFAULT_DEMO_DIR)
    return {c.id: c for c in clips}


def _run_benchmark_scenario(
    *,
    name: str,
    input_path: Path,
    output_dir: Path,
    wrap_mode: str,
    selection: SelectionSpec | None,
    mesh_spec: str | Path | None = None,
    intrinsics: tuple[float, float, float, float] | None = None,
    threads: int = 2,
    spatial_bucketing: bool = False,
    overlap_io: bool = False,
    offline_refine: bool = False,
) -> dict[str, Any]:
    out_mp4 = output_dir / f"bench_{name}.mp4"
    design = DesignSpec(
        source="recipe",
        pattern="stripes" if wrap_mode != "mesh_6dof" else "checker",
        colors=["#ff595e", "#1982c4", "#ffca3a"],
        text="BENCH",
    )
    placement = PlacementSpec(
        wrap_mode=wrap_mode,  # type: ignore[arg-type]
        blend_mode="luminance",
        opacity=0.90,
    )
    pipeline = VideoPipeline()
    rep = pipeline.process_video(
        input_path=input_path,
        output_path=out_mp4,
        design=design,
        placement=placement,
        selection=selection,
        mesh_spec=mesh_spec,
        intrinsics=intrinsics,
        tracking_height=360,
        threads=threads,
        raster_backend="auto",
        spatial_bucketing=spatial_bucketing,
        overlap_io=overlap_io,
        offline_refine=offline_refine,
    )
    return rep.to_dict()


def _cmd_benchmark(args: argparse.Namespace) -> int:
    min_fps = float(args.min_fps)
    if not math.isfinite(min_fps) or min_fps <= 0.0:
        raise ContractValidationError(f"--min-fps must be a finite number > 0, got {args.min_fps!r}")
    if args.threads < 1:
        raise ContractValidationError(f"--threads must be >= 1, got {args.threads}")

    out_dir = Path(os.path.abspath(args.output_dir)) if args.output_dir else (DEFAULT_DEMO_DIR.parent / "tmp" / "benchmarks")
    posix = out_dir.as_posix()
    if posix == "/tmp" or posix.startswith("/tmp/"):
        raise ContractValidationError(f"Refusing to use /tmp for benchmark --output-dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        info = probe_video_info(args.input)
        # Try matching demo manifest selection if input is a demo clip
        sel: SelectionSpec | None = None
        wrap = "planar"
        try:
            demos = _ensure_demo_manifest()
            in_resolved = Path(info["path"]).resolve()
            for clip in demos.values():
                if Path(clip.path).resolve() == in_resolved:
                    sel = clip.default_selection
                    wrap = clip.recommended_wrap
                    break
        except Exception:
            pass
        rep = _run_benchmark_scenario(
            name=Path(info["path"]).stem,
            input_path=Path(info["path"]),
            output_dir=out_dir,
            wrap_mode=wrap,
            selection=sel,
            threads=args.threads,
            spatial_bucketing=bool(args.spatial_bucketing),
            overlap_io=bool(args.overlap_io),
            offline_refine=bool(args.offline_refine),
        )
        if args.report:
            rep_p = Path(os.path.abspath(args.report))
            rep_p.parent.mkdir(parents=True, exist_ok=True)
            rep_p.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(rep, indent=2))
        else:
            print(f"Benchmark {rep['input_path']}: {rep['effective_fps']:.2f} FPS (min_fps={min_fps:.2f})")
        return 0 if float(rep["effective_fps"]) >= min_fps else 1

    demos = _ensure_demo_manifest()
    mesh_meta_path = DEFAULT_DEMO_DIR / "mesh_box_meta.json"
    mesh_intr: tuple[float, float, float, float] | None = None
    if mesh_meta_path.is_file():
        m = json.loads(mesh_meta_path.read_text(encoding="utf-8"))
        intr_raw = m.get("intrinsics")
        if isinstance(intr_raw, list) and len(intr_raw) == 4:
            mesh_intr = (float(intr_raw[0]), float(intr_raw[1]), float(intr_raw[2]), float(intr_raw[3]))

    preset_map: dict[str, tuple[str, str, Path | None, tuple[float, float, float, float] | None]] = {
        "quick": ("planar_card_640x480", "planar", None, None),
        "standard": ("planar_card_640x480", "planar", None, None),
        "hd": ("planar_hd_1280x720", "planar", None, None),
        "cylinder": ("cylinder_mug_640x480", "cylinder", None, None),
        "occlusion": ("occlusion_recovery_640x480", "planar", None, None),
        "mesh_6dof": ("mesh_box_640x480", "mesh_6dof", DEFAULT_DEMO_DIR / "mesh_box.ply", mesh_intr),
    }

    if args.preset == "all":
        scenarios = []
        all_passed = True
        for pname in ("standard", "hd", "cylinder", "occlusion", "mesh_6dof"):
            cid, wrap, mpath, intr = preset_map[pname]
            clip = demos[cid]
            rep = _run_benchmark_scenario(
                name=pname,
                input_path=Path(clip.path),
                output_dir=out_dir,
                wrap_mode=wrap,
                selection=clip.default_selection,
                mesh_spec=mpath,
                intrinsics=intr,
                threads=args.threads,
                spatial_bucketing=bool(args.spatial_bucketing),
                overlap_io=bool(args.overlap_io),
                offline_refine=bool(args.offline_refine),
            )
            scenarios.append(rep)
            if float(rep["effective_fps"]) < min_fps:
                all_passed = False
        payload = {
            "preset": "all",
            "min_fps": min_fps,
            "passed": all_passed,
            "scenarios": scenarios,
        }
        if args.report:
            rep_p = Path(os.path.abspath(args.report))
            rep_p.parent.mkdir(parents=True, exist_ok=True)
            rep_p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for r in scenarios:
                print(f"  {r['wrap_mode']:10s} {r['width']}x{r['height']}: {r['effective_fps']:.2f} FPS")
        return 0 if all_passed else 1

    if args.preset not in preset_map:
        raise ContractValidationError(f"Unknown benchmark preset: {args.preset!r}")

    cid, wrap, mpath, intr = preset_map[args.preset]
    clip = demos[cid]
    rep = _run_benchmark_scenario(
        name=args.preset,
        input_path=Path(clip.path),
        output_dir=out_dir,
        wrap_mode=wrap,
        selection=clip.default_selection,
        mesh_spec=mpath,
        intrinsics=intr,
        threads=args.threads,
        spatial_bucketing=bool(args.spatial_bucketing),
        overlap_io=bool(args.overlap_io),
        offline_refine=bool(args.offline_refine),
    )
    if args.report:
        rep_p = Path(os.path.abspath(args.report))
        rep_p.parent.mkdir(parents=True, exist_ok=True)
        rep_p.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"Benchmark [{args.preset}]: {rep['effective_fps']:.2f} FPS (min_fps={min_fps:.2f})")
    return 0 if float(rep["effective_fps"]) >= min_fps else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    port = int(args.port)
    if port < 1 or port > 65535:
        raise ContractValidationError(f"--port must be in [1, 65535], got {port}")
    from .server import run_server

    return run_server(host=str(args.host), port=port, storage_dir=str(args.storage_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = _CleanArgumentParser(
        prog="visualizeit",
        description="visualizeIT — Real-Time Causal Surface Design & Video Engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # demo-assets
    p_demo = sub.add_parser("demo-assets", help="Generate deterministic multi-scenario demo videos and 3D PLY")
    p_demo.add_argument("--output-dir", default=str(DEFAULT_DEMO_DIR), help="Target directory")
    p_demo.add_argument("--json", action="store_true", help="Print JSON manifest to stdout")
    p_demo.set_defaults(func=_cmd_demo_assets)

    # info
    p_info = sub.add_parser("info", help="Probe video container metadata")
    p_info.add_argument("--input", required=True, help="Input video file path")
    p_info.add_argument("--json", action="store_true", help="Print JSON metadata to stdout")
    p_info.set_defaults(func=_cmd_info)

    # segment
    p_seg = sub.add_parser("segment", help="Interactive foreground segmentation onboarding on one frame")
    p_seg.add_argument("--input", required=True, help="Input video file path")
    p_seg.add_argument("--frame", type=int, default=0, help="Frame index to segment (default: 0)")
    p_seg.add_argument("--point", action="append", default=[], help="Positive foreground point X,Y (repeatable)")
    p_seg.add_argument("--neg-point", action="append", default=[], help="Negative background point X,Y (repeatable)")
    p_seg.add_argument("--box", default=None, help="Bounding box X,Y,W,H")
    p_seg.add_argument("--output-mask", default=None, help="Optional PNG path to save foreground mask")
    p_seg.add_argument("--json", action="store_true", help="Print JSON result to stdout")
    p_seg.set_defaults(func=_cmd_segment)

    # process
    p_proc = sub.add_parser("process", help="Process 100%% of frames of a video with artwork mapping")
    p_proc.add_argument("--input", required=True, help="Input video file path")
    p_proc.add_argument("--output", required=True, help="Output MP4 file path")
    p_proc.add_argument("--report", default=None, help="Optional path to write VideoRunReport JSON")
    p_proc.add_argument("--prompt", default=None, help="Natural language design prompt")
    p_proc.add_argument("--pattern", default=None, choices=VALID_PATTERNS, help="Procedural pattern")
    p_proc.add_argument("--colors", default=None, help="Comma-separated colors (2-5 hex or named)")
    p_proc.add_argument("--text", default=None, help="Optional centered text overlay")
    p_proc.add_argument("--design-image", default=None, help="Custom RGBA/PNG/JPEG/WebP artwork path")
    p_proc.add_argument("--wrap-mode", default="planar", choices=VALID_WRAP_MODES, help="Surface wrap mode")
    p_proc.add_argument("--blend-mode", default="luminance", choices=VALID_BLEND_MODES, help="Blend mode")
    p_proc.add_argument("--opacity", type=float, default=0.90, help="Overlay opacity [0.0, 1.0]")
    p_proc.add_argument("--scale", type=float, default=1.0, help="Artwork scale [0.1, 5.0]")
    p_proc.add_argument("--rotation-deg", type=float, default=0.0, help="Artwork rotation in degrees")
    p_proc.add_argument("--offset-x", type=float, default=0.0, help="Horizontal UV offset")
    p_proc.add_argument("--offset-y", type=float, default=0.0, help="Vertical UV offset")
    p_proc.add_argument("--cylinder-radius-frac", type=float, default=0.5, help="Cylinder curvature factor")
    p_proc.add_argument("--tile-repeat", action="store_true", help="Tile artwork periodically")
    p_proc.add_argument("--select-frame", type=int, default=0, help="Selection frame index")
    p_proc.add_argument("--point", action="append", default=[], help="Positive selection point X,Y")
    p_proc.add_argument("--neg-point", action="append", default=[], help="Negative selection point X,Y")
    p_proc.add_argument("--box", default=None, help="Selection bounding box X,Y,W,H")
    p_proc.add_argument("--mesh", default=None, help="3D .ply mesh path or parametric shape (plane|cylinder|box|curved_panel)")
    p_proc.add_argument("--intrinsics", default=None, help="Camera intrinsics fx,fy,cx,cy")
    p_proc.add_argument("--init-pose", default=None, help="Explicit known-pose control rx,ry,rz,tx,ty,tz (separate control only)")
    p_proc.add_argument("--tracking-height", type=int, default=360, help="Working tracker height (360 or 480)")
    p_proc.add_argument("--threads", type=int, default=2, help="CPU worker threads")
    p_proc.add_argument("--raster-backend", default="auto", choices=("auto", "native_cpp", "opencv"), help="Rasterizer backend")
    p_proc.add_argument("--spatial-bucketing", action=argparse.BooleanOptionalAction, default=True, help="Enable 8x6 fixed-budget spatial feature bucketing (default: on)")
    p_proc.add_argument("--adaptive-appearance-gate", action=argparse.BooleanOptionalAction, default=True, help="Enable robust 4x4 blockwise appearance gating (default: on)")
    p_proc.add_argument("--local-occlusion-mask", action=argparse.BooleanOptionalAction, default=True, help="Enable illumination-invariant per-cell local hand/background occlusion mask (default: on)")
    p_proc.add_argument("--sharp-onboarding", action=argparse.BooleanOptionalAction, default=True, help="Enable area-capped ROI-cropped GrabCut frame-0 onboarding (default: on)")
    p_proc.add_argument("--grazing-angle-gate", action=argparse.BooleanOptionalAction, default=False, help="Enable experimental grazing-angle aspect-pinch suppression (default: off)")
    p_proc.add_argument("--overlap-io", action="store_true", help="Enable bounded decode/encode thread overlap")
    p_proc.add_argument("--offline-refine", action="store_true", help="Opt-in 2-pass offline bidirectional refinement")
    p_proc.add_argument("--json", action="store_true", help="Print VideoRunReport JSON to stdout")
    p_proc.set_defaults(func=_cmd_process)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run honest >=15/30 FPS video benchmark gate")
    p_bench.add_argument("--input", default=None, help="Optional custom video file to benchmark")
    p_bench.add_argument(
        "--preset",
        default="quick",
        choices=("quick", "standard", "hd", "cylinder", "occlusion", "mesh_6dof", "all"),
        help="Benchmark preset",
    )
    p_bench.add_argument("--min-fps", type=float, default=15.0, help="Minimum required processed FPS")
    p_bench.add_argument("--output-dir", default=None, help="Directory for output benchmark MP4s")
    p_bench.add_argument("--report", default=None, help="Optional path to write JSON report")
    p_bench.add_argument("--threads", type=int, default=2, help="CPU worker threads")
    p_bench.add_argument("--spatial-bucketing", action=argparse.BooleanOptionalAction, default=True, help="Enable 8x6 spatial feature bucketing (default: on)")
    p_bench.add_argument("--overlap-io", action="store_true", help="Enable bounded decode/encode thread overlap")
    p_bench.add_argument("--offline-refine", action="store_true", help="Opt-in 2-pass offline refinement")
    p_bench.add_argument("--json", action="store_true", help="Print JSON report to stdout")
    p_bench.set_defaults(func=_cmd_benchmark)

    # serve (deferred M3 placeholder)
    p_serve = sub.add_parser("serve", help="Start FastAPI + Unified Web Studio server (Milestone 3)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8790)
    p_serve.add_argument("--storage-dir", default=".cache/studio-runtime")
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return int(args.func(args))
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
        return code
    except (ContractValidationError, ValueError, FileNotFoundError, PermissionError, IsADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
