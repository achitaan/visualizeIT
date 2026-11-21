"""FastAPI HTTP/1.1 REST, RFC 7233 Range Media, and Real-Time WebSocket Server for visualizeIT."""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
import uuid

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from .contracts import (
    ContractValidationError,
    DesignSpec,
    PlacementSpec,
    SelectionSpec,
    VideoRunReport,
)
from .demo_assets import DEFAULT_DEMO_DIR, generate_demo_corpus, load_demo_manifest
from .designs import load_design_image, parse_design_prompt, render_design
from .native_raster import is_native_raster_available
from .pipeline import (
    REPO_ROOT,
    VideoPipeline,
    VideoPipelineSession,
    _default_selection_for_frame,
    _percentiles,
    probe_video_info,
)
from .segmentation import segment_frame
from .tracker import resolve_mesh

WEB_DIR = Path(__file__).resolve().parent / "web"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

DESIGN_PRESETS: list[dict[str, Any]] = [
    {
        "id": "sunset_stripes",
        "name": "Sunset Stripes",
        "pattern": "stripes",
        "colors": ["#ff595e", "#ffca3a", "#1982c4"],
        "text": "VIZIT",
    },
    {
        "id": "studio_checker",
        "name": "Studio Checker",
        "pattern": "checker",
        "colors": ["#1d3557", "#f1faee", "#e63946"],
        "text": "",
    },
    {
        "id": "ocean_waves",
        "name": "Ocean Waves",
        "pattern": "waves",
        "colors": ["#0077b6", "#00b4d8", "#90e0ef"],
        "text": "WAVE",
    },
    {
        "id": "coral_dots",
        "name": "Coral Polka",
        "pattern": "dots",
        "colors": ["#ff6b6b", "#ffd166", "#073b4c"],
        "text": "",
    },
    {
        "id": "blueprint_grid",
        "name": "Blueprint Grid",
        "pattern": "grid",
        "colors": ["#0f4c5c", "#e36414", "#fb8b24"],
        "text": "GRID",
    },
    {
        "id": "emerald_solid",
        "name": "Emerald Solid",
        "pattern": "solid",
        "colors": ["#2a9d8f", "#264653"],
        "text": "LAB",
    },
]


def _validate_storage_dir(path: str | os.PathLike[str]) -> Path:
    resolved = Path(os.path.abspath(os.fspath(path)))
    posix = resolved.as_posix()
    if posix == "/tmp" or posix.startswith("/tmp/"):
        raise ContractValidationError(f"Refusing to use /tmp for server storage_dir: {resolved}")
    if ".agents/teamwork" in posix:
        raise ContractValidationError(f"Refusing to use .agents/teamwork for storage_dir: {resolved}")
    return resolved


def _validate_id(raw_id: str, kind: str = "resource") -> str:
    if not raw_id or not _SAFE_ID_RE.match(raw_id) or ".." in raw_id or "/" in raw_id or "\\" in raw_id:
        raise HTTPException(status_code=404, detail=f"Invalid {kind} id")
    return raw_id


def _guess_video_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".webm":
        return "video/webm"
    if ext == ".mov":
        return "video/quicktime"
    if ext == ".avi":
        return "video/x-msvideo"
    return "video/mp4"


def _build_range_response(file_path: Path, request: Request, media_type: str) -> Response:
    """Serve full 200 or RFC 7233 206 Partial Content with strict 416/400 boundary handling."""
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Media file not found")
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range") or request.headers.get("Range")

    if not range_header:
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            headers={"Accept-Ranges": "bytes"},
        )

    m = re.match(r"^\s*bytes=(\d*)-(\d*)\s*$", range_header, re.IGNORECASE)
    if not m:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}", "Accept-Ranges": "bytes"},
        )

    start_s, end_s = m.group(1), m.group(2)
    if not start_s and not end_s:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}", "Accept-Ranges": "bytes"},
        )

    if start_s:
        start = int(start_s)
        end = int(end_s) if end_s else (file_size - 1)
    else:
        # Suffix range: bytes=-N
        suffix_len = int(end_s)
        if suffix_len <= 0:
            return Response(
                status_code=416,
                headers={"Content-Range": f"bytes */{file_size}", "Accept-Ranges": "bytes"},
            )
        start = max(0, file_size - suffix_len)
        end = file_size - 1

    if start < 0 or start >= file_size or end < start:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}", "Accept-Ranges": "bytes"},
        )
    end = min(end, file_size - 1)
    chunk_len = end - start + 1

    def _iter_range():
        with file_path.open("rb") as fh:
            fh.seek(start)
            remaining = chunk_len
            while remaining > 0:
                buf = fh.read(min(65536, remaining))
                if not buf:
                    break
                remaining -= len(buf)
                yield buf

    return StreamingResponse(
        _iter_range(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(chunk_len),
            "Accept-Ranges": "bytes",
        },
    )


def _decode_exact_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Failed to open video file")
    try:
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        ok, frm = cap.read()
        if ok and frm is not None:
            return frm
        # Sequential fallback for containers without accurate keyframe seeking
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        for i in range(frame_index + 1):
            ok, frm = cap.read()
            if not ok or frm is None:
                break
            if i == frame_index:
                return frm
    finally:
        cap.release()
    raise HTTPException(status_code=404, detail=f"Frame {frame_index} could not be decoded")


def _parse_design_from_dict(raw: dict[str, Any]) -> DesignSpec:
    if not isinstance(raw, dict):
        raise ContractValidationError("design_recipe must be a JSON object")
    prompt = str(raw.get("prompt") or "").strip()
    width = int(raw.get("width", 512))
    height = int(raw.get("height", 512))
    seed = int(raw.get("seed", 42))

    if "colors" in raw and raw["colors"] is not None:
        if not isinstance(raw["colors"], (list, tuple)):
            raise ContractValidationError("colors must be a list of color strings")
        explicit_colors: list[Any] | None = list(raw["colors"])
    else:
        explicit_colors = None

    if prompt:
        parsed = parse_design_prompt(prompt)
        pattern = str(raw.get("pattern") or parsed.pattern)
        colors = explicit_colors if explicit_colors is not None else list(parsed.colors)
        text = str(raw["text"]) if "text" in raw and raw["text"] is not None else parsed.text
        return DesignSpec(
            source="recipe",
            pattern=pattern,  # type: ignore[arg-type]
            colors=colors,
            text=text,
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
        )

    pattern = str(raw.get("pattern") or "stripes")
    colors = explicit_colors if explicit_colors is not None else ["#ff595e", "#1982c4", "#ffca3a"]
    text = str(raw.get("text") or "")
    return DesignSpec(
        source="recipe",
        pattern=pattern,  # type: ignore[arg-type]
        colors=colors,
        text=text,
        prompt="",
        width=width,
        height=height,
        seed=seed,
    )


def _parse_placement_from_dict(raw: dict[str, Any] | None) -> PlacementSpec:
    if raw is None:
        return PlacementSpec()
    if not isinstance(raw, dict):
        raise ContractValidationError("placement must be a JSON object")
    return PlacementSpec(
        wrap_mode=str(raw.get("wrap_mode", "planar")),  # type: ignore[arg-type]
        blend_mode=str(raw.get("blend_mode", "luminance")),  # type: ignore[arg-type]
        opacity=float(raw.get("opacity", 0.90)),
        scale=float(raw.get("scale", 1.0)),
        rotation_deg=float(raw.get("rotation_deg", 0.0)),
        offset_x=float(raw.get("offset_x", 0.0)),
        offset_y=float(raw.get("offset_y", 0.0)),
        cylinder_radius_frac=float(raw.get("cylinder_radius_frac", 0.5)),
        tile_repeat=bool(raw.get("tile_repeat", False)),
    )


def _parse_selection_from_dict(raw: dict[str, Any]) -> SelectionSpec:
    if not isinstance(raw, dict):
        raise ContractValidationError("selection must be a JSON object")
    return SelectionSpec.from_dict(raw)


@dataclass
class _VideoRecord:
    video_id: str
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str
    is_demo: bool = False
    default_selection: dict[str, Any] | None = None
    recommended_wrap: str = "planar"
    mesh_path: Path | None = None
    intrinsics: tuple[float, float, float, float] | None = None

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "codec": self.codec,
            "media_url": f"/api/videos/{self.video_id}/media",
        }


@dataclass
class _DesignRecord:
    design_id: str
    image_path: Path
    width: int
    height: int
    has_alpha: bool
    recipe: dict[str, Any] | None = None


@dataclass
class _SelectionRecord:
    selection_id: str
    video_id: str
    spec: SelectionSpec


@dataclass
class _RenderRecord:
    render_id: str
    video_path: Path
    report_path: Path
    report: dict[str, Any]


class StudioState:
    """Thread-safe in-memory + XFS storage registry for videos, designs, selections, and renders."""

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.videos_dir = storage_dir / "videos"
        self.designs_dir = storage_dir / "designs"
        self.renders_dir = storage_dir / "renders"
        for d in (self.videos_dir, self.designs_dir, self.renders_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self.videos: dict[str, _VideoRecord] = {}
        self.designs: dict[str, _DesignRecord] = {}
        self.selections: dict[str, _SelectionRecord] = {}
        self.renders: dict[str, _RenderRecord] = {}
        self.demo_clips_ready = False
        self.demo_payload: list[dict[str, Any]] = []
        self._load_demos()

    def _load_demos(self) -> None:
        try:
            if not (DEFAULT_DEMO_DIR / "demo_manifest.json").is_file():
                generate_demo_corpus(DEFAULT_DEMO_DIR)
            clips = load_demo_manifest(DEFAULT_DEMO_DIR)
        except Exception:
            self.demo_clips_ready = False
            self.demo_payload = []
            return

        mesh_intr: tuple[float, float, float, float] | None = None
        mesh_meta_path = DEFAULT_DEMO_DIR / "mesh_box_meta.json"
        if mesh_meta_path.is_file():
            try:
                mm = json.loads(mesh_meta_path.read_text(encoding="utf-8"))
                iv = mm.get("intrinsics")
                if isinstance(iv, list) and len(iv) == 4:
                    mesh_intr = (float(iv[0]), float(iv[1]), float(iv[2]), float(iv[3]))
            except Exception:
                pass

        demo_list: list[dict[str, Any]] = []
        with self._lock:
            for c in clips:
                p = Path(c.path)
                if not p.is_file():
                    continue
                is_mesh = c.id == "mesh_box_640x480"
                mpath = (DEFAULT_DEMO_DIR / "mesh_box.ply") if is_mesh and (DEFAULT_DEMO_DIR / "mesh_box.ply").is_file() else None
                dur = round(float(c.frame_count) / max(float(c.fps), 1.0), 4)
                rec = _VideoRecord(
                    video_id=c.id,
                    path=p,
                    width=int(c.width),
                    height=int(c.height),
                    fps=float(c.fps),
                    frame_count=int(c.frame_count),
                    duration_s=dur,
                    codec="h264",
                    is_demo=True,
                    default_selection=c.default_selection.to_dict(),
                    recommended_wrap=c.recommended_wrap,
                    mesh_path=mpath,
                    intrinsics=mesh_intr if is_mesh else None,
                )
                self.videos[c.id] = rec
                d_entry = c.to_dict()
                d_entry["media_url"] = f"/api/videos/{c.id}/media"
                demo_list.append(d_entry)
        self.demo_payload = demo_list
        self.demo_clips_ready = len(demo_list) >= 5

    def get_video(self, video_id: str) -> _VideoRecord:
        vid = _validate_id(video_id, "video")
        with self._lock:
            rec = self.videos.get(vid)
        if rec is None or not rec.path.is_file():
            raise HTTPException(status_code=404, detail=f"Video not found: {vid}")
        return rec

    def register_video(self, rec: _VideoRecord) -> None:
        with self._lock:
            self.videos[rec.video_id] = rec

    def register_design(self, rec: _DesignRecord) -> None:
        with self._lock:
            self.designs[rec.design_id] = rec

    def get_design(self, design_id: str) -> _DesignRecord:
        did = _validate_id(design_id, "design")
        with self._lock:
            rec = self.designs.get(did)
        if rec is None or not rec.image_path.is_file():
            raise HTTPException(status_code=404, detail=f"Design not found: {did}")
        return rec

    def register_selection(self, rec: _SelectionRecord) -> None:
        with self._lock:
            self.selections[rec.selection_id] = rec

    def get_selection(self, selection_id: str) -> _SelectionRecord:
        sid = _validate_id(selection_id, "selection")
        with self._lock:
            rec = self.selections.get(sid)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Selection not found: {sid}")
        return rec

    def register_render(self, rec: _RenderRecord) -> None:
        with self._lock:
            self.renders[rec.render_id] = rec

    def get_render(self, render_id: str) -> _RenderRecord:
        rid = _validate_id(render_id, "render")
        with self._lock:
            rec = self.renders.get(rid)
        if rec is None or not rec.video_path.is_file() or not rec.report_path.is_file():
            raise HTTPException(status_code=404, detail=f"Render not found: {rid}")
        return rec


def _resolve_render_inputs(
    state: StudioState,
    vid_rec: _VideoRecord,
    payload: dict[str, Any],
) -> tuple[SelectionSpec, DesignSpec, PlacementSpec, Any, tuple[float, float, float, float] | None]:
    # 1. Selection
    sel: SelectionSpec | None = None
    if payload.get("selection_id"):
        sel_rec = state.get_selection(str(payload["selection_id"]))
        sel = sel_rec.spec
    elif payload.get("selection"):
        sel = _parse_selection_from_dict(payload["selection"])
    elif vid_rec.default_selection is not None:
        sel = SelectionSpec.from_dict(vid_rec.default_selection)
    else:
        sel = _default_selection_for_frame(vid_rec.width, vid_rec.height, frame_index=0)
    sel.validate_for_frame(vid_rec.width, vid_rec.height, frame_count=vid_rec.frame_count)

    # 2. Design
    design: DesignSpec
    if payload.get("design_id"):
        d_rec = state.get_design(str(payload["design_id"]))
        design = DesignSpec(source="image", image_path=str(d_rec.image_path), colors=[])
    elif payload.get("design_recipe"):
        design = _parse_design_from_dict(payload["design_recipe"])
    else:
        design = DesignSpec(
            source="recipe",
            pattern="stripes",
            colors=["#ff595e", "#1982c4", "#ffca3a"],
            text="",
        )

    # 3. Placement
    placement = _parse_placement_from_dict(payload.get("placement"))

    # 4. Optional mesh & intrinsics for mesh_6dof
    mesh_spec = payload.get("mesh")
    if placement.wrap_mode == "mesh_6dof" and mesh_spec is None:
        mesh_spec = vid_rec.mesh_path if vid_rec.mesh_path is not None else "box"

    intr: tuple[float, float, float, float] | None = None
    raw_intr = payload.get("intrinsics")
    if isinstance(raw_intr, (list, tuple)) and len(raw_intr) == 4:
        intr = (float(raw_intr[0]), float(raw_intr[1]), float(raw_intr[2]), float(raw_intr[3]))
    elif placement.wrap_mode == "mesh_6dof":
        intr = vid_rec.intrinsics

    return sel, design, placement, mesh_spec, intr


def create_app(*, storage_dir: str | os.PathLike[str] = ".cache/studio-runtime") -> FastAPI:
    """Create the FastAPI application serving REST, RFC Range media, WebSocket stream, and Web Studio."""
    resolved_storage = _validate_storage_dir(storage_dir)
    state = StudioState(resolved_storage)

    app = FastAPI(title="visualizeIT Studio", version="1.0.0")
    app.state.studio = state

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.exception_handler(ContractValidationError)
    async def _contract_err_handler(request: Request, exc: ContractValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/", response_class=HTMLResponse)
    async def index() -> Response:
        idx = WEB_DIR / "index.html"
        if not idx.is_file():
            raise HTTPException(status_code=404, detail="Web Studio index.html not found")
        return FileResponse(str(idx), media_type="text/html; charset=utf-8")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "1.0.0",
            "native_raster_available": bool(is_native_raster_available(auto_build=False)),
            "demo_clips_ready": bool(state.demo_clips_ready),
        }

    @app.get("/api/demos")
    async def demos() -> dict[str, Any]:
        return {"clips": list(state.demo_payload)}

    @app.get("/api/designs/presets")
    async def design_presets() -> dict[str, Any]:
        return {"presets": DESIGN_PRESETS}

    @app.post("/api/designs/render")
    async def api_design_render(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Malformed JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="Request body must be a JSON object")
        try:
            spec = _parse_design_from_dict(body)
            rgba = render_design(spec)
        except (ContractValidationError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        design_id = f"dsn_{uuid.uuid4().hex[:12]}"
        out_png = state.designs_dir / f"{design_id}.png"
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        ok, buf = cv2.imencode(".png", bgra)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode design PNG")
        out_png.write_bytes(buf.tobytes())
        has_alpha = bool(int(rgba[:, :, 3].min()) < 255)
        state.register_design(
            _DesignRecord(
                design_id=design_id,
                image_path=out_png,
                width=int(rgba.shape[1]),
                height=int(rgba.shape[0]),
                has_alpha=has_alpha,
                recipe=spec.to_dict(),
            )
        )
        return {
            "design_id": design_id,
            "image_url": f"/api/designs/{design_id}/image.png",
            "recipe": spec.to_dict(),
        }

    @app.post("/api/designs/upload")
    async def api_design_upload(file: UploadFile = File(...)) -> dict[str, Any]:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded design file is empty")
        design_id = f"img_{uuid.uuid4().hex[:12]}"
        suffix = Path(file.filename or "upload.png").suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            suffix = ".bin"
        tmp_path = state.designs_dir / f".pending_{design_id}{suffix}"
        tmp_path.write_bytes(raw)
        try:
            rgba = load_design_image(tmp_path)
            # Detect whether original upload carried an alpha channel or transparent pixels
            from PIL import Image

            with Image.open(tmp_path) as pil_im:
                has_alpha = ("A" in pil_im.getbands()) or (int(rgba[:, :, 3].min()) < 255)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=415, detail=f"Invalid or corrupt design image: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        out_png = state.designs_dir / f"{design_id}.png"
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        ok, buf = cv2.imencode(".png", bgra)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to store design image")
        out_png.write_bytes(buf.tobytes())
        state.register_design(
            _DesignRecord(
                design_id=design_id,
                image_path=out_png,
                width=int(rgba.shape[1]),
                height=int(rgba.shape[0]),
                has_alpha=bool(has_alpha),
            )
        )
        return {
            "design_id": design_id,
            "image_url": f"/api/designs/{design_id}/image.png",
            "width": int(rgba.shape[1]),
            "height": int(rgba.shape[0]),
            "has_alpha": bool(has_alpha),
        }

    @app.get("/api/designs/{design_id}/image.png")
    async def api_design_image(design_id: str) -> Response:
        rec = state.get_design(design_id)
        return FileResponse(str(rec.image_path), media_type="image/png")

    @app.post("/api/videos/upload")
    async def api_video_upload(file: UploadFile = File(...)) -> dict[str, Any]:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty")
        video_id = f"vid_{uuid.uuid4().hex[:12]}"
        orig_ext = Path(file.filename or "upload.mp4").suffix.lower()
        ext = orig_ext if orig_ext in {".mp4", ".webm", ".mov", ".avi", ".mkv"} else ".mp4"
        dest_path = state.videos_dir / f"{video_id}{ext}"
        dest_path.write_bytes(raw)
        try:
            info = await asyncio.to_thread(probe_video_info, dest_path)
        except Exception as exc:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(status_code=415, detail=f"Invalid or unsupported video file: {exc}") from exc

        rec = _VideoRecord(
            video_id=video_id,
            path=dest_path,
            width=int(info["width"]),
            height=int(info["height"]),
            fps=float(info["fps"]),
            frame_count=int(info["frame_count"]),
            duration_s=float(info["duration_s"]),
            codec=str(info["codec"]),
        )
        state.register_video(rec)
        return rec.to_meta_dict()

    @app.get("/api/videos/{video_id}")
    async def api_video_meta(video_id: str) -> dict[str, Any]:
        rec = state.get_video(video_id)
        return rec.to_meta_dict()

    @app.get("/api/videos/{video_id}/media")
    async def api_video_media(video_id: str, request: Request) -> Response:
        rec = state.get_video(video_id)
        return _build_range_response(rec.path, request, _guess_video_media_type(rec.path))

    @app.get("/api/videos/{video_id}/frame/{frame_index}")
    async def api_video_frame(video_id: str, frame_index: int) -> Response:
        rec = state.get_video(video_id)
        if frame_index < 0 or frame_index >= rec.frame_count:
            raise HTTPException(
                status_code=400,
                detail=f"frame_index {frame_index} out of bounds for video with {rec.frame_count} frames",
            )
        frm = await asyncio.to_thread(_decode_exact_frame, rec.path, frame_index)
        ok, buf = cv2.imencode(".jpg", frm, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode frame JPEG")
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    @app.post("/api/videos/{video_id}/select")
    async def api_video_select(video_id: str, request: Request) -> dict[str, Any]:
        rec = state.get_video(video_id)
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Malformed JSON body: {exc}") from exc
        try:
            sel = _parse_selection_from_dict(body)
            sel.validate_for_frame(rec.width, rec.height, frame_count=rec.frame_count)
        except (ContractValidationError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        frm = await asyncio.to_thread(_decode_exact_frame, rec.path, int(sel.frame_index))
        seg = await asyncio.to_thread(segment_frame, frm, sel, sharp_onboarding=True)
        ok, mask_buf = cv2.imencode(".png", seg.mask)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode mask PNG")
        selection_id = f"sel_{uuid.uuid4().hex[:12]}"
        state.register_selection(_SelectionRecord(selection_id=selection_id, video_id=rec.video_id, spec=sel))
        return {
            "selection_id": selection_id,
            "frame_index": int(seg.frame_index),
            "foreground_pixels": int(seg.foreground_pixels),
            "coverage_ratio": float(seg.coverage_ratio),
            "bbox": [int(v) for v in seg.bbox],
            "polygon": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in seg.polygon],
            "anchor_count": int(seg.anchor_count),
            "init_ms": round(float(seg.init_ms), 3),
            "mask_png_base64": base64.b64encode(mask_buf.tobytes()).decode("ascii"),
        }

    @app.post("/api/videos/{video_id}/render")
    async def api_video_render(video_id: str, request: Request) -> dict[str, Any]:
        rec = state.get_video(video_id)
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Malformed JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="Render payload must be a JSON object")

        try:
            sel, design, placement, mesh_spec, intr = _resolve_render_inputs(state, rec, body)
        except HTTPException:
            raise
        except (ContractValidationError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        render_id = f"ren_{uuid.uuid4().hex[:12]}"
        render_dir = state.renders_dir / render_id
        render_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = render_dir / "video.mp4"
        out_report = render_dir / "report.json"

        def _do_render() -> VideoRunReport:
            pipe = VideoPipeline()
            return pipe.process_video(
                input_path=rec.path,
                output_path=out_mp4,
                design=design,
                placement=placement,
                selection=sel,
                mesh_spec=mesh_spec,
                intrinsics=intr,
                tracking_height=360,
                threads=2,
                raster_backend="auto",
                report_path=out_report,
            )

        try:
            report = await asyncio.to_thread(_do_render)
        except ContractValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Render failed: {exc}") from exc

        rep_dict = report.to_dict()
        state.register_render(
            _RenderRecord(
                render_id=render_id,
                video_path=out_mp4,
                report_path=out_report,
                report=rep_dict,
            )
        )
        return {
            "render_id": render_id,
            "status": "completed",
            "video_url": f"/api/renders/{render_id}/video.mp4",
            "report_url": f"/api/renders/{render_id}/report.json",
            "report": rep_dict,
        }

    @app.get("/api/renders/{render_id}/video.mp4")
    async def api_render_video(render_id: str, request: Request) -> Response:
        rec = state.get_render(render_id)
        return _build_range_response(rec.video_path, request, "video/mp4")

    @app.get("/api/renders/{render_id}/report.json")
    async def api_render_report(render_id: str) -> Response:
        rec = state.get_render(render_id)
        return FileResponse(str(rec.report_path), media_type="application/json")

    @app.websocket("/ws/videos/{video_id}/stream")
    async def ws_video_stream(websocket: WebSocket, video_id: str) -> None:
        # Validate video_id before accepting so unknown IDs cleanly return error or close
        try:
            vid_rec = state.get_video(video_id)
        except HTTPException as exc:
            await websocket.accept()
            await websocket.send_json({"type": "error", "error": exc.detail})
            await websocket.close(code=4004)
            return

        await websocket.accept()

        try:
            raw_text = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
            payload = json.loads(raw_text)
            if not isinstance(payload, dict):
                raise ContractValidationError("WebSocket config payload must be a JSON object")
        except WebSocketDisconnect:
            return
        except Exception as exc:
            try:
                await websocket.send_json({"type": "error", "error": f"Invalid config JSON: {exc}"})
                await websocket.close(code=1008)
            except Exception:
                pass
            return

        # Validate max_frames if provided
        max_frames: int | None = None
        if "max_frames" in payload and payload["max_frames"] is not None:
            try:
                raw_mf = payload["max_frames"]
                if isinstance(raw_mf, bool):
                    raise ValueError("max_frames cannot be bool")
                max_frames = int(raw_mf)
                if max_frames <= 0:
                    raise ValueError(f"max_frames must be >= 1, got {max_frames}")
            except Exception as exc:
                await websocket.send_json({"type": "error", "error": str(exc)})
                await websocket.close(code=1008)
                return

        try:
            sel, design, placement, mesh_spec, intr = _resolve_render_inputs(state, vid_rec, payload)
        except (HTTPException, ContractValidationError, ValueError, TypeError) as exc:
            msg = exc.detail if isinstance(exc, HTTPException) else str(exc)
            await websocket.send_json({"type": "error", "error": msg})
            await websocket.close(code=1008)
            return

        # Run streaming decode + causal tracking + native rasterization in a dedicated worker thread
        # so the FastAPI event loop remains 100% responsive for concurrent Range/HTTP/WS requests!
        loop = asyncio.get_running_loop()
        out_q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=8)
        cancel_ev = threading.Event()

        def _put_sync(item: tuple[str, Any]) -> None:
            fut = asyncio.run_coroutine_threadsafe(out_q.put(item), loop)
            while not cancel_ev.is_set():
                try:
                    fut.result(timeout=0.1)
                    return
                except TimeoutError:
                    continue
                except Exception:
                    return

        def _stream_worker() -> None:
            cv2.setNumThreads(2)
            mesh = resolve_mesh(mesh_spec) if placement.wrap_mode == "mesh_6dof" else None
            session = VideoPipelineSession(
                design=design,
                placement=placement,
                tracking_height=360,
                raster_backend="auto",
                mesh=mesh,
                intrinsics=intr,
            )

            cap = cv2.VideoCapture(str(vid_rec.path))
            if not cap.isOpened():
                _put_sync(("error", "Failed to open video stream"))
                return

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
            processed = 0
            select_idx = int(sel.frame_index)
            source_fps = float(vid_rec.fps) if vid_rec.fps > 0 else 30.0
            backend_used = "native_cpp"

            try:
                f_idx = 0
                while not cancel_ev.is_set():
                    if max_frames is not None and f_idx >= max_frames:
                        break
                    tf0 = time.perf_counter()
                    td0 = time.perf_counter()
                    ok, frame_bgr = cap.read()
                    dec_ms = (time.perf_counter() - td0) * 1000.0
                    if not ok or frame_bgr is None:
                        break
                    ts = float(f_idx) / source_fps

                    if f_idx < select_idx:
                        out_bgr = frame_bgr
                        state_str = "lost"
                        inliers = 0
                        prep_ms = trk_ms = cmp_ms = 0.0
                    elif f_idx == select_idx:
                        res_f = session.initialize(frame_bgr, sel, timestamp_s=ts)
                        out_bgr = res_f.frame_bgr
                        state_str = res_f.track_result.state
                        inliers = int(res_f.track_result.inliers)
                        prep_ms, trk_ms, cmp_ms = res_f.prep_ms, res_f.track_ms, res_f.composite_ms
                        if session.compositor is not None:
                            backend_used = session.compositor.backend
                    else:
                        res_f = session.process_frame(frame_bgr, f_idx, ts)
                        out_bgr = res_f.frame_bgr
                        state_str = res_f.track_result.state
                        inliers = int(res_f.track_result.inliers)
                        prep_ms, trk_ms, cmp_ms = res_f.prep_ms, res_f.track_ms, res_f.composite_ms

                    te0 = time.perf_counter()
                    ok_enc, jpg_buf = cv2.imencode(".jpg", out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    enc_ms = (time.perf_counter() - te0) * 1000.0
                    if not ok_enc:
                        raise RuntimeError(f"JPEG encode failed on frame {f_idx}")

                    total_ms = max(0.05, (time.perf_counter() - tf0) * 1000.0)
                    decode_times.append(dec_ms)
                    prep_times.append(prep_ms)
                    track_times.append(trk_ms)
                    comp_times.append(cmp_ms)
                    encode_times.append(enc_ms)
                    frame_totals.append(total_ms)
                    inlier_counts.append(inliers)

                    if state_str == "tracking":
                        tracked_count += 1
                        if had_tracked_before and prev_state in ("lost", "recovering"):
                            recovered_count += 1
                        had_tracked_before = True
                    elif state_str == "lost":
                        lost_count += 1
                    prev_state = state_str

                    processed += 1
                    elapsed_s = max(1e-5, time.perf_counter() - t_wall_start)
                    inst_fps = float(processed) / elapsed_s

                    msg = {
                        "type": "frame",
                        "frame_index": f_idx,
                        "state": state_str,
                        "fps": round(inst_fps, 2),
                        "latency_ms": round(total_ms, 3),
                        "inliers": inliers,
                        "jpeg_base64": base64.b64encode(jpg_buf.tobytes()).decode("ascii"),
                    }
                    _put_sync(("frame", msg))
                    f_idx += 1

                if not cancel_ev.is_set():
                    wall_time_s = max(1e-6, time.perf_counter() - t_wall_start)
                    eff_fps = float(processed) / wall_time_s
                    steady_slice = frame_totals[3:] if len(frame_totals) > 6 else frame_totals
                    steady_mean_ms = float(np.mean(steady_slice)) if steady_slice else 0.0
                    steady_fps = (1000.0 / steady_mean_ms) if steady_mean_ms > 1e-6 else eff_fps

                    report = VideoRunReport(
                        input_path=str(vid_rec.path),
                        output_path=None,
                        width=vid_rec.width,
                        height=vid_rec.height,
                        source_fps=source_fps,
                        wrap_mode=placement.wrap_mode,
                        blend_mode=placement.blend_mode,
                        raster_backend=backend_used,  # type: ignore[arg-type]
                        render_stride=1,
                        input_frame_count=processed,
                        processed_frame_count=processed,
                        rendered_frame_count=processed,
                        output_frame_count=processed,
                        tracked_frame_count=tracked_count,
                        lost_frame_count=lost_count,
                        recovered_count=recovered_count,
                        tracking_availability=float(tracked_count) / float(max(processed, 1)),
                        wall_time_seconds=round(wall_time_s, 6),
                        effective_fps=round(eff_fps, 4),
                        steady_state_fps=round(steady_fps, 4),
                        frame_latency_ms=_percentiles(frame_totals),
                        stage_latency_mean_ms={
                            "decode_ms": round(float(np.mean(decode_times)) if decode_times else 0.0, 4),
                            "prep_ms": round(float(np.mean(prep_times)) if prep_times else 0.0, 4),
                            "track_ms": round(float(np.mean(track_times)) if track_times else 0.0, 4),
                            "composite_ms": round(float(np.mean(comp_times)) if comp_times else 0.0, 4),
                            "encode_ms": round(float(np.mean(encode_times)) if encode_times else 0.0, 4),
                        },
                        inlier_count_median=round(float(np.median(inlier_counts)) if inlier_counts else 0.0, 2),
                        reprojection_rmse_px=(
                            round(session.tracker.reprojection_rmse_px, 4)
                            if session.tracker.reprojection_rmse_px is not None
                            else None
                        ),
                    )
                    _put_sync(("complete", {"type": "complete", "report": report.to_dict()}))
            except Exception as exc:
                if not cancel_ev.is_set():
                    _put_sync(("error", str(exc)))
            finally:
                cap.release()
                _put_sync(("done", None))

        worker = threading.Thread(target=_stream_worker, daemon=True)
        worker.start()

        try:
            while True:
                kind, data = await out_q.get()
                if kind == "frame":
                    await websocket.send_json(data)
                elif kind == "complete":
                    await websocket.send_json(data)
                    await websocket.close(code=1000)
                    break
                elif kind == "error":
                    await websocket.send_json({"type": "error", "error": str(data)})
                    await websocket.close(code=1011)
                    break
                elif kind == "done":
                    break
        except (WebSocketDisconnect, RuntimeError, OSError):
            cancel_ev.set()
        finally:
            cancel_ev.set()
            worker.join(timeout=5.0)

    return app


def run_server(*, host: str = "127.0.0.1", port: int = 8790, storage_dir: str | os.PathLike[str] = ".cache/studio-runtime") -> int:
    """Start Uvicorn HTTP/1.1 + WebSocket server."""
    import uvicorn

    app = create_app(storage_dir=storage_dir)
    uvicorn.run(
        app,
        host=host,
        port=int(port),
        http="h11",
        timeout_keep_alive=30,
        log_level="warning",
    )
    return 0
