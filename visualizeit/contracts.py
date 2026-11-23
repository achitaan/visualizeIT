"""Typed data contracts, validation rules, and JSON serialization for visualizeIT."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Literal, Mapping, Sequence

import numpy as np


class ContractValidationError(ValueError):
    """Raised when a contract field or cross-field invariant fails validation."""


VALID_DESIGN_SOURCES: tuple[str, ...] = ("recipe", "image")
VALID_PATTERNS: tuple[str, ...] = ("solid", "stripes", "checker", "dots", "waves", "grid")
VALID_WRAP_MODES: tuple[str, ...] = ("planar", "cylinder", "mesh_grid", "mesh_6dof")
VALID_BLEND_MODES: tuple[str, ...] = ("luminance", "normal", "ink")
VALID_TRACK_STATES: tuple[str, ...] = ("tracking", "recovering", "lost")
VALID_RASTER_BACKENDS: tuple[str, ...] = ("native_cpp", "opencv")

REQUIRED_FRAME_LATENCY_KEYS: tuple[str, ...] = ("mean", "p50", "p95", "p99", "max")
REQUIRED_STAGE_LATENCY_KEYS: tuple[str, ...] = (
    "decode_ms",
    "prep_ms",
    "track_ms",
    "composite_ms",
    "encode_ms",
)

# Exact 16 studio colors from bench/web/design-studio/design-engine.js plus standard CSS colors.
NAMED_CSS_COLORS: dict[str, str] = {
    "cream": "#f4ebd8",
    "ink": "#202820",
    "green": "#42634b",
    "orange": "#dc693f",
    "pink": "#e5a1ac",
    "blue": "#376aa0",
    "red": "#bd3f3f",
    "yellow": "#e9bb42",
    "purple": "#795b9b",
    "black": "#111111",
    "white": "#ffffff",
    "navy": "#203957",
    "teal": "#287e80",
    "brown": "#805b41",
    "gray": "#808080",
    "grey": "#808080",
    "cyan": "#00bcd4",
    "magenta": "#d81b60",
    "gold": "#f4b400",
    "coral": "#ff6f61",
    "lime": "#7cb342",
    "indigo": "#3949ab",
    "violet": "#8e24aa",
    "silver": "#c0c0c0",
    "maroon": "#800000",
    "olive": "#808000",
    "turquoise": "#1abc9c",
    "crimson": "#dc143c",
}

_HEX6_RE = re.compile(r"^#[0-9a-f]{6}$")
_CLIP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _check_int(
    val: Any,
    name: str,
    *,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise ContractValidationError(f"{name} must be an integer, got {type(val).__name__}")
    out = int(val)
    if min_val is not None and out < min_val:
        raise ContractValidationError(f"{name} must be >= {min_val}, got {out}")
    if max_val is not None and out > max_val:
        raise ContractValidationError(f"{name} must be <= {max_val}, got {out}")
    return out


def _check_float(
    val: Any,
    name: str,
    *,
    min_val: float | None = None,
    max_val: float | None = None,
    strict_gt: float | None = None,
) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ContractValidationError(f"{name} must be a finite number, got {type(val).__name__}")
    out = float(val)
    if not math.isfinite(out):
        raise ContractValidationError(f"{name} must be finite, got {val!r}")
    if strict_gt is not None and not (out > strict_gt):
        raise ContractValidationError(f"{name} must be > {strict_gt}, got {out}")
    if min_val is not None and out < min_val:
        raise ContractValidationError(f"{name} must be >= {min_val}, got {out}")
    if max_val is not None and out > max_val:
        raise ContractValidationError(f"{name} must be <= {max_val}, got {out}")
    return out


def _check_enum(val: Any, name: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(val, str) or val not in allowed:
        raise ContractValidationError(f"{name} must be one of {allowed}, got {val!r}")
    return val


def normalize_color(value: Any) -> str:
    """Normalize a 6-digit #RRGGBB hex color or known CSS color name to lowercase #rrggbb."""
    if not isinstance(value, str):
        raise ContractValidationError(f"Color must be a string, got {type(value).__name__}")
    cleaned = value.strip().lower()
    if cleaned in NAMED_CSS_COLORS:
        return NAMED_CSS_COLORS[cleaned]
    if _HEX6_RE.fullmatch(cleaned):
        return cleaned
    raise ContractValidationError(
        f"Invalid color {value!r}; expected 6-digit #RRGGBB or supported color name."
    )


@dataclass
class DesignSpec:
    source: Literal["recipe", "image"] = "recipe"
    pattern: Literal["solid", "stripes", "checker", "dots", "waves", "grid"] = "stripes"
    colors: list[str] = field(default_factory=lambda: ["#ff595e", "#1982c4", "#ffca3a"])
    text: str = ""
    prompt: str = ""
    image_path: str | None = None
    width: int = 512
    height: int = 512
    seed: int = 42
    scale: float = 1.0
    rotation_deg: float = 0.0

    def __post_init__(self) -> None:
        self.source = _check_enum(self.source, "source", VALID_DESIGN_SOURCES)  # type: ignore[assignment]
        self.pattern = _check_enum(self.pattern, "pattern", VALID_PATTERNS)  # type: ignore[assignment]

        if isinstance(self.colors, (str, bytes)) or not isinstance(self.colors, Sequence):
            raise ContractValidationError("colors must be a list or tuple of color strings")

        raw_colors = list(self.colors)
        if self.source == "image" and len(raw_colors) == 0:
            self.colors = ["#ff595e", "#1982c4", "#ffca3a"]
        else:
            if not (2 <= len(raw_colors) <= 5):
                raise ContractValidationError(
                    f"colors must contain 2 to 5 entries, got {len(raw_colors)}"
                )
            self.colors = [normalize_color(c) for c in raw_colors]

        if not isinstance(self.text, str) or len(self.text) > 80:
            raise ContractValidationError("text must be a string of at most 80 characters")
        if not isinstance(self.prompt, str) or len(self.prompt) > 500:
            raise ContractValidationError("prompt must be a string of at most 500 characters")

        if self.image_path is not None:
            if not isinstance(self.image_path, str) or not self.image_path.strip():
                raise ContractValidationError("image_path must be a non-empty string when provided")
        if self.source == "image" and self.image_path is None:
            raise ContractValidationError("image_path must be provided when source='image'")

        self.width = _check_int(self.width, "width", min_val=16, max_val=4096)
        self.height = _check_int(self.height, "height", min_val=16, max_val=4096)
        self.seed = _check_int(self.seed, "seed", min_val=0, max_val=4294967295)
        self.scale = _check_float(self.scale, "scale", min_val=0.25, max_val=4.0)
        self.rotation_deg = _check_float(self.rotation_deg, "rotation_deg", min_val=-180.0, max_val=180.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "pattern": self.pattern,
            "colors": list(self.colors),
            "text": self.text,
            "prompt": self.prompt,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "scale": self.scale,
            "rotation_deg": self.rotation_deg,
        }

    def to_recipe_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "colors": list(self.colors),
            "text": self.text,
            "scale": self.scale,
            "rotation_deg": self.rotation_deg,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DesignSpec:
        if not isinstance(data, Mapping):
            raise ContractValidationError("DesignSpec.from_dict expects a mapping")
        allowed = {
            "source",
            "pattern",
            "colors",
            "text",
            "prompt",
            "image_path",
            "width",
            "height",
            "seed",
            "scale",
            "rotation_deg",
        }
        unknown = set(data.keys()) - allowed
        if unknown:
            raise ContractValidationError(f"Unknown DesignSpec fields: {sorted(unknown)}")
        kwargs: dict[str, Any] = dict(data)
        if "source" not in kwargs and kwargs.get("image_path"):
            kwargs["source"] = "image"
        return cls(**kwargs)


@dataclass
class PlacementSpec:
    wrap_mode: Literal["planar", "cylinder", "mesh_grid", "mesh_6dof"] = "planar"
    blend_mode: Literal["luminance", "normal", "ink"] = "luminance"
    opacity: float = 0.90
    scale: float = 1.0
    rotation_deg: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    cylinder_radius_frac: float = 0.5
    tile_repeat: bool = False

    def __post_init__(self) -> None:
        self.wrap_mode = _check_enum(self.wrap_mode, "wrap_mode", VALID_WRAP_MODES)  # type: ignore[assignment]
        self.blend_mode = _check_enum(self.blend_mode, "blend_mode", VALID_BLEND_MODES)  # type: ignore[assignment]
        self.opacity = _check_float(self.opacity, "opacity", min_val=0.0, max_val=1.0)
        self.scale = _check_float(self.scale, "scale", min_val=0.1, max_val=5.0)
        self.rotation_deg = _check_float(self.rotation_deg, "rotation_deg", min_val=-3600.0, max_val=3600.0)
        self.offset_x = _check_float(self.offset_x, "offset_x", min_val=-5.0, max_val=5.0)
        self.offset_y = _check_float(self.offset_y, "offset_y", min_val=-5.0, max_val=5.0)
        self.cylinder_radius_frac = _check_float(
            self.cylinder_radius_frac, "cylinder_radius_frac", min_val=0.05, max_val=5.0
        )
        if not isinstance(self.tile_repeat, bool):
            raise ContractValidationError("tile_repeat must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "wrap_mode": self.wrap_mode,
            "blend_mode": self.blend_mode,
            "opacity": self.opacity,
            "scale": self.scale,
            "rotation_deg": self.rotation_deg,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "cylinder_radius_frac": self.cylinder_radius_frac,
            "tile_repeat": self.tile_repeat,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlacementSpec:
        if not isinstance(data, Mapping):
            raise ContractValidationError("PlacementSpec.from_dict expects a mapping")
        allowed = {
            "wrap_mode",
            "blend_mode",
            "opacity",
            "scale",
            "rotation_deg",
            "offset_x",
            "offset_y",
            "cylinder_radius_frac",
            "tile_repeat",
        }
        unknown = set(data.keys()) - allowed
        if unknown:
            raise ContractValidationError(f"Unknown PlacementSpec fields: {sorted(unknown)}")
        return cls(**dict(data))


def _parse_point_list(points: Any, name: str) -> list[tuple[float, float]]:
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise ContractValidationError(f"{name} must be a sequence of (x, y) coordinates")
    parsed: list[tuple[float, float]] = []
    for idx, pt in enumerate(points):
        if isinstance(pt, (str, bytes)) or not isinstance(pt, Sequence) or len(pt) != 2:
            raise ContractValidationError(f"{name}[{idx}] must be a 2-element (x, y) coordinate")
        px = _check_float(pt[0], f"{name}[{idx}].x", min_val=0.0)
        py = _check_float(pt[1], f"{name}[{idx}].y", min_val=0.0)
        parsed.append((px, py))
    return parsed


@dataclass
class SelectionSpec:
    frame_index: int = 0
    positive_points: list[tuple[float, float]] = field(default_factory=list)
    negative_points: list[tuple[float, float]] = field(default_factory=list)
    bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        self.frame_index = _check_int(self.frame_index, "frame_index", min_val=0)
        self.positive_points = _parse_point_list(self.positive_points, "positive_points")
        self.negative_points = _parse_point_list(self.negative_points, "negative_points")
        if self.bbox is not None:
            if isinstance(self.bbox, (str, bytes)) or not isinstance(self.bbox, Sequence) or len(self.bbox) != 4:
                raise ContractValidationError("bbox must be a 4-element (x, y, w, h) tuple")
            bx = _check_float(self.bbox[0], "bbox.x", min_val=0.0)
            by = _check_float(self.bbox[1], "bbox.y", min_val=0.0)
            bw = _check_float(self.bbox[2], "bbox.w", strict_gt=0.0)
            bh = _check_float(self.bbox[3], "bbox.h", strict_gt=0.0)
            self.bbox = (bx, by, bw, bh)

    @property
    def has_target(self) -> bool:
        return bool(self.positive_points or self.bbox is not None)

    def validate_for_frame(
        self,
        width: int,
        height: int,
        *,
        frame_count: int | None = None,
    ) -> None:
        w = _check_int(width, "width", min_val=1)
        h = _check_int(height, "height", min_val=1)
        if frame_count is not None:
            fc = _check_int(frame_count, "frame_count", min_val=1)
            if self.frame_index >= fc:
                raise ContractValidationError(
                    f"frame_index {self.frame_index} out of range for video with {fc} frames"
                )
        if not self.has_target:
            raise ContractValidationError(
                "SelectionSpec requires at least one positive_point or a non-empty bbox"
            )
        for idx, (px, py) in enumerate(self.positive_points):
            if px >= w or py >= h:
                raise ContractValidationError(
                    f"positive_points[{idx}]=({px}, {py}) outside frame bounds ({w}x{h})"
                )
        for idx, (px, py) in enumerate(self.negative_points):
            if px >= w or py >= h:
                raise ContractValidationError(
                    f"negative_points[{idx}]=({px}, {py}) outside frame bounds ({w}x{h})"
                )
        if self.bbox is not None:
            bx, by, bw, bh = self.bbox
            x0 = max(0.0, bx)
            y0 = max(0.0, by)
            x1 = min(float(w), bx + bw)
            y1 = min(float(h), by + bh)
            if x1 - x0 < 2.0 or y1 - y0 < 2.0:
                raise ContractValidationError(
                    f"bbox {self.bbox} has insufficient overlap with ({w}x{h}) frame"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "positive_points": [[x, y] for x, y in self.positive_points],
            "negative_points": [[x, y] for x, y in self.negative_points],
            "bbox": list(self.bbox) if self.bbox is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SelectionSpec:
        if not isinstance(data, Mapping):
            raise ContractValidationError("SelectionSpec.from_dict expects a mapping")
        allowed = {"frame_index", "positive_points", "negative_points", "bbox"}
        unknown = set(data.keys()) - allowed
        if unknown:
            raise ContractValidationError(f"Unknown SelectionSpec fields: {sorted(unknown)}")
        return cls(
            frame_index=data.get("frame_index", 0),
            positive_points=data.get("positive_points", []),
            negative_points=data.get("negative_points", []),
            bbox=data.get("bbox"),
        )


def _validate_numeric_elements(obj: Any, name: str) -> None:
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0 or np.issubdtype(obj.dtype, np.bool_) or obj.dtype.kind not in ("i", "u", "f"):
            raise ContractValidationError(
                f"{name} must be an ndarray of integer or floating-point values, got {obj.dtype}"
            )
        return
    if isinstance(obj, (str, bytes, bytearray)) or not isinstance(obj, Sequence):
        raise ContractValidationError(f"{name} must be a sequence or ndarray of real numbers")
    for item in obj:
        if isinstance(item, (bool, np.bool_)):
            raise ContractValidationError(f"{name} cannot contain boolean values")
        if isinstance(item, (int, float, np.integer, np.floating)):
            continue
        if isinstance(item, np.ndarray) or (
            isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray))
        ):
            _validate_numeric_elements(item, name)
        else:
            raise ContractValidationError(
                f"{name} must contain only real numeric values, got {type(item).__name__}"
            )


def _coerce_finite_array(val: Any, name: str, expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
    _validate_numeric_elements(val, name)
    try:
        arr = np.array(val, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractValidationError(f"{name} must be convertible to float64 ndarray") from exc
    if arr.ndim == 0 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ContractValidationError(f"{name} must contain finite numeric values")
    if expected_shape is not None and arr.shape != expected_shape:
        raise ContractValidationError(f"{name} must have shape {expected_shape}, got {arr.shape}")
    return arr


@dataclass
class TrackFrameResult:
    frame_index: int
    timestamp_s: float
    state: Literal["tracking", "recovering", "lost"]
    homography_3x3: np.ndarray | None = None
    pose_rvec: np.ndarray | None = None
    pose_tvec: np.ndarray | None = None
    grid_src_pts: np.ndarray | None = None
    grid_dst_pts: np.ndarray | None = None
    inliers: int = 0
    confidence: float = 0.0
    track_ms: float = 0.0
    confidence_state: Literal["TRACKING", "AMBIGUOUS", "LOST", "RECOVERING"] | None = None

    def __post_init__(self) -> None:
        self.frame_index = _check_int(self.frame_index, "frame_index", min_val=0)
        self.timestamp_s = _check_float(self.timestamp_s, "timestamp_s", min_val=0.0)
        self.state = _check_enum(self.state, "state", VALID_TRACK_STATES)  # type: ignore[assignment]
        self.inliers = _check_int(self.inliers, "inliers", min_val=0)
        self.confidence = _check_float(self.confidence, "confidence", min_val=0.0, max_val=1.0)
        self.track_ms = _check_float(self.track_ms, "track_ms", min_val=0.0)
        if self.confidence_state is None:
            self.confidence_state = (
                "TRACKING" if self.state == "tracking" else ("RECOVERING" if self.state == "recovering" else "LOST")
            )
        elif self.confidence_state not in ("TRACKING", "AMBIGUOUS", "LOST", "RECOVERING"):
            raise ContractValidationError(f"Invalid confidence_state: {self.confidence_state!r}")

        if self.state != "tracking":
            if (
                self.homography_3x3 is not None
                or self.pose_rvec is not None
                or self.pose_tvec is not None
                or self.grid_src_pts is not None
                or self.grid_dst_pts is not None
            ):
                raise ContractValidationError(
                    "Overlay transforms (homography_3x3/pose_rvec/pose_tvec/grid_src_pts/grid_dst_pts) "
                    f"must be None when state={self.state!r}"
                )

        if self.homography_3x3 is not None:
            self.homography_3x3 = _coerce_finite_array(
                self.homography_3x3, "homography_3x3", expected_shape=(3, 3)
            )

        if (self.pose_rvec is None) != (self.pose_tvec is None):
            raise ContractValidationError("pose_rvec and pose_tvec must both be provided or both be None")
        if self.pose_rvec is not None and self.pose_tvec is not None:
            rvec = _coerce_finite_array(self.pose_rvec, "pose_rvec")
            tvec = _coerce_finite_array(self.pose_tvec, "pose_tvec")
            if rvec.size != 3 or tvec.size != 3:
                raise ContractValidationError("pose_rvec and pose_tvec must each have 3 elements")
            self.pose_rvec = rvec.reshape(3)
            self.pose_tvec = tvec.reshape(3)

        if (self.grid_src_pts is None) != (self.grid_dst_pts is None):
            raise ContractValidationError(
                "grid_src_pts and grid_dst_pts must both be provided or both be None"
            )
        if self.grid_src_pts is not None and self.grid_dst_pts is not None:
            src = _coerce_finite_array(self.grid_src_pts, "grid_src_pts")
            dst = _coerce_finite_array(self.grid_dst_pts, "grid_dst_pts")
            if src.ndim != 2 or src.shape[1] != 2 or src.shape[0] < 3 or dst.shape != src.shape:
                raise ContractValidationError(
                    "grid_src_pts and grid_dst_pts must have matching shape (N, 2) with N >= 3"
                )
            self.grid_src_pts = src
            self.grid_dst_pts = dst

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "state": self.state,
            "homography_3x3": self.homography_3x3.tolist() if self.homography_3x3 is not None else None,
            "pose_rvec": self.pose_rvec.tolist() if self.pose_rvec is not None else None,
            "pose_tvec": self.pose_tvec.tolist() if self.pose_tvec is not None else None,
            "grid_src_pts": self.grid_src_pts.tolist() if self.grid_src_pts is not None else None,
            "grid_dst_pts": self.grid_dst_pts.tolist() if self.grid_dst_pts is not None else None,
            "inliers": self.inliers,
            "confidence": self.confidence,
            "track_ms": self.track_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrackFrameResult:
        if not isinstance(data, Mapping):
            raise ContractValidationError("TrackFrameResult.from_dict expects a mapping")
        required = {"frame_index", "timestamp_s", "state"}
        allowed = required | {
            "homography_3x3",
            "pose_rvec",
            "pose_tvec",
            "grid_src_pts",
            "grid_dst_pts",
            "inliers",
            "confidence",
            "track_ms",
        }
        missing = required - set(data.keys())
        if missing:
            raise ContractValidationError(f"Missing TrackFrameResult fields: {sorted(missing)}")
        unknown = set(data.keys()) - allowed
        if unknown:
            raise ContractValidationError(f"Unknown TrackFrameResult fields: {sorted(unknown)}")
        return cls(
            frame_index=data["frame_index"],
            timestamp_s=data["timestamp_s"],
            state=data["state"],
            homography_3x3=data.get("homography_3x3"),
            pose_rvec=data.get("pose_rvec"),
            pose_tvec=data.get("pose_tvec"),
            grid_src_pts=data.get("grid_src_pts"),
            grid_dst_pts=data.get("grid_dst_pts"),
            inliers=data.get("inliers", 0),
            confidence=data.get("confidence", 0.0),
            track_ms=data.get("track_ms", 0.0),
        )


def _validate_metric_dict(
    mapping: Any,
    name: str,
    required_keys: tuple[str, ...],
) -> dict[str, float]:
    if not isinstance(mapping, Mapping):
        raise ContractValidationError(f"{name} must be a dict")
    actual_keys = set(mapping.keys())
    expected_keys = set(required_keys)
    if actual_keys != expected_keys:
        raise ContractValidationError(
            f"{name} must have exact keys {list(required_keys)}, got {sorted(actual_keys)}"
        )
    return {k: _check_float(mapping[k], f"{name}.{k}", min_val=0.0) for k in required_keys}


@dataclass
class VideoRunReport:
    input_path: str
    output_path: str | None
    width: int
    height: int
    source_fps: float
    wrap_mode: str
    blend_mode: str
    raster_backend: str
    render_stride: int
    input_frame_count: int
    processed_frame_count: int
    rendered_frame_count: int
    output_frame_count: int
    tracked_frame_count: int
    lost_frame_count: int
    recovered_count: int
    tracking_availability: float
    wall_time_seconds: float
    effective_fps: float
    steady_state_fps: float
    frame_latency_ms: dict[str, float]
    stage_latency_mean_ms: dict[str, float]
    inlier_count_median: float
    reprojection_rmse_px: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input_path, str) or not self.input_path.strip():
            raise ContractValidationError("input_path must be a non-empty string")
        if self.output_path is not None:
            if not isinstance(self.output_path, str) or not self.output_path.strip():
                raise ContractValidationError("output_path must be None or a non-empty string")

        self.width = _check_int(self.width, "width", min_val=1)
        self.height = _check_int(self.height, "height", min_val=1)
        self.source_fps = _check_float(self.source_fps, "source_fps", strict_gt=0.0)
        self.wrap_mode = _check_enum(self.wrap_mode, "wrap_mode", VALID_WRAP_MODES)
        self.blend_mode = _check_enum(self.blend_mode, "blend_mode", VALID_BLEND_MODES)
        self.raster_backend = _check_enum(self.raster_backend, "raster_backend", VALID_RASTER_BACKENDS)

        self.render_stride = _check_int(self.render_stride, "render_stride")
        if self.render_stride != 1:
            raise ContractValidationError(
                f"render_stride must equal 1 (anti-cheating invariant), got {self.render_stride}"
            )

        self.input_frame_count = _check_int(self.input_frame_count, "input_frame_count", min_val=1)
        self.processed_frame_count = _check_int(
            self.processed_frame_count, "processed_frame_count", min_val=0, max_val=self.input_frame_count
        )
        self.rendered_frame_count = _check_int(
            self.rendered_frame_count, "rendered_frame_count", min_val=0, max_val=self.input_frame_count
        )
        self.output_frame_count = _check_int(
            self.output_frame_count, "output_frame_count", min_val=0, max_val=self.input_frame_count
        )

        if self.processed_frame_count != self.rendered_frame_count:
            raise ContractValidationError(
                f"processed_frame_count ({self.processed_frame_count}) must equal "
                f"rendered_frame_count ({self.rendered_frame_count})"
            )

        if self.output_path is not None:
            if not (
                self.input_frame_count
                == self.processed_frame_count
                == self.rendered_frame_count
                == self.output_frame_count
            ):
                raise ContractValidationError(
                    "When output_path is set, input_frame_count, processed_frame_count, "
                    "rendered_frame_count, and output_frame_count must all be equal; got "
                    f"({self.input_frame_count}, {self.processed_frame_count}, "
                    f"{self.rendered_frame_count}, {self.output_frame_count})"
                )

        self.tracked_frame_count = _check_int(self.tracked_frame_count, "tracked_frame_count", min_val=0)
        self.lost_frame_count = _check_int(self.lost_frame_count, "lost_frame_count", min_val=0)
        self.recovered_count = _check_int(self.recovered_count, "recovered_count", min_val=0)
        if self.tracked_frame_count + self.lost_frame_count > self.processed_frame_count:
            raise ContractValidationError(
                "tracked_frame_count + lost_frame_count cannot exceed processed_frame_count"
            )

        self.tracking_availability = _check_float(
            self.tracking_availability, "tracking_availability", min_val=0.0, max_val=1.0
        )
        self.wall_time_seconds = _check_float(self.wall_time_seconds, "wall_time_seconds", strict_gt=0.0)
        self.effective_fps = _check_float(self.effective_fps, "effective_fps", min_val=0.0)
        self.steady_state_fps = _check_float(self.steady_state_fps, "steady_state_fps", min_val=0.0)

        self.frame_latency_ms = _validate_metric_dict(
            self.frame_latency_ms, "frame_latency_ms", REQUIRED_FRAME_LATENCY_KEYS
        )
        self.stage_latency_mean_ms = _validate_metric_dict(
            self.stage_latency_mean_ms, "stage_latency_mean_ms", REQUIRED_STAGE_LATENCY_KEYS
        )
        self.inlier_count_median = _check_float(self.inlier_count_median, "inlier_count_median", min_val=0.0)
        if self.reprojection_rmse_px is not None:
            self.reprojection_rmse_px = _check_float(
                self.reprojection_rmse_px, "reprojection_rmse_px", min_val=0.0
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "width": self.width,
            "height": self.height,
            "source_fps": self.source_fps,
            "wrap_mode": self.wrap_mode,
            "blend_mode": self.blend_mode,
            "raster_backend": self.raster_backend,
            "render_stride": self.render_stride,
            "input_frame_count": self.input_frame_count,
            "processed_frame_count": self.processed_frame_count,
            "rendered_frame_count": self.rendered_frame_count,
            "output_frame_count": self.output_frame_count,
            "tracked_frame_count": self.tracked_frame_count,
            "lost_frame_count": self.lost_frame_count,
            "recovered_count": self.recovered_count,
            "tracking_availability": self.tracking_availability,
            "wall_time_seconds": self.wall_time_seconds,
            "effective_fps": self.effective_fps,
            "steady_state_fps": self.steady_state_fps,
            "frame_latency_ms": dict(self.frame_latency_ms),
            "stage_latency_mean_ms": dict(self.stage_latency_mean_ms),
            "inlier_count_median": self.inlier_count_median,
            "reprojection_rmse_px": self.reprojection_rmse_px,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VideoRunReport:
        if not isinstance(data, Mapping):
            raise ContractValidationError("VideoRunReport.from_dict expects a mapping")
        required = {
            "input_path",
            "output_path",
            "width",
            "height",
            "source_fps",
            "wrap_mode",
            "blend_mode",
            "raster_backend",
            "render_stride",
            "input_frame_count",
            "processed_frame_count",
            "rendered_frame_count",
            "output_frame_count",
            "tracked_frame_count",
            "lost_frame_count",
            "recovered_count",
            "tracking_availability",
            "wall_time_seconds",
            "effective_fps",
            "steady_state_fps",
            "frame_latency_ms",
            "stage_latency_mean_ms",
            "inlier_count_median",
        }
        missing = required - set(data.keys())
        if missing:
            raise ContractValidationError(f"Missing VideoRunReport fields: {sorted(missing)}")
        unknown = set(data.keys()) - (required | {"reprojection_rmse_px"})
        if unknown:
            raise ContractValidationError(f"Unknown VideoRunReport fields: {sorted(unknown)}")
        return cls(**dict(data))


@dataclass
class DemoClipInfo:
    id: str
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    default_selection: SelectionSpec
    recommended_wrap: Literal["planar", "cylinder", "mesh_grid", "mesh_6dof"] = "planar"

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _CLIP_ID_RE.fullmatch(self.id):
            raise ContractValidationError(f"Invalid demo clip id {self.id!r}")
        if not isinstance(self.path, str) or not self.path.strip():
            raise ContractValidationError("path must be a non-empty string")
        self.width = _check_int(self.width, "width", min_val=1)
        self.height = _check_int(self.height, "height", min_val=1)
        self.fps = _check_float(self.fps, "fps", strict_gt=0.0)
        self.frame_count = _check_int(self.frame_count, "frame_count", min_val=1)
        if isinstance(self.default_selection, Mapping):
            self.default_selection = SelectionSpec.from_dict(self.default_selection)
        elif not isinstance(self.default_selection, SelectionSpec):
            raise ContractValidationError("default_selection must be a SelectionSpec or dict")
        self.default_selection.validate_for_frame(
            self.width, self.height, frame_count=self.frame_count
        )
        self.recommended_wrap = _check_enum(
            self.recommended_wrap, "recommended_wrap", VALID_WRAP_MODES
        )  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "default_selection": self.default_selection.to_dict(),
            "recommended_wrap": self.recommended_wrap,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DemoClipInfo:
        if not isinstance(data, Mapping):
            raise ContractValidationError("DemoClipInfo.from_dict expects a mapping")
        required = {
            "id",
            "path",
            "width",
            "height",
            "fps",
            "frame_count",
            "default_selection",
            "recommended_wrap",
        }
        missing = required - set(data.keys())
        if missing:
            raise ContractValidationError(f"Missing DemoClipInfo fields: {sorted(missing)}")
        unknown = set(data.keys()) - required
        if unknown:
            raise ContractValidationError(f"Unknown DemoClipInfo fields: {sorted(unknown)}")
        return cls(**dict(data))
