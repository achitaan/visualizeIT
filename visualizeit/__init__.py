"""visualizeIT — Unified >=15 FPS Video & Interactive Surface Design Studio."""
from __future__ import annotations

from .contracts import (
    ContractValidationError,
    DemoClipInfo,
    DesignSpec,
    PlacementSpec,
    SelectionSpec,
    TrackFrameResult,
    VALID_BLEND_MODES,
    VALID_DESIGN_SOURCES,
    VALID_PATTERNS,
    VALID_RASTER_BACKENDS,
    VALID_TRACK_STATES,
    VALID_WRAP_MODES,
    VideoRunReport,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "ContractValidationError",
    "DemoClipInfo",
    "DesignSpec",
    "PlacementSpec",
    "SelectionSpec",
    "TrackFrameResult",
    "VideoRunReport",
    "VALID_PATTERNS",
    "VALID_DESIGN_SOURCES",
    "VALID_WRAP_MODES",
    "VALID_BLEND_MODES",
    "VALID_TRACK_STATES",
    "VALID_RASTER_BACKENDS",
]
