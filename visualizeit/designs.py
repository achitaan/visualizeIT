"""Procedural 2D design engine, natural-language prompt parser, and validated RGBA artwork loader."""
from __future__ import annotations

import io
import math
import os
from pathlib import Path
import re
import struct
from typing import Any, Callable, Mapping
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from .contracts import (
    ContractValidationError,
    DesignSpec,
    NAMED_CSS_COLORS,
    VALID_PATTERNS,
)


class DesignValidationError(ValueError):
    """Raised on invalid design recipe, prompt, or uploaded/loaded artwork file."""


PATTERNS: tuple[str, ...] = VALID_PATTERNS
NAMED_COLORS: dict[str, str] = dict(NAMED_CSS_COLORS)

MAX_UPLOAD_BYTES: int = 20 * 1024 * 1024
MAX_SOURCE_DIM: int = 8192
MAX_SOURCE_PIXELS: int = 36_000_000

_FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

_QUOTED_RE = re.compile(r'"([^"\n]*)"|\'([^\'\n]*)\'')
_TEXT_SUFFIX_RE = re.compile(r"\b(?:text|say|saying)\s*:?\s+(.+)$", re.IGNORECASE)
_COLOR_TOKEN_RE = re.compile(
    r"#[0-9a-z]+|\b(?:"
    + "|".join(sorted(NAMED_COLORS.keys(), key=len, reverse=True))
    + r")\b"
)
_HEX6_LOWER_RE = re.compile(r"^#[0-9a-f]{6}$")
_PATTERN_TOKEN_RE = re.compile(
    r"\b(solid|stripes?|striped|checker|checkerboard|checkered|dots?|dotted|waves?|wavy|grid)\b"
)
_PATTERN_ALIASES: dict[str, str] = {
    "stripe": "stripes",
    "striped": "stripes",
    "checkerboard": "checker",
    "checkered": "checker",
    "dot": "dots",
    "dotted": "dots",
    "wave": "waves",
    "wavy": "waves",
}
_NUMERIC_MOD_RE = re.compile(
    r"\b(scale|rotation|rotate|seed)\s*:?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:degrees?|deg)?"
)
_STOP_WORDS_RE = re.compile(
    r"\b(?:a|an|the|and|with|in|on|of|make|create|draw|pattern|colors?|background|please|text|say|saying|using)\b"
)
# Matches letters, numbers, and '#' (without '_') in parity with JS /[\p{L}\p{N}#]+/gu.
_WORD_OR_HASH_RE = re.compile(r"(?:[^\W_]|#)+", re.UNICODE)


def _imul32(a: int, b: int) -> int:
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) & 0xFFFFFFFF


def mulberry32(seed: int) -> Callable[[], float]:
    """Deterministic 32-bit Mulberry32 PRNG matching bench/web/design-studio/design-engine.js."""
    state = int(seed) & 0xFFFFFFFF

    def _next() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = _imul32(t ^ (t >> 15), t | 1)
        t = (t ^ (t + _imul32(t ^ (t >> 7), t | 61))) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return _next


def _hex_to_rgb_f32(color_hex: str) -> tuple[float, float, float]:
    return (
        float(int(color_hex[1:3], 16)),
        float(int(color_hex[3:5], 16)),
        float(int(color_hex[5:7], 16)),
    )


def _fmt_js_num(val: int | float) -> str:
    if isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
        return str(int(val))
    return f"{val:g}"


def parse_design_prompt_detailed(
    prompt: str,
    base: DesignSpec | Mapping[str, Any] | None = None,
) -> tuple[DesignSpec, list[str], list[str]]:
    """Parse a finite-grammar natural-language design prompt into (DesignSpec, warnings, unsupported_words)."""
    if not isinstance(prompt, str):
        raise DesignValidationError("Prompt must be a string.")
    if len(prompt) > 500:
        raise DesignValidationError("Prompt must contain at most 500 characters.")

    try:
        if base is None:
            spec = DesignSpec()
        elif isinstance(base, DesignSpec):
            spec = DesignSpec.from_dict(base.to_dict())
        elif isinstance(base, Mapping):
            base_dict = dict(base)
            spec = DesignSpec.from_dict(base_dict)
        else:
            raise DesignValidationError("base must be a DesignSpec, mapping, or None.")
    except ContractValidationError as exc:
        raise DesignValidationError(str(exc)) from exc

    spec.source = "recipe"
    spec.image_path = None
    spec.prompt = prompt

    warnings: list[str] = []
    words = prompt

    quoted_matches = list(_QUOTED_RE.finditer(words))
    if quoted_matches:
        first_m = quoted_matches[0]
        raw_text = first_m.group(1) if first_m.group(1) is not None else first_m.group(2)
        assert raw_text is not None
        spec.text = raw_text[:80]
        if len(raw_text) > 80:
            warnings.append("Text was shortened to 80 characters.")
        if len(quoted_matches) > 1:
            warnings.append("Only the first quoted text is used.")
        words = _QUOTED_RE.sub(" ", words)
    else:
        text_m = _TEXT_SUFFIX_RE.search(words)
        if text_m:
            trimmed = text_m.group(1).strip()
            spec.text = trimmed[:80]
            if len(trimmed) > 80:
                warnings.append("Text was shortened to 80 characters.")
            words = words[: text_m.start()]

    words = words.lower()

    found_colors: list[str] = []
    for cm in _COLOR_TOKEN_RE.finditer(words):
        tok = cm.group(0)
        mapped = NAMED_COLORS.get(tok) or (tok if _HEX6_LOWER_RE.fullmatch(tok) else None)
        if mapped and mapped not in found_colors:
            found_colors.append(mapped)
        elif not mapped:
            warnings.append(f"Ignored invalid color {tok}; use #RRGGBB.")

    if found_colors:
        chosen = found_colors[:5]
        if len(chosen) == 1:
            chosen.append("#202820" if chosen[0] == "#ffffff" else "#ffffff")
        spec.colors = chosen
        if len(found_colors) > 5:
            warnings.append("Only the first five colors are used.")

    words = _COLOR_TOKEN_RE.sub(" ", words)

    found_patterns: list[str] = []

    def _collect_pattern(m: re.Match[str]) -> str:
        w = m.group(0)
        found_patterns.append(_PATTERN_ALIASES.get(w, w))
        return " "

    words = _PATTERN_TOKEN_RE.sub(_collect_pattern, words)
    if found_patterns:
        spec.pattern = found_patterns[0]  # type: ignore[assignment]
    if len(set(found_patterns)) > 1:
        warnings.append("Multiple patterns found; the first one is used.")

    def _apply_numeric(m: re.Match[str]) -> str:
        key = m.group(1)
        raw_str = m.group(2)
        field_name = "rotation_deg" if key in ("rotate", "rotation") else key
        if field_name == "scale":
            min_v, max_v = 0.25, 4.0
        elif field_name == "seed":
            min_v, max_v = 0, 4294967295
        else:
            min_v, max_v = -180.0, 180.0

        raw = float(raw_str)
        if field_name == "seed":
            if math.isinf(raw):
                bounded_i = int(max_v) if raw > 0 else int(min_v)
            else:
                bounded_i = max(int(min_v), min(int(max_v), math.trunc(raw)))
            spec.seed = bounded_i
            if raw != bounded_i:
                warnings.append(
                    f"{field_name} adjusted to {_fmt_js_num(bounded_i)} "
                    f"(allowed {_fmt_js_num(min_v)}\u2013{_fmt_js_num(max_v)})."
                )
        else:
            bounded_f = max(float(min_v), min(float(max_v), raw))
            setattr(spec, field_name, bounded_f)
            if raw != bounded_f:
                warnings.append(
                    f"{field_name} adjusted to {_fmt_js_num(bounded_f)} "
                    f"(allowed {_fmt_js_num(min_v)}\u2013{_fmt_js_num(max_v)})."
                )
        return " "

    words = _NUMERIC_MOD_RE.sub(_apply_numeric, words)
    words = _STOP_WORDS_RE.sub(" ", words)

    unsupported_words = list(dict.fromkeys(_WORD_OR_HASH_RE.findall(words)))
    if unsupported_words:
        warnings.append(
            f"Unsupported wording: {', '.join(unsupported_words)}. "
            "Only the listed pattern, color, text and numeric terms affect the artwork."
        )
    if not prompt.strip():
        warnings.append("Empty prompt: the current recipe was kept.")

    try:
        validated = DesignSpec.from_dict(spec.to_dict())
    except ContractValidationError as exc:
        raise DesignValidationError(str(exc)) from exc
    return validated, warnings, unsupported_words


def parse_design_prompt(
    prompt: str,
    base: DesignSpec | Mapping[str, Any] | None = None,
) -> DesignSpec:
    """Parse a natural-language design prompt into a validated DesignSpec."""
    spec, _, _ = parse_design_prompt_detailed(prompt, base=base)
    return spec


def get_design_presets() -> list[dict[str, Any]]:
    """Return curated procedural studio presets spanning all 6 supported patterns."""
    raw_presets = [
        (
            "sunset_stripes",
            "Sunset Stripes",
            "Bold diagonal warm-cool studio stripes",
            'orange and navy and yellow stripes rotation -25 "STUDIO"',
            DesignSpec(
                pattern="stripes",
                colors=["#dc693f", "#203957", "#e9bb42"],
                text="STUDIO",
                rotation_deg=-25.0,
                scale=1.0,
                seed=731,
            ),
        ),
        (
            "studio_checker",
            "Editorial Checker",
            "Crisp architectural cream & forest checkerboard",
            "cream and green checker scale 1.1 rotation -12",
            DesignSpec(
                pattern="checker",
                colors=["#f4ebd8", "#42634b", "#e9bb42"],
                text="",
                rotation_deg=-12.0,
                scale=1.1,
                seed=108,
            ),
        ),
        (
            "pop_dots",
            "Pop Polka Dots",
            "Playful multi-radius halftone dots on ink canvas",
            "ink and pink and teal and yellow dots scale 0.9",
            DesignSpec(
                pattern="dots",
                colors=["#202820", "#e5a1ac", "#287e80", "#e9bb42"],
                text="",
                rotation_deg=10.0,
                scale=0.9,
                seed=404,
            ),
        ),
        (
            "ocean_waves",
            "Tidal Waves",
            "Flowing sinusoidal coastal ribbons",
            'navy and teal and cream waves rotation -8 "WAVE"',
            DesignSpec(
                pattern="waves",
                colors=["#203957", "#287e80", "#f4ebd8"],
                text="WAVE",
                rotation_deg=-8.0,
                scale=1.0,
                seed=219,
            ),
        ),
        (
            "blueprint_grid",
            "Blueprint Grid",
            "Precision technical grid lines with accent cells",
            "blue and white and yellow grid scale 0.85 rotation 0",
            DesignSpec(
                pattern="grid",
                colors=["#203957", "#ffffff", "#e9bb42"],
                text="GRID 01",
                rotation_deg=0.0,
                scale=0.85,
                seed=512,
            ),
        ),
        (
            "minimal_badge",
            "Minimal Badge",
            "Clean solid color field with high-contrast badge typography",
            'cream and red solid "VISUALIZE IT"',
            DesignSpec(
                pattern="solid",
                colors=["#f4ebd8", "#bd3f3f"],
                text="VISUALIZE IT",
                rotation_deg=0.0,
                scale=1.0,
                seed=42,
            ),
        ),
    ]
    out: list[dict[str, Any]] = []
    for preset_id, name, desc, prompt, spec in raw_presets:
        out.append(
            {
                "id": preset_id,
                "name": name,
                "description": desc,
                "prompt": prompt,
                "recipe": spec.to_recipe_dict(),
                "spec": spec.to_dict(),
            }
        )
    return out


def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    sz = max(10, int(font_size))
    for candidate in _FONT_CANDIDATES:
        if os.path.isfile(candidate):
            try:
                return ImageFont.truetype(candidate, sz)
            except OSError:
                continue
    return ImageFont.load_default()


def render_recipe_rgba(spec: DesignSpec | Mapping[str, Any]) -> np.ndarray:
    """Render a procedural DesignSpec recipe into a C-contiguous (H, W, 4) uint8 RGBA image."""
    try:
        design = spec if isinstance(spec, DesignSpec) else DesignSpec.from_dict(spec)
    except ContractValidationError as exc:
        raise DesignValidationError(str(exc)) from exc

    w, h = design.width, design.height
    palette = np.array([_hex_to_rgb_f32(c) for c in design.colors], dtype=np.float32)
    num_colors = len(palette)

    cell = max(8.0, 96.0 * design.scale * (min(w, h) / 1024.0))
    extent = math.hypot(w, h) * 0.5 + cell * 2.0
    rand = mulberry32(design.seed)
    phase_x = rand() * cell
    phase_y = rand() * cell

    start_cell = -math.ceil(extent / cell)
    end_cell = math.ceil(extent / cell)
    grid_span = end_cell - start_cell + 1

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - (w * 0.5)
    dy = yy - (h * 0.5)
    theta = math.radians(design.rotation_deg)
    cos_t = float(math.cos(theta))
    sin_t = float(math.sin(theta))
    # Inverse canvas transform (translate(w/2,h/2) -> rotate(theta) -> translate(phaseX, phaseY)).
    x_local = (dx * cos_t + dy * sin_t) - phase_x
    y_local = (-dx * sin_t + dy * cos_t) - phase_y

    rgb = np.empty((h, w, 3), dtype=np.float32)
    rgb[:] = palette[0]

    pattern = design.pattern
    if pattern == "solid":
        # Subtle multi-color inner frame accent when >2 colors are supplied on solid recipes
        if num_colors > 2:
            border = max(2, int(round(min(w, h) * 0.03 * min(design.scale, 1.5))))
            for extra_i in range(2, num_colors):
                inset = (extra_i - 2) * border
                if inset * 2 + border * 2 < min(w, h):
                    rgb[inset : inset + border, inset : w - inset] = palette[extra_i]
                    rgb[h - inset - border : h - inset, inset : w - inset] = palette[extra_i]
                    rgb[inset : h - inset, inset : inset + border] = palette[extra_i]
                    rgb[inset : h - inset, w - inset - border : w - inset] = palette[extra_i]
    elif pattern == "stripes":
        u_scaled = x_local / cell
        stripe_idx = np.mod(np.floor(u_scaled).astype(np.int64), num_colors)
        next_idx = np.mod(stripe_idx + 1, num_colors)
        frac = u_scaled - np.floor(u_scaled)
        aa = np.clip((frac - (1.0 - 1.0 / cell)) * cell, 0.0, 1.0)[:, :, None]
        rgb = (1.0 - aa) * palette[stripe_idx] + aa * palette[next_idx]
    elif pattern == "checker":
        gx = np.floor(x_local / cell).astype(np.int64)
        gy = np.floor(y_local / cell).astype(np.int64)
        idx = np.mod(gx + gy, num_colors)
        rgb = palette[idx]
    elif pattern == "dots":
        col_table = np.empty((grid_span, grid_span), dtype=np.int32)
        rad_table = np.empty((grid_span, grid_span), dtype=np.float32)
        fg_count = num_colors - 1
        for iy in range(grid_span):
            for ix in range(grid_span):
                col_table[iy, ix] = 1 + int(math.floor(rand() * fg_count))
                rad_table[iy, ix] = cell * (0.17 + rand() * 0.13)

        cx_i = np.rint(x_local / cell).astype(np.int32)
        cy_i = np.rint(y_local / cell).astype(np.int32)
        tx = np.clip(cx_i - start_cell, 0, grid_span - 1)
        ty = np.clip(cy_i - start_cell, 0, grid_span - 1)

        dist = np.hypot(x_local - cx_i.astype(np.float32) * cell, y_local - cy_i.astype(np.float32) * cell)
        radii = rad_table[ty, tx]
        cov = np.clip(radii + 0.5 - dist, 0.0, 1.0)[:, :, None]
        dot_colors = palette[col_table[ty, tx]]
        rgb = (1.0 - cov) * rgb + cov * dot_colors
    elif pattern == "grid":
        col_table = np.empty((grid_span, grid_span), dtype=np.int32)
        fg_count = num_colors - 1
        for iy in range(grid_span):
            for ix in range(grid_span):
                col_table[iy, ix] = 1 + int(math.floor(rand() * fg_count))

        gx = np.floor(x_local / cell).astype(np.int32)
        gy = np.floor(y_local / cell).astype(np.int32)
        tx = np.clip(gx - start_cell, 0, grid_span - 1)
        ty = np.clip(gy - start_cell, 0, grid_span - 1)

        fx_cell = x_local - gx.astype(np.float32) * cell
        fy_cell = y_local - gy.astype(np.float32) * cell
        d_edge = np.minimum(np.minimum(fx_cell, cell - fx_cell), np.minimum(fy_cell, cell - fy_cell))
        half_lw = max(1.0, cell * 0.045 * 0.5)
        cov = np.clip(half_lw + 0.5 - d_edge, 0.0, 1.0)[:, :, None]
        stroke_colors = palette[col_table[ty, tx]]
        rgb = (1.0 - cov) * rgb + cov * stroke_colors
    elif pattern == "waves":
        fg_count = num_colors - 1
        y_wave = y_local - np.sin(x_local / (cell * 0.75) + phase_x) * (cell * 0.30)
        wy = np.rint(y_wave / cell).astype(np.int64)
        d_wave = np.abs(y_wave - wy.astype(np.float32) * cell)
        half_lw = cell * 0.225
        cov = np.clip(half_lw + 0.5 - d_wave, 0.0, 1.0)[:, :, None]
        wave_idx = 1 + np.mod(wy, fg_count)
        rgb = (1.0 - cov) * rgb + cov * palette[wave_idx]

    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(rgb + 0.5, 0.0, 255.0).astype(np.uint8)
    rgba[:, :, 3] = 255

    if design.text:
        pil_img = Image.fromarray(rgba, mode="RGBA")
        draw = ImageDraw.Draw(pil_img)

        font_size = max(14, int(round(min(w, h) * 0.12)))
        stroke_w = max(2, font_size // 14)
        font = _load_font(font_size)

        bbox = draw.textbbox((0, 0), design.text, font=font, stroke_width=stroke_w)
        text_w = max(1, bbox[2] - bbox[0])
        if text_w > w * 0.84 and isinstance(font, ImageFont.FreeTypeFont):
            font_size = max(10, int(font_size * (w * 0.84) / text_w))
            stroke_w = max(1, font_size // 14)
            font = _load_font(font_size)
            bbox = draw.textbbox((0, 0), design.text, font=font, stroke_width=stroke_w)
            text_w = max(1, bbox[2] - bbox[0])

        text_h = max(1, bbox[3] - bbox[1])
        pad_x = max(8, int(round(font_size * 0.32)))
        pad_y = max(6, int(round(font_size * 0.28)))
        rx0 = max(2, int(round(w * 0.5 - text_w * 0.5 - pad_x)))
        ry0 = max(2, int(round(h * 0.5 - text_h * 0.5 - pad_y)))
        rx1 = min(w - 2, int(round(w * 0.5 + text_w * 0.5 + pad_x)))
        ry1 = min(h - 2, int(round(h * 0.5 + text_h * 0.5 + pad_y)))

        bg_hex = design.colors[0]
        fg_hex = design.colors[1]
        fg_r, fg_g, fg_b = _hex_to_rgb_f32(fg_hex)
        fg_lum = 0.299 * fg_r + 0.587 * fg_g + 0.114 * fg_b
        outline_hex = "#111111" if fg_lum > 145.0 else "#ffffff"

        draw.rounded_rectangle(
            [rx0, ry0, rx1, ry1],
            radius=max(4, font_size // 4),
            fill=bg_hex,
            outline=fg_hex,
            width=max(2, font_size // 18),
        )

        cx_mid = w * 0.5 - (bbox[0] + bbox[2]) * 0.5
        cy_mid = h * 0.5 - (bbox[1] + bbox[3]) * 0.5
        draw.text(
            (cx_mid, cy_mid),
            design.text,
            font=font,
            fill=fg_hex,
            stroke_width=stroke_w,
            stroke_fill=outline_hex,
        )
        rgba = np.ascontiguousarray(np.asarray(pil_img, dtype=np.uint8))

    return rgba


def _validate_png_stream(data: bytes) -> None:
    """Strictly verify PNG chunk structure, CRCs, IHDR bounds, non-APNG, and IEND termination."""
    offset = 8
    index = 0
    ended = False
    total = len(data)
    while offset < total:
        if offset + 8 > total:
            raise DesignValidationError("Truncated PNG chunk header.")
        length, kind = struct.unpack_from(">I4s", data, offset)
        end = offset + 12 + length
        if end > total or length > MAX_UPLOAD_BYTES:
            raise DesignValidationError("Truncated or oversized PNG chunk.")
        payload = data[offset + 8 : offset + 8 + length]
        crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        if (zlib.crc32(kind + payload) & 0xFFFFFFFF) != crc:
            raise DesignValidationError(f"Corrupt PNG chunk CRC in {kind!r}.")
        if index == 0:
            if kind != b"IHDR" or length != 13:
                raise DesignValidationError("PNG missing valid IHDR header.")
            width, height = struct.unpack_from(">II", payload)
            if width < 1 or height < 1 or width > MAX_SOURCE_DIM or height > MAX_SOURCE_DIM:
                raise DesignValidationError(f"PNG dimensions ({width}x{height}) out of allowed bounds.")
        elif kind == b"IHDR":
            raise DesignValidationError("Duplicate PNG IHDR chunk.")
        if kind == b"acTL":
            raise DesignValidationError("Animated PNG (APNG) is not supported.")
        offset = end
        index += 1
        if kind == b"IEND":
            if length != 0 or offset != total:
                raise DesignValidationError("Invalid PNG IEND trailer.")
            ended = True
            break
    if not ended:
        raise DesignValidationError("PNG missing IEND chunk.")


def _resize_rgba_premultiplied(rgba: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    f = rgba.astype(np.float32)
    alpha = f[:, :, 3:4] * (1.0 / 255.0)
    premult = np.empty_like(f)
    premult[:, :, :3] = f[:, :, :3] * alpha
    premult[:, :, 3:4] = f[:, :, 3:4]

    pil_pre = Image.fromarray(np.clip(premult + 0.5, 0.0, 255.0).astype(np.uint8), mode="RGBA")
    pil_resized = pil_pre.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    res_f = np.asarray(pil_resized, dtype=np.float32)

    out = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    out_a = res_f[:, :, 3]
    vis = out_a > 0.5
    if np.any(vis):
        scale = 255.0 / out_a[vis][:, None]
        out_rgb = np.clip(res_f[:, :, :3][vis] * scale + 0.5, 0.0, 255.0).astype(np.uint8)
        out[:, :, :3][vis] = out_rgb
        out[:, :, 3][vis] = np.clip(out_a[vis] + 0.5, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(out)


def load_design_image_with_info(
    path_or_bytes: str | os.PathLike[str] | bytes | bytearray | memoryview,
    *,
    max_dim: int = 2048,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate and load PNG/JPEG/WebP custom artwork as RGBA uint8 alongside metadata."""
    if isinstance(max_dim, bool) or not isinstance(max_dim, int) or max_dim < 16:
        raise DesignValidationError("max_dim must be an integer >= 16.")

    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        raw = bytes(path_or_bytes)
    elif isinstance(path_or_bytes, (str, os.PathLike)):
        p = Path(path_or_bytes)
        if not p.exists():
            raise DesignValidationError(f"Artwork file does not exist: {p}")
        if not p.is_file():
            raise DesignValidationError(f"Artwork path is not a regular file: {p}")
        try:
            size = p.stat().st_size
            if size == 0:
                raise DesignValidationError(f"Artwork file is empty: {p}")
            if size > MAX_UPLOAD_BYTES:
                raise DesignValidationError(f"Artwork file exceeds {MAX_UPLOAD_BYTES} bytes: {p}")
            raw = p.read_bytes()
        except OSError as exc:
            raise DesignValidationError(f"Failed reading artwork file {p}: {exc}") from exc
    else:
        raise DesignValidationError(
            f"Expected file path or bytes for artwork, got {type(path_or_bytes).__name__}."
        )

    if len(raw) == 0:
        raise DesignValidationError("Artwork data is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise DesignValidationError(f"Artwork data exceeds {MAX_UPLOAD_BYTES} bytes.")

    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        _validate_png_stream(raw)

    allowed_formats = {"PNG", "JPEG", "WEBP"}
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            fmt = (probe.format or "").upper()
            if fmt not in allowed_formats:
                raise DesignValidationError(
                    f"Unsupported image format {fmt or 'UNKNOWN'!r}; expected PNG, JPEG, or WebP."
                )
            if getattr(probe, "is_animated", False) and getattr(probe, "n_frames", 1) > 1:
                raise DesignValidationError("Animated images are not supported.")
            w0, h0 = probe.size
            if w0 < 1 or h0 < 1 or w0 > MAX_SOURCE_DIM or h0 > MAX_SOURCE_DIM or (w0 * h0) > MAX_SOURCE_PIXELS:
                raise DesignValidationError(f"Image dimensions ({w0}x{h0}) exceed safe limits.")
            probe.verify()

        with Image.open(io.BytesIO(raw)) as decoded:
            decoded.load()
            transposed = ImageOps.exif_transpose(decoded)
            has_alpha = bool(
                transposed.mode in ("RGBA", "LA", "PA")
                or (transposed.mode == "P" and "transparency" in transposed.info)
            )
            orig_w, orig_h = transposed.size
            rgba_img = transposed.convert("RGBA")
            rgba = np.ascontiguousarray(np.asarray(rgba_img, dtype=np.uint8))
    except DesignValidationError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
        struct.error,
        zlib.error,
         Image.DecompressionBombError,
    ) as exc:
        raise DesignValidationError(f"Invalid or corrupt artwork image: {exc}") from exc

    cur_h, cur_w = rgba.shape[:2]
    if max(cur_w, cur_h) > max_dim:
        scale = float(max_dim) / float(max(cur_w, cur_h))
        new_w = max(1, int(round(cur_w * scale)))
        new_h = max(1, int(round(cur_h * scale)))
        rgba = _resize_rgba_premultiplied(rgba, new_w, new_h)

    out_h, out_w = rgba.shape[:2]
    info = {
        "width": int(out_w),
        "height": int(out_h),
        "orig_width": int(orig_w),
        "orig_height": int(orig_h),
        "format": fmt,
        "has_alpha": has_alpha,
    }
    return rgba, info


def load_design_image(
    path_or_bytes: str | os.PathLike[str] | bytes | bytearray | memoryview,
    *,
    max_dim: int = 2048,
) -> np.ndarray:
    """Validate and load PNG/JPEG/WebP custom artwork as a C-contiguous (H, W, 4) uint8 RGBA array."""
    rgba, _ = load_design_image_with_info(path_or_bytes, max_dim=max_dim)
    return rgba


def render_design(spec: DesignSpec | Mapping[str, Any], *, max_dim: int = 2048) -> np.ndarray:
    """Unified entrypoint rendering either a procedural recipe or custom image DesignSpec to RGBA uint8."""
    try:
        design = spec if isinstance(spec, DesignSpec) else DesignSpec.from_dict(spec)
    except ContractValidationError as exc:
        raise DesignValidationError(str(exc)) from exc

    if design.source == "image":
        assert design.image_path is not None
        return load_design_image(design.image_path, max_dim=max_dim)
    if design.prompt and design.text == "" and design.colors == ["#ff595e", "#1982c4", "#ffca3a"]:
        design = parse_design_prompt(design.prompt, base=design)
    return render_recipe_rgba(design)
