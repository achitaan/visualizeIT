"""Milestone 1 Foundation Unit & Performance Suite (F1, F2, F3, F4, F5)."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import warnings
import zlib

import cv2
import numpy as np
from PIL import Image
import pytest

from bench.renderer import ray_triangle
from visualizeit import (
    ContractValidationError,
    DemoClipInfo,
    DesignSpec,
    PlacementSpec,
    SelectionSpec,
    TrackFrameResult,
    VideoRunReport,
    __version__,
)
from visualizeit.demo_assets import (
    DEFAULT_DEMO_DIR,
    DEMO_CLIP_IDS,
    generate_demo_corpus,
    load_demo_manifest,
    main as demo_assets_main,
)
from visualizeit.designs import (
    DesignValidationError,
    PATTERNS,
    get_design_presets,
    load_design_image,
    load_design_image_with_info,
    mulberry32,
    parse_design_prompt,
    parse_design_prompt_detailed,
    render_design,
    render_recipe_rgba,
)
from visualizeit.native_raster import (
    cv_gltf_pose_to_unity,
    is_native_raster_available,
    rasterize_mesh_rgbd,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_TMP_DIR = REPO_ROOT / ".cache" / "tmp"


def _subproc_env() -> dict[str, str]:
    CACHE_TMP_DIR.mkdir(parents=True, exist_ok=True)
    return {
        **os.environ,
        "TMPDIR": str(CACHE_TMP_DIR),
        "TEMP": str(CACHE_TMP_DIR),
        "TMP": str(CACHE_TMP_DIR),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


class TestContracts:
    """Verify F3 / PROJECT.md §3 typed schemas, round-trips, and fail-fast invariants."""

    def test_package_version_and_tmp_path_discipline(self, tmp_path: Path) -> None:
        assert __version__ == "1.0.0"
        resolved = tmp_path.resolve().as_posix()
        assert not resolved.startswith("/tmp/")
        assert ".agents/teamwork" not in resolved

    def test_design_spec_round_trip_and_image_mode_empty_colors(self) -> None:
        spec = DesignSpec(
            source="recipe",
            pattern="waves",
            colors=["#FF595E", "teal", "#ffca3a"],
            text="SURF 26",
            prompt="waves surf",
            width=640,
            height=480,
            seed=777,
            scale=1.25,
            rotation_deg=-15.0,
        )
        assert spec.colors == ["#ff595e", "#287e80", "#ffca3a"]
        raw = json.dumps(spec.to_dict(), allow_nan=False)
        restored = DesignSpec.from_dict(json.loads(raw))
        assert restored == spec

        recipe = spec.to_recipe_dict()
        assert set(recipe.keys()) == {
            "pattern",
            "colors",
            "text",
            "scale",
            "rotation_deg",
            "seed",
            "width",
            "height",
        }

        img_spec = DesignSpec(source="image", colors=[], image_path="/path/to/art.png")
        assert img_spec.source == "image"
        assert len(img_spec.colors) >= 2
        assert DesignSpec.from_dict(img_spec.to_dict()) == img_spec

    @pytest.mark.parametrize(
        "bad_kwargs",
        [
            {"source": "svg"},
            {"pattern": "plaid"},
            {"colors": ["#112233"]},
            {"colors": ["#112233", "#223344", "#334455", "#445566", "#556677", "#667788"]},
            {"colors": ["#fff", "#000000"]},
            {"colors": ["#zzzzzz", "#000000"]},
            {"text": "x" * 81},
            {"prompt": "p" * 501},
            {"scale": 0.1},
            {"scale": 4.5},
            {"scale": True},
            {"scale": float("nan")},
            {"rotation_deg": -181.0},
            {"rotation_deg": 181.0},
            {"seed": -1},
            {"seed": 2**32},
            {"seed": True},
            {"seed": 1.5},  # type: ignore[dict-item]
            {"source": "image", "image_path": None},
            {"source": "image", "image_path": "   "},
        ],
    )
    def test_design_spec_rejects_invalid_inputs(self, bad_kwargs: dict) -> None:
        with pytest.raises(ContractValidationError):
            DesignSpec(**bad_kwargs)

    def test_placement_and_selection_round_trip_and_validation(self) -> None:
        p = PlacementSpec(
            wrap_mode="cylinder",
            blend_mode="luminance",
            opacity=0.85,
            scale=1.2,
            rotation_deg=15.0,
            offset_x=0.1,
            offset_y=-0.1,
            cylinder_radius_frac=0.6,
            tile_repeat=True,
        )
        assert PlacementSpec.from_dict(json.loads(json.dumps(p.to_dict(), allow_nan=False))) == p

        for bad_p in [
            {"wrap_mode": "sphere"},
            {"blend_mode": "screen"},
            {"opacity": -0.01},
            {"opacity": 1.01},
            {"opacity": float("nan")},
            {"opacity": True},
            {"scale": 0.05},
            {"scale": 5.5},
            {"tile_repeat": 1},  # type: ignore[dict-item]
        ]:
            with pytest.raises(ContractValidationError):
                PlacementSpec(**bad_p)

        sel = SelectionSpec(
            frame_index=0,
            positive_points=[(120.5, 200.0)],
            negative_points=[(10.0, 15.0)],
            bbox=(80.0, 120.0, 200.0, 160.0),
        )
        assert sel.has_target is True
        sel.validate_for_frame(640, 480, frame_count=180)
        assert SelectionSpec.from_dict(json.loads(json.dumps(sel.to_dict(), allow_nan=False))) == sel

        for bad_s in [
            {"frame_index": -1},
            {"frame_index": True},
            {"positive_points": [(-1.0, 5.0)]},
            {"bbox": (10.0, 10.0, 0.0, 20.0)},
            {"bbox": (10.0, 10.0, 20.0, -5.0)},
        ]:
            with pytest.raises(ContractValidationError):
                SelectionSpec(**bad_s)

        empty_sel = SelectionSpec()
        assert empty_sel.has_target is False
        with pytest.raises(ContractValidationError):
            empty_sel.validate_for_frame(640, 480, frame_count=10)

    def test_track_frame_result_suppression_invariants(self) -> None:
        h_mat = np.eye(3, dtype=np.float64)
        ok_res = TrackFrameResult(
            frame_index=5,
            timestamp_s=5.0 / 30.0,
            state="tracking",
            homography_3x3=h_mat,
            pose_rvec=np.array([0.1, -0.2, 0.05]),
            pose_tvec=np.array([0.0, 0.0, 0.45]),
            inliers=64,
            confidence=0.92,
            track_ms=4.1,
        )
        rt = TrackFrameResult.from_dict(json.loads(json.dumps(ok_res.to_dict(), allow_nan=False)))
        assert rt.state == "tracking"
        assert np.allclose(rt.homography_3x3, h_mat)
        assert np.allclose(rt.pose_tvec, [0.0, 0.0, 0.45])

        # Non-tracking states MUST strictly suppress all transforms
        for st in ("lost", "recovering"):
            clean_lost = TrackFrameResult(frame_index=10, timestamp_s=0.33, state=st)  # type: ignore[arg-type]
            assert clean_lost.homography_3x3 is None
            with pytest.raises(ContractValidationError):
                TrackFrameResult(frame_index=10, timestamp_s=0.33, state=st, homography_3x3=h_mat)  # type: ignore[arg-type]
            with pytest.raises(ContractValidationError):
                TrackFrameResult(
                    frame_index=10,
                    timestamp_s=0.33,
                    state=st,  # type: ignore[arg-type]
                    pose_rvec=np.zeros(3),
                    pose_tvec=np.ones(3),
                )

        # Malformed homography or half-pose
        with pytest.raises(ContractValidationError):
            TrackFrameResult(frame_index=0, timestamp_s=0.0, state="tracking", homography_3x3=np.eye(2))
        with pytest.raises(ContractValidationError):
            TrackFrameResult(
                frame_index=0,
                timestamp_s=0.0,
                state="tracking",
                homography_3x3=np.array([[1.0, np.nan, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            )
        with pytest.raises(ContractValidationError):
            TrackFrameResult(frame_index=0, timestamp_s=0.0, state="tracking", pose_rvec=np.zeros(3))

    def test_track_frame_result_bool_str_rejection_isolation_and_from_dict_keys(self) -> None:
        # 1. Reject bool, mixed bool, string, complex, object, and scalar arrays across all matrix fields
        for bad_mat in [
            np.eye(3, dtype=bool),
            [[True, False, False], [False, True, False], [False, False, True]],
            [[1.0, True, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]],
            np.eye(3, dtype=np.complex128),
            "1,0,0,0,1,0,0,0,1",
            42.0,
        ]:
            with pytest.raises(ContractValidationError):
                TrackFrameResult(0, 0.0, "tracking", homography_3x3=bad_mat)  # type: ignore[arg-type]

        with pytest.raises(ContractValidationError):
            TrackFrameResult(0, 0.0, "tracking", pose_rvec=[True, False, False], pose_tvec=[0.0, 0.0, 1.0])  # type: ignore[arg-type]
        with pytest.raises(ContractValidationError):
            TrackFrameResult(
                0,
                0.0,
                "tracking",
                grid_src_pts=np.ones((4, 2), dtype=bool),
                grid_dst_pts=np.ones((4, 2), dtype=np.float64),
            )

        # 2. Mutating caller scratch arrays after construction MUST NOT corrupt TrackFrameResult
        h_scratch = np.eye(3, dtype=np.float64)
        rvec_scratch = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        tvec_scratch = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        src_scratch = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], dtype=np.float64)
        dst_scratch = np.array([[1.0, 1.0], [11.0, 1.0], [11.0, 11.0]], dtype=np.float64)

        isolated = TrackFrameResult(
            frame_index=1,
            timestamp_s=1.0 / 30.0,
            state="tracking",
            homography_3x3=h_scratch,
            pose_rvec=rvec_scratch,
            pose_tvec=tvec_scratch,
            grid_src_pts=src_scratch,
            grid_dst_pts=dst_scratch,
        )
        h_scratch[:, :] = np.nan
        rvec_scratch[:] = np.nan
        tvec_scratch[:] = np.inf
        src_scratch[:, :] = np.nan
        dst_scratch[:, :] = -np.inf

        encoded = json.dumps(isolated.to_dict(), allow_nan=False)
        round_tripped = TrackFrameResult.from_dict(json.loads(encoded))
        assert np.allclose(round_tripped.homography_3x3, np.eye(3))
        assert np.allclose(round_tripped.pose_rvec, [0.1, 0.2, 0.3])
        assert np.allclose(round_tripped.pose_tvec, [0.0, 0.0, 1.0])
        assert np.allclose(round_tripped.grid_dst_pts, [[1.0, 1.0], [11.0, 1.0], [11.0, 11.0]])

        # 3. TrackFrameResult.from_dict missing/unknown key and non-mapping validation
        with pytest.raises(ContractValidationError):
            TrackFrameResult.from_dict({})
        with pytest.raises(ContractValidationError):
            TrackFrameResult.from_dict({"frame_index": 0, "timestamp_s": 0.0})
        with pytest.raises(ContractValidationError):
            TrackFrameResult.from_dict({"frame_index": 0, "timestamp_s": 0.0, "state": "lost", "bogus": 1})
        with pytest.raises(ContractValidationError):
            TrackFrameResult.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_video_run_report_24_keys_and_anti_cheating_gates(self) -> None:
        base_kwargs = {
            "input_path": "/videos/in.mp4",
            "output_path": "/videos/out.mp4",
            "width": 640,
            "height": 480,
            "source_fps": 30.0,
            "wrap_mode": "planar",
            "blend_mode": "luminance",
            "raster_backend": "native_cpp",
            "render_stride": 1,
            "input_frame_count": 180,
            "processed_frame_count": 180,
            "rendered_frame_count": 180,
            "output_frame_count": 180,
            "tracked_frame_count": 175,
            "lost_frame_count": 5,
            "recovered_count": 1,
            "tracking_availability": 175.0 / 180.0,
            "wall_time_seconds": 5.0,
            "effective_fps": 36.0,
            "steady_state_fps": 38.5,
            "frame_latency_ms": {"mean": 27.7, "p50": 26.5, "p95": 33.0, "p99": 38.0, "max": 42.0},
            "stage_latency_mean_ms": {
                "decode_ms": 2.5,
                "prep_ms": 1.2,
                "track_ms": 8.4,
                "composite_ms": 3.1,
                "encode_ms": 5.2,
            },
            "inlier_count_median": 88.0,
            "reprojection_rmse_px": 1.15,
        }
        report = VideoRunReport(**base_kwargs)
        as_dict = report.to_dict()
        assert len(as_dict) == 24
        assert list(as_dict.keys()) == list(base_kwargs.keys())
        assert VideoRunReport.from_dict(json.loads(json.dumps(as_dict, allow_nan=False))) == report

        for bad_stride in (0, 2, 5):
            with pytest.raises(ContractValidationError):
                VideoRunReport(**{**base_kwargs, "render_stride": bad_stride})

        with pytest.raises(ContractValidationError):
            VideoRunReport(**{**base_kwargs, "processed_frame_count": 179, "rendered_frame_count": 179})
        with pytest.raises(ContractValidationError):
            VideoRunReport(**{**base_kwargs, "output_frame_count": 179})
        with pytest.raises(ContractValidationError):
            VideoRunReport(**{**base_kwargs, "frame_latency_ms": {"mean": 10.0}})

    def test_demo_clip_info_exact_8_keys(self) -> None:
        clip = DemoClipInfo(
            id="planar_card_640x480",
            path="/videos/planar_card_640x480.mp4",
            width=640,
            height=480,
            fps=30.0,
            frame_count=180,
            default_selection=SelectionSpec(frame_index=0, positive_points=[(320.0, 240.0)]),
            recommended_wrap="planar",
        )
        d = clip.to_dict()
        assert set(d.keys()) == {
            "id",
            "path",
            "width",
            "height",
            "fps",
            "frame_count",
            "default_selection",
            "recommended_wrap",
        }
        assert DemoClipInfo.from_dict(json.loads(json.dumps(d, allow_nan=False))) == clip


class TestDesignsAndArtworkLoader:
    """Verify F3 procedural design engine + JS parity and F4 custom RGBA artwork loader."""

    def test_all_patterns_determinism_and_parameter_sensitivity(self) -> None:
        for pat in PATTERNS:
            spec_a = DesignSpec(pattern=pat, width=256, height=256, seed=731)  # type: ignore[arg-type]
            spec_b = DesignSpec(pattern=pat, width=256, height=256, seed=731)  # type: ignore[arg-type]
            img_a = render_recipe_rgba(spec_a)
            img_b = render_recipe_rgba(spec_b)
            assert img_a.shape == (256, 256, 4)
            assert img_a.dtype == np.uint8
            assert img_a.flags["C_CONTIGUOUS"]
            assert np.all(img_a[:, :, 3] == 255)
            assert np.array_equal(img_a, img_b)

            if pat != "solid":
                img_seed = render_recipe_rgba(DesignSpec(pattern=pat, width=256, height=256, seed=732))  # type: ignore[arg-type]
                assert not np.array_equal(img_a, img_seed)

        # Scale, rotation, and 2..5 palette colors affect rendered pixels
        base = DesignSpec(pattern="dots", colors=["#202820", "#dc693f"], width=256, height=256, seed=42)
        r_base = render_recipe_rgba(base)
        r_scale = render_recipe_rgba(DesignSpec(**{**base.to_dict(), "scale": 2.0}))
        r_rot = render_recipe_rgba(DesignSpec(**{**base.to_dict(), "rotation_deg": 35.0}))
        r_five = render_recipe_rgba(
            DesignSpec(
                **{
                    **base.to_dict(),
                    "colors": ["#202820", "#dc693f", "#376aa0", "#e9bb42", "#e5a1ac"],
                }
            )
        )
        assert not np.array_equal(r_base, r_scale)
        assert not np.array_equal(r_base, r_rot)
        assert not np.array_equal(r_base, r_five)

    def test_text_overlay_and_presets(self) -> None:
        no_txt = render_recipe_rgba(DesignSpec(pattern="stripes", text="", width=320, height=320))
        with_txt = render_recipe_rgba(
            DesignSpec(pattern="stripes", text="<img src=x> STUDIO 😀", width=320, height=320)
        )
        # Center ROI must differ when text banner is rendered
        assert not np.array_equal(no_txt[120:200, 40:280], with_txt[120:200, 40:280])

        presets = get_design_presets()
        assert len(presets) >= 6
        seen_patterns = set()
        for preset in presets:
            seen_patterns.add(preset["recipe"]["pattern"])
            rgba = render_recipe_rgba(preset["spec"])
            assert rgba.shape == (512, 512, 4)
        assert seen_patterns == set(PATTERNS)

    def test_mulberry32_and_prompt_parser_node_js_parity(self) -> None:
        js_engine = REPO_ROOT / "bench" / "web" / "design-studio" / "design-engine.js"
        prompts = [
            "blue and #ffcc00 dots, scale 2 rotation 45 seed 42 text Hello world",
            'red waves "<img src=x onerror=alert(1)>"',
            "photorealistic dragon with green stripes",
            f'dots scale 900 rotation -999 seed 1.9 "{"😀" * 81}"',
            "dots waves #fff #zzzzzz blue",
            "white solid",
            "   ",
            "navy and cream and orange checkered rotate -30 scale 0.5",
            "seed " + ("9" * 309),
            "seed -" + ("9" * 309),
            "scale " + ("9" * 309),
            "rotate -" + ("9" * 309),
        ]
        node_script = (
            "const E = require(process.argv[1]);"
            "const prompts = JSON.parse(process.argv[2]);"
            "const r = E.seededRandom(731);"
            "const seq = [r(), r(), r(), r(), r()];"
            "const parsed = prompts.map(p => E.parsePrompt(p, E.DEFAULT));"
            "console.log(JSON.stringify({seq, parsed, def: E.DEFAULT}));"
        )
        proc = subprocess.run(
            ["node", "-e", node_script, str(js_engine), json.dumps(prompts)],
            check=True,
            capture_output=True,
            text=True,
            env=_subproc_env(),
        )
        js_out = json.loads(proc.stdout)

        py_rng = mulberry32(731)
        py_seq = [py_rng() for _ in range(5)]
        assert np.allclose(py_seq, js_out["seq"], atol=1e-15, rtol=0.0)

        js_default = DesignSpec(**js_out["def"])
        snapshot_before = js_default.to_dict()

        for prompt_str, js_item in zip(prompts, js_out["parsed"]):
            spec, warnings, unsupported = parse_design_prompt_detailed(prompt_str, base=js_default)
            assert js_default.to_dict() == snapshot_before, "base DesignSpec must not be mutated"
            assert spec.to_recipe_dict() == js_item["recipe"]
            assert unsupported == js_item["unsupportedWords"]
            assert warnings == js_item["warnings"]

        with pytest.raises(DesignValidationError):
            parse_design_prompt("x" * 501)
        with pytest.raises(DesignValidationError):
            parse_design_prompt(None)  # type: ignore[arg-type]

    def test_custom_rgba_artwork_loader_formats_alpha_and_aspect_resize(self, tmp_path: Path) -> None:
        # 1. RGBA PNG with transparent corner (0), semi-transparent band (128), opaque center (255)
        src = np.zeros((160, 240, 4), dtype=np.uint8)
        src[:, :, 0] = 220
        src[:, :, 1] = 80
        src[:, :, 2] = 40
        src[:, :, 3] = 255
        src[:40, :40, 3] = 0
        src[40:80, :80, 3] = 128

        png_path = tmp_path / "badge_rgba.png"
        Image.fromarray(src, mode="RGBA").save(png_path, format="PNG")
        loaded_png, info_png = load_design_image_with_info(png_path)
        assert info_png["format"] == "PNG"
        assert info_png["has_alpha"] is True
        assert loaded_png.shape == (160, 240, 4)
        assert int(loaded_png[10, 10, 3]) == 0
        assert int(loaded_png[60, 40, 3]) == 128
        assert int(loaded_png[120, 160, 3]) == 255

        # Also verify render_design(source="image")
        via_spec = render_design(DesignSpec(source="image", colors=[], image_path=str(png_path)))
        assert np.array_equal(via_spec, loaded_png)

        # 2. RGB JPEG -> has_alpha False, alpha channel promoted to 255
        jpg_path = tmp_path / "photo.jpg"
        Image.fromarray(src[:, :, :3], mode="RGB").save(jpg_path, format="JPEG", quality=95)
        loaded_jpg, info_jpg = load_design_image_with_info(jpg_path)
        assert info_jpg["format"] == "JPEG"
        assert info_jpg["has_alpha"] is False
        assert loaded_jpg.shape == (160, 240, 4)
        assert np.all(loaded_jpg[:, :, 3] == 255)

        # 3. RGBA WebP & RGB WebP
        webp_rgba_path = tmp_path / "sticker.webp"
        Image.fromarray(src, mode="RGBA").save(webp_rgba_path, format="WEBP", lossless=True)
        loaded_webp, info_webp = load_design_image_with_info(webp_rgba_path)
        assert info_webp["format"] == "WEBP"
        assert info_webp["has_alpha"] is True
        assert int(loaded_webp[5, 5, 3]) == 0

        # 4. Aspect-ratio preserving resize (1200w x 600h -> 300w x 150h)
        wide = np.full((600, 1200, 4), 200, dtype=np.uint8)
        wide_buf = io.BytesIO()
        Image.fromarray(wide, mode="RGBA").save(wide_buf, format="PNG")
        resized, info_res = load_design_image_with_info(wide_buf.getvalue(), max_dim=300)
        assert resized.shape == (150, 300, 4)
        assert info_res["orig_width"] == 1200 and info_res["orig_height"] == 600

    def test_custom_artwork_loader_rejects_corrupt_and_unsupported_inputs(self, tmp_path: Path) -> None:
        # Missing file & directory
        with pytest.raises(DesignValidationError):
            load_design_image(tmp_path / "does_not_exist.png")
        with pytest.raises(DesignValidationError):
            load_design_image(tmp_path)

        # Empty file & empty bytes
        empty_file = tmp_path / "empty.png"
        empty_file.write_bytes(b"")
        with pytest.raises(DesignValidationError):
            load_design_image(empty_file)
        with pytest.raises(DesignValidationError):
            load_design_image(b"")

        # Valid PNG mutated with truncated bytes & bit-flipped IDAT CRC
        valid_buf = io.BytesIO()
        Image.new("RGBA", (64, 64), (10, 20, 30, 255)).save(valid_buf, format="PNG")
        valid_png = valid_buf.getvalue()

        with pytest.raises(DesignValidationError):
            load_design_image(valid_png[:-10])

        corrupted_crc = bytearray(valid_png)
        corrupted_crc[-15] ^= 0xFF
        with pytest.raises(DesignValidationError):
            load_design_image(bytes(corrupted_crc))

        # Synthetic APNG with injected acTL chunk
        ihdr_end = 8 + 12 + 13
        actl_payload = struct.pack(">II", 2, 0)
        actl_crc = zlib.crc32(b"acTL" + actl_payload) & 0xFFFFFFFF
        actl_chunk = struct.pack(">I4s", 8, b"acTL") + actl_payload + struct.pack(">I", actl_crc)
        apng_bytes = valid_png[:ihdr_end] + actl_chunk + valid_png[ihdr_end:]
        with pytest.raises(DesignValidationError):
            load_design_image(apng_bytes)

        # Corrupt JPEG, plain text/JSON, and unsupported BMP/GIF
        with pytest.raises(DesignValidationError):
            load_design_image(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00corrupt_stream")
        with pytest.raises(DesignValidationError):
            load_design_image(b'{"not": "an image"}')

        bmp_buf = io.BytesIO()
        Image.new("RGB", (32, 32), (100, 100, 100)).save(bmp_buf, format="BMP")
        with pytest.raises(DesignValidationError):
            load_design_image(bmp_buf.getvalue())


class TestNativeRaster:
    """Verify F2 C++17 AVX-512 rasterizer, OpenCV fallback parity, ray oracle, and <=4.0 ms HD gate."""

    def test_native_library_and_pose_abi_preserved(self) -> None:
        assert is_native_raster_available(auto_build=False) is True
        pose_cv = np.array(
            [
                [1.0, 0.0, 0.0, 0.12],
                [0.0, 1.0, 0.0, -0.08],
                [0.0, 0.0, 1.0, 0.60],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        unity_pose = cv_gltf_pose_to_unity(pose_cv)
        round_trip = cv_gltf_pose_to_unity(unity_pose)
        assert np.allclose(round_trip, pose_cv, atol=1e-6)

        bad_pose = pose_cv.copy()
        bad_pose[0, 0] = 2.0
        with pytest.raises(ValueError):
            cv_gltf_pose_to_unity(bad_pose)

    @pytest.mark.parametrize("blend_mode", ["normal", "luminance", "ink"])
    @pytest.mark.parametrize("wrap_repeat", [False, True])
    def test_native_cpp_vs_opencv_fallback_parity(self, blend_mode: str, wrap_repeat: bool) -> None:
        w, h = 320, 240
        verts = np.array(
            [
                [-0.18, -0.14, 0.85],
                [0.20, -0.12, 1.05],
                [0.19, 0.15, 1.15],
                [-0.17, 0.13, 0.92],
            ],
            dtype=np.float32,
        )
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        uvs = (
            np.array([[-0.25, -0.25], [1.25, -0.25], [1.25, 1.25], [-0.25, 1.25]], dtype=np.float32)
            if wrap_repeat
            else np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        )
        tex = render_recipe_rgba(DesignSpec(pattern="checker", width=128, height=128, seed=99))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base_frame = np.stack(
            [
                90.0 + 90.0 * (xx / w),
                100.0 + 70.0 * (yy / h),
                120.0 + 40.0 * np.sin(xx * 0.03),
            ],
            axis=-1,
        ).astype(np.uint8)

        f_cpp = base_frame.copy()
        f_cv = base_frame.copy()
        intr = (280.0, 280.0, 160.0, 120.0)

        res_cpp = rasterize_mesh_rgbd(
            verts,
            tris,
            uvs,
            tex,
            intr,
            f_cpp,
            wrap_repeat=wrap_repeat,
            opacity=0.88,
            blend_mode=blend_mode,
            backend="native_cpp",
        )
        res_cv = rasterize_mesh_rgbd(
            verts,
            tris,
            uvs,
            tex,
            intr,
            f_cv,
            wrap_repeat=wrap_repeat,
            opacity=0.88,
            blend_mode=blend_mode,
            backend="opencv",
        )
        assert res_cpp.backend_used == "native_cpp"
        assert res_cv.backend_used == "opencv"

        m_cpp = res_cpp.mask > 0
        m_cv = res_cv.mask > 0
        iou = float(np.count_nonzero(m_cpp & m_cv)) / float(max(1, np.count_nonzero(m_cpp | m_cv)))
        assert iou >= 0.98

        interior = cv2.erode((m_cpp & m_cv).astype(np.uint8) * 255, np.ones((5, 5), np.uint8)) > 0
        mad = float(np.mean(np.abs(f_cpp[interior].astype(np.float32) - f_cv[interior].astype(np.float32))))
        assert mad <= 2.5

    def test_perspective_uv_against_bench_ray_triangle_oracle(self) -> None:
        w, h = 160, 120
        tw, th = 256, 256
        ug, vg = np.meshgrid(np.arange(tw, dtype=np.uint8), np.arange(th, dtype=np.uint8))
        tex = np.stack([ug, vg, np.full_like(ug, 100), np.full_like(ug, 255)], axis=-1)

        verts = np.array(
            [[-0.35, -0.25, 0.75], [0.90, -0.28, 2.60], [-0.08, 0.42, 1.35]],
            dtype=np.float32,
        )
        uvs = np.array([[0.05, 0.10], [0.95, 0.25], [0.25, 0.90]], dtype=np.float32)
        tris = np.array([[0, 1, 2]], dtype=np.int32)
        fx, fy, cx, cy = 140.0, 140.0, 80.0, 60.0

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        res = rasterize_mesh_rgbd(
            verts, tris, uvs, tex, (fx, fy, cx, cy), frame, opacity=1.0, blend_mode="normal", backend="native_cpp"
        )
        assert res.rendered_pixels > 400

        eroded = cv2.erode(res.mask, np.ones((3, 3), np.uint8)) > 0
        ys, xs = np.nonzero(eroded)
        origin = np.zeros(3, dtype=np.float64)
        diffs: list[float] = []
        for py, px in zip(ys[::4], xs[::4]):
            ray_dir = np.array([(px + 0.5 - cx) / fx, (py + 0.5 - cy) / fy, 1.0], dtype=np.float64)
            hit = ray_triangle(origin, ray_dir, verts.astype(np.float64), uvs.astype(np.float64))
            assert hit is not None
            _, oracle_uv = hit
            exp_rg = np.array([oracle_uv[0] * 255.0, oracle_uv[1] * 255.0])
            obs_rg = frame[py, px, :2].astype(np.float64)
            diffs.append(float(np.max(np.abs(obs_rg - exp_rg))))
        assert max(diffs) <= 2.0

    def test_non_contiguous_slice_and_invalid_inputs(self) -> None:
        verts = np.array([[-0.2, -0.2, 1.0], [0.2, -0.2, 1.0], [0.0, 0.2, 1.0]], dtype=np.float32)
        tris = np.array([[0, 1, 2]], dtype=np.int64)  # int64 coerced cleanly
        uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
        tex = np.full((16, 16, 4), 220, dtype=np.uint8)

        big = np.zeros((100, 200, 3), dtype=np.uint8)
        sub_view = big[:, ::2, :]  # non-contiguous (100, 100, 3) view
        assert not sub_view.flags["C_CONTIGUOUS"]
        res = rasterize_mesh_rgbd(verts, tris, uvs, tex, (90.0, 90.0, 50.0, 50.0), sub_view, backend="native_cpp")
        assert res.rendered_pixels > 100
        assert int(sub_view[50, 50, 0]) > 0

        for bad_call in [
            lambda: rasterize_mesh_rgbd(verts, np.array([[0, 1, 9]]), uvs, tex, (90, 90, 50, 50), np.zeros((64, 64, 3), np.uint8)),
            lambda: rasterize_mesh_rgbd(verts * np.nan, tris, uvs, tex, (90, 90, 50, 50), np.zeros((64, 64, 3), np.uint8)),
            lambda: rasterize_mesh_rgbd(verts, tris, uvs, tex, (0.0, 90, 50, 50), np.zeros((64, 64, 3), np.uint8)),
            lambda: rasterize_mesh_rgbd(verts, tris, uvs, tex, (90, 90, 50, 50), np.zeros((64, 64, 3), np.uint8), opacity=1.5),
            lambda: rasterize_mesh_rgbd(verts, tris, uvs, tex, (90, 90, 50, 50), np.zeros((64, 64, 3), np.uint8), opacity=np.bool_(True)),  # type: ignore[arg-type]
            lambda: rasterize_mesh_rgbd(verts, tris, uvs, tex, (90, 90, 50, 50), np.zeros((64, 64, 3), np.uint8), blend_mode=np.bool_(False)),  # type: ignore[arg-type]
            lambda: rasterize_mesh_rgbd(verts, tris, uvs, tex, (90, 90, 50, 50), np.zeros((64, 64, 3), np.uint8), blend_mode="dodge"),
            lambda: rasterize_mesh_rgbd(verts, tris, uvs, tex, (90, 90, 50, 50), np.zeros((64, 64, 3), np.uint8), backend="vulkan"),  # type: ignore[arg-type]
        ]:
            with pytest.raises(ValueError):
                bad_call()

        # Read-only (WRITEABLE=False) output buffers MUST raise ValueError before mutating anything
        raw_bytes = bytes([77] * (32 * 32 * 3))
        ro_frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(32, 32, 3)
        ro_depth = np.full((32, 32), np.inf, dtype=np.float32)
        ro_depth.setflags(write=False)
        ro_mask = np.zeros((32, 32), dtype=np.uint8)
        ro_mask.setflags(write=False)

        for backend_name in ("native_cpp", "opencv"):
            rw_depth = np.full((32, 32), np.inf, dtype=np.float32)
            with pytest.raises(ValueError, match="writeable"):
                rasterize_mesh_rgbd(
                    verts, tris, uvs, tex, (30.0, 30.0, 16.0, 16.0), ro_frame,
                    depth_buf=rw_depth, backend=backend_name,  # type: ignore[arg-type]
                )
            assert raw_bytes == bytes([77] * (32 * 32 * 3)), "immutable bytes buffer was corrupted"
            assert np.all(np.isinf(rw_depth)), "depth_buf was mutated when frame_rgb was read-only"

            with pytest.raises(ValueError, match="writeable"):
                rasterize_mesh_rgbd(
                    verts, tris, uvs, tex, (30.0, 30.0, 16.0, 16.0),
                    np.zeros((32, 32, 3), dtype=np.uint8),
                    depth_buf=ro_depth, backend=backend_name,  # type: ignore[arg-type]
                )
            with pytest.raises(ValueError, match="writeable"):
                rasterize_mesh_rgbd(
                    verts, tris, uvs, tex, (30.0, 30.0, 16.0, 16.0),
                    np.zeros((32, 32, 3), dtype=np.uint8),
                    mask_out=ro_mask, backend=backend_name,  # type: ignore[arg-type]
                )

        # NumPy scalar arguments (np.float32 opacity, np.int32 blend_mode, np.bool_ wrap_repeat) succeed
        for backend_name in ("native_cpp", "opencv"):
            out_np = np.full((64, 64, 3), 100, dtype=np.uint8)
            res_np = rasterize_mesh_rgbd(
                verts,
                tris,
                uvs,
                tex,
                np.array([90.0, 90.0, 32.0, 32.0], dtype=np.float32),
                out_np,
                opacity=np.float32(0.65),
                blend_mode=np.int32(1),
                wrap_repeat=np.bool_(True),
                backend=backend_name,  # type: ignore[arg-type]
            )
            assert res_np.rendered_pixels > 100

    def test_finite_uv_overflow_and_hd_frustum_clipping_and_zero_edge_bloat(self) -> None:
        tex = np.full((8, 8, 4), 220, dtype=np.uint8)
        tris = np.array([[0, 1, 2]], dtype=np.int32)

        # 1. Finite extreme UVs (3e38f and 1e37f) must produce 0 pixels, 0 crashes, and 0 RuntimeWarnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for verts_ov, uvs_ov, intr_ov, shape_ov in [
                (
                    np.array([[-0.5, -0.5, 0.001], [0.5, -0.5, 0.001], [0.0, 0.5, 0.001]], dtype=np.float32),
                    np.array([[1e37, 1e37], [-1e37, 1e37], [0.0, -1e37]], dtype=np.float32),
                    (10.0, 10.0, 8.0, 8.0),
                    (17, 17, 3),
                ),
                (
                    np.array([[-0.25, -0.25, 0.5], [0.25, -0.25, 0.5], [0.0, 0.25, 0.5]], dtype=np.float32),
                    np.full((3, 2), 3e38, dtype=np.float32),
                    (40.0, 40.0, 16.5, 16.5),
                    (33, 33, 3),
                ),
            ]:
                for wrap_rep in (False, True):
                    for backend_name in ("native_cpp", "opencv"):
                        frm = np.full(shape_ov, 99, dtype=np.uint8)
                        res = rasterize_mesh_rgbd(
                            verts_ov,
                            tris,
                            uvs_ov,
                            tex,
                            intr_ov,
                            frm,
                            wrap_repeat=wrap_rep,
                            backend=backend_name,  # type: ignore[arg-type]
                        )
                        assert res.rendered_pixels == 0
                        assert np.all(frm == 99)

        # 2. 1280x720 HD off-axis near-plane straddling + 1e8 guard band + zero silhouette bloat
        w, h = 1280, 720
        fx, fy, cx, cy = 1000.0, 1000.0, 640.0, 360.0
        uv = np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], dtype=np.float32)

        v_straddle = np.array([[-0.4, 0.20, 1.5], [0.4, 0.20, 1.5], [0.0, 0.20, -0.5]], dtype=np.float32)
        res_s_cpp = rasterize_mesh_rgbd(
            v_straddle, tris, uv, tex, (fx, fy, cx, cy), np.zeros((h, w, 3), np.uint8),
            wrap_repeat=True, blend_mode="normal", backend="native_cpp",
        )
        res_s_cv = rasterize_mesh_rgbd(
            v_straddle, tris, uv, tex, (fx, fy, cx, cy), np.zeros((h, w, 3), np.uint8),
            wrap_repeat=True, blend_mode="normal", backend="opencv",
        )
        assert res_s_cpp.rendered_pixels > 140_000
        assert abs(res_s_cpp.rendered_pixels - res_s_cv.rendered_pixels) <= 10

        v_guard_1e8 = np.array([[-0.2, -0.1, 1.0], [0.2, -0.1, 1.0], [0.0, 10.0, 1e-4]], dtype=np.float32)
        res_g_cpp = rasterize_mesh_rgbd(
            v_guard_1e8, tris, uv, tex, (fx, fy, cx, cy), np.zeros((h, w, 3), np.uint8),
            wrap_repeat=True, blend_mode="normal", backend="native_cpp",
        )
        res_g_cv = rasterize_mesh_rgbd(
            v_guard_1e8, tris, uv, tex, (fx, fy, cx, cy), np.zeros((h, w, 3), np.uint8),
            wrap_repeat=True, blend_mode="normal", backend="opencv",
        )
        assert res_g_cpp.rendered_pixels > 100_000
        assert abs(res_g_cpp.rendered_pixels - res_g_cv.rendered_pixels) <= 10

        v_bloat = np.array(
            [
                [(280.0 - cx) / fx, (200.0 - cy) / fy, 1.0],
                [(320.0 - cx) / fx, (400.0 - cy) / fy, 1.0],
                [(800_000.0 - cx) / fx, (300.0 - cy) / fy, 1.0],
            ],
            dtype=np.float32,
        )
        res_b_cpp = rasterize_mesh_rgbd(
            v_bloat, tris, uv, tex, (fx, fy, cx, cy), np.zeros((h, w, 3), np.uint8),
            wrap_repeat=True, blend_mode="normal", backend="native_cpp",
        )
        res_b_cv = rasterize_mesh_rgbd(
            v_bloat, tris, uv, tex, (fx, fy, cx, cy), np.zeros((h, w, 3), np.uint8),
            wrap_repeat=True, blend_mode="normal", backend="opencv",
        )
        assert int(np.flatnonzero(res_b_cpp.mask[300])[0]) == 300
        assert int(np.flatnonzero(res_b_cv.mask[300])[0]) == 300

    def test_hd_1280x720_native_raster_speed_gate(self) -> None:
        w, h = 1280, 720
        grid_n = 28  # 28x28x2 = 1,568 triangles
        ug, vg = np.meshgrid(
            np.linspace(0.0, 1.0, grid_n + 1, dtype=np.float32),
            np.linspace(0.0, 1.0, grid_n + 1, dtype=np.float32),
        )
        xs = (ug - 0.5) * 0.72
        ys = (vg - 0.5) * 0.42
        zs = 1.0 + 0.07 * np.cos((ug - 0.5) * np.pi)
        verts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=-1).astype(np.float32)
        uvs = np.stack([ug.ravel(), vg.ravel()], axis=-1).astype(np.float32)

        tris_list: list[list[int]] = []
        stride = grid_n + 1
        for gy in range(grid_n):
            for gx in range(grid_n):
                i0 = gy * stride + gx
                tris_list.append([i0, i0 + 1, i0 + stride + 1])
                tris_list.append([i0, i0 + stride + 1, i0 + stride])
        tris = np.asarray(tris_list, dtype=np.int32)

        tex = render_recipe_rgba(DesignSpec(pattern="waves", width=512, height=512, seed=123))
        base_frame = np.full((h, w, 3), 150, dtype=np.uint8)
        frame = base_frame.copy()
        depth = np.full((h, w), np.inf, dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        intr = (1000.0, 1000.0, 640.0, 360.0)

        for _ in range(10):
            depth.fill(np.inf)
            rasterize_mesh_rgbd(
                verts,
                tris,
                uvs,
                tex,
                intr,
                frame,
                depth_buf=depth,
                mask_out=mask,
                opacity=0.90,
                blend_mode="luminance",
                backend="native_cpp",
            )

        latencies_ms: list[float] = []
        for _ in range(100):
            depth.fill(np.inf)
            t0 = time.perf_counter()
            res = rasterize_mesh_rgbd(
                verts,
                tris,
                uvs,
                tex,
                intr,
                frame,
                depth_buf=depth,
                mask_out=mask,
                opacity=0.90,
                blend_mode="luminance",
                backend="native_cpp",
            )
            t1 = time.perf_counter()
            assert res.rendered_pixels > 180_000
            latencies_ms.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(latencies_ms))
        median_ms = float(np.median(latencies_ms))
        assert median_ms <= 4.0, f"HD 720p median raster latency {median_ms:.3f} ms > 4.0 ms"
        assert mean_ms <= 4.0, f"HD 720p mean raster latency {mean_ms:.3f} ms > 4.0 ms"


class TestDemoAssets:
    """Verify F5 multi-scenario H.264 MP4 corpus, ffprobe streams, ORB/GrabCut, occlusion, and 3D PLY."""

    def test_demo_assets_cli_idempotency_and_path_guards(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            generate_demo_corpus("/tmp/forbidden_demo_videos")
        with pytest.raises(ValueError):
            generate_demo_corpus(REPO_ROOT / ".agents" / "teamwork" / "forbidden_videos")
        assert demo_assets_main(["--output-dir", "/tmp/forbidden_demo_videos"]) == 2
        assert demo_assets_main(["--output-dir", ""]) == 2
        assert demo_assets_main(["--output-dir", "   "]) == 2

        existing_file = tmp_path / "not_a_dir.txt"
        existing_file.write_text("file", encoding="utf-8")
        assert demo_assets_main(["--output-dir", str(existing_file)]) == 2
        assert demo_assets_main(["--output-dir", str(existing_file / "sub")]) == 2
        bad_cli = subprocess.run(
            [sys.executable, "-B", "-m", "visualizeit.demo_assets", "--output-dir", str(existing_file)],
            capture_output=True,
            text=True,
            env=_subproc_env(),
        )
        assert bad_cli.returncode == 2

        proc = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "visualizeit.demo_assets",
                "--output-dir",
                str(DEFAULT_DEMO_DIR),
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=_subproc_env(),
        )
        manifest = json.loads(proc.stdout)
        assert [c["id"] for c in manifest["clips"]] == list(DEMO_CLIP_IDS)

        clips = load_demo_manifest(DEFAULT_DEMO_DIR)
        assert len(clips) == 5

    def test_ffprobe_streams_orb_features_and_grabcut_onboarding(self) -> None:
        clips = generate_demo_corpus(DEFAULT_DEMO_DIR, overwrite=False)
        orb = cv2.ORB_create(nfeatures=800)

        for clip in clips:
            mp4_path = Path(clip.path)
            assert mp4_path.is_file()

            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=codec_name,width,height,r_frame_rate,nb_read_frames,pix_fmt",
                    "-of",
                    "json",
                    str(mp4_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=_subproc_env(),
            )
            stream = json.loads(probe.stdout)["streams"][0]
            assert stream["codec_name"] == "h264"
            assert stream["pix_fmt"] == "yuv420p"
            assert stream["r_frame_rate"] == "30/1"
            assert int(stream["nb_read_frames"]) == 180
            assert int(stream["width"]) == clip.width
            assert int(stream["height"]) == clip.height

            cap = cv2.VideoCapture(str(mp4_path))
            ok, frame0 = cap.read()
            cap.release()
            assert ok and frame0 is not None and frame0.shape == (clip.height, clip.width, 3)

            sel = clip.default_selection
            assert sel.bbox is not None
            bx, by, bw, bh = [int(round(v)) for v in sel.bbox]
            roi_mask = np.zeros((clip.height, clip.width), dtype=np.uint8)
            roi_mask[max(0, by) : min(clip.height, by + bh), max(0, bx) : min(clip.width, bx + bw)] = 255

            kps, des = orb.detectAndCompute(frame0, roi_mask)
            assert des is not None and len(kps) >= 80, (
                f"{clip.id} had only {len(kps)} ORB keypoints inside default_selection.bbox"
            )

            # Verify GrabCut cleanly segments the foreground object using default_selection
            gc_mask = np.full((clip.height, clip.width), cv2.GC_BGD, dtype=np.uint8)
            gc_mask[max(0, by) : min(clip.height, by + bh), max(0, bx) : min(clip.width, bx + bw)] = cv2.GC_PR_FGD
            for px, py in sel.positive_points:
                cv2.circle(gc_mask, (int(round(px)), int(round(py))), 6, int(cv2.GC_FGD), -1)
            for nx, ny in sel.negative_points:
                cv2.circle(gc_mask, (int(round(nx)), int(round(ny))), 8, int(cv2.GC_BGD), -1)

            bgd_model = np.zeros((1, 65), dtype=np.float64)
            fgd_model = np.zeros((1, 65), dtype=np.float64)
            cv2.grabCut(frame0, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
            fg_px = int(np.count_nonzero(fg))
            assert 2_000 <= fg_px <= int(bw * bh * 0.98), f"{clip.id} GrabCut foreground pixels={fg_px}"
            for px, py in sel.positive_points:
                assert fg[int(round(py)), int(round(px))], f"{clip.id} positive point missed by GrabCut"
            for nx, ny in sel.negative_points:
                assert not fg[int(round(ny)), int(round(nx))], f"{clip.id} negative point included by GrabCut"

    def test_occlusion_recovery_schedule_and_ply_mesh_validity(self) -> None:
        gt_eval = json.loads((DEFAULT_DEMO_DIR / "ground_truth_eval.json").read_text(encoding="utf-8"))
        occ_frames = gt_eval["clips"]["occlusion_recovery_640x480"]["frames"]
        assert len(occ_frames) == 180
        for f in occ_frames[:70]:
            assert f["visible"] is True and f["visibility_fraction"] == 1.0
        for f in occ_frames[70:110]:
            assert f["visible"] is False and f["visibility_fraction"] == 0.0
        for f in occ_frames[110:]:
            assert f["visible"] is True and f["visibility_fraction"] == 1.0

        # Empirical 180-frame motion & ORB verification on decoded occlusion_recovery_640x480.mp4
        cap = cv2.VideoCapture(str(DEFAULT_DEMO_DIR / "occlusion_recovery_640x480.mp4"))
        all_occ_frames: list[np.ndarray] = []
        for _ in range(180):
            ok, frm = cap.read()
            assert ok and frm is not None
            all_occ_frames.append(frm)
        cap.release()

        frame_mads = [
            float(np.mean(np.abs(all_occ_frames[i].astype(np.float32) - all_occ_frames[i - 1].astype(np.float32))))
            for i in range(1, 180)
        ]
        assert min(frame_mads) > 0.25, f"Static/near-static frame pair detected: min MAD={min(frame_mads):.4f}"

        orb = cv2.ORB_create(nfeatures=600)
        roi0 = np.zeros((480, 640), dtype=np.uint8)
        roi0[140:340, 185:455] = 255
        _, des0 = orb.detectAndCompute(all_occ_frames[0], roi0)
        assert des0 is not None and len(des0) >= 80
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        def _good_matches(target_img: np.ndarray) -> int:
            _, des_t = orb.detectAndCompute(target_img, None)
            if des_t is None or len(des_t) < 2:
                return 0
            knn = matcher.knnMatch(des0, des_t, k=2)
            return sum(1 for pair in knn if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance)

        assert _good_matches(all_occ_frames[40]) >= 30
        for occ_idx in range(70, 110):
            assert _good_matches(all_occ_frames[occ_idx]) <= 3, f"Target card leaked on occluded frame {occ_idx}"
        assert _good_matches(all_occ_frames[140]) >= 30

        # Verify mesh_box.ply and mesh_box_meta.json
        ply_lines = (DEFAULT_DEMO_DIR / "mesh_box.ply").read_text(encoding="utf-8").splitlines()
        assert ply_lines[0] == "ply"
        assert "element vertex 150" in ply_lines
        assert "element face 192" in ply_lines
        hdr_end = ply_lines.index("end_header")
        vert_rows = [
            [float(tok) for tok in line.split()]
            for line in ply_lines[hdr_end + 1 : hdr_end + 1 + 150]
        ]
        verts_arr = np.asarray(vert_rows, dtype=np.float64)
        assert verts_arr.shape == (150, 11)
        assert np.max(np.abs(verts_arr[:, :3])) <= 0.10  # metric units in metres
        norms = np.linalg.norm(verts_arr[:, 3:6], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3)
        assert np.all((verts_arr[:, 6:8] >= 0.0) & (verts_arr[:, 6:8] <= 1.0))

        meta = json.loads((DEFAULT_DEMO_DIR / "mesh_box_meta.json").read_text(encoding="utf-8"))
        assert meta["vertex_units"] == "meters"
        assert meta["intrinsics"] == [560.0, 560.0, 320.0, 240.0]
        assert len(meta["ground_truth_poses"]) == 180
        for p in meta["ground_truth_poses"][::15]:
            r_mat = np.asarray(p["R_3x3"], dtype=np.float64)
            assert np.allclose(r_mat @ r_mat.T, np.eye(3), atol=1e-6)
            assert abs(float(np.linalg.det(r_mat)) - 1.0) <= 1e-6


class TestPrototypeBootstrapLinux:
    """Verify F1 Linux -B -S tools/prototype_bootstrap.py pyvenv.cfg site-packages resolution."""

    def test_bootstrap_resolves_linux_venv_under_no_site(self) -> None:
        bootstrap = REPO_ROOT / "tools" / "prototype_bootstrap.py"
        proc = subprocess.run(
            [
                sys.executable,
                "-B",
                "-S",
                str(bootstrap),
                "unittest",
                "tools.test_phone_static",
            ],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
            env=_subproc_env(),
        )
        assert proc.returncode == 0

        # Verify cv2, numpy, PIL, fastapi, and visualizeit all import cleanly under -B -S bootstrap
        probe_cmd = (
            "import sys; sys.argv = ['bootstrap', 'json.tool']; "
            "from tools.prototype_bootstrap import _resolve_site_packages; "
            "pkgs = _resolve_site_packages(); "
            "sys.path[:0] = ['.', *pkgs]; "
            "import cv2, numpy, PIL, fastapi, visualizeit; "
            "assert any('.cache/venv' in p for p in pkgs), pkgs"
        )
        subprocess.run(
            [sys.executable, "-B", "-S", "-c", probe_cmd],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
            env=_subproc_env(),
        )
