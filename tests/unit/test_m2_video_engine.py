"""Unit tests for Milestone 2 video engine: segmentation, causal tracking, compositor, 30 Hz live replay, and CLI."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from visualizeit.compositor import SurfaceCompositor
from visualizeit.contracts import DesignSpec, PlacementSpec, SelectionSpec
from visualizeit.demo_assets import DEFAULT_DEMO_DIR, generate_demo_corpus, load_demo_manifest
from visualizeit.designs import render_design
from visualizeit.pipeline import VideoPipeline, VideoPipelineSession, probe_video_info, replay_live_30hz
from visualizeit.segmentation import segment_frame
from visualizeit.tracker import VideoSurfaceTracker, build_parametric_mesh, load_ply_mesh


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def demo_clips() -> dict[str, Any]:
    if not (DEFAULT_DEMO_DIR / "demo_manifest.json").is_file():
        generate_demo_corpus(DEFAULT_DEMO_DIR)
    return {c.id: c for c in load_demo_manifest(DEFAULT_DEMO_DIR)}


def _read_frames(path: str | Path, count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while len(frames) < count:
            ok, frm = cap.read()
            if not ok or frm is None:
                break
            frames.append(frm)
    finally:
        cap.release()
    return frames


def test_m2_segmentation_point_and_box_bounds(demo_clips: dict[str, Any], tmp_path: Path) -> None:
    clip = demo_clips["planar_card_640x480"]
    f0 = _read_frames(clip.path, 1)[0]
    mask_out = REPO_ROOT / ".cache" / "tmp" / "unit_m2_seg_mask.png"
    mask_out.unlink(missing_ok=True)

    res = segment_frame(f0, clip.default_selection, output_mask_path=mask_out)
    assert res.width == 640 and res.height == 480
    assert 0.05 <= res.coverage_ratio <= 0.70
    assert res.anchor_count >= 16
    assert res.init_ms < 400.0
    assert mask_out.is_file()

    # Enforce strict bbox containment
    bx, by, bw, bh = [int(v) for v in clip.default_selection.bbox]
    outside = res.mask.copy()
    outside[by : by + bh + 1, bx : bx + bw + 1] = 0
    assert int(np.count_nonzero(outside)) == 0
    mask_out.unlink(missing_ok=True)


def test_m2_direct_causality_mutating_tail_leaves_prefix_byte_identical(demo_clips: dict[str, Any]) -> None:
    """Direct causality proof: mutating frames k..N-1 changes ZERO tracker decisions or rendered bytes on 0..k-1."""
    clip = demo_clips["planar_card_640x480"]
    frames_a = _read_frames(clip.path, 40)
    frames_b = [f.copy() for f in frames_a]
    # Mutate tail frames 20..39 radically (pitch black + random noise)
    rng = np.random.default_rng(12345)
    for i in range(20, 40):
        frames_b[i] = rng.integers(0, 255, size=frames_b[i].shape, dtype=np.uint8)

    design = DesignSpec(source="recipe", pattern="stripes", colors=["#ff595e", "#1982c4"], text="CAUSAL")
    placement = PlacementSpec(wrap_mode="planar", blend_mode="luminance", opacity=0.9)

    def run_seq(frames: list[np.ndarray]) -> tuple[list[dict], list[np.ndarray]]:
        sess = VideoPipelineSession(design=design, placement=placement, tracking_height=360)
        decisions, rendered = [], []
        out0 = sess.initialize(frames[0], clip.default_selection, timestamp_s=0.0)
        decisions.append((out0.track_result.state, out0.track_result.inliers, out0.track_result.homography_3x3))
        rendered.append(out0.frame_bgr.copy())
        for idx in range(1, len(frames)):
            out = sess.process_frame(frames[idx], idx, float(idx) / 30.0)
            decisions.append((out.track_result.state, out.track_result.inliers, out.track_result.homography_3x3))
            rendered.append(out.frame_bgr.copy())
        return decisions, rendered

    dec_a, ren_a = run_seq(frames_a)
    dec_b, ren_b = run_seq(frames_b)

    for idx in range(20):
        assert dec_a[idx][0] == dec_b[idx][0], f"State diverged at prefix frame {idx}"
        assert dec_a[idx][1] == dec_b[idx][1], f"Inliers diverged at prefix frame {idx}"
        if dec_a[idx][2] is not None:
            assert np.array_equal(dec_a[idx][2], dec_b[idx][2]), f"Homography diverged at prefix frame {idx}"
        assert np.array_equal(ren_a[idx], ren_b[idx]), f"Rendered RGB frame {idx} not byte-identical!"

    # And tail frames MUST actually diverge (proving the mutation was exercised)
    assert dec_b[25][0] == "lost"


def test_m2_shared_per_frame_core_and_30hz_live_replay(demo_clips: dict[str, Any]) -> None:
    clip = demo_clips["planar_card_640x480"]
    design = DesignSpec(source="recipe", pattern="checker", colors=["#ff006e", "#00f5d4"])
    placement = PlacementSpec(wrap_mode="planar", blend_mode="luminance", opacity=0.85)

    live = replay_live_30hz(
        input_path=clip.path,
        design=design,
        placement=placement,
        selection=clip.default_selection,
        target_hz=30.0,
        max_frames=30,
        threads=2,
    )
    assert live["frames_replayed"] == 29
    assert live["arrival_to_overlay_ready_ms"]["p95"] < 33.33
    assert live["deadline_misses"] == 0
    assert live["max_queue_depth"] == 0
    assert live["kept_up_live"] is True


def test_m2_occlusion_suppression_in_causal_and_offline_refine(demo_clips: dict[str, Any]) -> None:
    """Even opt-in --offline-refine must strictly suppress overlay across verified full occlusion."""
    clip = demo_clips["occlusion_recovery_640x480"]
    design = DesignSpec(source="recipe", pattern="solid", colors=["#00ff00", "#00ff00"])
    placement = PlacementSpec(wrap_mode="planar", blend_mode="normal", opacity=1.0)
    pipeline = VideoPipeline()

    for offline in (False, True):
        trace: list[dict] = []
        rep = pipeline.process_video(
            input_path=clip.path,
            output_path=None,
            design=design,
            placement=placement,
            selection=clip.default_selection,
            offline_refine=offline,
            trace_records=trace,
        )
        assert rep.render_stride == 1
        assert rep.lost_frame_count >= 10
        assert rep.recovered_count >= 1
        # Every lost/recovering frame MUST have zero rendered pixels
        for row in trace:
            if row["state"] != "tracking":
                assert row["rendered_pixels"] == 0, f"Offline={offline}: leaked overlay on {row['state']} frame {row['frame_index']}"


def test_m2_all_four_wrap_modes_and_parametric_meshes(demo_clips: dict[str, Any]) -> None:
    clip = demo_clips["planar_card_640x480"]
    f0 = _read_frames(clip.path, 1)[0]
    seg = segment_frame(f0, clip.default_selection)
    art = render_design(DesignSpec(source="recipe", pattern="stripes", colors=["#ff0000", "#0000ff"]))

    outputs = {}
    for mode in ("planar", "cylinder", "mesh_grid"):
        pl = PlacementSpec(wrap_mode=mode, blend_mode="normal", opacity=1.0)
        trk = VideoSurfaceTracker(wrap_mode=mode)
        tr = trk.initialize(f0, seg)
        comp = SurfaceCompositor(art, pl, ref_bbox=trk.ref_bbox, ref_mask=trk.ref_mask)
        out, mask, _ = comp.composite(f0, tr)
        assert mask is not None and int(np.count_nonzero(mask)) > 500
        outputs[mode] = out

    # Planar vs cylinder vs mesh_grid must produce distinct projections
    assert float(np.mean(np.abs(outputs["planar"].astype(np.float32) - outputs["cylinder"].astype(np.float32)))) > 0.5
    assert float(np.mean(np.abs(outputs["planar"].astype(np.float32) - outputs["mesh_grid"].astype(np.float32)))) > 0.2

    # All 4 parametric shapes load cleanly
    for shape in ("plane", "cylinder", "box", "curved_panel"):
        m = build_parametric_mesh(shape)
        assert len(m.vertices) >= 4 and len(m.triangles) >= 2


def test_m2_cli_clean_error_exit_codes() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "visualizeit", "info", "--input", "nonexistent_video_xyz.mp4"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "error:" in proc.stderr


def test_m2_local_occlusion_mask_suppresses_foreign_occluder_but_preserves_shadow() -> None:
    """Illumination-invariant local occlusion mask hides overlay on a foreign hand bar while preserving shadows."""
    h, w = 240, 320
    # Textured blue/cyan surface in center on dark background
    ref = np.full((h, w, 3), 25, dtype=np.uint8)
    rng = np.random.default_rng(7)
    patch = np.zeros((120, 160, 3), dtype=np.uint8)
    patch[:, :, 0] = 200  # strong blue channel (BGR)
    patch[:, :, 1] = 140
    patch[:, :, 2] = 35
    noise = rng.integers(-25, 26, size=(120, 160, 1))
    patch = np.clip(patch.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    ref[60:180, 80:240] = patch

    sel = SelectionSpec(frame_index=0, positive_points=[(160.0, 120.0)], bbox=(80.0, 60.0, 160.0, 120.0))
    seg = segment_frame(ref, sel, sharp_onboarding=True)
    art = render_design(DesignSpec(source="recipe", pattern="solid", colors=["#00ff00", "#00ff00"]))
    pl = PlacementSpec(wrap_mode="planar", blend_mode="normal", opacity=1.0)
    trk = VideoSurfaceTracker(wrap_mode="planar")
    tr0 = trk.initialize(ref, seg)
    comp = SurfaceCompositor(
        art, pl, ref_bbox=trk.ref_bbox, ref_mask=trk.ref_mask, ref_frame_bgr=ref, local_occlusion_mask=True
    )

    # Frame A: warm skin-toned hand occluding the right half of the surface
    hand_frame = ref.copy()
    hand_frame[60:180, 160:240] = (110, 155, 220)  # warm peach BGR
    out_hand, _, _ = comp.composite(hand_frame, tr0)
    left_diff = float(np.mean(np.abs(out_hand[75:165, 95:145].astype(np.float32) - hand_frame[75:165, 95:145].astype(np.float32))))
    right_diff = float(np.mean(np.abs(out_hand[75:165, 175:225].astype(np.float32) - hand_frame[75:165, 175:225].astype(np.float32))))
    assert left_diff > 60.0, f"Unoccluded left half should receive artwork, got diff={left_diff:.1f}"
    assert right_diff < 8.0, f"Foreign hand occluder on right half should suppress artwork, got diff={right_diff:.1f}"

    # Frame B: luminance-only shadow (0.55x brightness) across right half -> artwork MUST remain visible
    shadow_frame = ref.copy()
    shadow_frame[60:180, 160:240] = np.clip(shadow_frame[60:180, 160:240].astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
    out_shadow, _, _ = comp.composite(shadow_frame, tr0)
    shadow_diff = float(np.mean(np.abs(out_shadow[75:165, 175:225].astype(np.float32) - shadow_frame[75:165, 175:225].astype(np.float32))))
    assert shadow_diff > 60.0, f"Luminance shadow should NOT falsely suppress artwork, got diff={shadow_diff:.1f}"


def test_m2_sharp_onboarding_720p_bbox_precision() -> None:
    """ROI-cropped sharp_onboarding stays fast and aligns tightly within 3 px of an exact 720p target."""
    frame = np.full((720, 1280, 3), 24, dtype=np.uint8)
    rng = np.random.default_rng(19)
    card = np.zeros((360, 640, 3), dtype=np.uint8)
    card[:, :, 0] = 60
    card[:, :, 1] = 175
    card[:, :, 2] = 225
    card = np.clip(card.astype(np.int32) + rng.integers(-30, 31, size=(360, 640, 1)), 0, 255).astype(np.uint8)
    frame[180:540, 320:960] = card
    sel = SelectionSpec(
        frame_index=0,
        positive_points=[(640.0, 360.0)],
        negative_points=[(100.0, 100.0), (1180.0, 620.0)],
        bbox=(320.0, 180.0, 640.0, 360.0),
    )
    seg = segment_frame(frame, sel, sharp_onboarding=True)
    assert seg.init_ms < 150.0
    sx, sy, sw, sh = [float(v) for v in seg.bbox]
    assert abs(sx - 320.0) <= 3.0 and abs(sy - 180.0) <= 3.0
    assert abs((sx + sw) - 960.0) <= 3.0 and abs((sy + sh) - 540.0) <= 3.0


def test_m2_spatial_bucketing_respects_fixed_point_budget(demo_clips: dict[str, Any]) -> None:
    """8x6 fixed-budget spatial bucketing never exceeds max_klt_points and tracks cleanly."""
    clip = demo_clips["planar_card_640x480"]
    frames = _read_frames(clip.path, 15)
    seg = segment_frame(frames[0], clip.default_selection, sharp_onboarding=True)
    trk = VideoSurfaceTracker(wrap_mode="planar", max_klt_points=120, spatial_bucketing=True)
    tr0 = trk.initialize(frames[0], seg)
    assert tr0.state == "tracking"
    assert len(trk._curr_pts_track) <= 120
    for i in range(1, len(frames)):
        tr = trk.track_frame(frames[i], i, i / 30.0)
        assert tr.state == "tracking"
        assert len(trk._curr_pts_track) <= 120

