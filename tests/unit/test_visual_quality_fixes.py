"""Unit tests for Phase 1-4 visual quality fixes (initialization, monochrome/color occlusion, 6DoF cylinder fit, 4-state confidence)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from visualizeit.compositor import SurfaceCompositor
from visualizeit.contracts import ContractValidationError, PlacementSpec, SelectionSpec, TrackFrameResult
from visualizeit.segmentation import segment_frame
from visualizeit.tracker import VideoSurfaceTracker, build_parametric_mesh


def test_phase1_init_verification_and_distractor_suppression() -> None:
    """Positive-seed Lab affinity suppresses a differently-colored adjacent distractor inside explicit_box."""
    img = np.full((240, 320, 3), (40, 40, 40), dtype=np.uint8)
    # Blue target on left half of box, bright yellow distractor on right half
    img[60:180, 70:155] = (200, 90, 30)   # BGR blue
    img[60:180, 165:230] = (30, 220, 235) # BGR yellow distractor
    sel = SelectionSpec(
        frame_index=0,
        positive_points=[(110.0, 120.0)],
        negative_points=[],
        bbox=(65.0, 55.0, 175.0, 130.0),
    )
    seg = segment_frame(img, sel, sharp_onboarding=True)
    assert seg.init_valid is True
    assert seg.quality_score > 0.5
    assert seg.rejection_reason is None
    # Blue target should be heavily selected; yellow distractor should be mostly excluded
    blue_hit = float(np.mean(seg.mask[70:170, 80:145] > 0))
    yellow_hit = float(np.mean(seg.mask[70:170, 175:220] > 0))
    assert blue_hit > 0.70
    assert yellow_hit < 0.25


def test_phase2_monochrome_foreground_occlusion_mask() -> None:
    """Upgraded 96x96 local occlusion mask suppresses foreground occluders even on zero-chroma monochrome video."""
    rng = np.random.default_rng(7)
    ref = np.full((240, 320), 120, dtype=np.uint8)
    # Textured target card inside [60:180, 80:240]
    for _ in range(30):
        x, y = int(rng.integers(85, 225)), int(rng.integers(65, 170))
        cv2.rectangle(ref, (x, y), (x + 14, y + 10), int(rng.integers(20, 235)), -1)
    ref_bgr = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

    cur_bgr = ref_bgr.copy()
    # Smooth monochrome occluder bar covering middle of card
    cur_bgr[40:210, 135:185] = (175, 175, 175)

    art = np.zeros((128, 128, 4), dtype=np.uint8)
    art[:, :, :3] = (220, 40, 40)
    art[:, :, 3] = 255
    placement = PlacementSpec(wrap_mode="planar", blend_mode="normal", opacity=0.95)
    comp = SurfaceCompositor(
        art,
        placement,
        ref_bbox=(80, 60, 160, 120),
        ref_frame_bgr=ref_bgr,
        raster_backend="native_cpp",
        local_occlusion_mask=True,
    )
    tr = TrackFrameResult(frame_index=1, timestamp_s=0.033, state="tracking", homography_3x3=np.eye(3), inliers=50, confidence=0.9)
    out_bgr, _, _ = comp.composite(cur_bgr, tr)

    diff = np.max(np.abs(out_bgr.astype(np.float32) - cur_bgr.astype(np.float32)), axis=2)
    occ_bleed = float(np.mean(diff[75:165, 145:175] > 14.0))
    vis_cov = float(np.mean(diff[75:165, 90:125] > 14.0))
    assert occ_bleed < 0.35, f"Monochrome occluder bleed too high: {occ_bleed:.3f}"
    assert vis_cov > 0.70, f"Visible target over-eroded: {vis_cov:.3f}"


def test_phase3_causal_6dof_cylinder_aspect_fit() -> None:
    """Parametric 6DoF cylinder fits its radius to the frame-0 mask aspect ratio without GT leakage."""
    img = np.full((360, 480, 3), (80, 80, 80), dtype=np.uint8)
    # Narrow cylinder silhouette: 80px wide x 200px tall (aspect 0.40 -> radius ~ 0.5 * 0.40 * 0.14 = 0.028m)
    img[80:280, 200:280] = (60, 160, 210)
    cv2.putText(img, "CYL", (206, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    sel = SelectionSpec(frame_index=0, positive_points=[(240.0, 180.0)], bbox=(198.0, 78.0, 84.0, 204.0))
    seg = segment_frame(img, sel, sharp_onboarding=True)

    tracker = VideoSurfaceTracker(
        wrap_mode="mesh_6dof",
        mesh=build_parametric_mesh("cylinder"),
        intrinsics=(500.0, 500.0, 240.0, 180.0),
    )
    res = tracker.initialize(img, seg)
    assert res.state == "tracking"
    assert res.pose_rvec is not None and res.pose_tvec is not None
    # Fitted cylinder radius should be much narrower than default 0.06m
    assert tracker.mesh is not None
    fitted_r = float(np.max(np.abs(tracker.mesh.vertices[:, 0])))
    assert 0.022 <= fitted_r <= 0.038, f"Unexpected fitted cylinder radius: {fitted_r}"
    assert float(np.linalg.norm(res.pose_rvec)) < 0.1


def test_phase4_four_state_confidence_contract() -> None:
    """TrackFrameResult exposes explicit TRACKING/AMBIGUOUS/LOST/RECOVERING states."""
    tr_ok = TrackFrameResult(frame_index=0, timestamp_s=0.0, state="tracking", homography_3x3=np.eye(3), inliers=40, confidence=0.9)
    assert tr_ok.confidence_state == "TRACKING"
    tr_amb = TrackFrameResult(frame_index=1, timestamp_s=0.033, state="recovering", inliers=11, confidence=0.0, confidence_state="AMBIGUOUS")
    assert tr_amb.confidence_state == "AMBIGUOUS"
    assert tr_amb.homography_3x3 is None
    tr_lost = TrackFrameResult(frame_index=2, timestamp_s=0.066, state="lost", inliers=0, confidence=0.0)
    assert tr_lost.confidence_state == "LOST"
    with pytest.raises(ContractValidationError):
        TrackFrameResult(frame_index=3, timestamp_s=0.1, state="lost", confidence_state="INVALID")  # type: ignore[arg-type]
