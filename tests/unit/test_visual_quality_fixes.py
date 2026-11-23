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


def _make_textured_card_frame(
    w: int = 320,
    h: int = 240,
    dx: int = 0,
    dy: int = 0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    rng = np.random.default_rng(123)
    img = np.full((h, w, 3), (30, 35, 40), dtype=np.uint8)
    x0, y0, bw, bh = 80 + dx, 50 + dy, 150, 130
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (210, 190, 160), -1)
    for _ in range(45):
        px = int(rng.integers(x0 + 8, x0 + bw - 18))
        py = int(rng.integers(y0 + 8, y0 + bh - 18))
        col = tuple(int(c) for c in rng.integers(20, 230, size=3))
        cv2.rectangle(img, (px, py), (px + 12, py + 10), col, -1)
    cv2.putText(img, "VIZ", (x0 + 25, y0 + 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 180), 3)
    return img, (float(x0), float(y0), float(bw), float(bh))


def test_confidence_state_transitions_suppress_rendering_and_freeze_memory() -> None:
    """TRACKING -> LOST -> RECOVERING -> TRACKING suppresses rendering and freezes memory on non-TRACKING states."""
    from visualizeit.contracts import DesignSpec
    from visualizeit.pipeline import VideoPipelineSession

    f0, bbox = _make_textured_card_frame()
    art = np.zeros((96, 96, 4), dtype=np.uint8)
    art[:, :] = (30, 40, 220, 255)
    from pathlib import Path
    tmp_art = Path(".cache/tmp/test_vq_art.png")
    tmp_art.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tmp_art), art)

    sess = VideoPipelineSession(
        design=DesignSpec(source="image", image_path=str(tmp_art)),
        placement=PlacementSpec(wrap_mode="planar", blend_mode="normal", opacity=0.9),
        local_occlusion_mask=True,
    )
    sel = SelectionSpec(
        frame_index=0,
        positive_points=[(bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5)],
        bbox=bbox,
    )
    r0 = sess.initialize(f0, sel, timestamp_s=0.0)
    assert r0.track_result.confidence_state == "TRACKING"
    assert r0.rendered_mask is not None and int(np.count_nonzero(r0.rendered_mask)) > 1000
    assert not np.array_equal(r0.frame_bgr, f0)
    kf_before_loss = len(sess.tracker._ref_keyframes)

    # Full blackout -> LOST: must suppress rendering, reset compositor occlusion memory, and NOT add keyframes
    blank = np.zeros_like(f0)
    r_lost = sess.process_frame(blank, 1, 0.033)
    assert r_lost.track_result.state == "lost"
    assert r_lost.track_result.confidence_state == "LOST"
    assert r_lost.rendered_mask is None
    assert np.array_equal(r_lost.frame_bgr, blank)
    assert sess.compositor._occ_prev_vis is None
    assert len(sess.tracker._ref_keyframes) == kf_before_loss

    # 1st clean frame back -> RECOVERING (single-frame unconfirmed candidate MUST still suppress rendering!)
    f_back, _ = _make_textured_card_frame(dx=2, dy=1)
    r_rec1 = sess.process_frame(f_back, 2, 0.066)
    assert r_rec1.track_result.state == "recovering"
    assert r_rec1.track_result.confidence_state == "RECOVERING"
    assert r_rec1.rendered_mask is None
    assert np.array_equal(r_rec1.frame_bgr, f_back)

    # 2nd consistent clean frame -> confirmed TRACKING, overlay restored!
    f_back2, _ = _make_textured_card_frame(dx=3, dy=1)
    r_rec2 = sess.process_frame(f_back2, 3, 0.100)
    assert r_rec2.track_result.state == "tracking"
    assert r_rec2.track_result.confidence_state == "TRACKING"
    assert r_rec2.rendered_mask is not None and int(np.count_nonzero(r_rec2.rendered_mask)) > 1000
    assert not np.array_equal(r_rec2.frame_bgr, f_back2)


def test_false_recovery_rejection_on_inconsistent_pose_jump() -> None:
    """Inconsistent 2nd recovery candidate (large pose jump) MUST NOT falsely confirm TRACKING."""
    f0, bbox = _make_textured_card_frame()
    sel = SelectionSpec(
        frame_index=0,
        positive_points=[(bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5)],
        bbox=bbox,
    )
    seg = segment_frame(f0, sel, sharp_onboarding=True)
    tracker = VideoSurfaceTracker(wrap_mode="planar")
    r0 = tracker.initialize(f0, seg)
    assert r0.state == "tracking"

    # Force LOST
    blank = np.zeros_like(f0)
    r_lost = tracker.track_frame(blank, 1, 0.033)
    assert r_lost.state == "lost"

    # 1st candidate at dx=0 -> RECOVERING
    r_c1 = tracker.track_frame(f0, 2, 0.066)
    assert r_c1.state == "recovering"
    assert r_c1.homography_3x3 is None

    # 2nd candidate shifted wildly by +75px -> inconsistent with 1st candidate -> MUST NOT confirm TRACKING!
    f_jump, _ = _make_textured_card_frame(dx=75, dy=30)
    r_c2 = tracker.track_frame(f_jump, 3, 0.100)
    assert r_c2.state != "tracking", "Tracker falsely confirmed recovery across a 75px inconsistent pose jump!"
    assert r_c2.homography_3x3 is None


def test_ambiguous_state_suppresses_compositor_and_resets_temporal_memory() -> None:
    """Compositor returns untouched input frame and clears temporal visibility memory on AMBIGUOUS/LOST/RECOVERING."""
    f0, bbox = _make_textured_card_frame()
    art = np.full((64, 64, 4), (40, 40, 220, 255), dtype=np.uint8)
    comp = SurfaceCompositor(
        art,
        PlacementSpec(wrap_mode="planar", blend_mode="normal", opacity=0.9),
        ref_bbox=tuple(int(v) for v in bbox),
        ref_frame_bgr=f0,
        local_occlusion_mask=True,
    )
    tr_ok = TrackFrameResult(frame_index=0, timestamp_s=0.0, state="tracking", homography_3x3=np.eye(3), inliers=50, confidence=0.95)
    out0, m0, _ = comp.composite(f0, tr_ok)
    assert m0 is not None and comp._occ_prev_vis is not None
    assert not np.array_equal(out0, f0)

    for st, cst in [("recovering", "AMBIGUOUS"), ("recovering", "RECOVERING"), ("lost", "LOST")]:
        tr_bad = TrackFrameResult(frame_index=1, timestamp_s=0.033, state=st, inliers=0, confidence=0.0, confidence_state=cst)
        out_b, m_b, _ = comp.composite(f0, tr_bad)
        assert m_b is None
        assert np.array_equal(out_b, f0)
        assert comp._occ_prev_vis is None
        assert comp.last_occluder_mask_img is None


def test_6dof_target_color_lut_protects_reddish_bottle_and_excludes_gripping_hand() -> None:
    """Target Lab LUT prevents reddish-orange bottles from being misclassified as skin while still excluding exterior beige hands."""
    img = np.full((360, 480, 3), (25, 25, 25), dtype=np.uint8)
    # Reddish-orange bottle (deep orange-red BGR=(28, 62, 168)) in center
    img[90:270, 200:280] = (28, 62, 168)
    cv2.putText(img, "BBQ", (208, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (235, 235, 235), 2)
    mask = np.zeros((360, 480), dtype=np.uint8)
    mask[90:270, 200:280] = 255

    mesh = build_parametric_mesh("cylinder", cyl_radius=0.032, cyl_height=0.14)
    intr = (500.0, 500.0, 240.0, 180.0)
    init_pose = (np.zeros(3, dtype=np.float64), np.array([0.0, 0.0, 0.42], dtype=np.float64))
    art = np.full((96, 96, 4), (220, 40, 40, 255), dtype=np.uint8)
    comp = SurfaceCompositor(
        art,
        PlacementSpec(wrap_mode="mesh_6dof", blend_mode="normal", opacity=0.9),
        ref_bbox=(200, 90, 80, 180),
        ref_mask=mask,
        ref_frame_bgr=img,
        mesh=mesh,
        intrinsics=intr,
        init_pose=init_pose,
        local_occlusion_mask=True,
    )
    # Frame 0 (no hand): target Lab LUT must produce ZERO false hand pixels on the reddish-orange bottle
    hand0 = comp._compute_6dof_exterior_hand_mask(img, mask)
    assert hand0 is None or int(np.count_nonzero(hand0 & mask)) == 0

    # Frame 1: beige/pink hand entering from left exterior across middle of bottle
    img_hand = img.copy()
    cv2.rectangle(img_hand, (130, 155), (255, 200), (155, 175, 218), -1)  # BGR realistic light skin
    hand1 = comp._compute_6dof_exterior_hand_mask(img_hand, mask)
    assert hand1 is not None
    hand_on_bottle = float(np.mean(hand1[160:195, 205:250] > 0))
    clean_bottle = float(np.mean(hand1[100:145, 210:270] > 0))
    assert hand_on_bottle > 0.70, f"Exterior gripping hand not detected: {hand_on_bottle:.3f}"
    assert clean_bottle < 0.05, f"Non-occluded reddish bottle falsely flagged as hand: {clean_bottle:.3f}"


def test_6dof_degraded_frame_never_inserts_corrupted_keyframe() -> None:
    """6DoF tracker must not corrupt _mesh_keyframes when input degrades to LOST/AMBIGUOUS."""
    img = np.full((360, 480, 3), (40, 40, 40), dtype=np.uint8)
    img[90:270, 190:290] = (170, 130, 70)
    rng = np.random.default_rng(9)
    for _ in range(30):
        x, y = int(rng.integers(198, 276)), int(rng.integers(98, 256))
        cv2.rectangle(img, (x, y), (x + 10, y + 10), (int(rng.integers(20, 240)), 60, 200), -1)
    sel = SelectionSpec(frame_index=0, positive_points=[(240.0, 180.0)], bbox=(188.0, 88.0, 104.0, 184.0))
    seg = segment_frame(img, sel, sharp_onboarding=True)
    tracker = VideoSurfaceTracker(
        wrap_mode="mesh_6dof",
        mesh=build_parametric_mesh("cylinder"),
        intrinsics=(500.0, 500.0, 240.0, 180.0),
    )
    r0 = tracker.initialize(img, seg)
    assert r0.state == "tracking"
    kf_init = len(tracker._mesh_keyframes)
    assert kf_init >= 1

    blank = np.zeros_like(img)
    r1 = tracker.track_frame(blank, 1, 0.033)
    assert r1.state == "lost"
    assert r1.confidence_state == "LOST"
    assert len(tracker._mesh_keyframes) == kf_init


def test_neural_modules_availability_and_fallback() -> None:
    """Verify EdgeTAM and XFeat availability functions report clean booleans."""
    from visualizeit.matcher_xfeat import is_xfeat_available
    from visualizeit.vos_edgetam import is_edgetam_available

    has_vos = is_edgetam_available()
    has_xf = is_xfeat_available()
    assert isinstance(has_vos, bool)
    assert isinstance(has_xf, bool)


def test_edgetam_session_lifecycle_and_pruning() -> None:
    """Test EdgeTAM session initialization, optical flow interpolation, and O(1) memory pruning."""
    from visualizeit.vos_edgetam import EdgeTAMSession, is_edgetam_available

    if not is_edgetam_available():
        pytest.skip("EdgeTAM weights not cached in test environment")

    sess = EdgeTAMSession(vos_stride=2, memory_window=4)
    frame = np.full((120, 160, 3), 40, dtype=np.uint8)
    cv2.circle(frame, (80, 60), 25, (220, 220, 220), -1)

    sel = SelectionSpec(
        frame_index=0,
        positive_points=[(80.0, 60.0)],
        bbox=(50.0, 30.0, 60.0, 60.0),
    )
    seg = sess.initialize(frame, sel)
    assert seg.init_valid
    assert seg.mask is not None
    assert int(np.count_nonzero(seg.mask)) > 400

    # Step frame 1 (intermediate -> flow)
    frame_next = np.roll(frame, 2, axis=1)
    m1 = sess.step(frame_next, 1)
    assert m1.shape == (120, 160)
    assert int(np.count_nonzero(m1)) > 300

    # Step frame 2 (VOS keyframe)
    m2 = sess.step(frame_next, 2)
    assert m2.shape == (120, 160)
    assert int(np.count_nonzero(m2)) > 300

    # Step through frame 6 and confirm state images dict does not grow unboundedly
    for idx in range(3, 7):
        sess.step(frame_next, idx)
    assert len(sess._state["images"]) <= 6


def test_xfeat_matcher_small_roi_upsampling() -> None:
    """Test that XFeat extractor upsamples small handheld ROIs so valid keypoints are detected."""
    from visualizeit.matcher_xfeat import XFeatMatcher, is_xfeat_available

    if not is_xfeat_available():
        pytest.skip("XFeat weights not cached in test environment")

    matcher = XFeatMatcher()
    tiny_roi = np.full((64, 64, 3), 50, dtype=np.uint8)
    rng = np.random.default_rng(42)
    for _ in range(12):
        x, y = int(rng.integers(10, 50)), int(rng.integers(10, 50))
        cv2.rectangle(tiny_roi, (x, y), (x + 6, y + 6), (200, 220, 240), -1)

    kpts, feats = matcher.extract(tiny_roi)
    assert kpts.ndim == 2 and kpts.shape[1] == 2
    assert feats.ndim == 2 and feats.shape[1] == 64
    assert len(kpts) > 0


def test_pipeline_session_with_neural_backends() -> None:
    """Test full VideoPipelineSession end-to-end with neural backends."""
    from visualizeit.contracts import DesignSpec, PlacementSpec
    from visualizeit.matcher_xfeat import is_xfeat_available
    from visualizeit.pipeline import VideoPipelineSession
    from visualizeit.vos_edgetam import is_edgetam_available

    if not (is_edgetam_available() and is_xfeat_available()):
        pytest.skip("Neural weights not cached")

    design = DesignSpec(
        source="recipe",
        pattern="checker",
        width=128,
        height=128,
    )
    placement = PlacementSpec(wrap_mode="planar", opacity=0.9)
    session = VideoPipelineSession(
        design=design,
        placement=placement,
        segmentation_backend="edgetam",
        matcher_backend="xfeat",
        vos_stride=2,
    )

    frame = np.full((120, 160, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (40, 30), (120, 90), (200, 180, 160), -1)
    rng = np.random.default_rng(123)
    for _ in range(25):
        x, y = int(rng.integers(45, 110)), int(rng.integers(35, 82))
        cv2.rectangle(frame, (x, y), (x + 4, y + 4), (int(rng.integers(10, 80)), int(rng.integers(10, 80)), int(rng.integers(10, 80))), -1)

    sel = SelectionSpec(
        frame_index=0,
        positive_points=[(80.0, 60.0)],
        bbox=(38.0, 28.0, 84.0, 64.0),
    )
    out0 = session.initialize(frame, sel, timestamp_s=0.0)
    assert out0.rendered_mask is not None
    assert out0.track_result.confidence_state in ("TRACKING", "RECOVERING")

    frame1 = np.roll(frame, 1, axis=1)
    out1 = session.process_frame(frame1, 1, timestamp_s=0.033)
    assert out1.rendered_mask is not None
    assert session.last_vos_mask is not None


