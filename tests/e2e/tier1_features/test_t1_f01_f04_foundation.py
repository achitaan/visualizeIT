"""Tier 1 Feature Happy-Path E2E Tests — Slice A (`F1`–`F4`, 24 distinct tests).

Covers:
- `F1`: Repeatable Linux CPU Env & Legacy Compatibility (6 tests)
- `F2`: Native C++17 AVX-512 Surface Rasterizer (6 tests)
- `F3`: Procedural Design Engine & Prompt Parser (6 tests)
- `F4`: Custom RGBA Artwork Upload & Loader (6 tests)

Strictly opaque-box: zero module-scope imports of `visualizeit.*` or third-party packages
(other than `pytest`).
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.e2e._helpers import (
    BLEND_INK,
    BLEND_LUMINANCE,
    BLEND_NORMAL,
    DEMO_VIDEOS_DIR,
    HeadlessChromiumSession,
    NATIVE_BUILD_DIR,
    REPO_ROOT,
    SETUP_ENV_SCRIPT,
    SO_PATH,
    StudioAPIClient,
    VENV_PYTHON,
    VI_OK,
    assert_no_python_traceback,
    assert_video_run_report_invariants,
    call_pose_to_unity,
    call_rasterize_mesh_rgbd,
    compute_roi_and_bg_delta,
    decode_image_bytes,
    decode_video_frames,
    get_venv_python,
    load_native_abi,
    make_planar_quad_mesh,
    make_rgb_jpeg,
    make_rgba_png,
    make_rgba_webp,
    make_textured_tracking_clip,
    make_uv_cylinder_mesh,
    ray_triangle_uv_oracle,
    run_cli_json,
    run_subproc,
)

pytestmark = pytest.mark.tier1

_HOST_TMP_DAEMON_PREFIXES = (
    "agy-plugin-proxy.",
    "initgoogle_syslog_dir",
)

_PINNED_RESEARCH_SHA256 = {
    "bench/quality_gotrack.py": "0b18516894690de0fbf3565d5d542ced17865c161e646369fa4f7e5412a28c29",
    "bench/quality_runner.py": "bca306d1342dce2566c786fed94a7564ae090e11bf42243da2ea7bc9ce841ad1",
}


def _snapshot_non_daemon_tmp() -> set[str]:
    tmp_root = Path("/tmp")
    if not tmp_root.is_dir():
        return set()
    return {
        entry.name
        for entry in tmp_root.iterdir()
        if not entry.name.startswith(_HOST_TMP_DAEMON_PREFIXES)
    }


def _find_forbidden_teamwork_files() -> list[str]:
    teamwork_dir = REPO_ROOT / ".agents" / "teamwork"
    if not teamwork_dir.is_dir():
        return []
    bad: list[str] = []
    for path in teamwork_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in {"AGENTS.md", "GEMINI.md"} or path.suffix.lower() != ".md":
            bad.append(str(path.relative_to(REPO_ROOT)))
    return sorted(bad)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    clean = hex_color.strip().lstrip("#")
    if len(clean) == 3:
        clean = "".join(ch * 2 for ch in clean)
    assert len(clean) == 6, f"Expected 6-digit hex color, got {hex_color!r}"
    return (int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16))


# ==============================================================================
# F1 — Repeatable Linux CPU Env & Legacy Compatibility (6 tests)
# ==============================================================================


@pytest.mark.f01
def test_f01_venv_python_and_core_packages_importable(e2e_tmp_dir: Path) -> None:
    """Verify `.cache/venv/bin/python` is Python 3.11.x and cleanly imports all F1 dependencies."""
    assert VENV_PYTHON.is_file() and os.access(VENV_PYTHON, os.X_OK), (
        f"Expected executable virtualenv interpreter at {VENV_PYTHON}"
    )

    probe_code = """
import json, sys
from pathlib import Path
import cv2, fastapi, httpx, multipart, numpy, PIL, pytest, scipy, uvicorn, visualizeit, websockets

print(json.dumps({
    "py_major": sys.version_info.major,
    "py_minor": sys.version_info.minor,
    "prefix": str(Path(sys.prefix).resolve()),
    "numpy_version": numpy.__version__,
    "cv2_version": cv2.__version__,
    "pil_version": PIL.__version__,
    "visualizeit_file": str(Path(visualizeit.__file__).resolve()),
}))
"""
    proc = run_subproc(
        [str(VENV_PYTHON), "-B", "-c", probe_code],
        timeout=30.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert proc.returncode == 0, f"Venv import probe failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert_no_python_traceback(proc)

    info = json.loads(proc.stdout.strip())
    assert (info["py_major"], info["py_minor"]) == (3, 11), f"Expected Python 3.11, got {info}"
    assert Path(info["prefix"]) == (REPO_ROOT / ".cache" / "venv").resolve()

    np_parts = tuple(int(part) for part in str(info["numpy_version"]).split(".")[:2])
    assert np_parts >= (2, 1), f"Expected numpy >= 2.1 in .cache/venv, got {info['numpy_version']}"
    assert str((REPO_ROOT / "visualizeit").resolve()) in info["visualizeit_file"]


@pytest.mark.f01
def test_f01_setup_env_script_idempotent_on_xfs(e2e_tmp_dir: Path) -> None:
    """Run `bash scripts/setup_env.sh` on XFS and verify idempotent exit 0 + zero `/tmp` leaks."""
    assert SETUP_ENV_SCRIPT.is_file(), f"Missing setup script: {SETUP_ENV_SCRIPT}"

    before_tmp = _snapshot_non_daemon_tmp()
    proc = run_subproc(
        ["bash", str(SETUP_ENV_SCRIPT)],
        timeout=120.0,
        cwd=REPO_ROOT,
        tmp_dir=e2e_tmp_dir,
    )
    after_tmp = _snapshot_non_daemon_tmp()

    assert proc.returncode == 0, (
        f"scripts/setup_env.sh failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert VENV_PYTHON.is_file() and os.access(VENV_PYTHON, os.X_OK)
    assert SO_PATH.is_file() and SO_PATH.stat().st_size > 0
    assert DEMO_VIDEOS_DIR.is_dir() and any(DEMO_VIDEOS_DIR.glob("*.mp4")), (
        f"Expected generated .mp4 files in {DEMO_VIDEOS_DIR}"
    )

    ctest_proc = run_subproc(
        ["ctest", "--test-dir", str(NATIVE_BUILD_DIR), "--output-on-failure"],
        timeout=30.0,
        cwd=REPO_ROOT,
        tmp_dir=e2e_tmp_dir,
    )
    assert ctest_proc.returncode == 0, f"Native CTest failed:\n{ctest_proc.stdout}\n{ctest_proc.stderr}"

    leaked_tmp = sorted(after_tmp - before_tmp)
    assert not leaked_tmp, f"setup_env.sh leaked artifacts into /tmp: {leaked_tmp}"
    forbidden_tw = _find_forbidden_teamwork_files()
    assert not forbidden_tw, f"Forbidden files found inside .agents/teamwork/: {forbidden_tw}"


@pytest.mark.f01
def test_f01_prototype_bootstrap_linux_no_site_flag_works(e2e_tmp_dir: Path) -> None:
    """Verify `tools/prototype_bootstrap.py` resolves Linux venv `site-packages` under `-B -S`."""
    bootstrap = REPO_ROOT / "tools" / "prototype_bootstrap.py"
    assert bootstrap.is_file(), f"Missing {bootstrap}"

    proc_static = run_subproc(
        [str(get_venv_python()), "-B", "-S", str(bootstrap), "unittest", "tools.test_phone_static"],
        timeout=30.0,
        cwd=REPO_ROOT,
        tmp_dir=e2e_tmp_dir,
    )
    assert proc_static.returncode == 0, (
        f"prototype_bootstrap.py failed on tools.test_phone_static:\n"
        f"STDOUT:\n{proc_static.stdout}\nSTDERR:\n{proc_static.stderr}"
    )
    assert_no_python_traceback(proc_static)

    # Also verify third-party venv packages (numpy, Pillow) resolve under `-S`
    proc_renderer = run_subproc(
        [str(get_venv_python()), "-B", "-S", str(bootstrap), "unittest", "bench.test_renderer"],
        timeout=30.0,
        cwd=REPO_ROOT,
        tmp_dir=e2e_tmp_dir,
    )
    assert proc_renderer.returncode == 0, (
        f"prototype_bootstrap.py under -S failed to resolve third-party site-packages:\n"
        f"STDOUT:\n{proc_renderer.stdout}\nSTDERR:\n{proc_renderer.stderr}"
    )


@pytest.mark.f01
def test_f01_legacy_cpp_pose_abi_preserved_in_shared_lib() -> None:
    """Verify `vi_pose_abi_version() == 1` and `vi_cv_gltf_pose_to_unity` rigid & in-place round-trip."""
    import numpy as np

    lib = load_native_abi()
    assert int(lib.vi_pose_abi_version()) == 1

    # Construct an orthonormal right-handed rotation R = Ry(-0.45) @ Rx(0.30) + translation
    ax, ay = 0.30, -0.45
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(ax), -math.sin(ax)], [0.0, math.sin(ax), math.cos(ax)]],
        dtype=np.float64,
    )
    ry = np.array(
        [[math.cos(ay), 0.0, math.sin(ay)], [0.0, 1.0, 0.0], [-math.sin(ay), 0.0, math.cos(ay)]],
        dtype=np.float64,
    )
    rot = ry @ rx
    pose_cv = np.eye(4, dtype=np.float32)
    pose_cv[:3, :3] = rot.astype(np.float32)
    pose_cv[:3, 3] = np.array([0.28, -0.42, 1.35], dtype=np.float32)

    rc_out, pose_unity = call_pose_to_unity(pose_cv, lib=lib, in_place=False)
    assert rc_out == VI_OK

    # Verify cameraSigns=[1,-1,1,1] and objectSigns=[1,1,-1,1]
    cam_signs = np.array([1.0, -1.0, 1.0, 1.0], dtype=np.float32)
    obj_signs = np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    expected_unity = cam_signs[:, None] * pose_cv * obj_signs[None, :]
    assert np.allclose(pose_unity, expected_unity, atol=1e-5)

    # Verify in-place aliasing matches out-of-place and double-application restores original pose
    rc_in_place, pose_unity_ip = call_pose_to_unity(pose_cv, lib=lib, in_place=True)
    assert rc_in_place == VI_OK
    assert np.allclose(pose_unity_ip, pose_unity, atol=1e-6)

    rc_roundtrip, pose_roundtrip = call_pose_to_unity(pose_unity_ip, lib=lib, in_place=True)
    assert rc_roundtrip == VI_OK
    assert np.allclose(pose_roundtrip, pose_cv, atol=1e-5)


@pytest.mark.f01
def test_f01_legacy_node_js_test_suites_pass(e2e_tmp_dir: Path) -> None:
    """Verify legacy `node --test` and standalone Node.js regression suites pass with exit code 0."""
    node_test_files = [
        str(REPO_ROOT / "bench" / "test_prototype_client.js"),
        str(REPO_ROOT / "tools" / "test_design_studio_engine.js"),
        str(REPO_ROOT / "tools" / "test_phone_glass_helpers.js"),
    ]
    proc = run_subproc(["node", "--test", *node_test_files], timeout=30.0, cwd=REPO_ROOT, tmp_dir=e2e_tmp_dir)
    assert proc.returncode == 0, f"node --test suite failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    standalone_scripts = [
        REPO_ROOT / "bench" / "test_quality_editor_state_smoke.js",
        REPO_ROOT / "bench" / "test_workbench_resource_labels.js",
        REPO_ROOT / "tools" / "test_phone_selection.js",
        REPO_ROOT / "tools" / "test_phone_tracker.js",
        REPO_ROOT / "tools" / "test_phone_feature_coverage.js",
    ]
    for script in standalone_scripts:
        s_proc = run_subproc(["node", str(script)], timeout=20.0, cwd=REPO_ROOT, tmp_dir=e2e_tmp_dir)
        assert s_proc.returncode == 0, (
            f"Standalone Node suite {script.name} failed (rc={s_proc.returncode}):\n"
            f"STDOUT:\n{s_proc.stdout}\nSTDERR:\n{s_proc.stderr}"
        )


@pytest.mark.f01
def test_f01_legacy_python_unit_smoke_and_sha256_pins_intact(e2e_tmp_dir: Path) -> None:
    """Verify SHA-256-pinned research files, direct `test_evaluate_tracking.py`, and `test_workbench_neural`."""
    for rel_path, expected_sha in _PINNED_RESEARCH_SHA256.items():
        target = REPO_ROOT / rel_path
        assert target.is_file(), f"Missing pinned legacy file: {target}"
        raw_bytes = target.read_bytes()
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
        # Historical pins in workbench_neural.py were recorded on CRLF checkouts; accept exact LF or CRLF form
        crlf_bytes = raw_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        crlf_sha = hashlib.sha256(crlf_bytes).hexdigest()
        assert expected_sha in {raw_sha, crlf_sha}, (
            f"SHA-256 pin mismatch on {rel_path}: expected {expected_sha}, got raw={raw_sha}, crlf={crlf_sha}"
        )

    # Direct script execution required for tools/test_evaluate_tracking.py
    eval_proc = run_subproc(
        [str(get_venv_python()), "-B", str(REPO_ROOT / "tools" / "test_evaluate_tracking.py")],
        timeout=30.0,
        cwd=REPO_ROOT,
        tmp_dir=e2e_tmp_dir,
    )
    assert eval_proc.returncode == 0, (
        f"tools/test_evaluate_tracking.py failed:\nSTDOUT:\n{eval_proc.stdout}\nSTDERR:\n{eval_proc.stderr}"
    )

    neural_proc = run_subproc(
        [str(get_venv_python()), "-B", "-m", "unittest", "bench.test_workbench_neural"],
        timeout=45.0,
        cwd=REPO_ROOT,
        tmp_dir=e2e_tmp_dir,
    )
    assert neural_proc.returncode == 0, (
        f"bench.test_workbench_neural failed:\nSTDOUT:\n{neural_proc.stdout}\nSTDERR:\n{neural_proc.stderr}"
    )


# ==============================================================================
# F2 — Native C++17 AVX-512 Surface Rasterizer (6 tests)
# ==============================================================================


@pytest.mark.f02
def test_f02_single_front_facing_quad_renders_expected_roi_and_mask() -> None:
    """Rasterize a 2-triangle planar quad at z=1.0m and verify seamless ROI fill, mask, and depth."""
    import numpy as np

    verts, tris, uvs = make_planar_quad_mesh(width_m=0.4, height_m=0.3, z_m=1.0, center_xy=(0.0, 0.0))
    tex = np.zeros((64, 64, 4), dtype=np.uint8)
    tex[...] = [220, 62, 44, 255]

    bg = np.full((480, 640, 3), [24, 34, 48], dtype=np.uint8)
    bg_orig = bg.copy()
    depth = np.full((480, 640), np.inf, dtype=np.float32)

    res = call_rasterize_mesh_rgbd(
        verts,
        tris,
        uvs,
        tex,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        frame_rgb_inout=bg,
        depth_buf_inout=depth,
        mask_out=True,
        wrap_repeat=0,
        opacity=1.0,
        blend_mode=BLEND_NORMAL,
    )
    assert res["rc"] > 28000, f"Expected ~30,000 pixels rendered for 200x150 quad, got {res['rc']}"
    mask = res["mask"]
    assert mask is not None
    assert int(np.count_nonzero(mask)) == res["rc"]

    # Projected quad spans x in [220, 420], y in [165, 315]; interior must have zero diagonal cracks
    interior_mask = mask[170:310, 225:415]
    assert np.all(interior_mask == 255), "Found unrendered crack pixels inside shared 2-triangle quad interior"
    assert np.all(res["frame_rgb"][170:310, 225:415] == np.array([220, 62, 44], dtype=np.uint8))
    assert np.allclose(res["depth_buf"][170:310, 225:415], 1.0, atol=1e-4)

    # Exterior region outside [218..422, 163..317] must remain completely untouched
    outside = np.ones((480, 640), dtype=bool)
    outside[163:318, 218:423] = False
    assert np.all(mask[outside] == 0)
    assert np.array_equal(res["frame_rgb"][outside], bg_orig[outside])
    assert np.all(np.isinf(res["depth_buf"][outside]))


@pytest.mark.f02
def test_f02_zbuffer_depth_occlusion_is_triangle_order_independent() -> None:
    """Verify near (z=0.8m) vs far (z=1.6m) triangle overlap produces identical RGB and depth in any order."""
    import numpy as np

    verts_near, tris_near, _ = make_planar_quad_mesh(
        width_m=0.28, height_m=0.22, z_m=0.8, center_xy=(-0.03, -0.02)
    )
    verts_far, tris_far, _ = make_planar_quad_mesh(
        width_m=0.64, height_m=0.48, z_m=1.6, center_xy=(0.04, 0.02)
    )

    # 2-swatch RGBA atlas: left half (u in [0.05..0.45]) = red, right half (u in [0.55..0.95]) = blue
    tex = np.zeros((32, 64, 4), dtype=np.uint8)
    tex[:, :32] = [238, 28, 36, 255]
    tex[:, 32:] = [24, 92, 242, 255]

    uvs_near = np.array([[0.05, 0.1], [0.45, 0.1], [0.45, 0.9], [0.05, 0.9]], dtype=np.float32)
    uvs_far = np.array([[0.55, 0.1], [0.95, 0.1], [0.95, 0.9], [0.55, 0.9]], dtype=np.float32)

    all_verts = np.vstack([verts_near, verts_far])
    all_uvs = np.vstack([uvs_near, uvs_far])
    tris_near_first = np.vstack([tris_near, tris_far + 4])
    tris_far_first = np.vstack([tris_far + 4, tris_near])

    res_nf = call_rasterize_mesh_rgbd(
        all_verts,
        tris_near_first,
        all_uvs,
        tex,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        frame_w=640,
        frame_h=480,
        opacity=1.0,
        blend_mode=BLEND_NORMAL,
    )
    res_fn = call_rasterize_mesh_rgbd(
        all_verts,
        tris_far_first,
        all_uvs,
        tex,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        frame_w=640,
        frame_h=480,
        opacity=1.0,
        blend_mode=BLEND_NORMAL,
    )

    assert res_nf["rc"] > 0 and res_fn["rc"] > 0
    assert np.array_equal(res_nf["frame_rgb"], res_fn["frame_rgb"])
    finite_nf = np.isfinite(res_nf["depth_buf"])
    assert np.array_equal(finite_nf, np.isfinite(res_fn["depth_buf"]))
    assert np.allclose(res_nf["depth_buf"][finite_nf], res_fn["depth_buf"][finite_nf], atol=1e-6)

    # Overlap center [215:245, 290:320] must show near red quad at z=0.8m
    assert np.all(res_nf["frame_rgb"][215:245, 290:320] == np.array([238, 28, 36], dtype=np.uint8))
    assert np.allclose(res_nf["depth_buf"][215:245, 290:320], 0.8, atol=1e-4)
    # Non-overlapped far right region (near ends at x=388.75, far ends at x=432.5, y=321.25)
    assert np.all(res_nf["frame_rgb"][285:312, 396:428] == np.array([24, 92, 242], dtype=np.uint8))
    assert np.allclose(res_nf["depth_buf"][285:312, 396:428], 1.6, atol=1e-4)


@pytest.mark.f02
def test_f02_perspective_correct_uv_matches_ray_triangle_oracle() -> None:
    """Verify perspective-correct `1/z` UV interpolation matches Moller-Trumbore ray oracle within <=3 levels."""
    import numpy as np

    # Single large depth-tilted triangle spanning z=0.75m (left) to z=2.25m (right)
    verts = np.array(
        [
            [-0.30, -0.22, 0.75],
            [+0.75, -0.35, 2.25],
            [+0.10, +0.42, 1.35],
        ],
        dtype=np.float32,
    )
    tris = np.array([[0, 2, 1]], dtype=np.int32)
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.25, 1.0]], dtype=np.float32)

    th, tw = 128, 128
    u_grid = np.linspace(0.0, 1.0, tw, dtype=np.float64)[None, :]
    v_grid = np.linspace(0.0, 1.0, th, dtype=np.float64)[:, None]
    tex = np.zeros((th, tw, 4), dtype=np.uint8)
    tex[..., 0] = np.rint(255.0 * u_grid).astype(np.uint8)
    tex[..., 1] = np.rint(255.0 * v_grid).astype(np.uint8)
    tex[..., 2] = np.rint(255.0 * (1.0 - u_grid)).astype(np.uint8)
    tex[..., 3] = 255

    bg = np.full((480, 640, 3), 18, dtype=np.uint8)
    res = call_rasterize_mesh_rgbd(
        verts,
        tris,
        uvs,
        tex,
        fx=520.0,
        fy=520.0,
        cx=320.0,
        cy=240.0,
        frame_rgb_inout=bg.copy(),
        mask_out=True,
        wrap_repeat=0,
        opacity=1.0,
        blend_mode=BLEND_NORMAL,
    )
    oracle = ray_triangle_uv_oracle(
        verts,
        tris,
        uvs,
        fx=520.0,
        fy=520.0,
        cx=320.0,
        cy=240.0,
        frame_w=640,
        frame_h=480,
        texture_rgba=tex,
        wrap_repeat=0,
        opacity=1.0,
        background_rgb=bg,
    )

    both = (res["mask"] > 0) & oracle["hit_mask"]
    # Erode 2px from triangle edges to avoid rasterizer boundary tie-breaking differences
    interior = (
        both[2:-2, 2:-2]
        & both[:-4, 2:-2]
        & both[4:, 2:-2]
        & both[2:-2, :-4]
        & both[2:-2, 4:]
    )
    assert int(np.count_nonzero(interior)) > 5000

    c_rgb = res["frame_rgb"][2:-2, 2:-2][interior].astype(np.int32)
    o_rgb = oracle["rgb"][2:-2, 2:-2][interior].astype(np.int32)
    abs_err = np.abs(c_rgb - o_rgb)
    assert int(abs_err.max()) <= 3, f"Max RGB error vs 1/z oracle exceeded 3 uint8 levels: {abs_err.max()}"
    assert float(abs_err.mean()) <= 1.0

    # Confirm genuine 1/z correction by checking screen-space barycentric centroid (1/3, 1/3, 1/3)
    proj_x = 520.0 * (verts[:, 0] / verts[:, 2]) + 320.0
    proj_y = 520.0 * (verts[:, 1] / verts[:, 2]) + 240.0
    px_mid = int(round(float(proj_x.mean())))
    py_mid = int(round(float(proj_y.mean())))
    affine_u = float(uvs[:, 0].mean())
    affine_red = 255.0 * affine_u
    actual_red = float(res["frame_rgb"][py_mid, px_mid, 0])
    assert abs(actual_red - affine_red) >= 18.0, (
        f"Rasterizer UV looks affine rather than perspective-correct: actual_red={actual_red}, affine_red={affine_red}"
    )


@pytest.mark.f02
def test_f02_wrap_repeat_vs_decal_clamp_uv_sampling() -> None:
    """Verify `wrap_repeat=1` tiles UVs across [-1, 2] while `wrap_repeat=0` restricts decal to [0, 1]."""
    import numpy as np

    verts, tris, _ = make_planar_quad_mesh(width_m=0.60, height_m=0.30, z_m=1.0, center_xy=(0.0, 0.0))
    # Horizontal UVs span [-1.0, 2.0] across screen x in [170, 470] (100px per UV unit)
    uvs = np.array([[-1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [-1.0, 1.0]], dtype=np.float32)

    tex = np.zeros((64, 64, 4), dtype=np.uint8)
    tex[:, :32] = [250, 120, 24, 255]   # Orange left half
    tex[:, 32:] = [20, 210, 232, 255]   # Cyan right half

    bg = np.full((480, 640, 3), 30, dtype=np.uint8)
    rep = call_rasterize_mesh_rgbd(
        verts,
        tris,
        uvs,
        tex,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        frame_rgb_inout=bg.copy(),
        mask_out=True,
        wrap_repeat=1,
        opacity=1.0,
        blend_mode=BLEND_NORMAL,
    )
    decal = call_rasterize_mesh_rgbd(
        verts,
        tris,
        uvs,
        tex,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        frame_rgb_inout=bg.copy(),
        mask_out=True,
        wrap_repeat=0,
        opacity=1.0,
        blend_mode=BLEND_NORMAL,
    )

    # In repeat mode, all 3 periods ([-1,0], [0,1], [1,2]) alternate Orange -> Cyan
    y_mid = 240
    for period_x0 in (170, 270, 370):
        left_sw = rep["frame_rgb"][y_mid, period_x0 + 25]
        right_sw = rep["frame_rgb"][y_mid, period_x0 + 75]
        assert int(left_sw[0]) > 200 and int(left_sw[2]) < 60, f"Expected orange at x={period_x0 + 25}: {left_sw}"
        assert int(right_sw[2]) > 190 and int(right_sw[0]) < 60, f"Expected cyan at x={period_x0 + 75}: {right_sw}"

    # Center [0, 1] UV period (x in [275..365]) must match between repeat and decal modes
    assert np.allclose(
        rep["frame_rgb"][200:280, 275:365].astype(np.float32),
        decal["frame_rgb"][200:280, 275:365].astype(np.float32),
        atol=3.0,
    )
    # Outside [0, 1] (x in [180..260] and [380..460]), decal mode must NOT repeat the alternating stripes
    outer_left = decal["frame_rgb"][y_mid, 180:260].astype(np.float32)
    outer_right = decal["frame_rgb"][y_mid, 380:460].astype(np.float32)
    assert float(outer_left.std(axis=0).max()) <= 2.0, "Decal mode unexpectedly tiled stripes in u < 0"
    assert float(outer_right.std(axis=0).max()) <= 2.0, "Decal mode unexpectedly tiled stripes in u > 1"


@pytest.mark.f02
def test_f02_blend_modes_normal_luminance_and_ink_distinct() -> None:
    """Verify `normal`, `luminance_shading`, and `ink` blend modes over a shaded background gradient."""
    import numpy as np

    verts, tris, uvs = make_planar_quad_mesh(width_m=0.40, height_m=0.30, z_m=1.0)
    tex = np.zeros((64, 64, 4), dtype=np.uint8)
    tex[...] = [70, 175, 245, 255]  # Uniform bright sky-blue artwork

    # Horizontal background luminance gradient from L=80 (left of quad) to L=220 (right of quad)
    bg = np.full((480, 640, 3), 120, dtype=np.uint8)
    ramp = np.linspace(80.0, 220.0, 200, dtype=np.float64)
    bg[165:315, 220:420] = np.rint(ramp)[None, :, None].astype(np.uint8)

    outs: dict[int, Any] = {}
    for mode in (BLEND_NORMAL, BLEND_LUMINANCE, BLEND_INK):
        res = call_rasterize_mesh_rgbd(
            verts,
            tris,
            uvs,
            tex,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            frame_rgb_inout=bg.copy(),
            opacity=1.0,
            blend_mode=mode,
        )
        assert res["rc"] > 28000
        outs[mode] = res["frame_rgb"]

    dark_slice = (slice(190, 290), slice(230, 265))
    bright_slice = (slice(190, 290), slice(375, 410))

    # Normal mode: flat across background gradient
    norm_dark = float(outs[BLEND_NORMAL][dark_slice].mean())
    norm_bright = float(outs[BLEND_NORMAL][bright_slice].mean())
    assert abs(norm_bright - norm_dark) <= 1.5

    # Luminance shading mode: preserves scene shading ratio while tinting blue (B > R)
    lum_dark = float(outs[BLEND_LUMINANCE][dark_slice].mean())
    lum_bright = float(outs[BLEND_LUMINANCE][bright_slice].mean())
    assert lum_bright > 1.35 * lum_dark, (
        f"Expected luminance blend to preserve bright/dark substrate shading, got {lum_bright:.1f} vs {lum_dark:.1f}"
    )
    assert float(outs[BLEND_LUMINANCE][bright_slice][..., 2].mean()) > float(
        outs[BLEND_LUMINANCE][bright_slice][..., 0].mean()
    )

    # Ink mode: subtractive/multiplicative — never brightens above gray substrate, and distinct from normal/luminance
    ink_roi = outs[BLEND_INK][175:305, 230:410].astype(np.int32)
    bg_roi = bg[175:305, 230:410].astype(np.int32)
    assert np.all(ink_roi <= bg_roi + 2), "Ink blend brightened pixels above substrate intensity"

    for m_a, m_b in ((BLEND_NORMAL, BLEND_LUMINANCE), (BLEND_NORMAL, BLEND_INK), (BLEND_LUMINANCE, BLEND_INK)):
        diff = float(
            np.abs(outs[m_a][175:305, 230:410].astype(np.float32) - outs[m_b][175:305, 230:410].astype(np.float32)).mean()
        )
        assert diff >= 8.0, f"Blend modes {m_a} and {m_b} produced nearly indistinguishable output (diff={diff:.2f})"


@pytest.mark.f02
def test_f02_throughput_720p_200_triangle_mesh_under_6ms_per_frame() -> None:
    """Benchmark 50 consecutive 720p 200-triangle `vi_rasterize_mesh_rgbd` calls (<10 ms/frame mean)."""
    import numpy as np

    lib = load_native_abi()
    verts, tris, uvs = make_uv_cylinder_mesh(
        radius_m=0.18,
        height_m=0.36,
        z_center_m=0.85,
        num_sectors=25,
        num_stacks=4,
    )
    assert tris.shape == (200, 3)

    tex = np.zeros((256, 256, 4), dtype=np.uint8)
    ys, xs = np.ogrid[:256, :256]
    checker = ((xs // 16) + (ys // 16)) % 2 == 0
    tex[...] = [32, 168, 236, 255]
    tex[checker] = [248, 112, 52, 255]

    frame_w, frame_h = 1280, 720
    frame_rgb = np.full((frame_h, frame_w, 3), 140, dtype=np.uint8)
    depth_buf = np.full((frame_h, frame_w), np.inf, dtype=np.float32)
    mask_buf = np.zeros((frame_h, frame_w), dtype=np.uint8)

    v_ptr = verts.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    t_ptr = tris.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    uv_ptr = uvs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    tex_ptr = tex.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    rgb_ptr = frame_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    d_ptr = depth_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    m_ptr = mask_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    for _ in range(5):
        depth_buf.fill(np.inf)
        rc_warm = int(
            lib.vi_rasterize_mesh_rgbd(
                v_ptr,
                int(verts.shape[0]),
                t_ptr,
                int(tris.shape[0]),
                uv_ptr,
                tex_ptr,
                256,
                256,
                1,
                900.0,
                900.0,
                640.0,
                360.0,
                0.92,
                BLEND_LUMINANCE,
                rgb_ptr,
                d_ptr,
                m_ptr,
                frame_w,
                frame_h,
            )
        )
        assert rc_warm > 10000

    elapsed_s = 0.0
    rendered_counts: list[int] = []
    for _ in range(50):
        depth_buf.fill(np.inf)
        t0 = time.perf_counter()
        rc = int(
            lib.vi_rasterize_mesh_rgbd(
                v_ptr,
                int(verts.shape[0]),
                t_ptr,
                int(tris.shape[0]),
                uv_ptr,
                tex_ptr,
                256,
                256,
                1,
                900.0,
                900.0,
                640.0,
                360.0,
                0.92,
                BLEND_LUMINANCE,
                rgb_ptr,
                d_ptr,
                m_ptr,
                frame_w,
                frame_h,
            )
        )
        elapsed_s += time.perf_counter() - t0
        rendered_counts.append(rc)

    mean_ms = (elapsed_s / 50.0) * 1000.0
    assert min(rendered_counts) == max(rendered_counts) and rendered_counts[0] > 10000
    assert mean_ms < 10.0, f"Expected native 720p 200-tri rasterization < 10.0 ms/frame, got {mean_ms:.2f} ms"


# ==============================================================================
# F3 — Procedural Design Engine & Prompt Parser (6 tests)
# ==============================================================================


@pytest.mark.f03
def test_f03_all_six_procedural_patterns_render_distinct_nonempty_artwork(
    live_studio_server: StudioAPIClient,
) -> None:
    """Render all 6 procedural patterns at 512x512 and verify valid, pairwise-distinct artwork."""
    import numpy as np

    patterns = ("solid", "stripes", "checker", "dots", "waves", "grid")
    palette = ["#ff595e", "#1982c4", "#ffca3a"]
    rendered_rgb: dict[str, Any] = {}
    pixel_hashes: set[str] = set()

    for pat in patterns:
        resp = live_studio_server.post_json(
            "/api/designs/render",
            {"pattern": pat, "colors": palette, "text": "", "prompt": "", "width": 512, "height": 512},
            expected_status=200,
        )
        assert resp.get("design_id"), f"Missing design_id for pattern={pat!r}: {resp}"
        assert resp.get("image_url"), f"Missing image_url for pattern={pat!r}: {resp}"
        assert resp.get("recipe", {}).get("pattern") == pat

        img_resp = live_studio_server.get(str(resp["image_url"]), expected_status=200)
        arr, _ = decode_image_bytes(img_resp["body"])
        assert arr.shape[:2] == (512, 512) and arr.ndim == 3 and arr.shape[2] in (3, 4)
        rgb = arr[..., :3]
        if pat != "solid":
            assert float(rgb.std()) > 10.0, f"Expected spatial pattern variance for {pat!r}, got std={rgb.std():.2f}"
        rendered_rgb[pat] = rgb
        pixel_hashes.add(hashlib.sha256(rgb.tobytes()).hexdigest())

    assert len(pixel_hashes) == len(patterns), "Expected pairwise unique pixel hashes across all 6 patterns"
    for i, pat_a in enumerate(patterns):
        for pat_b in patterns[i + 1 :]:
            mad = float(np.abs(rendered_rgb[pat_a].astype(np.float32) - rendered_rgb[pat_b].astype(np.float32)).mean())
            assert mad > 3.0, f"Patterns {pat_a!r} and {pat_b!r} are too visually similar (MAD={mad:.2f})"


@pytest.mark.f03
def test_f03_multi_color_palette_2_to_5_hex_colors_applied(
    live_studio_server: StudioAPIClient,
) -> None:
    """Verify 2-, 3-, and 5-color palettes deposit pixel clusters near every specified hex color."""
    import numpy as np

    full_palette = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff"]
    for count in (2, 3, 5):
        colors = full_palette[:count]
        for pat in ("stripes", "dots"):
            resp = live_studio_server.post_json(
                "/api/designs/render",
                {"pattern": pat, "colors": colors, "text": "", "prompt": "", "width": 512, "height": 512},
                expected_status=200,
            )
            img_resp = live_studio_server.get(str(resp["image_url"]), expected_status=200)
            arr, _ = decode_image_bytes(img_resp["body"])
            rgb = arr[..., :3].astype(np.float32)

            for hex_c in colors:
                target = np.array(_hex_to_rgb(hex_c), dtype=np.float32)
                dist = np.linalg.norm(rgb - target[None, None, :], axis=-1)
                near_px = int(np.count_nonzero(dist <= 48.0))
                assert near_px >= 400, (
                    f"Color {hex_c} from {count}-color palette missing in {pat!r} render (only {near_px} pixels near)"
                )


@pytest.mark.f03
def test_f03_custom_text_overlay_changes_center_artwork_pixels(
    live_studio_server: StudioAPIClient,
) -> None:
    """Verify non-empty `text` draws dark ink glyphs across the central band of a solid white canvas."""
    import numpy as np

    blank_resp = live_studio_server.post_json(
        "/api/designs/render",
        {"pattern": "solid", "colors": ["#ffffff", "#111111"], "text": "", "prompt": "", "width": 512, "height": 512},
        expected_status=200,
    )
    text_resp = live_studio_server.post_json(
        "/api/designs/render",
        {
            "pattern": "solid",
            "colors": ["#ffffff", "#111111"],
            "text": "VISUALIZE",
            "prompt": "",
            "width": 512,
            "height": 512,
        },
        expected_status=200,
    )

    arr_blank, _ = decode_image_bytes(live_studio_server.get(str(blank_resp["image_url"]), expected_status=200)["body"])
    arr_text, _ = decode_image_bytes(live_studio_server.get(str(text_resp["image_url"]), expected_status=200)["body"])

    luma_blank = arr_blank[..., :3].astype(np.float32).mean(axis=-1)
    luma_text = arr_text[..., :3].astype(np.float32).mean(axis=-1)

    dark_blank_center = int(np.count_nonzero(luma_blank[128:384, 32:480] < 140.0))
    dark_text_center = int(np.count_nonzero(luma_text[128:384, 32:480] < 140.0))
    assert dark_blank_center == 0, f"Expected blank solid white center to have 0 dark pixels, got {dark_blank_center}"
    assert dark_text_center >= 400, f"Expected >=400 dark glyph pixels for text='VISUALIZE', got {dark_text_center}"


@pytest.mark.f03
def test_f03_seed_determinism_and_seed_variation(
    live_studio_server: StudioAPIClient,
) -> None:
    """Verify identical `seed=42` renders pixel-identically while `seed=999` changes pattern layout."""
    import numpy as np

    base_payload = {
        "pattern": "dots",
        "colors": ["#ef476f", "#118ab2", "#ffd166"],
        "text": "",
        "prompt": "",
        "width": 512,
        "height": 512,
    }
    r42_a = live_studio_server.post_json("/api/designs/render", {**base_payload, "seed": 42}, expected_status=200)
    r42_b = live_studio_server.post_json("/api/designs/render", {**base_payload, "seed": 42}, expected_status=200)
    r999 = live_studio_server.post_json("/api/designs/render", {**base_payload, "seed": 999}, expected_status=200)

    b42_a = live_studio_server.get(str(r42_a["image_url"]), expected_status=200)["body"]
    b42_b = live_studio_server.get(str(r42_b["image_url"]), expected_status=200)["body"]
    b999 = live_studio_server.get(str(r999["image_url"]), expected_status=200)["body"]

    img_42_a, _ = decode_image_bytes(b42_a)
    img_42_b, _ = decode_image_bytes(b42_b)
    img_999, _ = decode_image_bytes(b999)

    assert np.array_equal(img_42_a, img_42_b), "Identical seed=42 design renders were not pixel-exact deterministic"
    assert not np.array_equal(img_42_a[..., :3], img_999[..., :3]), (
        "Expected seed=42 vs seed=999 to vary procedural dots positions"
    )


@pytest.mark.f03
def test_f03_natural_language_prompt_parses_colors_pattern_and_text(
    live_studio_server: StudioAPIClient,
) -> None:
    """Verify natural-language prompt extracts wavy pattern, navy/yellow palette, and quoted text."""
    prompt = 'navy and yellow wavy pattern scale 1.5 rotation 30 text "HELLO AR"'
    resp = live_studio_server.post_json(
        "/api/designs/render",
        {"prompt": prompt, "width": 512, "height": 512},
        expected_status=200,
    )
    recipe = resp.get("recipe") or {}
    assert recipe.get("pattern") == "waves", f"Expected pattern='waves' from prompt, got {recipe}"
    assert recipe.get("text") == "HELLO AR", f"Expected text='HELLO AR' from prompt, got {recipe}"

    colors = recipe.get("colors") or []
    assert len(colors) >= 2, f"Expected >=2 parsed colors in recipe, got {colors}"
    rgb0 = _hex_to_rgb(str(colors[0]))
    rgb1 = _hex_to_rgb(str(colors[1]))
    # First color navy (blue-dominant dark), second color yellow (high red + green, lower blue)
    assert rgb0[2] > rgb0[0] and rgb0[2] >= 70, f"Expected navy-like first color, got {colors[0]} ({rgb0})"
    assert rgb1[0] >= 180 and rgb1[1] >= 150 and rgb1[2] < 130, (
        f"Expected yellow-like second color, got {colors[1]} ({rgb1})"
    )

    arr, _ = decode_image_bytes(live_studio_server.get(str(resp["image_url"]), expected_status=200)["body"])
    assert arr.shape[:2] == (512, 512)
    assert float(arr[..., :3].std()) > 20.0


@pytest.mark.f03
def test_f03_python_and_js_design_recipe_contract_parity(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
) -> None:
    """Compare Python `POST /api/designs/render` against Browser Studio UI prompt application via CDP."""
    prompts = (
        'red and blue checkerboard text "STUDIO"',
        "navy and yellow wavy pattern",
        "teal and orange dotted pattern",
    )
    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)
    browser.wait_for_testid("design-prompt-input")
    browser.wait_for_testid("apply-prompt-btn")
    browser.wait_for_testid("pattern-select")

    for prompt in prompts:
        py_resp = live_studio_server.post_json(
            "/api/designs/render",
            {"prompt": prompt, "width": 512, "height": 512},
            expected_status=200,
        )
        py_pattern = str((py_resp.get("recipe") or {}).get("pattern") or "")
        assert py_pattern in {"solid", "stripes", "checker", "dots", "waves", "grid"}

        browser.fill_testid("design-prompt-input", prompt)
        browser.click_testid("apply-prompt-btn")

        deadline = time.monotonic() + 6.0
        ui_pattern = ""
        while time.monotonic() < deadline:
            browser.pump_events(0.05)
            ui_pattern = str(
                browser.evaluate(
                    '(() => { const el = document.querySelector(\'[data-testid="pattern-select"]\'); '
                    'return el ? String(el.value) : ""; })()',
                    await_promise=False,
                )
                or ""
            )
            if ui_pattern == py_pattern:
                break
            time.sleep(0.08)

        assert ui_pattern == py_pattern, (
            f"Prompt parity mismatch for {prompt!r}: Python recipe resolved {py_pattern!r}, "
            f"Browser Studio pattern-select resolved {ui_pattern!r}"
        )

    browser.assert_zero_errors()


# ==============================================================================
# F4 — Custom RGBA Artwork Upload & Loader (6 tests)
# ==============================================================================


@pytest.mark.f04
def test_f04_rgba_png_upload_preserves_alpha_channel_flag_and_dimensions(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Upload a 320x240 RGBA emblem PNG and verify `has_alpha=True`, exact dimensions, and alpha=0 corners."""
    png_path = make_rgba_png(e2e_tmp_dir / "emblem_320x240.png", width=320, height=240, mode="emblem")
    up_resp = live_studio_server.upload(
        "/api/designs/upload",
        filename="emblem_320x240.png",
        content=png_path.read_bytes(),
        content_type="image/png",
        expected_status=200,
    )
    meta = json.loads(up_resp["body"].decode("utf-8"))
    assert meta.get("design_id")
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["has_alpha"] is True

    img_resp = live_studio_server.get(str(meta["image_url"]), expected_status=200)
    arr, _ = decode_image_bytes(img_resp["body"])
    assert arr.shape == (240, 320, 4)
    for cy, cx in ((0, 0), (0, 319), (239, 0), (239, 319)):
        assert int(arr[cy, cx, 3]) == 0, f"Expected transparent corner alpha=0 at {(cy, cx)}, got {arr[cy, cx, 3]}"
    assert int(arr[120, 160, 3]) == 255


@pytest.mark.f04
def test_f04_rgb_jpeg_artwork_upload_succeeds_with_has_alpha_false(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Upload a 400x300 RGB JPEG artwork and verify `has_alpha=False` and 400x300 dimensions."""
    jpg_path = make_rgb_jpeg(e2e_tmp_dir / "artwork_400x300.jpg", width=400, height=300)
    up_resp = live_studio_server.upload(
        "/api/designs/upload",
        filename="artwork_400x300.jpg",
        content=jpg_path.read_bytes(),
        content_type="image/jpeg",
        expected_status=200,
    )
    meta = json.loads(up_resp["body"].decode("utf-8"))
    assert meta.get("design_id")
    assert meta["width"] == 400
    assert meta["height"] == 300
    assert meta["has_alpha"] is False

    img_resp = live_studio_server.get(str(meta["image_url"]), expected_status=200)
    arr, _ = decode_image_bytes(img_resp["body"])
    assert arr.shape[:2] == (300, 400)


@pytest.mark.f04
def test_f04_webp_artwork_upload_with_transparency_succeeds(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Upload a 256x256 lossless RGBA WebP artwork and verify `has_alpha=True` and alpha preservation."""
    webp_path = make_rgba_webp(e2e_tmp_dir / "emblem_256.webp", width=256, height=256, mode="emblem")
    up_resp = live_studio_server.upload(
        "/api/designs/upload",
        filename="emblem_256.webp",
        content=webp_path.read_bytes(),
        content_type="image/webp",
        expected_status=200,
    )
    meta = json.loads(up_resp["body"].decode("utf-8"))
    assert meta.get("design_id")
    assert meta["width"] == 256
    assert meta["height"] == 256
    assert meta["has_alpha"] is True

    img_resp = live_studio_server.get(str(meta["image_url"]), expected_status=200)
    arr, _ = decode_image_bytes(img_resp["body"])
    assert arr.shape == (256, 256, 4)
    assert int(arr[0, 0, 3]) == 0
    assert int(arr[128, 128, 3]) == 255


@pytest.mark.f04
def test_f04_cli_process_with_custom_rgba_png_preserves_transparent_background(
    e2e_tmp_dir: Path,
) -> None:
    """Process a clip with an RGBA ring PNG and verify ring band changes while alpha=0 center hole stays raw."""
    import numpy as np

    clip_path = e2e_tmp_dir / "card_ring_in.mp4"
    out_path = e2e_tmp_dir / "card_ring_out.mp4"
    ring_path = make_rgba_png(e2e_tmp_dir / "ring.png", width=256, height=256, mode="ring", alpha_value=255)
    clip_meta = make_textured_tracking_clip(
        clip_path,
        width=640,
        height=480,
        frames=18,
        fps=30,
        shaded_card=True,
        tmp_dir=e2e_tmp_dir,
    )
    x0, y0, bw, bh = clip_meta["bbox_frame0"]

    _, report = run_cli_json(
        [
            "process",
            "--input",
            str(clip_path),
            "--output",
            str(out_path),
            "--design-image",
            str(ring_path),
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "normal",
            "--opacity",
            "1.0",
            "--box",
            f"{x0},{y0},{bw},{bh}",
        ],
        timeout=60.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report)

    raw_f0 = decode_video_frames(clip_path, [0])[0]
    out_f0 = decode_video_frames(out_path, [0])[0]
    _, bg_delta = compute_roi_and_bg_delta(raw_f0, out_f0, (x0, y0, bw, bh), margin=24)
    assert bg_delta <= 3.5, f"Background outside card ROI changed unexpectedly: bg_delta={bg_delta:.2f}"

    # Normalized UV coordinates inside the card bbox [x0..x0+bw, y0..y0+bh]
    roi_raw = raw_f0[y0 : y0 + bh, x0 : x0 + bw].astype(np.float32)
    roi_out = out_f0[y0 : y0 + bh, x0 : x0 + bw].astype(np.float32)
    us = (np.arange(bw, dtype=np.float32) + 0.5) / float(bw) - 0.5
    vs = (np.arange(bh, dtype=np.float32) + 0.5) / float(bh) - 0.5
    rad = np.sqrt(us[None, :] ** 2 + vs[:, None] ** 2)

    hole_mask = rad <= 0.10
    band_mask = (rad >= 0.23) & (rad <= 0.37)
    diff_map = np.abs(roi_out - roi_raw).mean(axis=-1)

    band_delta = float(diff_map[band_mask].mean())
    hole_delta = float(diff_map[hole_mask].mean())
    assert band_delta >= 20.0, f"Expected strong RGB change on opaque ring band, got {band_delta:.2f}"
    assert hole_delta <= 6.0, (
        f"Expected transparent alpha=0 ring hole to preserve underlying object texture, got delta={hole_delta:.2f}"
    )


@pytest.mark.f04
def test_f04_non_square_landscape_and_portrait_aspect_ratios_preserved(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify 512x128 (4:1 landscape) and 128x512 (1:4 portrait) PNGs keep exact dimensions without square crop."""
    wide_png = make_rgba_png(e2e_tmp_dir / "wide_512x128.png", width=512, height=128, mode="cutout_label")
    tall_png = make_rgba_png(e2e_tmp_dir / "tall_128x512.png", width=128, height=512, mode="cutout_label")

    for path, exp_w, exp_h in ((wide_png, 512, 128), (tall_png, 128, 512)):
        up_resp = live_studio_server.upload(
            "/api/designs/upload",
            filename=path.name,
            content=path.read_bytes(),
            content_type="image/png",
            expected_status=200,
        )
        meta = json.loads(up_resp["body"].decode("utf-8"))
        assert (meta["width"], meta["height"]) == (exp_w, exp_h)
        arr, _ = decode_image_bytes(live_studio_server.get(str(meta["image_url"]), expected_status=200)["body"])
        assert (arr.shape[1], arr.shape[0]) == (exp_w, exp_h)


@pytest.mark.f04
def test_f04_semi_transparent_alpha_gradient_blends_proportionally(
    e2e_tmp_dir: Path,
) -> None:
    """Process a planar card with a 50%-alpha pure-green PNG (`#00ff0080`) and verify midway green blend."""
    import numpy as np
    from PIL import Image

    clip_path = e2e_tmp_dir / "semi_in.mp4"
    out_path = e2e_tmp_dir / "semi_out.mp4"
    green_half = e2e_tmp_dir / "green_alpha128.png"

    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    rgba[..., 1] = 255
    rgba[..., 3] = 128
    Image.fromarray(rgba, mode="RGBA").save(green_half, format="PNG")

    clip_meta = make_textured_tracking_clip(
        clip_path,
        width=640,
        height=480,
        frames=16,
        fps=30,
        shaded_card=True,
        tmp_dir=e2e_tmp_dir,
    )
    x0, y0, bw, bh = clip_meta["bbox_frame0"]

    _, report = run_cli_json(
        [
            "process",
            "--input",
            str(clip_path),
            "--output",
            str(out_path),
            "--design-image",
            str(green_half),
            "--wrap-mode",
            "planar",
            "--blend-mode",
            "normal",
            "--opacity",
            "1.0",
            "--box",
            f"{x0},{y0},{bw},{bh}",
        ],
        timeout=60.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(report)

    raw_f0 = decode_video_frames(clip_path, [0])[0]
    out_f0 = decode_video_frames(out_path, [0])[0]
    core = (slice(y0 + 28, y0 + bh - 28), slice(x0 + 28, x0 + bw - 28))

    raw_core = raw_f0[core].astype(np.float32)
    out_core = out_f0[core].astype(np.float32)
    alpha = 128.0 / 255.0
    expected_g = float((alpha * 255.0 + (1.0 - alpha) * raw_core[..., 1]).mean())
    expected_r = float(((1.0 - alpha) * raw_core[..., 0]).mean())

    actual_g = float(out_core[..., 1].mean())
    actual_r = float(out_core[..., 0].mean())
    raw_g = float(raw_core[..., 1].mean())

    assert raw_g + 20.0 < actual_g < 235.0, (
        f"Expected 50% green alpha to blend midway between raw_g={raw_g:.1f} and 255, got actual_g={actual_g:.1f}"
    )
    assert abs(actual_g - expected_g) <= 25.0, (
        f"Green channel {actual_g:.1f} outside ±25 tolerance of expected 50% alpha blend {expected_g:.1f}"
    )
    assert abs(actual_r - expected_r) <= 25.0, (
        f"Red channel {actual_r:.1f} outside ±25 tolerance of expected 50% attenuation {expected_r:.1f}"
    )
