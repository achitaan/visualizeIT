"""Tier 2 Boundary, Corner-Case & Negative Error-Handling E2E Tests — Slice A (`F1`–`F4`, 24 tests).

Covers:
- `F1`: Env, Bootstrap & Legacy Compatibility Boundaries (6 tests)
- `F2`: Native C++17 Rasterizer Boundaries (6 tests)
- `F3`: Procedural Design & Prompt Parser Boundaries (6 tests)
- `F4`: Custom RGBA Artwork Upload & Loader Boundaries (6 tests)

Strictly opaque-box: zero module-scope imports of `visualizeit.*` or third-party packages
(other than `pytest`).
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path
from typing import Callable

import pytest

from tests.e2e._helpers import (
    BLEND_NORMAL,
    HeadlessChromiumSession,
    REPO_ROOT,
    SETUP_ENV_SCRIPT,
    StudioAPIClient,
    VI_INVALID_ARGUMENT,
    assert_cli_usage_error,
    assert_no_python_traceback,
    assert_video_run_report_invariants,
    call_pose_to_unity,
    call_rasterize_mesh_rgbd,
    compute_psnr,
    compute_roi_and_bg_delta,
    decode_image_bytes,
    decode_video_frames,
    get_venv_python,
    load_native_abi,
    make_planar_quad_mesh,
    make_rgba_png,
    make_textured_tracking_clip,
    run_cli,
    run_cli_json,
    run_subproc,
)

pytestmark = pytest.mark.tier2

_HOST_TMP_DAEMON_PREFIXES = (
    "agy-plugin-proxy.",
    "initgoogle_syslog_dir",
)


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


# ==============================================================================
# F1 — Env, Bootstrap & Legacy Compatibility Boundaries (6 tests)
# ==============================================================================


@pytest.mark.f01
def test_f01_bound_prototype_bootstrap_without_no_site_flag_exits_cleanly(
    e2e_tmp_dir: Path,
) -> None:
    """Verify invoking `tools/prototype_bootstrap.py` without `-S` exits non-zero with clean message."""
    bootstrap = REPO_ROOT / "tools" / "prototype_bootstrap.py"
    proc = run_subproc(
        [str(get_venv_python()), "-B", str(bootstrap), "tools.evaluate_tracking"],
        timeout=15.0,
        cwd=REPO_ROOT,
        tmp_dir=e2e_tmp_dir,
    )
    assert proc.returncode != 0, "Expected non-zero exit when `-S` flag is omitted"
    combined = f"{proc.stdout}\n{proc.stderr}"
    assert "Use python -B -S" in combined, f"Expected '-B -S' usage guidance, got:\n{combined}"
    assert_no_python_traceback(proc)


@pytest.mark.f01
def test_f01_bound_prototype_bootstrap_missing_module_arg_exits_cleanly(
    e2e_tmp_dir: Path,
) -> None:
    """Verify invoking `tools/prototype_bootstrap.py` under `-B -S` with no module name exits cleanly."""
    bootstrap = REPO_ROOT / "tools" / "prototype_bootstrap.py"
    proc = run_subproc(
        [str(get_venv_python()), "-B", "-S", str(bootstrap)],
        timeout=15.0,
        cwd=REPO_ROOT,
        tmp_dir=e2e_tmp_dir,
    )
    assert proc.returncode != 0, "Expected non-zero exit when module argument is missing"
    combined = f"{proc.stdout}\n{proc.stderr}"
    assert "A module is required." in combined, f"Expected 'A module is required.', got:\n{combined}"
    assert_no_python_traceback(proc)


@pytest.mark.f01
def test_f01_bound_setup_env_respects_custom_xfs_tmpdir_and_writes_nothing_to_tmp(
    e2e_tmp_dir: Path,
) -> None:
    """Verify `TMPDIR=<xfs_probe> bash scripts/setup_env.sh` writes zero entries into `/tmp` or `.agents/teamwork/`."""
    probe_tmp = e2e_tmp_dir / "custom_setup_tmp"
    probe_tmp.mkdir(parents=True, exist_ok=True)

    before_tmp = _snapshot_non_daemon_tmp()
    proc = run_subproc(
        ["bash", str(SETUP_ENV_SCRIPT)],
        timeout=120.0,
        cwd=REPO_ROOT,
        tmp_dir=probe_tmp,
    )
    after_tmp = _snapshot_non_daemon_tmp()

    assert proc.returncode == 0, (
        f"scripts/setup_env.sh failed under custom XFS TMPDIR:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    leaked_tmp = sorted(after_tmp - before_tmp)
    assert not leaked_tmp, f"setup_env.sh created unexpected entries in /tmp: {leaked_tmp}"
    forbidden_tw = _find_forbidden_teamwork_files()
    assert not forbidden_tw, f"Forbidden files created inside .agents/teamwork/: {forbidden_tw}"


@pytest.mark.f01
def test_f01_bound_pose_abi_rejects_null_nan_inf_and_nonrigid_matrices() -> None:
    """Verify `vi_cv_gltf_pose_to_unity` returns `VI_INVALID_ARGUMENT (1)` on NULL, NaN, Inf, scale, and reflection."""
    import numpy as np

    lib = load_native_abi()
    valid = np.eye(4, dtype=np.float32)
    out_buf = np.zeros((4, 4), dtype=np.float32)
    in_ptr = valid.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    assert int(lib.vi_cv_gltf_pose_to_unity(None, out_ptr)) == VI_INVALID_ARGUMENT
    assert int(lib.vi_cv_gltf_pose_to_unity(in_ptr, None)) == VI_INVALID_ARGUMENT
    assert int(lib.vi_cv_gltf_pose_to_unity(None, None)) == VI_INVALID_ARGUMENT

    nan_mat = np.eye(4, dtype=np.float32)
    nan_mat[0, 3] = np.float32(np.nan)
    assert call_pose_to_unity(nan_mat, lib=lib)[0] == VI_INVALID_ARGUMENT

    pos_inf_mat = np.eye(4, dtype=np.float32)
    pos_inf_mat[1, 3] = np.float32(np.inf)
    assert call_pose_to_unity(pos_inf_mat, lib=lib)[0] == VI_INVALID_ARGUMENT

    neg_inf_mat = np.eye(4, dtype=np.float32)
    neg_inf_mat[2, 2] = np.float32(-np.inf)
    assert call_pose_to_unity(neg_inf_mat, lib=lib)[0] == VI_INVALID_ARGUMENT

    scaled_mat = np.diag(np.array([2.0, 2.0, 2.0, 1.0], dtype=np.float32))
    assert call_pose_to_unity(scaled_mat, lib=lib)[0] == VI_INVALID_ARGUMENT

    reflected_mat = np.diag(np.array([-1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    assert call_pose_to_unity(reflected_mat, lib=lib)[0] == VI_INVALID_ARGUMENT

    bad_bottom_row = np.eye(4, dtype=np.float32)
    bad_bottom_row[3, 2] = 0.5
    assert call_pose_to_unity(bad_bottom_row, lib=lib)[0] == VI_INVALID_ARGUMENT


@pytest.mark.f01
def test_f01_bound_unknown_cli_subcommand_and_flag_exit_2_no_traceback(
    e2e_tmp_dir: Path,
) -> None:
    """Verify unknown CLI subcommands and bogus flags exit with code 2 and zero Python traceback."""
    proc_subcmd = run_cli(["nonexistent-cmd"], timeout=15.0, tmp_dir=e2e_tmp_dir)
    assert_cli_usage_error(proc_subcmd)

    proc_flag = run_cli(["process", "--bogus-flag"], timeout=15.0, tmp_dir=e2e_tmp_dir)
    assert_cli_usage_error(proc_flag)


@pytest.mark.f01
def test_f01_bound_legacy_design_studio_store_passes_on_non_symlink_xfs_root(
    e2e_tmp_dir: Path,
) -> None:
    """Verify `tools.test_design_studio` passes all tests when `TMPDIR` is on symlink-free XFS `.cache/tmp`."""
    for ancestor in (e2e_tmp_dir, *e2e_tmp_dir.parents):
        assert not ancestor.is_symlink(), f"Unexpected symlink ancestor on XFS temp path: {ancestor}"

    proc = run_subproc(
        [str(get_venv_python()), "-B", "-m", "unittest", "tools.test_design_studio"],
        timeout=45.0,
        cwd=REPO_ROOT,
        tmp_dir=e2e_tmp_dir,
    )
    assert proc.returncode == 0, (
        f"tools.test_design_studio failed on XFS TMPDIR:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert_no_python_traceback(proc)


# ==============================================================================
# F2 — Native C++17 Rasterizer Boundaries (6 tests)
# ==============================================================================


@pytest.mark.f02
def test_f02_bound_null_pointers_return_negative_error_code_without_segfault() -> None:
    """Pass `NULL` individually for all 6 required `vi_rasterize_mesh_rgbd` pointers and verify `rc < 0`."""
    import numpy as np

    lib = load_native_abi()
    verts, tris, uvs = make_planar_quad_mesh()
    tex = np.full((16, 16, 4), 200, dtype=np.uint8)
    frame_rgb = np.full((64, 64, 3), 77, dtype=np.uint8)
    depth_buf = np.full((64, 64), 99.0, dtype=np.float32)
    mask_buf = np.zeros((64, 64), dtype=np.uint8)

    v_ptr = verts.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    t_ptr = tris.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    uv_ptr = uvs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    tex_ptr = tex.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    rgb_ptr = frame_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    d_ptr = depth_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    m_ptr = mask_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    null_cases = [
        ("vertices_cam", None, t_ptr, uv_ptr, tex_ptr, rgb_ptr, d_ptr),
        ("triangles", v_ptr, None, uv_ptr, tex_ptr, rgb_ptr, d_ptr),
        ("vertex_uvs", v_ptr, t_ptr, None, tex_ptr, rgb_ptr, d_ptr),
        ("texture_rgba", v_ptr, t_ptr, uv_ptr, None, rgb_ptr, d_ptr),
        ("frame_rgb_inout", v_ptr, t_ptr, uv_ptr, tex_ptr, None, d_ptr),
        ("depth_buf_inout", v_ptr, t_ptr, uv_ptr, tex_ptr, rgb_ptr, None),
    ]
    for label, vp, tp, uvp, texp, rgbp, dp in null_cases:
        rc = int(
            lib.vi_rasterize_mesh_rgbd(
                vp,
                4,
                tp,
                2,
                uvp,
                texp,
                16,
                16,
                0,
                100.0,
                100.0,
                32.0,
                32.0,
                1.0,
                BLEND_NORMAL,
                rgbp,
                dp,
                m_ptr,
                64,
                64,
            )
        )
        assert rc < 0, f"Expected negative error return code when {label}=NULL, got rc={rc}"

    # Verify non-null sentinel buffers were not mutated on rejected calls
    assert np.all(frame_rgb == 77)
    assert np.all(depth_buf == 99.0)
    assert np.all(mask_buf == 0)


@pytest.mark.f02
def test_f02_bound_null_optional_mask_out_succeeds_cleanly() -> None:
    """Verify `mask_out=NULL` is accepted and produces identical RGB + depth output to non-null `mask_out`."""
    import numpy as np

    verts, tris, uvs = make_planar_quad_mesh(width_m=0.36, height_m=0.26, z_m=1.0)
    tex = np.zeros((32, 32, 4), dtype=np.uint8)
    tex[...] = [44, 196, 128, 255]

    bg = np.full((240, 320, 3), 50, dtype=np.uint8)
    res_null = call_rasterize_mesh_rgbd(
        verts,
        tris,
        uvs,
        tex,
        fx=260.0,
        fy=260.0,
        cx=160.0,
        cy=120.0,
        frame_rgb_inout=bg.copy(),
        mask_out=None,
        opacity=1.0,
        blend_mode=BLEND_NORMAL,
    )
    res_mask = call_rasterize_mesh_rgbd(
        verts,
        tris,
        uvs,
        tex,
        fx=260.0,
        fy=260.0,
        cx=160.0,
        cy=120.0,
        frame_rgb_inout=bg.copy(),
        mask_out=True,
        opacity=1.0,
        blend_mode=BLEND_NORMAL,
    )
    assert res_null["mask"] is None
    assert res_null["rc"] > 3000 and res_null["rc"] == res_mask["rc"]
    assert np.array_equal(res_null["frame_rgb"], res_mask["frame_rgb"])
    assert np.array_equal(res_null["depth_buf"], res_mask["depth_buf"])


@pytest.mark.f02
def test_f02_bound_zero_or_negative_dimensions_and_out_of_bounds_indices_rejected() -> None:
    """Verify non-positive dimensions/intrinsics and out-of-bounds vertex indices return `< 0` safely."""
    import numpy as np

    lib = load_native_abi()
    verts, tris, uvs = make_planar_quad_mesh()
    tex = np.full((16, 16, 4), 255, dtype=np.uint8)
    frame_rgb = np.full((64, 64, 3), 42, dtype=np.uint8)
    depth_buf = np.full((64, 64), np.inf, dtype=np.float32)

    v_ptr = verts.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    t_ptr = tris.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    uv_ptr = uvs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    tex_ptr = tex.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    rgb_ptr = frame_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    d_ptr = depth_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def _call(
        *,
        nv: int = 4,
        tp: object = t_ptr,
        nt: int = 2,
        tw: int = 16,
        th: int = 16,
        fx: float = 100.0,
        fy: float = 100.0,
        fw: int = 64,
        fh: int = 64,
    ) -> int:
        return int(
            lib.vi_rasterize_mesh_rgbd(
                v_ptr,
                nv,
                tp,
                nt,
                uv_ptr,
                tex_ptr,
                tw,
                th,
                0,
                fx,
                fy,
                32.0,
                32.0,
                1.0,
                BLEND_NORMAL,
                rgb_ptr,
                d_ptr,
                None,
                fw,
                fh,
            )
        )

    assert _call(fw=0) < 0
    assert _call(fh=-1) < 0
    assert _call(tw=0) < 0
    assert _call(th=-4) < 0
    assert _call(fx=0.0) < 0
    assert _call(fy=-200.0) < 0
    assert _call(nv=0, nt=1) < 0

    bad_hi = np.array([[0, 1, 99]], dtype=np.int32)
    assert _call(tp=bad_hi.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), nt=1) < 0

    bad_neg = np.array([[0, -1, 2]], dtype=np.int32)
    assert _call(tp=bad_neg.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), nt=1) < 0
    assert _call(nt=-3) < 0

    # Zero triangles: either rejected as non-positive dimension (< 0) or 0-pixel no-op (== 0), never mutating frame
    assert _call(nt=0) <= 0
    assert np.all(frame_rgb == 42)
    assert np.all(np.isinf(depth_buf))


@pytest.mark.f02
def test_f02_bound_opacity_exact_0_and_1_vs_out_of_range_negative_and_above_1() -> None:
    """Test exact `opacity=0.0` (unchanged), `opacity=1.0` (full), and safe handling of `-0.1` / `1.5`."""
    import numpy as np

    verts, tris, uvs = make_planar_quad_mesh(width_m=0.4, height_m=0.3, z_m=1.0)
    tex = np.zeros((32, 32, 4), dtype=np.uint8)
    tex[...] = [242, 46, 28, 255]
    bg_orig = np.full((240, 320, 3), [55, 95, 135], dtype=np.uint8)

    res_zero = call_rasterize_mesh_rgbd(
        verts,
        tris,
        uvs,
        tex,
        fx=250.0,
        fy=250.0,
        cx=160.0,
        cy=120.0,
        frame_rgb_inout=bg_orig.copy(),
        opacity=0.0,
        blend_mode=BLEND_NORMAL,
    )
    assert res_zero["rc"] >= 0
    assert np.array_equal(res_zero["frame_rgb"], bg_orig), "opacity=0.0 mutated background RGB pixels"

    res_one = call_rasterize_mesh_rgbd(
        verts,
        tris,
        uvs,
        tex,
        fx=250.0,
        fy=250.0,
        cx=160.0,
        cy=120.0,
        frame_rgb_inout=bg_orig.copy(),
        opacity=1.0,
        blend_mode=BLEND_NORMAL,
    )
    assert res_one["rc"] > 2000
    assert np.all(res_one["frame_rgb"][95:145, 120:200] == np.array([242, 46, 28], dtype=np.uint8))

    # Out-of-range opacity in C++ ABI: either rejected (`rc < 0`, untouched buffer) or safely clamped to [0.0, 1.0]
    res_neg = call_rasterize_mesh_rgbd(
        verts,
        tris,
        uvs,
        tex,
        fx=250.0,
        fy=250.0,
        cx=160.0,
        cy=120.0,
        frame_rgb_inout=bg_orig.copy(),
        opacity=-0.1,
        blend_mode=BLEND_NORMAL,
    )
    assert np.array_equal(res_neg["frame_rgb"], bg_orig), (
        f"opacity=-0.1 caused unexpected pixel mutation/underflow (rc={res_neg['rc']})"
    )

    res_hi = call_rasterize_mesh_rgbd(
        verts,
        tris,
        uvs,
        tex,
        fx=250.0,
        fy=250.0,
        cx=160.0,
        cy=120.0,
        frame_rgb_inout=bg_orig.copy(),
        opacity=1.5,
        blend_mode=BLEND_NORMAL,
    )
    if res_hi["rc"] < 0:
        assert np.array_equal(res_hi["frame_rgb"], bg_orig)
    else:
        assert np.array_equal(res_hi["frame_rgb"], res_one["frame_rgb"]), (
            "opacity=1.5 accepted without clamping to 1.0 (possible uint8 wraparound)"
        )


@pytest.mark.f02
def test_f02_bound_vertices_behind_camera_and_near_plane_straddling_triangle() -> None:
    """Verify vertices at `z <= 0` render 0 pixels safely and near-plane straddling triangles never corrupt depth."""
    import numpy as np

    tex = np.full((32, 32, 4), [200, 100, 50, 255], dtype=np.uint8)
    bg_orig = np.full((240, 320, 3), 28, dtype=np.uint8)

    # All vertices behind camera (z = -1.0m)
    verts_behind, tris, uvs = make_planar_quad_mesh(width_m=0.4, height_m=0.3, z_m=-1.0)
    res_behind = call_rasterize_mesh_rgbd(
        verts_behind,
        tris,
        uvs,
        tex,
        fx=250.0,
        fy=250.0,
        cx=160.0,
        cy=120.0,
        frame_rgb_inout=bg_orig.copy(),
        mask_out=True,
    )
    assert res_behind["rc"] == 0
    assert np.array_equal(res_behind["frame_rgb"], bg_orig)
    assert np.all(res_behind["mask"] == 0)
    assert not np.any(np.isnan(res_behind["depth_buf"]))

    # Triangle straddling camera near plane: v0 behind (z=-0.2m), v1 & v2 in front (z=+1.0m)
    verts_straddle = np.array(
        [
            [0.0, -0.10, -0.20],
            [-0.20, +0.18, +1.00],
            [+0.20, +0.18, +1.00],
        ],
        dtype=np.float32,
    )
    tris_one = np.array([[0, 1, 2]], dtype=np.int32)
    uvs_one = np.array([[0.5, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    res_straddle = call_rasterize_mesh_rgbd(
        verts_straddle,
        tris_one,
        uvs_one,
        tex,
        fx=250.0,
        fy=250.0,
        cx=160.0,
        cy=120.0,
        frame_rgb_inout=bg_orig.copy(),
        mask_out=True,
    )
    assert res_straddle["rc"] >= 0
    depth = res_straddle["depth_buf"]
    assert not np.any(np.isnan(depth))
    written = np.isfinite(depth)
    if np.any(written):
        assert np.all(depth[written] > 0.0), "Near-plane straddling triangle wrote non-positive depth values"


@pytest.mark.f02
def test_f02_bound_degenerate_collinear_and_offscreen_triangles_render_zero_pixels() -> None:
    """Verify zero-area collinear triangles and completely off-screen quads write 0 pixels."""
    import numpy as np

    tex = np.full((32, 32, 4), 255, dtype=np.uint8)
    bg_orig = np.full((240, 320, 3), 64, dtype=np.uint8)

    verts_collinear = np.array(
        [[-0.20, -0.10, 1.0], [0.00, 0.00, 1.0], [0.20, 0.10, 1.0]],
        dtype=np.float32,
    )
    tris_col = np.array([[0, 1, 2]], dtype=np.int32)
    uvs_col = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
    res_col = call_rasterize_mesh_rgbd(
        verts_collinear,
        tris_col,
        uvs_col,
        tex,
        fx=250.0,
        fy=250.0,
        cx=160.0,
        cy=120.0,
        frame_rgb_inout=bg_orig.copy(),
        mask_out=True,
    )
    assert res_col["rc"] == 0
    assert np.array_equal(res_col["frame_rgb"], bg_orig)
    assert np.all(res_col["mask"] == 0)

    for center_xy in ((50.0, 50.0), (-50.0, -50.0)):
        verts_off, tris_off, uvs_off = make_planar_quad_mesh(
            width_m=0.4, height_m=0.3, z_m=1.0, center_xy=center_xy
        )
        res_off = call_rasterize_mesh_rgbd(
            verts_off,
            tris_off,
            uvs_off,
            tex,
            fx=250.0,
            fy=250.0,
            cx=160.0,
            cy=120.0,
            frame_rgb_inout=bg_orig.copy(),
            mask_out=True,
        )
        assert res_off["rc"] == 0
        assert np.array_equal(res_off["frame_rgb"], bg_orig)
        assert np.all(res_off["mask"] == 0)


# ==============================================================================
# F3 — Procedural Design & Prompt Parser Boundaries (6 tests)
# ==============================================================================


@pytest.mark.f03
def test_f03_bound_invalid_pattern_name_rejected_in_cli_and_rest(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify unsupported `--pattern dragon` exits 2 in CLI and returns 400/422 in REST."""
    clip_path = e2e_tmp_dir / "tiny.mp4"
    make_textured_tracking_clip(clip_path, width=64, height=64, frames=10, fps=30, tmp_dir=e2e_tmp_dir)

    proc = run_cli(
        [
            "process",
            "--input",
            str(clip_path),
            "--output",
            str(e2e_tmp_dir / "out.mp4"),
            "--pattern",
            "dragon",
        ],
        timeout=20.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_cli_usage_error(proc)

    live_studio_server.post_json(
        "/api/designs/render",
        {"pattern": "dragon", "colors": ["#ff0000", "#00ff00"], "width": 256, "height": 256},
        expected_status=(400, 422),
    )


@pytest.mark.f03
def test_f03_bound_malformed_hex_colors_and_empty_palette_rejected(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify unambiguously invalid hex tokens (`#gggggg`, `not_a_hex_color`) and empty palettes are rejected."""
    clip_path = e2e_tmp_dir / "tiny_colors.mp4"
    make_textured_tracking_clip(clip_path, width=64, height=64, frames=10, fps=30, tmp_dir=e2e_tmp_dir)

    for bad_cli_colors in ("#gggggg,#1982c4", "not_a_hex_color", ""):
        proc = run_cli(
            [
                "process",
                "--input",
                str(clip_path),
                "--output",
                str(e2e_tmp_dir / "out.mp4"),
                "--pattern",
                "stripes",
                "--colors",
                bad_cli_colors,
            ],
            timeout=20.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)

    for bad_palette in (["#gggggg", "#1982c4"], ["not_a_hex_color", "#1982c4"], []):
        live_studio_server.post_json(
            "/api/designs/render",
            {"pattern": "stripes", "colors": bad_palette, "width": 256, "height": 256},
            expected_status=(400, 422),
        )

    assert live_studio_server.get_json("/api/health", expected_status=200)["status"] == "ok"


@pytest.mark.f03
def test_f03_bound_xss_html_and_unicode_emoji_in_text_and_prompt_safe(
    live_studio_server: StudioAPIClient,
    chromium_session_factory: Callable[..., HeadlessChromiumSession],
) -> None:
    """Verify `<script>` XSS and Unicode/emoji in text/prompt render literally and never execute in DOM."""
    xss_payload = "<script>window.__XSS=1</script><img src=x onerror='window.__XSS=2'>"
    unicode_payload = "🚀 Café 日本語"

    for txt in (xss_payload, unicode_payload):
        resp = live_studio_server.post_json(
            "/api/designs/render",
            {
                "pattern": "stripes",
                "colors": ["#ffffff", "#111111"],
                "text": txt,
                "prompt": f'blue stripes text "{txt}"',
                "width": 256,
                "height": 256,
            },
            expected_status=200,
        )
        img_resp = live_studio_server.get(str(resp["image_url"]), expected_status=200)
        arr, _ = decode_image_bytes(img_resp["body"])
        assert arr.shape[:2] == (256, 256)

    browser = chromium_session_factory()
    browser.navigate(live_studio_server.base_url)
    browser.wait_for_testid("design-prompt-input")
    browser.wait_for_testid("apply-prompt-btn")

    browser.fill_testid("design-prompt-input", f'red checker text "{xss_payload}"')
    browser.click_testid("apply-prompt-btn")
    browser.pump_events(0.30)
    assert browser.evaluate("typeof window.__XSS", await_promise=False) == "undefined", (
        "XSS script payload executed in Browser Studio DOM!"
    )

    browser.fill_testid("design-prompt-input", f'blue waves text "{unicode_payload}"')
    browser.click_testid("apply-prompt-btn")
    browser.pump_events(0.30)
    assert browser.evaluate("typeof window.__XSS", await_promise=False) == "undefined"
    browser.assert_zero_errors(allow_console_substrings=["src=x"])


@pytest.mark.f03
def test_f03_bound_empty_prompt_and_unsupported_words_handled_gracefully(
    live_studio_server: StudioAPIClient,
) -> None:
    """Verify empty prompts and unknown descriptive words return clean fallback artwork (200) or 4xx, never 500."""
    for prompt in ("", "   ", "photorealistic quantum spaceship hyperdrive nebula"):
        resp = live_studio_server.request(
            "POST",
            "/api/designs/render",
            json_body={"prompt": prompt, "width": 256, "height": 256},
        )
        assert resp["status"] in {200, 400, 422}, (
            f"Unexpected status {resp['status']} for prompt={prompt!r}: {resp['body'][:300]!r}"
        )
        if resp["status"] == 200:
            body = json.loads(resp["body"].decode("utf-8"))
            assert body.get("recipe", {}).get("pattern") in {
                "solid",
                "stripes",
                "checker",
                "dots",
                "waves",
                "grid",
            }
            arr, _ = decode_image_bytes(live_studio_server.get(str(body["image_url"]), expected_status=200)["body"])
            assert arr.shape[:2] == (256, 256)

    assert live_studio_server.get_json("/api/health", expected_status=200)["status"] == "ok"


@pytest.mark.f03
def test_f03_bound_extreme_design_dimensions_tiny_negative_and_huge_rejected(
    live_studio_server: StudioAPIClient,
) -> None:
    """Verify zero, negative, and OOM-sized (32768x32768) design dimensions return 400/413/422 without crash."""
    bad_dims = (
        {"pattern": "stripes", "width": 0, "height": 256},
        {"pattern": "stripes", "width": 256, "height": -64},
        {"pattern": "stripes", "width": -128, "height": -128},
        {"pattern": "stripes", "width": 32768, "height": 32768},
    )
    for payload in bad_dims:
        live_studio_server.post_json(
            "/api/designs/render",
            payload,
            expected_status=(400, 413, 422),
        )

    assert live_studio_server.get_json("/api/health", expected_status=200)["status"] == "ok"


@pytest.mark.f03
def test_f03_bound_palette_boundary_exact_2_and_5_colors_accepted(
    live_studio_server: StudioAPIClient,
) -> None:
    """Verify exact inclusive palette length boundaries of 2 colors and 5 colors succeed with HTTP 200."""
    pal2 = ["#e63946", "#1d3557"]
    pal5 = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

    for pal in (pal2, pal5):
        resp = live_studio_server.post_json(
            "/api/designs/render",
            {"pattern": "stripes", "colors": pal, "width": 256, "height": 256},
            expected_status=200,
        )
        recipe_colors = resp.get("recipe", {}).get("colors") or []
        assert len(recipe_colors) == len(pal), f"Expected {len(pal)} colors in recipe, got {recipe_colors}"
        arr, _ = decode_image_bytes(live_studio_server.get(str(resp["image_url"]), expected_status=200)["body"])
        assert arr.shape[:2] == (256, 256)


# ==============================================================================
# F4 — Custom RGBA Artwork Upload & Loader Boundaries (6 tests)
# ==============================================================================


@pytest.mark.f04
def test_f04_bound_zero_alpha_fully_transparent_rgba_png_leaves_scene_unchanged(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify a 100% `alpha=0` PNG (`RGB=[255,32,180]`) uploads with `has_alpha=True` and leaves video untinted."""
    zero_png = make_rgba_png(e2e_tmp_dir / "zero_alpha.png", width=256, height=256, mode="zero_alpha")

    up_resp = live_studio_server.upload(
        "/api/designs/upload",
        filename="zero_alpha.png",
        content=zero_png.read_bytes(),
        content_type="image/png",
        expected_status=200,
    )
    meta = json.loads(up_resp["body"].decode("utf-8"))
    assert meta["has_alpha"] is True
    assert (meta["width"], meta["height"]) == (256, 256)

    clip_path = e2e_tmp_dir / "zero_alpha_in.mp4"
    out_path = e2e_tmp_dir / "zero_alpha_out.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path, width=640, height=480, frames=15, fps=30, tmp_dir=e2e_tmp_dir
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
            str(zero_png),
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

    raw_frames = decode_video_frames(clip_path, [0, 7])
    out_frames = decode_video_frames(out_path, [0, 7])
    for idx in (0, 7):
        psnr = compute_psnr(raw_frames[idx], out_frames[idx])
        roi_delta, bg_delta = compute_roi_and_bg_delta(raw_frames[idx], out_frames[idx], (x0, y0, bw, bh))
        assert psnr >= 35.0, f"Zero-alpha PNG bled hidden RGB channels onto frame {idx} (PSNR={psnr:.2f} dB)"
        assert roi_delta <= 4.0 and bg_delta <= 3.5


@pytest.mark.f04
def test_f04_bound_empty_zero_byte_design_file_rejected(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify a 0-byte `.png` file is rejected with HTTP 400/415/422 and CLI exit code 2."""
    empty_png = e2e_tmp_dir / "empty.png"
    empty_png.write_bytes(b"")

    live_studio_server.upload(
        "/api/designs/upload",
        filename="empty.png",
        content=b"",
        content_type="image/png",
        expected_status=(400, 415, 422),
    )

    clip_path = e2e_tmp_dir / "tiny.mp4"
    make_textured_tracking_clip(clip_path, width=64, height=64, frames=10, fps=30, tmp_dir=e2e_tmp_dir)
    proc = run_cli(
        [
            "process",
            "--input",
            str(clip_path),
            "--output",
            str(e2e_tmp_dir / "out.mp4"),
            "--design-image",
            str(empty_png),
        ],
        timeout=20.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_cli_usage_error(proc)


@pytest.mark.f04
def test_f04_bound_non_image_text_and_mp4_disguised_as_png_rejected(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify plain text and `.mp4` bytes disguised as `.png` are rejected cleanly by REST and CLI."""
    clip_path = e2e_tmp_dir / "src_video.mp4"
    make_textured_tracking_clip(clip_path, width=64, height=64, frames=10, fps=30, tmp_dir=e2e_tmp_dir)

    fake_text = e2e_tmp_dir / "fake_text.png"
    fake_text.write_bytes(b"not a valid png image payload\n")

    disguised_mp4 = e2e_tmp_dir / "video_disguised.png"
    disguised_mp4.write_bytes(clip_path.read_bytes())

    for bad_file in (fake_text, disguised_mp4):
        live_studio_server.upload(
            "/api/designs/upload",
            filename=bad_file.name,
            content=bad_file.read_bytes(),
            content_type="image/png",
            expected_status=(400, 415, 422),
        )
        proc = run_cli(
            [
                "process",
                "--input",
                str(clip_path),
                "--output",
                str(e2e_tmp_dir / f"out_{bad_file.stem}.mp4"),
                "--design-image",
                str(bad_file),
            ],
            timeout=20.0,
            tmp_dir=e2e_tmp_dir,
        )
        assert_cli_usage_error(proc)


@pytest.mark.f04
def test_f04_bound_truncated_corrupt_png_payload_rejected(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify a PNG truncated mid-IDAT stream is rejected cleanly with 400/415/422 and CLI exit 2."""
    valid_png = make_rgba_png(e2e_tmp_dir / "valid_src.png", width=256, height=256, mode="emblem")
    raw_bytes = valid_png.read_bytes()
    corrupt_bytes = raw_bytes[: max(48, len(raw_bytes) // 2)]
    corrupt_png = e2e_tmp_dir / "truncated_corrupt.png"
    corrupt_png.write_bytes(corrupt_bytes)

    live_studio_server.upload(
        "/api/designs/upload",
        filename="truncated_corrupt.png",
        content=corrupt_bytes,
        content_type="image/png",
        expected_status=(400, 415, 422),
    )

    clip_path = e2e_tmp_dir / "tiny.mp4"
    make_textured_tracking_clip(clip_path, width=64, height=64, frames=10, fps=30, tmp_dir=e2e_tmp_dir)
    proc = run_cli(
        [
            "process",
            "--input",
            str(clip_path),
            "--output",
            str(e2e_tmp_dir / "out.mp4"),
            "--design-image",
            str(corrupt_png),
        ],
        timeout=20.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_cli_usage_error(proc)


@pytest.mark.f04
def test_f04_bound_tiny_1x1_and_16x16_rgba_artwork_uploads_and_warps_cleanly(
    live_studio_server: StudioAPIClient,
    e2e_tmp_dir: Path,
) -> None:
    """Verify tiny `1x1` and `16x16` RGBA PNGs upload/warp without `(width - 1)` division-by-zero bugs."""
    import numpy as np
    from PIL import Image

    png_1x1 = e2e_tmp_dir / "tiny_1x1.png"
    Image.fromarray(np.array([[[20, 220, 240, 255]]], dtype=np.uint8), mode="RGBA").save(png_1x1, format="PNG")
    png_16x16 = make_rgba_png(e2e_tmp_dir / "tiny_16x16.png", width=16, height=16, mode="semi", alpha_value=255)

    up_16 = live_studio_server.upload(
        "/api/designs/upload",
        filename="tiny_16x16.png",
        content=png_16x16.read_bytes(),
        content_type="image/png",
        expected_status=200,
    )
    meta_16 = json.loads(up_16["body"].decode("utf-8"))
    assert (meta_16["width"], meta_16["height"]) == (16, 16)

    up_1 = live_studio_server.upload(
        "/api/designs/upload",
        filename="tiny_1x1.png",
        content=png_1x1.read_bytes(),
        content_type="image/png",
        expected_status=(200, 400, 422),
    )
    if up_1["status"] == 200:
        meta_1 = json.loads(up_1["body"].decode("utf-8"))
        assert (meta_1["width"], meta_1["height"]) == (1, 1)

    clip_path = e2e_tmp_dir / "warp_tiny_in.mp4"
    clip_meta = make_textured_tracking_clip(
        clip_path, width=640, height=480, frames=12, fps=30, tmp_dir=e2e_tmp_dir
    )
    x0, y0, bw, bh = clip_meta["bbox_frame0"]

    out_16 = e2e_tmp_dir / "warp_16x16_out.mp4"
    _, rep_16 = run_cli_json(
        [
            "process",
            "--input",
            str(clip_path),
            "--output",
            str(out_16),
            "--design-image",
            str(png_16x16),
            "--blend-mode",
            "normal",
            "--box",
            f"{x0},{y0},{bw},{bh}",
        ],
        timeout=60.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_video_run_report_invariants(rep_16)
    raw_f0 = decode_video_frames(clip_path, [0])[0]
    out_16_f0 = decode_video_frames(out_16, [0])[0]
    roi_delta_16, _ = compute_roi_and_bg_delta(raw_f0, out_16_f0, (x0, y0, bw, bh))
    assert roi_delta_16 >= 10.0

    out_1 = e2e_tmp_dir / "warp_1x1_out.mp4"
    proc_1 = run_cli(
        [
            "process",
            "--input",
            str(clip_path),
            "--output",
            str(out_1),
            "--design-image",
            str(png_1x1),
            "--blend-mode",
            "normal",
            "--box",
            f"{x0},{y0},{bw},{bh}",
            "--json",
        ],
        timeout=60.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_no_python_traceback(proc_1)
    assert proc_1.returncode in (0, 2), f"Unexpected exit code {proc_1.returncode} for 1x1 PNG:\n{proc_1.stderr}"
    if proc_1.returncode == 0:
        rep_1 = json.loads(proc_1.stdout.strip())
        assert_video_run_report_invariants(rep_1)
        out_1_f0 = decode_video_frames(out_1, [0])[0]
        roi_delta_1, _ = compute_roi_and_bg_delta(raw_f0, out_1_f0, (x0, y0, bw, bh))
        assert roi_delta_1 >= 10.0


@pytest.mark.f04
def test_f04_bound_missing_nonexistent_design_image_path_exits_2(
    e2e_tmp_dir: Path,
) -> None:
    """Verify `--design-image` pointing to a non-existent path exits with code 2 and zero traceback."""
    clip_path = e2e_tmp_dir / "tiny.mp4"
    make_textured_tracking_clip(clip_path, width=64, height=64, frames=10, fps=30, tmp_dir=e2e_tmp_dir)
    missing_png = e2e_tmp_dir / "does_not_exist_artwork.png"

    proc = run_cli(
        [
            "process",
            "--input",
            str(clip_path),
            "--output",
            str(e2e_tmp_dir / "out.mp4"),
            "--design-image",
            str(missing_png),
        ],
        timeout=20.0,
        tmp_dir=e2e_tmp_dir,
    )
    assert_cli_usage_error(proc)
