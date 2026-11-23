"""Causal real-time KLT + ORB video surface tracker (planar, cylinder, grid, 6-DoF mesh)."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Literal

import cv2
import numpy as np

from .contracts import ContractValidationError, TrackFrameResult
from .native_raster import rasterize_mesh_rgbd
from .segmentation import SegmentationResult


@dataclass(slots=True)
class MeshGeometry:
    """3D triangle mesh in meters (right-handed object frame) with vertex UVs."""
    name: str
    vertices: np.ndarray   # (V, 3) float32 meters
    normals: np.ndarray    # (V, 3) float32
    uvs: np.ndarray        # (V, 2) float32
    triangles: np.ndarray  # (T, 3) int32
    source_kind: Literal["ply", "parametric"]


def load_ply_mesh(ply_path: str | Path) -> MeshGeometry:
    """Load an ASCII or binary-little-endian PLY mesh into meters."""
    path = Path(ply_path)
    if not path.is_file():
        raise ContractValidationError(f"PLY mesh file not found: {path}")

    raw = path.read_bytes()
    header_end_idx = raw.find(b"end_header\n")
    sep_len = len(b"end_header\n")
    if header_end_idx < 0:
        header_end_idx = raw.find(b"end_header\r\n")
        sep_len = len(b"end_header\r\n")
    if header_end_idx < 0:
        raise ContractValidationError(f"Invalid PLY file (missing end_header): {path}")

    header_lines = raw[:header_end_idx].decode("ascii", errors="strict").splitlines()
    if not header_lines or header_lines[0].strip() != "ply":
        raise ContractValidationError(f"Invalid PLY header in {path}")

    fmt = "ascii"
    num_verts = 0
    num_faces = 0
    current_elem = ""
    vert_props: list[tuple[str, str]] = []

    for line in header_lines[1:]:
        s = line.strip()
        if not s or s.startswith("comment"):
            continue
        parts = s.split()
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current_elem = parts[1]
            if current_elem == "vertex":
                num_verts = int(parts[2])
            elif current_elem == "face":
                num_faces = int(parts[2])
        elif parts[0] == "property" and current_elem == "vertex" and parts[1] != "list":
            vert_props.append((parts[1], parts[2]))

    if num_verts < 3 or num_faces < 1:
        raise ContractValidationError(f"PLY mesh has insufficient geometry ({num_verts} verts, {num_faces} faces)")

    prop_names = [name for _, name in vert_props]
    ix = prop_names.index("x") if "x" in prop_names else 0
    iy = prop_names.index("y") if "y" in prop_names else 1
    iz = prop_names.index("z") if "z" in prop_names else 2
    inx = prop_names.index("nx") if "nx" in prop_names else -1
    iny = prop_names.index("ny") if "ny" in prop_names else -1
    inz = prop_names.index("nz") if "nz" in prop_names else -1
    iu = prop_names.index("s") if "s" in prop_names else (prop_names.index("u") if "u" in prop_names else (prop_names.index("texture_u") if "texture_u" in prop_names else -1))
    iv = prop_names.index("t") if "t" in prop_names else (prop_names.index("v") if "v" in prop_names else (prop_names.index("texture_v") if "texture_v" in prop_names else -1))

    body = raw[header_end_idx + sep_len :]
    verts = np.zeros((num_verts, 3), dtype=np.float32)
    norms = np.zeros((num_verts, 3), dtype=np.float32)
    uvs = np.zeros((num_verts, 2), dtype=np.float32)
    tris: list[list[int]] = []

    if fmt == "ascii":
        lines = [ln for ln in body.decode("ascii", errors="strict").splitlines() if ln.strip()]
        if len(lines) < num_verts + num_faces:
            raise ContractValidationError(f"Truncated ASCII PLY file: {path}")
        for i in range(num_verts):
            toks = lines[i].split()
            verts[i, 0] = float(toks[ix])
            verts[i, 1] = float(toks[iy])
            verts[i, 2] = float(toks[iz])
            if inx >= 0 and iny >= 0 and inz >= 0:
                norms[i, 0] = float(toks[inx])
                norms[i, 1] = float(toks[iny])
                norms[i, 2] = float(toks[inz])
            if iu >= 0 and iv >= 0:
                uvs[i, 0] = float(toks[iu])
                uvs[i, 1] = float(toks[iv])
        for i in range(num_faces):
            toks = lines[num_verts + i].split()
            cnt = int(toks[0])
            idx_list = [int(toks[1 + k]) for k in range(cnt)]
            for k in range(1, cnt - 1):
                tris.append([idx_list[0], idx_list[k], idx_list[k + 1]])
    elif fmt == "binary_little_endian":
        type_map = {
            "char": "<i1", "int8": "<i1", "uchar": "<u1", "uint8": "<u1",
            "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
            "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
            "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        }
        v_dtype = np.dtype([(name, type_map[ptype]) for ptype, name in vert_props])
        v_bytes = num_verts * v_dtype.itemsize
        v_arr = np.frombuffer(body[:v_bytes], dtype=v_dtype)
        verts[:, 0] = v_arr["x"]
        verts[:, 1] = v_arr["y"]
        verts[:, 2] = v_arr["z"]
        if inx >= 0 and iny >= 0 and inz >= 0:
            norms[:, 0] = v_arr[prop_names[inx]]
            norms[:, 1] = v_arr[prop_names[iny]]
            norms[:, 2] = v_arr[prop_names[inz]]
        if iu >= 0 and iv >= 0:
            uvs[:, 0] = v_arr[prop_names[iu]]
            uvs[:, 1] = v_arr[prop_names[iv]]
        offset = v_bytes
        for _ in range(num_faces):
            cnt = int(body[offset])
            offset += 1
            idxs = np.frombuffer(body[offset : offset + 4 * cnt], dtype="<i4")
            offset += 4 * cnt
            for k in range(1, cnt - 1):
                tris.append([int(idxs[0]), int(idxs[k]), int(idxs[k + 1])])
    else:
        raise ContractValidationError(f"Unsupported PLY format {fmt!r}")

    triangles = np.asarray(tris, dtype=np.int32)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ContractValidationError(f"PLY mesh contains no valid triangles: {path}")
    if not np.all(np.isfinite(verts)):
        raise ContractValidationError(f"PLY mesh contains non-finite (NaN/Inf) vertex coordinates: {path}")
    if int(triangles.min()) < 0 or int(triangles.max()) >= num_verts:
        raise ContractValidationError(
            f"PLY mesh face indices out of range [0, {num_verts - 1}]: {path}"
        )

    # Convert millimeter PLY meshes (e.g. BOP/YCB scans) to meters automatically
    if float(np.max(np.abs(verts))) > 10.0:
        verts = (verts * 1e-3).astype(np.float32)

    # Synthesize planar/spherical UVs if missing
    if iu < 0 or iv < 0 or float(np.ptp(uvs)) < 1e-6:
        mins = verts.min(axis=0)
        spans = np.maximum(np.ptp(verts, axis=0), 1e-6)
        uvs[:, 0] = (verts[:, 0] - mins[0]) / spans[0]
        uvs[:, 1] = 1.0 - (verts[:, 1] - mins[1]) / spans[1]

    # Compute vertex normals if missing
    if float(np.linalg.norm(norms)) < 1e-6:
        norms = _compute_vertex_normals(verts, triangles)

    return MeshGeometry(
        name=path.stem,
        vertices=np.ascontiguousarray(verts, dtype=np.float32),
        normals=np.ascontiguousarray(norms, dtype=np.float32),
        uvs=np.ascontiguousarray(uvs, dtype=np.float32),
        triangles=np.ascontiguousarray(triangles, dtype=np.int32),
        source_kind="ply",
    )


def _compute_vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    vn = np.zeros_like(verts, dtype=np.float64)
    np.add.at(vn, tris[:, 0], fn)
    np.add.at(vn, tris[:, 1], fn)
    np.add.at(vn, tris[:, 2], fn)
    lens = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.divide(vn, np.maximum(lens, 1e-12))
    return vn.astype(np.float32)


def build_parametric_mesh(
    shape: str,
    *,
    cyl_radius: float = 0.06,
    cyl_height: float = 0.14,
    radius_profile: list[float] | np.ndarray | None = None,
) -> MeshGeometry:
    """Build a metric parametric 3D mesh for plane, cylinder (or axisymmetric lathe), box, or curved_panel."""
    s = shape.strip().lower()
    if s == "plane":
        nx, ny = 8, 8
        hx, hy = 0.10, 0.075
        xs = np.linspace(-hx, hx, nx + 1, dtype=np.float32)
        ys = np.linspace(-hy, hy, ny + 1, dtype=np.float32)
        verts, norms, uvs, tris = [], [], [], []
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                verts.append([x, y, 0.0])
                norms.append([0.0, 0.0, -1.0])
                uvs.append([i / nx, j / ny])
        for j in range(ny):
            for i in range(nx):
                a = j * (nx + 1) + i
                b = a + 1
                c = a + (nx + 1)
                d = c + 1
                tris.append([a, b, d])
                tris.append([a, d, c])
        return MeshGeometry("plane", np.asarray(verts, np.float32), np.asarray(norms, np.float32), np.asarray(uvs, np.float32), np.asarray(tris, np.int32), "parametric")

    if s == "cylinder":
        n_theta = 32
        if radius_profile is not None and len(radius_profile) >= 3:
            prof = np.asarray(radius_profile, dtype=np.float64).ravel()
            n_y = len(prof) - 1
        else:
            n_y = 8
            prof = np.ones(n_y + 1, dtype=np.float64)
        radius, height = float(cyl_radius), float(cyl_height)
        verts, norms, uvs, tris = [], [], [], []
        dy = height / max(n_y, 1)
        for j in range(n_y + 1):
            v_frac = j / n_y
            y = (v_frac - 0.5) * height
            r_j = radius * float(prof[j])
            # Slope dr/dy for accurate axisymmetric surface normal
            r_prev = radius * float(prof[max(0, j - 1)])
            r_next = radius * float(prof[min(n_y, j + 1)])
            dr_dy = (r_next - r_prev) / (max(1, min(n_y, j + 1) - max(0, j - 1)) * dy)
            ny_v = -dr_dy
            n_len = math.hypot(1.0, ny_v)
            radial_s = 1.0 / n_len
            ny_norm = ny_v / n_len
            for i in range(n_theta + 1):
                u_frac = i / n_theta
                theta = (u_frac - 0.5) * 2.0 * math.pi
                nx_v = math.sin(theta)
                nz_v = -math.cos(theta)
                verts.append([r_j * nx_v, y, r_j * nz_v])
                norms.append([nx_v * radial_s, ny_norm, nz_v * radial_s])
                uvs.append([u_frac, v_frac])
        for j in range(n_y):
            for i in range(n_theta):
                a = j * (n_theta + 1) + i
                b = a + 1
                c = a + (n_theta + 1)
                d = c + 1
                tris.append([a, b, d])
                tris.append([a, d, c])
        return MeshGeometry("cylinder", np.asarray(verts, np.float32), np.asarray(norms, np.float32), np.asarray(uvs, np.float32), np.asarray(tris, np.int32), "parametric")

    if s == "curved_panel":
        n_theta, n_y = 20, 8
        radius, height, arc = 0.14, 0.14, math.radians(100.0)
        verts, norms, uvs, tris = [], [], [], []
        for j in range(n_y + 1):
            v_frac = j / n_y
            y = (v_frac - 0.5) * height
            for i in range(n_theta + 1):
                u_frac = i / n_theta
                theta = (u_frac - 0.5) * arc
                nx_v = math.sin(theta)
                nz_v = -math.cos(theta)
                verts.append([radius * nx_v, y, radius * (nz_v + 1.0)])
                norms.append([nx_v, 0.0, nz_v])
                uvs.append([u_frac, v_frac])
        for j in range(n_y):
            for i in range(n_theta):
                a = j * (n_theta + 1) + i
                b = a + 1
                c = a + (n_theta + 1)
                d = c + 1
                tris.append([a, b, d])
                tris.append([a, d, c])
        return MeshGeometry("curved_panel", np.asarray(verts, np.float32), np.asarray(norms, np.float32), np.asarray(uvs, np.float32), np.asarray(tris, np.int32), "parametric")

    if s == "box":
        hx, hy, hz = 0.095, 0.072, 0.062
        faces_def = [
            # (normal, u_axis, v_axis, center)
            ([0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, -hz]),
            ([0, 0, 1], [-1, 0, 0], [0, 1, 0], [0, 0, hz]),
            ([1, 0, 0], [0, 0, 1], [0, 1, 0], [hx, 0, 0]),
            ([-1, 0, 0], [0, 0, -1], [0, 1, 0], [-hx, 0, 0]),
            ([0, -1, 0], [1, 0, 0], [0, 0, 1], [0, -hy, 0]),
            ([0, 1, 0], [1, 0, 0], [0, 0, -1], [0, hy, 0]),
        ]
        sub = 3
        verts, norms, uvs, tris = [], [], [], []
        for nrm, uax, vax, ctr in faces_def:
            n_v = np.asarray(nrm, dtype=np.float32)
            u_v = np.asarray(uax, dtype=np.float32)
            v_v = np.asarray(vax, dtype=np.float32)
            c_v = np.asarray(ctr, dtype=np.float32)
            ext = np.array([hx, hy, hz], dtype=np.float32)
            u_len = float(np.abs(u_v) @ ext)
            v_len = float(np.abs(v_v) @ ext)
            base = len(verts)
            for j in range(sub + 1):
                vf = j / sub
                for i in range(sub + 1):
                    uf = i / sub
                    pos = c_v + (2.0 * uf - 1.0) * u_len * u_v + (2.0 * vf - 1.0) * v_len * v_v
                    verts.append(pos.tolist())
                    norms.append(n_v.tolist())
                    uvs.append([uf, vf])
            for j in range(sub):
                for i in range(sub):
                    a = base + j * (sub + 1) + i
                    b = a + 1
                    c = a + (sub + 1)
                    d = c + 1
                    tris.append([a, b, d])
                    tris.append([a, d, c])
        return MeshGeometry("box", np.asarray(verts, np.float32), np.asarray(norms, np.float32), np.asarray(uvs, np.float32), np.asarray(tris, np.int32), "parametric")

    raise ContractValidationError(
        f"Unsupported parametric mesh shape {shape!r}; expected one of plane, cylinder, box, curved_panel or a .ply path"
    )


def resolve_mesh(mesh_spec: str | Path | MeshGeometry | None) -> MeshGeometry:
    if isinstance(mesh_spec, MeshGeometry):
        return mesh_spec
    if mesh_spec is None:
        return build_parametric_mesh("box")
    s = str(mesh_spec).strip()
    if s.lower() in ("plane", "cylinder", "box", "curved_panel"):
        return build_parametric_mesh(s)
    return load_ply_mesh(s)


def _rot_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _spatial_bucket_points(
    pts: np.ndarray,
    scores: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    grid_cols: int = 8,
    grid_rows: int = 6,
    max_total: int = 180,
) -> np.ndarray:
    """Fixed-budget 8x6 spatial cell bucketing preserving spatial diversity at exact same feature cap."""
    if len(pts) <= max_total:
        return np.arange(len(pts), dtype=np.int64)
    bx, by, bw, bh = bbox
    bw = max(bw, 1.0)
    bh = max(bh, 1.0)
    per_cell = max(1, max_total // (grid_cols * grid_rows))
    order = np.argsort(-scores)
    buckets: dict[tuple[int, int], list[int]] = {}
    selected: list[int] = []
    remaining: list[int] = []
    for idx in order:
        cx = int(np.clip(((pts[idx, 0] - bx) / bw) * grid_cols, 0, grid_cols - 1))
        cy = int(np.clip(((pts[idx, 1] - by) / bh) * grid_rows, 0, grid_rows - 1))
        cell = buckets.setdefault((cx, cy), [])
        if len(cell) < per_cell and len(selected) < max_total:
            cell.append(int(idx))
            selected.append(int(idx))
        else:
            remaining.append(int(idx))
    if len(selected) < max_total:
        selected.extend(remaining[: max_total - len(selected)])
    return np.asarray(selected[:max_total], dtype=np.int64)


def _extract_patch_thumb(gray: np.ndarray, quad: np.ndarray, size: int = 32) -> np.ndarray | None:
    """Warp the quad into a tiny 32x32 normalized grayscale thumbnail for fast occlusion checks."""
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    try:
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        patch = cv2.warpPerspective(gray, M, (size, size), flags=cv2.INTER_LINEAR)
    except cv2.error:
        return None
    return patch


def _patch_ncc(ref_patch: np.ndarray, cur_patch: np.ndarray) -> float:
    a = ref_patch.astype(np.float32).ravel()
    b = cur_patch.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-3 or nb < 1e-3:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _patch_block_ncc_p75(ref_patch: np.ndarray, cur_patch: np.ndarray, grid: int = 4) -> float:
    """75th-percentile local NCC over a 4x4 grid of 8x8 blocks (robust to partial hand occlusion & local shadow)."""
    bs = ref_patch.shape[0] // grid
    a = ref_patch.reshape(grid, bs, grid, bs).transpose(0, 2, 1, 3).reshape(grid * grid, bs * bs).astype(np.float32)
    b = cur_patch.reshape(grid, bs, grid, bs).transpose(0, 2, 1, 3).reshape(grid * grid, bs * bs).astype(np.float32)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    valid = (na > 8.0) & (nb > 8.0)
    if not np.any(valid):
        return 0.0
    corrs = np.sum(a[valid] * b[valid], axis=1) / (na[valid] * nb[valid])
    return float(np.percentile(corrs, 75))


def _patch_grad_energy(patch: np.ndarray) -> float:
    arr = patch.astype(np.float32)
    gx = np.abs(arr[:, 1:] - arr[:, :-1]).mean()
    gy = np.abs(arr[1:, :] - arr[:-1, :]).mean()
    return float(gx + gy)


def _is_valid_quad(
    quad: np.ndarray,
    ref_area: float,
    frame_w: int,
    frame_h: int,
) -> bool:
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        return False
    # Convexity check via signed cross products of consecutive edges
    edges = np.roll(quad, -1, axis=0) - quad
    crosses = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if not (np.all(crosses > 0) or np.all(crosses < 0)):
        return False
    area = float(cv2.contourArea(quad.astype(np.float32)))
    if area < max(100.0, 0.12 * ref_area) or area > max(4.5 * ref_area, float(frame_w * frame_h) * 2.5):
        return False
    # Must overlap image frame meaningfully
    x_min, y_min = float(quad[:, 0].min()), float(quad[:, 1].min())
    x_max, y_max = float(quad[:, 0].max()), float(quad[:, 1].max())
    if x_max < frame_w * 0.05 or x_min > frame_w * 0.95 or y_max < frame_h * 0.05 or y_min > frame_h * 0.95:
        return False
    # Check aspect sanity (no extreme collapsed sliver)
    side_lens = np.linalg.norm(edges, axis=1)
    if float(side_lens.min()) < 8.0 or float(side_lens.max()) / max(float(side_lens.min()), 1.0) > 12.0:
        return False
    return True


def _estimate_initial_pose_from_mask(
    mask_u8: np.ndarray,
    mesh: MeshGeometry,
    K: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Estimate top-K plausible initial 6-DoF poses purely from mask silhouette + K + mesh (zero GT).

    Returns list of `(rvec, tvec, iou)` sorted descending by silhouette IoU.
    """
    h, w = mask_u8.shape[:2]
    ys, xs = np.nonzero(mask_u8)
    if len(xs) < 16:
        return [(np.zeros(3, dtype=np.float64), np.array([0.0, 0.0, 0.5], dtype=np.float64), 0.0)]

    bx0, by0 = float(xs.min()), float(ys.min())
    bw = max(8.0, float(xs.max() - bx0 + 1.0))
    bh = max(8.0, float(ys.max() - by0 + 1.0))
    mcx = float(xs.mean())
    mcy = float(ys.mean())
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    # Downscale mask to <= 160 px wide for fast (<15 ms) silhouette search
    scale = min(1.0, 160.0 / float(max(w, h)))
    sw = max(32, int(round(w * scale)))
    sh = max(24, int(round(h * scale)))
    small_mask = cv2.resize(mask_u8, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    small_area = float(np.count_nonzero(small_mask))
    sfx, sfy, scx, scy = fx * scale, fy * scale, cx * scale, cy * scale

    verts = mesh.vertices.astype(np.float64)
    # Use unique hull / extreme vertices for fast 2D convex hull silhouette projection
    if len(verts) > 64:
        step = max(1, len(verts) // 64)
        sample_v = verts[::step]
    else:
        sample_v = verts

    extents = np.ptp(verts, axis=0)
    diag = float(np.linalg.norm(extents))
    # Initial depth estimate from apparent bbox diagonal vs 3D diameter
    tz0 = max(0.12, fx * max(float(extents[0]), float(extents[1]), diag * 0.72) / max(bw, bh))
    tx0 = (mcx - cx) * tz0 / fx
    ty0 = (mcy - cy) * tz0 / fy

    def _eval_pose(rx: float, ry: float, rz: float, tx: float, ty: float, tz: float) -> float:
        R = _rot_xyz(rx, ry, rz)
        pc = sample_v @ R.T + np.array([tx, ty, tz], dtype=np.float64)
        if np.any(pc[:, 2] <= 0.03):
            return 0.0
        u = pc[:, 0] / pc[:, 2] * sfx + scx
        v = pc[:, 1] / pc[:, 2] * sfy + scy
        pts2d = np.column_stack([u, v]).astype(np.float32)
        hull = cv2.convexHull(pts2d)
        proj = np.zeros((sh, sw), dtype=np.uint8)
        cv2.fillConvexPoly(proj, np.round(hull).astype(np.int32), 1)
        pbool = proj > 0
        inter = float(np.count_nonzero(pbool & small_mask))
        if inter <= 0.0:
            return 0.0
        union = small_area + float(np.count_nonzero(pbool)) - inter
        return inter / max(union, 1.0)

    # Coarse grid over (rx, ry, tz_scale)
    candidates: list[tuple[float, float, float, float, float, float, float]] = []
    for rx_deg in (-35.0, -18.0, 0.0, 18.0, 35.0):
        for ry_deg in (-40.0, -22.0, 0.0, 22.0, 40.0):
            for tz_mult in (0.82, 0.95, 1.08, 1.22):
                tz = tz0 * tz_mult
                tx = (mcx - cx) * tz / fx
                ty = (mcy - cy) * tz / fy
                iou = _eval_pose(math.radians(rx_deg), math.radians(ry_deg), 0.0, tx, ty, tz)
                candidates.append((iou, math.radians(rx_deg), math.radians(ry_deg), 0.0, tx, ty, tz))

    candidates.sort(key=lambda x: x[0], reverse=True)
    # Keep top distinct angle basins and refine each with coordinate ascent
    refined: list[tuple[np.ndarray, np.ndarray, float]] = []
    used_basins: list[tuple[float, float]] = []
    for best_iou, rx, ry, rz, tx, ty, tz in candidates:
        if any(abs(rx - urx) < 0.25 and abs(ry - ury) < 0.25 for urx, ury in used_basins):
            continue
        used_basins.append((rx, ry))
        # Fine coordinate refinement (2 rounds)
        for step_ang, step_tz, step_xy in ((0.12, 0.04, 0.012), (0.04, 0.015, 0.004)):
            improved = True
            iters = 0
            while improved and iters < 6:
                improved = False
                iters += 1
                for drx, dry, drz, dtx, dty, dtz in (
                    (step_ang, 0, 0, 0, 0, 0), (-step_ang, 0, 0, 0, 0, 0),
                    (0, step_ang, 0, 0, 0, 0), (0, -step_ang, 0, 0, 0, 0),
                    (0, 0, step_ang, 0, 0, 0), (0, 0, -step_ang, 0, 0, 0),
                    (0, 0, 0, 0, 0, step_tz), (0, 0, 0, 0, 0, -step_tz),
                    (0, 0, 0, step_xy, 0, 0), (0, 0, 0, -step_xy, 0, 0),
                    (0, 0, 0, 0, step_xy, 0), (0, 0, 0, 0, -step_xy, 0),
                ):
                    nt = max(0.08, tz + dtz)
                    sc = _eval_pose(rx + drx, ry + dry, rz + drz, tx + dtx, ty + dty, nt)
                    if sc > best_iou + 1e-4:
                        best_iou = sc
                        rx, ry, rz = rx + drx, ry + dry, rz + drz
                        tx, ty, tz = tx + dtx, ty + dty, nt
                        improved = True
        R = _rot_xyz(rx, ry, rz)
        rvec, _ = cv2.Rodrigues(R)
        tvec = np.array([tx, ty, tz], dtype=np.float64)
        refined.append((rvec.reshape(3), tvec, best_iou))
        if len(refined) >= 3:
            break

    refined.sort(key=lambda x: x[2], reverse=True)
    return refined


def _extract_oriented_mask_frame(mask_u8: np.ndarray) -> tuple[float, float, float, float, float] | None:
    """Return (cx, cy, eff_w, eff_h, axis_rad) oriented bounding frame for a binary mask."""
    ys, xs = np.nonzero(mask_u8 > 127)
    if len(xs) < 48:
        return None
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    (cx, cy), (rw, rh), ang_deg = cv2.minAreaRect(pts)
    if rw < 4.0 or rh < 4.0:
        return None
    if rw > rh:
        eff_w, eff_h = float(rh), float(rw)
        axis_deg = float(ang_deg) + 90.0
    else:
        eff_w, eff_h = float(rw), float(rh)
        axis_deg = float(ang_deg)
    while axis_deg > 90.0:
        axis_deg -= 180.0
    while axis_deg < -90.0:
        axis_deg += 180.0
    return (float(cx), float(cy), eff_w, eff_h, math.radians(axis_deg))


class VideoSurfaceTracker:
    """Causal per-frame KLT + ORB/XFeat tracker for planar, cylinder, mesh_grid, and mesh_6dof."""

    def __init__(
        self,
        *,
        wrap_mode: Literal["planar", "cylinder", "mesh_grid", "mesh_6dof"] = "planar",
        tracking_height: int = 360,
        max_klt_points: int = 180,
        orb_features: int = 600,
        spatial_bucketing: bool = True,
        adaptive_appearance_gate: bool = True,
        grazing_angle_gate: bool = False,
        mesh: MeshGeometry | None = None,
        intrinsics: tuple[float, float, float, float] | None = None,
        init_pose: tuple[np.ndarray, np.ndarray] | None = None,
        matcher_backend: Literal["orb", "xfeat"] = "orb",
    ) -> None:
        if tracking_height not in (360, 480) and not (120 <= tracking_height <= 1080):
            raise ContractValidationError(f"Invalid tracking_height: {tracking_height}")
        if matcher_backend not in ("orb", "xfeat"):
            raise ContractValidationError(f"Invalid matcher_backend: {matcher_backend!r}")
        self.wrap_mode = wrap_mode
        self.tracking_height = int(tracking_height)
        self.max_klt_points = int(max_klt_points)
        self.orb_features = int(orb_features)
        self.spatial_bucketing = bool(spatial_bucketing)
        self.adaptive_appearance_gate = bool(adaptive_appearance_gate)
        self.grazing_angle_gate = bool(grazing_angle_gate)
        self.mesh = mesh
        self.intrinsics = intrinsics
        self.init_pose_control = init_pose
        self.init_pose_source: str = "known_pose_control" if init_pose is not None else "mask_silhouette_fit"
        self.matcher_backend: Literal["orb", "xfeat"] = matcher_backend
        self._xfeat: Any = None

        self._initialized = False
        self._frame_w = 0
        self._frame_h = 0
        self._track_w = 0
        self._track_h = 0
        self._sx = 1.0  # native -> track scale
        self._sy = 1.0

        self._orb = cv2.ORB_create(nfeatures=self.orb_features, nlevels=6, scaleFactor=1.2)
        self._orb_recovery = cv2.ORB_create(
            nfeatures=2000, nlevels=8, scaleFactor=1.2, edgeThreshold=15, patchSize=15, fastThreshold=8
        )
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Shared state
        self._state: Literal["tracking", "recovering", "lost"] = "lost"
        self._prev_gray: np.ndarray | None = None
        self._ref_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._ref_quad: np.ndarray = np.zeros((4, 2), dtype=np.float32)
        self._ref_area: float = 1.0
        self._ref_mask: np.ndarray | None = None
        self._ref_patch: np.ndarray | None = None
        self._prev_patch: np.ndarray | None = None
        self._last_block_ncc: float = 1.0
        self._ref_grad_energy: float = 0.0
        self._ref_kps_native: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self._ref_descs: np.ndarray | None = None
        self._ref_keyframes: list[tuple[np.ndarray, np.ndarray]] = []
        self._xfeat_ref_keyframes: list[tuple[np.ndarray, np.ndarray]] = []
        self._use_native_recovery: bool = False
        self._frames_since_kf: int = 0
        self._init_vos_area: float = 0.0
        self._init_vos_frame: tuple[float, float, float, float, float] | None = None
        self._last_cyl_roll_rad: float = 0.0
        self._last_cyl_scale: float = 1.0

        # Planar/cylinder/grid state
        self._curr_pts_track: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self._curr_ref_pts_native: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self._last_H: np.ndarray | None = None
        self._cand_H: np.ndarray | None = None
        self._grid_src: np.ndarray | None = None
        self._lost_streak = 0
        self._frames_since_replenish = 0

        # 6-DoF mesh state
        self._K: np.ndarray | None = None
        self._mesh_pts3d_bank: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._mesh_descs_bank: np.ndarray | None = None
        self._mesh_keyframes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._xfeat_mesh_keyframes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._init_tz: float = 0.5
        self._last_occluder_mask_img: np.ndarray | None = None
        self._curr_pts3d: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._last_rvec: np.ndarray | None = None
        self._last_tvec: np.ndarray | None = None
        self._last_valid_rvec: np.ndarray | None = None
        self._last_valid_tvec: np.ndarray | None = None
        self._last_ang_vel: np.ndarray = np.zeros(3, dtype=np.float64)
        self._vos_coast_frames: int = 0
        self._cand_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._alt_hypotheses: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._reproj_errors: list[float] = []
        self._last_cyl_H_ema: np.ndarray | None = None

    def set_occluder_mask(self, mask_img: np.ndarray | None) -> None:
        """Receive detected foreground/hand occluder mask (native resolution uint8) to exclude from tracking memory."""
        self._last_occluder_mask_img = mask_img

    def _ensure_xfeat(self) -> Any:
        if self._xfeat is None and self.matcher_backend == "xfeat":
            from .matcher_xfeat import XFeatMatcher

            self._xfeat = XFeatMatcher()
        return self._xfeat

    def _erode_vos_mask(self, mask_u8: np.ndarray) -> np.ndarray:
        """Erode VOS foreground mask slightly so tracked keypoints never sit on finger/background edges."""
        fg_px = int(np.count_nonzero(mask_u8))
        if fg_px > 900:
            eroded = cv2.erode(mask_u8, np.ones((5, 5), np.uint8), iterations=1)
            if int(np.count_nonzero(eroded)) >= 180:
                return eroded
        if fg_px > 200:
            eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
            if int(np.count_nonzero(eroded)) >= 64:
                return eroded
        return mask_u8

    def _mask_bbox_roi(
        self,
        mask_u8: np.ndarray | None,
        *,
        pad_frac: float = 0.35,
        min_dim: int = 128,
    ) -> tuple[int, int, int, int] | None:
        if mask_u8 is None:
            return None
        ys, xs = np.nonzero(mask_u8 > 127)
        if len(xs) < 16:
            return None
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bw, bh = max(8, x1 - x0), max(8, y1 - y0)
        px = max(int(round(bw * pad_frac)), max(0, (min_dim - bw) // 2))
        py = max(int(round(bh * pad_frac)), max(0, (min_dim - bh) // 2))
        return (
            max(0, x0 - px),
            max(0, y0 - py),
            min(self._frame_w, x1 + px),
            min(self._frame_h, y1 + py),
        )

    def _match_xfeat_2d(
        self,
        frame_bgr: np.ndarray,
        mask_u8: np.ndarray | None = None,
        *,
        roi: tuple[int, int, int, int] | None = None,
        min_cossim: float = 0.78,
        top_k: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match current frame XFeat descriptors against stored 2D reference keyframes."""
        if self.matcher_backend != "xfeat" or not self._xfeat_ref_keyframes:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        xf = self._ensure_xfeat()
        if roi is None and mask_u8 is not None:
            roi = self._mask_bbox_roi(mask_u8)
        cur_pts, cur_descs = xf.extract(frame_bgr, mask_u8, roi=roi, top_k=top_k)
        if len(cur_pts) < 6:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        best_ref = np.zeros((0, 2), dtype=np.float32)
        best_cur = np.zeros((0, 2), dtype=np.float32)
        for kf_pts, kf_descs in self._xfeat_ref_keyframes:
            m_ref, m_cur = xf.match(kf_pts, kf_descs, cur_pts, cur_descs, min_cossim=min_cossim)
            if len(m_ref) > len(best_ref):
                best_ref, best_cur = m_ref, m_cur
        return best_ref, best_cur

    def _maybe_store_xfeat_keyframe_2d(
        self,
        frame_bgr: np.ndarray,
        H: np.ndarray,
        mask_u8: np.ndarray | None = None,
    ) -> None:
        if self.matcher_backend != "xfeat" or len(self._xfeat_ref_keyframes) >= 6 or self._ref_mask is None:
            return
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return
        xf = self._ensure_xfeat()
        if mask_u8 is None:
            warped_quad = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), H).reshape(4, 2)
            mask_use = np.zeros((self._frame_h, self._frame_w), dtype=np.uint8)
            cv2.fillConvexPoly(mask_use, np.round(warped_quad).astype(np.int32), 255)
        else:
            mask_use = mask_u8
        roi = self._mask_bbox_roi(mask_use)
        pts, descs = xf.extract(frame_bgr, mask_use, roi=roi, top_k=512)
        if len(pts) < 16:
            return
        back_ref = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        iu = np.clip(np.round(back_ref[:, 0]).astype(np.int32), 0, self._frame_w - 1)
        iv = np.clip(np.round(back_ref[:, 1]).astype(np.int32), 0, self._frame_h - 1)
        inside = (
            (back_ref[:, 0] >= 0) & (back_ref[:, 0] < self._frame_w)
            & (back_ref[:, 1] >= 0) & (back_ref[:, 1] < self._frame_h)
            & (self._ref_mask[iv, iu] > 0)
        )
        if int(np.count_nonzero(inside)) >= 16:
            self._xfeat_ref_keyframes.append((back_ref[inside].astype(np.float32), descs[inside].copy()))

    def _match_xfeat_6dof(
        self,
        frame_bgr: np.ndarray,
        mask_u8: np.ndarray | None = None,
        *,
        roi: tuple[int, int, int, int] | None = None,
        min_cossim: float = 0.76,
        top_k: int = 640,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match current frame XFeat descriptors against verified 3D mesh keyframes."""
        if self.matcher_backend != "xfeat" or not self._xfeat_mesh_keyframes:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.float32)
        xf = self._ensure_xfeat()
        if roi is None and mask_u8 is not None:
            roi = self._mask_bbox_roi(mask_u8)
        cur_pts, cur_descs = xf.extract(frame_bgr, mask_u8, roi=roi, top_k=top_k)
        if len(cur_pts) < 6:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.float32)
        best_3d = np.zeros((0, 3), dtype=np.float64)
        best_2d = np.zeros((0, 2), dtype=np.float32)
        for _, _, kf_pts3d, kf_descs in self._xfeat_mesh_keyframes:
            idx0, idx1 = xf.match_indices(kf_descs, cur_descs, min_cossim=min_cossim)
            if len(idx0) > len(best_3d):
                best_3d = kf_pts3d[idx0].copy()
                best_2d = cur_pts[idx1].copy()
        return best_3d, best_2d

    def _try_xfeat_kf0_loop_closure(
        self,
        frame_bgr: np.ndarray,
        mask_u8: np.ndarray | None,
        vos_mask: np.ndarray | None,
        res_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int] | None:
        """Independently solve PnP against canonical frame-0 XFeat keyframe to eliminate 360-deg rotational drift."""
        if self.matcher_backend != "xfeat" or not self._xfeat_mesh_keyframes or self._K is None:
            return None
        xf = self._ensure_xfeat()
        roi = self._mask_bbox_roi(vos_mask if vos_mask is not None else mask_u8)
        cur_pts, cur_descs = xf.extract(frame_bgr, mask_u8, roi=roi, top_k=640)
        if len(cur_pts) < 10:
            return None
        # Prioritize canonical frame-0 keyframe (index 0, zero accumulated drift!)
        _, _, kf0_pts3d, kf0_descs = self._xfeat_mesh_keyframes[0]
        idx0, idx1 = xf.match_indices(kf0_descs, cur_descs, min_cossim=0.72)
        if len(idx0) < 10:
            return None
        pts3d = kf0_pts3d[idx0]
        pts2d = cur_pts[idx1]
        ok, rvec, tvec, inl_bool, rmse = self._solve_pnp_robust(pts3d, pts2d, None)
        if not ok or rvec is None or tvec is None:
            return None
        n_inl = int(np.count_nonzero(inl_bool))
        if n_inl < 18 or rmse > 3.2 * res_scale:
            return None
        if not (max(0.06, 0.55 * self._init_tz) <= float(tvec[2]) <= 1.65 * self._init_tz):
            return None
        # Never allow noisy cap/rim matches on a featureless backside to 180-deg flip the front face onto the back!
        if self._last_rvec is not None:
            R_kf, _ = cv2.Rodrigues(rvec)
            R_prev, _ = cv2.Rodrigues(self._last_rvec)
            d_ang_kf = math.degrees(math.acos(float(np.clip((np.trace(R_kf @ R_prev.T) - 1.0) * 0.5, -1.0, 1.0))))
            if d_ang_kf > 65.0 and not ((self._vos_coast_frames >= 20 and n_inl >= 24) or n_inl >= 36):
                return None
        if vos_mask is not None:
            ys_v, xs_v = np.nonzero(vos_mask > 127)
            if len(xs_v) >= 32:
                vos_w = max(12.0, float(xs_v.max() - xs_v.min()))
                vos_h = max(12.0, float(ys_v.max() - ys_v.min()))
                inl_pts = pts2d[inl_bool]
                if (
                    float(inl_pts[:, 0].max() - inl_pts[:, 0].min()) < 0.34 * vos_w
                    or float(inl_pts[:, 1].max() - inl_pts[:, 1].min()) < 0.30 * vos_h
                ):
                    return None
            v_iou, v_cov, _ = self._eval_6dof_vos_iou(rvec, tvec, vos_mask)
            if v_iou < 0.62 or v_cov < 0.65:
                return None
        return rvec, tvec, pts3d[inl_bool], pts2d[inl_bool], rmse, n_inl

    def _maybe_store_xfeat_keyframe_6dof(
        self,
        frame_bgr: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        depth_buf: np.ndarray,
        mask_u8: np.ndarray | None = None,
    ) -> None:
        if self.matcher_backend != "xfeat" or len(self._xfeat_mesh_keyframes) >= 16 or self._K is None:
            return
        R_cur, _ = cv2.Rodrigues(rvec.astype(np.float64))
        for kf_r, _, _, _ in self._xfeat_mesh_keyframes:
            R_kf, _ = cv2.Rodrigues(kf_r.astype(np.float64))
            ang = math.acos(float(np.clip((np.trace(R_cur @ R_kf.T) - 1.0) * 0.5, -1.0, 1.0)))
            if math.degrees(ang) < 18.0:
                return
        sil_u8 = (depth_buf < 1e20).astype(np.uint8) * 255
        if mask_u8 is not None and mask_u8.shape == sil_u8.shape:
            sil_u8 = cv2.bitwise_and(sil_u8, self._erode_vos_mask(mask_u8))
        else:
            sil_u8 = cv2.erode(sil_u8, np.ones((5, 5), np.uint8), iterations=1)
        if int(np.count_nonzero(sil_u8)) < 120:
            return
        xf = self._ensure_xfeat()
        roi = self._mask_bbox_roi(sil_u8)
        pts_nat, descs = xf.extract(frame_bgr, sil_u8, roi=roi, top_k=512, erode_mask=False)
        if len(pts_nat) < 12:
            return
        fx, fy, cx, cy = float(self._K[0, 0]), float(self._K[1, 1]), float(self._K[0, 2]), float(self._K[1, 2])
        iu = np.clip(np.round(pts_nat[:, 0]).astype(np.int32), 0, self._frame_w - 1)
        iv = np.clip(np.round(pts_nat[:, 1]).astype(np.int32), 0, self._frame_h - 1)
        z = depth_buf[iv, iu].astype(np.float64)
        valid = np.isfinite(z) & (z > 0.03) & (z < 100.0)
        if int(np.count_nonzero(valid)) < 12:
            return
        uv_ok = pts_nat[valid].astype(np.float64)
        z_ok = z[valid]
        p_cam = np.column_stack([(uv_ok[:, 0] - cx) * z_ok / fx, (uv_ok[:, 1] - cy) * z_ok / fy, z_ok])
        p_obj = (p_cam - tvec.reshape(1, 3)) @ R_cur
        self._xfeat_mesh_keyframes.append((rvec.copy(), tvec.copy(), p_obj[:512], descs[valid][:512].copy()))

    def _eval_6dof_vos_iou_fast(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        vos_small_b: np.ndarray,
        scale: float,
    ) -> tuple[float, float, tuple[float, float, float, float] | None]:
        """Ultra-fast (<=320px) reusable-buffer 6-DoF mesh vs VOS IoU (~0.35ms)."""
        assert self.mesh is not None and self._K is not None
        sh, sw = vos_small_b.shape[:2]
        if (
            getattr(self, "_vos_iou_rgb", None) is None
            or self._vos_iou_rgb.shape[0] != sh
            or self._vos_iou_rgb.shape[1] != sw
        ):
            self._vos_iou_rgb = np.zeros((sh, sw, 3), dtype=np.uint8)
            self._vos_iou_depth = np.full((sh, sw), 1e30, dtype=np.float32)
            self._vos_iou_mask = np.zeros((sh, sw), dtype=np.uint8)
            self._vos_iou_tex = np.full((4, 4, 4), 255, dtype=np.uint8)
        self._vos_iou_depth.fill(1e30)
        self._vos_iou_mask.fill(0)

        R, _ = cv2.Rodrigues(rvec.astype(np.float64))
        verts_cam = (self.mesh.vertices.astype(np.float64) @ R.T + tvec.reshape(1, 3)).astype(np.float32)
        fx = float(self._K[0, 0]) * scale
        fy = float(self._K[1, 1]) * scale
        cx = float(self._K[0, 2]) * scale
        cy = float(self._K[1, 2]) * scale
        res = rasterize_mesh_rgbd(
            verts_cam,
            self.mesh.triangles,
            self.mesh.uvs,
            self._vos_iou_tex,
            (fx, fy, cx, cy),
            self._vos_iou_rgb,
            depth_buf=self._vos_iou_depth,
            mask_out=self._vos_iou_mask,
            opacity=0.0,
            blend_mode="normal",
        )
        proj_b = res.depth_buf < 1e20
        n_vos = int(np.count_nonzero(vos_small_b))
        n_proj = int(np.count_nonzero(proj_b))
        if n_vos == 0 or n_proj == 0:
            return 0.0, 0.0, None
        inter = int(np.count_nonzero(proj_b & vos_small_b))
        union = n_vos + n_proj - inter
        ys_p, xs_p = np.nonzero(proj_b)
        inv_s = 1.0 / max(scale, 1e-4)
        pbbox = (
            float(xs_p.min()) * inv_s,
            float(ys_p.min()) * inv_s,
            float(xs_p.max()) * inv_s,
            float(ys_p.max()) * inv_s,
        )
        return float(inter) / float(max(union, 1)), float(inter) / float(max(n_vos, 1)), pbbox

    def _rasterize_full_depth(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Rasterize full-resolution 3D mesh depth buffer using cached RGB buffer."""
        assert self.mesh is not None and self._K is not None
        if (
            getattr(self, "_full_depth_rgb", None) is None
            or self._full_depth_rgb.shape[0] != self._frame_h
            or self._full_depth_rgb.shape[1] != self._frame_w
        ):
            self._full_depth_rgb = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
            self._full_depth_tex = np.full((4, 4, 4), 255, dtype=np.uint8)
        R, _ = cv2.Rodrigues(rvec.astype(np.float64))
        verts_cam = (self.mesh.vertices.astype(np.float64) @ R.T + tvec.reshape(1, 3)).astype(np.float32)
        fx, fy, cx, cy = float(self._K[0, 0]), float(self._K[1, 1]), float(self._K[0, 2]), float(self._K[1, 2])
        res = rasterize_mesh_rgbd(
            verts_cam,
            self.mesh.triangles,
            self.mesh.uvs,
            self._full_depth_tex,
            (fx, fy, cx, cy),
            self._full_depth_rgb,
            opacity=0.0,
            blend_mode="normal",
        )
        return res.depth_buf

    def _eval_6dof_vos_iou(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        vos_mask: np.ndarray,
        *,
        need_depth: bool = False,
    ) -> tuple[float, float, np.ndarray | None]:
        """Compute (IoU, fraction_of_vos_covered, optional full-res depth_buf) via fast low-res rasterization."""
        scale = min(1.0, 320.0 / float(max(self._frame_w, self._frame_h, 1)))
        if scale < 0.99:
            sw = max(32, int(round(self._frame_w * scale)))
            sh = max(24, int(round(self._frame_h * scale)))
            vos_s = cv2.resize(vos_mask, (sw, sh), interpolation=cv2.INTER_NEAREST) > 127
        else:
            vos_s = vos_mask > 127
        iou, cov, _ = self._eval_6dof_vos_iou_fast(rvec, tvec, vos_s, scale)
        db = self._rasterize_full_depth(rvec, tvec) if need_depth else None
        return iou, cov, db

    def _refine_6dof_pose_to_vos(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        depth_buf: np.ndarray | None,
        vos_iou: float,
        vos_cov: float,
        vos_mask: np.ndarray,
        pts3d_inl: np.ndarray,
        pts2d_native_inl: np.ndarray,
        base_rmse: float,
        res_scale: float,
        *,
        coasting: bool = False,
    ) -> tuple[np.ndarray, float, float, np.ndarray | None, float]:
        """Align (tx, ty, tz) cleanly to the live VOS foreground silhouette (~1ms via low-res buffers)."""
        assert self._K is not None
        scale = min(1.0, 320.0 / float(max(self._frame_w, self._frame_h, 1)))
        if scale < 0.99:
            sw = max(32, int(round(self._frame_w * scale)))
            sh = max(24, int(round(self._frame_h * scale)))
            vos_s = cv2.resize(vos_mask, (sw, sh), interpolation=cv2.INTER_NEAREST) > 127
        else:
            vos_s = vos_mask > 127

        cur_iou, cur_cov, pbbox = self._eval_6dof_vos_iou_fast(rvec, tvec, vos_s, scale)
        ys_v, xs_v = np.nonzero(vos_s)
        if pbbox is None or len(xs_v) < 12:
            return tvec, cur_iou, cur_cov, depth_buf, base_rmse

        inv_s = 1.0 / max(scale, 1e-4)
        fx, fy = float(self._K[0, 0]), float(self._K[1, 1])
        px0, py0, px1, py1 = pbbox
        pcx = 0.5 * (px0 + px1)
        pcy = 0.5 * (py0 + py1)
        vx0 = float(np.percentile(xs_v, 1.0)) * inv_s
        vx1 = float(np.percentile(xs_v, 99.0)) * inv_s
        vy0 = float(np.percentile(ys_v, 1.0)) * inv_s
        vy1 = float(np.percentile(ys_v, 99.0)) * inv_s
        vcx = 0.5 * (vx0 + vx1)
        vcy = 0.5 * (vy0 + vy1)
        dx = vcx - pcx
        dy = vcy - pcy

        pw = max(8.0, px1 - px0)
        ph = max(8.0, py1 - py0)
        vw = max(8.0, vx1 - vx0)
        vh = max(8.0, vy1 - vy0)

        # Only adjust scale when VOS mask isn't heavily occluded/truncated relative to frame-0 area
        n_vos_nat = float(len(xs_v)) * (inv_s * inv_s)
        allow_scale = (self._init_vos_area > 0.0) and (n_vos_nat >= 0.72 * self._init_vos_area)
        scale_ratio = float(np.clip(max(vw / pw, vh / ph), 0.85, 1.18)) if allow_scale else 1.0
        # When partially occluded, still pull back inflated silhouettes that grew way bigger than VOS!
        if not allow_scale and (pw > vw * 1.15 and ph > vh * 1.15):
            scale_ratio = float(np.clip(max(vw / pw, vh / ph), 0.85, 1.0))

        best_t = tvec.copy()
        best_iou = cur_iou
        best_cov = cur_cov
        best_rmse = base_rmse

        step_fracs = (0.92, 0.65, 0.35) if coasting else (0.85, 0.55, 0.30)
        for frac in step_fracs:
            tz_old = float(best_t[2])
            s_step = 1.0 + frac * (scale_ratio - 1.0)
            tz_new = float(np.clip(tz_old / max(s_step, 0.5), max(0.06, 0.55 * self._init_tz), 1.65 * self._init_tz))
            t_cand = best_t.copy()
            t_cand[0] = (best_t[0] + (dx * frac) * tz_old / fx) * (tz_new / tz_old)
            t_cand[1] = (best_t[1] + (dy * frac) * tz_old / fy) * (tz_new / tz_old)
            t_cand[2] = tz_new

            if not coasting and len(pts3d_inl) >= 8:
                proj_c, _ = cv2.projectPoints(pts3d_inl, rvec, t_cand, self._K, None)
                errs = np.linalg.norm(proj_c.reshape(-1, 2) - pts2d_native_inl, axis=1)
                rmse_c = float(math.sqrt(float(np.mean(errs ** 2))))
                rmse_lim = (
                    max(base_rmse + 4.0 * res_scale, 8.5 * res_scale)
                    if cur_iou < 0.78
                    else max(base_rmse + 1.2 * res_scale, 4.0 * res_scale)
                )
            else:
                rmse_c = base_rmse
                rmse_lim = 999.0

            c_iou, c_cov, _ = self._eval_6dof_vos_iou_fast(rvec, t_cand, vos_s, scale)
            allow_snap = (rmse_c <= rmse_lim) or (cur_iou < 0.65 and c_iou >= cur_iou + 0.08)
            if allow_snap and (c_iou > best_iou + 0.012 or (coasting and (c_cov > best_cov + 0.015 or c_iou > best_iou + 0.004))):
                best_t = t_cand
                best_iou = c_iou
                best_cov = c_cov
                best_rmse = rmse_c
                break

        return best_t, best_iou, best_cov, None, best_rmse

    def _refine_6dof_cylinder_rot_to_vos(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        vos_iou: float,
        vos_cov: float,
        vos_mask: np.ndarray,
        pts3d_inl: np.ndarray,
        pts2d_native_inl: np.ndarray,
        base_rmse: float,
        res_scale: float,
        *,
        coasting: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """Fast (~2ms) 5-DoF silhouette-contour roll+pitch+translation alignment for parametric cylinders.

        Preserves cylinder spin yaw while snapping drifted in-plane roll and out-of-plane pitch
        (e.g. featureless backside pitch collapse on bottles or tilted soup cans) directly to the
        live VOS foreground silhouette.
        """
        assert self._K is not None
        if (
            self.mesh is None
            or self.mesh.source_kind != "parametric"
            or self.mesh.name != "cylinder"
            or vos_iou >= 0.86
        ):
            return rvec, tvec, vos_iou, vos_cov, base_rmse

        cur_f = _extract_oriented_mask_frame(vos_mask)
        if cur_f is None:
            return rvec, tvec, vos_iou, vos_cov, base_rmse
        vcx, vcy, vw, vh, v_roll = cur_f

        scale = min(1.0, 320.0 / float(max(self._frame_w, self._frame_h, 1)))
        if scale < 0.99:
            sw = max(32, int(round(self._frame_w * scale)))
            sh = max(24, int(round(self._frame_h * scale)))
            vos_s = cv2.resize(vos_mask, (sw, sh), interpolation=cv2.INTER_NEAREST) > 127
        else:
            vos_s = vos_mask > 127
        ys_v, xs_v = np.nonzero(vos_s)
        if len(xs_v) < 16:
            return rvec, tvec, vos_iou, vos_cov, base_rmse

        inv_s = 1.0 / max(scale, 1e-4)
        vx0 = float(np.percentile(xs_v, 1.0)) * inv_s
        vx1 = float(np.percentile(xs_v, 99.0)) * inv_s
        vy0 = float(np.percentile(ys_v, 1.0)) * inv_s
        vy1 = float(np.percentile(ys_v, 99.0)) * inv_s
        v_bbox_cx = 0.5 * (vx0 + vx1)
        v_bbox_cy = 0.5 * (vy0 + vy1)
        v_bbox_w = max(8.0, vx1 - vx0)
        v_bbox_h = max(8.0, vy1 - vy0)
        n_vos_nat = float(len(xs_v)) * (inv_s * inv_s)
        allow_scale = (self._init_vos_area > 0.0) and (n_vos_nat >= 0.65 * self._init_vos_area)
        fx, fy = float(self._K[0, 0]), float(self._K[1, 1])

        def _align_t_and_eval(r_cand: np.ndarray, t_seed: np.ndarray) -> tuple[np.ndarray, float, float]:
            ci, cc, pbbox = self._eval_6dof_vos_iou_fast(r_cand, t_seed, vos_s, scale)
            if pbbox is None:
                return t_seed, ci, cc
            px0, py0, px1, py1 = pbbox
            pcx = 0.5 * (px0 + px1)
            pcy = 0.5 * (py0 + py1)
            pw = max(8.0, px1 - px0)
            ph = max(8.0, py1 - py0)
            tz_old = float(t_seed[2])
            sr = float(np.clip(max(v_bbox_w / pw, v_bbox_h / ph), 0.78, 1.28)) if allow_scale else 1.0
            tz_new = float(np.clip(tz_old / max(sr, 0.5), max(0.06, 0.55 * self._init_tz), 1.65 * self._init_tz))
            t_new = t_seed.copy()
            t_new[0] = (t_seed[0] + (v_bbox_cx - pcx) * tz_old / fx) * (tz_new / tz_old)
            t_new[1] = (t_seed[1] + (v_bbox_cy - pcy) * tz_old / fy) * (tz_new / tz_old)
            t_new[2] = tz_new
            ci2, cc2, _ = self._eval_6dof_vos_iou_fast(r_cand, t_new, vos_s, scale)
            if ci2 >= ci:
                return t_new, ci2, cc2
            return t_seed, ci, cc

        best_r = rvec.copy()
        best_t = tvec.copy()
        best_iou = vos_iou
        best_cov = vos_cov
        best_rmse = base_rmse

        R_cur, _ = cv2.Rodrigues(best_r.astype(np.float64))
        proj_roll = math.atan2(-float(R_cur[0, 1]), float(R_cur[1, 1]))
        best_v_roll = v_roll
        best_d = abs(v_roll - proj_roll)
        for cand_a in (v_roll + math.pi, v_roll - math.pi):
            if abs(cand_a - proj_roll) < best_d:
                best_d = abs(cand_a - proj_roll)
                best_v_roll = cand_a

        candidates: list[np.ndarray] = []
        # 1) Direct in-plane roll alignment to live VOS principal axis
        if (vh / max(vw, 1.0)) >= 1.18 and abs(best_v_roll - proj_roll) >= math.radians(2.0):
            d_roll = float(np.clip(best_v_roll - proj_roll, -math.radians(22.0), math.radians(22.0)))
            R_dz, _ = cv2.Rodrigues(np.array([0.0, 0.0, d_roll], dtype=np.float64))
            r_roll, _ = cv2.Rodrigues(R_dz @ R_cur)
            candidates.append(r_roll.ravel())

        # 2) Pitch foreshortening alignment (recovers from backside pitch collapse or tracks forward tilts)
        if best_iou < 0.84:
            # Upright zero-pitch candidate preserving yaw and live VOS roll when silhouette is clearly upright
            if (vh / max(vw, 1.0)) >= 1.38:
                yaw_cur = math.atan2(float(R_cur[0, 2]), float(R_cur[2, 2]))
                R_y, _ = cv2.Rodrigues(np.array([0.0, yaw_cur, 0.0], dtype=np.float64))
                R_z, _ = cv2.Rodrigues(np.array([0.0, 0.0, best_v_roll], dtype=np.float64))
                r_up, _ = cv2.Rodrigues(R_z @ R_y)
                candidates.append(r_up.ravel())
            R_base = R_cur
            if candidates:
                # Also search pitch around roll-aligned base
                R_base, _ = cv2.Rodrigues(candidates[0].astype(np.float64))
            for dp_deg in (-16.0, -8.0, 8.0, 16.0):
                R_dx, _ = cv2.Rodrigues(np.array([math.radians(dp_deg), 0.0, 0.0], dtype=np.float64))
                r_p, _ = cv2.Rodrigues(R_dx @ R_base)
                candidates.append(r_p.ravel())

        for r_cand in candidates:
            t_cand, c_iou, c_cov = _align_t_and_eval(r_cand, best_t)
            if c_iou <= best_iou + 0.022:
                continue
            # Check KLT RMSE when tracking strongly; allow pure silhouette ICP when pose visibly diverged (vos_iou < 0.76)
            if (not coasting) and len(pts3d_inl) >= 8:
                proj_c, _ = cv2.projectPoints(pts3d_inl, r_cand, t_cand, self._K, None)
                errs = np.linalg.norm(proj_c.reshape(-1, 2) - pts2d_native_inl, axis=1)
                rmse_c = float(math.sqrt(float(np.mean(errs ** 2))))
                rmse_lim = max(base_rmse + 3.5 * res_scale, 7.5 * res_scale)
                if rmse_c > rmse_lim and not (vos_iou < 0.78 and c_iou >= vos_iou + 0.06):
                    continue
            else:
                rmse_c = base_rmse
            best_r = r_cand.copy()
            best_t = t_cand.copy()
            best_iou = c_iou
            best_cov = c_cov
            best_rmse = rmse_c

        return best_r, best_t, best_iou, best_cov, best_rmse

    def _estimate_cylinder_similarity_from_vos(self, vos_mask: np.ndarray) -> np.ndarray | None:
        """Compute a smooth 4-DoF similarity homography from frame-0 VOS frame to current VOS mask."""
        if self._init_vos_frame is None:
            return None
        cur_frame = _extract_oriented_mask_frame(vos_mask)
        if cur_frame is None:
            return None
        cx0, cy0, w0, h0, ang0 = self._init_vos_frame
        cx1, cy1, w1, h1, ang1 = cur_frame
        # Unwrap axis angle modulo pi closest to previous roll so cylinders never flip 180 deg
        best_ang = ang1
        best_diff = abs(ang1 - self._last_cyl_roll_rad)
        for cand_ang in (ang1 + math.pi, ang1 - math.pi):
            d = abs(cand_ang - self._last_cyl_roll_rad)
            if d < best_diff:
                best_diff = d
                best_ang = cand_ang
        # Smooth roll slightly
        d_ang = best_ang - ang0
        fg_area = float(np.count_nonzero(vos_mask > 127))
        if self._init_vos_area > 0.0 and fg_area >= 0.72 * self._init_vos_area:
            raw_scale = float(np.clip(0.5 * (w1 / max(w0, 1.0) + h1 / max(h0, 1.0)), 0.55, 1.85))
            scale = 0.65 * self._last_cyl_scale + 0.35 * raw_scale
        else:
            # Partially occluded by a gripping hand -> keep scale anchored to previous verified scale
            scale = self._last_cyl_scale

        ca, sa = math.cos(d_ang) * scale, math.sin(d_ang) * scale
        tx = cx1 - (ca * cx0 - sa * cy0)
        ty = cy1 - (sa * cx0 + ca * cy0)
        H = np.array([[ca, -sa, tx], [sa, ca, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
        self._last_cyl_roll_rad = best_ang
        self._last_cyl_scale = scale
        return H

    @property
    def ref_bbox(self) -> tuple[int, int, int, int]:
        return self._ref_bbox

    @property
    def ref_quad(self) -> np.ndarray:
        return self._ref_quad

    @property
    def ref_mask(self) -> np.ndarray | None:
        return self._ref_mask

    @property
    def reprojection_rmse_px(self) -> float | None:
        if not self._reproj_errors:
            return None
        arr = np.asarray(self._reproj_errors, dtype=np.float64)
        return float(math.sqrt(float(np.mean(arr * arr))))

    def _to_track_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Resize-first BGR -> tracking-resolution grayscale (fast on 720p/1080p)."""
        if self._track_w != self._frame_w or self._track_h != self._frame_h:
            small = cv2.resize(frame_bgr, (self._track_w, self._track_h), interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def _native_to_track(self, pts: np.ndarray) -> np.ndarray:
        out = pts.astype(np.float32, copy=True)
        out[:, 0] *= self._sx
        out[:, 1] *= self._sy
        return out

    def _track_to_native(self, pts: np.ndarray) -> np.ndarray:
        out = pts.astype(np.float32, copy=True)
        out[:, 0] /= self._sx
        out[:, 1] /= self._sy
        return out

    def _detect_corners_in_mask(self, gray_track: np.ndarray, mask_track: np.ndarray) -> np.ndarray:
        """Detect corners in tracking-space mask (ROI-cropped for speed on high-res frames) with optional 8x6 bucketing."""
        ys, xs = np.nonzero(mask_track)
        if len(xs) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        h, w = gray_track.shape[:2]
        x0 = max(0, int(xs.min()) - 6)
        y0 = max(0, int(ys.min()) - 6)
        x1 = min(w, int(xs.max()) + 7)
        y1 = min(h, int(ys.max()) + 7)
        sub_gray = gray_track[y0:y1, x0:x1]
        sub_mask = mask_track[y0:y1, x0:x1]

        if not self.spatial_bucketing:
            corners = cv2.goodFeaturesToTrack(
                sub_gray,
                maxCorners=self.max_klt_points,
                qualityLevel=0.01,
                minDistance=4.0,
                mask=sub_mask,
                blockSize=5,
            )
            if corners is None or len(corners) == 0:
                return np.zeros((0, 2), dtype=np.float32)
            pts = corners.reshape(-1, 2).astype(np.float32)
            pts[:, 0] += float(x0)
            pts[:, 1] += float(y0)
            return pts

        # Fixed-budget 8x6 spatial bucketing: detect up to 3x candidates, reserve per cell, cap at exact max_klt_points
        corners = cv2.goodFeaturesToTrack(
            sub_gray,
            maxCorners=self.max_klt_points * 3,
            qualityLevel=0.005,
            minDistance=3.0,
            mask=sub_mask,
            blockSize=5,
        )
        if corners is None or len(corners) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = corners.reshape(-1, 2).astype(np.float32)
        pts[:, 0] += float(x0)
        pts[:, 1] += float(y0)
        bbox = (float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
        # goodFeaturesToTrack returns points sorted by descending corner quality
        scores = np.linspace(1.0, 0.1, len(pts), dtype=np.float32)
        keep = _spatial_bucket_points(pts, scores, bbox, max_total=self.max_klt_points)
        return pts[keep]

    def initialize(
        self,
        frame_bgr: np.ndarray,
        seg: SegmentationResult,
    ) -> TrackFrameResult:
        """Initialize the tracker on the selection frame (frame_index = seg.frame_index)."""
        t0 = time.perf_counter()
        cv2.setRNGSeed(17)

        self._frame_h, self._frame_w = frame_bgr.shape[:2]
        bx, by, bw, bh = seg.bbox
        min_obj_dim = float(max(1, min(bw, bh)))
        scale_base = min(1.0, float(self.tracking_height) / float(self._frame_h))
        # Adaptive object-scale tracking: ensure small objects (e.g., portrait SHOW3D mug/keyboard)
        # retain >= 96 px on their shorter side at tracking resolution (capped at native resolution).
        scale_obj = min(1.0, 96.0 / min_obj_dim)
        scale = max(scale_base, scale_obj)
        if scale < 0.98:
            self._track_h = max(32, int(round(self._frame_h * scale)))
            self._track_w = max(32, int(round(self._frame_w * scale)))
        else:
            self._track_h = self._frame_h
            self._track_w = self._frame_w
        self._sx = float(self._track_w) / float(self._frame_w)
        self._sy = float(self._track_h) / float(self._frame_h)

        gray_track = self._to_track_gray(frame_bgr)
        self._prev_gray = gray_track
        self._ref_mask = seg.mask.copy()
        self._ref_bbox = seg.bbox
        self._last_2d_bbox: tuple[int, int, int, int] = tuple(int(v) for v in seg.bbox)
        self._ref_quad = np.array(
            [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]],
            dtype=np.float32,
        )
        self._ref_area = float(max(bw * bh, 1))

        # Reference 32x32 appearance thumbnail in tracking grayscale
        ref_quad_track = self._native_to_track(self._ref_quad)
        self._ref_patch = _extract_patch_thumb(gray_track, ref_quad_track, size=32)
        self._prev_patch = self._ref_patch.copy() if self._ref_patch is not None else None
        self._last_block_ncc = 1.0
        self._ref_grad_energy = _patch_grad_energy(self._ref_patch) if self._ref_patch is not None else 0.0

        # Build tracking-resolution eroded mask for clean interior feature seeding
        mask_track = (
            cv2.resize(seg.mask, (self._track_w, self._track_h), interpolation=cv2.INTER_NEAREST)
            if (self._track_w != self._frame_w or self._track_h != self._frame_h)
            else seg.mask.copy()
        )
        eroded = cv2.erode(mask_track, np.ones((3, 3), np.uint8), iterations=1)
        seed_mask = eroded if int(np.count_nonzero(eroded)) >= 64 else mask_track

        # Detect initial KLT points
        klt_pts = self._detect_corners_in_mask(gray_track, seed_mask)
        if len(klt_pts) < 16:
            ys, xs = np.nonzero(seed_mask)
            if len(xs) > 0:
                step = max(1, len(xs) // 48)
                grid = np.column_stack([xs[::step], ys[::step]]).astype(np.float32)
                klt_pts = np.vstack([klt_pts, grid[:48]]) if len(klt_pts) else grid[:48]

        # Detect reference ORB descriptors inside mask for recovery & re-anchoring
        self._use_native_recovery = (min(bw, bh) < 180.0) and (self._sx < 0.99)
        self._ref_keyframes = []
        self._frames_since_kf = 0
        if self._use_native_recovery:
            gray_native = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            kps, descs = self._orb_recovery.detectAndCompute(gray_native, seg.mask)
            if descs is not None and len(kps) > 0:
                self._ref_kps_native = np.array([kp.pt for kp in kps], dtype=np.float32)
                self._ref_descs = descs.copy()
                self._ref_keyframes.append((self._ref_kps_native.copy(), self._ref_descs.copy()))
            else:
                self._ref_kps_native = np.zeros((0, 2), dtype=np.float32)
                self._ref_descs = None
        else:
            kps, descs = self._orb_recovery.detectAndCompute(gray_track, mask_track)
            if descs is not None and len(kps) > 0:
                kp_track = np.array([kp.pt for kp in kps], dtype=np.float32)
                self._ref_kps_native = self._track_to_native(kp_track)
                self._ref_descs = descs.copy()
                self._ref_keyframes.append((self._ref_kps_native.copy(), self._ref_descs.copy()))
            else:
                self._ref_kps_native = np.zeros((0, 2), dtype=np.float32)
                self._ref_descs = None

        # Store initial VOS silhouette metadata for VOS-locked cylinder / 6-DoF alignment
        self._init_vos_area = float(np.count_nonzero(seg.mask > 127))
        self._init_vos_frame = _extract_oriented_mask_frame(seg.mask)
        if self._init_vos_frame is not None:
            self._last_cyl_roll_rad = self._init_vos_frame[4]
        self._last_cyl_scale = 1.0
        self._last_ang_vel = np.zeros(3, dtype=np.float64)
        self._vos_coast_frames = 0

        # Seed learned XFeat reference keyframe when matcher_backend == 'xfeat'
        self._xfeat_ref_keyframes = []
        if self.matcher_backend == "xfeat":
            xf = self._ensure_xfeat()
            roi0 = self._mask_bbox_roi(seg.mask)
            xf_pts0, xf_descs0 = xf.extract(frame_bgr, seg.mask, roi=roi0, top_k=640)
            if len(xf_pts0) < 12:
                xf_pts0, xf_descs0 = xf.extract(frame_bgr, seg.mask, roi=None, top_k=640)
            if len(xf_pts0) >= 8:
                self._xfeat_ref_keyframes.append((xf_pts0.copy(), xf_descs0.copy()))

        # Precompute 5x5 control grid for mesh_grid / cylinder mode
        gx = np.linspace(float(bx), float(bx + bw), 5, dtype=np.float32)
        gy = np.linspace(float(by), float(by + bh), 5, dtype=np.float32)
        gxx, gyy = np.meshgrid(gx, gy)
        self._grid_src = np.column_stack([gxx.ravel(), gyy.ravel()]).astype(np.float32)

        self._initialized = True
        self._lost_streak = 0
        self._frames_since_replenish = 0
        self._reproj_errors = []

        if self.wrap_mode == "mesh_6dof":
            return self._initialize_6dof(frame_bgr, gray_track, seg, klt_pts, t0)

        # If target patch is completely featureless (e.g., pitch-black video), start lost cleanly
        if self._ref_grad_energy < 0.4 or len(klt_pts) < 4:
            self._curr_pts_track = np.zeros((0, 2), dtype=np.float32)
            self._curr_ref_pts_native = np.zeros((0, 2), dtype=np.float32)
            self._last_H = None
            self._state = "lost"
            track_ms = (time.perf_counter() - t0) * 1000.0
            return TrackFrameResult(
                frame_index=int(seg.frame_index),
                timestamp_s=0.0,
                state="lost",
                inliers=0,
                confidence=0.0,
                track_ms=track_ms,
                confidence_state="LOST",
            )

        self._curr_pts_track = klt_pts.astype(np.float32)
        self._curr_ref_pts_native = self._track_to_native(klt_pts)
        self._last_H = np.eye(3, dtype=np.float64)
        self._state = "tracking"

        track_ms = (time.perf_counter() - t0) * 1000.0
        return TrackFrameResult(
            frame_index=int(seg.frame_index),
            timestamp_s=0.0,
            state="tracking",
            homography_3x3=self._last_H.copy(),
            grid_src_pts=self._grid_src.copy() if self.wrap_mode in ("mesh_grid", "cylinder") else None,
            grid_dst_pts=self._grid_src.copy() if self.wrap_mode in ("mesh_grid", "cylinder") else None,
            inliers=int(len(klt_pts)),
            confidence=1.0,
            track_ms=track_ms,
            confidence_state="TRACKING",
        )

    def _lift_2d_to_mesh3d(
        self,
        pts_native: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lift native 2D pixels onto 3D object-space mesh coordinates using fast native depth rasterization."""
        assert self.mesh is not None and self._K is not None
        if len(pts_native) == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=bool)
        R, _ = cv2.Rodrigues(rvec.astype(np.float64))
        verts_cam = (self.mesh.vertices.astype(np.float64) @ R.T + tvec.reshape(1, 3)).astype(np.float32)
        fx, fy, cx, cy = float(self._K[0, 0]), float(self._K[1, 1]), float(self._K[0, 2]), float(self._K[1, 2])
        dummy_rgb = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
        dummy_tex = np.full((4, 4, 4), 255, dtype=np.uint8)
        res = rasterize_mesh_rgbd(
            verts_cam,
            self.mesh.triangles,
            self.mesh.uvs,
            dummy_tex,
            (fx, fy, cx, cy),
            dummy_rgb,
            opacity=0.0,
            blend_mode="normal",
        )
        depth = res.depth_buf
        u = pts_native[:, 0]
        v = pts_native[:, 1]
        iu = np.clip(np.round(u).astype(np.int32), 0, self._frame_w - 1)
        iv = np.clip(np.round(v).astype(np.int32), 0, self._frame_h - 1)
        z = depth[iv, iu].astype(np.float64)
        valid = np.isfinite(z) & (z > 0.03) & (z < 100.0)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float64), valid
        uv_ok = pts_native[valid].astype(np.float64)
        z_ok = z[valid]
        x_cam = (uv_ok[:, 0] - cx) * z_ok / fx
        y_cam = (uv_ok[:, 1] - cy) * z_ok / fy
        p_cam = np.column_stack([x_cam, y_cam, z_ok])
        p_obj = (p_cam - tvec.reshape(1, 3)) @ R
        return p_obj, valid

    def _initialize_6dof(
        self,
        frame_bgr: np.ndarray,
        gray_track: np.ndarray,
        seg: SegmentationResult,
        klt_pts_track: np.ndarray,
        t0: float,
    ) -> TrackFrameResult:
        if self.mesh is None:
            self.mesh = build_parametric_mesh("box")
        if self.intrinsics is not None:
            fx, fy, cx, cy = self.intrinsics
        else:
            f = float(max(self._frame_w, self._frame_h)) * 0.9
            fx, fy, cx, cy = f, f, self._frame_w * 0.5, self._frame_h * 0.5
        self._K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        if self.init_pose_control is not None:
            rvec0, tvec0 = self.init_pose_control
            hypotheses = [(np.asarray(rvec0, dtype=np.float64).reshape(3), np.asarray(tvec0, dtype=np.float64).reshape(3), 1.0)]
        elif self.mesh.source_kind == "parametric" and self.mesh.name == "cylinder":
            # Exact perspective-correct causal cylinder/lathe fit from frame-0 mask (accounts for front-surface Z = tz - r)
            bx, by, bw, bh = seg.bbox
            ys_m, xs_m = np.nonzero(seg.mask)
            rad_prof: np.ndarray | None = None
            rz0 = 0.0
            if len(xs_m) >= 32:
                # Check if cylinder silhouette is clearly elongated and tilted (e.g. tilted handheld mug/bottle)
                pts_m = np.column_stack([xs_m.astype(np.float32), ys_m.astype(np.float32)])
                (rcx, rcy), (mr_w, mr_h), mr_ang = cv2.minAreaRect(pts_m)
                if mr_w > mr_h:
                    o_w, o_h = float(mr_h), float(mr_w)
                    axis_deg = float(mr_ang) + 90.0
                else:
                    o_w, o_h = float(mr_w), float(mr_h)
                    axis_deg = float(mr_ang)
                while axis_deg > 90.0:
                    axis_deg -= 180.0
                while axis_deg < -90.0:
                    axis_deg += 180.0
                if (o_h / max(o_w, 1.0)) >= 1.30 and 10.0 <= abs(axis_deg) <= 75.0:
                    rz0 = math.radians(axis_deg)
                    ca, sa = math.cos(-rz0), math.sin(-rz0)
                    dx_m = xs_m.astype(np.float64) - float(rcx)
                    dy_m = ys_m.astype(np.float64) - float(rcy)
                    rx_m = dx_m * ca - dy_m * sa
                    ry_m = dx_m * sa + dy_m * ca
                    x_lo, x_hi = float(np.percentile(rx_m, 1.0)), float(np.percentile(rx_m, 99.0))
                    y_lo, y_hi = float(np.percentile(ry_m, 1.0)), float(np.percentile(ry_m, 99.0))
                    eff_w = max(8.0, x_hi - x_lo)
                    eff_h = max(8.0, y_hi - y_lo)
                    mcx = float(rcx) + 0.5 * (x_lo + x_hi) * math.cos(rz0) - 0.5 * (y_lo + y_hi) * math.sin(rz0)
                    mcy = float(rcy) + 0.5 * (x_lo + x_hi) * math.sin(rz0) + 0.5 * (y_lo + y_hi) * math.cos(rz0)
                else:
                    # Use tight 0.5%-99.5% quantiles so stray pixels don't inflate size without shrinking true extent
                    x_lo, x_hi = float(np.percentile(xs_m, 0.5)), float(np.percentile(xs_m, 99.5))
                    y_lo, y_hi = float(np.percentile(ys_m, 0.5)), float(np.percentile(ys_m, 99.5))
                    eff_w = max(8.0, x_hi - x_lo)
                    eff_h = max(8.0, y_hi - y_lo)
                    mid_x = 0.5 * (x_lo + x_hi)
                    med_x = float(np.median(xs_m))
                    if abs(mid_x - med_x) > 0.05 * eff_w:
                        # Asymmetric protruding handle (e.g. pitcher): center axis on body median X
                        # rather than handle-shifted bbox midpoint so the opposite flank/spout is never clipped.
                        mcx = med_x
                    else:
                        mcx = mid_x
                    mcy = 0.5 * (y_lo + y_hi)

                    # Extract 17-slice axisymmetric radius profile r(v) so tapered bottles (e.g., BBQ sauce)
                    # fit their true 3D silhouette across ALL 360 deg without overhanging narrow necks!
                    n_slices = 17
                    row_widths = np.full(n_slices, eff_w, dtype=np.float64)
                    slice_h = eff_h / float(n_slices)
                    for sj in range(n_slices):
                        sy0 = int(round(y_lo + sj * slice_h))
                        sy1 = max(sy0 + 1, int(round(y_lo + (sj + 1) * slice_h)))
                        band = seg.mask[max(0, sy0):min(self._frame_h, sy1), :]
                        bxs = np.nonzero(band)[1]
                        if len(bxs) >= 4:
                            row_widths[sj] = float(np.percentile(bxs, 97) - np.percentile(bxs, 3))
                    # Smooth 3-tap along height
                    prof_raw = np.clip(row_widths / max(eff_w, 1.0), 0.30, 1.05)
                    prof_s = prof_raw.copy()
                    prof_s[1:-1] = 0.25 * prof_raw[:-2] + 0.50 * prof_raw[1:-1] + 0.25 * prof_raw[2:]
                    # Only enable tapered lathe when genuine TOP-neck tapering exists (top 30% narrower than 0.72x);
                    # never pinch the lower body due to foreground occluders (e.g. mustard bottle in front of pitcher).
                    if float(np.min(prof_s[:5])) < 0.72:
                        prof_s[5:] = np.maximum(prof_s[5:], 0.92)
                        rad_prof = prof_s
            else:
                eff_w, eff_h = max(8.0, float(bw)), max(8.0, float(bh))
                mcx, mcy = bx + bw * 0.5, by + bh * 0.5
            cyl_h = 0.14
            # Solve coupled perspective equations:
            #   eff_h = fy * cyl_h / (tz0 - cyl_r)
            #   eff_w/2 = fx * cyl_r / sqrt(tz0^2 - cyl_r^2)
            cyl_r = float(np.clip(0.5 * (eff_w / eff_h) * cyl_h, 0.015, 0.14))
            tz0 = max(0.14, fy * cyl_h / eff_h + cyl_r)
            for _ in range(3):
                half_w_rad = (0.5 * eff_w) / max(fx, 1.0)
                cyl_r = float(np.clip(tz0 * half_w_rad / math.sqrt(1.0 + half_w_rad * half_w_rad), 0.015, 0.14))
                tz0 = max(0.14, fy * cyl_h / eff_h + cyl_r)
            self.mesh = build_parametric_mesh("cylinder", cyl_radius=cyl_r, cyl_height=cyl_h, radius_profile=rad_prof)
            tx0 = (mcx - cx) * tz0 / fx
            ty0 = (mcy - cy) * tz0 / fy
            hypotheses = [(np.array([0.0, 0.0, rz0], dtype=np.float64), np.array([tx0, ty0, tz0], dtype=np.float64), 0.95)]
        else:
            hypotheses = _estimate_initial_pose_from_mask(seg.mask, self.mesh, self._K)

        # Primary hypothesis
        best_rvec, best_tvec, _ = hypotheses[0]
        self._init_tz = float(best_tvec[2])
        self._last_sil_bbox: tuple[int, int, int, int] = tuple(int(v) for v in seg.bbox)
        self._last_occluder_mask_img = None
        # Build frame-0 target Lab LUT (color frames only) to prevent background pixels from contaminating 3D replenishment
        # and to catch off-object drift into empty background space.
        self._ref_6dof_lab_lut: np.ndarray | None = None
        b0, g0, r0 = frame_bgr[::4, ::4, 0], frame_bgr[::4, ::4, 1], frame_bgr[::4, ::4, 2]
        chroma0 = float(np.mean(np.abs(r0.astype(np.int16) - g0.astype(np.int16)) + np.abs(g0.astype(np.int16) - b0.astype(np.int16))))
        if chroma0 > 6.0 and int(np.count_nonzero(seg.mask)) >= 100:
            lab0 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            t_pts = lab0[seg.mask > 0]
            lut_cnt = np.zeros((14, 33, 33), dtype=np.int32)
            np.add.at(
                lut_cnt,
                (
                    np.clip(t_pts[:, 0] // 20, 0, 13),
                    np.clip(t_pts[:, 1] // 8, 0, 32),
                    np.clip(t_pts[:, 2] // 8, 0, 32),
                ),
                1,
            )
            lut_b = lut_cnt >= max(10, int(0.015 * len(t_pts)))
            # L-only +/-1 dilation tolerates shading shifts without whitelisting unrelated background colors
            lut_d = lut_b.copy()
            lut_d[1:, :, :] |= lut_b[:-1, :, :]
            lut_d[:-1, :, :] |= lut_b[1:, :, :]
            self._ref_6dof_lab_lut = lut_d
        pts_native = self._track_to_native(klt_pts_track)
        pts3d, valid = self._lift_2d_to_mesh3d(pts_native, best_rvec, best_tvec)
        self._curr_pts_track = klt_pts_track[valid].astype(np.float32)
        self._curr_pts3d = pts3d.astype(np.float64)

        # Store secondary silhouette hypotheses for causal frame-1..3 symmetry disambiguation
        self._alt_hypotheses = []
        for alt_rvec, alt_tvec, _ in hypotheses[1:]:
            alt_pts3d, alt_valid = self._lift_2d_to_mesh3d(pts_native, alt_rvec, alt_tvec)
            if len(alt_pts3d) >= 12:
                self._alt_hypotheses.append((alt_rvec.copy(), alt_tvec.copy(), klt_pts_track[alt_valid].astype(np.float32), alt_pts3d))

        # Lift ORB reference bank to 3D for occlusion re-acquisition & seed verified multi-keyframe bank
        self._mesh_keyframes = []
        if self._ref_descs is not None and len(self._ref_kps_native) > 0:
            orb_pts3d, orb_valid = self._lift_2d_to_mesh3d(self._ref_kps_native, best_rvec, best_tvec)
            self._mesh_pts3d_bank = orb_pts3d[:1500]
            self._mesh_descs_bank = self._ref_descs[orb_valid][:1500].copy()
            if len(self._mesh_pts3d_bank) >= 12:
                self._mesh_keyframes.append((
                    best_rvec.copy(),
                    best_tvec.copy(),
                    self._mesh_pts3d_bank.copy(),
                    self._mesh_descs_bank.copy(),
                ))
        else:
            self._mesh_pts3d_bank = np.zeros((0, 3), dtype=np.float64)
            self._mesh_descs_bank = None

        # Lift learned XFeat reference features to 3D mesh for drift-free anchoring & recovery
        self._xfeat_mesh_keyframes = []
        if self.matcher_backend == "xfeat" and self._xfeat_ref_keyframes:
            xf_pts0, xf_descs0 = self._xfeat_ref_keyframes[0]
            xf_pts3d, xf_valid = self._lift_2d_to_mesh3d(xf_pts0, best_rvec, best_tvec)
            if len(xf_pts3d) >= 8:
                self._xfeat_mesh_keyframes.append((
                    best_rvec.copy(),
                    best_tvec.copy(),
                    xf_pts3d[:512].copy(),
                    xf_descs0[xf_valid][:512].copy(),
                ))

        self._last_rvec = best_rvec.copy()
        self._last_tvec = best_tvec.copy()
        self._state = "tracking"
        self._reproj_errors.append(0.0)

        track_ms = (time.perf_counter() - t0) * 1000.0
        return TrackFrameResult(
            frame_index=int(seg.frame_index),
            timestamp_s=0.0,
            state="tracking",
            pose_rvec=self._last_rvec.copy(),
            pose_tvec=self._last_tvec.copy(),
            inliers=int(len(self._curr_pts3d)),
            confidence=1.0,
            track_ms=track_ms,
            confidence_state="TRACKING",
        )

    def _match_orb_to_ref(
        self,
        gray_img: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
        *,
        high_density: bool = False,
        is_native: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match current frame ORB descriptors against reference keyframe bank."""
        if not self._ref_keyframes and (self._ref_descs is None or len(self._ref_descs) < 8):
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

        ox, oy = 0, 0
        if roi is not None:
            x0, y0, x1, y1 = roi
            sub = gray_img[y0:y1, x0:x1]
            ox, oy = x0, y0
        else:
            sub = gray_img

        det = self._orb_recovery if high_density else self._orb
        kps, descs = det.detectAndCompute(sub, None)
        if descs is None or len(kps) < 6:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

        bank = self._ref_keyframes if (high_density and self._ref_keyframes) else (
            [(self._ref_kps_native, self._ref_descs)] if self._ref_descs is not None else []
        )
        best_ref = np.zeros((0, 2), dtype=np.float32)
        best_cur = np.zeros((0, 2), dtype=np.float32)
        ratio_thr = 0.80 if high_density else 0.78
        dist_thr = 72 if high_density else 62

        for kf_pts_native, kf_descs in bank:
            if kf_descs is None or len(kf_descs) < 8:
                continue
            knn = self._bf.knnMatch(descs, kf_descs, k=2)
            best_by_ref: dict[int, tuple[float, tuple[float, float]]] = {}
            for pair in knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < ratio_thr * n.distance and m.distance <= dist_thr:
                    prev = best_by_ref.get(m.trainIdx)
                    if prev is None or m.distance < prev[0]:
                        pt = kps[m.queryIdx].pt
                        best_by_ref[m.trainIdx] = (m.distance, (pt[0] + ox, pt[1] + oy))
            if len(best_by_ref) > len(best_ref) and len(best_by_ref) >= 6:
                ref_idxs = sorted(best_by_ref.keys())
                cur_pts = np.array([best_by_ref[i][1] for i in ref_idxs], dtype=np.float32)
                cur_native = cur_pts if is_native else self._track_to_native(cur_pts)
                best_ref = kf_pts_native[ref_idxs]
                best_cur = cur_native

        return best_ref, best_cur

    def _maybe_store_keyframe(self, frame_bgr: np.ndarray, gray_track: np.ndarray, H: np.ndarray) -> None:
        """Store up to 5 high-confidence view keyframes mapped to frame-0 coordinates for robust recovery."""
        if len(self._ref_keyframes) >= 5 or self._ref_mask is None:
            return
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return
        warped_quad = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), H).reshape(4, 2)
        if self._use_native_recovery:
            gray_src = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            mask_kf = np.zeros((self._frame_h, self._frame_w), dtype=np.uint8)
            cv2.fillConvexPoly(mask_kf, np.round(warped_quad).astype(np.int32), 255)
            kps, descs = self._orb_recovery.detectAndCompute(gray_src, mask_kf)
            if descs is None or len(kps) < 20:
                return
            pts_native = np.array([kp.pt for kp in kps], dtype=np.float32)
        else:
            q_track = self._native_to_track(warped_quad)
            mask_kf = np.zeros((self._track_h, self._track_w), dtype=np.uint8)
            cv2.fillConvexPoly(mask_kf, np.round(q_track).astype(np.int32), 255)
            kps, descs = self._orb_recovery.detectAndCompute(gray_track, mask_kf)
            if descs is None or len(kps) < 20:
                return
            pts_native = self._track_to_native(np.array([kp.pt for kp in kps], dtype=np.float32))

        back_ref = cv2.perspectiveTransform(pts_native.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        iu = np.clip(np.round(back_ref[:, 0]).astype(np.int32), 0, self._frame_w - 1)
        iv = np.clip(np.round(back_ref[:, 1]).astype(np.int32), 0, self._frame_h - 1)
        inside = (
            (back_ref[:, 0] >= 0) & (back_ref[:, 0] < self._frame_w)
            & (back_ref[:, 1] >= 0) & (back_ref[:, 1] < self._frame_h)
            & (self._ref_mask[iv, iu] > 0)
        )
        if int(np.count_nonzero(inside)) >= 16:
            self._ref_keyframes.append((back_ref[inside].astype(np.float32), descs[inside].copy()))

    def _run_fb_klt(self, gray_track: np.ndarray, pts_track: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward-backward gated pyramidal Lucas-Kanade optical flow."""
        if self._prev_gray is None or len(pts_track) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)
        p0 = pts_track.reshape(-1, 1, 2).astype(np.float32)
        win_sz = (21, 21) if self._track_h > 600 else (15, 15)
        max_lvl = 4 if self._track_h > 600 else 3
        fb_max = 2.0 if self._track_h > 600 else 1.5
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray_track,
            p0,
            None,
            winSize=win_sz,
            maxLevel=max_lvl,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
        )
        if p1 is None or st1 is None:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(len(pts_track), dtype=bool)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(
            gray_track,
            self._prev_gray,
            p1,
            None,
            winSize=win_sz,
            maxLevel=max_lvl,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
        )
        if pb is None or st2 is None:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(len(pts_track), dtype=bool)
        fb_err = np.linalg.norm((pb - p0).reshape(-1, 2), axis=1)
        p1_2d = p1.reshape(-1, 2)
        good = (
            (st1.ravel() == 1)
            & (st2.ravel() == 1)
            & (fb_err <= fb_max)
            & np.all(np.isfinite(p1_2d), axis=1)
            & (p1_2d[:, 0] >= 0.0)
            & (p1_2d[:, 0] < float(self._track_w))
            & (p1_2d[:, 1] >= 0.0)
            & (p1_2d[:, 1] < float(self._track_h))
        )
        return p1_2d, good

    def _validate_homography_and_appearance(
        self,
        H: np.ndarray | None,
        gray_track: np.ndarray,
        inliers_mask: np.ndarray | None,
        cur_pts_native: np.ndarray,
        min_inliers: int,
        min_ncc: float,
        *,
        is_recovery: bool = False,
    ) -> tuple[bool, int, np.ndarray, np.ndarray | None, float]:
        if H is None or H.shape != (3, 3) or not np.all(np.isfinite(H)) or inliers_mask is None:
            return False, 0, np.zeros(0, dtype=bool), None, 0.0
        inl_bool = inliers_mask.ravel().astype(bool)
        n_inl = int(np.count_nonzero(inl_bool))
        if n_inl < min_inliers:
            return False, n_inl, inl_bool, None, 0.0
        inlier_ratio = float(n_inl) / float(max(len(inl_bool), 1))
        if inlier_ratio < 0.25:
            return False, n_inl, inl_bool, None, 0.0

        warped_quad = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), H).reshape(4, 2)
        if not _is_valid_quad(warped_quad, self._ref_area, self._frame_w, self._frame_h):
            return False, n_inl, inl_bool, None, 0.0

        # Spatial support spread check on inliers
        inl_pts = cur_pts_native[inl_bool]
        span_x = float(np.ptp(inl_pts[:, 0]))
        span_y = float(np.ptp(inl_pts[:, 1]))
        q_w = max(float(np.ptp(warped_quad[:, 0])), 1.0)
        q_h = max(float(np.ptp(warped_quad[:, 1])), 1.0)
        if span_x < 0.12 * q_w or span_y < 0.12 * q_h:
            return False, n_inl, inl_bool, None, 0.0

        # Fast 32x32 appearance & texture sanity check (prevents locking onto occluders/blank frames)
        cur_patch: np.ndarray | None = None
        block_ncc = 1.0
        if self._ref_patch is not None:
            cur_patch = _extract_patch_thumb(gray_track, self._native_to_track(warped_quad), size=32)
            if cur_patch is None:
                return False, n_inl, inl_bool, None, 0.0
            cur_grad = _patch_grad_energy(cur_patch)
            if self._ref_grad_energy > 4.0 and cur_grad < 0.22 * self._ref_grad_energy:
                return False, n_inl, inl_bool, cur_patch, 0.0

            if self.adaptive_appearance_gate and not is_recovery:
                block_ncc = _patch_block_ncc_p75(self._ref_patch, cur_patch, grid=4)
                temp_ncc = _patch_ncc(self._prev_patch, cur_patch) if self._prev_patch is not None else 0.0
                if self.grazing_angle_gate:
                    w_top = float(np.linalg.norm(warped_quad[1] - warped_quad[0]))
                    w_bot = float(np.linalg.norm(warped_quad[2] - warped_quad[3]))
                    h_l = float(np.linalg.norm(warped_quad[3] - warped_quad[0]))
                    h_r = float(np.linalg.norm(warped_quad[2] - warped_quad[1]))
                    w_r = min(w_top, w_bot) / max(float(self._ref_bbox[2]), 1.0)
                    h_r = min(h_l, h_r) / max(float(self._ref_bbox[3]), 1.0)
                    aspect_ratio = min(w_r, h_r) / max(w_r, h_r, 1e-6)
                    if block_ncc < 0.30 and aspect_ratio < 0.48:
                        return False, n_inl, inl_bool, cur_patch, block_ncc
                min_anchor = 0.14 if self.grazing_angle_gate else -1.0
                strong_geom_temporal = (
                    block_ncc >= min_anchor
                    and n_inl >= 18
                    and inlier_ratio >= 0.55
                    and span_x >= 0.25 * q_w
                    and span_y >= 0.25 * q_h
                    and temp_ncc >= 0.42
                )
                if block_ncc < 0.20 and not strong_geom_temporal:
                    return False, n_inl, inl_bool, cur_patch, block_ncc
            else:
                ncc = _patch_ncc(self._ref_patch, cur_patch)
                block_ncc = _patch_block_ncc_p75(self._ref_patch, cur_patch, grid=4)
                if max(ncc, block_ncc * 0.85) < min_ncc:
                    return False, n_inl, inl_bool, cur_patch, ncc

        return True, n_inl, inl_bool, cur_patch, block_ncc

    def _replenish_planar_points(
        self,
        gray_track: np.ndarray,
        H: np.ndarray,
        *,
        external_fg_mask: np.ndarray | None = None,
    ) -> None:
        """Detect new Shi-Tomasi corners inside the current tracked object (excluding occluders) and map back to frame-0 coords."""
        if self._ref_mask is None:
            return
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return
        warped_quad = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), H).reshape(4, 2)
        q_track = self._native_to_track(warped_quad)
        mask_cur = np.zeros((self._track_h, self._track_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask_cur, np.round(q_track).astype(np.int32), 255)
        if external_fg_mask is not None and external_fg_mask.shape == (self._frame_h, self._frame_w):
            vos_in = self._erode_vos_mask(external_fg_mask)
            vos_tr = (
                cv2.resize(vos_in, (self._track_w, self._track_h), interpolation=cv2.INTER_NEAREST)
                if (self._track_w != self._frame_w or self._track_h != self._frame_h)
                else vos_in
            )
            mask_cur = cv2.bitwise_and(mask_cur, vos_tr)
        elif (
            self.wrap_mode == "planar"
            and self._last_occluder_mask_img is not None
            and self._last_occluder_mask_img.shape == (self._frame_h, self._frame_w)
        ):
            occ_tr = (
                cv2.resize(self._last_occluder_mask_img, (self._track_w, self._track_h), interpolation=cv2.INTER_NEAREST)
                if (self._track_w != self._frame_w or self._track_h != self._frame_h)
                else self._last_occluder_mask_img
            )
            occ_tr = cv2.dilate(occ_tr, np.ones((7, 7), np.uint8), iterations=1)
            cand = cv2.bitwise_and(mask_cur, cv2.bitwise_not(occ_tr))
            if int(np.count_nonzero(cand)) >= int(0.45 * max(int(np.count_nonzero(mask_cur)), 1)):
                mask_cur = cand
        mask_cur = cv2.erode(mask_cur, np.ones((3, 3), np.uint8), iterations=1)
        new_pts_track = self._detect_corners_in_mask(gray_track, mask_cur)
        if len(new_pts_track) == 0:
            return
        new_native = self._track_to_native(new_pts_track)
        back_ref = cv2.perspectiveTransform(new_native.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        iu = np.clip(np.round(back_ref[:, 0]).astype(np.int32), 0, self._frame_w - 1)
        iv = np.clip(np.round(back_ref[:, 1]).astype(np.int32), 0, self._frame_h - 1)
        inside = (
            (back_ref[:, 0] >= 0) & (back_ref[:, 0] < self._frame_w)
            & (back_ref[:, 1] >= 0) & (back_ref[:, 1] < self._frame_h)
            & (self._ref_mask[iv, iu] > 0)
        )
        if not np.any(inside):
            return
        add_track = new_pts_track[inside]
        add_ref = back_ref[inside].astype(np.float32)

        if len(self._curr_pts_track) > 0:
            # Avoid duplicates within 4 tracking pixels
            d2 = np.sum((add_track[:, None, :] - self._curr_pts_track[None, :, :]) ** 2, axis=2)
            far = d2.min(axis=1) > 16.0
            add_track = add_track[far]
            add_ref = add_ref[far]
            if len(add_track) == 0:
                return
            merged_track = np.vstack([self._curr_pts_track, add_track])[: self.max_klt_points]
            merged_ref = np.vstack([self._curr_ref_pts_native, add_ref])[: self.max_klt_points]
        else:
            merged_track = add_track[: self.max_klt_points]
            merged_ref = add_ref[: self.max_klt_points]

        self._curr_pts_track = merged_track.astype(np.float32)
        self._curr_ref_pts_native = merged_ref.astype(np.float32)

    def _compute_grid_dst(self, H: np.ndarray) -> np.ndarray | None:
        if self._grid_src is None:
            return None
        return cv2.perspectiveTransform(self._grid_src.reshape(-1, 1, 2), H).reshape(-1, 2).astype(np.float32)

    def _eval_quad_vos_metrics(self, H: np.ndarray, vos_mask: np.ndarray) -> tuple[float, float, float]:
        """Return (IoU, recall=inter/vos, precision=inter/quad) between warped ref quad and live VOS mask."""
        n_vos = int(np.count_nonzero(vos_mask > 127))
        if n_vos < 16:
            return 0.0, 0.0, 0.0
        warped_q = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), H).reshape(4, 2)
        q_mask = np.zeros((self._frame_h, self._frame_w), dtype=np.uint8)
        cv2.fillConvexPoly(q_mask, np.round(warped_q).astype(np.int32), 255)
        n_proj = int(np.count_nonzero(q_mask))
        if n_proj < 16:
            return 0.0, 0.0, 0.0
        inter = int(np.count_nonzero((q_mask > 0) & (vos_mask > 127)))
        union = n_vos + n_proj - inter
        return float(inter) / float(max(union, 1)), float(inter) / float(n_vos), float(inter) / float(n_proj)

    def _eval_quad_vos_overlap(self, H: np.ndarray, vos_mask: np.ndarray) -> float:
        """Return fraction of live VOS pixels lying inside the warped reference quad."""
        _, rec, _ = self._eval_quad_vos_metrics(H, vos_mask)
        return rec

    def track_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_s: float,
        *,
        external_fg_mask: np.ndarray | None = None,
    ) -> TrackFrameResult:
        """Causal per-frame tracking step shared between file processing and live 30 Hz replay."""
        if not self._initialized:
            raise RuntimeError("VideoSurfaceTracker.initialize() must be called before track_frame()")

        t0 = time.perf_counter()
        gray_track = self._to_track_gray(frame_bgr)
        use_vos = external_fg_mask is not None and external_fg_mask.shape == (self._frame_h, self._frame_w)

        if self.wrap_mode == "mesh_6dof":
            return self._track_frame_6dof(
                frame_bgr,
                gray_track,
                frame_index,
                timestamp_s,
                t0,
                external_fg_mask=external_fg_mask if use_vos else None,
            )

        # If VOS confirms the target object is completely gone/occluded, suppress immediately
        if use_vos and external_fg_mask is not None and int(np.count_nonzero(external_fg_mask > 127)) < 48:
            self._state = "lost"
            self._last_H = None
            self._prev_patch = None
            self._cand_H = None
            self._curr_pts_track = np.zeros((0, 2), dtype=np.float32)
            self._curr_ref_pts_native = np.zeros((0, 2), dtype=np.float32)
            self._lost_streak += 1
            self._prev_gray = gray_track
            track_ms = (time.perf_counter() - t0) * 1000.0
            return TrackFrameResult(
                frame_index=int(frame_index),
                timestamp_s=float(timestamp_s),
                state="lost",
                inliers=0,
                confidence=0.0,
                track_ms=track_ms,
                confidence_state="LOST",
            )

        vos_inner = self._erode_vos_mask(external_fg_mask) if (use_vos and external_fg_mask is not None) else None

        # --- Planar / Cylinder / Mesh-Grid Tracking ---
        if self._state == "tracking":
            p1_track, good = self._run_fb_klt(gray_track, self._curr_pts_track)
            klt_cur_track = p1_track[good]
            klt_ref_native = self._curr_ref_pts_native[good]
            klt_cur_native = self._track_to_native(klt_cur_track)

            # Strictly gate KLT points inside live VOS mask (zero finger/bar/background contamination!)
            planar_occ_frac = 0.0
            if vos_inner is not None and len(klt_cur_native) > 0:
                iu = np.clip(np.round(klt_cur_native[:, 0]).astype(np.int32), 0, self._frame_w - 1)
                iv = np.clip(np.round(klt_cur_native[:, 1]).astype(np.int32), 0, self._frame_h - 1)
                on_obj = vos_inner[iv, iu] > 127
                klt_cur_track = klt_cur_track[on_obj]
                klt_ref_native = klt_ref_native[on_obj]
                klt_cur_native = klt_cur_native[on_obj]
            elif (
                self.wrap_mode == "planar"
                and self._last_occluder_mask_img is not None
                and len(klt_cur_native) >= 16
                and self._last_occluder_mask_img.shape == (self._frame_h, self._frame_w)
            ):
                occ_dil = cv2.dilate(self._last_occluder_mask_img, np.ones((15, 15), np.uint8), iterations=1)
                iu = np.clip(np.round(klt_cur_native[:, 0]).astype(np.int32), 0, self._frame_w - 1)
                iv = np.clip(np.round(klt_cur_native[:, 1]).astype(np.int32), 0, self._frame_h - 1)
                unocc = occ_dil[iv, iu] == 0
                planar_occ_frac = 1.0 - (float(np.count_nonzero(unocc)) / float(len(unocc)))
                if planar_occ_frac < 0.52 and int(np.count_nonzero(unocc)) >= 14:
                    klt_cur_track = klt_cur_track[unocc]
                    klt_ref_native = klt_ref_native[unocc]
                    klt_cur_native = klt_cur_native[unocc]

            ref_all = klt_ref_native
            cur_all = klt_cur_native

            # Periodic or low-support XFeat drift-anchoring (when matcher_backend == 'xfeat')
            if self.matcher_backend == "xfeat" and (frame_index % 5 == 0 or len(ref_all) < 28):
                xf_ref, xf_cur = self._match_xfeat_2d(frame_bgr, vos_inner)
                if len(xf_ref) >= 6:
                    ref_all = np.vstack([ref_all, xf_ref]) if len(ref_all) else xf_ref
                    cur_all = np.vstack([cur_all, xf_cur]) if len(cur_all) else xf_cur
            elif len(ref_all) < 24:
                orb_ref, orb_cur = self._match_orb_to_ref(gray_track)
                if len(orb_ref) >= 6:
                    ref_all = np.vstack([ref_all, orb_ref]) if len(ref_all) else orb_ref
                    cur_all = np.vstack([cur_all, orb_cur]) if len(cur_all) else orb_cur

            H_est: np.ndarray | None = None
            inl_mask: np.ndarray | None = None
            if len(ref_all) >= 8:
                cv2.setRNGSeed(17)
                H_est, inl_mask = cv2.findHomography(ref_all, cur_all, cv2.RANSAC, 3.0)
                if self.wrap_mode == "cylinder" and H_est is not None and abs(float(H_est[2, 2])) > 1e-8:
                    Hn = H_est / float(H_est[2, 2])
                    q_h = np.hstack([self._ref_quad.astype(np.float64), np.ones((4, 1), dtype=np.float64)])
                    w_q = (q_h @ Hn.T)[:, 2]
                    svals = np.linalg.svd(Hn[:2, :2], compute_uv=False)
                    aniso = float(svals[0] / max(svals[1], 1e-6))
                    w_spread = float(w_q.max() / max(float(w_q.min()), 1e-6)) if float(w_q.min()) > 1e-4 else 99.0
                    aniso_lim = 1.10 if use_vos else 1.42
                    w_lim = 1.06 if use_vos else 1.32
                    if aniso > aniso_lim or w_spread > w_lim:
                        M_sim, inl_sim = cv2.estimateAffinePartial2D(
                            ref_all.astype(np.float32),
                            cur_all.astype(np.float32),
                            method=cv2.RANSAC,
                            ransacReprojThreshold=3.0,
                        )
                        if M_sim is not None and inl_sim is not None and int(np.count_nonzero(inl_sim)) >= 8:
                            H_sim = np.eye(3, dtype=np.float64)
                            H_sim[:2, :] = M_sim.astype(np.float64)
                            H_est = H_sim
                            inl_mask = inl_sim

            ok, n_inl, inl_bool, cur_patch, block_ncc = self._validate_homography_and_appearance(
                H_est, gray_track, inl_mask, cur_all, min_inliers=10, min_ncc=0.18, is_recovery=False
            )

            # VOS consistency check & graceful appearance-tolerance when VOS proves exact surface alignment
            vos_ov = 1.0
            if use_vos and external_fg_mask is not None and H_est is not None and inl_mask is not None:
                vos_ov = self._eval_quad_vos_overlap(H_est, external_fg_mask)
                if ok and vos_ov < 0.25:
                    ok = False
                elif (not ok) and int(np.count_nonzero(inl_mask)) >= 10 and vos_ov >= 0.55:
                    warped_q_v = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), H_est).reshape(4, 2)
                    if _is_valid_quad(warped_q_v, self._ref_area, self._frame_w, self._frame_h):
                        inl_bool = inl_mask.ravel().astype(bool)
                        n_inl = int(np.count_nonzero(inl_bool))
                        ok = True
                        block_ncc = max(block_ncc, 0.30)

            # Continuity jump check while tracking (reject implausible single-frame teleport)
            if ok and self._last_H is not None and H_est is not None:
                prev_q = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), self._last_H).reshape(4, 2)
                cur_q = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), H_est).reshape(4, 2)
                max_shift = float(np.linalg.norm(cur_q - prev_q, axis=1).max())
                if max_shift > max(self._frame_w, self._frame_h) * 0.35:
                    ok = False

            # Smooth VOS-locked 4-DoF cylinder fallback & specular-drift clamp for smooth rotating bottles
            vos_cyl_fallback = False
            if self.wrap_mode == "cylinder" and use_vos and external_fg_mask is not None:
                vos_fg_2d = int(np.count_nonzero(external_fg_mask > 127))
                if self._init_vos_area > 0.0 and vos_fg_2d < max(160, int(0.36 * self._init_vos_area)):
                    # Object heavily obscured by hands (e.g., <36% visible stub) -> suppress cleanly!
                    ok = False
                else:
                    cyl_drifted = False
                    if ok and H_est is not None:
                        cur_f = _extract_oriented_mask_frame(external_fg_mask)
                        if cur_f is not None:
                            wq = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), H_est).reshape(4, 2)
                            q_cx = float(wq[:, 0].mean())
                            q_w = float(np.linalg.norm(wq[1] - wq[0]))
                            q_span_x = float(wq[:, 0].max() - wq[:, 0].min())
                            if (
                                abs(q_cx - cur_f[0]) > max(3.0, 0.03 * cur_f[2])
                                or not (0.92 <= q_w / max(cur_f[2], 1.0) <= 1.08)
                                or q_span_x > 1.12 * max(cur_f[2], 1.0)
                            ):
                                cyl_drifted = True
                    if (not ok) or vos_ov < 0.65 or cyl_drifted:
                        H_vos = self._estimate_cylinder_similarity_from_vos(external_fg_mask)
                        if H_vos is not None and self._eval_quad_vos_overlap(H_vos, external_fg_mask) >= 0.48:
                            if self._last_cyl_H_ema is not None:
                                H_vos = 0.60 * H_vos + 0.40 * self._last_cyl_H_ema
                                H_vos /= max(abs(float(H_vos[2, 2])), 1e-8)
                            self._last_cyl_H_ema = H_vos.copy()
                            H_est = H_vos
                            ok = True
                            vos_cyl_fallback = True
                            n_inl = max(n_inl, 16)
                            block_ncc = max(block_ncc, 0.35)
                    elif ok and H_est is not None:
                        if self._last_cyl_H_ema is not None:
                            H_est = 0.72 * H_est + 0.28 * self._last_cyl_H_ema
                            H_est /= max(abs(float(H_est[2, 2])), 1e-8)
                        self._last_cyl_H_ema = H_est.copy()

            if ok and H_est is not None:
                # Reprojection RMSE on inliers
                if not vos_cyl_fallback and len(ref_all) > 0 and len(inl_bool) == len(ref_all) and np.any(inl_bool):
                    proj = cv2.perspectiveTransform(ref_all[inl_bool].reshape(-1, 1, 2), H_est).reshape(-1, 2)
                    err = np.linalg.norm(proj - cur_all[inl_bool], axis=1)
                    med_err = float(np.median(err)) if len(err) else 0.0
                else:
                    med_err = 0.0
                self._reproj_errors.append(med_err)

                # Explicit AMBIGUOUS gate: suppress overlay when support is thin and appearance is weak
                if (not vos_cyl_fallback) and n_inl < 14 and block_ncc < 0.25 and med_err > 2.5 and (not use_vos or vos_ov < 0.55):
                    self._state = "recovering"
                    self._cand_H = H_est
                    self._prev_gray = gray_track
                    track_ms = (time.perf_counter() - t0) * 1000.0
                    return TrackFrameResult(
                        frame_index=int(frame_index),
                        timestamp_s=float(timestamp_s),
                        state="recovering",
                        inliers=n_inl,
                        confidence=0.0,
                        track_ms=track_ms,
                        confidence_state="AMBIGUOUS",
                    )

                self._last_H = H_est
                if self.wrap_mode == "cylinder" and use_vos and external_fg_mask is not None and not vos_cyl_fallback:
                    cur_f = _extract_oriented_mask_frame(external_fg_mask)
                    if cur_f is not None:
                        self._last_cyl_roll_rad = cur_f[4]
                warped_q_2d = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), H_est).reshape(4, 2)
                x0_2 = max(0, int(math.floor(float(warped_q_2d[:, 0].min()))))
                y0_2 = max(0, int(math.floor(float(warped_q_2d[:, 1].min()))))
                x1_2 = min(self._frame_w, int(math.ceil(float(warped_q_2d[:, 0].max()))))
                y1_2 = min(self._frame_h, int(math.ceil(float(warped_q_2d[:, 1].max()))))
                if (x1_2 - x0_2) >= 12 and (y1_2 - y0_2) >= 12:
                    self._last_2d_bbox = (x0_2, y0_2, x1_2 - x0_2, y1_2 - y0_2)
                self._prev_patch = cur_patch
                self._last_block_ncc = block_ncc
                if not vos_cyl_fallback and len(cur_all) > 0 and len(inl_bool) == len(cur_all) and np.any(inl_bool):
                    self._curr_pts_track = self._native_to_track(cur_all[inl_bool])
                    self._curr_ref_pts_native = ref_all[inl_bool].astype(np.float32)
                else:
                    self._curr_pts_track = np.zeros((0, 2), dtype=np.float32)
                    self._curr_ref_pts_native = np.zeros((0, 2), dtype=np.float32)
                self._frames_since_replenish += 1
                self._frames_since_kf += 1
                # Prevent occluded frames from contaminating KLT memory
                allow_replenish = use_vos or (((not self.adaptive_appearance_gate) or (block_ncc >= 0.28)) and (planar_occ_frac < 0.08))
                if allow_replenish and (
                    len(self._curr_pts_track) < int(self.max_klt_points * 0.55) or self._frames_since_replenish >= 12
                ):
                    self._replenish_planar_points(gray_track, H_est, external_fg_mask=external_fg_mask if use_vos else None)
                    self._frames_since_replenish = 0
                if (block_ncc >= 0.35 or (use_vos and vos_ov >= 0.70)) and n_inl >= 25 and self._frames_since_kf >= 15:
                    self._maybe_store_keyframe(frame_bgr, gray_track, H_est)
                    if self.matcher_backend == "xfeat" and not vos_cyl_fallback:
                        self._maybe_store_xfeat_keyframe_2d(frame_bgr, H_est, vos_inner)
                    self._frames_since_kf = 0

                self._prev_gray = gray_track
                self._lost_streak = 0
                conf = float(np.clip(n_inl / max(float(len(ref_all)), 1.0), 0.0, 1.0)) if not vos_cyl_fallback else 0.75
                track_ms = (time.perf_counter() - t0) * 1000.0
                return TrackFrameResult(
                    frame_index=int(frame_index),
                    timestamp_s=float(timestamp_s),
                    state="tracking",
                    homography_3x3=H_est.copy(),
                    grid_src_pts=self._grid_src.copy() if self.wrap_mode in ("mesh_grid", "cylinder") else None,
                    grid_dst_pts=self._compute_grid_dst(H_est) if self.wrap_mode in ("mesh_grid", "cylinder") else None,
                    inliers=n_inl,
                    confidence=conf,
                    track_ms=track_ms,
                    confidence_state="TRACKING",
                )

            # Tracking lost -> immediately suppress overlay!
            self._state = "lost"
            self._last_H = None
            self._prev_patch = None
            self._cand_H = None
            self._curr_pts_track = np.zeros((0, 2), dtype=np.float32)
            self._curr_ref_pts_native = np.zeros((0, 2), dtype=np.float32)
            self._lost_streak = 1
            self._prev_gray = gray_track
            track_ms = (time.perf_counter() - t0) * 1000.0
            return TrackFrameResult(
                frame_index=int(frame_index),
                timestamp_s=float(timestamp_s),
                state="lost",
                inliers=n_inl,
                confidence=0.0,
                track_ms=track_ms,
                confidence_state="LOST",
            )

        # --- Lost or Recovering: XFeat (when enabled) or high-density multi-keyframe ORB re-acquisition ---
        orb_ref = np.zeros((0, 2), dtype=np.float32)
        orb_cur = np.zeros((0, 2), dtype=np.float32)

        if self.matcher_backend == "xfeat" and self._xfeat_ref_keyframes:
            xf_ref, xf_cur = self._match_xfeat_2d(frame_bgr, vos_inner, min_cossim=0.73, top_k=640)
            if len(xf_ref) < 12:
                xf_ref_f, xf_cur_f = self._match_xfeat_2d(frame_bgr, None, roi=None, min_cossim=0.75, top_k=800)
                if len(xf_ref_f) > len(xf_ref):
                    xf_ref, xf_cur = xf_ref_f, xf_cur_f
            # If upright XFeat finds too few matches after >= 4 lost frames (e.g., object rotated ~90 deg
            # in-plane from vertical to horizontal like a handheld keyboard), test +/-90 deg rotated ROI!
            if len(xf_ref) < 14 and self._lost_streak >= 4 and vos_inner is not None and int(np.count_nonzero(vos_inner)) >= 200:
                xf_m = self._ensure_xfeat()
                roi_r = self._mask_bbox_roi(vos_inner, pad_frac=0.18)
                if roi_r is not None:
                    rx0, ry0, rx1, ry1 = roi_r
                    crop_r = frame_bgr[ry0:ry1, rx0:rx1]
                    mask_r = vos_inner[ry0:ry1, rx0:rx1]
                    ch, cw = crop_r.shape[:2]
                    for rot_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                        rot_img = cv2.rotate(crop_r, rot_code)
                        rot_msk = cv2.rotate(mask_r, rot_code)
                        r_pts, r_descs = xf_m.extract(rot_img, rot_msk, roi=None, top_k=512)
                        if len(r_pts) < 12:
                            continue
                        # Map rotated crop coords back to original full-frame coords
                        if rot_code == cv2.ROTATE_90_CLOCKWISE:
                            orig_x = r_pts[:, 1] + float(rx0)
                            orig_y = (float(ch - 1) - r_pts[:, 0]) + float(ry0)
                        else:
                            orig_x = (float(cw - 1) - r_pts[:, 1]) + float(rx0)
                            orig_y = r_pts[:, 0] + float(ry0)
                        mapped_pts = np.column_stack([orig_x, orig_y]).astype(np.float32)
                        for kf_pts, kf_descs in self._xfeat_ref_keyframes:
                            m_ref_r, m_cur_r = xf_m.match(kf_pts, kf_descs, mapped_pts, r_descs, min_cossim=0.73)
                            if len(m_ref_r) > len(xf_ref):
                                xf_ref, xf_cur = m_ref_r, m_cur_r
            orb_ref, orb_cur = xf_ref, xf_cur
        else:
            if self._use_native_recovery:
                gray_rec = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                rec_h, rec_w = self._frame_h, self._frame_w
                is_nat = True
            else:
                gray_rec = gray_track
                rec_h, rec_w = self._track_h, self._track_w
                is_nat = False

            # First try causal generous 3.5x ROI around last verified 2D bbox (concentrates ORB keypoints on small objects)
            lbx, lby, lbw, lbh = self._last_2d_bbox
            if not is_nat:
                lbx = int(round(lbx * self._sx))
                lby = int(round(lby * self._sy))
                lbw = max(12, int(round(lbw * self._sx)))
                lbh = max(12, int(round(lbh * self._sy)))
            pad_x = max(lbw, int(0.20 * rec_w))
            pad_y = max(lbh, int(0.20 * rec_h))
            rx0 = max(0, lbx - pad_x)
            ry0 = max(0, lby - pad_y)
            rx1 = min(rec_w, lbx + lbw + pad_x)
            ry1 = min(rec_h, lby + lbh + pad_y)
            if (rx1 - rx0) >= 48 and (ry1 - ry0) >= 48 and (rx1 - rx0) * (ry1 - ry0) < int(0.75 * rec_w * rec_h):
                r_ref, r_cur = self._match_orb_to_ref(gray_rec, roi=(rx0, ry0, rx1, ry1), high_density=True, is_native=is_nat)
                if len(r_ref) > len(orb_ref):
                    orb_ref, orb_cur = r_ref, r_cur
            # Fallback to full-frame search if ROI yielded thin matches (handles large motion during blackout)
            if len(orb_ref) < 20:
                f_ref, f_cur = self._match_orb_to_ref(gray_rec, roi=None, high_density=True, is_native=is_nat)
                if len(f_ref) > len(orb_ref):
                    orb_ref, orb_cur = f_ref, f_cur

        H_cand: np.ndarray | None = None
        inl_mask = None
        if len(orb_ref) >= 9:
            cv2.setRNGSeed(17)
            H_cand, inl_mask = cv2.findHomography(orb_ref, orb_cur, cv2.RANSAC, 4.0)

        n_inl_raw = int(np.count_nonzero(inl_mask)) if inl_mask is not None else 0
        rec_min_ncc = 0.14 if n_inl_raw >= 20 else 0.22
        ok, n_inl, inl_bool, cur_patch, block_ncc = self._validate_homography_and_appearance(
            H_cand, gray_track, inl_mask, orb_cur, min_inliers=9, min_ncc=rec_min_ncc, is_recovery=True
        )
        if use_vos and external_fg_mask is not None and H_cand is not None and inl_mask is not None:
            vos_iou_r, vos_rec_r, vos_prec_r = self._eval_quad_vos_metrics(H_cand, external_fg_mask)
            warped_q_r = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), H_cand).reshape(4, 2)
            # Check aspect-ratio consistency so 90-deg flipped trapezoids are strictly rejected!
            ref_w_q = float(np.linalg.norm(self._ref_quad[1] - self._ref_quad[0]))
            ref_h_q = float(np.linalg.norm(self._ref_quad[3] - self._ref_quad[0]))
            cur_w_q = 0.5 * (float(np.linalg.norm(warped_q_r[1] - warped_q_r[0])) + float(np.linalg.norm(warped_q_r[2] - warped_q_r[3])))
            cur_h_q = 0.5 * (float(np.linalg.norm(warped_q_r[3] - warped_q_r[0])) + float(np.linalg.norm(warped_q_r[2] - warped_q_r[1])))
            ref_ar = ref_w_q / max(ref_h_q, 1.0)
            cur_ar = cur_w_q / max(cur_h_q, 1.0)
            ar_ratio = cur_ar / max(ref_ar, 1e-3)
            aspect_ok = 0.42 <= ar_ratio <= 2.35
            if ok and (vos_iou_r < 0.32 or vos_prec_r < 0.42 or not aspect_ok):
                ok = False
            elif (not ok) and n_inl_raw >= 14 and vos_iou_r >= 0.48 and vos_prec_r >= 0.58 and vos_rec_r >= 0.55 and aspect_ok:
                if _is_valid_quad(warped_q_r, self._ref_area, self._frame_w, self._frame_h):
                    inl_bool = inl_mask.ravel().astype(bool)
                    n_inl = int(np.count_nonzero(inl_bool))
                    ok = True
                    block_ncc = max(block_ncc, 0.28)
        if (
            not ok
            and self.wrap_mode == "cylinder"
            and use_vos
            and external_fg_mask is not None
            and self._init_vos_area > 0.0
            and int(np.count_nonzero(external_fg_mask > 127)) >= max(240, int(0.50 * self._init_vos_area))
        ):
            H_vos_r = self._estimate_cylinder_similarity_from_vos(external_fg_mask)
            if H_vos_r is not None and self._eval_quad_vos_overlap(H_vos_r, external_fg_mask) >= 0.55:
                H_cand = H_vos_r
                ok = True
                n_inl = 16
                inl_bool = np.zeros((0,), dtype=bool)
                orb_ref = np.zeros((0, 2), dtype=np.float32)
                orb_cur = np.zeros((0, 2), dtype=np.float32)
                cur_patch = None
                block_ncc = 0.35
        self._prev_gray = gray_track

        if ok and H_cand is not None:
            if self._state == "recovering" and self._cand_H is not None:
                prev_q = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), self._cand_H).reshape(4, 2)
                cur_q = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), H_cand).reshape(4, 2)
                disp = float(np.linalg.norm(cur_q - prev_q, axis=1).mean())
                if disp <= max(self._frame_w, self._frame_h) * 0.12:
                    # Two consecutive consistent detections -> REACQUIRED!
                    self._state = "tracking"
                    self._last_H = H_cand
                    self._prev_patch = cur_patch
                    self._last_block_ncc = block_ncc
                    self._cand_H = None
                    self._curr_pts_track = self._native_to_track(orb_cur[inl_bool])
                    self._curr_ref_pts_native = orb_ref[inl_bool].astype(np.float32)
                    self._replenish_planar_points(gray_track, H_cand, external_fg_mask=external_fg_mask if use_vos else None)
                    self._lost_streak = 0
                    conf = float(np.clip(n_inl / max(float(len(orb_ref)), 1.0), 0.0, 1.0))
                    track_ms = (time.perf_counter() - t0) * 1000.0
                    return TrackFrameResult(
                        frame_index=int(frame_index),
                        timestamp_s=float(timestamp_s),
                        state="tracking",
                        homography_3x3=H_cand.copy(),
                        grid_src_pts=self._grid_src.copy() if self.wrap_mode in ("mesh_grid", "cylinder") else None,
                        grid_dst_pts=self._compute_grid_dst(H_cand) if self.wrap_mode in ("mesh_grid", "cylinder") else None,
                        inliers=n_inl,
                        confidence=conf,
                        track_ms=track_ms,
                        confidence_state="TRACKING",
                    )
            # First valid candidate after loss -> hold in recovering (overlay remains suppressed)
            self._state = "recovering"
            self._cand_H = H_cand
            track_ms = (time.perf_counter() - t0) * 1000.0
            return TrackFrameResult(
                frame_index=int(frame_index),
                timestamp_s=float(timestamp_s),
                state="recovering",
                inliers=n_inl,
                confidence=0.0,
                track_ms=track_ms,
                confidence_state="RECOVERING",
            )

        self._state = "lost"
        self._cand_H = None
        self._lost_streak += 1
        track_ms = (time.perf_counter() - t0) * 1000.0
        return TrackFrameResult(
            frame_index=int(frame_index),
            timestamp_s=float(timestamp_s),
            state="lost",
            inliers=n_inl,
            confidence=0.0,
            track_ms=track_ms,
            confidence_state="LOST",
        )

    def _solve_pnp_robust(
        self,
        pts3d: np.ndarray,
        pts2d_native: np.ndarray,
        prior: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray, float]:
        assert self._K is not None
        if len(pts3d) < 8:
            return False, None, None, np.zeros(0, dtype=bool), 999.0
        obj = pts3d.astype(np.float64)
        img = pts2d_native.astype(np.float64)
        reproj_thresh = max(3.5, 3.5 * float(self._frame_h) / 480.0)

        proposals: list[tuple[np.ndarray, np.ndarray]] = []
        cv2.setRNGSeed(17)
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                obj, img, self._K, None,
                iterationsCount=120,
                reprojectionError=reproj_thresh,
                confidence=0.995,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if ok and inl is not None and len(inl) >= 6:
                proposals.append((rvec.reshape(3), tvec.reshape(3)))
        except cv2.error:
            pass

        if prior is not None:
            pr, pt = prior[0].reshape(3, 1).copy(), prior[1].reshape(3, 1).copy()
            try:
                ok_it, r_it, t_it = cv2.solvePnP(
                    obj, img, self._K, None, pr, pt, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if ok_it:
                    proposals.append((r_it.reshape(3), t_it.reshape(3)))
            except cv2.error:
                pass

        best_res: tuple[bool, int, float, np.ndarray, np.ndarray, np.ndarray] | None = None
        R_prior = None
        tz_prior = None
        if prior is not None:
            R_prior, _ = cv2.Rodrigues(prior[0].astype(np.float64).reshape(3))
            tz_prior = float(prior[1].reshape(3)[2])

        for rvec, tvec in proposals:
            if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))) or tvec[2] <= 0.04:
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, self._K, None)
            errs = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
            inl_bool = np.isfinite(errs) & (errs <= reproj_thresh)
            n_inl = int(np.count_nonzero(inl_bool))
            if n_inl < 8:
                continue
            try:
                r_ref, t_ref = cv2.solvePnPRefineLM(
                    obj[inl_bool], img[inl_bool], self._K, None,
                    rvec.reshape(3, 1), tvec.reshape(3, 1),
                )
                rvec = r_ref.reshape(3)
                tvec = t_ref.reshape(3)
            except cv2.error:
                pass
            if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))) or tvec[2] <= 0.04:
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, self._K, None)
            errs = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
            inl_bool = np.isfinite(errs) & (errs <= reproj_thresh)
            n_inl = int(np.count_nonzero(inl_bool))
            if n_inl < 8:
                continue
            rmse = float(math.sqrt(float(np.mean(errs[inl_bool] ** 2))))
            kin_ok = True
            if R_prior is not None and tz_prior is not None and tz_prior > 0.04:
                R_c, _ = cv2.Rodrigues(rvec)
                d_ang = math.degrees(math.acos(float(np.clip((np.trace(R_c @ R_prior.T) - 1.0) * 0.5, -1.0, 1.0))))
                z_ratio = float(tvec[2]) / tz_prior
                kin_ok = (d_ang <= 32.0) and (0.62 <= z_ratio <= 1.60)
            if best_res is None:
                best_res = (kin_ok, n_inl, rmse, rvec, tvec, inl_bool)
            else:
                b_kin, b_inl, b_rmse = best_res[0], best_res[1], best_res[2]
                if (kin_ok and not b_kin) or (
                    kin_ok == b_kin and ((n_inl > b_inl) or (n_inl == b_inl and rmse < b_rmse))
                ):
                    best_res = (kin_ok, n_inl, rmse, rvec, tvec, inl_bool)

        if best_res is None:
            return False, None, None, np.zeros(len(pts3d), dtype=bool), 999.0
        _, _, rmse, rvec, tvec, inl_bool = best_res
        return True, rvec, tvec, inl_bool, rmse

    def _eval_6dof_edge_support(
        self,
        gray_track: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute silhouette boundary edge-alignment support in [0, 1] and native depth buffer."""
        assert self.mesh is not None and self._K is not None
        R, _ = cv2.Rodrigues(rvec.astype(np.float64))
        verts_cam = (self.mesh.vertices.astype(np.float64) @ R.T + tvec.reshape(1, 3)).astype(np.float32)
        fx, fy, cx, cy = float(self._K[0, 0]), float(self._K[1, 1]), float(self._K[0, 2]), float(self._K[1, 2])
        dummy_rgb = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
        dummy_tex = np.full((4, 4, 4), 255, dtype=np.uint8)
        res = rasterize_mesh_rgbd(
            verts_cam, self.mesh.triangles, self.mesh.uvs, dummy_tex, (fx, fy, cx, cy), dummy_rgb,
            opacity=0.0, blend_mode="normal",
        )
        sil_u8 = (res.depth_buf < 1e20).astype(np.uint8) * 255
        sil_track = (
            cv2.resize(sil_u8, (self._track_w, self._track_h), interpolation=cv2.INTER_NEAREST)
            if (self._track_w != self._frame_w or self._track_h != self._frame_h)
            else sil_u8
        )
        contours, _ = cv2.findContours(sil_track, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0, res.depth_buf
        largest = max(contours, key=cv2.contourArea).reshape(-1, 2)
        if len(largest) < 16:
            return 0.0, res.depth_buf
        idxs = np.linspace(0, len(largest) - 1, min(72, len(largest)), dtype=np.int32)
        pts = largest[idxs].astype(np.float32)
        # Mask normal via Sobel on sil_track
        gx_m = cv2.Sobel(sil_track, cv2.CV_32F, 1, 0, ksize=3)
        gy_m = cv2.Sobel(sil_track, cv2.CV_32F, 0, 1, ksize=3)
        px_i = np.clip(pts[:, 0].astype(np.int32), 0, self._track_w - 1)
        py_i = np.clip(pts[:, 1].astype(np.int32), 0, self._track_h - 1)
        nrm = -np.column_stack([gx_m[py_i, px_i], gy_m[py_i, px_i]])
        nmag = np.linalg.norm(nrm, axis=1)
        ok_n = nmag > 1e-3
        if int(np.count_nonzero(ok_n)) < 12:
            return 0.5, res.depth_buf
        pts = pts[ok_n]
        nrm = nrm[ok_n] / nmag[ok_n, None]

        # ROI-cropped image gradient along normal within +/-3 tracking pixels (10x faster on high-res frames)
        pad_r = 12
        rx0 = max(0, int(np.floor(pts[:, 0].min())) - pad_r)
        ry0 = max(0, int(np.floor(pts[:, 1].min())) - pad_r)
        rx1 = min(self._track_w, int(np.ceil(pts[:, 0].max())) + pad_r + 1)
        ry1 = min(self._track_h, int(np.ceil(pts[:, 1].max())) + pad_r + 1)
        sub_g = gray_track[ry0:ry1, rx0:rx1]
        if sub_g.size == 0:
            return 0.0, res.depth_buf
        smooth = cv2.GaussianBlur(sub_g, (5, 5), 0.9)
        ix = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        iy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        sub_h, sub_w = sub_g.shape[:2]
        pts_loc = pts - np.array([[float(rx0), float(ry0)]], dtype=np.float32)
        steps = np.arange(-3, 4, dtype=np.float32)
        samp = pts_loc[:, None, :] + nrm[:, None, :] * steps[None, :, None]
        sx = np.clip(np.round(samp[:, :, 0]).astype(np.int32), 0, sub_w - 1)
        sy = np.clip(np.round(samp[:, :, 1]).astype(np.int32), 0, sub_h - 1)
        grad = np.stack([ix[sy, sx], iy[sy, sx]], axis=-1)
        gmag = np.linalg.norm(grad, axis=-1)
        align = np.abs(np.sum(grad * nrm[:, None, :], axis=-1)) / np.maximum(gmag, 1.0)
        supported = np.any((gmag > 22.0) & (align > 0.58), axis=1)
        return float(np.mean(supported)), res.depth_buf

    def _refine_6dof_pose_silhouette(
        self,
        gray_track: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        depth_buf: np.ndarray,
        edge_sup: float,
        pts3d_inl: np.ndarray,
        pts2d_native_inl: np.ndarray,
        base_rmse: float,
        res_scale: float,
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        """Fast 1-step IRLS Huber silhouette-edge alignment for (tx, ty, tz), strictly verified against KLT RMSE."""
        assert self.mesh is not None and self._K is not None
        sil_u8 = (depth_buf < 1e20).astype(np.uint8) * 255
        sil_track = (
            cv2.resize(sil_u8, (self._track_w, self._track_h), interpolation=cv2.INTER_NEAREST)
            if (self._track_w != self._frame_w or self._track_h != self._frame_h)
            else sil_u8
        )
        contours, _ = cv2.findContours(sil_track, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return tvec, edge_sup, depth_buf, base_rmse
        largest = max(contours, key=cv2.contourArea).reshape(-1, 2)
        if len(largest) < 24:
            return tvec, edge_sup, depth_buf, base_rmse
        idxs = np.linspace(0, len(largest) - 1, min(64, len(largest)), dtype=np.int32)
        pts = largest[idxs].astype(np.float32)

        # Exclude contour samples touching detected gripping hands/occluders so fingers never pull the pose
        if self._last_occluder_mask_img is not None and self._last_occluder_mask_img.shape == (self._frame_h, self._frame_w):
            p_nat = self._track_to_native(pts)
            iu = np.clip(np.round(p_nat[:, 0]).astype(np.int32), 0, self._frame_w - 1)
            iv = np.clip(np.round(p_nat[:, 1]).astype(np.int32), 0, self._frame_h - 1)
            keep_c = self._last_occluder_mask_img[iv, iu] == 0
            if int(np.count_nonzero(keep_c)) < 16:
                return tvec, edge_sup, depth_buf, base_rmse
            pts = pts[keep_c]

        gx_m = cv2.Sobel(sil_track, cv2.CV_32F, 1, 0, ksize=3)
        gy_m = cv2.Sobel(sil_track, cv2.CV_32F, 0, 1, ksize=3)
        px_i = np.clip(pts[:, 0].astype(np.int32), 0, self._track_w - 1)
        py_i = np.clip(pts[:, 1].astype(np.int32), 0, self._track_h - 1)
        nrm = -np.column_stack([gx_m[py_i, px_i], gy_m[py_i, px_i]])
        nmag = np.linalg.norm(nrm, axis=1)
        ok_n = nmag > 1e-3
        if int(np.count_nonzero(ok_n)) < 16:
            return tvec, edge_sup, depth_buf, base_rmse
        pts = pts[ok_n]
        nrm = nrm[ok_n] / nmag[ok_n, None]

        pad_r = 14
        rx0 = max(0, int(np.floor(pts[:, 0].min())) - pad_r)
        ry0 = max(0, int(np.floor(pts[:, 1].min())) - pad_r)
        rx1 = min(self._track_w, int(np.ceil(pts[:, 0].max())) + pad_r + 1)
        ry1 = min(self._track_h, int(np.ceil(pts[:, 1].max())) + pad_r + 1)
        sub_g = gray_track[ry0:ry1, rx0:rx1]
        if sub_g.size == 0:
            return tvec, edge_sup, depth_buf, base_rmse
        smooth = cv2.GaussianBlur(sub_g, (5, 5), 0.9)
        ix = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        iy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        sub_h, sub_w = sub_g.shape[:2]
        pts_loc = pts - np.array([[float(rx0), float(ry0)]], dtype=np.float32)

        steps = np.arange(-6, 7, dtype=np.float32)
        samp = pts_loc[:, None, :] + nrm[:, None, :] * steps[None, :, None]
        sx = np.clip(np.round(samp[:, :, 0]).astype(np.int32), 0, sub_w - 1)
        sy = np.clip(np.round(samp[:, :, 1]).astype(np.int32), 0, sub_h - 1)
        grad = np.stack([ix[sy, sx], iy[sy, sx]], axis=-1)
        gmag = np.linalg.norm(grad, axis=-1)
        align = np.abs(np.sum(grad * nrm[:, None, :], axis=-1)) / np.maximum(gmag, 1.0)
        valid_e = (gmag > 22.0) & (align > 0.60)
        # Prefer strong aligned edges close to current silhouette
        score = np.where(valid_e, gmag * align / (1.0 + 0.14 * np.abs(steps)[None, :]), 0.0)
        best_idx = np.argmax(score, axis=1)
        has_e = score[np.arange(len(pts)), best_idx] > 0.0
        if int(np.count_nonzero(has_e)) < 18:
            return tvec, edge_sup, depth_buf, base_rmse

        d_obs = steps[best_idx[has_e]].astype(np.float64)
        med_abs_d = float(np.median(np.abs(d_obs)))
        if med_abs_d < 0.80 or med_abs_d > 4.5:
            return tvec, edge_sup, depth_buf, base_rmse

        p_e = pts[has_e].astype(np.float64)
        n_e = nrm[has_e].astype(np.float64)
        ctr = p_e.mean(axis=0)
        rad_n = np.sum((p_e - ctr[None, :]) * n_e, axis=1)
        A = np.column_stack([n_e[:, 0], n_e[:, 1], rad_n])

        # 2-iteration IRLS Huber least squares for (dx_track, dy_track, d_scale)
        w = np.ones(len(d_obs), dtype=np.float64)
        sol = np.zeros(3, dtype=np.float64)
        for _ in range(2):
            Aw = A * w[:, None]
            H_sys = Aw.T @ A + np.diag([1e-3, 1e-3, 4.0])  # mild scale damping
            try:
                sol = np.linalg.solve(H_sys, Aw.T @ d_obs)
            except np.linalg.LinAlgError:
                return tvec, edge_sup, depth_buf, base_rmse
            res_r = np.abs(A @ sol - d_obs)
            w = np.where(res_r <= 1.8, 1.0, 1.8 / np.maximum(res_r, 1e-6))

        dx_tr = float(np.clip(sol[0], -3.0, 3.0))
        dy_tr = float(np.clip(sol[1], -3.0, 3.0))
        ds = float(np.clip(sol[2], -0.035, 0.035))
        if abs(dx_tr) < 0.35 and abs(dy_tr) < 0.35 and abs(ds) < 0.008:
            return tvec, edge_sup, depth_buf, base_rmse

        fx, fy = float(self._K[0, 0]), float(self._K[1, 1])
        tz_old = float(tvec[2])
        tz_new = float(np.clip(tz_old / (1.0 + ds), max(0.06, 0.55 * self._init_tz), 1.65 * self._init_tz))
        tvec_new = tvec.copy()
        tvec_new[0] = (tvec[0] + (dx_tr / max(self._sx, 1e-4)) * tz_old / fx) * (tz_new / tz_old)
        tvec_new[1] = (tvec[1] + (dy_tr / max(self._sy, 1e-4)) * tz_old / fy) * (tz_new / tz_old)
        tvec_new[2] = tz_new

        # Verify KLT 3D-to-2D reprojection consistency before accepting!
        if len(pts3d_inl) >= 8:
            proj, _ = cv2.projectPoints(pts3d_inl, rvec, tvec_new, self._K, None)
            errs = np.linalg.norm(proj.reshape(-1, 2) - pts2d_native_inl, axis=1)
            new_rmse = float(math.sqrt(float(np.mean(errs ** 2))))
            if new_rmse > base_rmse + 0.35 * res_scale:
                return tvec, edge_sup, depth_buf, base_rmse
        else:
            new_rmse = base_rmse

        new_es, new_dbuf = self._eval_6dof_edge_support(gray_track, rvec, tvec_new)
        if new_es >= edge_sup + 0.05:
            return tvec_new, new_es, new_dbuf, new_rmse
        return tvec, edge_sup, depth_buf, base_rmse

    def _eval_6dof_bg_empty_mask(
        self,
        frame_bgr: np.ndarray,
        depth_buf: np.ndarray,
    ) -> tuple[float, np.ndarray | None]:
        """Return fraction and uint8 mask of 3D silhouette pixels matching the immediate exterior background color (and not target color)."""
        lut = getattr(self, "_ref_6dof_lab_lut", None)
        if lut is None:
            return 0.0, None
        sil = depth_buf < 1e20
        ys, xs = np.nonzero(sil)
        n_sil = len(xs)
        if n_sil < 64:
            return 0.0, None
        h, w = frame_bgr.shape[:2]
        pad = 22
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(w, int(xs.max()) + pad + 1)
        y1 = min(h, int(ys.max()) + pad + 1)

        roi_bgr = frame_bgr[y0:y1:2, x0:x1:2]
        roi_sil = sil[y0:y1:2, x0:x1:2]
        n_sil_sub = int(np.count_nonzero(roi_sil))
        if n_sil_sub < 24:
            return 0.0, None
        sil_u8 = roi_sil.astype(np.uint8) * 255
        ext_ring = (cv2.dilate(sil_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0) & (~roi_sil)
        if int(np.count_nonzero(ext_ring)) < 16:
            return 0.0, None

        roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        li = np.clip(roi_lab[:, :, 0] // 20, 0, 13)
        ai = np.clip(roi_lab[:, :, 1] // 8, 0, 32)
        bi = np.clip(roi_lab[:, :, 2] // 8, 0, 32)
        in_target = lut[li, ai, bi]

        bg_pts = roi_lab[ext_ring]
        bg_cnt = np.zeros((14, 33, 33), dtype=np.int32)
        np.add.at(
            bg_cnt,
            (
                np.clip(bg_pts[:, 0] // 20, 0, 13),
                np.clip(bg_pts[:, 1] // 8, 0, 32),
                np.clip(bg_pts[:, 2] // 8, 0, 32),
            ),
            1,
        )
        bg_lut = bg_cnt >= max(6, int(0.03 * len(bg_pts)))
        in_bg = bg_lut[li, ai, bi]

        # Only count exterior-connected background regions leaking inside the silhouette
        cand = (~roi_sil) | (roi_sil & (~in_target) & in_bg)
        n_cc, lbls, _, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=4)
        bg_in_sil = np.zeros(roi_sil.shape, dtype=bool)
        if n_cc > 1:
            flat = lbls.ravel()
            ext_c = np.bincount(flat, weights=(~roi_sil).ravel().astype(np.float32), minlength=n_cc)
            in_c = np.bincount(flat, weights=roi_sil.ravel().astype(np.float32), minlength=n_cc)
            keep = np.zeros(n_cc, dtype=bool)
            for c_id in range(1, n_cc):
                if ext_c[c_id] >= 12 and in_c[c_id] >= 6:
                    keep[c_id] = True
            if np.any(keep):
                bg_in_sil = keep[lbls] & roi_sil

        n_bg_in = int(np.count_nonzero(bg_in_sil))
        if n_bg_in == 0:
            return 0.0, None
        frac = float(n_bg_in) / float(n_sil_sub)
        bg_up = cv2.resize(
            np.where(bg_in_sil, np.uint8(255), np.uint8(0)),
            (x1 - x0, y1 - y0),
            interpolation=cv2.INTER_NEAREST,
        )
        full_m = np.zeros((h, w), dtype=np.uint8)
        full_m[y0:y1, x0:x1] = bg_up
        return frac, full_m

    def _replenish_6dof_points(
        self,
        gray_track: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        *,
        depth_buf: np.ndarray | None = None,
        bg_exclude_mask: np.ndarray | None = None,
        external_fg_mask: np.ndarray | None = None,
    ) -> None:
        """Project current mesh silhouette, exclude detected hand/background pixels, and lift interior corners to 3D."""
        assert self.mesh is not None and self._K is not None
        R, _ = cv2.Rodrigues(rvec)
        fx, fy, cx, cy = float(self._K[0, 0]), float(self._K[1, 1]), float(self._K[0, 2]), float(self._K[1, 2])
        if depth_buf is None:
            verts_cam = (self.mesh.vertices.astype(np.float64) @ R.T + tvec.reshape(1, 3)).astype(np.float32)
            dummy_rgb = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
            dummy_tex = np.full((4, 4, 4), 255, dtype=np.uint8)
            res = rasterize_mesh_rgbd(
                verts_cam, self.mesh.triangles, self.mesh.uvs, dummy_tex, (fx, fy, cx, cy), dummy_rgb,
                opacity=0.0, blend_mode="normal",
            )
            depth_buf = res.depth_buf
        sil_u8 = (depth_buf < 1e20).astype(np.uint8) * 255
        if external_fg_mask is not None and external_fg_mask.shape == sil_u8.shape:
            sil_u8 = cv2.bitwise_and(sil_u8, self._erode_vos_mask(external_fg_mask))
        else:
            # Strictly exclude detected foreground/hand occluder pixels so hands NEVER contaminate 3D points
            if self._last_occluder_mask_img is not None and self._last_occluder_mask_img.shape == sil_u8.shape:
                occ_dil = cv2.dilate(self._last_occluder_mask_img, np.ones((7, 7), np.uint8), iterations=1)
                sil_u8 = cv2.bitwise_and(sil_u8, cv2.bitwise_not(occ_dil))
            # Also exclude exterior-connected background overhang so empty tray/wall pixels NEVER get lifted to 3D
            if bg_exclude_mask is not None and bg_exclude_mask.shape == sil_u8.shape:
                bg_dil = cv2.dilate(bg_exclude_mask, np.ones((5, 5), np.uint8), iterations=1)
                sil_u8 = cv2.bitwise_and(sil_u8, cv2.bitwise_not(bg_dil))
        sil_track = (
            cv2.resize(sil_u8, (self._track_w, self._track_h), interpolation=cv2.INTER_NEAREST)
            if (self._track_w != self._frame_w or self._track_h != self._frame_h)
            else sil_u8
        )
        sil_track = cv2.erode(sil_track, np.ones((3, 3), np.uint8), iterations=1) if external_fg_mask is not None else cv2.erode(sil_track, np.ones((5, 5), np.uint8), iterations=1)
        if int(np.count_nonzero(sil_track)) < 48:
            return
        new_track = self._detect_corners_in_mask(gray_track, sil_track)
        if len(new_track) == 0:
            return
        new_native = self._track_to_native(new_track)
        iu = np.clip(np.round(new_native[:, 0]).astype(np.int32), 0, self._frame_w - 1)
        iv = np.clip(np.round(new_native[:, 1]).astype(np.int32), 0, self._frame_h - 1)
        z = depth_buf[iv, iu].astype(np.float64)
        valid = np.isfinite(z) & (z > 0.03) & (z < 100.0)
        if not np.any(valid):
            return
        uv_ok = new_native[valid].astype(np.float64)
        z_ok = z[valid]
        p_cam = np.column_stack([(uv_ok[:, 0] - cx) * z_ok / fx, (uv_ok[:, 1] - cy) * z_ok / fy, z_ok])
        p_obj = (p_cam - tvec.reshape(1, 3)) @ R
        add_track = new_track[valid]

        if len(self._curr_pts_track) > 0:
            d2 = np.sum((add_track[:, None, :] - self._curr_pts_track[None, :, :]) ** 2, axis=2)
            far = d2.min(axis=1) > 16.0
            add_track = add_track[far]
            p_obj = p_obj[far]
            if len(add_track) == 0:
                return
            self._curr_pts_track = np.vstack([self._curr_pts_track, add_track])[: self.max_klt_points].astype(np.float32)
            self._curr_pts3d = np.vstack([self._curr_pts3d, p_obj])[: self.max_klt_points]
        else:
            self._curr_pts_track = add_track[: self.max_klt_points].astype(np.float32)
            self._curr_pts3d = p_obj[: self.max_klt_points]

    def _maybe_store_6dof_keyframe(
        self,
        frame_bgr: np.ndarray,
        gray_track: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        depth_buf: np.ndarray,
    ) -> None:
        """Insert a new verified 3D viewpoint into the multi-keyframe bank ONLY when pose & silhouette are independently verified."""
        if len(self._mesh_keyframes) >= 12 or self._K is None:
            return
        R_cur, _ = cv2.Rodrigues(rvec.astype(np.float64))
        for kf_r, _, _, _ in self._mesh_keyframes:
            R_kf, _ = cv2.Rodrigues(kf_r.astype(np.float64))
            ang = math.acos(float(np.clip((np.trace(R_cur @ R_kf.T) - 1.0) * 0.5, -1.0, 1.0)))
            if math.degrees(ang) < 16.0:
                return  # Already have a verified keyframe near this viewpoint

        sil_u8 = (depth_buf < 1e20).astype(np.uint8) * 255
        if self._last_occluder_mask_img is not None and self._last_occluder_mask_img.shape == sil_u8.shape:
            occ_dil = cv2.dilate(self._last_occluder_mask_img, np.ones((7, 7), np.uint8), iterations=1)
            sil_u8 = cv2.bitwise_and(sil_u8, cv2.bitwise_not(occ_dil))
        sil_u8 = cv2.erode(sil_u8, np.ones((5, 5), np.uint8), iterations=1)

        if self._use_native_recovery:
            gray_src = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            kps, descs = self._orb_recovery.detectAndCompute(gray_src, sil_u8)
            if descs is None or len(kps) < 16:
                return
            pts_native = np.array([kp.pt for kp in kps], dtype=np.float32)
        else:
            sil_tr = (
                cv2.resize(sil_u8, (self._track_w, self._track_h), interpolation=cv2.INTER_NEAREST)
                if (self._track_w != self._frame_w or self._track_h != self._frame_h)
                else sil_u8
            )
            kps, descs = self._orb_recovery.detectAndCompute(gray_track, sil_tr)
            if descs is None or len(kps) < 16:
                return
            pts_native = self._track_to_native(np.array([kp.pt for kp in kps], dtype=np.float32))

        fx, fy, cx, cy = float(self._K[0, 0]), float(self._K[1, 1]), float(self._K[0, 2]), float(self._K[1, 2])
        iu = np.clip(np.round(pts_native[:, 0]).astype(np.int32), 0, self._frame_w - 1)
        iv = np.clip(np.round(pts_native[:, 1]).astype(np.int32), 0, self._frame_h - 1)
        z = depth_buf[iv, iu].astype(np.float64)
        valid = np.isfinite(z) & (z > 0.03) & (z < 100.0)
        if int(np.count_nonzero(valid)) < 16:
            return
        uv_ok = pts_native[valid].astype(np.float64)
        z_ok = z[valid]
        p_cam = np.column_stack([(uv_ok[:, 0] - cx) * z_ok / fx, (uv_ok[:, 1] - cy) * z_ok / fy, z_ok])
        p_obj = (p_cam - tvec.reshape(1, 3)) @ R_cur
        self._mesh_keyframes.append((rvec.copy(), tvec.copy(), p_obj[:600], descs[valid][:600].copy()))

    def _track_frame_6dof(
        self,
        frame_bgr: np.ndarray,
        gray_track: np.ndarray,
        frame_index: int,
        timestamp_s: float,
        t0: float,
        *,
        external_fg_mask: np.ndarray | None = None,
    ) -> TrackFrameResult:
        assert self._K is not None and self.mesh is not None
        use_vos = external_fg_mask is not None and external_fg_mask.shape == (self._frame_h, self._frame_w)
        vos_fg_px = int(np.count_nonzero(external_fg_mask > 127)) if (use_vos and external_fg_mask is not None) else 0

        if use_vos and vos_fg_px < 48:
            self._state = "lost"
            self._last_rvec = None
            self._last_tvec = None
            self._cand_pose = None
            self._vos_coast_frames = 0
            self._curr_pts_track = np.zeros((0, 2), dtype=np.float32)
            self._curr_pts3d = np.zeros((0, 3), dtype=np.float64)
            self._prev_gray = gray_track
            track_ms = (time.perf_counter() - t0) * 1000.0
            return TrackFrameResult(
                frame_index=int(frame_index),
                timestamp_s=float(timestamp_s),
                state="lost",
                inliers=0,
                confidence=0.0,
                track_ms=track_ms,
                confidence_state="LOST",
            )

        vos_inner = self._erode_vos_mask(external_fg_mask) if (use_vos and external_fg_mask is not None) else None

        if self._state == "tracking":
            p1_track, good = self._run_fb_klt(gray_track, self._curr_pts_track)
            prev_track = self._curr_pts_track[good]
            cur_track = p1_track[good]
            cur_pts3d = self._curr_pts3d[good]
            cur_native = self._track_to_native(cur_track)

            # Strictly gate KLT points inside live VOS mask (or fallback heuristic occluder mask) BEFORE PnP
            if vos_inner is not None and len(cur_native) > 0:
                iu = np.clip(np.round(cur_native[:, 0]).astype(np.int32), 0, self._frame_w - 1)
                iv = np.clip(np.round(cur_native[:, 1]).astype(np.int32), 0, self._frame_h - 1)
                on_obj = vos_inner[iv, iu] > 127
                prev_track = prev_track[on_obj]
                cur_track = cur_track[on_obj]
                cur_pts3d = cur_pts3d[on_obj]
                cur_native = cur_native[on_obj]
            elif (
                self._last_occluder_mask_img is not None
                and len(cur_native) >= 16
                and self._last_occluder_mask_img.shape == (self._frame_h, self._frame_w)
            ):
                iu = np.clip(np.round(cur_native[:, 0]).astype(np.int32), 0, self._frame_w - 1)
                iv = np.clip(np.round(cur_native[:, 1]).astype(np.int32), 0, self._frame_h - 1)
                unocc = self._last_occluder_mask_img[iv, iu] == 0
                if int(np.count_nonzero(unocc)) >= 12:
                    prev_track = prev_track[unocc]
                    cur_track = cur_track[unocc]
                    cur_pts3d = cur_pts3d[unocc]
                    cur_native = cur_native[unocc]

            pnp_3d = cur_pts3d
            pnp_2d = cur_native
            res_scale = max(1.0, float(self._frame_h) / 480.0)
            kf0_snapped = False
            ok = False
            rvec = None
            tvec = None
            inl_bool = np.zeros((0,), dtype=bool)
            rmse = 999.0

            # Independent Canonical Keyframe-0 Loop Closure: never let drifted KLT drown out frame-0 XFeat lock!
            kf0_ran = False
            if (
                self.matcher_backend == "xfeat"
                and self._xfeat_mesh_keyframes
                and (frame_index % 3 == 0 or len(pnp_3d) < 28 or self._vos_coast_frames > 0)
            ):
                kf0_ran = True
                kf0_res = self._try_xfeat_kf0_loop_closure(
                    frame_bgr,
                    vos_inner if use_vos else None,
                    external_fg_mask if use_vos else None,
                    res_scale,
                )
                if kf0_res is not None:
                    kf_r, kf_t, kf_3d, kf_2d, kf_rmse, kf_n_inl = kf0_res
                    # Check if KF0 strongly locks onto canonical front face (even across >32 deg backside slip!)
                    if kf_n_inl >= 12 or (kf_n_inl >= 10 and (len(pnp_3d) < 24 or self._vos_coast_frames > 0)):
                        rvec, tvec = kf_r, kf_t
                        pnp_3d, pnp_2d = kf_3d, kf_2d
                        inl_bool = np.ones((len(kf_3d),), dtype=bool)
                        rmse = kf_rmse
                        ok = True
                        kf0_snapped = True

            if not kf0_snapped:
                # Periodic or low-support XFeat multi-keyframe drift-anchoring when away from frame-0 face
                if (
                    not kf0_ran
                    and self.matcher_backend == "xfeat"
                    and self._xfeat_mesh_keyframes
                    and (frame_index % 8 == 3 or len(pnp_3d) < 24)
                ):
                    xf_3d, xf_2d = self._match_xfeat_6dof(frame_bgr, vos_inner)
                    if len(xf_3d) >= 6:
                        pnp_3d = np.vstack([pnp_3d, xf_3d]) if len(pnp_3d) else xf_3d
                        pnp_2d = np.vstack([pnp_2d, xf_2d]) if len(pnp_2d) else xf_2d

                prior = (self._last_rvec, self._last_tvec) if (self._last_rvec is not None and self._last_tvec is not None) else None
                ok, rvec, tvec, inl_bool, rmse = self._solve_pnp_robust(pnp_3d, pnp_2d, prior)

            # Causal symmetry disambiguation on early frames (frame 1..3) if alternate silhouette hypotheses exist
            if self._alt_hypotheses and frame_index <= 3:
                best_rmse = rmse if ok else 999.0
                best_choice: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None = None
                surviving_alts = []
                for alt_r, alt_t, alt_trk, alt_3d in self._alt_hypotheses:
                    ap1, ag = self._run_fb_klt(gray_track, alt_trk)
                    if int(np.count_nonzero(ag)) >= 8:
                        a_nat = self._track_to_native(ap1[ag])
                        aok, ar, at, ainl, armse = self._solve_pnp_robust(alt_3d[ag], a_nat, (alt_r, alt_t))
                        if aok and ar is not None and at is not None:
                            surviving_alts.append((ar, at, ap1[ag][ainl], alt_3d[ag][ainl]))
                            if armse + 0.25 < best_rmse and int(np.count_nonzero(ainl)) >= 12:
                                best_rmse = armse
                                best_choice = (ar, at, a_nat, alt_3d[ag], ainl, armse)
                if best_choice is not None:
                    rvec, tvec, pnp_2d, pnp_3d, inl_bool, rmse = best_choice
                    ok = True
                self._alt_hypotheses = surviving_alts if frame_index < 3 else []

            # Kinematic jump & implausible depth-drift guard (allow canonical KF0 loop-closure snap!)
            if ok and rvec is not None and tvec is not None:
                if tvec[2] < max(0.06, 0.50 * self._init_tz) or tvec[2] > 1.75 * self._init_tz:
                    ok = False
                elif (not kf0_snapped) and self._last_rvec is not None and self._last_tvec is not None:
                    R_new, _ = cv2.Rodrigues(rvec)
                    R_old, _ = cv2.Rodrigues(self._last_rvec)
                    d_ang = math.degrees(math.acos(float(np.clip((np.trace(R_new @ R_old.T) - 1.0) * 0.5, -1.0, 1.0))))
                    d_tr = float(np.linalg.norm(tvec - self._last_tvec))
                    if d_ang > 32.0 or d_tr > max(0.20, 0.45 * self._init_tz):
                        ok = False

            if ok and rvec is not None and tvec is not None and rmse <= 6.0 * res_scale:
                n_inl = int(np.count_nonzero(inl_bool))
                inl_native = pnp_2d[inl_bool]
                inl_pts3d = pnp_3d[inl_bool]

                vos_snapped_6dof = False
                if use_vos and external_fg_mask is not None:
                    vos_iou, vos_cov, depth_buf = self._eval_6dof_vos_iou(rvec, tvec, external_fg_mask)
                    if vos_iou < 0.86:
                        pre_iou = vos_iou
                        tvec, vos_iou, vos_cov, depth_buf, rmse = self._refine_6dof_pose_to_vos(
                            rvec,
                            tvec,
                            depth_buf,
                            vos_iou,
                            vos_cov,
                            external_fg_mask,
                            inl_pts3d,
                            inl_native,
                            rmse,
                            res_scale,
                            coasting=False,
                        )
                        if vos_iou >= pre_iou + 0.05:
                            vos_snapped_6dof = True
                        if vos_iou < 0.80:
                            try:
                                rv2, tv2, i2, c2, r2 = self._refine_6dof_cylinder_rot_to_vos(
                                    rvec, tvec, vos_iou, vos_cov, external_fg_mask,
                                    inl_pts3d, inl_native, rmse, res_scale, coasting=False,
                                )
                                if i2 >= vos_iou + 0.03:
                                    rvec, tvec, vos_iou, vos_cov, rmse = rv2, tv2, i2, c2, r2
                                    _, _, depth_buf = self._eval_6dof_vos_iou(rvec, tvec, external_fg_mask)
                                    vos_snapped_6dof = True
                            except Exception:
                                pass
                    edge_sup = vos_iou
                    bg_empty_frac = max(0.0, 1.0 - vos_iou)
                    bg_mask_img = None
                    xs_sil = np.nonzero(depth_buf < 1e20)[1] if depth_buf is not None else np.nonzero(external_fg_mask > 127)[1]
                    sil_w_native = float(xs_sil.max() - xs_sil.min()) if len(xs_sil) >= 16 else 32.0
                    xspan_native = float(inl_native[:, 0].max() - inl_native[:, 0].min()) if len(inl_native) >= 4 else 0.0
                    xspan_ratio = xspan_native / max(sil_w_native, 12.0)
                    tz_rel = float(tvec[2]) / max(self._init_tz, 1e-4)
                    is_ambiguous = (vos_iou < 0.45 and vos_cov < 0.62) or (tz_rel > 1.55 or tz_rel < 0.55)
                else:
                    edge_sup, depth_buf = self._eval_6dof_edge_support(gray_track, rvec, tvec)

                    # Measure 2D horizontal span of inliers relative to projected mesh width (detects grazing-strip PnP degeneracy)
                    xs_sil = np.nonzero(depth_buf < 1e20)[1]
                    sil_w_native = float(xs_sil.max() - xs_sil.min()) if len(xs_sil) >= 16 else 32.0
                    xspan_native = float(inl_native[:, 0].max() - inl_native[:, 0].min()) if len(inl_native) >= 4 else 0.0
                    xspan_ratio = xspan_native / max(sil_w_native, 12.0)

                    # 1-step IRLS Huber silhouette-edge refinement strictly bounded by KLT RMSE (+0.35*res_scale)
                    if edge_sup < 0.72:
                        tvec, edge_sup, depth_buf, rmse = self._refine_6dof_pose_silhouette(
                            gray_track,
                            rvec,
                            tvec,
                            depth_buf,
                            edge_sup,
                            inl_pts3d,
                            inl_native,
                            rmse,
                            res_scale,
                        )

                    # Guard against grazing-strip depth runaway ONLY when points collapsed onto a narrow limb (< 35% width)
                    if (
                        xspan_ratio < 0.35
                        and self._last_tvec is not None
                        and self._last_tvec[2] > 0.04
                    ):
                        prev_tz = float(self._last_tvec[2])
                        z_ratio = float(tvec[2]) / prev_tz
                        if z_ratio > 1.035 or z_ratio < 0.965:
                            clamped_tz = float(np.clip(tvec[2], prev_tz * 0.965, prev_tz * 1.035))
                            scale_z = clamped_tz / max(float(tvec[2]), 1e-4)
                            t_cand = tvec.copy()
                            t_cand[0] *= scale_z
                            t_cand[1] *= scale_z
                            t_cand[2] = clamped_tz
                            proj_c, _ = cv2.projectPoints(inl_pts3d, rvec, t_cand, self._K, None)
                            rmse_c = float(math.sqrt(float(np.mean(np.sum((proj_c.reshape(-1, 2) - inl_native) ** 2, axis=1)))))
                            if rmse_c <= max(rmse + 1.2 * res_scale, 4.0 * res_scale):
                                tvec = t_cand
                                rmse = rmse_c
                                edge_sup, depth_buf = self._eval_6dof_edge_support(gray_track, rvec, tvec)

                    # Explicit AMBIGUOUS gate & silhouette-consistent depth-scale regularization
                    bg_empty_frac, bg_mask_img = self._eval_6dof_bg_empty_mask(frame_bgr, depth_buf)
                    if bg_empty_frac > 0.12 and float(tvec[2]) < 0.96 * self._init_tz:
                        scale_z = min(1.045, (0.98 * self._init_tz) / max(float(tvec[2]), 1e-4))
                        if scale_z > 1.008:
                            t_cand = tvec.copy()
                            t_cand[0] *= scale_z
                            t_cand[1] *= scale_z
                            t_cand[2] *= scale_z
                            proj_c, _ = cv2.projectPoints(inl_pts3d, rvec, t_cand, self._K, None)
                            rmse_c = float(math.sqrt(float(np.mean(np.sum((proj_c.reshape(-1, 2) - inl_native) ** 2, axis=1)))))
                            if rmse_c <= max(rmse + 1.2 * res_scale, 4.5 * res_scale):
                                es_c, db_c = self._eval_6dof_edge_support(gray_track, rvec, t_cand)
                                bg_c, bgm_c = self._eval_6dof_bg_empty_mask(frame_bgr, db_c)
                                if bg_c < bg_empty_frac - 0.012:
                                    tvec = t_cand
                                    rmse = rmse_c
                                    edge_sup = es_c
                                    depth_buf = db_c
                                    bg_empty_frac = bg_c
                                    bg_mask_img = bgm_c
                    tz_rel = float(tvec[2]) / max(self._init_tz, 1e-4)
                    strong_geom = (n_inl >= 35) and (rmse <= 3.5 * res_scale) and (0.65 <= tz_rel <= 1.40)
                    bg_drifted = (bg_empty_frac > 0.52) or (bg_empty_frac > 0.42 and edge_sup < 0.45)
                    is_ambiguous = bg_drifted or (
                        (not strong_geom)
                        and (
                            (n_inl < 14 and edge_sup < 0.42)
                            or (rmse > 4.2 * res_scale and edge_sup < 0.40)
                            or (xspan_ratio < 0.24 and n_inl < 30)
                            or ((tz_rel > 1.42 or tz_rel < 0.62) and edge_sup < 0.62)
                        )
                    )

                can_vos_coast = (
                    use_vos
                    and external_fg_mask is not None
                    and self._last_rvec is not None
                    and self._last_tvec is not None
                    and vos_fg_px >= max(120, int(0.25 * self._init_vos_area))
                )
                if is_ambiguous and not can_vos_coast:
                    self._state = "recovering"
                    self._cand_pose = (rvec.copy(), tvec.copy())
                    self._prev_gray = gray_track
                    track_ms = (time.perf_counter() - t0) * 1000.0
                    return TrackFrameResult(
                        frame_index=int(frame_index),
                        timestamp_s=float(timestamp_s),
                        state="recovering",
                        inliers=n_inl,
                        confidence=0.0,
                        track_ms=track_ms,
                        confidence_state="AMBIGUOUS",
                    )

                if not is_ambiguous:
                    # Update damped angular velocity estimate for smooth featureless-backside coasting
                    if self._last_rvec is not None:
                        R_new_v, _ = cv2.Rodrigues(rvec)
                        R_old_v, _ = cv2.Rodrigues(self._last_rvec)
                        d_rvec, _ = cv2.Rodrigues(R_new_v @ R_old_v.T)
                        d_rvec = d_rvec.ravel()
                        if float(np.linalg.norm(d_rvec)) < math.radians(20.0):
                            self._last_ang_vel = 0.55 * self._last_ang_vel + 0.45 * d_rvec
                    was_coasting = self._vos_coast_frames > 0
                    self._vos_coast_frames = 0
                    self._last_rvec = rvec.copy()
                    self._last_tvec = tvec.copy()
                    self._last_valid_rvec = rvec.copy()
                    self._last_valid_tvec = tvec.copy()
                    if depth_buf is not None:
                        ys_sil, xs_sil_u = np.nonzero(depth_buf < 1e20)
                    elif use_vos and external_fg_mask is not None:
                        ys_sil, xs_sil_u = np.nonzero(external_fg_mask > 127)
                    else:
                        ys_sil, xs_sil_u = np.zeros((0,), dtype=np.intp), np.zeros((0,), dtype=np.intp)
                    if len(xs_sil_u) >= 16:
                        x0_s, x1_s = int(xs_sil_u.min()), int(xs_sil_u.max())
                        y0_s, y1_s = int(ys_sil.min()), int(ys_sil.max())
                        self._last_sil_bbox = (x0_s, y0_s, max(8, x1_s - x0_s), max(8, y1_s - y0_s))
                    self._curr_pts_track = self._native_to_track(inl_native)[: self.max_klt_points].astype(np.float32)
                    self._curr_pts3d = inl_pts3d[: self.max_klt_points].copy()
                    self._reproj_errors.append(rmse)
                    self._frames_since_replenish += 1

                    # Only replenish 3D points & store verified keyframes when pose, span, depth, and silhouette are independently solid
                    can_replenish = (
                        (use_vos and edge_sup >= 0.42 and rmse <= 4.0 * res_scale and (0.62 <= tz_rel <= 1.45))
                        or ((not use_vos) and edge_sup >= 0.38 and rmse <= 3.5 * res_scale and (0.68 <= tz_rel <= 1.38) and bg_empty_frac < 0.40)
                    )
                    need_replenish = can_replenish and (
                        kf0_snapped
                        or vos_snapped_6dof
                        or len(self._curr_pts_track) < int(self.max_klt_points * 0.55)
                        or self._frames_since_replenish >= 10
                    )
                    can_store_kf = (
                        (not was_coasting)
                        and (
                            (use_vos and edge_sup >= 0.55 and rmse <= 3.2 * res_scale and n_inl >= 18 and (0.68 <= tz_rel <= 1.38))
                            or ((not use_vos) and edge_sup >= 0.48 and rmse <= 2.6 * res_scale and n_inl >= 24 and xspan_ratio >= 0.40 and (0.72 <= tz_rel <= 1.35) and bg_empty_frac < 0.28)
                        )
                    )
                    if (need_replenish or can_store_kf) and depth_buf is None:
                        depth_buf = self._rasterize_full_depth(rvec, tvec)
                    if need_replenish:
                        self._replenish_6dof_points(
                            gray_track,
                            rvec,
                            tvec,
                            depth_buf=depth_buf,
                            bg_exclude_mask=bg_mask_img,
                            external_fg_mask=external_fg_mask if use_vos else None,
                        )
                        self._frames_since_replenish = 0
                    if can_store_kf and depth_buf is not None:
                        self._maybe_store_6dof_keyframe(frame_bgr, gray_track, rvec, tvec, depth_buf)
                        if self.matcher_backend == "xfeat":
                            self._maybe_store_xfeat_keyframe_6dof(frame_bgr, rvec, tvec, depth_buf, external_fg_mask if use_vos else None)

                    self._prev_gray = gray_track
                    conf = float(np.clip(0.65 * (n_inl / max(float(len(pnp_3d)), 1.0)) + 0.35 * edge_sup, 0.0, 1.0))
                    track_ms = (time.perf_counter() - t0) * 1000.0
                    return TrackFrameResult(
                        frame_index=int(frame_index),
                        timestamp_s=float(timestamp_s),
                        state="tracking",
                        pose_rvec=rvec.copy(),
                        pose_tvec=tvec.copy(),
                        inliers=n_inl,
                        confidence=conf,
                        track_ms=track_ms,
                        confidence_state="TRACKING",
                    )

            # Smooth VOS-locked coasting across featureless backsides (e.g. plain white mug / smooth can back)
            if (
                use_vos
                and external_fg_mask is not None
                and self._last_rvec is not None
                and self._last_tvec is not None
                and vos_fg_px >= max(120, int(0.25 * self._init_vos_area))
                and self._vos_coast_frames < 200
            ):
                R_prev_c, _ = cv2.Rodrigues(self._last_rvec)
                if self.mesh.source_kind == "parametric" and self.mesh.name == "cylinder":
                    # Coast primarily around the cylinder's own Y-axis spin so featureless backsides
                    # never integrate noisy out-of-plane pitch that tips an upright bottle flat!
                    cyl_axis = R_prev_c[:, 1]
                    spin_rate = float(np.dot(self._last_ang_vel, cyl_axis))
                    d_R, _ = cv2.Rodrigues(0.80 * spin_rate * cyl_axis)
                else:
                    d_R, _ = cv2.Rodrigues(0.80 * self._last_ang_vel)
                rvec_c, _ = cv2.Rodrigues(d_R @ R_prev_c)
                rvec_c = rvec_c.ravel()
                self._last_ang_vel *= 0.88
                c_iou, c_cov, c_db = self._eval_6dof_vos_iou(rvec_c, self._last_tvec, external_fg_mask)
                tvec_c, c_iou, c_cov, c_db, _ = self._refine_6dof_pose_to_vos(
                    rvec_c,
                    self._last_tvec,
                    c_db,
                    c_iou,
                    c_cov,
                    external_fg_mask,
                    np.zeros((0, 3), dtype=np.float64),
                    np.zeros((0, 2), dtype=np.float32),
                    0.0,
                    res_scale,
                    coasting=True,
                )
                # If a parametric cylinder lost silhouette IoU (e.g. spurious pitch) while the live
                # VOS mask is clearly elongated/upright, test an upright zero-pitch pose preserving yaw.
                if c_iou < 0.62 and self.mesh.source_kind == "parametric" and self.mesh.name == "cylinder":
                    cur_fc = _extract_oriented_mask_frame(external_fg_mask)
                    if cur_fc is not None and (cur_fc[3] / max(cur_fc[2], 1.0)) >= 1.35:
                        R_c_now, _ = cv2.Rodrigues(rvec_c)
                        yaw_c = math.atan2(float(R_c_now[0, 2]), float(R_c_now[2, 2]))
                        R_y_up, _ = cv2.Rodrigues(np.array([0.0, yaw_c, 0.0], dtype=np.float64))
                        R_z_up, _ = cv2.Rodrigues(np.array([0.0, 0.0, float(cur_fc[4])], dtype=np.float64))
                        r_up, _ = cv2.Rodrigues(R_z_up @ R_y_up)
                        r_up = r_up.ravel()
                        u_iou, u_cov, u_db = self._eval_6dof_vos_iou(r_up, tvec_c, external_fg_mask)
                        t_up, u_iou, u_cov, u_db, _ = self._refine_6dof_pose_to_vos(
                            r_up,
                            tvec_c,
                            u_db,
                            u_iou,
                            u_cov,
                            external_fg_mask,
                            np.zeros((0, 3), dtype=np.float64),
                            np.zeros((0, 2), dtype=np.float32),
                            0.0,
                            res_scale,
                            coasting=True,
                        )
                        if u_iou > c_iou + 0.10:
                            rvec_c, tvec_c, c_iou, c_cov, c_db = r_up, t_up, u_iou, u_cov, u_db
                if c_iou >= 0.30 or c_cov >= 0.50:
                    self._vos_coast_frames += 1
                    self._last_rvec = rvec_c.copy()
                    self._last_tvec = tvec_c.copy()
                    self._last_valid_rvec = rvec_c.copy()
                    self._last_valid_tvec = tvec_c.copy()
                    self._curr_pts_track = np.zeros((0, 2), dtype=np.float32)
                    self._curr_pts3d = np.zeros((0, 3), dtype=np.float64)
                    self._replenish_6dof_points(
                        gray_track,
                        rvec_c,
                        tvec_c,
                        depth_buf=c_db,
                        external_fg_mask=external_fg_mask,
                    )
                    self._prev_gray = gray_track
                    track_ms = (time.perf_counter() - t0) * 1000.0
                    return TrackFrameResult(
                        frame_index=int(frame_index),
                        timestamp_s=float(timestamp_s),
                        state="tracking",
                        pose_rvec=rvec_c.copy(),
                        pose_tvec=tvec_c.copy(),
                        inliers=0,
                        confidence=float(np.clip(c_iou, 0.35, 0.85)),
                        track_ms=track_ms,
                        confidence_state="TRACKING",
                    )

            self._state = "lost"
            self._last_rvec = None
            self._last_tvec = None
            self._cand_pose = None
            self._vos_coast_frames = 0
            self._curr_pts_track = np.zeros((0, 2), dtype=np.float32)
            self._curr_pts3d = np.zeros((0, 3), dtype=np.float64)
            self._prev_gray = gray_track
            track_ms = (time.perf_counter() - t0) * 1000.0
            return TrackFrameResult(
                frame_index=int(frame_index),
                timestamp_s=float(timestamp_s),
                state="lost",
                inliers=0,
                confidence=0.0,
                track_ms=track_ms,
                confidence_state="LOST",
            )

        # Lost / recovering in 6-DoF: XFeat (when enabled) or ORB multi-keyframe 3D re-acquisition
        res_scale = max(1.0, float(self._frame_h) / 480.0)
        best_cand: tuple[int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

        if self.matcher_backend == "xfeat" and self._xfeat_mesh_keyframes:
            xf = self._ensure_xfeat()
            roi_rec = self._mask_bbox_roi(external_fg_mask) if use_vos else None
            cur_pts, cur_descs = xf.extract(frame_bgr, vos_inner, roi=roi_rec, top_k=640)
            if len(cur_pts) < 10 and roi_rec is not None:
                cur_pts, cur_descs = xf.extract(frame_bgr, None, roi=None, top_k=800)
            if len(cur_pts) >= 8:
                for _, _, kf_pts3d, kf_descs in self._xfeat_mesh_keyframes:
                    idx0, idx1 = xf.match_indices(kf_descs, cur_descs, min_cossim=0.72)
                    if len(idx0) >= 10:
                        pts3d = kf_pts3d[idx0]
                        cur_native = cur_pts[idx1]
                        ok, rvec, tvec, inl_bool, rmse = self._solve_pnp_robust(pts3d, cur_native, None)
                        if ok and rvec is not None and tvec is not None and rmse <= 4.5 * res_scale:
                            if not (max(0.06, 0.55 * self._init_tz) <= tvec[2] <= 1.65 * self._init_tz):
                                continue
                            n_inl = int(np.count_nonzero(inl_bool))
                            if n_inl < 10:
                                continue
                            if best_cand is None or n_inl > best_cand[0] or (n_inl == best_cand[0] and rmse < best_cand[1]):
                                best_cand = (
                                    n_inl,
                                    rmse,
                                    rvec,
                                    tvec,
                                    self._native_to_track(cur_native[inl_bool]),
                                    pts3d[inl_bool],
                                    inl_bool,
                                )
        else:
            kf_bank = self._mesh_keyframes if self._mesh_keyframes else (
                [(np.zeros(3), np.array([0.0, 0.0, self._init_tz]), self._mesh_pts3d_bank, self._mesh_descs_bank)]
                if (self._mesh_descs_bank is not None and len(self._mesh_descs_bank) >= 10)
                else []
            )
            if kf_bank:
                if self._use_native_recovery:
                    gray_rec = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    is_native = True
                else:
                    gray_rec = gray_track
                    is_native = False

                search_views: list[tuple[list[cv2.KeyPoint], np.ndarray | None, float, float]] = []
                kps_full, descs_full = self._orb_recovery.detectAndCompute(gray_rec, None)
                if descs_full is not None and len(kps_full) >= 8:
                    search_views.append((kps_full, descs_full, 0.0, 0.0))

                # Causal local 3x ROI around last verified silhouette bbox increases keypoint density on small rotating objects
                if is_native and getattr(self, "_last_sil_bbox", None) is not None:
                    lbx, lby, lbw, lbh = self._last_sil_bbox
                    rx0 = max(0, lbx - lbw)
                    ry0 = max(0, lby - lbh)
                    rx1 = min(self._frame_w, lbx + 2 * lbw)
                    ry1 = min(self._frame_h, lby + 2 * lbh)
                    if (rx1 - rx0) >= 48 and (ry1 - ry0) >= 48 and (rx1 - rx0) * (ry1 - ry0) < int(0.65 * self._frame_w * self._frame_h):
                        kps_roi, descs_roi = self._orb_recovery.detectAndCompute(gray_rec[ry0:ry1, rx0:rx1], None)
                        if descs_roi is not None and len(kps_roi) >= 8:
                            search_views.append((kps_roi, descs_roi, float(rx0), float(ry0)))

                for kps, descs, ox, oy in search_views:
                    if descs is None:
                        continue
                    for _, _, kf_pts3d, kf_descs in kf_bank:
                        if kf_descs is None or len(kf_descs) < 10:
                            continue
                        knn = self._bf.knnMatch(descs, kf_descs, k=2)
                        best_by_ref: dict[int, tuple[float, tuple[float, float]]] = {}
                        for pair in knn:
                            if len(pair) < 2:
                                continue
                            m, n = pair
                            if m.distance < 0.78 * n.distance and m.distance <= 62:
                                prev = best_by_ref.get(m.trainIdx)
                                if prev is None or m.distance < prev[0]:
                                    pt = kps[m.queryIdx].pt
                                    best_by_ref[m.trainIdx] = (m.distance, (pt[0] + ox, pt[1] + oy))
                        if len(best_by_ref) >= 12:
                            idxs = sorted(best_by_ref.keys())
                            cur_pts = np.array([best_by_ref[i][1] for i in idxs], dtype=np.float32)
                            cur_native = cur_pts if is_native else self._track_to_native(cur_pts)
                            cur_track = self._native_to_track(cur_native) if is_native else cur_pts
                            pts3d = kf_pts3d[idxs]
                            ok, rvec, tvec, inl_bool, rmse = self._solve_pnp_robust(pts3d, cur_native, None)
                            if ok and rvec is not None and tvec is not None and rmse <= 4.2 * res_scale:
                                if not (max(0.06, 0.55 * self._init_tz) <= tvec[2] <= 1.60 * self._init_tz):
                                    continue
                                n_inl = int(np.count_nonzero(inl_bool))
                                if n_inl < 12:
                                    continue
                                if best_cand is None or n_inl > best_cand[0] or (n_inl == best_cand[0] and rmse < best_cand[1]):
                                    best_cand = (n_inl, rmse, rvec, tvec, cur_track[inl_bool], pts3d[inl_bool], inl_bool)
                    if best_cand is not None and best_cand[0] >= 24:
                        break

        if best_cand is not None:
            n_inl, rmse, rvec, tvec, inl_track, inl_pts3d, _ = best_cand
            if use_vos and external_fg_mask is not None:
                vos_iou, vos_cov, depth_buf = self._eval_6dof_vos_iou(rvec, tvec, external_fg_mask)
                sil_ok = (vos_iou >= 0.40) or (vos_cov >= 0.55)
                edge_sup = vos_iou
            else:
                edge_sup, depth_buf = self._eval_6dof_edge_support(gray_track, rvec, tvec)
                sil_ok = edge_sup >= 0.40
            self._prev_gray = gray_track
            if sil_ok:
                if self._state == "recovering" and self._cand_pose is not None:
                    dt = float(np.linalg.norm(tvec - self._cand_pose[1]))
                    R1, _ = cv2.Rodrigues(rvec)
                    R0, _ = cv2.Rodrigues(self._cand_pose[0])
                    d_ang = math.degrees(math.acos(float(np.clip((np.trace(R1 @ R0.T) - 1.0) * 0.5, -1.0, 1.0))))
                    if dt <= 0.14 and d_ang <= 22.0:
                        self._state = "tracking"
                        self._last_rvec = rvec.copy()
                        self._last_tvec = tvec.copy()
                        self._cand_pose = None
                        self._vos_coast_frames = 0
                        self._curr_pts_track = inl_track.astype(np.float32)
                        self._curr_pts3d = inl_pts3d
                        self._reproj_errors.append(rmse)
                        self._replenish_6dof_points(
                            gray_track,
                            rvec,
                            tvec,
                            depth_buf=depth_buf,
                            external_fg_mask=external_fg_mask if use_vos else None,
                        )
                        track_ms = (time.perf_counter() - t0) * 1000.0
                        return TrackFrameResult(
                            frame_index=int(frame_index),
                            timestamp_s=float(timestamp_s),
                            state="tracking",
                            pose_rvec=rvec.copy(),
                            pose_tvec=tvec.copy(),
                            inliers=n_inl,
                            confidence=float(np.clip(0.65 * (n_inl / 24.0) + 0.35 * edge_sup, 0.0, 1.0)),
                            track_ms=track_ms,
                            confidence_state="TRACKING",
                        )
                self._state = "recovering"
                self._cand_pose = (rvec.copy(), tvec.copy())
                track_ms = (time.perf_counter() - t0) * 1000.0
                return TrackFrameResult(
                    frame_index=int(frame_index),
                    timestamp_s=float(timestamp_s),
                    state="recovering",
                    inliers=n_inl,
                    confidence=0.0,
                    track_ms=track_ms,
                    confidence_state="RECOVERING",
                )

        # Featureless-object VOS silhouette re-acquisition (e.g. plain white mug re-emerging from hands)
        if (
            use_vos
            and external_fg_mask is not None
            and self._last_valid_rvec is not None
            and self._last_valid_tvec is not None
            and self._init_vos_area > 0.0
            and vos_fg_px >= max(280, int(0.45 * self._init_vos_area))
            and vos_fg_px <= int(1.60 * self._init_vos_area)
        ):
            n_cc_r, _, stats_r, _ = cv2.connectedComponentsWithStats((external_fg_mask > 127).astype(np.uint8), connectivity=8)
            max_cc_r = int(stats_r[1:, cv2.CC_STAT_AREA].max()) if n_cc_r > 1 else 0
            if max_cc_r >= int(0.75 * vos_fg_px):
                # Align recovered cylinder orientation cleanly with the live VOS principal axis (no stale random tilt!)
                r_seed = self._last_valid_rvec.copy()
                cur_f_6 = _extract_oriented_mask_frame(external_fg_mask)
                if cur_f_6 is not None and self.mesh.source_kind == "parametric" and self.mesh.name == "cylinder":
                    # Extract current camera-plane axis roll of cylinder Y-axis from r_seed
                    R_prev_s, _ = cv2.Rodrigues(r_seed)
                    prev_y_ang = math.atan2(-float(R_prev_s[0, 1]), float(R_prev_s[1, 1]))
                    obs_ang = float(cur_f_6[4])
                    # Unwrap obs_ang modulo pi closest to prev_y_ang
                    best_obs = obs_ang
                    best_d = abs(obs_ang - prev_y_ang)
                    for cand_a in (obs_ang + math.pi, obs_ang - math.pi):
                        if abs(cand_a - prev_y_ang) < best_d:
                            best_d = abs(cand_a - prev_y_ang)
                            best_obs = cand_a
                    d_roll = float(np.clip(best_obs - prev_y_ang, -math.radians(40.0), math.radians(40.0)))
                    R_dz, _ = cv2.Rodrigues(np.array([0.0, 0.0, d_roll], dtype=np.float64))
                    r_seed, _ = cv2.Rodrigues(R_dz @ R_prev_s)
                    r_seed = r_seed.ravel()
                t_seed = self._last_valid_tvec.copy()
                c_iou, c_cov, c_db = self._eval_6dof_vos_iou(r_seed, t_seed, external_fg_mask)
                t_rec, c_iou, c_cov, c_db, _ = self._refine_6dof_pose_to_vos(
                    r_seed,
                    t_seed,
                    c_db,
                    c_iou,
                    c_cov,
                    external_fg_mask,
                    np.zeros((0, 3), dtype=np.float64),
                    np.zeros((0, 2), dtype=np.float32),
                    0.0,
                    res_scale,
                    coasting=True,
                )
                if c_iou >= 0.54 or (c_iou >= 0.45 and c_cov >= 0.70):
                    self._state = "tracking"
                    self._vos_coast_frames = 1
                    self._last_rvec = r_seed.copy()
                    self._last_tvec = t_rec.copy()
                    self._last_valid_rvec = r_seed.copy()
                    self._last_valid_tvec = t_rec.copy()
                    self._cand_pose = None
                    self._curr_pts_track = np.zeros((0, 2), dtype=np.float32)
                    self._curr_pts3d = np.zeros((0, 3), dtype=np.float64)
                    self._replenish_6dof_points(
                        gray_track,
                        r_seed,
                        t_rec,
                        depth_buf=c_db,
                        external_fg_mask=external_fg_mask,
                    )
                    self._prev_gray = gray_track
                    track_ms = (time.perf_counter() - t0) * 1000.0
                    return TrackFrameResult(
                        frame_index=int(frame_index),
                        timestamp_s=float(timestamp_s),
                        state="tracking",
                        pose_rvec=r_seed.copy(),
                        pose_tvec=t_rec.copy(),
                        inliers=0,
                        confidence=float(np.clip(c_iou, 0.45, 0.85)),
                        track_ms=track_ms,
                        confidence_state="TRACKING",
                    )

        self._prev_gray = gray_track
        self._state = "lost"
        self._cand_pose = None
        track_ms = (time.perf_counter() - t0) * 1000.0
        return TrackFrameResult(
            frame_index=int(frame_index),
            timestamp_s=float(timestamp_s),
            state="lost",
            inliers=0,
            confidence=0.0,
            track_ms=track_ms,
            confidence_state="LOST",
        )
