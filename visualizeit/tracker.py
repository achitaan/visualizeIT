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
) -> MeshGeometry:
    """Build a metric parametric 3D mesh for plane, cylinder, box, or curved_panel."""
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
        n_theta, n_y = 32, 8
        radius, height = float(cyl_radius), float(cyl_height)
        verts, norms, uvs, tris = [], [], [], []
        for j in range(n_y + 1):
            v_frac = j / n_y
            y = (v_frac - 0.5) * height
            for i in range(n_theta + 1):
                u_frac = i / n_theta
                theta = (u_frac - 0.5) * 2.0 * math.pi
                nx_v = math.sin(theta)
                nz_v = -math.cos(theta)
                verts.append([radius * nx_v, y, radius * nz_v])
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


class VideoSurfaceTracker:
    """Causal per-frame KLT + ORB tracker for planar, cylinder, mesh_grid, and mesh_6dof."""

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
    ) -> None:
        if tracking_height not in (360, 480) and not (120 <= tracking_height <= 1080):
            raise ContractValidationError(f"Invalid tracking_height: {tracking_height}")
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

        self._initialized = False
        self._frame_w = 0
        self._frame_h = 0
        self._track_w = 0
        self._track_h = 0
        self._sx = 1.0  # native -> track scale
        self._sy = 1.0

        self._orb = cv2.ORB_create(nfeatures=self.orb_features, nlevels=6, scaleFactor=1.2)
        self._orb_recovery = cv2.ORB_create(nfeatures=1500, nlevels=8, scaleFactor=1.2, fastThreshold=10)
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
        self._use_native_recovery: bool = False
        self._frames_since_kf: int = 0

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
        self._curr_pts3d: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._last_rvec: np.ndarray | None = None
        self._last_tvec: np.ndarray | None = None
        self._cand_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._alt_hypotheses: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._reproj_errors: list[float] = []

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
        """Detect corners in tracking-space mask with optional 8x6 spatial bucketing."""
        if not self.spatial_bucketing:
            corners = cv2.goodFeaturesToTrack(
                gray_track,
                maxCorners=self.max_klt_points,
                qualityLevel=0.01,
                minDistance=4.0,
                mask=mask_track,
                blockSize=5,
            )
            if corners is None or len(corners) == 0:
                return np.zeros((0, 2), dtype=np.float32)
            return corners.reshape(-1, 2).astype(np.float32)

        # Fixed-budget 8x6 spatial bucketing: detect up to 3x candidates, reserve per cell, cap at exact max_klt_points
        corners = cv2.goodFeaturesToTrack(
            gray_track,
            maxCorners=self.max_klt_points * 3,
            qualityLevel=0.005,
            minDistance=3.0,
            mask=mask_track,
            blockSize=5,
        )
        if corners is None or len(corners) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = corners.reshape(-1, 2).astype(np.float32)
        ys, xs = np.nonzero(mask_track)
        if len(xs) == 0:
            return pts[: self.max_klt_points]
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
        if self._frame_h > self.tracking_height:
            scale = float(self.tracking_height) / float(self._frame_h)
            self._track_h = self.tracking_height
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
        bx, by, bw, bh = seg.bbox
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
        self._use_native_recovery = (min(bw * self._sx, bh * self._sy) < 95.0) and (self._sx < 0.85)
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
            kps, descs = self._orb_recovery.detectAndCompute(gray_track, seed_mask)
            if descs is not None and len(kps) > 0:
                kp_track = np.array([kp.pt for kp in kps], dtype=np.float32)
                self._ref_kps_native = self._track_to_native(kp_track)
                self._ref_descs = descs.copy()
                self._ref_keyframes.append((self._ref_kps_native.copy(), self._ref_descs.copy()))
            else:
                self._ref_kps_native = np.zeros((0, 2), dtype=np.float32)
                self._ref_descs = None

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
            # Causally fit parametric cylinder aspect ratio & upright pose directly from frame-0 mask
            bx, by, bw, bh = seg.bbox
            cyl_h = 0.14
            cyl_r = float(np.clip(0.5 * (float(bw) / max(float(bh), 1.0)) * cyl_h, 0.015, 0.14))
            self.mesh = build_parametric_mesh("cylinder", cyl_radius=cyl_r, cyl_height=cyl_h)
            ys_m, xs_m = np.nonzero(seg.mask)
            mcx = float(xs_m.mean()) if len(xs_m) > 0 else (bx + bw * 0.5)
            mcy = float(ys_m.mean()) if len(ys_m) > 0 else (by + bh * 0.5)
            tz0 = max(0.12, fy * cyl_h / max(float(bh), 8.0))
            tx0 = (mcx - cx) * tz0 / fx
            ty0 = (mcy - cy) * tz0 / fy
            hypotheses = [(np.zeros(3, dtype=np.float64), np.array([tx0, ty0, tz0], dtype=np.float64), 0.95)]
        else:
            hypotheses = _estimate_initial_pose_from_mask(seg.mask, self.mesh, self._K)

        # Primary hypothesis
        best_rvec, best_tvec, _ = hypotheses[0]
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

        # Lift ORB reference bank to 3D for occlusion re-acquisition
        if self._ref_descs is not None and len(self._ref_kps_native) > 0:
            orb_pts3d, orb_valid = self._lift_2d_to_mesh3d(self._ref_kps_native, best_rvec, best_tvec)
            self._mesh_pts3d_bank = orb_pts3d[:1500]
            self._mesh_descs_bank = self._ref_descs[orb_valid][:1500].copy()
        else:
            self._mesh_pts3d_bank = np.zeros((0, 3), dtype=np.float64)
            self._mesh_descs_bank = None

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
        dist_thr = 64 if high_density else 62

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
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray_track,
            p0,
            None,
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
        )
        if p1 is None or st1 is None:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(len(pts_track), dtype=bool)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(
            gray_track,
            self._prev_gray,
            p1,
            None,
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
        )
        if pb is None or st2 is None:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(len(pts_track), dtype=bool)
        fb_err = np.linalg.norm((pb - p0).reshape(-1, 2), axis=1)
        p1_2d = p1.reshape(-1, 2)
        good = (
            (st1.ravel() == 1)
            & (st2.ravel() == 1)
            & (fb_err <= 1.5)
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
                    and n_inl >= 28
                    and inlier_ratio >= 0.55
                    and span_x >= 0.30 * q_w
                    and span_y >= 0.30 * q_h
                    and temp_ncc >= 0.40
                )
                if block_ncc < 0.20 and not strong_geom_temporal:
                    return False, n_inl, inl_bool, cur_patch, block_ncc
            else:
                ncc = _patch_ncc(self._ref_patch, cur_patch)
                block_ncc = _patch_block_ncc_p75(self._ref_patch, cur_patch, grid=4)
                if max(ncc, block_ncc * 0.85) < min_ncc:
                    return False, n_inl, inl_bool, cur_patch, ncc

        return True, n_inl, inl_bool, cur_patch, block_ncc

    def _replenish_planar_points(self, gray_track: np.ndarray, H: np.ndarray) -> None:
        """Detect new Shi-Tomasi corners inside the current tracked object and map back to frame-0 coords."""
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

    def track_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_s: float,
    ) -> TrackFrameResult:
        """Causal per-frame tracking step shared between file processing and live 30 Hz replay."""
        if not self._initialized:
            raise RuntimeError("VideoSurfaceTracker.initialize() must be called before track_frame()")

        t0 = time.perf_counter()
        gray_track = self._to_track_gray(frame_bgr)

        if self.wrap_mode == "mesh_6dof":
            return self._track_frame_6dof(gray_track, frame_index, timestamp_s, t0)

        # --- Planar / Cylinder / Mesh-Grid Tracking ---
        if self._state == "tracking":
            p1_track, good = self._run_fb_klt(gray_track, self._curr_pts_track)
            klt_cur_track = p1_track[good]
            klt_ref_native = self._curr_ref_pts_native[good]
            klt_cur_native = self._track_to_native(klt_cur_track)

            ref_all = klt_ref_native
            cur_all = klt_cur_native

            # If KLT count is getting low, supplement with local ORB matches
            if len(ref_all) < 24:
                orb_ref, orb_cur = self._match_orb_to_ref(gray_track)
                if len(orb_ref) >= 6:
                    ref_all = np.vstack([ref_all, orb_ref]) if len(ref_all) else orb_ref
                    cur_all = np.vstack([cur_all, orb_cur]) if len(cur_all) else orb_cur

            H_est: np.ndarray | None = None
            inl_mask: np.ndarray | None = None
            if len(ref_all) >= 8:
                cv2.setRNGSeed(17)
                H_est, inl_mask = cv2.findHomography(ref_all, cur_all, cv2.RANSAC, 3.5)

            ok, n_inl, inl_bool, cur_patch, block_ncc = self._validate_homography_and_appearance(
                H_est, gray_track, inl_mask, cur_all, min_inliers=10, min_ncc=0.18, is_recovery=False
            )

            # Continuity jump check while tracking (reject implausible single-frame teleport)
            if ok and self._last_H is not None and H_est is not None:
                prev_q = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), self._last_H).reshape(4, 2)
                cur_q = cv2.perspectiveTransform(self._ref_quad.reshape(1, 4, 2), H_est).reshape(4, 2)
                max_shift = float(np.linalg.norm(cur_q - prev_q, axis=1).max())
                if max_shift > max(self._frame_w, self._frame_h) * 0.35:
                    ok = False

            if ok and H_est is not None:
                # Reprojection RMSE on inliers
                proj = cv2.perspectiveTransform(ref_all[inl_bool].reshape(-1, 1, 2), H_est).reshape(-1, 2)
                err = np.linalg.norm(proj - cur_all[inl_bool], axis=1)
                med_err = float(np.median(err)) if len(err) else 0.0
                self._reproj_errors.append(med_err)

                # Explicit AMBIGUOUS gate: suppress overlay when support is thin and appearance is weak
                if n_inl < 14 and block_ncc < 0.25 and med_err > 2.5:
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
                self._prev_patch = cur_patch
                self._last_block_ncc = block_ncc
                self._curr_pts_track = self._native_to_track(cur_all[inl_bool])
                self._curr_ref_pts_native = ref_all[inl_bool].astype(np.float32)
                self._frames_since_replenish += 1
                self._frames_since_kf += 1
                # Prevent occluded frames from contaminating KLT memory
                allow_replenish = (not self.adaptive_appearance_gate) or (block_ncc >= 0.28)
                if allow_replenish and (
                    len(self._curr_pts_track) < int(self.max_klt_points * 0.55) or self._frames_since_replenish >= 12
                ):
                    self._replenish_planar_points(gray_track, H_est)
                    self._frames_since_replenish = 0
                if block_ncc >= 0.35 and n_inl >= 25 and self._frames_since_kf >= 15:
                    self._maybe_store_keyframe(frame_bgr, gray_track, H_est)
                    self._frames_since_kf = 0

                self._prev_gray = gray_track
                self._lost_streak = 0
                conf = float(np.clip(n_inl / max(float(len(ref_all)), 1.0), 0.0, 1.0))
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

        # --- Lost or Recovering: high-density multi-keyframe ORB re-acquisition ---
        if self._use_native_recovery:
            gray_rec = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            orb_ref, orb_cur = self._match_orb_to_ref(gray_rec, roi=None, high_density=True, is_native=True)
        else:
            orb_ref, orb_cur = self._match_orb_to_ref(gray_track, roi=None, high_density=True, is_native=False)

        H_cand: np.ndarray | None = None
        inl_mask = None
        if len(orb_ref) >= 9:
            cv2.setRNGSeed(17)
            H_cand, inl_mask = cv2.findHomography(orb_ref, orb_cur, cv2.RANSAC, 4.0)

        ok, n_inl, inl_bool, cur_patch, block_ncc = self._validate_homography_and_appearance(
            H_cand, gray_track, inl_mask, orb_cur, min_inliers=9, min_ncc=0.22, is_recovery=True
        )
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
                    self._replenish_planar_points(gray_track, H_cand)
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

        best_res: tuple[int, float, np.ndarray, np.ndarray, np.ndarray] | None = None
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
            if best_res is None or (n_inl > best_res[0]) or (n_inl == best_res[0] and rmse < best_res[1]):
                best_res = (n_inl, rmse, rvec, tvec, inl_bool)

        if best_res is None:
            return False, None, None, np.zeros(len(pts3d), dtype=bool), 999.0
        _, rmse, rvec, tvec, inl_bool = best_res
        return True, rvec, tvec, inl_bool, rmse

    def _replenish_6dof_points(self, gray_track: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> None:
        """Project current mesh silhouette, detect interior corners, and lift onto 3D mesh surface."""
        assert self.mesh is not None and self._K is not None
        R, _ = cv2.Rodrigues(rvec)
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
        sil_track = cv2.erode(sil_track, np.ones((3, 3), np.uint8), iterations=1)
        new_track = self._detect_corners_in_mask(gray_track, sil_track)
        if len(new_track) == 0:
            return
        new_native = self._track_to_native(new_track)
        u = new_native[:, 0]
        v = new_native[:, 1]
        iu = np.clip(np.round(u).astype(np.int32), 0, self._frame_w - 1)
        iv = np.clip(np.round(v).astype(np.int32), 0, self._frame_h - 1)
        z = res.depth_buf[iv, iu].astype(np.float64)
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

    def _track_frame_6dof(
        self,
        gray_track: np.ndarray,
        frame_index: int,
        timestamp_s: float,
        t0: float,
    ) -> TrackFrameResult:
        assert self._K is not None and self.mesh is not None

        if self._state == "tracking":
            p1_track, good = self._run_fb_klt(gray_track, self._curr_pts_track)
            cur_track = p1_track[good]
            cur_pts3d = self._curr_pts3d[good]
            cur_native = self._track_to_native(cur_track)

            prior = (self._last_rvec, self._last_tvec) if (self._last_rvec is not None and self._last_tvec is not None) else None
            ok, rvec, tvec, inl_bool, rmse = self._solve_pnp_robust(cur_pts3d, cur_native, prior)

            # Causal symmetry disambiguation on early frames (frame 1..3) if alternate silhouette hypotheses exist
            if self._alt_hypotheses and frame_index <= 3:
                best_rmse = rmse if ok else 999.0
                best_choice: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None = None
                surviving_alts = []
                for alt_r, alt_t, alt_trk, alt_3d in self._alt_hypotheses:
                    ap1, ag = self._run_fb_klt(gray_track, alt_trk)
                    if int(np.count_nonzero(ag)) >= 8:
                        aok, ar, at, ainl, armse = self._solve_pnp_robust(
                            alt_3d[ag], self._track_to_native(ap1[ag]), (alt_r, alt_t)
                        )
                        if aok and ar is not None and at is not None:
                            surviving_alts.append((ar, at, ap1[ag][ainl], alt_3d[ag][ainl]))
                            if armse + 0.25 < best_rmse and int(np.count_nonzero(ainl)) >= 12:
                                best_rmse = armse
                                best_choice = (ar, at, ap1[ag], alt_3d[ag], ainl, armse)
                if best_choice is not None:
                    rvec, tvec, cur_track, cur_pts3d, inl_bool, rmse = best_choice
                    ok = True
                self._alt_hypotheses = surviving_alts if frame_index < 3 else []

            if ok and rvec is not None and tvec is not None and rmse <= 6.0:
                n_inl = int(np.count_nonzero(inl_bool))
                self._last_rvec = rvec.copy()
                self._last_tvec = tvec.copy()
                self._curr_pts_track = cur_track[inl_bool].astype(np.float32)
                self._curr_pts3d = cur_pts3d[inl_bool]
                self._reproj_errors.append(rmse)
                self._frames_since_replenish += 1
                if len(self._curr_pts_track) < int(self.max_klt_points * 0.55) or self._frames_since_replenish >= 10:
                    self._replenish_6dof_points(gray_track, rvec, tvec)
                    self._frames_since_replenish = 0
                self._prev_gray = gray_track
                conf = float(np.clip(n_inl / max(float(len(cur_pts3d)), 1.0), 0.0, 1.0))
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
                )

            self._state = "lost"
            self._last_rvec = None
            self._last_tvec = None
            self._cand_pose = None
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
            )

        # Lost / recovering in 6-DoF: ORB matching against 3D reference bank
        if self._mesh_descs_bank is not None and len(self._mesh_descs_bank) >= 10:
            kps, descs = self._orb.detectAndCompute(gray_track, None)
            if descs is not None and len(kps) >= 8:
                knn = self._bf.knnMatch(descs, self._mesh_descs_bank, k=2)
                best_by_ref: dict[int, tuple[float, tuple[float, float]]] = {}
                for pair in knn:
                    if len(pair) < 2:
                        continue
                    m, n = pair
                    if m.distance < 0.78 * n.distance and m.distance <= 62:
                        prev = best_by_ref.get(m.trainIdx)
                        if prev is None or m.distance < prev[0]:
                            best_by_ref[m.trainIdx] = (m.distance, kps[m.queryIdx].pt)
                if len(best_by_ref) >= 10:
                    idxs = sorted(best_by_ref.keys())
                    cur_track = np.array([best_by_ref[i][1] for i in idxs], dtype=np.float32)
                    cur_native = self._track_to_native(cur_track)
                    pts3d = self._mesh_pts3d_bank[idxs]
                    ok, rvec, tvec, inl_bool, rmse = self._solve_pnp_robust(pts3d, cur_native, None)
                    if ok and rvec is not None and tvec is not None and rmse <= 5.0:
                        n_inl = int(np.count_nonzero(inl_bool))
                        self._prev_gray = gray_track
                        if self._state == "recovering" and self._cand_pose is not None:
                            dt = float(np.linalg.norm(tvec - self._cand_pose[1]))
                            if dt <= 0.15:
                                self._state = "tracking"
                                self._last_rvec = rvec.copy()
                                self._last_tvec = tvec.copy()
                                self._cand_pose = None
                                self._curr_pts_track = cur_track[inl_bool].astype(np.float32)
                                self._curr_pts3d = pts3d[inl_bool]
                                self._reproj_errors.append(rmse)
                                self._replenish_6dof_points(gray_track, rvec, tvec)
                                track_ms = (time.perf_counter() - t0) * 1000.0
                                return TrackFrameResult(
                                    frame_index=int(frame_index),
                                    timestamp_s=float(timestamp_s),
                                    state="tracking",
                                    pose_rvec=rvec.copy(),
                                    pose_tvec=tvec.copy(),
                                    inliers=n_inl,
                                    confidence=float(np.clip(n_inl / max(float(len(pts3d)), 1.0), 0.0, 1.0)),
                                    track_ms=track_ms,
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
        )
