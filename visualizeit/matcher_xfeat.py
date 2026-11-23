"""Learned dense/sparse keypoint extraction and matching backed by XFeat (CVPR 2024)."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

_XFEAT_LOCK = threading.Lock()
_XFEAT_CACHE: dict[str, Any] = {}


def _resolve_xfeat_paths() -> tuple[Path, Path]:
    root = Path(os.environ.get("VISUALIZEIT_XFEAT_DIR", str(REPO_ROOT / ".cache" / "models" / "accelerated_features")))
    weights = Path(os.environ.get("VISUALIZEIT_XFEAT_WEIGHTS", str(root / "weights" / "xfeat.pt")))
    return root, weights


def is_xfeat_available() -> bool:
    """Return True if PyTorch, XFeat repo, and xfeat.pt weights are available locally."""
    root, weights = _resolve_xfeat_paths()
    if not (root / "modules" / "xfeat.py").is_file() or not weights.is_file() or weights.stat().st_size < 1_000_000:
        return False
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _get_xfeat_model(device: str = "cpu") -> Any:
    """Lazy-load and cache the singleton XFeat model."""
    with _XFEAT_LOCK:
        cached = _XFEAT_CACHE.get(device)
        if cached is not None:
            return cached

        root, weights = _resolve_xfeat_paths()
        if not (root / "modules" / "xfeat.py").is_file() or not weights.is_file():
            raise RuntimeError(
                f"XFeat model files not found at {root} / {weights}. "
                "Install accelerated_features into .cache/models/accelerated_features or set VISUALIZEIT_XFEAT_DIR."
            )

        import torch

        if device == "cpu":
            n_threads = max(1, min(4, os.cpu_count() or 2))
            try:
                torch.set_num_threads(n_threads)
            except RuntimeError:
                pass

        root_str = str(root.resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        from modules.xfeat import XFeat

        model = XFeat(weights=str(weights.resolve()), top_k=1024, detection_threshold=0.04)
        model.eval()
        _XFEAT_CACHE[device] = model
        return model


class XFeatMatcher:
    """Lightweight CPU/GPU XFeat keypoint extractor and mutual nearest-neighbor matcher."""

    def __init__(self, *, device: str = "cpu", max_dim: int = 480) -> None:
        self.device = device
        self.max_dim = max(160, int(max_dim))
        self._model = _get_xfeat_model(device)

    def extract(
        self,
        frame_bgr: np.ndarray,
        mask_u8: np.ndarray | None = None,
        *,
        max_dim: int | None = None,
        top_k: int = 512,
        roi: tuple[int, int, int, int] | None = None,
        erode_mask: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract native-coordinate (N, 2) float32 keypoints and (N, 64) L2-normalized float32 descriptors."""
        import torch

        h, w = frame_bgr.shape[:2]
        ox, oy = 0, 0
        crop = frame_bgr
        if roi is not None:
            x0, y0, x1, y1 = roi
            x0 = max(0, int(x0))
            y0 = max(0, int(y0))
            x1 = min(w, int(x1))
            y1 = min(h, int(y1))
            if (x1 - x0) >= 32 and (y1 - y0) >= 32:
                crop = frame_bgr[y0:y1, x0:x1]
                ox, oy = x0, y0

        ch, cw = crop.shape[:2]
        cap_dim = self.max_dim if max_dim is None else max(160, int(max_dim))
        max_side = max(ch, cw)
        if max_side > cap_dim:
            scale = float(cap_dim) / float(max_side)
            sw = max(32, int(round(cw * scale)))
            sh = max(32, int(round(ch * scale)))
            img_in = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
            inv_sx = float(cw) / float(sw)
            inv_sy = float(ch) / float(sh)
        elif roi is not None and max_side < 160:
            scale = min(2.0, 192.0 / float(max(max_side, 1)))
            sw = max(32, int(round(cw * scale)))
            sh = max(32, int(round(ch * scale)))
            img_in = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_CUBIC)
            inv_sx = float(cw) / float(sw)
            inv_sy = float(ch) / float(sh)
        else:
            img_in = crop
            inv_sx = 1.0
            inv_sy = 1.0

        rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self._model.dev) / 255.0

        with torch.inference_mode():
            out = self._model.detectAndCompute(tensor, top_k=int(top_k), detection_threshold=0.04)[0]

        kpts = out["keypoints"].detach().cpu().numpy().astype(np.float32)
        descs = out["descriptors"].detach().cpu().numpy().astype(np.float32)
        if len(kpts) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 64), dtype=np.float32)

        kpts[:, 0] = kpts[:, 0] * inv_sx + float(ox)
        kpts[:, 1] = kpts[:, 1] * inv_sy + float(oy)

        if mask_u8 is not None and mask_u8.shape == (h, w):
            m_use = mask_u8
            if erode_mask and int(np.count_nonzero(mask_u8)) > 400:
                eroded = cv2.erode(mask_u8, np.ones((5, 5), np.uint8), iterations=1)
                if int(np.count_nonzero(eroded)) >= 120:
                    m_use = eroded
            iu = np.clip(np.round(kpts[:, 0]).astype(np.int32), 0, w - 1)
            iv = np.clip(np.round(kpts[:, 1]).astype(np.int32), 0, h - 1)
            inside = m_use[iv, iu] > 127
            if not np.any(inside) and m_use is not mask_u8:
                inside = mask_u8[iv, iu] > 127
            kpts = kpts[inside]
            descs = descs[inside]

        if len(kpts) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 64), dtype=np.float32)

        # Ensure unit L2 norm
        norms = np.linalg.norm(descs, axis=1, keepdims=True)
        descs = descs / np.maximum(norms, 1e-8)
        return np.ascontiguousarray(kpts, dtype=np.float32), np.ascontiguousarray(descs, dtype=np.float32)

    @staticmethod
    def match_indices(
        desc0: np.ndarray,
        desc1: np.ndarray,
        *,
        min_cossim: float = 0.78,
        min_margin: float = 0.02,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mutual nearest-neighbor cosine matching returning (idx0, idx1) int64 arrays."""
        if desc0 is None or desc1 is None or len(desc0) < 4 or len(desc1) < 4:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        sim = np.matmul(desc0.astype(np.float32), desc1.astype(np.float32).T)  # (N0, N1)
        best1_for_0 = np.argmax(sim, axis=1)
        best0_for_1 = np.argmax(sim, axis=0)
        idx0_all = np.arange(len(desc0), dtype=np.int64)
        mutual = best0_for_1[best1_for_0] == idx0_all
        scores = sim[idx0_all, best1_for_0]
        keep = mutual & (scores >= float(min_cossim))

        if min_margin > 0.0 and sim.shape[1] >= 2 and np.any(keep):
            # Check separation from 2nd-best match to suppress ambiguous repetitive textures
            part = np.partition(sim, -2, axis=1)
            second = part[:, -2]
            keep &= (scores - second) >= float(min_margin)

        idx0 = idx0_all[keep]
        idx1 = best1_for_0[keep].astype(np.int64)
        return idx0, idx1

    @classmethod
    def match(
        cls,
        pts0: np.ndarray,
        desc0: np.ndarray,
        pts1: np.ndarray,
        desc1: np.ndarray,
        *,
        min_cossim: float = 0.78,
        min_margin: float = 0.02,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match two sets of XFeat features, returning (matched_pts0, matched_pts1) as (M, 2) float32."""
        idx0, idx1 = cls.match_indices(desc0, desc1, min_cossim=min_cossim, min_margin=min_margin)
        if len(idx0) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        return np.ascontiguousarray(pts0[idx0], dtype=np.float32), np.ascontiguousarray(pts1[idx1], dtype=np.float32)
