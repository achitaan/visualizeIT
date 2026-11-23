#!/usr/bin/env python3
"""Generate Dev-only visual-quality diagnostic assets with exact per-pixel ground truth.

Outputs (under .cache/vq-diagnostics/ by default):
  design_asym.png                 asymmetric logo + 4 distinct corner markers (orientation-revealing test artwork)
  vqdev_occluder_planar.mp4       textured planar card + moving skin-tone hand + same-colour occluder
  vqdev_occluder_planar_mono.mp4  identical scene rendered monochrome (SHOW3D-like, zero chroma)
  vqdev_spin_cylinder.mp4         textured cylinder spinning +/-75 deg about its axis (exact surface UV GT)
  <clip>_gt.npz                   per-frame exact GT (homography / UV maps / target & occluder masks)
  diagnostics_manifest.json       clip list (split='dev_diagnostic'; never mixed with the sealed holdout)

These clips are NEW Dev diagnostics. They deliberately do NOT reuse holdout clips such as
syn_occlusion_partial_hand or syn_cylinder_fast_spin, which stay sealed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_asym_design(size: tuple[int, int] = (512, 384)) -> np.ndarray:
    """RGBA artwork: asymmetric 'F'-arrow logo + four uniquely coloured corner markers."""
    w, h = size
    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, :3] = (40, 40, 200)  # BGR deep red background
    img[:, :, 3] = 255
    # Diagonal stripes give texture for visual inspection of sliding.
    for k in range(-h, w, 36):
        cv2.line(img, (k, 0), (k + h, h), (60, 60, 230, 255), 10)
    # Asymmetric logo: big 'F' + arrow pointing right-up.
    cx, cy = w // 2, h // 2
    cv2.rectangle(img, (cx - 90, cy - 110), (cx - 55, cy + 110), (255, 255, 255, 255), -1)
    cv2.rectangle(img, (cx - 90, cy - 110), (cx + 60, cy - 75), (255, 255, 255, 255), -1)
    cv2.rectangle(img, (cx - 90, cy - 20), (cx + 25, cy + 12), (255, 255, 255, 255), -1)
    pts = np.array([[cx + 40, cy + 90], [cx + 120, cy + 10], [cx + 120, cy + 60], [cx + 150, cy + 5], [cx + 95, cy - 5], [cx + 105, cy + 25]], np.int32)
    cv2.fillPoly(img, [pts], (0, 230, 255, 255))
    # Corner markers: TL red-white bullseye, TR green square, BR blue triangle, BL yellow circle.
    m = 26
    cv2.circle(img, (m + 10, m + 10), 26, (255, 255, 255, 255), -1)
    cv2.circle(img, (m + 10, m + 10), 15, (0, 0, 0, 255), -1)
    cv2.rectangle(img, (w - m - 40, m - 14), (w - m + 10, m + 36), (0, 200, 0, 255), -1)
    tri = np.array([[w - m + 10, h - m + 10], [w - m - 45, h - m + 10], [w - m + 10, h - m - 45]], np.int32)
    cv2.fillPoly(img, [tri], (255, 120, 0, 255))
    cv2.circle(img, (m + 12, h - m - 12), 24, (0, 255, 255, 255), -1)
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), (255, 255, 255, 255), 4)
    return img


def _scene_texture(w: int, h: int, seed: int, base: tuple[int, int, int]) -> np.ndarray:
    """Print-like textured surface (text blocks, shapes) so the tracker has real features."""
    rng = np.random.default_rng(seed)
    tex = np.full((h, w, 3), base, np.uint8)
    for _ in range(40):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(tex, (x, y), int(rng.integers(6, 28)), c, -1)
        elif kind == 1:
            cv2.rectangle(tex, (x, y), (x + int(rng.integers(10, 60)), y + int(rng.integers(6, 30))), c, -1)
        else:
            cv2.putText(tex, "".join(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ0123456789"), 4)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.5, 1.3)), c, 2)
    return cv2.GaussianBlur(tex, (3, 3), 0)


def _background(w: int, h: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bg = np.full((h, w, 3), (95, 105, 110), np.uint8)
    for _ in range(90):
        c = tuple(int(v) for v in rng.integers(40, 200, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.rectangle(bg, (x, y), (x + int(rng.integers(8, 90)), y + int(rng.integers(8, 90))), c, -1)
    return cv2.GaussianBlur(bg, (5, 5), 0)


def _write_mp4(frames: list[np.ndarray], path: Path, fps: float) -> None:
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0", "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def _hand_mask(w: int, h: int, cx: float, cy: float, ang: float, scale: float) -> np.ndarray:
    """Hand-like occluder: palm ellipse + 4 fingers + thumb + forearm extending off-frame."""
    m = np.zeros((h, w), np.uint8)
    ca, sa = math.cos(ang), math.sin(ang)

    def tr(px: float, py: float) -> tuple[int, int]:
        return int(round(cx + scale * (ca * px - sa * py))), int(round(cy + scale * (sa * px + ca * py)))

    cv2.ellipse(m, tr(0, 0), (int(42 * scale), int(52 * scale)), math.degrees(ang), 0, 360, 255, -1)
    for k, (fx, flen) in enumerate([(-30, 70), (-10, 82), (10, 78), (30, 64)]):
        p0, p1 = tr(fx, -35), tr(fx, -35 - flen)
        cv2.line(m, p0, p1, 255, int(17 * scale))
        cv2.circle(m, p1, int(8.5 * scale), 255, -1)
    cv2.line(m, tr(-38, 10), tr(-85, -25), 255, int(19 * scale))
    cv2.line(m, tr(0, 30), tr(0, 400), 255, int(70 * scale))  # forearm
    return m


def gen_occluder_planar(out_dir: Path, n: int = 120, w: int = 640, h: int = 480) -> dict:
    card_w, card_h = 300, 210
    card = _scene_texture(card_w, card_h, seed=11, base=(150, 175, 205))
    cv2.rectangle(card, (0, 0), (card_w - 1, card_h - 1), (30, 30, 30), 3)
    bg = _background(w, h, seed=5)
    src = np.array([[0, 0], [card_w, 0], [card_w, card_h], [0, card_h]], np.float32)
    frames, Hs, tmasks, omasks = [], [], [], []
    skin = np.array([120, 150, 200], np.float32)  # BGR skin
    rng = np.random.default_rng(3)
    for i in range(n):
        t = i / (n - 1)
        cx = w * 0.5 + 60 * math.sin(2 * math.pi * t)
        cy = h * 0.5 + 25 * math.sin(4 * math.pi * t)
        ang = math.radians(12 * math.sin(2 * math.pi * t * 1.3))
        persp = 0.18 * math.sin(2 * math.pi * t * 0.8)
        sc = 1.0 + 0.12 * math.sin(2 * math.pi * t * 0.6)
        corners = []
        for (x, y) in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            xx = x * card_w * 0.5 * sc * (1 + persp * y * 0.5)
            yy = y * card_h * 0.5 * sc * (1 - persp * x * 0.25)
            corners.append([cx + math.cos(ang) * xx - math.sin(ang) * yy, cy + math.sin(ang) * xx + math.cos(ang) * yy])
        dst = np.array(corners, np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(card, H, (w, h), flags=cv2.INTER_LINEAR)
        tm = cv2.warpPerspective(np.full((card_h, card_w), 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
        gain = 1.0 + 0.15 * math.sin(2 * math.pi * t * 1.7)  # global illumination drift
        frame = bg.copy()
        frame[tm > 0] = warped[tm > 0]
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255)
        occ = np.zeros((h, w), np.uint8)
        # Occluder A: skin-tone hand sweeping across the card (frames 18-62), covers <= ~45% of card.
        if 18 <= i <= 62:
            u = (i - 18) / 44.0
            hm = _hand_mask(w, h, cx - 210 + 300 * u, cy + 40 - 30 * math.sin(math.pi * u), math.radians(-25 + 30 * u), 0.85)
            occ = np.maximum(occ, hm)
            shade = skin * (0.85 + 0.15 * math.sin(i))
            frame[hm > 0] = shade
        # Occluder B: neutral grey-blue box with card-like colour (frames 72-108) -> chroma-poor occluder.
        if 72 <= i <= 108:
            u = (i - 72) / 36.0
            bm = np.zeros((h, w), np.uint8)
            bx = int(cx + 170 - 230 * u)
            cv2.rectangle(bm, (bx, int(cy - 150)), (bx + 95, int(cy + 160)), 255, -1)
            occ = np.maximum(occ, bm)
            frame[bm > 0] = np.array([140, 165, 195], np.float32) * gain
        frame = cv2.GaussianBlur(frame.astype(np.uint8), (3, 3), 0)
        noise = rng.normal(0, 2.0, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
        Hs.append(H)
        tmasks.append(((tm > 0) & (occ == 0)).astype(np.uint8))
        omasks.append((occ > 0).astype(np.uint8))
    full_target = []
    for H in Hs:
        full_target.append((cv2.warpPerspective(np.full((card_h, card_w), 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST) > 0).astype(np.uint8))
    for mono in (False, True):
        cid = "vqdev_occluder_planar_mono" if mono else "vqdev_occluder_planar"
        fr = [cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) for f in frames] if mono else frames
        _write_mp4(fr, out_dir / f"{cid}.mp4", 30.0)
        np.savez_compressed(
            out_dir / f"{cid}_gt.npz",
            kind="planar",
            H_card_to_img=np.stack(Hs),
            card_size=np.array([card_w, card_h]),
            target_visible=np.stack(tmasks),
            target_full=np.stack(full_target),
            occluder=np.stack(omasks),
        )
    c0 = cv2.perspectiveTransform(src.reshape(1, 4, 2), Hs[0]).reshape(4, 2)
    x0, y0 = c0.min(axis=0)
    x1, y1 = c0.max(axis=0)
    sel = {"frame_index": 0, "positive_points": [[float((x0 + x1) / 2), float((y0 + y1) / 2)]], "negative_points": [], "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]}
    return {"selection": sel, "wrap": "planar", "w": w, "h": h, "n": n}


def gen_spin_cylinder(out_dir: Path, n: int = 120, w: int = 640, h: int = 480) -> dict:
    f = 620.0
    cxk, cyk = w / 2, h / 2
    R, HGT = 0.045, 0.13
    lab_w, lab_h = 720, 260
    label = _scene_texture(lab_w, lab_h, seed=21, base=(70, 160, 200))
    for k in range(0, lab_w, 90):  # vertical seams make rotation visible
        cv2.line(label, (k, 0), (k, lab_h), (20, 20, 20), 2)
    bg = _background(w, h, seed=9)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    dirs = np.stack([(xs - cxk) / f, (ys - cyk) / f, np.ones_like(xs)], axis=-1)
    frames, uvs_all, masks = [], [], []
    poses = []
    rng = np.random.default_rng(4)
    for i in range(n):
        t = i / (n - 1)
        theta = math.radians(75.0) * math.sin(2 * math.pi * t)  # spin about axis
        tilt = math.radians(6.0) * math.sin(2 * math.pi * t * 0.7)
        tx = 0.03 * math.sin(2 * math.pi * t * 0.5)
        tvec = np.array([tx, 0.0, 0.42])
        cyr, syr = math.cos(theta), math.sin(theta)
        Ry = np.array([[cyr, 0, syr], [0, 1, 0], [-syr, 0, cyr]])
        cz, sz = math.cos(tilt), math.sin(tilt)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Rm = Rz @ Ry  # object -> camera
        o = Rm.T @ (-tvec)
        d = dirs @ Rm  # (R^T d) for each pixel
        a = d[..., 0] ** 2 + d[..., 2] ** 2
        b = 2 * (o[0] * d[..., 0] + o[2] * d[..., 2])
        c = o[0] ** 2 + o[2] ** 2 - R * R
        disc = b * b - 4 * a * c
        hit = disc > 0
        s = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a + 1e-12), np.inf)
        p = o[None, None, :] + s[..., None] * d
        hit &= (np.abs(p[..., 1]) <= HGT / 2) & (s > 0)
        phi = np.arctan2(p[..., 0], -p[..., 2])  # 0 = facing camera at theta=0
        u = phi / (2 * math.pi) + 0.5
        v = p[..., 1] / HGT + 0.5
        mapx = (u * lab_w).astype(np.float32) % lab_w
        mapy = np.clip(v * lab_h, 0, lab_h - 1).astype(np.float32)
        col = cv2.remap(label, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        nrm = np.stack([p[..., 0], np.zeros_like(u), p[..., 2]], -1) / R
        nrm_cam = nrm @ Rm.T
        light = np.array([-0.4, -0.3, -0.85])
        light /= np.linalg.norm(light)
        lam = np.clip(-(nrm_cam @ light) * -1.0, 0, 1)
        shade = 0.45 + 0.6 * np.clip((nrm_cam * np.array([0.4, 0.3, -0.85])).sum(-1) * -1 + 0.5, 0, 1) * 0.0 + 0.55 * lam
        frame = bg.astype(np.float32).copy()
        frame[hit] = col[hit].astype(np.float32) * np.clip(shade[hit], 0.35, 1.15)[:, None]
        frame = cv2.GaussianBlur(np.clip(frame, 0, 255).astype(np.uint8), (3, 3), 0)
        frame = np.clip(frame.astype(np.float32) + rng.normal(0, 2.0, frame.shape), 0, 255).astype(np.uint8)
        frames.append(frame)
        uv = np.stack([np.where(hit, u, -1), np.where(hit, v, -1)], -1).astype(np.float32)
        uvs_all.append(uv.astype(np.float16))
        masks.append(hit.astype(np.uint8))
        poses.append(np.concatenate([cv2.Rodrigues(Rm)[0].ravel(), tvec]))
    _write_mp4(frames, out_dir / "vqdev_spin_cylinder.mp4", 30.0)
    np.savez_compressed(
        out_dir / "vqdev_spin_cylinder_gt.npz",
        kind="cylinder",
        surface_uv=np.stack(uvs_all),
        target_full=np.stack(masks),
        target_visible=np.stack(masks),
        occluder=np.zeros((n, h, w), np.uint8),
        pose=np.stack(poses),
        K=np.array([[f, 0, cxk], [0, f, cyk], [0, 0, 1]]),
        radius=R,
        height=HGT,
    )
    ys0, xs0 = np.nonzero(masks[0])
    x0, y0, x1, y1 = xs0.min(), ys0.min(), xs0.max(), ys0.max()
    sel = {"frame_index": 0, "positive_points": [[float((x0 + x1) / 2), float((y0 + y1) / 2)]], "negative_points": [], "bbox": [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]}
    return {"selection": sel, "wrap": "cylinder", "w": w, "h": h, "n": n, "intrinsics": [f, f, cxk, cyk]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / ".cache" / "vq-diagnostics")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "design_asym.png"), make_asym_design())
    occ = gen_occluder_planar(out)
    cyl = gen_spin_cylinder(out)
    clips = []
    for cid, meta in (("vqdev_occluder_planar", occ), ("vqdev_occluder_planar_mono", occ), ("vqdev_spin_cylinder", cyl)):
        clips.append({
            "id": cid,
            "split": "dev_diagnostic",
            "path": str(out / f"{cid}.mp4"),
            "gt_npz": str(out / f"{cid}_gt.npz"),
            "width": meta["w"],
            "height": meta["h"],
            "frame_count": meta["n"],
            "recommended_wrap": meta["wrap"],
            "user_selection": meta["selection"],
            **({"intrinsics": meta["intrinsics"]} if "intrinsics" in meta else {}),
        })
    (out / "diagnostics_manifest.json").write_text(json.dumps({"clips": clips}, indent=2) + "\n")
    print(json.dumps([c["id"] for c in clips]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
