/* Dedicated Uncapped Causal Video/Camera Tracking Web Worker (Pyramidal KLT + Similarity Quad + NCC Re-Acquisition) */
"use strict";

let currentEpoch = 0;
let active = false;
let state = "idle"; // "idle" | "tracking" | "recovering" | "lost"

let trackW = 0;
let trackH = 0;
let fullW = 0;
let fullH = 0;
let sx = 1.0;
let sy = 1.0;

let prevGray = null;
let prevHalf = null;

// Reference box/quad in tracking coordinates
let refBox = null; // { x, y, w, h }
let refArea = 1.0;
let curQuad = null; // [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
let refPatch = null; // Float32Array(24*24) zero-mean unit-norm
let refPatchCoarse = null; // Float32Array(12*12) zero-mean unit-norm for fast rotated re-acquisition
let prevPatch = null;
let refGradEnergy = 0;

// Tracked feature points in tracking coordinates
let ptsX = new Float32Array(0);
let ptsY = new Float32Array(0);
let numPts = 0;
let framesSinceReplenish = 0;
let recoverCand = null; // { cx, cy, quad, score }

const PATCH_N = 24;
const COARSE_N = 12;

function downsampleHalf(gray, w, h) {
  const hw = Math.max(8, w >> 1);
  const hh = Math.max(8, h >> 1);
  const out = new Uint8Array(hw * hh);
  for (let y = 0; y < hh; y++) {
    const sy0 = Math.min(h - 1, y << 1);
    const sy1 = Math.min(h - 1, sy0 + 1);
    const r0 = sy0 * w;
    const r1 = sy1 * w;
    const dstRow = y * hw;
    for (let x = 0; x < hw; x++) {
      const sx0 = Math.min(w - 1, x << 1);
      const sx1 = Math.min(w - 1, sx0 + 1);
      out[dstRow + x] = (gray[r0 + sx0] + gray[r0 + sx1] + gray[r1 + sx0] + gray[r1 + sx1]) >> 2;
    }
  }
  return { data: out, w: hw, h: hh };
}

function sampleBilinear(gray, w, h, x, y) {
  if (x < 0) x = 0;
  else if (x > w - 1.001) x = w - 1.001;
  if (y < 0) y = 0;
  else if (y > h - 1.001) y = h - 1.001;
  const x0 = x | 0;
  const y0 = y | 0;
  const fx = x - x0;
  const fy = y - y0;
  const idx = y0 * w + x0;
  const v00 = gray[idx];
  const v10 = gray[idx + 1];
  const v01 = gray[idx + w];
  const v11 = gray[idx + w + 1];
  return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
}

function boxToQuad(box) {
  return [
    [box.x, box.y],
    [box.x + box.w, box.y],
    [box.x + box.w, box.y + box.h],
    [box.x, box.y + box.h],
  ];
}

function rotatedBoxQuad(cx, cy, w, h, angleRad) {
  const c = Math.cos(angleRad);
  const s = Math.sin(angleRad);
  const hw = w * 0.5;
  const hh = h * 0.5;
  const corners = [
    [-hw, -hh],
    [hw, -hh],
    [hw, hh],
    [-hw, hh],
  ];
  return corners.map(([dx, dy]) => [cx + dx * c - dy * s, cy + dx * s + dy * c]);
}

function quadArea(q) {
  const a1 = Math.abs((q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[1][1] - q[0][1]) * (q[2][0] - q[0][0])) * 0.5;
  const a2 = Math.abs((q[2][0] - q[0][0]) * (q[3][1] - q[0][1]) - (q[2][1] - q[0][1]) * (q[3][0] - q[0][0])) * 0.5;
  return a1 + a2;
}

function quadBounds(q) {
  let minX = q[0][0], maxX = q[0][0], minY = q[0][1], maxY = q[0][1];
  for (let i = 1; i < 4; i++) {
    if (q[i][0] < minX) minX = q[i][0];
    if (q[i][0] > maxX) maxX = q[i][0];
    if (q[i][1] < minY) minY = q[i][1];
    if (q[i][1] > maxY) maxY = q[i][1];
  }
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

function quadCentroid(q) {
  return [
    (q[0][0] + q[1][0] + q[2][0] + q[3][0]) * 0.25,
    (q[0][1] + q[1][1] + q[2][1] + q[3][1]) * 0.25,
  ];
}

function pointInConvexQuad(px, py, q) {
  let pos = 0;
  let neg = 0;
  for (let i = 0; i < 4; i++) {
    const a = q[i];
    const b = q[(i + 1) & 3];
    const cross = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]);
    if (cross > 0) pos++;
    else if (cross < 0) neg++;
  }
  return pos === 0 || neg === 0;
}

function quadToFullQuad(q) {
  const invX = 1.0 / sx;
  const invY = 1.0 / sy;
  return [
    [q[0][0] * invX, q[0][1] * invY],
    [q[1][0] * invX, q[1][1] * invY],
    [q[2][0] * invX, q[2][1] * invY],
    [q[3][0] * invX, q[3][1] * invY],
  ];
}

function extractNormalizedPatchQuad(gray, w, h, q, patchN) {
  const n = patchN || PATCH_N;
  const patch = new Float32Array(n * n);
  let sum = 0;
  const q0x = q[0][0], q0y = q[0][1];
  const q1x = q[1][0], q1y = q[1][1];
  const q2x = q[2][0], q2y = q[2][1];
  const q3x = q[3][0], q3y = q[3][1];

  for (let j = 0; j < n; j++) {
    const v = (j + 0.5) / n;
    const iv = 1.0 - v;
    const lx = iv * q0x + v * q3x;
    const ly = iv * q0y + v * q3y;
    const rx = iv * q1x + v * q2x;
    const ry = iv * q1y + v * q2y;
    const rowOff = j * n;
    for (let i = 0; i < n; i++) {
      const u = (i + 0.5) / n;
      const px = lx + u * (rx - lx);
      const py = ly + u * (ry - ly);
      const val = sampleBilinear(gray, w, h, px, py);
      patch[rowOff + i] = val;
      sum += val;
    }
  }
  const mean = sum / patch.length;
  let normSq = 0;
  for (let k = 0; k < patch.length; k++) {
    const d = patch[k] - mean;
    patch[k] = d;
    normSq += d * d;
  }
  let gradSum = 0;
  for (let j = 0; j < n - 1; j++) {
    const r = j * n;
    for (let i = 0; i < n - 1; i++) {
      const idx = r + i;
      gradSum += Math.abs(patch[idx + 1] - patch[idx]) + Math.abs(patch[idx + n] - patch[idx]);
    }
  }
  const std = Math.sqrt(normSq / patch.length);
  const gradEnergy = gradSum / ((n - 1) * (n - 1));
  const norm = Math.sqrt(normSq);
  if (norm > 1e-4) {
    const inv = 1.0 / norm;
    for (let k = 0; k < patch.length; k++) patch[k] *= inv;
  }
  return { patch, std, gradEnergy, norm };
}

function patchNCC(pA, pB) {
  let dot = 0;
  for (let k = 0; k < pA.length; k++) dot += pA[k] * pB[k];
  return dot;
}

function detectCornersInQuad(gray, w, h, q, maxCount) {
  const b = quadBounds(q);
  const x0 = Math.max(4, Math.floor(b.x + 2));
  const y0 = Math.max(4, Math.floor(b.y + 2));
  const x1 = Math.min(w - 5, Math.ceil(b.x + b.w - 2));
  const y1 = Math.min(h - 5, Math.ceil(b.y + b.h - 2));
  if (x1 - x0 < 6 || y1 - y0 < 6) return { xs: new Float32Array(0), ys: new Float32Array(0), n: 0 };

  const cands = [];
  const step = Math.max(2, Math.floor(Math.min(x1 - x0, y1 - y0) / 18));
  for (let y = y0; y <= y1; y += step) {
    for (let x = x0; x <= x1; x += step) {
      if (!pointInConvexQuad(x, y, q)) continue;
      let sxx = 0, syy = 0, sxy = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const row = (y + dy) * w;
        for (let dx = -1; dx <= 1; dx++) {
          const idx = row + x + dx;
          const gx = gray[idx + 1] - gray[idx - 1];
          const gy = gray[idx + w] - gray[idx - w];
          sxx += gx * gx;
          syy += gy * gy;
          sxy += gx * gy;
        }
      }
      const trace = sxx + syy;
      const det = sxx * syy - sxy * sxy;
      const disc = Math.max(0, trace * trace - 4 * det);
      const minEig = 0.5 * (trace - Math.sqrt(disc));
      if (minEig > 40.0) {
        cands.push({ x, y, score: minEig });
      }
    }
  }

  cands.sort((a, b) => b.score - a.score);
  const chosen = [];
  const minDistSq = 10.0;
  for (let i = 0; i < cands.length && chosen.length < maxCount; i++) {
    const c = cands[i];
    let tooClose = false;
    for (let j = 0; j < chosen.length; j++) {
      const dx = c.x - chosen[j].x;
      const dy = c.y - chosen[j].y;
      if (dx * dx + dy * dy < minDistSq) {
        tooClose = true;
        break;
      }
    }
    if (!tooClose) chosen.push(c);
  }

  const n = chosen.length;
  const xs = new Float32Array(n);
  const ys = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = chosen[i].x;
    ys[i] = chosen[i].y;
  }
  return { xs, ys, n };
}

function lkSingleLevel(g0, g1, w, h, x0, y0, initDx, initDy) {
  const half = 3;
  let dx = initDx;
  let dy = initDy;

  const ix0 = Math.round(x0);
  const iy0 = Math.round(y0);
  if (ix0 < half + 1 || ix0 >= w - half - 1 || iy0 < half + 1 || iy0 >= h - half - 1) {
    return null;
  }

  let sxx = 0, syy = 0, sxy = 0;
  const patch0 = new Float32Array(49);
  const gxArr = new Float32Array(49);
  const gyArr = new Float32Array(49);
  let k = 0;
  for (let wy = -half; wy <= half; wy++) {
    const row = (iy0 + wy) * w;
    for (let wx = -half; wx <= half; wx++) {
      const idx = row + ix0 + wx;
      patch0[k] = g0[idx];
      const gx = 0.5 * (g0[idx + 1] - g0[idx - 1]);
      const gy = 0.5 * (g0[idx + w] - g0[idx - w]);
      gxArr[k] = gx;
      gyArr[k] = gy;
      sxx += gx * gx;
      syy += gy * gy;
      sxy += gx * gy;
      k++;
    }
  }
  const det = sxx * syy - sxy * sxy;
  if (det < 1e-2) return null;
  const invDet = 1.0 / det;

  for (let iter = 0; iter < 5; iter++) {
    const cx = x0 + dx;
    const cy = y0 + dy;
    if (cx < half + 1 || cx >= w - half - 2 || cy < half + 1 || cy >= h - half - 2) {
      return null;
    }
    let bx = 0, by = 0;
    k = 0;
    for (let wy = -half; wy <= half; wy++) {
      for (let wx = -half; wx <= half; wx++) {
        const it = sampleBilinear(g1, w, h, cx + wx, cy + wy) - patch0[k];
        bx += gxArr[k] * it;
        by += gyArr[k] * it;
        k++;
      }
    }
    const stepX = -(syy * bx - sxy * by) * invDet;
    const stepY = -(-sxy * bx + sxx * by) * invDet;
    dx += stepX;
    dy += stepY;
    if (stepX * stepX + stepY * stepY < 0.002) break;
  }

  return { x: x0 + dx, y: y0 + dy };
}

function trackPyramidFB(g0, h0, g1, h1, w, h, x0, y0) {
  const coarse = lkSingleLevel(h0.data, h1.data, h0.w, h0.h, x0 * 0.5, y0 * 0.5, 0, 0);
  const guessDx = coarse ? (coarse.x - x0 * 0.5) * 2.0 : 0.0;
  const guessDy = coarse ? (coarse.y - y0 * 0.5) * 2.0 : 0.0;
  const fwd = lkSingleLevel(g0, g1, w, h, x0, y0, guessDx, guessDy);
  if (!fwd) return null;
  const bwd = lkSingleLevel(g1, g0, w, h, fwd.x, fwd.y, x0 - fwd.x, y0 - fwd.y);
  if (!bwd) return null;
  const fbErr = Math.hypot(bwd.x - x0, bwd.y - y0);
  if (fbErr > 2.0) return null;
  return fwd;
}

function fitAffine(px, py, nx, ny) {
  const n = px.length;
  if (n < 4) return null;
  let mPx = 0, mPy = 0, mNx = 0, mNy = 0;
  for (let i = 0; i < n; i++) {
    mPx += px[i]; mPy += py[i];
    mNx += nx[i]; mNy += ny[i];
  }
  mPx /= n; mPy /= n; mNx /= n; mNy /= n;

  // Solve 2x2 centered normal equations for linear part [a b; c d]
  let sxx = 0, sxy = 0, syy = 0;
  let rx0 = 0, ry0 = 0, rx1 = 0, ry1 = 0;
  for (let i = 0; i < n; i++) {
    const x0 = px[i] - mPx;
    const y0 = py[i] - mPy;
    const x1 = nx[i] - mNx;
    const y1 = ny[i] - mNy;
    sxx += x0 * x0;
    sxy += x0 * y0;
    syy += y0 * y0;
    rx0 += x0 * x1;
    ry0 += y0 * x1;
    rx1 += x0 * y1;
    ry1 += y0 * y1;
  }

  const detS = sxx * syy - sxy * sxy;
  let a = 1.0, b = 0.0, c = 0.0, d = 1.0;
  if (detS > 32.0) {
    const inv = 1.0 / detS;
    a = (syy * rx0 - sxy * ry0) * inv;
    b = (-sxy * rx0 + sxx * ry0) * inv;
    c = (syy * rx1 - sxy * ry1) * inv;
    d = (-sxy * rx1 + sxx * ry1) * inv;
    const detA = a * d - b * c;
    const frob = Math.sqrt(a * a + b * b + c * c + d * d);
    // Reject degenerate/flipping/extreme-shear steps and fall back to similarity
    if (!(detA >= 0.65 && detA <= 1.55 && frob <= 2.4)) {
      const sim = fitSimilarity(px, py, nx, ny);
      if (!sim) return null;
      return { a: sim.a, b: -sim.b, c: sim.b, d: sim.a, tx: sim.tx, ty: sim.ty };
    }
  } else {
    const sim = fitSimilarity(px, py, nx, ny);
    if (!sim) return null;
    return { a: sim.a, b: -sim.b, c: sim.b, d: sim.a, tx: sim.tx, ty: sim.ty };
  }

  const tx = mNx - (a * mPx + b * mPy);
  const ty = mNy - (c * mPx + d * mPy);
  return { a, b, c, d, tx, ty };
}

function applyAffineToQuad(q, aff) {
  const { a, b, c, d, tx, ty } = aff;
  return q.map(([x, y]) => [a * x + b * y + tx, c * x + d * y + ty]);
}

function fitSimilarity(px, py, nx, ny) {
  const n = px.length;
  if (n < 3) return null;
  let mPx = 0, mPy = 0, mNx = 0, mNy = 0;
  for (let i = 0; i < n; i++) {
    mPx += px[i]; mPy += py[i];
    mNx += nx[i]; mNy += ny[i];
  }
  mPx /= n; mPy /= n; mNx /= n; mNy /= n;

  let denom = 0, numA = 0, numB = 0;
  for (let i = 0; i < n; i++) {
    const dx0 = px[i] - mPx;
    const dy0 = py[i] - mPy;
    const dx1 = nx[i] - mNx;
    const dy1 = ny[i] - mNy;
    denom += dx0 * dx0 + dy0 * dy0;
    numA += dx0 * dx1 + dy0 * dy1;
    numB += dx0 * dy1 - dy0 * dx1;
  }
  let a = 1.0, b = 0.0;
  if (denom > 8.0) {
    a = numA / denom;
    b = numB / denom;
    const sc = Math.hypot(a, b);
    const ang = Math.atan2(b, a);
    const clampedSc = Math.max(0.85, Math.min(1.18, sc));
    const clampedAng = Math.max(-0.35, Math.min(0.35, ang));
    a = clampedSc * Math.cos(clampedAng);
    b = clampedSc * Math.sin(clampedAng);
  }
  const tx = mNx - (a * mPx - b * mPy);
  const ty = mNy - (b * mPx + a * mPy);
  return { a, b, tx, ty };
}

function estimatePointBox(gray, w, h, px, py) {
  const defaultW = Math.max(28, Math.round(w * 0.34));
  const defaultH = Math.max(24, Math.round(h * 0.34));
  const bx = Math.max(2, Math.min(w - defaultW - 2, Math.round(px - defaultW * 0.5)));
  const by = Math.max(2, Math.min(h - defaultH - 2, Math.round(py - defaultH * 0.5)));
  return { x: bx, y: by, w: Math.min(defaultW, w - 4), h: Math.min(defaultH, h - 4) };
}

function searchTemplateReacquire(gray, w, h) {
  if (!refPatch || !refPatchCoarse || !refBox) return null;
  const bw = refBox.w;
  const bh = refBox.h;
  const halfW = bw * 0.5;
  const halfH = bh * 0.5;
  const minCx = Math.max(4, Math.floor(halfW * 0.6));
  const maxCx = Math.min(w - 5, Math.ceil(w - halfW * 0.6));
  const minCy = Math.max(4, Math.floor(halfH * 0.6));
  const maxCy = Math.min(h - 5, Math.ceil(h - halfH * 0.6));

  let bestScore = -1.0;
  let bestCx = w * 0.5;
  let bestCy = h * 0.5;
  let bestAng = 0.0;
  let bestScale = 1.0;

  // Coarse grid search on 12x12 patch across 3 rotations (0, +/-14 deg)
  const strideX = Math.max(5, Math.round(w / 20));
  const strideY = Math.max(5, Math.round(h / 16));
  const coarseAngles = [0.0, -0.24, 0.24];

  for (let ai = 0; ai < coarseAngles.length; ai++) {
    const ang = coarseAngles[ai];
    for (let cy = minCy; cy <= maxCy; cy += strideY) {
      for (let cx = minCx; cx <= maxCx; cx += strideX) {
        const q = rotatedBoxQuad(cx, cy, bw, bh, ang);
        const cand = extractNormalizedPatchQuad(gray, w, h, q, COARSE_N);
        if (cand.std < 5.0 || cand.gradEnergy < Math.max(1.8, refGradEnergy * 0.22)) continue;
        const sc = patchNCC(refPatchCoarse, cand.patch);
        if (sc > bestScore) {
          bestScore = sc;
          bestCx = cx;
          bestCy = cy;
          bestAng = ang;
        }
      }
    }
  }

  if (bestScore < 0.36) return null;

  // Fine local refinement on full 24x24 patch over position, angle, and mild scale
  let fineScore = -1.0;
  let fineQuad = null;
  let fineCx = bestCx;
  let fineCy = bestCy;
  const rx0 = Math.max(minCx, bestCx - strideX);
  const rx1 = Math.min(maxCx, bestCx + strideX);
  const ry0 = Math.max(minCy, bestCy - strideY);
  const ry1 = Math.min(maxCy, bestCy + strideY);
  const fineStep = Math.max(2, Math.floor(strideX / 2));
  const fineAngles = [bestAng - 0.12, bestAng, bestAng + 0.12];
  const fineScales = [0.90, 1.0, 1.12];

  for (let si = 0; si < fineScales.length; si++) {
    const scale = fineScales[si];
    for (let ai = 0; ai < fineAngles.length; ai++) {
      const ang = fineAngles[ai];
      for (let cy = ry0; cy <= ry1; cy += fineStep) {
        for (let cx = rx0; cx <= rx1; cx += fineStep) {
          const q = rotatedBoxQuad(cx, cy, bw * scale, bh * scale, ang);
          const cand = extractNormalizedPatchQuad(gray, w, h, q, PATCH_N);
          if (cand.std < 5.0 || cand.gradEnergy < Math.max(1.8, refGradEnergy * 0.22)) continue;
          const sc = patchNCC(refPatch, cand.patch);
          if (sc > fineScore) {
            fineScore = sc;
            fineQuad = q;
            fineCx = cx;
            fineCy = cy;
            bestScale = scale;
          }
        }
      }
    }
  }

  if (!fineQuad || fineScore < 0.48) return null;
  return { cx: fineCx, cy: fineCy, quad: fineQuad, score: fineScore, scale: bestScale };
}

function handleInit(msg) {
  const t0 = performance.now();
  currentEpoch = msg.epoch;
  fullW = msg.fullWidth;
  fullH = msg.fullHeight;
  trackW = msg.width;
  trackH = msg.height;
  sx = trackW / Math.max(1, fullW);
  sy = trackH / Math.max(1, fullH);

  const gray = new Uint8Array(msg.gray);
  prevGray = gray;
  prevHalf = downsampleHalf(gray, trackW, trackH);

  let box;
  if (msg.bbox && msg.bbox.length === 4 && msg.bbox[2] >= 8 && msg.bbox[3] >= 8) {
    box = {
      x: Math.max(1, Math.min(trackW - 10, msg.bbox[0] * sx)),
      y: Math.max(1, Math.min(trackH - 10, msg.bbox[1] * sy)),
      w: Math.max(10, Math.min(trackW - 2, msg.bbox[2] * sx)),
      h: Math.max(10, Math.min(trackH - 2, msg.bbox[3] * sy)),
    };
    if (box.x + box.w >= trackW - 1) box.w = Math.max(8, trackW - 2 - box.x);
    if (box.y + box.h >= trackH - 1) box.h = Math.max(8, trackH - 2 - box.y);
  } else if (msg.point && msg.point.length === 2) {
    box = estimatePointBox(gray, trackW, trackH, msg.point[0] * sx, msg.point[1] * sy);
  } else {
    box = {
      x: Math.round(trackW * 0.25),
      y: Math.round(trackH * 0.25),
      w: Math.round(trackW * 0.5),
      h: Math.round(trackH * 0.5),
    };
  }

  const initQuad = boxToQuad(box);
  const info = extractNormalizedPatchQuad(gray, trackW, trackH, initQuad, PATCH_N);
  const infoCoarse = extractNormalizedPatchQuad(gray, trackW, trackH, initQuad, COARSE_N);
  const corners = detectCornersInQuad(gray, trackW, trackH, initQuad, 64);

  // Reject featureless / pitch-black frames cleanly (never falsely lock!)
  if (info.std < 5.0 || info.gradEnergy < 1.8 || corners.n < 6) {
    active = true;
    refBox = box;
    refArea = Math.max(1.0, box.w * box.h);
    curQuad = null;
    refPatch = null;
    refPatchCoarse = null;
    refGradEnergy = 0;
    numPts = 0;
    state = "lost";
    self.postMessage({
      type: "result",
      epoch: currentEpoch,
      seq: msg.seq || 0,
      state: "lost",
      inliers: 0,
      quad: null,
      trackMs: performance.now() - t0,
    });
    return;
  }

  active = true;
  refBox = { ...box };
  refArea = Math.max(1.0, box.w * box.h);
  curQuad = initQuad;
  refPatch = info.patch;
  refPatchCoarse = infoCoarse.patch;
  prevPatch = info.patch;
  refGradEnergy = info.gradEnergy;
  ptsX = corners.xs;
  ptsY = corners.ys;
  numPts = corners.n;
  framesSinceReplenish = 0;
  recoverCand = null;
  state = "tracking";

  self.postMessage({
    type: "result",
    epoch: currentEpoch,
    seq: msg.seq || 0,
    state: "tracking",
    inliers: numPts,
    quad: quadToFullQuad(curQuad),
    trackMs: performance.now() - t0,
  });
}

function handleTrack(msg) {
  if (msg.epoch !== currentEpoch) return;
  const t0 = performance.now();
  const gray = new Uint8Array(msg.gray);
  const half = downsampleHalf(gray, trackW, trackH);

  if (!active || !refPatch || !refBox) {
    prevGray = gray;
    prevHalf = half;
    self.postMessage({
      type: "result",
      epoch: currentEpoch,
      seq: msg.seq || 0,
      state: active ? "lost" : "idle",
      inliers: 0,
      quad: null,
      trackMs: performance.now() - t0,
    });
    return;
  }

  if (state === "tracking" && curQuad && prevGray && prevHalf && numPts >= 6) {
    const goodPrevX = [];
    const goodPrevY = [];
    const goodNewX = [];
    const goodNewY = [];
    const dxs = [];
    const dys = [];

    for (let i = 0; i < numPts; i++) {
      const res = trackPyramidFB(prevGray, prevHalf, gray, half, trackW, trackH, ptsX[i], ptsY[i]);
      if (res) {
        dxs.push(res.x - ptsX[i]);
        dys.push(res.y - ptsY[i]);
        goodPrevX.push(ptsX[i]);
        goodPrevY.push(ptsY[i]);
        goodNewX.push(res.x);
        goodNewY.push(res.y);
      }
    }

    let ok = false;
    let inlierCount = 0;
    let nextQuad = null;
    let nextPatch = null;
    let finalNewX = null;
    let finalNewY = null;

    if (dxs.length >= 6) {
      // Pass 1: coarse median translation outlier gate
      const sortedDx = dxs.slice().sort((a, b) => a - b);
      const sortedDy = dys.slice().sort((a, b) => a - b);
      const medDx = sortedDx[sortedDx.length >> 1];
      const medDy = sortedDy[sortedDy.length >> 1];

      const p1Px = [], p1Py = [], p1Nx = [], p1Ny = [];
      for (let i = 0; i < dxs.length; i++) {
        if (Math.hypot(dxs[i] - medDx, dys[i] - medDy) <= 5.0) {
          p1Px.push(goodPrevX[i]);
          p1Py.push(goodPrevY[i]);
          p1Nx.push(goodNewX[i]);
          p1Ny.push(goodNewY[i]);
        }
      }

      if (p1Px.length >= 6) {
        // Pass 2: closed-form 6-DoF 2D affine (handles perspective foreshortening + tilt) + residual pruning
        const aff1 = fitAffine(p1Px, p1Py, p1Nx, p1Ny);
        if (aff1) {
          const p2Px = [], p2Py = [], p2Nx = [], p2Ny = [];
          for (let i = 0; i < p1Px.length; i++) {
            const predX = aff1.a * p1Px[i] + aff1.b * p1Py[i] + aff1.tx;
            const predY = aff1.c * p1Px[i] + aff1.d * p1Py[i] + aff1.ty;
            if (Math.hypot(p1Nx[i] - predX, p1Ny[i] - predY) <= 2.4) {
              p2Px.push(p1Px[i]);
              p2Py.push(p1Py[i]);
              p2Nx.push(p1Nx[i]);
              p2Ny.push(p1Ny[i]);
            }
          }

          inlierCount = p2Px.length;
          if (inlierCount >= 6) {
            const aff2 = fitAffine(p2Px, p2Py, p2Nx, p2Ny) || aff1;
            const candQuad = applyAffineToQuad(curQuad, aff2);
            const area = quadArea(candQuad);
            const [cx, cy] = quadCentroid(candQuad);
            const geomOk =
              area >= refArea * 0.20 &&
              area <= refArea * 3.5 &&
              cx >= -trackW * 0.15 &&
              cx <= trackW * 1.15 &&
              cy >= -trackH * 0.15 &&
              cy <= trackH * 1.15;

            if (geomOk) {
              const curInfo = extractNormalizedPatchQuad(gray, trackW, trackH, candQuad, PATCH_N);
              const nccRef = patchNCC(refPatch, curInfo.patch);
              const nccPrev = prevPatch ? patchNCC(prevPatch, curInfo.patch) : nccRef;
              const gradOk = curInfo.std >= 4.5 && curInfo.gradEnergy >= Math.max(1.5, refGradEnergy * 0.20);
              if (gradOk && (nccRef >= 0.25 || (nccPrev >= 0.65 && inlierCount >= 16 && nccRef >= 0.05))) {
                ok = true;
                nextQuad = candQuad;
                nextPatch = curInfo.patch;
                finalNewX = Float32Array.from(p2Nx);
                finalNewY = Float32Array.from(p2Ny);
              }
            }
          }
        }
      }
    }

    if (ok && nextQuad) {
      curQuad = nextQuad;
      prevPatch = nextPatch || prevPatch;
      ptsX = finalNewX;
      ptsY = finalNewY;
      numPts = inlierCount;
      framesSinceReplenish++;
      if (numPts < 26 || framesSinceReplenish >= 12) {
        const fresh = detectCornersInQuad(gray, trackW, trackH, curQuad, 64);
        if (fresh.n >= 8) {
          ptsX = fresh.xs;
          ptsY = fresh.ys;
          numPts = fresh.n;
          framesSinceReplenish = 0;
        }
      }
      prevGray = gray;
      prevHalf = half;
      self.postMessage({
        type: "result",
        epoch: currentEpoch,
        seq: msg.seq || 0,
        state: "tracking",
        inliers: numPts,
        quad: quadToFullQuad(curQuad),
        trackMs: performance.now() - t0,
      });
      return;
    }

    // Lost tracking -> immediately suppress overlay!
    state = "lost";
    curQuad = null;
    recoverCand = null;
    numPts = 0;
    prevGray = gray;
    prevHalf = half;
    self.postMessage({
      type: "result",
      epoch: currentEpoch,
      seq: msg.seq || 0,
      state: "lost",
      inliers: 0,
      quad: null,
      trackMs: performance.now() - t0,
    });
    return;
  }

  // Lost or recovering -> run fast rotated/scaled NCC template search for automatic re-acquisition
  const match = searchTemplateReacquire(gray, trackW, trackH);
  prevGray = gray;
  prevHalf = half;

  if (match) {
    if (state === "recovering" && recoverCand) {
      const dist = Math.hypot(match.cx - recoverCand.cx, match.cy - recoverCand.cy);
      if (dist <= Math.max(trackW, trackH) * 0.16) {
        // 2 consecutive consistent detections -> RE-ACQUIRED!
        const corners = detectCornersInQuad(gray, trackW, trackH, match.quad, 64);
        if (corners.n >= 6) {
          state = "tracking";
          curQuad = match.quad;
          prevPatch = extractNormalizedPatchQuad(gray, trackW, trackH, curQuad, PATCH_N).patch;
          ptsX = corners.xs;
          ptsY = corners.ys;
          numPts = corners.n;
          recoverCand = null;
          framesSinceReplenish = 0;
          self.postMessage({
            type: "result",
            epoch: currentEpoch,
            seq: msg.seq || 0,
            state: "tracking",
            inliers: numPts,
            quad: quadToFullQuad(curQuad),
            trackMs: performance.now() - t0,
          });
          return;
        }
      }
    }
    state = "recovering";
    recoverCand = match;
    self.postMessage({
      type: "result",
      epoch: currentEpoch,
      seq: msg.seq || 0,
      state: "recovering",
      inliers: 0,
      quad: null,
      trackMs: performance.now() - t0,
    });
    return;
  }

  state = "lost";
  recoverCand = null;
  self.postMessage({
    type: "result",
    epoch: currentEpoch,
    seq: msg.seq || 0,
    state: "lost",
    inliers: 0,
    quad: null,
    trackMs: performance.now() - t0,
  });
}

self.onmessage = function (ev) {
  const msg = ev.data;
  if (!msg || !msg.type) return;
  if (msg.type === "clear") {
    currentEpoch = msg.epoch || currentEpoch + 1;
    active = false;
    state = "idle";
    curQuad = null;
    refBox = null;
    refPatch = null;
    refPatchCoarse = null;
    prevPatch = null;
    numPts = 0;
    recoverCand = null;
    self.postMessage({ type: "cleared", epoch: currentEpoch });
  } else if (msg.type === "init") {
    handleInit(msg);
  } else if (msg.type === "track") {
    handleTrack(msg);
  }
};
