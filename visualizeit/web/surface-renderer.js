/* Fast Multi-Mode Surface Renderer (planar, cylinder, mesh_grid, mesh_6dof x luminance, normal, ink) */
(function (global) {
  "use strict";

  function clampNum(v, lo, hi, fallback) {
    const n = Number(v);
    if (!Number.isFinite(n)) return fallback;
    return Math.max(lo, Math.min(hi, n));
  }

  function prepareTransformedArtwork(sourceImgOrCanvas, placement, workCanvas) {
    const size = 384;
    const canvas = workCanvas || document.createElement("canvas");
    if (canvas.width !== size) canvas.width = size;
    if (canvas.height !== size) canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (!ctx) return canvas;

    const scale = clampNum(placement.scale, 0.1, 5.0, 1.0);
    const rotRad = (clampNum(placement.rotation_deg, -360, 360, 0.0) * Math.PI) / 180.0;
    const ox = clampNum(placement.offset_x, -1.0, 1.0, 0.0) * size;
    const oy = clampNum(placement.offset_y, -1.0, 1.0, 0.0) * size;

    ctx.clearRect(0, 0, size, size);
    ctx.save();
    ctx.translate(size * 0.5 + ox, size * 0.5 + oy);
    ctx.rotate(rotRad);
    ctx.scale(scale, scale);
    ctx.drawImage(sourceImgOrCanvas, -size * 0.5, -size * 0.5, size, size);
    ctx.restore();
    return canvas;
  }

  function renderSurfaceOverlay(ctx, quad, artworkSource, placement, scratchCanvas) {
    if (!quad || quad.length !== 4 || !artworkSource) return;
    const opacity = clampNum(placement.opacity, 0.0, 1.0, 0.9);
    if (opacity <= 0.001) return;

    const x0 = quad[0][0], y0 = quad[0][1];
    const x1 = quad[1][0], y1 = quad[1][1];
    const x2 = quad[2][0], y2 = quad[2][1];
    const x3 = quad[3][0], y3 = quad[3][1];
    const w = Math.hypot(x1 - x0, y1 - y0);
    const h = Math.hypot(x3 - x0, y3 - y0);
    if (!Number.isFinite(w) || !Number.isFinite(h) || w < 4 || h < 4) return;

    const transformed = prepareTransformedArtwork(artworkSource, placement, scratchCanvas);
    const texW = transformed.width;
    const texH = transformed.height;
    const wrapMode = placement.wrap_mode || "planar";
    const blendMode = placement.blend_mode || "luminance";

    ctx.save();
    ctx.globalAlpha = opacity;
    if (blendMode === "ink") {
      ctx.globalCompositeOperation = "multiply";
    } else if (blendMode === "luminance") {
      ctx.globalCompositeOperation = "hard-light";
    } else {
      ctx.globalCompositeOperation = "source-over";
    }

    const angle = Math.atan2(y1 - y0, x1 - x0);
    ctx.translate(x0, y0);
    ctx.rotate(angle);

    if (wrapMode === "cylinder") {
      // Multi-strip cylindrical foreshortening + subtle edge shading
      const slices = 16;
      const halfArc = (72 * Math.PI) / 180;
      const sinMax = Math.sin(halfArc);
      for (let i = 0; i < slices; i++) {
        const u0 = i / slices;
        const u1 = (i + 1) / slices;
        const t0 = (u0 * 2 - 1) * halfArc;
        const t1 = (u1 * 2 - 1) * halfArc;
        const dx0 = (0.5 + 0.5 * (Math.sin(t0) / sinMax)) * w;
        const dx1 = (0.5 + 0.5 * (Math.sin(t1) / sinMax)) * w;
        const sw = Math.max(0.5, dx1 - dx0);
        const midCos = Math.cos(0.5 * (t0 + t1));
        const bow = 0.035 * h * (midCos - Math.cos(halfArc));
        ctx.drawImage(
          transformed,
          u0 * texW,
          0,
          (u1 - u0) * texW,
          texH,
          dx0,
          bow,
          sw + 0.6,
          h - bow * 0.5
        );
      }
      // Subtle cylindrical specular/edge roll-off
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, "rgba(0,0,0,0.28)");
      grad.addColorStop(0.28, "rgba(255,255,255,0.08)");
      grad.addColorStop(0.5, "rgba(255,255,255,0.0)");
      grad.addColorStop(0.75, "rgba(0,0,0,0.08)");
      grad.addColorStop(1, "rgba(0,0,0,0.30)");
      ctx.globalCompositeOperation = "source-atop";
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);
    } else if (wrapMode === "mesh_grid") {
      const cols = 4;
      const rows = 4;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const u0 = c / cols;
          const v0 = r / rows;
          const uc = (c + 0.5) / cols - 0.5;
          const vc = (r + 0.5) / rows - 0.5;
          const bulge = 1.0 + 0.04 * (0.5 - (uc * uc + vc * vc));
          const dw = (w / cols) * bulge;
          const dh = (h / rows) * bulge;
          ctx.drawImage(
            transformed,
            u0 * texW,
            v0 * texH,
            texW / cols,
            texH / rows,
            u0 * w,
            v0 * h,
            dw + 0.5,
            dh + 0.5
          );
        }
      }
    } else {
      // planar or mesh_6dof preview
      ctx.drawImage(transformed, 0, 0, texW, texH, 0, 0, w, h);
    }

    ctx.restore();
  }

  global.VisualizeITRenderer = {
    clampNum,
    renderSurfaceOverlay,
  };
})(window);
