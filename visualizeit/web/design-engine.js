/* Procedural Design Engine & Prompt Parser (browser parity with visualizeit/designs.py) */
(function (global) {
  "use strict";

  const NAMED_COLORS = {
    cream: "#f4ebd8",
    ink: "#202820",
    green: "#42634b",
    orange: "#dc693f",
    pink: "#e5a1ac",
    blue: "#376aa0",
    red: "#bd3f3f",
    yellow: "#e9bb42",
    purple: "#795b9b",
    black: "#111111",
    white: "#ffffff",
    navy: "#203957",
    teal: "#287e80",
    brown: "#805b41",
    gray: "#808080",
    grey: "#808080",
    cyan: "#00bcd4",
    magenta: "#d81b60",
    gold: "#f4b400",
    coral: "#ff6f61",
    lime: "#7cb342",
    indigo: "#3949ab",
    violet: "#8e24aa",
    silver: "#c0c0c0",
    maroon: "#800000",
    olive: "#808000",
    turquoise: "#1abc9c",
    crimson: "#dc143c",
  };

  const VALID_PATTERNS = ["solid", "stripes", "checker", "dots", "waves", "grid"];
  const PATTERN_TOKEN_RE = /\b(solid|stripes?|striped|checker|checkerboard|checkered|dots?|dotted|waves?|wavy|grid)\b/;
  const PATTERN_ALIASES = {
    stripe: "stripes",
    striped: "stripes",
    checkerboard: "checker",
    checkered: "checker",
    dot: "dots",
    dotted: "dots",
    wave: "waves",
    wavy: "waves",
  };

  function parseDesignPrompt(prompt) {
    const raw = String(prompt || "").trim();
    const unquoted = raw.replace(/"[^"\n]*"|'[^'\n]*'/g, " ");
    const lower = unquoted.toLowerCase();

    // Pattern match (exact parity with visualizeit/designs.py)
    let pattern = "stripes";
    const pm = lower.match(PATTERN_TOKEN_RE);
    if (pm && pm[1]) {
      pattern = PATTERN_ALIASES[pm[1]] || pm[1];
    } else if (lower.includes("polka") || lower.includes("circle")) {
      pattern = "dots";
    }

    // Extract quoted text: text "..." or '...' or suffix text: ...
    let text = "";
    const textMatch = raw.match(/"([^"\n]*)"|'([^'\n]*)'/);
    if (textMatch) {
      text = (textMatch[1] !== undefined ? textMatch[1] : textMatch[2] || "").trim().slice(0, 80);
    } else {
      const suf = raw.match(/\b(?:text|say|saying)\s*:?\s+(.+)$/i);
      if (suf && suf[1]) {
        text = suf[1].trim().slice(0, 80);
      }
    }

    // Extract colors in order of appearance (hex + named)
    const found = [];
    const tokens = unquoted.split(/[\s,;]+/);
    for (const tok of tokens) {
      const clean = tok.trim().toLowerCase().replace(/[^a-z0-9#]/g, "");
      if (!clean) continue;
      if (/^#[0-9a-f]{6}$/i.test(clean)) {
        if (!found.includes(clean)) found.push(clean);
      } else if (/^#[0-9a-f]{3}$/i.test(clean)) {
        const exp = "#" + clean[1] + clean[1] + clean[2] + clean[2] + clean[3] + clean[3];
        if (!found.includes(exp)) found.push(exp);
      } else if (NAMED_COLORS[clean]) {
        const hex = NAMED_COLORS[clean];
        if (!found.includes(hex)) found.push(hex);
      }
    }
    const colors = found.length >= 2
      ? found.slice(0, 5)
      : found.length === 1
      ? [found[0], "#ffffff"]
      : ["#f4ebd8", "#202820", "#dc693f"];

    // Optional scale cue
    let scale = null;
    const scaleMatch = lower.match(/\bscale\s*:?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))/);
    if (scaleMatch) {
      const sv = parseFloat(scaleMatch[1]);
      if (Number.isFinite(sv) && sv >= 0.1 && sv <= 5.0) scale = sv;
    }

    return {
      pattern,
      colors,
      text,
      prompt: raw,
      scale,
      width: 512,
      height: 512,
    };
  }

  function renderDesignToCanvas(recipe, targetCanvas) {
    const w = Math.max(64, Number(recipe.width) || 512);
    const h = Math.max(64, Number(recipe.height) || 512);
    const canvas = targetCanvas || document.createElement("canvas");
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return canvas;

    const pattern = VALID_PATTERNS.includes(recipe.pattern) ? recipe.pattern : "stripes";
    const colors = Array.isArray(recipe.colors) && recipe.colors.length >= 1
      ? recipe.colors
      : ["#ff595e", "#1982c4", "#ffca3a"];
    const c0 = colors[0] || "#ff595e";
    const c1 = colors[1] || "#1982c4";
    const c2 = colors[2] || c0;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = c0;
    ctx.fillRect(0, 0, w, h);

    if (pattern === "stripes") {
      const bands = Math.max(6, colors.length * 3);
      const bw = w / bands;
      for (let i = 0; i < bands; i++) {
        ctx.fillStyle = colors[i % colors.length];
        ctx.fillRect(Math.floor(i * bw), 0, Math.ceil(bw + 1), h);
      }
    } else if (pattern === "checker") {
      const cells = 8;
      const cw = w / cells;
      const ch = h / cells;
      for (let r = 0; r < cells; r++) {
        for (let c = 0; c < cells; c++) {
          ctx.fillStyle = colors[(r + c) % colors.length];
          ctx.fillRect(Math.floor(c * cw), Math.floor(r * ch), Math.ceil(cw + 1), Math.ceil(ch + 1));
        }
      }
    } else if (pattern === "dots") {
      const cols = 7;
      const rows = 7;
      const rad = Math.min(w, h) / (cols * 2.9);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const cx = ((c + 0.5) / cols) * w;
          const cy = ((r + 0.5) / rows) * h;
          ctx.fillStyle = colors[1 + ((r + c) % Math.max(1, colors.length - 1))] || c1;
          ctx.beginPath();
          ctx.arc(cx, cy, rad, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    } else if (pattern === "waves") {
      const strips = 8;
      const sh = h / strips;
      for (let s = 0; s < strips; s++) {
        ctx.fillStyle = colors[s % colors.length];
        ctx.beginPath();
        const baseY = s * sh;
        ctx.moveTo(0, h);
        ctx.lineTo(0, baseY);
        for (let x = 0; x <= w; x += 8) {
          const y = baseY + Math.sin((x / w) * Math.PI * 4 + s * 0.9) * (sh * 0.32);
          ctx.lineTo(x, y);
        }
        ctx.lineTo(w, h);
        ctx.closePath();
        ctx.fill();
      }
    } else if (pattern === "grid") {
      const step = Math.max(16, Math.round(Math.min(w, h) / 10));
      ctx.strokeStyle = c1;
      ctx.lineWidth = Math.max(2, Math.round(Math.min(w, h) * 0.012));
      for (let x = 0; x <= w; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y <= h; y += step) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }
      if (colors.length >= 3) {
        ctx.fillStyle = c2;
        for (let x = step; x < w; x += step * 2) {
          for (let y = step; y < h; y += step * 2) {
            ctx.fillRect(x - 4, y - 4, 8, 8);
          }
        }
      }
    }

    const label = String(recipe.text || "").trim();
    if (label) {
      const fontSize = Math.max(20, Math.round(Math.min(w, h) * 0.16));
      ctx.save();
      ctx.font = `800 ${fontSize}px Inter, -apple-system, BlinkMacSystemFont, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.lineWidth = Math.max(4, Math.round(fontSize * 0.12));
      ctx.strokeStyle = "rgba(18, 20, 24, 0.85)";
      ctx.strokeText(label, w * 0.5, h * 0.5, w * 0.88);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, w * 0.5, h * 0.5, w * 0.88);
      ctx.restore();
    }

    return canvas;
  }

  global.VisualizeITDesign = {
    VALID_PATTERNS,
    parseDesignPrompt,
    renderDesignToCanvas,
  };
})(window);
