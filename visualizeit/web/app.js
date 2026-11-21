/* Unified Browser Video & Live Camera Studio Controller (Uncapped Web Worker + REST MP4 Export) */
(function () {
  "use strict";

  const { parseDesignPrompt, renderDesignToCanvas } = window.VisualizeITDesign;
  const { clampNum, renderSurfaceOverlay } = window.VisualizeITRenderer;

  // Global read-only telemetry contract (PROJECT.md §6 — all 11 required keys)
  const telemetry = {
    mode: "video",
    playing: false,
    currentFps: 0.0,
    medianFrameMs: 0.0,
    p95FrameMs: 0.0,
    processedFrames: 0,
    trackingState: "idle",
    inliers: 0,
    wrapMode: "planar",
    blendMode: "luminance",
    lastExportReport: null,
  };
  window.__VISUALIZEIT_TELEMETRY__ = telemetry;

  // DOM references
  const modeVideoBtn = document.getElementById("mode-video-btn");
  const modeCameraBtn = document.getElementById("mode-camera-btn");
  const demoSelect = document.getElementById("demo-clip-select");
  const loadDemoBtn = document.getElementById("load-demo-btn");
  const videoFileInput = document.getElementById("video-file-input");
  const playPauseBtn = document.getElementById("play-pause-btn");
  const exportMp4Btn = document.getElementById("export-mp4-btn");
  const exportDownloadLink = document.getElementById("export-download-link");

  const stageCanvas = document.getElementById("stage-canvas");
  const stageCtx = stageCanvas.getContext("2d", { willReadFrequently: false });
  const stageStatus = document.getElementById("stage-status");

  const selectModePointBtn = document.getElementById("select-mode-point-btn");
  const selectModeBoxBtn = document.getElementById("select-mode-box-btn");
  const autoSelectBtn = document.getElementById("auto-select-btn");
  const clearSelectionBtn = document.getElementById("clear-selection-btn");

  const designPreviewCanvas = document.getElementById("design-preview-canvas");
  const designPromptInput = document.getElementById("design-prompt-input");
  const applyPromptBtn = document.getElementById("apply-prompt-btn");
  const patternSelect = document.getElementById("pattern-select");
  const designImageInput = document.getElementById("design-image-input");

  const wrapModeSelect = document.getElementById("wrap-mode-select");
  const blendModeSelect = document.getElementById("blend-mode-select");
  const opacitySlider = document.getElementById("opacity-slider");
  const scaleSlider = document.getElementById("scale-slider");
  const rotationSlider = document.getElementById("rotation-slider");
  const opacityVal = document.getElementById("opacity-val");
  const scaleVal = document.getElementById("scale-val");
  const rotationVal = document.getElementById("rotation-val");

  const hudFps = document.getElementById("hud-fps");
  const hudLatency = document.getElementById("hud-latency-ms");
  const hudTrackingState = document.getElementById("hud-tracking-state");
  const hudProcessedFrames = document.getElementById("hud-processed-frames");

  const sourceVideo = document.getElementById("source-video");
  const cameraVideo = document.getElementById("camera-video");

  // Offscreen canvases for artwork, transforms, and fast worker grayscale extraction
  const proceduralCanvas = document.createElement("canvas");
  proceduralCanvas.width = 512;
  proceduralCanvas.height = 512;
  const overlayScratchCanvas = document.createElement("canvas");
  const lumaCanvas = document.createElement("canvas");
  const lumaCtx = lumaCanvas.getContext("2d", { willReadFrequently: true });

  // Application state
  let demoMetaById = {};
  let currentVideoId = null;
  let videoLoaded = false;
  let posterFrame0Image = null;
  let activeCameraStream = null;

  let selectTool = "point"; // "point" | "box"
  let currentSelection = null; // { frame_index, positive_points, negative_points, bbox }
  let dragStart = null;
  let dragCurrent = null;

  let currentDesignRecipe = {
    pattern: "stripes",
    colors: ["#ff595e", "#1982c4", "#ffca3a"],
    text: "",
    prompt: "",
    width: 512,
    height: 512,
  };
  let currentDesignId = null; // non-null only when custom uploaded image is active
  let customArtworkImage = null;

  let currentPlacement = {
    wrap_mode: "planar",
    blend_mode: "luminance",
    opacity: 0.9,
    scale: 1.0,
    rotation_deg: 0.0,
    offset_x: 0.0,
    offset_y: 0.0,
  };

  // Worker & uncapped frame loop state
  let trackerWorker = null;
  let workerEpoch = 1;
  let modeEpoch = 1;
  let workerBusy = false;
  let pendingSeq = 0;
  let pendingStartMs = 0;
  let latestQuad = null;
  let loopRafId = null;
  let loopTimerId = null;

  // Rolling latency & FPS windows
  const latencyWindow = [];
  const timestampWindow = [];

  function setStatus(message, isError) {
    if (!stageStatus) return;
    stageStatus.textContent = message;
    stageStatus.classList.toggle("is-error", Boolean(isError));
  }

  function updateHud() {
    hudFps.textContent = `${telemetry.currentFps.toFixed(1)}`;
    hudLatency.textContent = `${telemetry.medianFrameMs.toFixed(1)} ms`;
    hudTrackingState.textContent = telemetry.trackingState;
    hudTrackingState.className = `metric-val state-${telemetry.trackingState}`;
    hudProcessedFrames.textContent = String(telemetry.processedFrames);
    playPauseBtn.textContent = telemetry.playing ? "Pause" : "Play";
  }

  function percentile(arr, p) {
    if (!arr.length) return 0.0;
    const sorted = arr.slice().sort((a, b) => a - b);
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((p / 100) * (sorted.length - 1))));
    return sorted[idx];
  }

  function recordFrameMetrics(frameMs, stateStr, inliers) {
    const now = performance.now();
    const safeMs = Number.isFinite(frameMs) && frameMs > 0 ? frameMs : 0.5;
    latencyWindow.push(safeMs);
    if (latencyWindow.length > 60) latencyWindow.shift();

    timestampWindow.push(now);
    while (timestampWindow.length > 60 || (timestampWindow.length > 2 && now - timestampWindow[0] > 1500)) {
      timestampWindow.shift();
    }

    telemetry.processedFrames += 1;
    telemetry.medianFrameMs = Number(percentile(latencyWindow, 50).toFixed(2));
    telemetry.p95FrameMs = Number(percentile(latencyWindow, 95).toFixed(2));

    if (timestampWindow.length >= 2) {
      const spanS = (timestampWindow[timestampWindow.length - 1] - timestampWindow[0]) / 1000.0;
      const fps = spanS > 0.01 ? (timestampWindow.length - 1) / spanS : 1000.0 / Math.max(1.0, safeMs);
      telemetry.currentFps = Number(Math.min(240.0, Math.max(0.0, fps)).toFixed(1));
    } else {
      telemetry.currentFps = Number(Math.min(240.0, 1000.0 / Math.max(1.0, safeMs)).toFixed(1));
    }

    telemetry.trackingState = String(stateStr || "idle");
    telemetry.inliers = Math.max(0, Math.round(Number(inliers) || 0));
    updateHud();
  }

  function resetMetrics() {
    latencyWindow.length = 0;
    timestampWindow.length = 0;
    telemetry.currentFps = 0.0;
    telemetry.medianFrameMs = 0.0;
    telemetry.p95FrameMs = 0.0;
    telemetry.processedFrames = 0;
    updateHud();
  }

  function getActiveArtworkSource() {
    return customArtworkImage || proceduralCanvas;
  }

  function refreshDesignPreview() {
    if (!customArtworkImage) {
      renderDesignToCanvas(currentDesignRecipe, proceduralCanvas);
    }
    const pctx = designPreviewCanvas.getContext("2d");
    if (!pctx) return;
    pctx.clearRect(0, 0, designPreviewCanvas.width, designPreviewCanvas.height);
    const src = getActiveArtworkSource();
    if (src) {
      pctx.drawImage(src, 0, 0, designPreviewCanvas.width, designPreviewCanvas.height);
    }
    drawCurrentStageFrame();
  }

  function getActiveVideoEl() {
    return telemetry.mode === "camera" ? cameraVideo : sourceVideo;
  }

  function getDrawableMediaSource() {
    if (telemetry.mode === "video" && !telemetry.playing && posterFrame0Image && posterFrame0Image.naturalWidth > 0) {
      return {
        el: posterFrame0Image,
        w: posterFrame0Image.naturalWidth,
        h: posterFrame0Image.naturalHeight,
      };
    }
    const vid = getActiveVideoEl();
    if (vid && vid.readyState >= 2 && vid.videoWidth > 0 && vid.videoHeight > 0) {
      return { el: vid, w: vid.videoWidth, h: vid.videoHeight };
    }
    if (telemetry.mode === "video" && posterFrame0Image && posterFrame0Image.naturalWidth > 0) {
      return {
        el: posterFrame0Image,
        w: posterFrame0Image.naturalWidth,
        h: posterFrame0Image.naturalHeight,
      };
    }
    return null;
  }

  function drawCurrentStageFrame() {
    const src = getDrawableMediaSource();
    if (!src) return;
    if (stageCanvas.width !== src.w || stageCanvas.height !== src.h) {
      stageCanvas.width = src.w;
      stageCanvas.height = src.h;
    }
    stageCtx.drawImage(src.el, 0, 0, stageCanvas.width, stageCanvas.height);

    if (telemetry.trackingState === "tracking" && latestQuad) {
      renderSurfaceOverlay(
        stageCtx,
        latestQuad,
        getActiveArtworkSource(),
        currentPlacement,
        overlayScratchCanvas
      );
    }

    // Draw interactive box drag preview if dragging
    if (dragStart && dragCurrent && selectTool === "box") {
      const x = Math.min(dragStart.x, dragCurrent.x);
      const y = Math.min(dragStart.y, dragCurrent.y);
      const w = Math.abs(dragCurrent.x - dragStart.x);
      const h = Math.abs(dragCurrent.y - dragStart.y);
      stageCtx.save();
      stageCtx.strokeStyle = "#ffca3a";
      stageCtx.lineWidth = 2;
      stageCtx.setLineDash([6, 4]);
      stageCtx.strokeRect(x, y, w, h);
      stageCtx.restore();
    }
  }

  function extractGrayscaleForWorker(src) {
    const vw = src.w || 640;
    const vh = src.h || 480;
    const targetW = Math.min(200, vw);
    const targetH = Math.max(24, Math.round(vh * (targetW / vw)));
    if (lumaCanvas.width !== targetW || lumaCanvas.height !== targetH) {
      lumaCanvas.width = targetW;
      lumaCanvas.height = targetH;
    }
    lumaCtx.drawImage(src.el, 0, 0, targetW, targetH);
    const rgba = lumaCtx.getImageData(0, 0, targetW, targetH).data;
    const gray = new Uint8Array(targetW * targetH);
    for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
      gray[i] = (rgba[j] * 77 + rgba[j + 1] * 150 + rgba[j + 2] * 29) >> 8;
    }
    return { gray, width: targetW, height: targetH, fullWidth: vw, fullHeight: vh };
  }

  function initWorker() {
    if (trackerWorker) {
      trackerWorker.terminate();
    }
    trackerWorker = new Worker("/static/video-tracker-worker.js");
    trackerWorker.onmessage = function (ev) {
      const msg = ev.data;
      if (!msg) return;
      if (msg.epoch !== workerEpoch) return;
      workerBusy = false;

      if (msg.type === "cleared") {
        latestQuad = null;
        telemetry.trackingState = "idle";
        telemetry.inliers = 0;
        updateHud();
        drawCurrentStageFrame();
        return;
      }

      if (msg.type === "result") {
        const frameMs = Math.max(0.25, performance.now() - pendingStartMs);
        latestQuad = msg.state === "tracking" ? msg.quad : null;
        if (telemetry.playing) {
          recordFrameMetrics(frameMs, msg.state, msg.inliers);
        } else {
          telemetry.trackingState = String(msg.state || "idle");
          telemetry.inliers = Math.max(0, Math.round(Number(msg.inliers) || 0));
          updateHud();
        }
        drawCurrentStageFrame();
      }
    };
    trackerWorker.onerror = function (err) {
      workerBusy = false;
      setStatus(`Tracker worker error: ${err.message || "unknown"}`, true);
    };
  }

  function clearTrackingSelection() {
    workerEpoch += 1;
    workerBusy = false;
    currentSelection = null;
    latestQuad = null;
    telemetry.trackingState = "idle";
    telemetry.inliers = 0;
    updateHud();
    if (trackerWorker) {
      trackerWorker.postMessage({ type: "clear", epoch: workerEpoch });
    }
    drawCurrentStageFrame();
  }

  function submitSelectionToWorker(selObj) {
    const src = getDrawableMediaSource();
    if (!src) {
      setStatus("Load a video or start Live Camera before selecting a surface.", false);
      return;
    }
    currentSelection = selObj;
    workerEpoch += 1;
    workerBusy = true;
    pendingSeq += 1;
    pendingStartMs = performance.now();

    const frame = extractGrayscaleForWorker(src);
    const point =
      selObj.positive_points && selObj.positive_points.length > 0 ? selObj.positive_points[0] : null;
    trackerWorker.postMessage(
      {
        type: "init",
        epoch: workerEpoch,
        seq: pendingSeq,
        gray: frame.gray.buffer,
        width: frame.width,
        height: frame.height,
        fullWidth: frame.fullWidth,
        fullHeight: frame.fullHeight,
        point,
        bbox: selObj.bbox || null,
      },
      [frame.gray.buffer]
    );
  }

  function stopFramePump() {
    if (loopRafId !== null) {
      cancelAnimationFrame(loopRafId);
      loopRafId = null;
    }
    if (loopTimerId !== null) {
      clearTimeout(loopTimerId);
      loopTimerId = null;
    }
  }

  function pumpOneFrame() {
    if (!telemetry.playing) return;
    const src = getDrawableMediaSource();
    if (src) {
      if (!workerBusy && trackerWorker) {
        if (currentSelection) {
          drawCurrentStageFrame();
          workerBusy = true;
          pendingSeq += 1;
          pendingStartMs = performance.now();
          const frame = extractGrayscaleForWorker(src);
          trackerWorker.postMessage(
            {
              type: "track",
              epoch: workerEpoch,
              seq: pendingSeq,
              gray: frame.gray.buffer,
              width: frame.width,
              height: frame.height,
            },
            [frame.gray.buffer]
          );
        } else {
          // Even before a selection is anchored, count rendered video/camera frames uncapped
          const t0 = performance.now();
          drawCurrentStageFrame();
          const dt = Math.max(0.4, performance.now() - t0);
          recordFrameMetrics(dt, "idle", 0);
        }
      } else {
        drawCurrentStageFrame();
      }
    }
    // Uncapped scheduling: requestAnimationFrame (60 Hz native) without any 66 ms throttle
    loopRafId = requestAnimationFrame(pumpOneFrame);
  }

  function startFramePump() {
    stopFramePump();
    loopRafId = requestAnimationFrame(pumpOneFrame);
  }

  function stopCameraTracks() {
    if (activeCameraStream) {
      const tracks = activeCameraStream.getTracks();
      for (const t of tracks) {
        try {
          t.stop();
        } catch (_) {}
      }
      activeCameraStream = null;
    }
    try {
      cameraVideo.pause();
      cameraVideo.srcObject = null;
    } catch (_) {}
  }

  async function switchToVideoMode() {
    modeEpoch += 1;
    stopFramePump();
    stopCameraTracks();
    telemetry.mode = "video";
    telemetry.playing = false;
    modeVideoBtn.classList.add("is-active");
    modeVideoBtn.setAttribute("aria-pressed", "true");
    modeCameraBtn.classList.remove("is-active");
    modeCameraBtn.setAttribute("aria-pressed", "false");
    clearTrackingSelection();
    updateHud();
    if (videoLoaded) {
      drawCurrentStageFrame();
      setStatus("Recorded Video mode ready.", false);
    } else {
      setStatus("Load a demo clip or upload a video file to begin.", false);
    }
  }

  async function switchToCameraMode() {
    const myEpoch = ++modeEpoch;
    stopFramePump();
    try {
      sourceVideo.pause();
    } catch (_) {}
    stopCameraTracks();

    telemetry.mode = "camera";
    telemetry.playing = false;
    modeCameraBtn.classList.add("is-active");
    modeCameraBtn.setAttribute("aria-pressed", "true");
    modeVideoBtn.classList.remove("is-active");
    modeVideoBtn.setAttribute("aria-pressed", "false");
    clearTrackingSelection();
    resetMetrics();
    updateHud();

    if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
      setStatus("Camera API (getUserMedia) is unavailable in this browser.", true);
      return;
    }

    setStatus("Requesting live camera stream...", false);
    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60 } },
        audio: false,
      });
    } catch (err) {
      if (myEpoch !== modeEpoch) return;
      telemetry.playing = false;
      updateHud();
      const name = (err && err.name) || "CameraError";
      setStatus(`Camera unavailable (${name}): ${(err && err.message) || "Permission denied or no device"}`, true);
      return;
    }

    // If user rapidly switched back to video mode while getUserMedia was pending, stop immediately!
    if (myEpoch !== modeEpoch || telemetry.mode !== "camera") {
      for (const t of stream.getTracks()) {
        try {
          t.stop();
        } catch (_) {}
      }
      return;
    }

    activeCameraStream = stream;
    for (const track of stream.getTracks()) {
      track.addEventListener("ended", () => {
        if (myEpoch !== modeEpoch) return;
        telemetry.playing = false;
        stopFramePump();
        updateHud();
        setStatus("Camera track ended.", false);
      });
    }

    cameraVideo.srcObject = stream;
    try {
      await cameraVideo.play();
    } catch (_) {
      if (myEpoch !== modeEpoch) return;
      telemetry.playing = false;
      updateHud();
      setStatus("Unable to start camera playback.", true);
      return;
    }

    if (myEpoch !== modeEpoch || telemetry.mode !== "camera") {
      stopCameraTracks();
      return;
    }

    telemetry.playing = true;
    updateHud();
    setStatus("Live Camera active. Click or drag on an object to attach artwork.", false);
    startFramePump();
  }

  function loadVideoUrl(videoId, mediaUrl, clipMeta) {
    const myEpoch = ++modeEpoch;
    stopFramePump();
    telemetry.playing = false;
    videoLoaded = false;
    posterFrame0Image = null;
    currentVideoId = videoId;
    telemetry.lastExportReport = null;
    exportDownloadLink.setAttribute("href", "#");
    exportDownloadLink.removeAttribute("download");
    exportDownloadLink.classList.add("is-disabled");
    exportDownloadLink.setAttribute("aria-disabled", "true");

    clearTrackingSelection();
    resetMetrics();
    setStatus(`Loading ${videoId}...`, false);

    // Fetch decoded frame 0 poster JPEG so paused frame-0 rendering and pre-play selection
    // work immediately even in headless Chromium (which may not paint paused <video> frames).
    const poster = new Image();
    poster.crossOrigin = "anonymous";
    poster.onload = () => {
      if (myEpoch !== modeEpoch || telemetry.mode !== "video") return;
      posterFrame0Image = poster;
      videoLoaded = true;
      stageCanvas.width = poster.naturalWidth || 640;
      stageCanvas.height = poster.naturalHeight || 480;
      drawCurrentStageFrame();
      setStatus(`Loaded ${videoId} (${stageCanvas.width}×${stageCanvas.height}). Select an object or press Play.`, false);
    };
    poster.src = `/api/videos/${encodeURIComponent(videoId)}/frame/0`;

    sourceVideo.pause();
    sourceVideo.src = mediaUrl;
    sourceVideo.load();

    const onLoadedData = () => {
      sourceVideo.removeEventListener("loadeddata", onLoadedData);
      sourceVideo.removeEventListener("error", onLoadErr);
      if (myEpoch !== modeEpoch || telemetry.mode !== "video") return;
      videoLoaded = true;
      if (sourceVideo.videoWidth > 0 && sourceVideo.videoHeight > 0) {
        stageCanvas.width = sourceVideo.videoWidth;
        stageCanvas.height = sourceVideo.videoHeight;
      }
      drawCurrentStageFrame();
      setStatus(`Loaded ${videoId} (${stageCanvas.width}×${stageCanvas.height}). Select an object or press Play.`, false);
    };
    const onLoadErr = () => {
      sourceVideo.removeEventListener("loadeddata", onLoadedData);
      sourceVideo.removeEventListener("error", onLoadErr);
      if (myEpoch !== modeEpoch) return;
      if (!posterFrame0Image) {
        videoLoaded = false;
        setStatus(`Failed to decode video ${videoId}.`, true);
      }
    };
    sourceVideo.addEventListener("loadeddata", onLoadedData);
    sourceVideo.addEventListener("error", onLoadErr);
  }

  async function fetchDemos() {
    try {
      const resp = await fetch("/api/demos");
      if (!resp.ok) return;
      const data = await resp.json();
      const clips = Array.isArray(data.clips) ? data.clips : [];
      if (!clips.length) return;
      demoMetaById = {};
      const prevVal = demoSelect.value;
      demoSelect.innerHTML = "";
      for (const c of clips) {
        demoMetaById[c.id] = c;
        const opt = document.createElement("option");
        opt.value = c.id;
        opt.textContent = c.id;
        demoSelect.appendChild(opt);
      }
      if (prevVal && demoMetaById[prevVal]) {
        demoSelect.value = prevVal;
      }
    } catch (_) {
      // Keep HTML fallback options intact
    }
  }

  function eventToCanvasPoint(ev) {
    const rect = stageCanvas.getBoundingClientRect();
    const cw = stageCanvas.width || rect.width || 640;
    const ch = stageCanvas.height || rect.height || 480;
    const x = clampNum(((ev.clientX - rect.left) / Math.max(1, rect.width)) * cw, 0, cw - 1, cw * 0.5);
    const y = clampNum(((ev.clientY - rect.top) / Math.max(1, rect.height)) * ch, 0, ch - 1, ch * 0.5);
    return { x, y };
  }

  // Event listeners
  modeVideoBtn.addEventListener("click", () => {
    switchToVideoMode();
  });

  modeCameraBtn.addEventListener("click", () => {
    switchToCameraMode();
  });

  loadDemoBtn.addEventListener("click", () => {
    if (telemetry.mode !== "video") {
      switchToVideoMode();
    }
    const clipId = demoSelect.value || "planar_card_640x480";
    const meta = demoMetaById[clipId];
    const mediaUrl = (meta && meta.media_url) || `/api/videos/${encodeURIComponent(clipId)}/media`;
    loadVideoUrl(clipId, mediaUrl, meta || null);
  });

  videoFileInput.addEventListener("change", async () => {
    const file = videoFileInput.files && videoFileInput.files[0];
    if (!file) return;
    if (telemetry.mode !== "video") {
      await switchToVideoMode();
    }
    setStatus(`Uploading ${file.name}...`, false);
    const form = new FormData();
    form.append("file", file, file.name);
    try {
      const resp = await fetch("/api/videos/upload", { method: "POST", body: form });
      if (!resp.ok) {
        let detail = `Upload failed (HTTP ${resp.status})`;
        try {
          const j = await resp.json();
          if (j && j.detail) detail = String(j.detail);
        } catch (_) {}
        setStatus(detail, true);
        return;
      }
      const data = await resp.json();
      loadVideoUrl(data.video_id, data.media_url, data);
    } catch (err) {
      setStatus(`Video upload error: ${(err && err.message) || "network error"}`, true);
    }
  });

  playPauseBtn.addEventListener("click", async () => {
    if (telemetry.mode === "camera") {
      if (!activeCameraStream) {
        await switchToCameraMode();
        return;
      }
      telemetry.playing = !telemetry.playing;
      if (telemetry.playing) {
        try {
          await cameraVideo.play();
        } catch (_) {}
        startFramePump();
      } else {
        stopFramePump();
      }
      updateHud();
      return;
    }

    if (!videoLoaded || !currentVideoId) {
      setStatus("No video loaded yet. Select a demo clip or upload a video first.", false);
      return;
    }

    if (telemetry.playing) {
      telemetry.playing = false;
      stopFramePump();
      sourceVideo.pause();
      updateHud();
    } else {
      try {
        await sourceVideo.play();
      } catch (_) {}
      telemetry.playing = true;
      updateHud();
      startFramePump();
    }
  });

  selectModePointBtn.addEventListener("click", () => {
    selectTool = "point";
    selectModePointBtn.classList.add("is-active");
    selectModePointBtn.setAttribute("aria-pressed", "true");
    selectModeBoxBtn.classList.remove("is-active");
    selectModeBoxBtn.setAttribute("aria-pressed", "false");
  });

  selectModeBoxBtn.addEventListener("click", () => {
    selectTool = "box";
    selectModeBoxBtn.classList.add("is-active");
    selectModeBoxBtn.setAttribute("aria-pressed", "true");
    selectModePointBtn.classList.remove("is-active");
    selectModePointBtn.setAttribute("aria-pressed", "false");
  });

  autoSelectBtn.addEventListener("click", () => {
    const src = getDrawableMediaSource();
    if (!src) {
      setStatus("Load a video or start Live Camera before using Auto Select.", false);
      return;
    }
    const w = src.w;
    const h = src.h;
    if (telemetry.mode === "video" && currentVideoId && demoMetaById[currentVideoId] && demoMetaById[currentVideoId].default_selection) {
      const ds = demoMetaById[currentVideoId].default_selection;
      submitSelectionToWorker({
        frame_index: Number(ds.frame_index || 0),
        positive_points: Array.isArray(ds.positive_points) ? ds.positive_points : [[w * 0.5, h * 0.5]],
        negative_points: Array.isArray(ds.negative_points) ? ds.negative_points : [],
        bbox: Array.isArray(ds.bbox) ? ds.bbox : [w * 0.25, h * 0.25, w * 0.5, h * 0.5],
      });
    } else {
      submitSelectionToWorker({
        frame_index: 0,
        positive_points: [[w * 0.5, h * 0.5]],
        negative_points: [],
        bbox: [w * 0.25, h * 0.25, w * 0.5, h * 0.5],
      });
    }
    setStatus("Auto-selected target surface.", false);
  });

  clearSelectionBtn.addEventListener("click", () => {
    clearTrackingSelection();
    setStatus("Selection cleared.", false);
  });

  stageCanvas.addEventListener("pointerdown", (ev) => {
    const pt = eventToCanvasPoint(ev);
    dragStart = pt;
    dragCurrent = pt;
  });

  stageCanvas.addEventListener("pointermove", (ev) => {
    if (!dragStart) return;
    dragCurrent = eventToCanvasPoint(ev);
    if (selectTool === "box") {
      drawCurrentStageFrame();
    }
  });

  stageCanvas.addEventListener("pointerup", (ev) => {
    const endPt = eventToCanvasPoint(ev);
    const startPt = dragStart || endPt;
    dragStart = null;
    dragCurrent = null;

    if (selectTool === "box") {
      const x0 = Math.min(startPt.x, endPt.x);
      const y0 = Math.min(startPt.y, endPt.y);
      const bw = Math.abs(endPt.x - startPt.x);
      const bh = Math.abs(endPt.y - startPt.y);
      if (bw >= 8 && bh >= 8) {
        submitSelectionToWorker({
          frame_index: 0,
          positive_points: [[x0 + bw * 0.5, y0 + bh * 0.5]],
          negative_points: [],
          bbox: [x0, y0, bw, bh],
        });
        setStatus(`Selected box (${Math.round(x0)}, ${Math.round(y0)}, ${Math.round(bw)}×${Math.round(bh)}).`, false);
        return;
      }
    }

    // Point selection (or tiny drag fallback)
    submitSelectionToWorker({
      frame_index: 0,
      positive_points: [[endPt.x, endPt.y]],
      negative_points: [],
      bbox: null,
    });
    setStatus(`Selected point (${Math.round(endPt.x)}, ${Math.round(endPt.y)}).`, false);
  });

  patternSelect.addEventListener("change", () => {
    currentDesignRecipe = {
      ...currentDesignRecipe,
      pattern: patternSelect.value || "stripes",
    };
    currentDesignId = null;
    customArtworkImage = null;
    refreshDesignPreview();
  });

  applyPromptBtn.addEventListener("click", () => {
    const parsed = parseDesignPrompt(designPromptInput.value);
    currentDesignRecipe = {
      pattern: parsed.pattern,
      colors: parsed.colors,
      text: parsed.text,
      prompt: parsed.prompt,
      width: 512,
      height: 512,
    };
    currentDesignId = null;
    customArtworkImage = null;
    patternSelect.value = parsed.pattern;
    if (parsed.scale !== null) {
      currentPlacement.scale = clampNum(parsed.scale, 0.1, 5.0, 1.0);
      scaleSlider.value = String(currentPlacement.scale);
      scaleVal.textContent = `${currentPlacement.scale.toFixed(2)}×`;
    }
    refreshDesignPreview();
    setStatus(`Applied design prompt (${parsed.pattern}).`, false);
  });

  designImageInput.addEventListener("change", async () => {
    const file = designImageInput.files && designImageInput.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file, file.name);
    try {
      const resp = await fetch("/api/designs/upload", { method: "POST", body: form });
      if (!resp.ok) {
        let detail = `Unsupported design image (HTTP ${resp.status})`;
        try {
          const j = await resp.json();
          if (j && j.detail) detail = String(j.detail);
        } catch (_) {}
        setStatus(detail, true);
        return;
      }
      const data = await resp.json();
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        customArtworkImage = img;
        currentDesignId = data.design_id;
        refreshDesignPreview();
        setStatus(`Loaded custom artwork (${data.width}×${data.height}).`, false);
      };
      img.onerror = () => {
        setStatus("Uploaded artwork image failed to decode in browser.", true);
      };
      img.src = data.image_url;
    } catch (err) {
      setStatus(`Artwork upload error: ${(err && err.message) || "network error"}`, true);
    }
  });

  wrapModeSelect.addEventListener("change", () => {
    currentPlacement.wrap_mode = wrapModeSelect.value || "planar";
    telemetry.wrapMode = currentPlacement.wrap_mode;
    drawCurrentStageFrame();
  });

  blendModeSelect.addEventListener("change", () => {
    currentPlacement.blend_mode = blendModeSelect.value || "luminance";
    telemetry.blendMode = currentPlacement.blend_mode;
    drawCurrentStageFrame();
  });

  function syncSliders() {
    currentPlacement.opacity = clampNum(opacitySlider.value, 0.0, 1.0, 0.9);
    currentPlacement.scale = clampNum(scaleSlider.value, 0.1, 5.0, 1.0);
    currentPlacement.rotation_deg = clampNum(rotationSlider.value, -180.0, 180.0, 0.0);
    opacityVal.textContent = currentPlacement.opacity.toFixed(2);
    scaleVal.textContent = `${currentPlacement.scale.toFixed(2)}×`;
    rotationVal.textContent = `${Math.round(currentPlacement.rotation_deg)}°`;
    drawCurrentStageFrame();
  }

  for (const sl of [opacitySlider, scaleSlider, rotationSlider]) {
    sl.addEventListener("input", syncSliders);
    sl.addEventListener("change", syncSliders);
  }

  exportMp4Btn.addEventListener("click", async () => {
    if (!currentVideoId || telemetry.mode !== "video") {
      setStatus("Load a recorded video before exporting MP4.", false);
      return;
    }
    exportMp4Btn.disabled = true;
    telemetry.lastExportReport = null;
    setStatus("Rendering full-frame H.264 MP4 on server...", false);

    const payload = {
      placement: { ...currentPlacement },
    };
    if (currentSelection) {
      payload.selection = currentSelection;
    } else if (demoMetaById[currentVideoId] && demoMetaById[currentVideoId].default_selection) {
      payload.selection = demoMetaById[currentVideoId].default_selection;
    }
    if (currentDesignId) {
      payload.design_id = currentDesignId;
    } else {
      payload.design_recipe = { ...currentDesignRecipe };
    }

    try {
      const resp = await fetch(`/api/videos/${encodeURIComponent(currentVideoId)}/render`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        let detail = `Export failed (HTTP ${resp.status})`;
        try {
          const j = await resp.json();
          if (j && j.detail) detail = String(j.detail);
        } catch (_) {}
        setStatus(detail, true);
        return;
      }
      const data = await resp.json();
      telemetry.lastExportReport = data.report;
      exportDownloadLink.setAttribute("href", data.video_url);
      exportDownloadLink.setAttribute("download", `${currentVideoId}_visualizeit.mp4`);
      exportDownloadLink.classList.remove("is-disabled");
      exportDownloadLink.setAttribute("aria-disabled", "false");
      const fpsStr = data.report && Number(data.report.effective_fps).toFixed(1);
      setStatus(`Export complete (${fpsStr} FPS, ${data.report.output_frame_count} frames). Ready to download.`, false);
    } catch (err) {
      setStatus(`Export error: ${(err && err.message) || "network error"}`, true);
    } finally {
      exportMp4Btn.disabled = false;
    }
  });

  // Bootstrap
  initWorker();
  refreshDesignPreview();
  syncSliders();
  updateHud();
  fetchDemos();
})();
