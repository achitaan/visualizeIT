"""Zero-dependency headless `/usr/bin/chromium` (156.x) CDP harness for Browser Studio & Camera E2E.

Enforces all host/Chromium 156 constraints discovered empirically:
1. Short XFS `TMPDIR` (`<= 61` chars in `.cache/tmp/cXXXXXX`) so `SingletonSocket` stays `< 108` bytes
   and `/tmp` remains 100% untouched.
2. Strict `--flag=value` CLI flag syntax for Debian's `/usr/bin/chromium` wrapper.
3. `--no-sandbox --no-zygote` + graceful CDP `Browser.close` + `os.killpg` fallback for zero zombies.
4. `Page.addScriptToEvaluateOnNewDocument`, DOM testid actions, file uploads, MP4 downloads,
   `requestAnimationFrame`-synchronized Canvas2D/WebGL2 sampling, and `window.__VISUALIZEIT_TELEMETRY__`.
"""

from __future__ import annotations

import contextlib
import ctypes
import http.client
import json
import os
import re
import select
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from .http_and_ws import StdlibWebSocket, WebSocketClosedError
from .paths_and_proc import CACHE_TMP_ROOT, assert_xfs_temp_path, terminate_process_group

CHROMIUM_BIN = Path("/usr/bin/chromium")


def _to_selector(selector_or_testid: str) -> str:
    s = selector_or_testid.strip()
    if s.startswith(("[", "#", ".")) or re.match(r"^(input|button|select|canvas|a|div)([\s\[#\.:]|$)", s):
        return s
    return f'[data-testid={json.dumps(s)}]'


class HeadlessChromiumSession:
    """Isolated, zero-`/tmp`-leak, zero-zombie headless Chromium 156 session controlled via CDP."""

    def __init__(
        self,
        *,
        fake_video_y4m: Path | None = None,
        download_dir: Path | None = None,
        extra_args: list[str] | None = None,
        enable_fake_media_stream: bool = True,
        startup_timeout: float = 15.0,
    ) -> None:
        CACHE_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        assert_xfs_temp_path(CACHE_TMP_ROOT)
        # Keep <= 61 chars to satisfy Linux sockaddr_un 108-byte SingletonSocket limit!
        self._short_dir = Path(tempfile.mkdtemp(prefix="c", dir=str(CACHE_TMP_ROOT)))
        assert len(str(self._short_dir)) <= 61, (
            f"TMPDIR too long ({len(str(self._short_dir))} > 61) for Chromium SingletonSocket: {self._short_dir}"
        )
        if download_dir is not None:
            self.download_dir = assert_xfs_temp_path(download_dir)
        else:
            self.download_dir = self._short_dir / "dl"
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.fake_video_y4m = fake_video_y4m
        self.extra_args = list(extra_args or [])
        self.enable_fake_media_stream = enable_fake_media_stream
        self.startup_timeout = startup_timeout

        self.proc: subprocess.Popen[bytes] | None = None
        self.ws: StdlibWebSocket | None = None
        self.cdp_port: int | None = None
        self._next_id = 0
        self._closed = False
        self.events: list[dict[str, Any]] = []
        self.console_errors: list[str] = []
        self.page_exceptions: list[str] = []
        self._completed_downloads: set[str] = set()

    def __enter__(self) -> "HeadlessChromiumSession":
        if self.proc is not None and self.ws is not None:
            return self
        assert CHROMIUM_BIN.is_file(), f"Missing headless browser binary: {CHROMIUM_BIN}"

        xdg_cfg = self._short_dir / "xc"
        xdg_cache = self._short_dir / "xh"
        xdg_run = self._short_dir / "xr"
        prof = self._short_dir / "p"
        dcache = self._short_dir / "d"
        for d in (xdg_cfg, xdg_cache, xdg_run, prof, dcache):
            d.mkdir(parents=True, exist_ok=True)
        xdg_run.chmod(0o700)

        env = os.environ.copy()
        env.update(
            {
                "TMPDIR": str(self._short_dir),
                "TEMP": str(self._short_dir),
                "TMP": str(self._short_dir),
                "XDG_CONFIG_HOME": str(xdg_cfg),
                "XDG_CACHE_HOME": str(xdg_cache),
                "XDG_RUNTIME_DIR": str(xdg_run),
            }
        )

        # Every flag MUST use --flag=value syntax for Debian /usr/bin/chromium wrapper compatibility!
        cmd: list[str] = [
            str(CHROMIUM_BIN),
            "--headless",
            "--no-sandbox",
            "--no-zygote",
            "--disable-gpu",
            "--enable-unsafe-swiftshader",
            "--disable-dev-shm-usage",
            "--autoplay-policy=no-user-gesture-required",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--remote-debugging-port=0",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-breakpad",
            "--disable-crash-reporter",
            f"--user-data-dir={prof}",
            f"--disk-cache-dir={dcache}",
        ]
        if self.enable_fake_media_stream:
            cmd.append("--use-fake-ui-for-media-stream")
            cmd.append("--use-fake-device-for-media-stream")
            if self.fake_video_y4m is not None:
                y4m_abs = assert_xfs_temp_path(self.fake_video_y4m)
                assert y4m_abs.is_file(), f"Fake camera .y4m file missing: {y4m_abs}"
                cmd.append(f"--use-file-for-fake-video-capture={y4m_abs}")

        for flag in self.extra_args:
            assert flag.startswith("--"), (
                f"Chromium extra_args must use '--flag' or '--flag=value' syntax, got {flag!r}"
            )
            cmd.append(flag)
        cmd.append("about:blank")

        with contextlib.suppress(Exception):
            # PR_SET_CHILD_SUBREAPER = 36 on Linux: adopt double-forked crashpad grandchildren
            # instead of leaking them as <defunct> onto container PID 1.
            ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        assert self.proc.stderr is not None
        os.set_blocking(self.proc.stderr.fileno(), False)

        buf = b""
        browser_ws: str | None = None
        deadline = time.monotonic() + self.startup_timeout
        try:
            while time.monotonic() < deadline and browser_ws is None:
                if self.proc.poll() is not None:
                    buf += self.proc.stderr.read() or b""
                    raise RuntimeError(
                        f"Chromium exited early (rc={self.proc.returncode}):\n"
                        f"{buf.decode('utf-8', 'replace')}"
                    )
                if select.select([self.proc.stderr], [], [], 0.08)[0]:
                    buf += self.proc.stderr.read(4096) or b""
                    m = re.search(
                        rb"DevTools listening on (ws://127\.0\.0\.1:(\d+)/devtools/browser/\S+)",
                        buf,
                    )
                    if m:
                        browser_ws = m.group(1).decode("ascii")
                        self.cdp_port = int(m.group(2))
            if not browser_ws or self.cdp_port is None:
                raise TimeoutError(
                    f"Timed out after {self.startup_timeout}s waiting for Chromium DevTools banner.\n"
                    f"Stderr so far:\n{buf.decode('utf-8', 'replace')}"
                )

            targets: list[dict[str, Any]] = []
            page_ws: str | None = None
            target_deadline = time.monotonic() + 5.0
            while time.monotonic() < target_deadline and page_ws is None:
                conn = http.client.HTTPConnection("127.0.0.1", self.cdp_port, timeout=3.0)
                try:
                    conn.request("GET", "/json/list")
                    targets = json.loads(conn.getresponse().read().decode("utf-8"))
                finally:
                    conn.close()
                for t in targets:
                    if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                        page_ws = str(t["webSocketDebuggerUrl"])
                        break
                if page_ws is None:
                    time.sleep(0.05)

            if page_ws is None:
                conn = http.client.HTTPConnection("127.0.0.1", self.cdp_port, timeout=3.0)
                try:
                    conn.request("PUT", "/json/new?about:blank")
                    created = json.loads(conn.getresponse().read().decode("utf-8"))
                    page_ws = str(created["webSocketDebuggerUrl"])
                finally:
                    conn.close()

            self.ws = StdlibWebSocket(page_ws, timeout=10.0)
            for domain in ("Page.enable", "Runtime.enable", "DOM.enable", "Log.enable"):
                self.call(domain)
            self.call(
                "Browser.setDownloadBehavior",
                {
                    "behavior": "allow",
                    "downloadPath": str(self.download_dir),
                    "eventsEnabled": True,
                },
            )
            self.call("Page.bringToFront")
            return self
        except Exception:
            self._teardown()
            raise

    def _record_event(self, data: dict[str, Any]) -> None:
        self.events.append(data)
        method = data.get("method")
        params = data.get("params") or {}
        if method == "Runtime.consoleAPICalled" and params.get("type") == "error":
            rendered_args = [
                str(a.get("description") or a.get("value") or json.dumps(a))
                for a in params.get("args", [])
            ]
            self.console_errors.append(" ".join(rendered_args) or json.dumps(params))
        elif method == "Runtime.exceptionThrown":
            details = params.get("exceptionDetails") or {}
            exc_obj = details.get("exception") or {}
            desc = exc_obj.get("description") or details.get("text") or json.dumps(details)
            self.page_exceptions.append(str(desc))
        elif method == "Browser.downloadProgress" and params.get("state") == "completed":
            guid = params.get("guid")
            if guid:
                self._completed_downloads.add(str(guid))

    def pump_events(self, duration: float = 0.05) -> None:
        """Drain pending asynchronous CDP events for up to `duration` seconds."""
        if self.ws is None or self.proc is None or self.proc.poll() is not None:
            return
        end_t = time.monotonic() + max(0.01, duration)
        while True:
            rem = end_t - time.monotonic()
            if rem <= 0:
                break
            self.ws.settimeout(rem)
            try:
                op, raw = self.ws.recv_message()
            except (TimeoutError, WebSocketClosedError, OSError):
                break
            if op == 0x1:
                msg = json.loads(raw.decode("utf-8"))
                if "id" not in msg:
                    self._record_event(msg)

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Send a synchronous CDP command and return its `result` dictionary."""
        assert self.ws is not None and self.proc is not None, "HeadlessChromiumSession is not open"
        if self.proc.poll() is not None:
            raise RuntimeError(f"Chromium crashed with exit code {self.proc.returncode}")
        self._next_id += 1
        req_id = self._next_id
        self.ws.settimeout(timeout)
        self.ws.send_json({"id": req_id, "method": method, "params": params or {}})
        deadline = time.monotonic() + timeout
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError(f"CDP {method} timed out after {timeout}s")
            self.ws.settimeout(rem)
            op, raw = self.ws.recv_message()
            if op != 0x1:
                continue
            msg = json.loads(raw.decode("utf-8"))
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise RuntimeError(f"CDP {method} error: {msg['error']}")
                return msg.get("result") or {}
            self._record_event(msg)

    def wait_for_event(
        self,
        method: str,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Wait for a CDP event matching `method` and optional `predicate(params)`."""
        assert self.ws is not None, "HeadlessChromiumSession is not open"
        for idx, ev in enumerate(self.events):
            if ev.get("method") == method and (predicate is None or predicate(ev.get("params") or {})):
                return self.events.pop(idx)

        deadline = time.monotonic() + timeout
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError(f"Timed out after {timeout}s waiting for CDP event {method}")
            self.ws.settimeout(rem)
            op, raw = self.ws.recv_message()
            if op != 0x1:
                continue
            msg = json.loads(raw.decode("utf-8"))
            self._record_event(msg)
            if msg.get("method") == method and (predicate is None or predicate(msg.get("params") or {})):
                self.events.pop()
                return msg

    def add_init_script(self, source: str, *, timeout: float = 10.0) -> str:
        """Register JavaScript via `Page.addScriptToEvaluateOnNewDocument` prior to `navigate()`."""
        res = self.call("Page.addScriptToEvaluateOnNewDocument", {"source": source}, timeout=timeout)
        return str(res.get("identifier", ""))

    def navigate(self, url: str, *, timeout: float = 15.0) -> None:
        """Navigate to `url`, wait for `Page.loadEventFired`, and bring the tab to the foreground."""
        assert "//0.0.0.0" not in url, (
            f"Navigate to http://127.0.0.1:<port> (not 0.0.0.0) so window.isSecureContext is true: {url}"
        )
        self.events = [ev for ev in self.events if ev.get("method") != "Page.loadEventFired"]
        nav = self.call("Page.navigate", {"url": url}, timeout=timeout)
        if nav.get("errorText"):
            raise RuntimeError(f"Navigation to {url} failed: {nav['errorText']}")
        self.wait_for_event("Page.loadEventFired", timeout=timeout)
        self.call("Page.bringToFront")

    def evaluate(
        self,
        expression: str,
        *,
        await_promise: bool = True,
        timeout: float = 10.0,
    ) -> Any:
        """Evaluate JavaScript in the page context and return the JSON-serialized result value."""
        res = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": await_promise,
                "returnByValue": True,
            },
            timeout=timeout,
        )
        if "exceptionDetails" in res:
            raise RuntimeError(f"JS evaluation threw exception: {res['exceptionDetails']}")
        return (res.get("result") or {}).get("value")

    def wait_for_testid(self, testid: str, *, timeout: float = 10.0) -> dict[str, Any]:
        """Poll until `[data-testid="<testid>"]` exists in the DOM."""
        sel_json = json.dumps(_to_selector(testid))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = self.evaluate(
                f"""(() => {{
                    const el = document.querySelector({sel_json});
                    if (!el) return null;
                    const r = el.getBoundingClientRect();
                    return {{
                        tag: el.tagName.toLowerCase(),
                        disabled: Boolean(el.disabled),
                        width: r.width,
                        height: r.height
                    }};
                }})()""",
                await_promise=False,
                timeout=min(5.0, max(1.0, deadline - time.monotonic())),
            )
            if info is not None:
                return dict(info)
            time.sleep(0.05)
        raise LookupError(f'Element [data-testid="{testid}"] not found within {timeout}s')

    def click_testid(self, testid: str, *, timeout: float = 10.0) -> None:
        """Wait for `[data-testid="<testid>"]` to exist and not be disabled, then click it."""
        sel_json = json.dumps(_to_selector(testid))
        deadline = time.monotonic() + timeout
        last_state: Any = None
        while time.monotonic() < deadline:
            last_state = self.evaluate(
                f"""(() => {{
                    const el = document.querySelector({sel_json});
                    if (!el) return "missing";
                    if (el.disabled) return "disabled";
                    el.scrollIntoView({{ block: "center", inline: "center" }});
                    el.click();
                    return "clicked";
                }})()""",
                await_promise=False,
                timeout=min(5.0, max(1.0, deadline - time.monotonic())),
            )
            if last_state == "clicked":
                return
            time.sleep(0.05)
        if last_state == "missing":
            raise LookupError(f'Element [data-testid="{testid}"] not found within {timeout}s')
        raise TimeoutError(f'Element [data-testid="{testid}"] remained {last_state!r} after {timeout}s')

    def click_canvas_point(
        self,
        x: float,
        y: float,
        *,
        testid: str = "stage-canvas",
        timeout: float = 10.0,
    ) -> dict[str, float]:
        """Click internal canvas pixel coordinate `(x, y)` on `[data-testid="stage-canvas"]`."""
        self.wait_for_testid(testid, timeout=timeout)
        sel_json = json.dumps(_to_selector(testid))
        res = self.evaluate(
            f"""(() => {{
                const canvas = document.querySelector({sel_json});
                if (!canvas) throw new Error("Canvas not found");
                canvas.scrollIntoView({{ block: "center", inline: "center" }});
                const rect = canvas.getBoundingClientRect();
                const cw = canvas.width || rect.width || 640;
                const ch = canvas.height || rect.height || 480;
                const clientX = rect.left + ({float(x)} / cw) * (rect.width || cw);
                const clientY = rect.top + ({float(y)} / ch) * (rect.height || ch);
                const opts = {{
                    bubbles: true,
                    cancelable: true,
                    clientX,
                    clientY,
                    button: 0,
                    buttons: 1,
                    pointerId: 1,
                    isPrimary: true
                }};
                canvas.dispatchEvent(new PointerEvent("pointerdown", opts));
                canvas.dispatchEvent(new MouseEvent("mousedown", opts));
                canvas.dispatchEvent(new PointerEvent("pointerup", {{ ...opts, buttons: 0 }}));
                canvas.dispatchEvent(new MouseEvent("mouseup", {{ ...opts, buttons: 0 }}));
                canvas.dispatchEvent(new MouseEvent("click", {{ ...opts, buttons: 0 }}));
                return {{ clientX, clientY, canvasW: cw, canvasH: ch }};
            }})()""",
            await_promise=False,
            timeout=timeout,
        )
        return dict(res)

    def drag_canvas_box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        testid: str = "stage-canvas",
        steps: int = 5,
        timeout: float = 10.0,
    ) -> dict[str, float]:
        """Drag a bounding box `(x, y, w, h)` in canvas pixel coordinates on `[data-testid="stage-canvas"]`."""
        self.wait_for_testid(testid, timeout=timeout)
        sel_json = json.dumps(_to_selector(testid))
        n_steps = max(2, int(steps))
        res = self.evaluate(
            f"""(() => {{
                const canvas = document.querySelector({sel_json});
                if (!canvas) throw new Error("Canvas not found");
                canvas.scrollIntoView({{ block: "center", inline: "center" }});
                const rect = canvas.getBoundingClientRect();
                const cw = canvas.width || rect.width || 640;
                const ch = canvas.height || rect.height || 480;
                const toClient = (px, py) => ([
                    rect.left + (px / cw) * (rect.width || cw),
                    rect.top + (py / ch) * (rect.height || ch)
                ]);
                const [sx, sy] = toClient({float(x)}, {float(y)});
                const [ex, ey] = toClient({float(x + w)}, {float(y + h)});
                const base = {{ bubbles: true, cancelable: true, button: 0, pointerId: 1, isPrimary: true }};
                canvas.dispatchEvent(new PointerEvent("pointerdown", {{ ...base, clientX: sx, clientY: sy, buttons: 1 }}));
                canvas.dispatchEvent(new MouseEvent("mousedown", {{ ...base, clientX: sx, clientY: sy, buttons: 1 }}));
                for (let i = 1; i <= {n_steps}; i++) {{
                    const t = i / {n_steps};
                    const mx = sx + (ex - sx) * t;
                    const my = sy + (ey - sy) * t;
                    canvas.dispatchEvent(new PointerEvent("pointermove", {{ ...base, clientX: mx, clientY: my, buttons: 1 }}));
                    canvas.dispatchEvent(new MouseEvent("mousemove", {{ ...base, clientX: mx, clientY: my, buttons: 1 }}));
                }}
                canvas.dispatchEvent(new PointerEvent("pointerup", {{ ...base, clientX: ex, clientY: ey, buttons: 0 }}));
                canvas.dispatchEvent(new MouseEvent("mouseup", {{ ...base, clientX: ex, clientY: ey, buttons: 0 }}));
                return {{ startX: sx, startY: sy, endX: ex, endY: ey, canvasW: cw, canvasH: ch }};
            }})()""",
            await_promise=False,
            timeout=timeout,
        )
        return dict(res)

    def fill_testid(self, testid: str, value: str, *, timeout: float = 10.0) -> str:
        """Fill a text `<input>` or `<textarea>` identified by `data-testid` and fire `input`/`change`."""
        self.wait_for_testid(testid, timeout=timeout)
        sel_json = json.dumps(_to_selector(testid))
        val_json = json.dumps(str(value))
        out = self.evaluate(
            f"""(() => {{
                const el = document.querySelector({sel_json});
                if (!el) throw new Error("Missing element");
                el.focus();
                const proto = el instanceof HTMLTextAreaElement
                    ? HTMLTextAreaElement.prototype
                    : HTMLInputElement.prototype;
                const desc = Object.getOwnPropertyDescriptor(proto, "value");
                if (desc && desc.set) desc.set.call(el, {val_json});
                else el.value = {val_json};
                el.dispatchEvent(new Event("input", {{ bubbles: true }}));
                el.dispatchEvent(new Event("change", {{ bubbles: true }}));
                return el.value;
            }})()""",
            await_promise=False,
            timeout=timeout,
        )
        return str(out)

    def select_testid(self, testid: str, value: str, *, timeout: float = 10.0) -> str:
        """Select `<option value="...">` on a `<select data-testid="...">` and fire `input`/`change`."""
        self.wait_for_testid(testid, timeout=timeout)
        sel_json = json.dumps(_to_selector(testid))
        val_json = json.dumps(str(value))
        out = self.evaluate(
            f"""(() => {{
                const el = document.querySelector({sel_json});
                if (!el) throw new Error("Missing select element");
                const target = {val_json};
                if (el.tagName.toLowerCase() === "select") {{
                    const opts = Array.from(el.options || []);
                    const match = opts.find(o => o.value === target || o.textContent.trim() === target);
                    if (match) el.value = match.value;
                    else el.value = target;
                }} else {{
                    el.value = target;
                }}
                el.dispatchEvent(new Event("input", {{ bubbles: true }}));
                el.dispatchEvent(new Event("change", {{ bubbles: true }}));
                return String(el.value);
            }})()""",
            await_promise=False,
            timeout=timeout,
        )
        return str(out)

    def set_slider_testid(
        self,
        testid: str,
        value: float | int | str,
        *,
        timeout: float = 10.0,
    ) -> float:
        """Set `<input type="range" data-testid="...">` value and dispatch `input`/`change`."""
        self.wait_for_testid(testid, timeout=timeout)
        sel_json = json.dumps(_to_selector(testid))
        val_json = json.dumps(str(value))
        out = self.evaluate(
            f"""(() => {{
                const el = document.querySelector({sel_json});
                if (!el) throw new Error("Missing slider element");
                const desc = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
                if (desc && desc.set) desc.set.call(el, {val_json});
                else el.value = {val_json};
                el.dispatchEvent(new Event("input", {{ bubbles: true }}));
                el.dispatchEvent(new Event("change", {{ bubbles: true }}));
                return Number(el.value);
            }})()""",
            await_promise=False,
            timeout=timeout,
        )
        return float(out)

    def set_file_input(
        self,
        selector_or_testid: str,
        file_paths: list[Path] | Path,
        *,
        timeout: float = 10.0,
    ) -> None:
        """Attach one or more local files to `<input type="file">` via `DOM.setFileInputFiles`."""
        paths = [file_paths] if isinstance(file_paths, Path) else list(file_paths)
        resolved = [str(p.resolve()) for p in paths]
        selector = _to_selector(selector_or_testid)
        root_id = int(self.call("DOM.getDocument", {"depth": 1}, timeout=timeout)["root"]["nodeId"])
        q = self.call("DOM.querySelector", {"nodeId": root_id, "selector": selector}, timeout=timeout)
        node_id = int(q.get("nodeId") or 0)
        if not node_id:
            raise LookupError(f"File input selector not found: {selector}")
        self.call(
            "DOM.setFileInputFiles",
            {"nodeId": node_id, "files": resolved},
            timeout=timeout,
        )

    def get_testid_text(self, testid: str, *, timeout: float = 5.0) -> str:
        """Return trimmed `textContent` of `[data-testid="<testid>"]`."""
        self.wait_for_testid(testid, timeout=timeout)
        sel_json = json.dumps(_to_selector(testid))
        val = self.evaluate(
            f"""(() => {{
                const el = document.querySelector({sel_json});
                return el ? (el.textContent || "").trim() : "";
            }})()""",
            await_promise=False,
            timeout=timeout,
        )
        return str(val or "")

    def get_telemetry(self, *, timeout: float = 5.0) -> dict[str, Any] | None:
        """Read `window.__VISUALIZEIT_TELEMETRY__` (`PROJECT.md §6`)."""
        val = self.evaluate(
            """(() => {
                const raw = window.__VISUALIZEIT_TELEMETRY__;
                if (!raw) return null;
                const t = typeof raw === "function" ? raw() : raw;
                if (!t || typeof t !== "object") return null;
                const keys = new Set([
                    "mode", "playing", "currentFps", "medianFrameMs", "p95FrameMs",
                    "processedFrames", "trackingState", "inliers", "wrapMode",
                    "blendMode", "lastExportReport", ...Object.keys(t)
                ]);
                const out = {};
                for (const k of keys) {
                    if (k in t && typeof t[k] !== "function") {
                        out[k] = t[k];
                    }
                }
                return JSON.parse(JSON.stringify(out));
            })()""",
            await_promise=False,
            timeout=timeout,
        )
        return dict(val) if isinstance(val, dict) else None

    def wait_for_telemetry(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        timeout: float = 15.0,
        poll_interval: float = 0.1,
    ) -> dict[str, Any]:
        """Poll `window.__VISUALIZEIT_TELEMETRY__` until `predicate(telemetry)` returns `True`."""
        deadline = time.monotonic() + timeout
        last_tel: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            self.pump_events(0.02)
            last_tel = self.get_telemetry(timeout=min(4.0, max(1.0, deadline - time.monotonic())))
            if last_tel is not None and predicate(last_tel):
                return last_tel
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Timed out after {timeout}s waiting for telemetry predicate.\n"
            f"Last telemetry: {last_tel}\n"
            f"Console errors: {self.console_errors}\n"
            f"Page exceptions: {self.page_exceptions}"
        )

    def wait_for_download(
        self,
        *,
        timeout: float = 30.0,
        expected_suffix: str = ".mp4",
    ) -> Path:
        """Wait for a browser download in `self.download_dir` to finish and return its `Path`."""
        deadline = time.monotonic() + timeout
        suffix_low = expected_suffix.lower()
        while time.monotonic() < deadline:
            self.pump_events(0.12)
            candidates = [
                p
                for p in self.download_dir.iterdir()
                if p.is_file()
                and not p.name.endswith((".crdownload", ".tmp", ".part"))
                and p.name.lower().endswith(suffix_low)
                and p.stat().st_size > 0
            ]
            if candidates and (
                len(self._completed_downloads) > 0
                or not any(p.name.endswith(".crdownload") for p in self.download_dir.iterdir())
            ):
                newest = max(candidates, key=lambda p: p.stat().st_mtime)
                # Confirm file size is stable across a brief check
                sz1 = newest.stat().st_size
                time.sleep(0.05)
                self.pump_events(0.03)
                if newest.exists() and newest.stat().st_size == sz1 and sz1 > 0:
                    return newest
            time.sleep(0.08)
        present = [p.name for p in self.download_dir.iterdir()] if self.download_dir.exists() else []
        raise TimeoutError(
            f"Timed out after {timeout}s waiting for *{expected_suffix} download in {self.download_dir}. "
            f"Present files: {present}"
        )

    def sample_stage_canvas(
        self,
        *,
        testid: str = "stage-canvas",
        max_dim: int = 160,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Sample `[data-testid="stage-canvas"]` inside `requestAnimationFrame` (Canvas2D & WebGL2 safe)."""
        self.wait_for_testid(testid, timeout=timeout)
        sel_json = json.dumps(_to_selector(testid))
        sample = self.evaluate(
            f"""(() => new Promise((resolve, reject) => {{
                const capture = () => {{
                    try {{
                        const src = document.querySelector({sel_json});
                        if (!src) return reject(new Error("Canvas not found"));
                        const w = src.width || 0;
                        const h = src.height || 0;
                        if (w <= 0 || h <= 0) {{
                            return resolve({{
                                width: w, height: h, sample_w: 0, sample_h: 0,
                                mean_rgb: [0, 0, 0], std_rgb: [0, 0, 0],
                                non_black_ratio: 0.0, png_base64: ""
                            }});
                        }}
                        const scale = Math.min(1.0, {int(max_dim)} / Math.max(w, h));
                        const sw = Math.max(1, Math.round(w * scale));
                        const sh = Math.max(1, Math.round(h * scale));
                        const off = document.createElement("canvas");
                        off.width = sw;
                        off.height = sh;
                        const ctx = off.getContext("2d", {{ willReadFrequently: true }});
                        ctx.drawImage(src, 0, 0, sw, sh);
                        let img = ctx.getImageData(0, 0, sw, sh);
                        let data = img.data;
                        let nonZero = 0;
                        for (let i = 0; i < data.length; i += 4) {{
                            if (data[i] > 0 || data[i + 1] > 0 || data[i + 2] > 0) nonZero++;
                        }}
                        if (nonZero === 0) {{
                            const gl = src.getContext("webgl2") || src.getContext("webgl");
                            if (gl) {{
                                const raw = new Uint8Array(w * h * 4);
                                gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, raw);
                                const tmp = document.createElement("canvas");
                                tmp.width = w;
                                tmp.height = h;
                                const tctx = tmp.getContext("2d");
                                const timg = tctx.createImageData(w, h);
                                for (let row = 0; row < h; row++) {{
                                    const srcOff = (h - 1 - row) * w * 4;
                                    const dstOff = row * w * 4;
                                    timg.data.set(raw.subarray(srcOff, srcOff + w * 4), dstOff);
                                }}
                                tctx.putImageData(timg, 0, 0);
                                ctx.clearRect(0, 0, sw, sh);
                                ctx.drawImage(tmp, 0, 0, sw, sh);
                                img = ctx.getImageData(0, 0, sw, sh);
                                data = img.data;
                            }}
                        }}
                        const total = sw * sh;
                        let sr = 0, sg = 0, sb = 0, nb = 0;
                        for (let i = 0; i < data.length; i += 4) {{
                            const r = data[i], g = data[i + 1], b = data[i + 2];
                            sr += r; sg += g; sb += b;
                            if (r > 4 || g > 4 || b > 4) nb++;
                        }}
                        const mr = sr / total, mg = sg / total, mb = sb / total;
                        let vr = 0, vg = 0, vb = 0;
                        for (let i = 0; i < data.length; i += 4) {{
                            vr += (data[i] - mr) ** 2;
                            vg += (data[i + 1] - mg) ** 2;
                            vb += (data[i + 2] - mb) ** 2;
                        }}
                        const pngB64 = off.toDataURL("image/png").split(",")[1] || "";
                        resolve({{
                            width: w,
                            height: h,
                            sample_w: sw,
                            sample_h: sh,
                            mean_rgb: [mr, mg, mb],
                            std_rgb: [Math.sqrt(vr / total), Math.sqrt(vg / total), Math.sqrt(vb / total)],
                            non_black_ratio: nb / total,
                            png_base64: pngB64
                        }});
                    }} catch (err) {{
                        reject(err);
                    }}
                }};
                requestAnimationFrame(capture);
            }}))()""",
            await_promise=True,
            timeout=timeout,
        )
        return dict(sample)

    def assert_zero_errors(
        self,
        *,
        allow_console_substrings: list[str] | None = None,
    ) -> None:
        """Pump pending CDP events and assert zero uncaught JS exceptions or console errors."""
        self.pump_events(0.05)
        allowed = list(allow_console_substrings or [])
        filtered_console = [
            err
            for err in self.console_errors
            if "favicon.ico" not in err and not any(sub in err for sub in allowed)
        ]
        assert not self.page_exceptions, (
            f"Uncaught browser page exception(s) detected: {self.page_exceptions}"
        )
        assert not filtered_console, (
            f"Browser console.error message(s) detected: {filtered_console}"
        )

    def _teardown(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.ws is not None:
            with contextlib.suppress(Exception, WebSocketClosedError):
                self.call("Browser.close", timeout=2.0)
            self.ws.close()
            self.ws = None
        if self.proc is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=3.0)
            terminate_process_group(self.proc, sigterm_timeout=2.0)
            self.proc = None
        # Reap any adopted double-forked grandchildren (e.g. chrome_crashpad_handler)
        for _ in range(8):
            reaped_any = False
            while True:
                try:
                    rpid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    rpid = 0
                if rpid <= 0:
                    break
                reaped_any = True
            if not reaped_any:
                time.sleep(0.02)
            else:
                break
        shutil.rmtree(self._short_dir, ignore_errors=True)

    def close(self) -> None:
        self._teardown()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._teardown()
