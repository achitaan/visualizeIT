"""Pure-stdlib HTTP/1.1 Keep-Alive, Range, Multipart, and RFC 6455 WebSocket E2E client.

Supports `PROJECT.md §5` REST endpoints and dual-mode `/ws/videos/{id}/stream` (both `jpeg_base64`
JSON frames and paired metadata + binary JPEG frames), plus structured handshake/close diagnostics
for Tier 2 negative WebSocket boundary tests.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.client
import json
import os
import socket
import struct
import subprocess
import time
import urllib.parse
import uuid
from typing import Any

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WebSocketHandshakeError(ConnectionError):
    """Raised when an HTTP -> WebSocket upgrade request returns a non-101 status."""

    def __init__(
        self,
        status_code: int,
        status_line: str,
        headers: dict[str, str],
        body: bytes = b"",
    ) -> None:
        super().__init__(f"WebSocket upgrade failed ({status_code}): {status_line}")
        self.status_code = status_code
        self.status_line = status_line
        self.headers = headers
        self.body = body


class WebSocketClosedError(ConnectionError):
    """Raised when the remote peer closes the WebSocket connection."""

    def __init__(self, close_code: int | None, close_reason: str = "") -> None:
        super().__init__(f"WebSocket closed by server (code={close_code}, reason={close_reason!r})")
        self.close_code = close_code
        self.close_reason = close_reason


class StdlibWebSocket:
    """RFC 6455 localhost WebSocket client using only Python stdlib (`>64 KiB` frame capable)."""

    def __init__(self, ws_url: str, timeout: float = 10.0) -> None:
        parsed = urllib.parse.urlsplit(ws_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = f"{parsed.path or '/'}?{parsed.query}" if parsed.query else (parsed.path or "/")
        self.ws_url = ws_url
        self.close_code: int | None = None
        self.close_reason: str = ""
        self._sock = socket.create_connection((host, port), timeout=timeout)
        try:
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            handshake = (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            )
            self._sock.sendall(handshake.encode("ascii"))
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = self._sock.recv(4096)
                if not chunk:
                    raise ConnectionError(f"EOF during WebSocket handshake to {ws_url}")
                buf += chunk
            hdr_bytes, self._rx_buf = buf.split(b"\r\n\r\n", 1)
            hdr_lines = hdr_bytes.decode("latin1").split("\r\n")
            status_line = hdr_lines[0]
            parts = status_line.split(" ", 2)
            status_code = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
            resp_headers: dict[str, str] = {}
            for line in hdr_lines[1:]:
                if ":" in line:
                    hk, hv = line.split(":", 1)
                    resp_headers[hk.strip().lower()] = hv.strip()

            if status_code != 101:
                clen = int(resp_headers.get("content-length", "0") or "0")
                err_body = self._rx_buf
                while clen > len(err_body):
                    more = self._sock.recv(min(4096, clen - len(err_body)))
                    if not more:
                        break
                    err_body += more
                raise WebSocketHandshakeError(
                    status_code=status_code,
                    status_line=status_line,
                    headers=resp_headers,
                    body=err_body,
                )

            expected_accept = base64.b64encode(
                hashlib.sha1((key + WS_GUID).encode("ascii")).digest()
            ).decode("ascii")
            actual_accept = resp_headers.get("sec-websocket-accept", "")
            if actual_accept.lower() != expected_accept.lower():
                raise ConnectionError(
                    f"Invalid Sec-WebSocket-Accept header: {actual_accept!r} != {expected_accept!r}"
                )
        except Exception:
            with contextlib.suppress(OSError):
                self._sock.close()
            raise

    def settimeout(self, timeout: float) -> None:
        self._sock.settimeout(timeout)

    def _recv_exact(self, n: int) -> bytes:
        while len(self._rx_buf) < n:
            chunk = self._sock.recv(max(4096, n - len(self._rx_buf)))
            if not chunk:
                raise WebSocketClosedError(self.close_code or 1006, self.close_reason or "EOF")
            self._rx_buf += chunk
        out, self._rx_buf = self._rx_buf[:n], self._rx_buf[n:]
        return out

    def send_frame(self, payload: bytes, opcode: int = 0x1) -> None:
        mask_key = os.urandom(4)
        n = len(payload)
        hdr = bytearray([0x80 | (opcode & 0x0F)])
        if n < 126:
            hdr.append(0x80 | n)
        elif n < 65536:
            hdr.append(0x80 | 126)
            hdr.extend(struct.pack("!H", n))
        else:
            hdr.append(0x80 | 127)
            hdr.extend(struct.pack("!Q", n))
        hdr.extend(mask_key)
        if n:
            pad = (4 - (n % 4)) % 4
            padded = payload + (b"\x00" * pad)
            masked = (
                int.from_bytes(padded, "big")
                ^ int.from_bytes(mask_key * (len(padded) // 4), "big")
            ).to_bytes(len(padded), "big")[:n]
        else:
            masked = b""
        self._sock.sendall(bytes(hdr) + masked)

    def send_text(self, text: str) -> None:
        self.send_frame(text.encode("utf-8"), opcode=0x1)

    def send_json(self, obj: Any) -> None:
        self.send_text(json.dumps(obj))

    def recv_message(self) -> tuple[int, bytes]:
        msg_op: int | None = None
        parts: list[bytes] = []
        while True:
            b0, b1 = self._recv_exact(2)
            fin = bool(b0 & 0x80)
            op = b0 & 0x0F
            masked = bool(b1 & 0x80)
            n = b1 & 0x7F
            if n == 126:
                (n,) = struct.unpack("!H", self._recv_exact(2))
            elif n == 127:
                (n,) = struct.unpack("!Q", self._recv_exact(8))
            mkey = self._recv_exact(4) if masked else b""
            data = self._recv_exact(n) if n else b""
            if masked and data:
                data = bytes(b ^ mkey[i & 3] for i, b in enumerate(data))
            if op == 0x9:  # Ping -> Pong
                self.send_frame(data, opcode=0xA)
                continue
            if op == 0xA:  # Pong
                continue
            if op == 0x8:  # Close
                if len(data) >= 2:
                    (code,) = struct.unpack("!H", data[:2])
                    reason = data[2:].decode("utf-8", "replace")
                else:
                    code = 1005
                    reason = ""
                self.close_code = code
                self.close_reason = reason
                with contextlib.suppress(OSError):
                    self.send_frame(data[:2] if len(data) >= 2 else b"", opcode=0x8)
                raise WebSocketClosedError(code, reason)
            if op != 0x0:
                msg_op = op
            parts.append(data)
            if fin:
                assert msg_op is not None
                return msg_op, b"".join(parts)

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.send_frame(struct.pack("!H", 1000), opcode=0x8)
        with contextlib.suppress(OSError):
            self._sock.close()

    def __enter__(self) -> "StdlibWebSocket":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def encode_multipart_formdata(
    fields: dict[str, str] | None = None,
    files: list[tuple[str, str, bytes, str]] | None = None,
) -> tuple[str, bytes]:
    """Encode `multipart/form-data` returning `(content_type_header, body_bytes)`."""
    boundary = f"----VisualizeITE2E{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for key, val in (fields or {}).items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{val}\r\n'.encode("utf-8")
        )
    for field_name, filename, data, content_type in (files or []):
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
            + data
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def _assert_expected_status(
    resp: dict[str, Any],
    expected_status: int | tuple[int, ...] | set[int] | list[int] | None,
    method: str,
    path: str,
) -> None:
    if expected_status is None:
        return
    allowed = {expected_status} if isinstance(expected_status, int) else set(expected_status)
    if resp["status"] not in allowed:
        preview = resp["body"][:500].decode("utf-8", "replace")
        raise AssertionError(
            f"{method} {path} returned HTTP {resp['status']}, expected {sorted(allowed)}.\n"
            f"Headers: {resp['headers']}\nBody preview: {preview}"
        )


class StudioAPIClient:
    """HTTP/1.1 Keep-Alive + Range + Multipart + WebSocket opaque-box E2E client."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8790,
        server_proc: subprocess.Popen[Any] | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.host = host
        self.port = port
        self.server_proc = server_proc
        self.timeout = timeout
        self._conn: http.client.HTTPConnection | None = None
        self.client_ports_seen: list[int] = []
        self.last_response: dict[str, Any] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_base_url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def _check_server_alive(self) -> None:
        if self.server_proc is not None and self.server_proc.poll() is not None:
            raise AssertionError(
                f"`visualizeit serve` exited prematurely with code {self.server_proc.returncode}"
            )

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        json_body: Any = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Issue an HTTP/1.1 keep-alive request, draining the body so the TCP socket stays reusable."""
        self._check_server_alive()
        hdrs = {"Connection": "keep-alive", **(headers or {})}
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        if body is not None:
            hdrs["Content-Length"] = str(len(body))

        t_lim = float(timeout if timeout is not None else self.timeout)
        for attempt in range(2):
            if self._conn is None or self._conn.sock is None:
                self._conn = http.client.HTTPConnection(self.host, self.port, timeout=t_lim)
            else:
                self._conn.sock.settimeout(t_lim)
            try:
                self._conn.request(method.upper(), path, body=body, headers=hdrs)
                local_port = int(self._conn.sock.getsockname()[1]) if self._conn.sock is not None else None
                resp = self._conn.getresponse()
                raw = resp.read()
                resp_hdrs = {k.lower(): v for k, v in resp.getheaders()}
                if resp_hdrs.get("connection", "").lower() == "close" or resp.version == 10:
                    self.close()
                if local_port is not None:
                    self.client_ports_seen.append(local_port)

                parsed_json: Any = None
                if "json" in resp_hdrs.get("content-type", "").lower() and raw:
                    with contextlib.suppress(json.JSONDecodeError, UnicodeDecodeError):
                        parsed_json = json.loads(raw.decode("utf-8"))

                result = {
                    "status": int(resp.status),
                    "http_version": int(resp.version),
                    "headers": resp_hdrs,
                    "body": raw,
                    "json": parsed_json,
                    "client_port": local_port,
                }
                self.last_response = result
                return result
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
                self.close()
                if attempt == 1:
                    self._check_server_alive()
                    raise
        raise RuntimeError("Unreachable")

    def get(
        self,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        expected_status: int | tuple[int, ...] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        resp = self.request("GET", path, headers=headers, timeout=timeout)
        _assert_expected_status(resp, expected_status, "GET", path)
        return resp

    def get_json(
        self,
        path: str,
        *,
        expected_status: int | tuple[int, ...] | None = 200,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        resp = self.request("GET", path, headers=headers, timeout=timeout)
        assert resp["http_version"] == 11, f"Expected HTTP/1.1 (11), got {resp['http_version']}"
        _assert_expected_status(resp, expected_status, "GET", path)
        data = json.loads(resp["body"].decode("utf-8"))
        assert isinstance(data, dict), f"Expected JSON object from GET {path}, got {type(data).__name__}"
        return data

    def post_json(
        self,
        path: str,
        payload: Any,
        *,
        expected_status: int | tuple[int, ...] | None = 200,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        resp = self.request("POST", path, headers=headers, json_body=payload, timeout=timeout)
        assert resp["http_version"] == 11, f"Expected HTTP/1.1 (11), got {resp['http_version']}"
        _assert_expected_status(resp, expected_status, "POST", path)
        if not resp["body"]:
            return {}
        data = json.loads(resp["body"].decode("utf-8"))
        return data if isinstance(data, dict) else {"data": data}

    def get_range(
        self,
        path: str,
        start: int | None = 0,
        end: int | None = None,
        *,
        range_header: str | None = None,
        expected_status: int | tuple[int, ...] | None = 206,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if range_header is None:
            assert start is not None, "Either start or range_header must be provided"
            range_val = f"bytes={start}-{'' if end is None else end}"
        else:
            range_val = range_header
        resp = self.request("GET", path, headers={"Range": range_val}, timeout=timeout)
        assert resp["http_version"] == 11, f"Expected HTTP/1.1 (11), got {resp['http_version']}"
        _assert_expected_status(resp, expected_status, "GET (Range)", path)
        return resp

    def upload(
        self,
        path: str,
        *,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        field: str = "file",
        extra_fields: dict[str, str] | None = None,
        expected_status: int | tuple[int, ...] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        ctype, payload = encode_multipart_formdata(
            fields=extra_fields,
            files=[(field, filename, content, content_type)],
        )
        resp = self.request(
            "POST",
            path,
            headers={"Content-Type": ctype},
            body=payload,
            timeout=timeout,
        )
        assert resp["http_version"] == 11, f"Expected HTTP/1.1 (11), got {resp['http_version']}"
        _assert_expected_status(resp, expected_status, "POST (multipart)", path)
        return resp

    def open_ws(self, path: str, *, timeout: float = 10.0) -> StdlibWebSocket:
        self._check_server_alive()
        rel = path if path.startswith("/") else f"/{path}"
        return StdlibWebSocket(f"{self.ws_base_url}{rel}", timeout=timeout)

    def stream_video_ws_raw(
        self,
        video_id: str,
        payload: dict[str, Any] | str | bytes | None,
        *,
        frame_timeout: float = 10.0,
        total_timeout: float = 60.0,
    ) -> dict[str, Any]:
        """Execute `/ws/videos/{video_id}/stream` and return full structured outcome (including errors/close codes)."""
        self._check_server_alive()
        t0 = time.perf_counter()
        frames: list[dict[str, Any]] = []
        pending_meta: dict[str, Any] | None = None
        complete_report: dict[str, Any] | None = None
        error_message: dict[str, Any] | None = None
        close_code: int | None = None
        close_reason: str = ""

        ws_url = f"{self.ws_base_url}/ws/videos/{urllib.parse.quote(str(video_id), safe='')}/stream"
        try:
            ws = StdlibWebSocket(ws_url, timeout=frame_timeout)
        except WebSocketHandshakeError as exc:
            return {
                "upgraded": False,
                "handshake_status": exc.status_code,
                "handshake_body": exc.body,
                "frames": [],
                "complete_report": None,
                "error_message": None,
                "close_code": None,
                "close_reason": exc.status_line,
                "wall_time_s": time.perf_counter() - t0,
            }

        deadline = time.monotonic() + total_timeout
        try:
            if isinstance(payload, dict):
                ws.send_json(payload)
            elif isinstance(payload, str):
                ws.send_text(payload)
            elif isinstance(payload, bytes):
                ws.send_frame(payload, opcode=0x1)

            while True:
                self._check_server_alive()
                rem = deadline - time.monotonic()
                if rem <= 0:
                    raise TimeoutError(f"WebSocket stream exceeded total_timeout={total_timeout}s")
                ws.settimeout(min(frame_timeout, rem))
                try:
                    opcode, raw = ws.recv_message()
                except WebSocketClosedError as exc:
                    close_code = exc.close_code
                    close_reason = exc.close_reason
                    break

                if opcode == 0x1:
                    msg = json.loads(raw.decode("utf-8"))
                    mtype = msg.get("type")
                    if mtype == "frame":
                        if msg.get("jpeg_base64"):
                            msg["jpeg_bytes"] = base64.b64decode(msg["jpeg_base64"])
                            frames.append(msg)
                        else:
                            pending_meta = msg
                    elif mtype == "complete":
                        complete_report = msg.get("report")
                        close_code = 1000
                        break
                    elif mtype == "error" or "error" in msg:
                        error_message = msg
                        # Drain optional server close frame quickly if sent right after error JSON
                        ws.settimeout(0.25)
                        with contextlib.suppress(TimeoutError, OSError):
                            try:
                                ws.recv_message()
                            except WebSocketClosedError as c_exc:
                                close_code = c_exc.close_code
                                close_reason = c_exc.close_reason
                        break
                elif opcode == 0x2:
                    if pending_meta is None:
                        frames.append({"type": "binary_frame", "jpeg_bytes": raw})
                    else:
                        pending_meta["jpeg_bytes"] = raw
                        frames.append(pending_meta)
                        pending_meta = None
        finally:
            ws.close()

        return {
            "upgraded": True,
            "handshake_status": 101,
            "handshake_body": b"",
            "frames": frames,
            "complete_report": complete_report,
            "error_message": error_message,
            "close_code": close_code,
            "close_reason": close_reason,
            "wall_time_s": time.perf_counter() - t0,
        }

    def stream_video_ws(
        self,
        video_id: str,
        payload: dict[str, Any],
        *,
        frame_timeout: float = 10.0,
        total_timeout: float = 60.0,
        raise_on_error: bool = True,
    ) -> Any:
        """Stream `/ws/videos/{video_id}/stream` and return `(frames, complete_report)`.

        Supports both `jpeg_base64` JSON frames and paired JSON metadata + binary JPEG frames.
        If `raise_on_error=False`, returns the full raw outcome dict from `stream_video_ws_raw`.
        """
        outcome = self.stream_video_ws_raw(
            video_id,
            payload,
            frame_timeout=frame_timeout,
            total_timeout=total_timeout,
        )
        if not raise_on_error:
            return outcome

        if not outcome["upgraded"]:
            raise WebSocketHandshakeError(
                status_code=outcome["handshake_status"],
                status_line=outcome["close_reason"],
                headers={},
                body=outcome["handshake_body"],
            )
        if outcome["error_message"] is not None:
            raise RuntimeError(f"Server WebSocket error frame: {outcome['error_message']}")
        if outcome["complete_report"] is None:
            raise WebSocketClosedError(
                outcome["close_code"],
                outcome["close_reason"] or "Stream ended before terminal 'complete' message",
            )
        return outcome["frames"], outcome["complete_report"]

    def close(self) -> None:
        if self._conn is not None:
            with contextlib.suppress(OSError):
                self._conn.close()
            self._conn = None

    def __enter__(self) -> "StudioAPIClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
