"""Process execution, XFS path hygiene, CLI runners, and VideoRunReport invariant checks.

Strictly stdlib-only at module import time so `pytest --collect-only` always succeeds in <1s.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from .http_and_ws import StudioAPIClient

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
CACHE_DIR: Path = REPO_ROOT / ".cache"
CACHE_TMP_ROOT: Path = CACHE_DIR / "tmp"
E2E_RUNS_ROOT: Path = CACHE_TMP_ROOT / "e2e_runs"
E2E_TMP_ROOT: Path = E2E_RUNS_ROOT
NATIVE_BUILD_DIR: Path = CACHE_DIR / "native-build"
DEMO_VIDEOS_DIR: Path = CACHE_DIR / "demo-videos"
SETUP_ENV_SCRIPT: Path = REPO_ROOT / "scripts" / "setup_env.sh"
SO_PATH: Path = NATIVE_BUILD_DIR / "libvisualizeit_pose.so"
VENV_PYTHON: Path = CACHE_DIR / "venv" / "bin" / "python"

VALID_WRAP_MODES = frozenset({"planar", "cylinder", "mesh_grid", "mesh_6dof"})
VALID_BLEND_MODES = frozenset({"luminance", "normal", "ink"})
REQUIRED_FRAME_LATENCY_KEYS = ("mean", "p50", "p95", "p99", "max")
REQUIRED_STAGE_LATENCY_KEYS = ("decode_ms", "prep_ms", "track_ms", "composite_ms", "encode_ms")
REQUIRED_VIDEO_RUN_REPORT_KEYS = (
    "input_path",
    "output_path",
    "width",
    "height",
    "source_fps",
    "wrap_mode",
    "blend_mode",
    "raster_backend",
    "render_stride",
    "input_frame_count",
    "processed_frame_count",
    "rendered_frame_count",
    "output_frame_count",
    "tracked_frame_count",
    "lost_frame_count",
    "recovered_count",
    "tracking_availability",
    "wall_time_seconds",
    "effective_fps",
    "steady_state_fps",
    "frame_latency_ms",
    "stage_latency_mean_ms",
    "inlier_count_median",
    "reprojection_rmse_px",
)


def get_venv_python() -> Path:
    """Return `.cache/venv/bin/python` if provisioned and executable, else `sys.executable`."""
    if VENV_PYTHON.is_file() and os.access(VENV_PYTHON, os.X_OK):
        return VENV_PYTHON
    return Path(sys.executable)


def assert_xfs_temp_path(path: Path) -> Path:
    """Verify a temporary or runtime directory never targets `/tmp` or `.agents/teamwork`."""
    resolved = path.resolve()
    s = str(resolved)
    if s == "/tmp" or s.startswith("/tmp/"):
        raise AssertionError(f"Forbidden /tmp path in E2E harness: {resolved}")
    if ".agents/teamwork" in s:
        raise AssertionError(f"Forbidden .agents/teamwork path for runtime data: {resolved}")
    return resolved


def build_xfs_env(
    tmp_dir: Path | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Construct a subprocess environment with TMPDIR/TEMP/TMP anchored on XFS `.cache/tmp`."""
    target_tmp = assert_xfs_temp_path(tmp_dir or E2E_RUNS_ROOT)
    target_tmp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "TMPDIR": str(target_tmp),
            "TEMP": str(target_tmp),
            "TMP": str(target_tmp),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if extra:
        env.update(extra)
        for var in ("TMPDIR", "TEMP", "TMP"):
            if var in env and env[var]:
                assert_xfs_temp_path(Path(env[var]))
    return env


def find_free_port(host: str = "127.0.0.1") -> int:
    """Bind an ephemeral TCP port on `host` and return the port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def terminate_process_group(proc: subprocess.Popen[Any], *, sigterm_timeout: float = 3.0) -> None:
    """Cleanly terminate a `start_new_session=True` process group with zero orphans or open pipes."""
    if proc.poll() is None:
        pgid = proc.pid
        for sig, wait_s in ((signal.SIGTERM, sigterm_timeout), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            except PermissionError:
                try:
                    proc.kill() if sig == signal.SIGKILL else proc.terminate()
                except ProcessLookupError:
                    break
            try:
                proc.wait(timeout=wait_s)
                break
            except subprocess.TimeoutExpired:
                continue
    else:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=1.0)

    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.close()


def run_subproc(
    cmd: list[str],
    *,
    timeout: float = 60.0,
    check: bool = False,
    cwd: Path | None = None,
    tmp_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an external command in its own process group with finite timeout and XFS TMPDIR."""
    env = build_xfs_env(tmp_dir=tmp_dir, extra=extra_env)
    work_dir = str(cwd or REPO_ROOT)
    proc = subprocess.Popen(
        [str(part) for part in cmd],
        cwd=work_dir,
        env=env,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_process_group(proc, sigterm_timeout=1.0)
        raise
    finally:
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()

    completed = subprocess.CompletedProcess(
        args=proc.args,
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout or "",
        stderr=stderr or "",
    )
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def run_cli(
    args: list[str],
    *,
    timeout: float = 60.0,
    check: bool = False,
    cwd: Path | None = None,
    tmp_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke `python -m visualizeit <args>` via `.cache/venv/bin/python` with XFS TMPDIR."""
    cmd = [str(get_venv_python()), "-m", "visualizeit", *[str(a) for a in args]]
    return run_subproc(
        cmd,
        timeout=timeout,
        check=check,
        cwd=cwd or REPO_ROOT,
        tmp_dir=tmp_dir,
        extra_env=extra_env,
    )


def run_cli_json(
    args: list[str],
    *,
    timeout: float = 60.0,
    expected_rc: int = 0,
    tmp_dir: Path | None = None,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Invoke `python -m visualizeit ... --json`, verify exit code, and parse JSON stdout."""
    cli_args = list(args)
    if "--json" not in cli_args:
        cli_args.append("--json")
    proc = run_cli(
        cli_args,
        timeout=timeout,
        check=False,
        cwd=cwd,
        tmp_dir=tmp_dir,
        extra_env=extra_env,
    )
    assert proc.returncode == expected_rc, (
        f"Expected CLI exit code {expected_rc}, got {proc.returncode} for args={cli_args!r}\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
    raw_out = proc.stdout.strip()
    assert raw_out, (
        f"Expected JSON on stdout for args={cli_args!r}, got empty stdout.\nSTDERR:\n{proc.stderr}"
    )
    try:
        parsed = json.loads(raw_out)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"CLI stdout is not valid JSON for args={cli_args!r}: {exc}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        ) from exc
    assert isinstance(parsed, dict), f"Expected JSON object from CLI, got {type(parsed).__name__}"
    return proc, parsed


def assert_no_python_traceback(proc_or_text: subprocess.CompletedProcess[str] | str) -> None:
    """Assert that output contains zero uncaught Python stack traces."""
    if isinstance(proc_or_text, subprocess.CompletedProcess):
        combined = f"{proc_or_text.stdout or ''}\n{proc_or_text.stderr or ''}"
    else:
        combined = str(proc_or_text)
    assert "Traceback (most recent call last)" not in combined, (
        f"Unexpected uncaught Python traceback found in output:\n{combined}"
    )


def assert_cli_usage_error(proc: subprocess.CompletedProcess[str]) -> None:
    """Assert exit code 2, non-empty diagnostic message, and zero uncaught stack trace."""
    assert proc.returncode == 2, (
        f"Expected CLI usage/input error exit code 2, got {proc.returncode}.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    combined = ((proc.stderr or "") + (proc.stdout or "")).strip()
    assert combined, "CLI exited with code 2 but emitted empty diagnostic output on stderr/stdout"
    assert_no_python_traceback(proc)


def _assert_finite_number(val: Any, field_name: str) -> float:
    assert isinstance(val, (int, float)) and not isinstance(val, bool), (
        f"Expected numeric {field_name}, got {val!r} ({type(val).__name__})"
    )
    fval = float(val)
    assert math.isfinite(fval), f"Expected finite {field_name}, got {val!r}"
    return fval


def assert_video_run_report_invariants(
    report: dict[str, Any],
    *,
    require_min_fps: float | None = None,
    expect_output_video: bool = True,
) -> None:
    """Validate all PROJECT.md §3 `VideoRunReport` schema, stride=1, frame-count, and telemetry invariants."""
    assert isinstance(report, dict), f"Expected VideoRunReport dict, got {type(report).__name__}"
    missing = [k for k in REQUIRED_VIDEO_RUN_REPORT_KEYS if k not in report]
    assert not missing, f"VideoRunReport missing required keys: {missing}. Present: {sorted(report.keys())}"

    assert isinstance(report["input_path"], str) and report["input_path"].strip(), (
        f"Invalid input_path: {report['input_path']!r}"
    )
    if expect_output_video:
        assert isinstance(report["output_path"], str) and report["output_path"].strip(), (
            f"Expected non-empty output_path when expect_output_video=True, got {report['output_path']!r}"
        )
    else:
        assert report["output_path"] is None or isinstance(report["output_path"], str), (
            f"Invalid output_path: {report['output_path']!r}"
        )

    width = report["width"]
    height = report["height"]
    assert isinstance(width, int) and width > 0, f"Invalid width: {width!r}"
    assert isinstance(height, int) and height > 0, f"Invalid height: {height!r}"

    source_fps = _assert_finite_number(report["source_fps"], "source_fps")
    assert source_fps > 0.0, f"Expected source_fps > 0, got {source_fps}"

    assert report["wrap_mode"] in VALID_WRAP_MODES, (
        f"Invalid wrap_mode={report['wrap_mode']!r}, expected one of {sorted(VALID_WRAP_MODES)}"
    )
    assert report["blend_mode"] in VALID_BLEND_MODES, (
        f"Invalid blend_mode={report['blend_mode']!r}, expected one of {sorted(VALID_BLEND_MODES)}"
    )
    assert isinstance(report["raster_backend"], str) and report["raster_backend"].strip(), (
        f"Invalid raster_backend: {report['raster_backend']!r}"
    )

    assert report["render_stride"] == 1, (
        f"Anti-cheating invariant violated: render_stride must equal 1, got {report['render_stride']!r}"
    )

    in_f = report["input_frame_count"]
    proc_f = report["processed_frame_count"]
    rend_f = report["rendered_frame_count"]
    out_f = report["output_frame_count"]
    for fname, fval in (
        ("input_frame_count", in_f),
        ("processed_frame_count", proc_f),
        ("rendered_frame_count", rend_f),
        ("output_frame_count", out_f),
    ):
        assert isinstance(fval, int) and not isinstance(fval, bool) and fval >= 0, (
            f"Invalid {fname}: {fval!r}"
        )

    assert in_f > 0, f"Expected input_frame_count > 0, got {in_f}"
    assert proc_f > 0, f"Expected processed_frame_count > 0, got {proc_f}"
    if expect_output_video:
        assert in_f == proc_f == rend_f == out_f, (
            "4-way frame count equality violated: "
            f"input={in_f}, processed={proc_f}, rendered={rend_f}, output={out_f}"
        )
    else:
        assert proc_f == rend_f and proc_f <= in_f, (
            f"Stream frame count invariant violated: input={in_f}, processed={proc_f}, rendered={rend_f}"
        )

    tracked_f = report["tracked_frame_count"]
    lost_f = report["lost_frame_count"]
    rec_c = report["recovered_count"]
    for fname, fval in (
        ("tracked_frame_count", tracked_f),
        ("lost_frame_count", lost_f),
        ("recovered_count", rec_c),
    ):
        assert isinstance(fval, int) and not isinstance(fval, bool) and fval >= 0, (
            f"Invalid {fname}: {fval!r}"
        )
    assert tracked_f <= proc_f and lost_f <= proc_f and (tracked_f + lost_f) <= proc_f, (
        f"Inconsistent tracking counts: tracked={tracked_f}, lost={lost_f}, processed={proc_f}"
    )

    avail = _assert_finite_number(report["tracking_availability"], "tracking_availability")
    assert -1e-6 <= avail <= 1.0 + 1e-6, f"tracking_availability out of [0, 1]: {avail}"

    wall_s = _assert_finite_number(report["wall_time_seconds"], "wall_time_seconds")
    eff_fps = _assert_finite_number(report["effective_fps"], "effective_fps")
    steady_fps = _assert_finite_number(report["steady_state_fps"], "steady_state_fps")
    assert wall_s > 0.0, f"Expected wall_time_seconds > 0, got {wall_s}"
    assert eff_fps > 0.0, f"Expected effective_fps > 0, got {eff_fps}"
    assert steady_fps > 0.0, f"Expected steady_state_fps > 0, got {steady_fps}"

    fl = report["frame_latency_ms"]
    assert isinstance(fl, dict), f"Expected frame_latency_ms dict, got {fl!r}"
    for k in REQUIRED_FRAME_LATENCY_KEYS:
        assert k in fl, f"frame_latency_ms missing key {k!r}: {fl}"
    fl_vals = {k: _assert_finite_number(fl[k], f"frame_latency_ms.{k}") for k in REQUIRED_FRAME_LATENCY_KEYS}
    for k, v in fl_vals.items():
        assert v >= 0.0, f"frame_latency_ms.{k} must be >= 0, got {v}"
    assert fl_vals["mean"] > 0.0 and fl_vals["max"] > 0.0, f"frame_latency_ms must be > 0: {fl_vals}"
    assert fl_vals["p50"] <= fl_vals["p95"] + 1e-5, f"Non-monotonic p50 > p95: {fl_vals}"
    assert fl_vals["p95"] <= fl_vals["p99"] + 1e-5, f"Non-monotonic p95 > p99: {fl_vals}"
    assert fl_vals["p99"] <= fl_vals["max"] + 1e-5, f"Non-monotonic p99 > max: {fl_vals}"
    assert fl_vals["mean"] <= fl_vals["max"] + 1e-4, f"Inconsistent mean > max: {fl_vals}"

    sl = report["stage_latency_mean_ms"]
    assert isinstance(sl, dict), f"Expected stage_latency_mean_ms dict, got {sl!r}"
    for k in REQUIRED_STAGE_LATENCY_KEYS:
        assert k in sl, f"stage_latency_mean_ms missing key {k!r}: {sl}"
        sv = _assert_finite_number(sl[k], f"stage_latency_mean_ms.{k}")
        assert sv >= 0.0, f"stage_latency_mean_ms.{k} must be >= 0, got {sv}"

    inlier_med = _assert_finite_number(report["inlier_count_median"], "inlier_count_median")
    assert inlier_med >= 0.0, f"inlier_count_median must be >= 0, got {inlier_med}"

    rmse = report["reprojection_rmse_px"]
    if rmse is not None:
        rmse_f = _assert_finite_number(rmse, "reprojection_rmse_px")
        assert rmse_f >= 0.0, f"reprojection_rmse_px must be >= 0, got {rmse_f}"

    if require_min_fps is not None:
        assert eff_fps >= float(require_min_fps), (
            f"Performance gate failed: effective_fps={eff_fps:.2f} < require_min_fps={require_min_fps:.2f}"
        )


@contextlib.contextmanager
def spawn_live_server(
    *,
    storage_dir: Path,
    host: str = "127.0.0.1",
    port: int | None = None,
    startup_timeout: float = 15.0,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> Generator[StudioAPIClient, None, None]:
    """Spawn `python -m visualizeit serve` and yield a connected `StudioAPIClient`.

    Polls `proc.poll()` every 75 ms during readiness wait so an unimplemented or broken
    `serve` subcommand fails cleanly in <0.25s without hanging.
    """
    from .http_and_ws import StudioAPIClient

    resolved_storage = assert_xfs_temp_path(storage_dir)
    resolved_storage.mkdir(parents=True, exist_ok=True)
    srv_tmp = resolved_storage / "_srv_tmp"
    srv_tmp.mkdir(parents=True, exist_ok=True)

    bind_port = port if port is not None else find_free_port(host)
    cmd = [
        str(get_venv_python()),
        "-m",
        "visualizeit",
        "serve",
        "--host",
        host,
        "--port",
        str(bind_port),
        "--storage-dir",
        str(resolved_storage),
        *[str(a) for a in (extra_args or [])],
    ]
    env = build_xfs_env(tmp_dir=srv_tmp, extra=extra_env)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    client: StudioAPIClient | None = None
    try:
        deadline = time.monotonic() + startup_timeout
        ready = False
        last_err = ""
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=2.0)
                raise AssertionError(
                    f"`visualizeit serve` exited prematurely with code {proc.returncode}.\n"
                    f"CMD: {cmd}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                )
            conn = http.client.HTTPConnection(host, bind_port, timeout=0.5)
            try:
                conn.request("GET", "/api/health")
                resp = conn.getresponse()
                body = resp.read()
                if resp.status == 200:
                    ready = True
                    break
                last_err = f"HTTP {resp.status}: {body[:200]!r}"
            except OSError as exc:
                last_err = str(exc)
            finally:
                with contextlib.suppress(OSError):
                    conn.close()

            if proc.poll() is not None:
                continue
            time.sleep(0.075)

        if not ready:
            terminate_process_group(proc, sigterm_timeout=1.5)
            raise TimeoutError(
                f"Timed out after {startup_timeout}s waiting for GET http://{host}:{bind_port}/api/health == 200 "
                f"(last error: {last_err})"
            )

        client = StudioAPIClient(host=host, port=bind_port, server_proc=proc)
        yield client
    finally:
        if client is not None:
            client.close()
        terminate_process_group(proc, sigterm_timeout=3.0)
