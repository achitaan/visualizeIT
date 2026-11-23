"""Shared pytest configuration, XFS TMPDIR enforcement, marker registration, and opaque-box E2E fixtures.

Zero imports of `visualizeit.*` or third-party non-stdlib libraries (other than `pytest`) at
module/collection scope so `pytest --collect-only tests/e2e` always completes in <1.0s.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Generator

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.e2e._helpers.cdp_browser import HeadlessChromiumSession
from tests.e2e._helpers.http_and_ws import StudioAPIClient
from tests.e2e._helpers.paths_and_proc import (
    CACHE_TMP_ROOT,
    E2E_RUNS_ROOT,
    run_cli,
    spawn_live_server,
)


def _enforce_xfs_tmpdir() -> None:
    E2E_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    xfs_tmp = str(E2E_RUNS_ROOT)
    os.environ["TMPDIR"] = xfs_tmp
    os.environ["TEMP"] = xfs_tmp
    os.environ["TMP"] = xfs_tmp
    tempfile.tempdir = None


_enforce_xfs_tmpdir()

_DEMO_MANIFEST_CACHE: dict[str, dict[str, dict[str, Any]]] = {}
_MANAGED_PYTEST_BASETEMP: Path | None = None

_TIER_DIR_MARKERS: tuple[tuple[str, str], ...] = (
    ("tier1_features", "tier1"),
    ("tier2_boundaries", "tier2"),
    ("tier3_combinations", "tier3"),
    ("tier4_workloads", "tier4"),
)


def pytest_configure(config: pytest.Config) -> None:
    global _MANAGED_PYTEST_BASETEMP
    _enforce_xfs_tmpdir()

    basetemp_opt = getattr(config.option, "basetemp", None)
    if not basetemp_opt or str(basetemp_opt).startswith("/tmp"):
        basetemp_dir = CACHE_TMP_ROOT / "pytest_e2e" / f"run_{os.getpid()}"
        basetemp_dir.parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(basetemp_dir)
        _MANAGED_PYTEST_BASETEMP = basetemp_dir

    tier_descriptions = {
        "tier1": "Tier 1: Feature happy-path opaque-box E2E tests (F1-F16)",
        "tier2": "Tier 2: Boundary, corner-case, and negative error-handling opaque-box E2E tests",
        "tier3": "Tier 3: Cross-feature pairwise and multi-surface combination opaque-box E2E tests",
        "tier4": "Tier 4: Real-world end-to-end application workloads and >=15 FPS verification gates",
    }
    for name, desc in tier_descriptions.items():
        config.addinivalue_line("markers", f"{name}: {desc}")

    for idx in range(1, 17):
        config.addinivalue_line("markers", f"f{idx:02d}: Feature F{idx} E2E coverage")
        config.addinivalue_line("markers", f"f{idx}: Feature F{idx} E2E coverage")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        item_path = Path(str(item.fspath))
        parts = set(item_path.parts)
        for folder_name, marker_name in _TIER_DIR_MARKERS:
            if folder_name in parts:
                existing = {m.name for m in item.iter_markers()}
                if marker_name not in existing:
                    item.add_marker(getattr(pytest.mark, marker_name))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _MANAGED_PYTEST_BASETEMP is not None:
        shutil.rmtree(_MANAGED_PYTEST_BASETEMP, ignore_errors=True)
        pytest_e2e_root = CACHE_TMP_ROOT / "pytest_e2e"
        if pytest_e2e_root.is_dir() and not any(pytest_e2e_root.iterdir()):
            shutil.rmtree(pytest_e2e_root, ignore_errors=True)
    if E2E_RUNS_ROOT.is_dir() and not any(E2E_RUNS_ROOT.iterdir()):
        shutil.rmtree(E2E_RUNS_ROOT, ignore_errors=True)


@pytest.fixture(scope="function")
def e2e_tmp_dir() -> Generator[Path, None, None]:
    """Function-scoped XFS temporary directory inside `.cache/tmp/e2e_runs`, deleted on teardown."""
    E2E_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="t_", dir=str(E2E_RUNS_ROOT)))
    try:
        yield work_dir
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def demo_corpus_dir() -> Generator[Path, None, None]:
    """Session-scoped demo video/3D corpus generated via `python -m visualizeit demo-assets --json`."""
    corpus_dir = CACHE_TMP_ROOT / "e2e_demo_corpus"
    shutil.rmtree(corpus_dir, ignore_errors=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    proc = run_cli(
        ["demo-assets", "--output-dir", str(corpus_dir), "--json"],
        timeout=90.0,
        tmp_dir=corpus_dir,
    )
    if proc.returncode != 0:
        shutil.rmtree(corpus_dir, ignore_errors=True)
        pytest.fail(
            f"`visualizeit demo-assets` failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            pytrace=False,
        )

    try:
        payload = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as exc:
        shutil.rmtree(corpus_dir, ignore_errors=True)
        pytest.fail(
            f"`visualizeit demo-assets --json` emitted invalid JSON ({exc}):\nSTDOUT:\n{proc.stdout}",
            pytrace=False,
        )

    clips = payload.get("clips") if isinstance(payload, dict) else None
    if not isinstance(clips, list) or not clips:
        shutil.rmtree(corpus_dir, ignore_errors=True)
        pytest.fail(
            f"`visualizeit demo-assets --json` returned empty or invalid 'clips': {payload!r}",
            pytrace=False,
        )

    _DEMO_MANIFEST_CACHE[str(corpus_dir)] = {
        str(clip["id"]): dict(clip)
        for clip in clips
        if isinstance(clip, dict) and "id" in clip
    }
    try:
        yield corpus_dir
    finally:
        _DEMO_MANIFEST_CACHE.pop(str(corpus_dir), None)
        shutil.rmtree(corpus_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def demo_corpus_manifest(demo_corpus_dir: Path) -> dict[str, dict[str, Any]]:
    """Map of demo clip `id` -> clip metadata dict emitted by `demo-assets --json`."""
    manifest = _DEMO_MANIFEST_CACHE.get(str(demo_corpus_dir))
    if not manifest:
        pytest.fail(f"No cached demo corpus manifest for {demo_corpus_dir}", pytrace=False)
    return {cid: dict(meta) for cid, meta in manifest.items()}


@pytest.fixture(scope="module")
def live_studio_server() -> Generator[StudioAPIClient, None, None]:
    """Module-scoped live `python -m visualizeit serve` instance yielding `StudioAPIClient`.

    Checks `proc.poll()` every 75 ms so an unimplemented `serve` fails cleanly in <0.25s.
    """
    E2E_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    storage_dir = Path(tempfile.mkdtemp(prefix="srv_", dir=str(E2E_RUNS_ROOT)))
    cm = spawn_live_server(
        storage_dir=storage_dir,
        host="127.0.0.1",
        port=None,
        startup_timeout=15.0,
    )
    try:
        try:
            client = cm.__enter__()
        except (AssertionError, RuntimeError, TimeoutError, OSError) as exc:
            shutil.rmtree(storage_dir, ignore_errors=True)
            pytest.fail(str(exc), pytrace=False)
        try:
            yield client
        finally:
            cm.__exit__(None, None, None)
    finally:
        shutil.rmtree(storage_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def chromium_session_factory() -> Generator[Callable[..., HeadlessChromiumSession], None, None]:
    """Function-scoped factory opening isolated headless Chromium 156 sessions with guaranteed teardown."""
    opened: list[HeadlessChromiumSession] = []

    def make_session(
        *,
        fake_video_y4m: Path | None = None,
        download_dir: Path | None = None,
        extra_args: list[str] | None = None,
        enable_fake_media_stream: bool = True,
        startup_timeout: float = 15.0,
    ) -> HeadlessChromiumSession:
        sess = HeadlessChromiumSession(
            fake_video_y4m=fake_video_y4m,
            download_dir=download_dir,
            extra_args=extra_args,
            enable_fake_media_stream=enable_fake_media_stream,
            startup_timeout=startup_timeout,
        )
        opened.append(sess)
        sess.__enter__()
        return sess

    try:
        yield make_session
    finally:
        for sess in reversed(opened):
            sess._teardown()
