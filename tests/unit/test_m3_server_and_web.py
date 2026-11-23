"""Unit tests for Milestone 3 FastAPI REST, RFC 7233 Range media, WebSocket stream, and Web Studio."""
from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from visualizeit.server import WEB_DIR, create_app

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def client() -> TestClient:
    storage = REPO_ROOT / ".cache" / "tmp" / "unit_m3_storage"
    storage.mkdir(parents=True, exist_ok=True)
    app = create_app(storage_dir=storage)
    with TestClient(app) as tc:
        yield tc


def test_m3_health_demos_presets_and_web_index(client: TestClient) -> None:
    h = client.get("/api/health")
    assert h.status_code == 200
    hj = h.json()
    assert hj["status"] == "ok"
    assert hj["version"] == "1.0.0"
    assert hj["native_raster_available"] is True
    assert hj["demo_clips_ready"] is True

    d = client.get("/api/demos")
    assert d.status_code == 200
    clips = d.json()["clips"]
    assert len(clips) >= 5
    assert any(c["id"] == "planar_card_640x480" for c in clips)

    p = client.get("/api/designs/presets")
    assert p.status_code == 200
    assert len(p.json()["presets"]) >= 4

    idx = client.get("/")
    assert idx.status_code == 200
    assert "text/html" in idx.headers["content-type"]
    for tid in ("mode-video", "mode-camera", "stage-canvas", "export-mp4-btn", "hud-fps"):
        assert f'data-testid="{tid}"' in idx.text

    for fname in ("styles.css", "design-engine.js", "video-tracker-worker.js", "surface-renderer.js", "app.js"):
        assert (WEB_DIR / fname).is_file()
        r = client.get(f"/static/{fname}")
        assert r.status_code == 200


def test_m3_range_media_and_frame_extraction_and_4xx_bounds(client: TestClient) -> None:
    full = client.get("/api/videos/planar_card_640x480/media")
    assert full.status_code == 200
    assert "bytes" in full.headers.get("accept-ranges", "").lower()

    part = client.get("/api/videos/planar_card_640x480/media", headers={"Range": "bytes=0-1023"})
    assert part.status_code == 206
    assert len(part.content) == 1024
    assert part.content == full.content[:1024]

    bad_range = client.get("/api/videos/planar_card_640x480/media", headers={"Range": "bytes=999999999-999999999"})
    assert bad_range.status_code == 416

    f0 = client.get("/api/videos/planar_card_640x480/frame/0")
    assert f0.status_code == 200
    assert "image/jpeg" in f0.headers["content-type"]

    oob = client.get("/api/videos/planar_card_640x480/frame/99999")
    assert oob.status_code in (400, 404)

    nf = client.get("/api/videos/does_not_exist_999")
    assert nf.status_code == 404

    trav = client.get("/api/videos/..%2F..%2Fetc%2Fpasswd")
    assert trav.status_code in (400, 404, 422)


def test_m3_design_select_render_and_websocket_stream(client: TestClient) -> None:
    dr = client.post(
        "/api/designs/render",
        json={"prompt": 'crimson and gold stripes text "UNIT"', "width": 256, "height": 256},
    )
    assert dr.status_code == 200
    design_id = dr.json()["design_id"]

    sel = client.post(
        "/api/videos/planar_card_640x480/select",
        json={"frame_index": 0, "positive_points": [[320, 240]], "bbox": [185, 140, 270, 200]},
    )
    assert sel.status_code == 200
    sel_j = sel.json()
    assert sel_j["foreground_pixels"] > 500
    assert len(base64.b64decode(sel_j["mask_png_base64"])) > 100

    # WebSocket stream bounded to 10 frames
    with client.websocket_connect("/ws/videos/planar_card_640x480/stream") as ws:
        ws.send_json(
            {
                "selection_id": sel_j["selection_id"],
                "design_id": design_id,
                "placement": {"wrap_mode": "planar", "blend_mode": "luminance", "opacity": 0.9},
                "max_frames": 10,
            }
        )
        frames = []
        complete = None
        while True:
            msg = ws.receive_json()
            if msg["type"] == "frame":
                frames.append(msg)
            elif msg["type"] == "complete":
                complete = msg["report"]
                break
        assert len(frames) == 10
        assert [f["frame_index"] for f in frames] == list(range(10))
        assert complete is not None
        assert complete["render_stride"] == 1
        assert complete["processed_frame_count"] == 10
