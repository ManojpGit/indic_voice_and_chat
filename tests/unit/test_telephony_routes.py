"""Route-level tests for the Twilio telephony hooks."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import telephony_hooks


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(telephony_hooks.router)
    return app


def test_twilio_voice_returns_twiml_with_stream_url() -> None:
    app = _make_app()
    client = TestClient(app)
    resp = client.post("/telephony/twilio/voice")
    assert resp.status_code == 200
    assert "application/xml" in resp.headers.get("content-type", "")
    body = resp.text
    assert "<Response>" in body
    assert "<Stream url=" in body
    assert "/api/v1/telephony/twilio/stream" in body


def test_websocket_without_factory_closes() -> None:
    # Ensure no factory is registered for this isolated test
    telephony_hooks.set_bridge_factory(None)  # type: ignore[arg-type]
    app = _make_app()
    client = TestClient(app)
    # Connecting and immediately checking close: TestClient surfaces the close.
    with client.websocket_connect("/telephony/twilio/stream") as ws:
        # The server should close the connection because no factory is set.
        # Trying to receive from a closed ws raises WebSocketDisconnect.
        from starlette.websockets import WebSocketDisconnect
        import pytest

        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()


def test_websocket_drives_registered_bridge() -> None:
    received: list[bytes] = []

    class MiniBridge:
        def __init__(self, ws):
            self._ws = ws

        async def run(self):
            # Pretend a single text frame is the entire conversation.
            msg = await self._ws.receive_text()
            received.append(msg.encode())
            await self._ws.send_text("ack")

    telephony_hooks.set_bridge_factory(lambda ws: MiniBridge(ws))
    try:
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/telephony/twilio/stream") as ws:
            ws.send_text("hello")
            assert ws.receive_text() == "ack"
    finally:
        telephony_hooks.set_bridge_factory(None)  # type: ignore[arg-type]

    assert received == [b"hello"]
