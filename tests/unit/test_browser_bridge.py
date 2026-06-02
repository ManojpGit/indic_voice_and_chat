# tests/unit/test_browser_bridge.py
from __future__ import annotations

import json

import pytest

from src.api.browser_bridge import BrowserBridgeConfig, BrowserVoiceBridge
from src.pipeline.vad import EnergyVAD


class FakeWebSocket:
    """Scripted Starlette-style websocket for bridge tests."""

    def __init__(self, incoming: list[dict]):
        self._incoming = list(incoming)
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.closed = False

    async def accept(self) -> None:
        pass

    async def receive(self) -> dict:
        if self._incoming:
            return self._incoming.pop(0)
        return {"type": "websocket.disconnect", "code": 1000}

    async def send_text(self, data: str) -> None:
        self.sent_text.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self, code: int = 1000) -> None:
        self.closed = True


def _bridge(ws, agent=None):
    return BrowserVoiceBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30),
        config=BrowserBridgeConfig(),
    )


@pytest.mark.asyncio
async def test_send_json_emits_text_frame():
    ws = FakeWebSocket([])
    bridge = _bridge(ws)
    await bridge._send_json({"type": "status", "status": "listening"})
    assert json.loads(ws.sent_text[0]) == {"type": "status", "status": "listening"}


@pytest.mark.asyncio
async def test_send_pcm_writes_binary_frames():
    ws = FakeWebSocket([])
    bridge = _bridge(ws)
    await bridge._send_pcm(b"\x01\x02\x03\x04")
    # status:speaking, then the audio bytes, then status:listening
    assert b"".join(ws.sent_bytes) == b"\x01\x02\x03\x04"
    statuses = [json.loads(t).get("status") for t in ws.sent_text]
    assert "speaking" in statuses and statuses[-1] == "listening"
