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


# ---------------------------------------------------------------------------
# Task 3 — run() handshake, opening, turn loop, debug events
# ---------------------------------------------------------------------------
from src.agents.voicebot import TurnOutcome
from src.dialogue.response_parser import VoiceBotResponse
from src.pipeline.engine import TurnResult, TurnMetrics


class _FakeState:
    def __init__(self):
        self.state = type("S", (), {"value": "qualifying"})()
        self.is_terminal = False


class _FakeSlots:
    values = {"interested": True}


class FakeAgent:
    """Minimal agent matching the surface BrowserVoiceBridge touches."""

    def __init__(self):
        self.state = _FakeState()
        self.slots = _FakeSlots()
        self.started = False
        self.opening_played = False
        self.hung_up = False
        self.turns: list[bytes] = []

    async def start(self):
        self.started = True

    async def play_opening(self, sink):
        self.opening_played = True
        await sink(b"\x10\x11")  # fake opening audio

    async def handle_turn(self, captured: bytes, sink) -> TurnOutcome:
        self.turns.append(captured)
        await sink(b"\x20\x21")  # fake reply audio
        return TurnOutcome(
            response=VoiceBotResponse(response_text="Theek hai", action="continue"),
            pipeline=TurnResult(
                user_text="Namaste",
                user_language="hi",
                user_confidence=1.0,
                agent_text='{"response":"Theek hai"}',
                audio_bytes_sent=2,
                metrics=TurnMetrics(),
            ),
        )

    async def handle_hangup(self):
        self.hung_up = True


def _loud_frame(vad) -> bytes:
    return b"\xff\x7f" * (vad.frame_bytes // 2)


def _silent_frame(vad) -> bytes:
    return b"\x00\x00" * (vad.frame_bytes // 2)


@pytest.mark.asyncio
async def test_run_handshake_plays_opening_and_processes_a_turn():
    vad = EnergyVAD(sample_rate=16000, frame_ms=30)
    # hello, then 10 speech frames (300ms) + 25 silence frames (750ms) => one turn.
    incoming = [{"type": "websocket.receive", "text": json.dumps({"type": "hello", "tenant": "dev"})}]
    for _ in range(10):
        incoming.append({"type": "websocket.receive", "bytes": _loud_frame(vad)})
    for _ in range(25):
        incoming.append({"type": "websocket.receive", "bytes": _silent_frame(vad)})
    ws = FakeWebSocket(incoming)
    agent = FakeAgent()
    bridge = BrowserVoiceBridge(websocket=ws, agent=agent, vad=vad, config=BrowserBridgeConfig())

    await bridge.run()

    assert agent.started and agent.opening_played and agent.hung_up
    assert len(agent.turns) == 1 and len(agent.turns[0]) > 0  # captured audio dispatched
    events = [json.loads(t) for t in ws.sent_text]
    transcripts = [(e["role"], e["text"]) for e in events if e["type"] == "transcript"]
    assert ("user", "Namaste") in transcripts
    assert ("agent", "Theek hai") in transcripts
    states = [e for e in events if e["type"] == "state"]
    assert states and states[-1]["state"] == "qualifying"
    assert states[-1]["slots"] == {"interested": True}
