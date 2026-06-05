from __future__ import annotations

import asyncio
import base64
import json
import math
import struct
from dataclasses import dataclass
from typing import Optional

import pytest

from src.api.telephony_exotel import (
    EXOTEL_SAMPLE_RATE,
    ExotelBridgeConfig,
    ExotelMediaBridge,
    voicebot_xml,
)
from src.pipeline.vad import EndpointConfig, EnergyVAD


# --- Fakes ---------------------------------------------------------------


class FakeWebSocket:
    def __init__(self, incoming: list[dict]) -> None:
        self._incoming = list(incoming)
        self.sent: list[dict] = []
        self._closed = False

    async def send_text(self, data: str) -> None:
        if self._closed:
            raise RuntimeError("ws closed")
        self.sent.append(json.loads(data))

    async def receive_text(self) -> str:
        if not self._incoming:
            self._closed = True
            raise asyncio.CancelledError("no more frames")
        return json.dumps(self._incoming.pop(0))


@dataclass
class _AgentState:
    is_terminal: bool = False


class FakeAgent:
    def __init__(
        self,
        response_audio: Optional[bytes] = None,
        terminal_after_turn: bool = False,
    ) -> None:
        self._response_audio = response_audio or b""
        self._terminate = terminal_after_turn
        self.start_called = False
        self.turns: list[bytes] = []
        self.hangup_called = False
        self.extended_silence_called = False
        self.state = _AgentState()

    async def start(self) -> None:
        self.start_called = True

    async def handle_turn(self, captured_audio: bytes, audio_sink) -> None:
        self.turns.append(captured_audio)
        if self._response_audio:
            await audio_sink(self._response_audio)
        if self._terminate:
            self.state.is_terminal = True

    async def handle_extended_silence(self) -> None:
        self.extended_silence_called = True
        self.state.is_terminal = True

    async def handle_hangup(self) -> None:
        self.hangup_called = True


# --- Helpers -------------------------------------------------------------


def _media_frame(pcm16_bytes: bytes, track: str = "inbound") -> dict:
    """Exotel media frame — payload is raw PCM16 LE @ 8kHz, base64-encoded."""
    return {
        "event": "media",
        "stream_sid": "EXstreamtest",
        "media": {
            "payload": base64.b64encode(pcm16_bytes).decode("ascii"),
            "track": track,
            "timestamp": "0",
            "chunk": 1,
        },
    }


def _start_frame(stream_sid: str = "EXstreamtest") -> dict:
    return {
        "event": "start",
        "stream_sid": stream_sid,
        "call_sid": "EXcalltest",
        "media_format": {"encoding": "pcm", "sample_rate": 8000, "channels": 1},
    }


def _stop_frame() -> dict:
    return {"event": "stop"}


def _silent_pcm16(duration_ms: int) -> bytes:
    n_samples = int(EXOTEL_SAMPLE_RATE * duration_ms / 1000)
    return b"\x00\x00" * n_samples


def _loud_pcm16(duration_ms: int, amp: int = 8000) -> bytes:
    n_samples = int(EXOTEL_SAMPLE_RATE * duration_ms / 1000)
    return b"".join(
        struct.pack("<h", int(amp * math.sin(2 * math.pi * 440 * i / EXOTEL_SAMPLE_RATE)))
        for i in range(n_samples)
    )


# --- Tests ---------------------------------------------------------------


def test_voicebot_xml_includes_stream_url() -> None:
    body = voicebot_xml("wss://example/exotel/stream")
    assert "<Response>" in body
    assert '<Stream url="wss://example/exotel/stream"/>' in body


@pytest.mark.asyncio
async def test_bridge_starts_agent_and_handles_hangup_on_stop() -> None:
    agent = FakeAgent()
    ws = FakeWebSocket(incoming=[
        {"event": "connected"},
        _start_frame(),
        _stop_frame(),
    ])
    bridge = ExotelMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
    )
    await bridge.run()
    assert agent.start_called is True
    assert agent.hangup_called is True
    assert agent.turns == []


@pytest.mark.asyncio
async def test_bridge_dispatches_turn_after_endpoint_detected() -> None:
    agent = FakeAgent(response_audio=b"\x00\x00" * 800)  # 50ms PCM16 @ 16kHz

    frames = [{"event": "connected"}, _start_frame()]
    for _ in range(6):
        frames.append(_media_frame(_loud_pcm16(50)))
    for _ in range(30):
        frames.append(_media_frame(_silent_pcm16(50)))
    frames.append(_stop_frame())

    ws = FakeWebSocket(incoming=frames)
    bridge = ExotelMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
        config=ExotelBridgeConfig(
            pcm_sample_rate=16000,
            endpoint=EndpointConfig(min_speech_ms=60, min_silence_ms=300),
            max_idle_silence_s=60,
        ),
    )
    await bridge.run()

    assert len(agent.turns) >= 1
    assert agent.turns[0]
    sent_media = [m for m in ws.sent if m.get("event") == "media"]
    assert sent_media, "expected outbound media frames"
    assert all(m.get("stream_sid") == "EXstreamtest" for m in sent_media)
    # Verify outbound payloads are *not* μ-law: each frame should be 320 bytes
    # of PCM16 (= 20ms @ 8kHz mono) after base64 decode, or smaller for the last.
    for m in sent_media[:-1]:
        decoded = base64.b64decode(m["media"]["payload"])
        assert len(decoded) == 320, f"expected 320-byte PCM16 chunks, got {len(decoded)}"


@pytest.mark.asyncio
async def test_bridge_extended_silence_terminates() -> None:
    agent = FakeAgent()
    frames = [{"event": "connected"}, _start_frame()]
    for _ in range(200):  # 200 * 50ms = 10s silence
        frames.append(_media_frame(_silent_pcm16(50)))

    ws = FakeWebSocket(incoming=frames)
    bridge = ExotelMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
        config=ExotelBridgeConfig(
            max_idle_silence_s=2.0,
            endpoint=EndpointConfig(min_speech_ms=300, min_silence_ms=600),
        ),
    )
    await bridge.run()
    assert agent.extended_silence_called is True
    assert agent.hangup_called is True


@pytest.mark.asyncio
async def test_bridge_terminates_when_agent_state_terminal() -> None:
    agent = FakeAgent(response_audio=b"\x00\x00" * 100, terminal_after_turn=True)
    frames = [{"event": "connected"}, _start_frame()]
    for _ in range(6):
        frames.append(_media_frame(_loud_pcm16(50)))
    for _ in range(30):
        frames.append(_media_frame(_silent_pcm16(50)))
    for _ in range(50):
        frames.append(_media_frame(_silent_pcm16(50)))
    frames.append(_stop_frame())

    ws = FakeWebSocket(incoming=frames)
    bridge = ExotelMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
        config=ExotelBridgeConfig(
            pcm_sample_rate=16000,
            endpoint=EndpointConfig(min_speech_ms=60, min_silence_ms=300),
            max_idle_silence_s=60,
        ),
    )
    await bridge.run()
    assert agent.state.is_terminal
    assert len(agent.turns) == 1


@pytest.mark.asyncio
async def test_bridge_ignores_outbound_track_frames() -> None:
    agent = FakeAgent()
    frames = [
        {"event": "connected"},
        _start_frame(),
        _media_frame(_loud_pcm16(50), track="outbound"),
        _stop_frame(),
    ]
    ws = FakeWebSocket(incoming=frames)
    bridge = ExotelMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(),
    )
    await bridge.run()
    assert agent.turns == []


@pytest.mark.asyncio
async def test_send_pcm_skips_when_no_stream_sid() -> None:
    """`_send_pcm` must early-return if `start` hasn't populated stream_sid."""
    agent = FakeAgent()
    ws = FakeWebSocket(incoming=[])
    bridge = ExotelMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(),
    )
    assert bridge._stream_sid is None
    await bridge._send_pcm(b"\x00\x00" * 1000)
    assert ws.sent == []


@pytest.mark.asyncio
async def test_outbound_chunk_size_is_20ms_of_pcm16() -> None:
    """Pacing requirement: each frame should be 320 bytes (=20ms PCM16@8k)."""
    agent = FakeAgent()
    ws = FakeWebSocket(incoming=[])
    bridge = ExotelMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(),
        config=ExotelBridgeConfig(pcm_sample_rate=8000),  # skip resample for simpler accounting
    )
    bridge._stream_sid = "EXtest"
    # Send 100ms worth of PCM16 @ 8k = 1600 bytes -> expect 5 frames of 320 bytes
    payload = b"\x01\x02" * 800
    await bridge._send_pcm(payload)
    assert len(ws.sent) == 5
    for m in ws.sent:
        decoded = base64.b64decode(m["media"]["payload"])
        assert len(decoded) == 320


@pytest.mark.asyncio
async def test_exotel_records_outcome_on_hangup_when_llm_present(monkeypatch) -> None:
    import src.api.outcome_recorder as orec
    from src.campaign.models import CallAnalysis, LeadCallOutcome

    calls = []

    async def fake_analyze(agent, **kwargs):
        calls.append(kwargs)
        return CallAnalysis(outcome=LeadCallOutcome.NOT_INTERESTED, summary="s")

    monkeypatch.setattr(orec, "analyze_agent_call", fake_analyze)

    agent = FakeAgent()
    ws = FakeWebSocket(incoming=[{"event": "connected"}, _start_frame(), _stop_frame()])
    bridge = ExotelMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
        llm=object(),
        tenant_timezone="Asia/Kolkata",
    )
    await bridge.run()
    assert len(calls) == 1
    assert agent.hangup_called is True
