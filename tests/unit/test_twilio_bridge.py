from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Optional

import pytest

from src.api.telephony_twilio import (
    TWILIO_SAMPLE_RATE,
    TwilioBridgeConfig,
    TwilioMediaBridge,
    voice_twiml,
)
from src.pipeline.audio_utils import pcm16_to_mulaw
from src.pipeline.vad import EndpointConfig, EnergyVAD, VADFrame


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
    def __init__(self, response_audio: Optional[bytes] = None, terminal_after_turn: bool = False) -> None:
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


def _media_frame(mulaw_bytes: bytes, track: str = "inbound") -> dict:
    return {
        "event": "media",
        "media": {
            "payload": base64.b64encode(mulaw_bytes).decode("ascii"),
            "track": track,
            "timestamp": "0",
        },
    }


def _start_frame(stream_sid: str = "MZtest") -> dict:
    return {
        "event": "start",
        "start": {
            "streamSid": stream_sid,
            "callSid": "CAtest",
            "mediaFormat": {
                "encoding": "audio/x-mulaw",
                "sampleRate": 8000,
                "channels": 1,
            },
        },
    }


def _stop_frame() -> dict:
    return {"event": "stop"}


def _silent_mulaw(duration_ms: int) -> bytes:
    # μ-law silence is 0xFF (or 0x7F depending on convention; pcm16_to_mulaw
    # of zeros produces the canonical encoded silence).
    n_samples = int(TWILIO_SAMPLE_RATE * duration_ms / 1000)
    pcm = b"\x00\x00" * n_samples
    return pcm16_to_mulaw(pcm)


def _loud_mulaw(duration_ms: int, amp: int = 8000) -> bytes:
    import math
    import struct
    n_samples = int(TWILIO_SAMPLE_RATE * duration_ms / 1000)
    pcm = b"".join(
        struct.pack("<h", int(amp * math.sin(2 * math.pi * 440 * i / TWILIO_SAMPLE_RATE)))
        for i in range(n_samples)
    )
    return pcm16_to_mulaw(pcm)


# --- Tests ---------------------------------------------------------------


def test_voice_twiml_includes_stream_url() -> None:
    body = voice_twiml("wss://example/stream")
    assert "<Response>" in body
    assert '<Stream url="wss://example/stream"/>' in body


@pytest.mark.asyncio
async def test_bridge_starts_agent_and_handles_hangup_on_stop() -> None:
    agent = FakeAgent()
    ws = FakeWebSocket(incoming=[
        {"event": "connected"},
        _start_frame(),
        _stop_frame(),
    ])
    bridge = TwilioMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
    )
    await bridge.run()
    assert agent.start_called is True
    assert agent.hangup_called is True
    assert agent.turns == []  # no media frames -> no turn dispatched


@pytest.mark.asyncio
async def test_bridge_dispatches_turn_after_endpoint_detected() -> None:
    agent = FakeAgent(response_audio=b"\x00\x00" * 800)  # 50ms PCM16 @ 16kHz

    # 6 loud frames (300ms speech) then 30 silent frames (~900ms silence) — well over thresholds.
    frames = [{"event": "connected"}, _start_frame()]
    for _ in range(6):
        frames.append(_media_frame(_loud_mulaw(50)))  # 50 ms each
    for _ in range(30):
        frames.append(_media_frame(_silent_mulaw(50)))
    frames.append(_stop_frame())

    ws = FakeWebSocket(incoming=frames)
    bridge = TwilioMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
        config=TwilioBridgeConfig(
            pcm_sample_rate=16000,
            endpoint=EndpointConfig(min_speech_ms=60, min_silence_ms=300),
            max_idle_silence_s=60,  # don't hit extended-silence in this test
        ),
    )
    await bridge.run()

    assert len(agent.turns) >= 1
    assert agent.turns[0]  # has captured audio
    # Agent's TTS audio was sent back as media frames
    sent_media = [m for m in ws.sent if m.get("event") == "media"]
    assert sent_media, "expected outbound media frames"
    assert all(m.get("streamSid") == "MZtest" for m in sent_media)


@pytest.mark.asyncio
async def test_bridge_extended_silence_terminates() -> None:
    agent = FakeAgent()
    # All silence frames, enough to exceed max_idle_silence_s
    frames = [{"event": "connected"}, _start_frame()]
    # 200 silent frames * 50ms = 10s of silence
    for _ in range(200):
        frames.append(_media_frame(_silent_mulaw(50)))

    ws = FakeWebSocket(incoming=frames)
    bridge = TwilioMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
        config=TwilioBridgeConfig(
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
        frames.append(_media_frame(_loud_mulaw(50)))
    for _ in range(30):
        frames.append(_media_frame(_silent_mulaw(50)))
    # Lots more frames after — bridge should NOT process them
    for _ in range(50):
        frames.append(_media_frame(_silent_mulaw(50)))
    frames.append(_stop_frame())

    ws = FakeWebSocket(incoming=frames)
    bridge = TwilioMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
        config=TwilioBridgeConfig(
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
        _media_frame(_loud_mulaw(50), track="outbound"),  # echo, ignore
        _stop_frame(),
    ]
    ws = FakeWebSocket(incoming=frames)
    bridge = TwilioMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(),
    )
    await bridge.run()
    assert agent.turns == []


@pytest.mark.asyncio
async def test_records_outcome_on_hangup_when_llm_present(monkeypatch) -> None:
    import src.api.outcome_recorder as orec
    from src.campaign.models import CallAnalysis, LeadCallOutcome

    calls = []

    async def fake_analyze(agent, **kwargs):
        calls.append(kwargs)
        return CallAnalysis(outcome=LeadCallOutcome.INTERESTED, summary="s")

    monkeypatch.setattr(orec, "analyze_agent_call", fake_analyze)

    agent = FakeAgent()
    ws = FakeWebSocket(incoming=[{"event": "connected"}, _start_frame(), _stop_frame()])
    bridge = TwilioMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
        llm=object(),
        tenant_timezone="Asia/Kolkata",
    )
    await bridge.run()
    assert len(calls) == 1  # analyzed once on hangup
    assert calls[0]["tenant_timezone"] == "Asia/Kolkata"
    assert agent.hangup_called is True


@pytest.mark.asyncio
async def test_no_outcome_recording_when_llm_absent(monkeypatch) -> None:
    import src.api.outcome_recorder as orec

    calls = []

    async def fake_analyze(agent, **kwargs):  # pragma: no cover - must not run
        calls.append(kwargs)

    monkeypatch.setattr(orec, "analyze_agent_call", fake_analyze)

    agent = FakeAgent()
    ws = FakeWebSocket(incoming=[{"event": "connected"}, _start_frame(), _stop_frame()])
    bridge = TwilioMediaBridge(  # no llm -> recording is a no-op
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
    )
    await bridge.run()
    assert calls == []
    assert agent.hangup_called is True
