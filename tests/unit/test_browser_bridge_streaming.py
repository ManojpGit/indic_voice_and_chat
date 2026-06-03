from __future__ import annotations

import asyncio
import json

import pytest

from src.api.browser_bridge import BrowserVoiceBridge, BrowserBridgeConfig
from src.interfaces.stt import STTStreamEvent
from src.pipeline.vad import EnergyVAD


class _FakeWS:
    def __init__(self):
        self.sent_json = []
        self.sent_bytes = []

    async def send_text(self, t):
        self.sent_json.append(json.loads(t))

    async def send_bytes(self, b):
        self.sent_bytes.append(b)


class _ScriptedSession:
    def __init__(self, events):
        self._events = events
        self.sent = []
        self.closed = False

    async def send(self, pcm):
        self.sent.append(pcm)

    async def events(self):
        for ev in self._events:
            await asyncio.sleep(0)
            yield ev

    async def aclose(self):
        self.closed = True


class _FakeProvider:
    def __init__(self, session):
        self._session = session

    async def open_stream(self, config):
        return self._session


class _FakeAgent:
    def __init__(self):
        self.text_turns = []
        self.state = type("S", (), {
            "state": type("V", (), {"value": "qualifying"})(),
            "is_terminal": False,
        })()
        self.slots = type("Slots", (), {"values": {}})()

    async def handle_turn_text(self, text, sink):
        self.text_turns.append(text)
        from src.dialogue.response_parser import VoiceBotResponse
        from src.pipeline.engine import TurnMetrics, TurnResult

        class _O:
            response = VoiceBotResponse(response_text="जी", action="continue")
            pipeline = TurnResult("u", "hi", 1.0, "{}", 0, TurnMetrics())

        return _O()


def _bridge(events):
    session = _ScriptedSession(events)
    bridge = BrowserVoiceBridge(
        websocket=_FakeWS(),
        agent=_FakeAgent(),
        vad=EnergyVAD(sample_rate=16000, frame_ms=30),
        config=BrowserBridgeConfig(),
        stream_provider=_FakeProvider(session),
    )
    return bridge, session


@pytest.mark.asyncio
async def test_interim_event_emits_partial():
    bridge, session = _bridge([STTStreamEvent(type="interim", text="और कुछ")])
    await bridge._consume_stream_events(session)
    partials = [m for m in bridge._ws.sent_json if m.get("type") == "partial"]
    assert partials and partials[0]["text"] == "और कुछ" and partials[0]["role"] == "user"


@pytest.mark.asyncio
async def test_endpoint_event_dispatches_text_turn():
    bridge, session = _bridge([
        STTStreamEvent(type="interim", text="और कुछ"),
        STTStreamEvent(type="endpoint", text="और कुछ benefits हैं"),
    ])
    await bridge._consume_stream_events(session)
    assert bridge._agent.text_turns == ["और कुछ benefits हैं"]
    transcripts = [m for m in bridge._ws.sent_json
                   if m.get("type") == "transcript" and m.get("role") == "user"]
    assert transcripts and transcripts[-1]["text"] == "और कुछ benefits हैं"


@pytest.mark.asyncio
async def test_endpoint_ignored_while_agent_busy():
    bridge, session = _bridge([STTStreamEvent(type="endpoint", text="x")])
    bridge._agent_busy = True
    await bridge._consume_stream_events(session)
    assert bridge._agent.text_turns == []
