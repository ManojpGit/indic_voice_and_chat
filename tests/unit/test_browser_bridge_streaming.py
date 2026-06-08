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

    async def handle_turn_text(self, text, sink, cancel_event=None):
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


def test_build_streaming_provider_from_tenant():
    from types import SimpleNamespace
    from src.api.dev_console import _build_stream_provider

    tenant = SimpleNamespace(
        settings=SimpleNamespace(pipeline=SimpleNamespace(
            stt_streaming=SimpleNamespace(
                provider="deepgram", model="nova-2", language="hi",
                endpointing=300, utterance_end_ms=1000,
                api_key_env="TENANT_DEV_DEEPGRAM_KEY",
            )
        )),
        secret=lambda env: "dg_secret" if env == "TENANT_DEV_DEEPGRAM_KEY" else None,
    )
    provider = _build_stream_provider(tenant)
    assert provider.__class__.__name__ == "DeepgramSTTAdapter"


def test_build_streaming_provider_none_when_unconfigured():
    from types import SimpleNamespace
    from src.api.dev_console import _build_stream_provider

    tenant = SimpleNamespace(
        settings=SimpleNamespace(pipeline=SimpleNamespace(stt_streaming=None)),
        secret=lambda env: None,
    )
    assert _build_stream_provider(tenant) is None


@pytest.mark.asyncio
async def test_handle_barge_in_cancels_when_busy():
    bridge, session = _bridge([])
    import asyncio as _a
    bridge._agent_busy = True
    bridge._cancel_event = _a.Event()
    bridge._handle_barge_in()
    assert bridge._cancel_event.is_set() is True
    assert bridge._agent_busy is False


@pytest.mark.asyncio
async def test_handle_barge_in_noop_when_idle():
    bridge, session = _bridge([])
    import asyncio as _a
    bridge._agent_busy = False
    bridge._cancel_event = _a.Event()
    bridge._handle_barge_in()
    assert bridge._cancel_event.is_set() is False  # untouched


@pytest.mark.asyncio
async def test_cancelled_turn_skips_agent_transcript():
    from src.dialogue.response_parser import VoiceBotResponse
    from src.pipeline.engine import TurnMetrics, TurnResult

    class _CancelAgent:
        state = type("S", (), {"state": type("V", (), {"value": "listening"})(), "is_terminal": False})()
        slots = type("SL", (), {"values": {}})()

        async def handle_turn_text(self, text, sink, cancel_event=None):
            class _O:
                response = VoiceBotResponse(response_text="जी हाँ सुन", action="continue", parse_error="barge-in")
                pipeline = TurnResult("u", "hi", 1.0, "{}", 0, TurnMetrics(), cancelled=True)
            return _O()

    from src.api.browser_bridge import BrowserVoiceBridge, BrowserBridgeConfig
    from src.pipeline.vad import EnergyVAD
    bridge = BrowserVoiceBridge(
        websocket=_FakeWS(), agent=_CancelAgent(),
        vad=EnergyVAD(sample_rate=16000, frame_ms=30), config=BrowserBridgeConfig(),
    )
    await bridge._dispatch_text_turn("और कुछ?")
    agent_msgs = [m for m in bridge._ws.sent_json
                  if m.get("type") == "transcript" and m.get("role") == "agent"]
    assert agent_msgs == []  # abandoned reply not emitted
    statuses = [m["status"] for m in bridge._ws.sent_json if m.get("type") == "status"]
    assert statuses[-1] == "listening"


@pytest.mark.asyncio
async def test_dispatch_arms_then_disarms_barge():
    # A normal turn should arm barge-in at the start and disarm at the end, so
    # the browser only allows interruptions during a cancellable turn.
    bridge, session = _bridge([])
    await bridge._dispatch_text_turn("hi")
    barge = [m for m in bridge._ws.sent_json if m.get("type") == "barge"]
    assert barge[0] == {"type": "barge", "armed": True}
    assert barge[-1] == {"type": "barge", "armed": False}


@pytest.mark.asyncio
async def test_endpoint_gap_ms_logged(caplog):
    import logging
    bridge, session = _bridge([
        STTStreamEvent(type="interim", text="haan"),
        STTStreamEvent(type="endpoint", text="haan ji boliye"),
    ])
    with caplog.at_level(logging.INFO):
        await bridge._consume_stream_events(session)
    recs = [r for r in caplog.records if r.getMessage() == "browser turn (stream)"]
    assert recs, "no 'browser turn (stream)' log emitted"
    gap = getattr(recs[0], "endpoint_gap_ms", None)
    assert gap is not None and gap >= 0
