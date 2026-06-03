from __future__ import annotations

import asyncio
import json

import pytest

from src.interfaces.stt import STTConfig
from src.providers.stt.deepgram import DeepgramSTTAdapter, DeepgramStreamSession


def _results(transcript, is_final=False, speech_final=False):
    return json.dumps({
        "type": "Results",
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": transcript, "confidence": 0.9}]},
    })


def test_interim_emits_interim_event():
    s = DeepgramStreamSession(ws=None, start_tasks=False)
    ev = s._handle_raw(_results("और कुछ", is_final=False))
    assert ev is not None and ev.type == "interim" and ev.text == "और कुछ"


def test_final_emits_final_and_accumulates():
    s = DeepgramStreamSession(ws=None, start_tasks=False)
    ev = s._handle_raw(_results("और कुछ benefits", is_final=True))
    assert ev.type == "final" and ev.text == "और कुछ benefits"


def test_speech_final_emits_endpoint_with_accumulated_text():
    s = DeepgramStreamSession(ws=None, start_tasks=False)
    s._handle_raw(_results("और कुछ", is_final=True))
    ev = s._handle_raw(_results("benefits हैं", is_final=True, speech_final=True))
    assert ev.type == "endpoint"
    assert ev.text == "और कुछ benefits हैं"


def test_accumulator_resets_after_endpoint():
    s = DeepgramStreamSession(ws=None, start_tasks=False)
    s._handle_raw(_results("पहला", is_final=True, speech_final=True))
    ev = s._handle_raw(_results("दूसरा", is_final=True, speech_final=True))
    assert ev.text == "दूसरा"


def test_utterance_end_is_backup_only_when_no_speech_final():
    s = DeepgramStreamSession(ws=None, start_tasks=False)
    s._handle_raw(_results("कुछ", is_final=True))
    ev = s._handle_raw(json.dumps({"type": "UtteranceEnd"}))
    assert ev.type == "endpoint" and ev.text == "कुछ"


def test_utterance_end_suppressed_after_speech_final():
    s = DeepgramStreamSession(ws=None, start_tasks=False)
    s._handle_raw(_results("कुछ", is_final=True, speech_final=True))
    ev = s._handle_raw(json.dumps({"type": "UtteranceEnd"}))
    assert ev is None


def test_empty_transcript_ignored():
    s = DeepgramStreamSession(ws=None, start_tasks=False)
    assert s._handle_raw(_results("", is_final=False)) is None


class _FakeWS:
    def __init__(self, incoming):
        self._incoming = list(incoming)
        self.sent: list = []
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._incoming:
            raise StopAsyncIteration
        await asyncio.sleep(0)
        return self._incoming.pop(0)


@pytest.mark.asyncio
async def test_session_streams_events_end_to_end():
    ws = _FakeWS([
        _results("और कुछ", is_final=False),
        _results("और कुछ benefits हैं", is_final=True, speech_final=True),
    ])
    session = DeepgramStreamSession(ws=ws, keepalive_interval=999, start_tasks=True)
    types = []
    async for ev in session.events():
        types.append(ev.type)
    await session.aclose()
    assert types == ["interim", "endpoint"]
    assert ws.closed is True


@pytest.mark.asyncio
async def test_open_stream_builds_url_and_uses_connector():
    captured = {}

    async def fake_connector(url, headers):
        captured["url"] = url
        captured["headers"] = headers
        return _FakeWS([])

    adapter = DeepgramSTTAdapter({
        "api_key": "dg_key",
        "model": "nova-2",
        "language": "hi",
        "endpointing": 300,
        "utterance_end_ms": 1000,
        "connector": fake_connector,
    })
    session = await adapter.open_stream(STTConfig(language="hi", sample_rate=16000))
    await session.aclose()
    assert "model=nova-2" in captured["url"]
    assert "language=hi" in captured["url"]
    assert "endpointing=300" in captured["url"]
    assert "interim_results=true" in captured["url"]
    assert "utterance_end_ms=1000" in captured["url"]
    assert captured["headers"]["Authorization"] == "Token dg_key"


def test_adapter_rejects_missing_key(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(ValueError):
        DeepgramSTTAdapter({})


def test_registry_resolves_deepgram():
    from src.providers import STREAMING_STT_PROVIDERS, get_streaming_stt_provider
    assert STREAMING_STT_PROVIDERS["deepgram"].__name__ == "DeepgramSTTAdapter"
    provider = get_streaming_stt_provider({"provider": "deepgram", "api_key": "x"})
    assert provider.__class__.__name__ == "DeepgramSTTAdapter"


def test_registry_unknown_provider_raises():
    from src.providers import get_streaming_stt_provider, UnknownProviderError
    with pytest.raises(UnknownProviderError):
        get_streaming_stt_provider({"provider": "nope", "api_key": "x"})
