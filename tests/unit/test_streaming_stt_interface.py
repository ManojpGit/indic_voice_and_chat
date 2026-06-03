from __future__ import annotations

import inspect

import pytest

from src.interfaces.stt import (
    ISTTStreamSession,
    IStreamingSTTProvider,
    STTStreamEvent,
)


def test_stream_event_defaults():
    ev = STTStreamEvent(type="final", text="नमस्ते")
    assert ev.type == "final"
    assert ev.text == "नमस्ते"
    assert ev.confidence == 1.0
    assert ev.language is None


def test_session_is_abstract():
    assert inspect.isabstract(ISTTStreamSession)
    for name in ("send", "events", "aclose"):
        assert name in ISTTStreamSession.__abstractmethods__


def test_provider_is_abstract():
    assert inspect.isabstract(IStreamingSTTProvider)
    assert "open_stream" in IStreamingSTTProvider.__abstractmethods__
