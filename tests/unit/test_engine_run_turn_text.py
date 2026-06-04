from __future__ import annotations

import asyncio

import pytest

from src.interfaces.llm import LLMConfig, LLMMessage
from src.interfaces.stt import STTConfig
from src.interfaces.tts import TTSConfig, TTSResult
from src.pipeline.engine import PipelineConfig, PipelineEngine


class _FakeLLM:
    async def generate(self, messages, config):  # pragma: no cover - unused
        raise NotImplementedError

    async def generate_stream(self, messages, config):
        for tok in ['{"response_text": "', "नमस्ते जी।", '", "action": "continue"}']:
            yield tok


class _FakeTTS:
    async def synthesize(self, text, config):
        return TTSResult(audio=b"\x00\x00" * 80, duration_ms=10.0, sample_rate=16000)

    async def synthesize_stream(self, text_stream, config):  # pragma: no cover
        if False:
            yield b""


class _FakeSTT:
    async def transcribe(self, audio, config):  # pragma: no cover - unused here
        raise NotImplementedError

    async def transcribe_stream(self, audio_stream, config):  # pragma: no cover
        if False:
            yield None


def _engine():
    cfg = PipelineConfig(
        stt=STTConfig(language="hi-IN"),
        llm=LLMConfig(response_format="json", max_tokens=256),
        tts=TTSConfig(language="hi-IN", sample_rate=16000),
    )
    return PipelineEngine(_FakeSTT(), _FakeLLM(), _FakeTTS(), cfg)


@pytest.mark.asyncio
async def test_run_turn_text_skips_stt_and_speaks_response():
    engine = _engine()
    sink_calls = []

    async def sink(audio: bytes):
        sink_calls.append(audio)

    result = await engine.run_turn_text(
        "और कुछ benefits हैं?",
        history=[LLMMessage(role="system", content="be Anaaya")],
        audio_sink=sink,
    )
    assert result.user_text == "और कुछ benefits हैं?"
    assert result.metrics.stt_latency_ms == 0
    assert '"response_text"' in result.agent_text
    assert sink_calls
    assert "नमस्ते जी।" in "".join(result.sentences_spoken)


@pytest.mark.asyncio
async def test_run_turn_text_cancel_stops_before_audio():
    engine = _engine()
    sink_calls = []

    async def sink(audio: bytes):
        sink_calls.append(audio)

    cancel = asyncio.Event()
    cancel.set()  # pre-cancelled: the token loop breaks before processing/audio

    result = await engine.run_turn_text(
        "और कुछ benefits हैं?",
        history=[],
        audio_sink=sink,
        cancel_event=cancel,
    )
    assert result.cancelled is True
    assert sink_calls == []            # no audio synthesized/sent
    assert result.audio_bytes_sent == 0
    assert result.sentences_spoken == []
