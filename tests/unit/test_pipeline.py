from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult
from src.interfaces.stt import ISTTProvider, STTConfig, STTResult
from src.interfaces.tts import ITTSProvider, TTSConfig, TTSResult
from src.pipeline.engine import PipelineConfig, PipelineEngine


# --- Fakes ---------------------------------------------------------------


class FakeSTT(ISTTProvider):
    def __init__(self, text: str = "hello", language: str = "en", confidence: float = 0.9) -> None:
        self._text = text
        self._language = language
        self._confidence = confidence

    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        return STTResult(
            text=self._text,
            confidence=self._confidence,
            language=self._language,
            raw_response={},
        )

    async def transcribe_stream(self, audio_stream, config):
        if False:
            yield  # pragma: no cover

    def get_supported_languages(self):
        return ["en", "hi"]


class FakeLLM(ILLMProvider):
    def __init__(self, tokens: list[str], delay_per_token: float = 0.0) -> None:
        self._tokens = tokens
        self._delay = delay_per_token
        self.last_messages: list[LLMMessage] | None = None

    async def generate(self, messages, config):
        self.last_messages = messages
        return LLMResult(text="".join(self._tokens), finish_reason="stop")

    async def generate_stream(self, messages, config):
        self.last_messages = messages
        for tok in self._tokens:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield tok


class FakeTTS(ITTSProvider):
    def __init__(self) -> None:
        self.synthesized: list[str] = []

    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        self.synthesized.append(text)
        return TTSResult(
            audio=text.encode("utf-8"),  # cheap & deterministic for tests
            duration_ms=10.0,
            sample_rate=config.sample_rate,
        )

    async def synthesize_stream(self, text_stream, config):
        if False:
            yield  # pragma: no cover

    def get_available_voices(self, language: str):
        return []


def _config() -> PipelineConfig:
    return PipelineConfig(
        stt=STTConfig(language="en"),
        llm=LLMConfig(),
        tts=TTSConfig(language="en"),
    )


# --- Tests ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turn_full_happy_path() -> None:
    stt = FakeSTT(text="Aap kaise hain?", language="hi", confidence=0.85)
    llm = FakeLLM(tokens=['{"response_text": "Theek hoon. ', 'Aapka kya haal? ', '", "action": "continue"}'])
    tts = FakeTTS()
    engine = PipelineEngine(stt, llm, tts, _config())

    sink_buffer: list[bytes] = []

    async def sink(b: bytes) -> None:
        sink_buffer.append(b)

    result = await engine.run_turn(
        captured_audio=b"\x00\x00",
        history=[LLMMessage(role="system", content="be polite")],
        audio_sink=sink,
    )

    assert result.user_text == "Aap kaise hain?"
    assert result.user_language == "hi"
    assert result.user_confidence == 0.85
    assert result.cancelled is False
    assert result.audio_bytes_sent > 0
    # The LLM history was extended with the user turn (not mutating caller's list)
    assert llm.last_messages is not None
    assert llm.last_messages[-1].content == "Aap kaise hain?"
    # Streaming TTS got at least the first sentence
    assert any("Theek hoon" in s for s in tts.synthesized)


@pytest.mark.asyncio
async def test_run_turn_does_not_mutate_history() -> None:
    history = [LLMMessage(role="system", content="hello")]
    engine = PipelineEngine(
        FakeSTT(), FakeLLM(tokens=['{"x":1}']), FakeTTS(), _config()
    )
    await engine.run_turn(b"\x00", history, audio_sink=_drop_sink)
    # Caller's list is untouched
    assert len(history) == 1
    assert history[0].role == "system"


@pytest.mark.asyncio
async def test_run_turn_empty_stt_returns_early() -> None:
    stt = FakeSTT(text="", confidence=0.0)
    llm = FakeLLM(tokens=["should not be called"])
    tts = FakeTTS()
    engine = PipelineEngine(stt, llm, tts, _config())

    result = await engine.run_turn(b"\x00", history=[], audio_sink=_drop_sink)
    assert result.user_text == ""
    assert result.agent_text == ""
    assert result.audio_bytes_sent == 0
    assert llm.last_messages is None  # LLM never called
    assert tts.synthesized == []


@pytest.mark.asyncio
async def test_run_turn_streams_per_sentence() -> None:
    # 3 sentences: each should be synthesized separately as it completes.
    stt = FakeSTT(text="hi")
    llm = FakeLLM(tokens=["First. ", "Second sentence. ", "Third one."])
    tts = FakeTTS()
    engine = PipelineEngine(stt, llm, tts, _config())

    await engine.run_turn(b"\x00", [], audio_sink=_drop_sink)

    # All three sentences should have been synthesized as discrete units.
    assert len(tts.synthesized) >= 3
    joined = " | ".join(tts.synthesized)
    assert "First." in joined
    assert "Second sentence." in joined
    assert "Third one." in joined


@pytest.mark.asyncio
async def test_run_turn_cancellation_drops_remaining_audio() -> None:
    stt = FakeSTT(text="hi")
    llm = FakeLLM(
        tokens=["First. ", "Second. ", "Third."],
        delay_per_token=0.05,  # slow enough to cancel mid-stream
    )
    tts = FakeTTS()
    engine = PipelineEngine(stt, llm, tts, _config())

    cancel_event = asyncio.Event()

    async def cancelling_sink(b: bytes) -> None:
        # Cancel after the very first audio chunk.
        cancel_event.set()

    result = await engine.run_turn(
        b"\x00", [], audio_sink=cancelling_sink, cancel_event=cancel_event
    )

    assert result.cancelled is True
    # We should have synthesized fewer than the total 3 sentences
    assert len(tts.synthesized) < 3 or result.audio_bytes_sent < sum(
        len(s.encode()) for s in ("First.", "Second.", "Third.")
    )


@pytest.mark.asyncio
async def test_run_turn_metrics_populated() -> None:
    stt = FakeSTT(text="hi")
    llm = FakeLLM(tokens=["Hello there."], delay_per_token=0.005)
    tts = FakeTTS()
    engine = PipelineEngine(stt, llm, tts, _config())

    result = await engine.run_turn(b"\x00", [], audio_sink=_drop_sink)
    m = result.metrics
    assert m.stt_latency_ms >= 0
    assert m.llm_total_ms >= 0
    assert m.total_latency_ms >= m.stt_latency_ms


async def _drop_sink(b: bytes) -> None:
    pass
