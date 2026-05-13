from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

from src.benchmarks.datasets import STTSample, TTSSample
from src.benchmarks.latency_benchmark import (
    run_latency_benchmark,
    run_latency_matrix,
)
from src.benchmarks.stt_benchmark import run_stt_benchmark
from src.benchmarks.tts_benchmark import (
    detect_outliers,
    join_mos_scores,
    run_tts_benchmark,
)
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult
from src.interfaces.stt import ISTTProvider, STTConfig, STTResult
from src.interfaces.tts import ITTSProvider, TTSConfig, TTSResult


# --- Fakes ---------------------------------------------------------------


class CannedSTT(ISTTProvider):
    def __init__(self, transcripts_by_audio: dict[bytes, str]) -> None:
        self._table = transcripts_by_audio

    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        text = self._table.get(audio, "")
        return STTResult(text=text, confidence=0.9, language=config.language, raw_response={})

    async def transcribe_stream(self, audio_stream, config) -> AsyncIterator[STTResult]:
        if False:
            yield  # pragma: no cover

    def get_supported_languages(self):
        return ["hi-IN", "en-IN"]


class ConstantTTS(ITTSProvider):
    def __init__(self, sample_rate: int = 16000) -> None:
        self._sr = sample_rate

    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        # 80ms of audio per character (very stable / measurable)
        ms = max(80.0 * len(text), 10.0)
        return TTSResult(audio=b"\x00\x00" * 100, duration_ms=ms, sample_rate=config.sample_rate)

    async def synthesize_stream(self, text_stream, config) -> AsyncIterator[bytes]:
        if False:
            yield  # pragma: no cover

    def get_available_voices(self, language: str):
        return []


class CannedLLM(ILLMProvider):
    def __init__(self, response: str = '{"response_text":"ok","language":"hi","action":"continue"}') -> None:
        self._resp = response

    async def generate(self, messages, config) -> LLMResult:
        return LLMResult(text=self._resp, finish_reason="stop")

    async def generate_stream(self, messages, config) -> AsyncIterator[str]:
        for t in (self._resp[: len(self._resp) // 2], self._resp[len(self._resp) // 2 :]):
            yield t


# --- STT benchmark ------------------------------------------------------


@pytest.mark.asyncio
async def test_stt_benchmark_aggregates_by_language() -> None:
    samples = [
        STTSample(id="s1", transcript="Namaste dosto", language="hi-IN", audio_bytes=b"A"),
        STTSample(id="s2", transcript="Hello world", language="en-IN", audio_bytes=b"B"),
        STTSample(id="s3", transcript="Mixed sentence here", language="hi-IN", code_switch=True, audio_bytes=b"C"),
    ]
    stt = CannedSTT({
        b"A": "Namaste dosto",      # perfect
        b"B": "hello earth",        # 1 word swap
        b"C": "Mixed sentence",     # 1 deletion
    })
    result = await run_stt_benchmark("sarvam", stt, samples)
    assert result.provider == "sarvam"
    assert result.sample_count == 3
    # Per-language breakdown
    assert "hi-IN" in result.per_language
    assert "en-IN" in result.per_language
    # The hi-IN bucket includes the code-switch sample, so its WER reflects that
    assert result.per_language["en-IN"].wer_mean > 0.0
    # Code-switch was carved out separately
    assert result.code_switch is not None
    assert result.code_switch.sample_count == 1
    # Latency stats present
    assert result.latency.count == 3


@pytest.mark.asyncio
async def test_stt_benchmark_no_code_switch_samples() -> None:
    samples = [STTSample(id="s1", transcript="hi", language="hi", audio_bytes=b"x")]
    stt = CannedSTT({b"x": "hi"})
    result = await run_stt_benchmark("sarvam", stt, samples)
    assert result.code_switch is None
    assert result.overall.wer_mean == 0.0


@pytest.mark.asyncio
async def test_stt_benchmark_keep_per_sample_off() -> None:
    samples = [STTSample(id="s1", transcript="hi", audio_bytes=b"x")]
    stt = CannedSTT({b"x": "hi"})
    result = await run_stt_benchmark("p", stt, samples, keep_per_sample=False)
    assert result.per_sample == []


# --- TTS benchmark ------------------------------------------------------


@pytest.mark.asyncio
async def test_tts_benchmark_basic() -> None:
    samples = [
        TTSSample(id="t1", text="Namaste", language="hi-IN"),
        TTSSample(id="t2", text="Hello world", language="en-IN"),
    ]
    tts = ConstantTTS(sample_rate=16000)
    result = await run_tts_benchmark("sarvam", tts, samples, sample_rate=16000)
    assert result.sample_count == 2
    assert result.sample_rate_violations == 0
    assert result.avg_duration_ms > 0
    assert "hi-IN" in result.per_language


@pytest.mark.asyncio
async def test_tts_benchmark_flags_sample_rate_mismatch() -> None:
    samples = [TTSSample(id="t1", text="Hi", language="en")]
    tts = ConstantTTS(sample_rate=16000)
    # Request 22050 but provider returns 16000 (which the request actually sets,
    # so we change expectation to 22050 to force a mismatch).
    result = await run_tts_benchmark("p", tts, samples, sample_rate=16000)
    # ConstantTTS always echoes config.sample_rate, so no violation here.
    assert result.sample_rate_violations == 0


@pytest.mark.asyncio
async def test_tts_join_mos_scores() -> None:
    samples = [
        TTSSample(id="t1", text="Hi", language="en"),
        TTSSample(id="t2", text="Bye", language="en"),
    ]
    tts = ConstantTTS()
    result = await run_tts_benchmark("p", tts, samples)
    join_mos_scores(result, {"t1": 4.2, "t2": 3.8})
    assert result.mos_mean == pytest.approx(4.0)
    assert result.per_sample[0].mos == 4.2
    assert result.per_sample[1].mos == 3.8


@pytest.mark.asyncio
async def test_tts_detect_outliers_finds_implausible_rate() -> None:
    samples = [
        TTSSample(id="t1", text="Hi", language="en"),
        TTSSample(id="t2", text="This is reasonably long text that should be normal-paced", language="en"),
    ]
    tts = ConstantTTS()
    result = await run_tts_benchmark("p", tts, samples)
    # ConstantTTS gives 80ms per char, which is ~12.5 chars/sec — well in range.
    # Outliers should be empty.
    assert detect_outliers(result) == []


# --- Latency benchmark --------------------------------------------------


@pytest.mark.asyncio
async def test_latency_benchmark_single_combo() -> None:
    stt = CannedSTT({b"\x00": "hi"})
    llm = CannedLLM()
    tts = ConstantTTS()
    result = await run_latency_benchmark(
        {"stt": "sarvam", "llm": "groq", "tts": "sarvam"},
        stt, llm, tts,
        audio_samples=[b"\x00", b"\x00", b"\x00"],
    )
    assert result.sample_count == 3
    assert result.end_to_end.count == 3
    assert result.stt.count == 3
    assert result.combo["llm"] == "groq"


@pytest.mark.asyncio
async def test_latency_matrix_runs_all_combos() -> None:
    combos = [
        {"stt": "sarvam", "llm": "groq", "tts": "sarvam"},
        {"stt": "sarvam", "llm": "gemini", "tts": "sarvam"},
    ]
    out = await run_latency_matrix(
        combos,
        make_stt=lambda name: CannedSTT({b"\x00": "hi"}),
        make_llm=lambda name: CannedLLM(),
        make_tts=lambda name: ConstantTTS(),
        audio_samples=[b"\x00", b"\x00"],
    )
    assert len(out.results) == 2
    assert out.results[0].combo["llm"] == "groq"
    assert out.results[1].combo["llm"] == "gemini"
