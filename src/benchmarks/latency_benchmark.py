"""End-to-end latency benchmark.

Drives ``PipelineEngine`` over scripted utterances for one provider combination
``(stt, llm, tts)``, captures per-stage timing from ``TurnMetrics``, and emits
per-stage ``LatencyStats``.

A combinatorial helper, ``run_latency_matrix``, takes a list of provider
combos and yields a list of ``LatencyRunResult`` for each. Caller assembles
the recommendation matrix from the results.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Optional

from src.benchmarks.metrics import LatencyStats, latency_stats
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage
from src.interfaces.stt import ISTTProvider, STTConfig
from src.interfaces.tts import ITTSProvider, TTSConfig
from src.pipeline.engine import PipelineConfig, PipelineEngine


@dataclass
class LatencyRunResult:
    combo: dict[str, str]    # {"stt": "sarvam", "llm": "groq", "tts": "sarvam"}
    sample_count: int
    stt: LatencyStats
    llm_ttft: LatencyStats
    llm_total: LatencyStats
    tts_first_chunk: LatencyStats
    tts_total: LatencyStats
    end_to_end: LatencyStats


@dataclass
class LatencyMatrixResult:
    results: list[LatencyRunResult] = field(default_factory=list)


async def run_latency_benchmark(
    combo: dict[str, str],
    stt: ISTTProvider,
    llm: ILLMProvider,
    tts: ITTSProvider,
    *,
    audio_samples: list[bytes],
    history: Optional[list[LLMMessage]] = None,
    config: Optional[PipelineConfig] = None,
) -> LatencyRunResult:
    """Run one provider combo across ``audio_samples``, return aggregated stats."""
    cfg = config or PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig())
    engine = PipelineEngine(stt, llm, tts, cfg)
    history = history or []

    stt_samples: list[float] = []
    llm_ttft: list[float] = []
    llm_total: list[float] = []
    tts_first: list[float] = []
    tts_total_arr: list[float] = []
    e2e: list[float] = []

    async def sink(_: bytes) -> None:
        return None

    for audio in audio_samples:
        t0 = time.perf_counter()
        result = await engine.run_turn(audio, history, sink)
        dt = (time.perf_counter() - t0) * 1000.0
        e2e.append(dt)
        m = result.metrics
        stt_samples.append(m.stt_latency_ms)
        llm_ttft.append(m.llm_ttft_ms)
        llm_total.append(m.llm_total_ms)
        tts_first.append(m.tts_first_chunk_ms)
        tts_total_arr.append(m.tts_total_ms)

    return LatencyRunResult(
        combo=dict(combo),
        sample_count=len(audio_samples),
        stt=latency_stats(stt_samples),
        llm_ttft=latency_stats(llm_ttft),
        llm_total=latency_stats(llm_total),
        tts_first_chunk=latency_stats(tts_first),
        tts_total=latency_stats(tts_total_arr),
        end_to_end=latency_stats(e2e),
    )


# Factory: lazy provider creators so we can iterate combos without holding
# clients in memory simultaneously. Tests inject simple lambdas.
ProviderFactory = Callable[[str], object]


async def run_latency_matrix(
    combos: list[dict[str, str]],
    *,
    make_stt: Callable[[str], ISTTProvider],
    make_llm: Callable[[str], ILLMProvider],
    make_tts: Callable[[str], ITTSProvider],
    audio_samples: list[bytes],
    history: Optional[list[LLMMessage]] = None,
    config: Optional[PipelineConfig] = None,
) -> LatencyMatrixResult:
    """Run every combo in sequence; concurrency is intentional 1 so latency
    measurements aren't contaminated by parallel provider load."""
    out = LatencyMatrixResult()
    for combo in combos:
        stt = make_stt(combo["stt"])
        llm = make_llm(combo["llm"])
        tts = make_tts(combo["tts"])
        r = await run_latency_benchmark(
            combo, stt, llm, tts,
            audio_samples=audio_samples,
            history=history,
            config=config,
        )
        out.results.append(r)
    return out
