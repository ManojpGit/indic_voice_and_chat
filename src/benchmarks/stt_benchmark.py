"""STT benchmark runner.

Feeds each ``STTSample`` through an ``ISTTProvider``, scores the resulting
transcript against ground truth via WER + CER. Results are aggregated
overall, per-language, and (optionally) per code-switch subset so the
provider matrix shows where each STT actually struggles.

Per-sample latency is captured so the latency benchmark can re-use this
runner's output without re-running expensive STT calls.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from src.benchmarks.datasets import STTSample
from src.benchmarks.metrics import (
    AggregateAccuracy,
    aggregate_accuracy,
    latency_stats,
    LatencyStats,
)
from src.interfaces.stt import ISTTProvider, STTConfig


@dataclass
class STTSampleResult:
    sample_id: str
    reference: str
    hypothesis: str
    language: Optional[str]
    code_switch: bool
    confidence: float
    stt_latency_ms: float
    wer: float
    cer: float


@dataclass
class STTRunResult:
    provider: str
    sample_count: int
    overall: AggregateAccuracy
    per_language: dict[str, AggregateAccuracy] = field(default_factory=dict)
    code_switch: Optional[AggregateAccuracy] = None
    latency: LatencyStats = field(default_factory=lambda: latency_stats([]))
    per_sample: list[STTSampleResult] = field(default_factory=list)


async def run_stt_benchmark(
    provider_name: str,
    stt: ISTTProvider,
    samples: list[STTSample],
    *,
    audio_base_dir: Optional[str] = None,
    default_language: Optional[str] = None,
    keep_per_sample: bool = True,
) -> STTRunResult:
    """Score ``stt`` against ``samples``.

    Returns aggregated overall + per-language + code-switch stats plus the
    raw per-sample rows when ``keep_per_sample`` is True (default).
    """
    from pathlib import Path

    base_dir = Path(audio_base_dir) if audio_base_dir else None
    references: list[str] = []
    hypotheses: list[str] = []
    per_lang: dict[str, list[tuple[str, str]]] = defaultdict(list)
    code_switch_pairs: list[tuple[str, str]] = []
    latencies: list[float] = []
    rows: list[STTSampleResult] = []

    for sample in samples:
        try:
            audio = sample.resolve_audio(base_dir=base_dir)
        except (FileNotFoundError, ValueError) as e:
            # Treat missing audio as an empty hypothesis so accounting still works.
            audio = b""
            _missing_audio = str(e)  # noqa: F841 — kept for future logging
        cfg = STTConfig(language=sample.language or default_language)
        t0 = time.perf_counter()
        result = await stt.transcribe(audio, cfg)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        from src.benchmarks.metrics import word_error_rate, character_error_rate
        w = word_error_rate(sample.transcript, result.text)
        c = character_error_rate(sample.transcript, result.text)
        latencies.append(dt_ms)

        references.append(sample.transcript)
        hypotheses.append(result.text)
        lang_key = sample.language or "unknown"
        per_lang[lang_key].append((sample.transcript, result.text))
        if sample.code_switch:
            code_switch_pairs.append((sample.transcript, result.text))

        if keep_per_sample:
            rows.append(STTSampleResult(
                sample_id=sample.id,
                reference=sample.transcript,
                hypothesis=result.text,
                language=sample.language,
                code_switch=sample.code_switch,
                confidence=result.confidence,
                stt_latency_ms=dt_ms,
                wer=w.wer,
                cer=c.cer,
            ))

    overall = aggregate_accuracy(references, hypotheses)
    per_language = {
        lang: aggregate_accuracy([r for r, _ in pairs], [h for _, h in pairs])
        for lang, pairs in per_lang.items()
    }
    code_switch = None
    if code_switch_pairs:
        code_switch = aggregate_accuracy(
            [r for r, _ in code_switch_pairs],
            [h for _, h in code_switch_pairs],
        )

    return STTRunResult(
        provider=provider_name,
        sample_count=len(samples),
        overall=overall,
        per_language=per_language,
        code_switch=code_switch,
        latency=latency_stats(latencies),
        per_sample=rows,
    )
