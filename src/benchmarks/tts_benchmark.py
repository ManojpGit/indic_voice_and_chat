"""TTS benchmark runner.

TTS quality is fundamentally subjective — MOS (Mean Opinion Score) is the
canonical metric and requires human raters. This module ships two layers:

1. Objective metrics that we *can* compute mechanically:
   - sample-rate consistency vs the configured request
   - audio length plausibility (rate-adjusted seconds per character)
   - synthesis latency
   - duration variance across samples

2. A MOS-aggregation helper that takes ratings from an external CSV (the
   human-eval results) keyed by sample id, joins them with the objective
   pass-through, and produces a unified ``TTSRunResult``.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

from src.benchmarks.datasets import TTSSample
from src.benchmarks.metrics import LatencyStats, latency_stats
from src.interfaces.tts import ITTSProvider, TTSConfig


@dataclass
class TTSSampleResult:
    sample_id: str
    text: str
    language: str
    audio_bytes: int
    duration_ms: float
    sample_rate: int
    chars_per_second: float
    synth_latency_ms: float
    mos: Optional[float] = None       # joined from human-eval CSV later


@dataclass
class TTSRunResult:
    provider: str
    sample_count: int
    avg_duration_ms: float
    avg_chars_per_second: float
    expected_sample_rate: int
    sample_rate_violations: int
    latency: LatencyStats
    mos_mean: Optional[float] = None
    per_language: dict[str, dict[str, float]] = field(default_factory=dict)
    per_sample: list[TTSSampleResult] = field(default_factory=list)


_MIN_CHARS_PER_SECOND = 5.0   # absurdly slow speech
_MAX_CHARS_PER_SECOND = 35.0  # absurdly fast speech


async def run_tts_benchmark(
    provider_name: str,
    tts: ITTSProvider,
    samples: list[TTSSample],
    *,
    sample_rate: int = 16000,
    keep_per_sample: bool = True,
) -> TTSRunResult:
    """Score ``tts`` over ``samples``. No human MOS — that joins later."""
    rows: list[TTSSampleResult] = []
    latencies: list[float] = []
    rate_violations = 0
    durations: list[float] = []
    chars_per_sec: list[float] = []
    per_lang: dict[str, list[float]] = defaultdict(list)

    for sample in samples:
        cfg = TTSConfig(
            language=sample.language,
            voice_id=sample.voice_id,
            sample_rate=sample_rate,
        )
        t0 = time.perf_counter()
        result = await tts.synthesize(sample.text, cfg)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        if result.sample_rate != sample_rate:
            rate_violations += 1
        duration_s = max(result.duration_ms / 1000.0, 1e-6)
        cps = len(sample.text) / duration_s
        durations.append(result.duration_ms)
        chars_per_sec.append(cps)
        per_lang[sample.language].append(cps)

        if keep_per_sample:
            rows.append(TTSSampleResult(
                sample_id=sample.id,
                text=sample.text,
                language=sample.language,
                audio_bytes=len(result.audio),
                duration_ms=result.duration_ms,
                sample_rate=result.sample_rate,
                chars_per_second=cps,
                synth_latency_ms=dt_ms,
            ))

    per_lang_summary = {
        lang: {
            "samples": len(cps_list),
            "avg_chars_per_second": float(mean(cps_list)) if cps_list else 0.0,
        }
        for lang, cps_list in per_lang.items()
    }

    return TTSRunResult(
        provider=provider_name,
        sample_count=len(samples),
        avg_duration_ms=float(mean(durations)) if durations else 0.0,
        avg_chars_per_second=float(mean(chars_per_sec)) if chars_per_sec else 0.0,
        expected_sample_rate=sample_rate,
        sample_rate_violations=rate_violations,
        latency=latency_stats(latencies),
        per_language=per_lang_summary,
        per_sample=rows,
    )


# --- MOS join ----------------------------------------------------------


def join_mos_scores(result: TTSRunResult, mos_by_sample_id: dict[str, float]) -> TTSRunResult:
    """Fold human MOS ratings into the per-sample rows + overall mean.

    Sample ids missing from ``mos_by_sample_id`` keep ``mos = None``.
    """
    rated: list[float] = []
    for row in result.per_sample:
        score = mos_by_sample_id.get(row.sample_id)
        if score is not None:
            row.mos = float(score)
            rated.append(float(score))
    if rated:
        result.mos_mean = float(mean(rated))
    return result


def detect_outliers(result: TTSRunResult) -> list[str]:
    """Return sample ids where ``chars_per_second`` is implausible.

    Useful for flagging clipped or stuck synthesis in test data review.
    """
    out: list[str] = []
    for row in result.per_sample:
        if row.chars_per_second < _MIN_CHARS_PER_SECOND or row.chars_per_second > _MAX_CHARS_PER_SECOND:
            out.append(row.sample_id)
    return out
