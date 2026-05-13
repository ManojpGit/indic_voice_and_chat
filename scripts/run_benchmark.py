"""CLI to execute one slice of the benchmark suite.

Designed for the deferred-API path (no live keys):
    python scripts/run_benchmark.py stt --dataset data/stt.jsonl --out data/stt_results.csv

Sub-commands:
    stt        run STT benchmark against a JSONL dataset
    tts        run TTS benchmark against a JSONL dataset
    latency    run latency matrix across N replays of one audio sample
    rag        run RAG benchmark against a JSONL dataset

The CLI deliberately wires fake provider clients when ``--mock`` is given so
the suite is exercisable end-to-end without API keys. With real keys present
in the environment, the existing provider factories pick them up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.benchmarks.datasets import (
    load_rag_dataset,
    load_stt_dataset,
    load_tts_dataset,
)
from src.benchmarks.export import (
    write_latency_csv,
    write_rag_csv,
    write_stt_csv,
    write_tts_csv,
)
from src.benchmarks.stt_benchmark import run_stt_benchmark
from src.benchmarks.tts_benchmark import run_tts_benchmark


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vox-agent benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)

    p_stt = sub.add_parser("stt", help="Run STT benchmark")
    p_stt.add_argument("--dataset", required=True, type=Path)
    p_stt.add_argument("--out", required=True, type=Path)
    p_stt.add_argument("--provider", default="sarvam")
    p_stt.add_argument("--mock", action="store_true", help="Use a deterministic fake provider")

    p_tts = sub.add_parser("tts", help="Run TTS benchmark")
    p_tts.add_argument("--dataset", required=True, type=Path)
    p_tts.add_argument("--out", required=True, type=Path)
    p_tts.add_argument("--provider", default="sarvam")
    p_tts.add_argument("--mock", action="store_true")

    p_rag = sub.add_parser("rag", help="Run RAG benchmark")
    p_rag.add_argument("--dataset", required=True, type=Path)
    p_rag.add_argument("--out", required=True, type=Path)

    return parser


async def _run(args: argparse.Namespace) -> int:
    if args.command == "stt":
        samples = load_stt_dataset(args.dataset)
        if args.mock:
            stt = _MockSTT()
        else:
            from src.config import load_settings
            from src.providers import get_stt_provider
            stt = get_stt_provider({"provider": args.provider, **load_settings().pipeline.stt.model_dump()})
        result = await run_stt_benchmark(args.provider, stt, samples)
        n = write_stt_csv(args.out, result)
        print(f"wrote {n} rows to {args.out}; wer_mean={result.overall.wer_mean:.4f}")
        return 0

    if args.command == "tts":
        samples = load_tts_dataset(args.dataset)
        if args.mock:
            tts = _MockTTS()
        else:
            from src.config import load_settings
            from src.providers import get_tts_provider
            tts = get_tts_provider({"provider": args.provider, **load_settings().pipeline.tts.model_dump()})
        result = await run_tts_benchmark(args.provider, tts, samples)
        n = write_tts_csv(args.out, result)
        print(f"wrote {n} rows to {args.out}; avg_chars_per_second={result.avg_chars_per_second:.2f}")
        return 0

    if args.command == "rag":
        # RAG path requires a wired retriever + agent. The CLI form is a
        # thin compatibility shim that emits a clear message — full RAG runs
        # currently happen through ``tests/integration/test_benchmark_e2e.py``
        # or the API (Phase 6+).
        print(
            "rag CLI requires app bootstrap; use the API or the integration test for now",
            file=sys.stderr,
        )
        return 2

    return 1


# --- Mock providers for --mock mode -------------------------------------


class _MockSTT:
    async def transcribe(self, audio, config):
        from src.interfaces.stt import STTResult
        return STTResult(text="mock transcription", confidence=0.5, language=config.language)

    async def transcribe_stream(self, audio_stream, config):
        if False:
            yield  # pragma: no cover

    def get_supported_languages(self):
        return ["hi-IN"]


class _MockTTS:
    async def synthesize(self, text, config):
        from src.interfaces.tts import TTSResult
        return TTSResult(audio=b"\x00\x00" * 100, duration_ms=80.0 * len(text), sample_rate=config.sample_rate)

    async def synthesize_stream(self, text_stream, config):
        if False:
            yield  # pragma: no cover

    def get_available_voices(self, language):
        return []


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
