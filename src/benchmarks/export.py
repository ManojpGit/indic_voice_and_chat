"""CSV / JSON export for benchmark results.

Each benchmark result has its own flat shape, so we ship a small registry
of writers rather than a single generic flattener. All writers accept
``Path`` or any file-like object with ``.write()``.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from src.benchmarks.rag_benchmark import RAGRunResult
from src.benchmarks.stt_benchmark import STTRunResult
from src.benchmarks.task_benchmark import TaskRunResult
from src.benchmarks.tts_benchmark import TTSRunResult
from src.benchmarks.latency_benchmark import LatencyMatrixResult


PathLike = Union[Path, str]


# --- Helpers -----------------------------------------------------------


def _open_writer(target: Union[PathLike, io.TextIOBase], header: list[str]) -> tuple[Any, Any, bool]:
    """Return (csv_writer, closer, should_close)."""
    if isinstance(target, (str, Path)):
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        writer.writerow(header)
        return writer, fh, True
    writer = csv.writer(target)
    writer.writerow(header)
    return writer, None, False


# --- Per-benchmark writers ---------------------------------------------


def write_stt_csv(target: Union[PathLike, io.TextIOBase], result: STTRunResult) -> int:
    header = ["sample_id", "language", "code_switch", "reference", "hypothesis", "wer", "cer", "confidence", "stt_latency_ms"]
    writer, fh, should_close = _open_writer(target, header)
    n = 0
    try:
        for row in result.per_sample:
            writer.writerow([
                row.sample_id,
                row.language or "",
                "1" if row.code_switch else "0",
                row.reference,
                row.hypothesis,
                f"{row.wer:.4f}",
                f"{row.cer:.4f}",
                f"{row.confidence:.4f}",
                f"{row.stt_latency_ms:.2f}",
            ])
            n += 1
    finally:
        if should_close:
            fh.close()
    return n


def write_tts_csv(target: Union[PathLike, io.TextIOBase], result: TTSRunResult) -> int:
    header = [
        "sample_id", "language", "audio_bytes", "duration_ms", "sample_rate",
        "chars_per_second", "synth_latency_ms", "mos",
    ]
    writer, fh, should_close = _open_writer(target, header)
    n = 0
    try:
        for row in result.per_sample:
            writer.writerow([
                row.sample_id,
                row.language,
                row.audio_bytes,
                f"{row.duration_ms:.2f}",
                row.sample_rate,
                f"{row.chars_per_second:.2f}",
                f"{row.synth_latency_ms:.2f}",
                "" if row.mos is None else f"{row.mos:.2f}",
            ])
            n += 1
    finally:
        if should_close:
            fh.close()
    return n


def write_latency_csv(target: Union[PathLike, io.TextIOBase], result: LatencyMatrixResult) -> int:
    header = [
        "stt_provider", "llm_provider", "tts_provider", "samples",
        "stt_p95_ms", "llm_ttft_p95_ms", "llm_total_p95_ms",
        "tts_first_p95_ms", "tts_total_p95_ms", "e2e_p95_ms", "e2e_mean_ms",
    ]
    writer, fh, should_close = _open_writer(target, header)
    n = 0
    try:
        for r in result.results:
            writer.writerow([
                r.combo.get("stt", ""),
                r.combo.get("llm", ""),
                r.combo.get("tts", ""),
                r.sample_count,
                f"{r.stt.p95_ms:.2f}",
                f"{r.llm_ttft.p95_ms:.2f}",
                f"{r.llm_total.p95_ms:.2f}",
                f"{r.tts_first_chunk.p95_ms:.2f}",
                f"{r.tts_total.p95_ms:.2f}",
                f"{r.end_to_end.p95_ms:.2f}",
                f"{r.end_to_end.mean_ms:.2f}",
            ])
            n += 1
    finally:
        if should_close:
            fh.close()
    return n


def write_rag_csv(target: Union[PathLike, io.TextIOBase], result: RAGRunResult) -> int:
    header = [
        "sample_id", "query", "retrieved_ids", "precision_at_k", "recall_at_k",
        "reciprocal_rank", "answer_recall", "faithful", "latency_ms",
    ]
    writer, fh, should_close = _open_writer(target, header)
    n = 0
    try:
        for row in result.per_sample:
            writer.writerow([
                row.sample_id, row.query,
                ";".join(row.retrieved_ids),
                f"{row.precision_at_k:.4f}",
                f"{row.recall_at_k:.4f}",
                f"{row.reciprocal_rank:.4f}",
                f"{row.answer_recall:.4f}",
                "1" if row.faithful else "0",
                f"{row.latency_ms:.2f}",
            ])
            n += 1
    finally:
        if should_close:
            fh.close()
    return n


def write_task_csv(target: Union[PathLike, io.TextIOBase], result: TaskRunResult) -> int:
    header = [
        "scenario_id", "completed", "disposition_match", "expected_disposition",
        "actual_action", "slot_fill_rate", "slot_value_match_rate", "turn_count",
        "duration_ms", "final_state",
    ]
    writer, fh, should_close = _open_writer(target, header)
    n = 0
    try:
        for row in result.per_scenario:
            writer.writerow([
                row.scenario_id,
                "1" if row.completed else "0",
                "1" if row.disposition_match else "0",
                row.expected_disposition,
                row.actual_action,
                f"{row.required_slot_fill_rate:.4f}",
                f"{row.slot_value_match_rate:.4f}",
                row.turn_count,
                f"{row.duration_ms:.2f}",
                row.final_state,
            ])
            n += 1
    finally:
        if should_close:
            fh.close()
    return n


# --- JSON dump ---------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    """Convert nested dataclasses to JSON-friendly dicts."""
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


def write_json(path: PathLike, payload: Any) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False)
    p.write_text(data, encoding="utf-8")
    return len(data)


# --- Provider recommendation matrix ------------------------------------


def recommend_providers(
    stt_results: Iterable[STTRunResult],
    tts_results: Iterable[TTSRunResult],
    latency: Optional[LatencyMatrixResult] = None,
    *,
    wer_weight: float = 0.5,
    latency_weight: float = 0.5,
) -> dict[str, Any]:
    """Pick best STT + best LLM/TTS combo by combining quality + latency.

    For STT: lower WER is better. We score = wer_mean (lower wins).
    For TTS we leave the choice to MOS if present, otherwise tie-break by
    latency p95.
    For latency: we pick the (stt, llm, tts) combo with the lowest e2e p95.
    """
    stt_pick = None
    stt_ranking: list[tuple[str, float]] = []
    for r in stt_results:
        score = wer_weight * r.overall.wer_mean + latency_weight * (r.latency.p95_ms / 1000.0)
        stt_ranking.append((r.provider, score))
    if stt_ranking:
        stt_ranking.sort(key=lambda x: x[1])
        stt_pick = stt_ranking[0][0]

    tts_pick = None
    tts_ranking: list[tuple[str, float]] = []
    for r in tts_results:
        # Higher MOS = better -> negate; lower latency = better.
        mos_term = -(r.mos_mean or 0.0)
        score = mos_term + latency_weight * (r.latency.p95_ms / 1000.0)
        tts_ranking.append((r.provider, score))
    if tts_ranking:
        tts_ranking.sort(key=lambda x: x[1])
        tts_pick = tts_ranking[0][0]

    combo_pick = None
    combo_ranking: list[tuple[dict[str, str], float]] = []
    if latency is not None:
        for r in latency.results:
            combo_ranking.append((dict(r.combo), r.end_to_end.p95_ms))
        combo_ranking.sort(key=lambda x: x[1])
        if combo_ranking:
            combo_pick = combo_ranking[0][0]

    return {
        "stt_recommendation": stt_pick,
        "stt_ranking": stt_ranking,
        "tts_recommendation": tts_pick,
        "tts_ranking": tts_ranking,
        "latency_combo_recommendation": combo_pick,
        "latency_combo_ranking": combo_ranking,
    }
