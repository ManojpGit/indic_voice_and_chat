"""Top-level benchmark suite runner.

Persists each completed benchmark to the ``benchmark_runs`` table so the
``/benchmarks/*`` API endpoints can query historical results. The actual
DB write is optional (sessionmaker may be None for in-process / smoke runs).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from src.benchmarks.export import to_jsonable
from src.models.benchmark import BenchmarkRun
from src.benchmarks.latency_benchmark import LatencyMatrixResult
from src.benchmarks.rag_benchmark import RAGRunResult
from src.benchmarks.stt_benchmark import STTRunResult
from src.benchmarks.task_benchmark import TaskRunResult
from src.benchmarks.tts_benchmark import TTSRunResult

log = logging.getLogger(__name__)


@dataclass
class SuiteResults:
    stt: list[STTRunResult] = field(default_factory=list)
    tts: list[TTSRunResult] = field(default_factory=list)
    latency: Optional[LatencyMatrixResult] = None
    rag: Optional[RAGRunResult] = None
    task: Optional[TaskRunResult] = None


@dataclass
class SuiteRecord:
    id: str
    name: str
    description: str
    pipeline_config: dict[str, Any]
    language: str
    dataset: str
    results: dict[str, Any]
    created_at: datetime


class SuiteRunner:
    def __init__(self, sessionmaker=None) -> None:
        # ``sessionmaker`` matches ``async_sessionmaker[AsyncSession]`` from
        # ``src.models.database`` but we keep it untyped so tests can pass
        # None and skip DB writes.
        self._sm = sessionmaker
        self._in_memory: list[SuiteRecord] = []

    @property
    def records(self) -> list[SuiteRecord]:
        return list(self._in_memory)

    async def record(
        self,
        *,
        name: str,
        description: str,
        pipeline_config: dict[str, Any],
        language: str,
        dataset: str,
        results: SuiteResults,
    ) -> SuiteRecord:
        payload = {
            "stt": [to_jsonable(r) for r in results.stt],
            "tts": [to_jsonable(r) for r in results.tts],
            "latency": to_jsonable(results.latency) if results.latency else None,
            "rag": to_jsonable(results.rag) if results.rag else None,
            "task": to_jsonable(results.task) if results.task else None,
        }
        record = SuiteRecord(
            id=f"br_{uuid.uuid4().hex[:12]}",
            name=name,
            description=description,
            pipeline_config=pipeline_config,
            language=language,
            dataset=dataset,
            results=payload,
            created_at=datetime.utcnow(),
        )
        self._in_memory.append(record)

        if self._sm is not None:
            try:
                async with self._sm() as session:
                    row = BenchmarkRun(
                        name=record.name,
                        description=record.description,
                        pipeline_config=record.pipeline_config,
                        language=record.language,
                        dataset=record.dataset,
                        results=record.results,
                    )
                    session.add(row)
                    await session.commit()
            except Exception:  # noqa: BLE001
                log.exception("failed to persist benchmark run; kept in-memory")
        return record
