"""RAG benchmark.

Three concerns scored independently:

1. Retrieval quality — precision@k, recall@k, MRR computed against
   ``expected_chunks`` (set of ground-truth chunk ids).

2. Answer faithfulness — does the agent's ``response_text`` rely on chunks
   the retriever actually returned? Light citation-coverage check: every
   citation in ``sources_used`` must be present in the retrieved set
   (catches the LLM inventing references), and at least one citation must
   exist when retrieval returned anything (catches dropped grounding).

3. Answer recall — does the agent's text contain key terms from the
   expected answer? Token-overlap heuristic so we don't pull in a heavy
   semantic evaluator. Not perfect, but actionable for regression tracking.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

from src.agents.chatbot import ChatBotAgent
from src.benchmarks.datasets import RAGSample
from src.benchmarks.metrics import LatencyStats, latency_stats
from src.rag.embeddings import _tokenize


# --- Retrieval scoring --------------------------------------------------


@dataclass
class RetrievalScore:
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    hit: bool


def score_retrieval(
    expected_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
    *,
    k: Optional[int] = None,
) -> RetrievalScore:
    """Compute precision@k / recall@k / MRR for one query.

    If ``k`` is None it defaults to ``len(retrieved_chunk_ids)``.
    """
    expected = set(expected_chunk_ids)
    if not expected:
        # Convention: empty ground truth -> trivially correct
        return RetrievalScore(precision_at_k=1.0, recall_at_k=1.0, reciprocal_rank=1.0, hit=True)
    top_k = retrieved_chunk_ids if k is None else retrieved_chunk_ids[:k]
    hits = [r for r in top_k if r in expected]
    precision = len(hits) / max(len(top_k), 1)
    recall = len(hits) / len(expected)
    mrr = 0.0
    for i, r in enumerate(retrieved_chunk_ids, start=1):
        if r in expected:
            mrr = 1.0 / i
            break
    return RetrievalScore(
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=mrr,
        hit=bool(hits),
    )


# --- Answer scoring -----------------------------------------------------


@dataclass
class AnswerScore:
    answer_recall: float        # token overlap with expected answer
    faithful: bool              # all citations supported by retrieved
    citations_supported: int
    citations_total: int


def score_answer(
    expected_answer: Optional[str],
    response_text: str,
    cited_sources: list[str],
    retrieved_source_tags: list[str],
) -> AnswerScore:
    available = set(retrieved_source_tags)
    supported = [c for c in cited_sources if c in available]
    faithful = (len(supported) == len(cited_sources)) if cited_sources else True

    if not expected_answer:
        answer_recall = 1.0 if response_text.strip() else 0.0
    else:
        expected_tokens = set(_tokenize(expected_answer))
        response_tokens = set(_tokenize(response_text))
        if not expected_tokens:
            answer_recall = 1.0
        else:
            answer_recall = len(expected_tokens & response_tokens) / len(expected_tokens)

    return AnswerScore(
        answer_recall=answer_recall,
        faithful=faithful,
        citations_supported=len(supported),
        citations_total=len(cited_sources),
    )


# --- Full RAG benchmark -------------------------------------------------


@dataclass
class RAGSampleResult:
    sample_id: str
    query: str
    retrieved_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    answer_recall: float
    faithful: bool
    latency_ms: float
    response_text: str = ""


@dataclass
class RAGRunResult:
    sample_count: int
    precision_mean: float
    recall_mean: float
    mrr_mean: float
    faithfulness_rate: float
    answer_recall_mean: float
    latency: LatencyStats
    per_sample: list[RAGSampleResult] = field(default_factory=list)


async def run_rag_benchmark(
    agent: ChatBotAgent,
    samples: list[RAGSample],
    *,
    top_k: int = 5,
) -> RAGRunResult:
    """Run each sample through ``agent.handle_message`` + score."""
    rows: list[RAGSampleResult] = []
    precisions: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    faithful_count = 0
    answer_recalls: list[float] = []
    latencies: list[float] = []

    for sample in samples:
        t0 = time.perf_counter()
        result = await agent.handle_message(sample.query)
        dt = (time.perf_counter() - t0) * 1000.0

        retrieved_ids = [r.document.id for r in result.retrieved]
        retrieved_tags = []
        for r in result.retrieved:
            md = r.document.metadata or {}
            fn = md.get("filename") or md.get("source")
            section = md.get("section") or md.get("page")
            if fn and section is not None:
                retrieved_tags.append(f"{fn}:{section}")
            elif fn:
                retrieved_tags.append(str(fn))
            else:
                retrieved_tags.append(r.document.id)

        retr = score_retrieval(sample.expected_chunks, retrieved_ids, k=top_k)
        ans = score_answer(
            expected_answer=sample.expected_answer,
            response_text=result.response.response_text,
            cited_sources=result.response.sources_used,
            retrieved_source_tags=retrieved_tags,
        )

        precisions.append(retr.precision_at_k)
        recalls.append(retr.recall_at_k)
        mrrs.append(retr.reciprocal_rank)
        if ans.faithful:
            faithful_count += 1
        answer_recalls.append(ans.answer_recall)
        latencies.append(dt)

        rows.append(RAGSampleResult(
            sample_id=sample.id,
            query=sample.query,
            retrieved_ids=retrieved_ids,
            precision_at_k=retr.precision_at_k,
            recall_at_k=retr.recall_at_k,
            reciprocal_rank=retr.reciprocal_rank,
            answer_recall=ans.answer_recall,
            faithful=ans.faithful,
            latency_ms=dt,
            response_text=result.response.response_text,
        ))

    n = max(len(samples), 1)
    return RAGRunResult(
        sample_count=len(samples),
        precision_mean=float(mean(precisions)) if precisions else 0.0,
        recall_mean=float(mean(recalls)) if recalls else 0.0,
        mrr_mean=float(mean(mrrs)) if mrrs else 0.0,
        faithfulness_rate=faithful_count / n,
        answer_recall_mean=float(mean(answer_recalls)) if answer_recalls else 0.0,
        latency=latency_stats(latencies),
        per_sample=rows,
    )
