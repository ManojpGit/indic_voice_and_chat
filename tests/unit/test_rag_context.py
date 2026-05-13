from __future__ import annotations

from src.dialogue.response_parser import ChatBotResponse
from src.interfaces.vector_store import Document
from src.rag.context_builder import (
    GuardConfig,
    apply_hallucination_guard,
    build_rag_context,
)
from src.rag.retriever import RetrievedChunk


def _chunk(doc_id: str, content: str, **md) -> RetrievedChunk:
    return RetrievedChunk(
        document=Document(id=doc_id, content=content, metadata=md),
        score=0.9,
    )


# --- Context builder ---------------------------------------------------


def test_build_rag_context_empty_returns_marker() -> None:
    out = build_rag_context([])
    assert "no relevant sources" in out.text
    assert out.source_tags == []
    assert out.chunk_count == 0


def test_build_rag_context_uses_filename_section_tag() -> None:
    out = build_rag_context([
        _chunk("c1", "Plan B has 500GB.", filename="plans.pdf", page=2),
        _chunk("c2", "Plan A has 100GB.", filename="plans.pdf", page=1),
    ])
    assert "plans.pdf:2" in out.text
    assert "plans.pdf:1" in out.text
    assert out.source_tags == ["plans.pdf:2", "plans.pdf:1"]
    assert out.chunk_count == 2


def test_build_rag_context_falls_back_to_id_when_no_metadata() -> None:
    out = build_rag_context([_chunk("doc-42", "content")])
    assert "doc-42" in out.source_tags


def test_build_rag_context_truncates_at_max_chars() -> None:
    big = "X" * 1000
    chunks = [_chunk(f"c{i}", big, filename=f"f{i}.md") for i in range(10)]
    out = build_rag_context(chunks, max_chars=2500)
    # Should have included at most ~3 chunks before hitting the budget
    assert out.chunk_count <= 4
    assert len(out.text) <= 4500  # block headers add some overhead


# --- Hallucination guard -----------------------------------------------


def test_guard_passes_through_clean_response() -> None:
    rag = build_rag_context([_chunk("c1", "Plan B has 500GB.", filename="plans.pdf", page=2)])
    response = ChatBotResponse(
        response_text="Plan B has 500GB.",
        language="en",
        sources_used=["plans.pdf:2"],
        confidence="high",
        action="none",
    )
    out = apply_hallucination_guard(response, rag)
    assert out.response_text == "Plan B has 500GB."
    assert out.sources_used == ["plans.pdf:2"]
    assert out.confidence == "high"


def test_guard_strips_unsupported_citations_and_downgrades() -> None:
    rag = build_rag_context([_chunk("c1", "Plan B has 500GB.", filename="plans.pdf", page=2)])
    response = ChatBotResponse(
        response_text="Plan B has 500GB and supports 5G.",
        language="en",
        sources_used=["plans.pdf:2", "wireless.pdf:7"],  # second is invented
        confidence="high",
    )
    out = apply_hallucination_guard(response, rag)
    assert out.sources_used == ["plans.pdf:2"]
    assert out.confidence == "low"


def test_guard_no_retrieval_returns_fallback_in_english() -> None:
    rag = build_rag_context([])
    response = ChatBotResponse(
        response_text="Yes, the answer is 42.",
        language="en",
        confidence="high",
    )
    out = apply_hallucination_guard(response, rag)
    assert "not able to find" in out.response_text.lower()
    assert out.confidence == "low"
    assert out.sources_used == []


def test_guard_no_retrieval_returns_fallback_in_hindi() -> None:
    rag = build_rag_context([])
    response = ChatBotResponse(
        response_text="Plan B mein 500GB data hai.",
        language="hi",
        confidence="high",
    )
    out = apply_hallucination_guard(response, rag)
    assert "documentation mein nahi mil raha" in out.response_text.lower()
    assert out.confidence == "low"


def test_guard_no_retrieval_empty_response_unchanged() -> None:
    rag = build_rag_context([])
    response = ChatBotResponse(response_text="", language="en", confidence="medium")
    out = apply_hallucination_guard(response, rag)
    # No fallback substituted because there was nothing to override.
    assert out.response_text == ""
    assert out.confidence == "low"


def test_guard_does_not_mutate_input() -> None:
    rag = build_rag_context([_chunk("c1", "x", filename="f.md")])
    response = ChatBotResponse(
        response_text="x",
        language="en",
        sources_used=["INVALID"],
        confidence="high",
    )
    apply_hallucination_guard(response, rag)
    assert response.sources_used == ["INVALID"]
    assert response.confidence == "high"


def test_guard_low_confidence_with_no_sources_passes_through() -> None:
    rag = build_rag_context([_chunk("c1", "x", filename="f.md")])
    response = ChatBotResponse(
        response_text="I'm not sure",
        language="en",
        sources_used=[],
        confidence="low",
    )
    out = apply_hallucination_guard(response, rag)
    assert out.response_text == "I'm not sure"
    assert out.confidence == "low"
