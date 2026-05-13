from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

from src.agents.base import AgentSession
from src.agents.chatbot import ChatBotAgent
from src.dialogue.context import SessionStore
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult
from src.interfaces.vector_store import Document
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder, IdentityReranker
from src.rag.retriever import HybridRetriever, RetrievalConfig


# --- Fakes ---------------------------------------------------------------


class FakeLLM(ILLMProvider):
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls: list[list[LLMMessage]] = []

    async def generate(self, messages, config) -> LLMResult:
        self.calls.append(list(messages))
        return LLMResult(text=json.dumps(self._payload), finish_reason="stop")

    async def generate_stream(self, messages, config) -> AsyncIterator[str]:
        if False:
            yield  # pragma: no cover


# --- Fixtures ------------------------------------------------------------


@pytest.fixture
async def retriever(tmp_faiss_index: str) -> HybridRetriever:
    store = FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})
    r = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        reranker=IdentityReranker(),
        config=RetrievalConfig(
            strategy="hybrid",
            top_k=2,
            oversample_k=8,
            reranking=True,
            similarity_threshold=0.0,
        ),
    )
    await r.index([
        Document(id="c1", content="Plan B has 500GB unlimited data.", metadata={"filename": "plans.pdf", "page": 2}),
        Document(id="c2", content="Plan A is the basic 100GB plan for Rs 199.", metadata={"filename": "plans.pdf", "page": 1}),
        Document(id="c3", content="Cooking recipes for biryani and other dishes.", metadata={"filename": "cookbook.md"}),
    ])
    return r


def _make_agent(llm, retriever, store=None) -> ChatBotAgent:
    return ChatBotAgent(
        session=AgentSession(session_id="cb-1"),
        llm=llm,
        retriever=retriever,
        company_name="Acme",
        language_default="en",
        store=store,
    )


# --- Tests --------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_message_full_happy_path(retriever) -> None:
    llm = FakeLLM({
        "response_text": "Plan B has 500GB unlimited data.",
        "language": "en",
        "sources_used": ["plans.pdf:2"],
        "confidence": "high",
        "action": "none",
        "suggested_followups": ["What's the price?"],
    })
    agent = _make_agent(llm, retriever)

    result = await agent.handle_message("Tell me about Plan B")
    assert result.response.response_text == "Plan B has 500GB unlimited data."
    assert result.response.confidence == "high"
    assert "plans.pdf:2" in result.response.sources_used
    assert len(result.retrieved) >= 1
    # System prompt was built with retrieved context
    sent = llm.calls[0]
    assert sent[0].role == "system"
    assert "Plan B" in sent[0].content


@pytest.mark.asyncio
async def test_handle_message_empty_input_returns_early(retriever) -> None:
    llm = FakeLLM({})
    agent = _make_agent(llm, retriever)
    result = await agent.handle_message("   ")
    assert result.response.parse_error == "empty user input"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_hallucination_guard_strips_invented_citations(retriever) -> None:
    llm = FakeLLM({
        "response_text": "Plan B has 500GB and free 5G.",
        "language": "en",
        "sources_used": ["plans.pdf:2", "wireless.pdf:99"],  # second is invented
        "confidence": "high",
        "action": "none",
    })
    agent = _make_agent(llm, retriever)
    result = await agent.handle_message("Plan B details?")
    assert result.response.sources_used == ["plans.pdf:2"]
    assert result.response.confidence == "low"


@pytest.mark.asyncio
async def test_followup_turn_includes_prior_history_in_prompt(retriever) -> None:
    llm = FakeLLM({
        "response_text": "Plan B is Rs 699 per month.",
        "language": "en",
        "sources_used": ["plans.pdf:2"],
        "confidence": "high",
        "action": "none",
    })
    agent = _make_agent(llm, retriever)
    await agent.handle_message("Tell me about Plan B")
    await agent.handle_message("What's the price?")
    last_call = llm.calls[-1]
    # First message is system, then prior user/assistant pairs, then current user.
    roles = [m.role for m in last_call]
    assert roles[0] == "system"
    # The prior user message is in history
    assert any("Tell me about Plan B" in m.content for m in last_call if m.role == "user")
    assert last_call[-1].role == "user"
    assert last_call[-1].content == "What's the price?"


@pytest.mark.asyncio
async def test_persists_to_redis(retriever, fake_redis) -> None:
    llm = FakeLLM({
        "response_text": "Plan B has 500GB.",
        "language": "en",
        "sources_used": ["plans.pdf:2"],
        "confidence": "high",
        "action": "none",
    })
    store = SessionStore(fake_redis, ttl_seconds=300)
    agent = _make_agent(llm, retriever, store=store)
    await agent.handle_message("Plan B?")

    history = await store.get_history("cb-1")
    roles = [t["role"] for t in history]
    assert roles == ["user", "agent"]
    assert "Plan B" in history[0]["content"]
    state = await store.get_state("cb-1")
    assert state["agent_type"] == "chatbot"
    assert state["last_confidence"] == "high"
    assert state["turn_count"] == 1


@pytest.mark.asyncio
async def test_get_history_in_memory_when_no_store(retriever) -> None:
    llm = FakeLLM({
        "response_text": "Plan B has 500GB.",
        "language": "en",
        "sources_used": ["plans.pdf:2"],
        "confidence": "high",
        "action": "none",
    })
    agent = _make_agent(llm, retriever)
    await agent.handle_message("Plan B?")
    history = await agent.get_history()
    assert any(h["role"] == "user" for h in history)
    assert any(h["role"] == "assistant" for h in history)
