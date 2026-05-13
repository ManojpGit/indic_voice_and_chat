"""Route-level tests for /api/v1/chat/* endpoints."""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.base import AgentSession
from src.agents.chatbot import ChatBotAgent
from src.api import chat
from src.auth import TenantContext, register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import TenantSettings
from src.dialogue.context import SessionStore
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult
from src.interfaces.vector_store import Document
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder
from src.rag.retriever import HybridRetriever, RetrievalConfig


HEADERS = {"Authorization": "Bearer test-token"}


class _FakeLLM(ILLMProvider):
    def __init__(self, payload: dict) -> None:
        self._json = json.dumps(payload)

    async def generate(self, messages, config) -> LLMResult:
        return LLMResult(text=self._json, finish_reason="stop")

    async def generate_stream(self, messages, config) -> AsyncIterator[str]:
        if False:
            yield  # pragma: no cover


@pytest.fixture
async def app(tmp_faiss_index: str, fake_redis):
    store = FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=2, oversample_k=8, reranking=False),
    )
    await retriever.index([
        Document(id="c1", content="Plan B has 500GB unlimited.", metadata={"filename": "plans.md", "page": 0})
    ])

    session_store = SessionStore(fake_redis, ttl_seconds=300, tenant_id="t1")

    async def factory(tenant: TenantContext, session_id: str) -> ChatBotAgent:
        return ChatBotAgent(
            session=AgentSession(session_id=session_id),
            llm=_FakeLLM({
                "response_text": "Plan B has 500GB.",
                "language": "en",
                "sources_used": ["plans.md:0"],
                "confidence": "high",
                "action": "none",
            }),
            retriever=retriever,
            company_name=tenant.settings.name,
            store=session_store,
        )

    chat.set_chatbot_factory(factory)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        plaintext_tokens=["test-token"],
    )
    a = FastAPI()
    a.include_router(chat.router)
    yield a
    chat.set_chatbot_factory(None)
    set_tenant_resolver(None)


def test_post_message_creates_session_and_returns_answer(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post("/chat/message", json={"message": "Tell me about Plan B"}, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["response_text"] == "Plan B has 500GB."
    assert body["session_id"].startswith("chat_")
    assert "plans.md:0" in body["sources_used"]
    assert body["confidence"] == "high"


def test_post_message_uses_supplied_session_id(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post(
        "/chat/message",
        json={"session_id": "my-session", "message": "Plan B?"},
        headers=HEADERS,
    )
    assert resp.json()["session_id"] == "my-session"


def test_get_history_returns_persisted_turns(app: FastAPI) -> None:
    client = TestClient(app)
    client.post("/chat/message", json={"session_id": "hist-test", "message": "First Q"}, headers=HEADERS)
    client.post("/chat/message", json={"session_id": "hist-test", "message": "Second Q"}, headers=HEADERS)

    resp = client.get("/chat/history/hist-test", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "hist-test"
    assert len(body["history"]) == 4  # 2 user + 2 agent
    assert body["history"][0]["role"] == "user"
    assert body["history"][0]["content"] == "First Q"


def test_websocket_round_trip(app: FastAPI) -> None:
    client = TestClient(app)
    with client.websocket_connect("/chat/ws", headers=HEADERS) as ws:
        ws.send_text(json.dumps({"session_id": "ws-1", "message": "Plan B?"}))
        reply = json.loads(ws.receive_text())
    assert reply["session_id"] == "ws-1"
    assert reply["response_text"] == "Plan B has 500GB."


def test_websocket_invalid_json_returns_error(app: FastAPI) -> None:
    client = TestClient(app)
    with client.websocket_connect("/chat/ws", headers=HEADERS) as ws:
        ws.send_text("not json")
        err = json.loads(ws.receive_text())
        assert "error" in err


def test_websocket_missing_message_returns_error(app: FastAPI) -> None:
    client = TestClient(app)
    with client.websocket_connect("/chat/ws", headers=HEADERS) as ws:
        ws.send_text(json.dumps({"session_id": "x"}))
        err = json.loads(ws.receive_text())
        assert "error" in err


def test_post_when_factory_unset_returns_503() -> None:
    chat.set_chatbot_factory(None)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="X"),
        plaintext_tokens=["test-token"],
    )
    a = FastAPI()
    a.include_router(chat.router)
    client = TestClient(a)
    resp = client.post("/chat/message", json={"message": "x"}, headers=HEADERS)
    assert resp.status_code == 503
    set_tenant_resolver(None)


def test_post_without_auth_returns_401(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post("/chat/message", json={"message": "x"})
    assert resp.status_code == 401
