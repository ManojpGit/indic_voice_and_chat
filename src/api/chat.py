"""ChatBot endpoints (PRD §7.3).

Three surfaces, all backed by the same ``ChatBotAgent``:

- ``WS  /chat/ws``        real-time bidirectional, one JSON message per turn
- ``POST /chat/message``  single-turn HTTP, suitable for WhatsApp / async channels
- ``GET  /chat/history/{session_id}`` retrieve the persisted conversation

Agent construction happens in ``set_chatbot_factory(factory)`` — the factory
takes a session_id and returns a configured ``ChatBotAgent``. Wiring this
factory at app startup keeps the routes thin and easy to test.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Awaitable, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.agents.chatbot import ChatBotAgent
from src.auth import TenantContext, current_tenant

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


# --- DI -----------------------------------------------------------------


ChatBotFactory = Callable[[TenantContext, str], Awaitable[ChatBotAgent]]
_factory: Optional[ChatBotFactory] = None


def set_chatbot_factory(factory: Optional[ChatBotFactory]) -> None:
    """Register / unregister the per-session agent factory."""
    global _factory
    _factory = factory


async def _get_agent(tenant: TenantContext, session_id: str) -> ChatBotAgent:
    if _factory is None:
        raise HTTPException(
            status_code=503,
            detail="chatbot factory not initialized; set_chatbot_factory() not called",
        )
    return await _factory(tenant, session_id)


def _scoped_session(tenant: TenantContext, session_id: str) -> str:
    """Namespace ``session_id`` by tenant so two tenants can use the same id."""
    return f"{tenant.id}:{session_id}"


# --- Schemas ------------------------------------------------------------


class ChatMessageRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(min_length=1)


class ChatMessageResponse(BaseModel):
    session_id: str
    response_text: str
    language: str
    confidence: str
    sources_used: list[str]
    action: str
    suggested_followups: list[str] = []


class HistoryEntry(BaseModel):
    role: str
    content: str
    metadata: Optional[dict] = None


class HistoryResponse(BaseModel):
    session_id: str
    history: list[HistoryEntry]


# --- HTTP routes --------------------------------------------------------


@router.post("/message", response_model=ChatMessageResponse)
async def chat_message(
    req: ChatMessageRequest, tenant: TenantContext = Depends(current_tenant),
) -> ChatMessageResponse:
    session_id = req.session_id or _new_session_id()
    agent = await _get_agent(tenant, _scoped_session(tenant, session_id))
    result = await agent.handle_message(req.message)
    return ChatMessageResponse(
        session_id=session_id,
        response_text=result.response.response_text,
        language=result.response.language,
        confidence=result.response.confidence,
        sources_used=result.response.sources_used,
        action=result.response.action,
        suggested_followups=result.response.suggested_followups,
    )


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def chat_history(
    session_id: str = Path(min_length=1),
    tenant: TenantContext = Depends(current_tenant),
) -> HistoryResponse:
    agent = await _get_agent(tenant, _scoped_session(tenant, session_id))
    raw = await agent.get_history()
    return HistoryResponse(
        session_id=session_id,
        history=[HistoryEntry(**h) for h in raw],
    )


# --- WebSocket route ----------------------------------------------------


@router.websocket("/ws")
async def chat_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    if _factory is None:
        await websocket.close(code=1011, reason="chatbot factory unset")
        return

    # Resolve tenant from the WS bearer token (or X-Tenant-Slug header).
    from src.auth.middleware import _resolve as _resolve_tenant  # local import to avoid cycle

    tenant = await _resolve_tenant(websocket, allow_slug_header=True)
    if tenant is None:
        await websocket.close(code=1008, reason="missing/invalid tenant credentials")
        return

    session_id: Optional[str] = None
    agent: Optional[ChatBotAgent] = None
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid json"}))
                continue

            user_text = (msg.get("message") or "").strip()
            if not user_text:
                await websocket.send_text(json.dumps({"error": "missing 'message'"}))
                continue

            if agent is None:
                session_id = msg.get("session_id") or _new_session_id()
                agent = await _factory(tenant, _scoped_session(tenant, session_id))

            result = await agent.handle_message(user_text)
            await websocket.send_text(json.dumps({
                "session_id": session_id,
                "response_text": result.response.response_text,
                "language": result.response.language,
                "confidence": result.response.confidence,
                "sources_used": result.response.sources_used,
                "action": result.response.action,
                "suggested_followups": result.response.suggested_followups,
            }))
    except WebSocketDisconnect:
        log.info("chat ws client disconnected", extra={"session_id": session_id})
    except Exception:  # noqa: BLE001 — never let the websocket task escape
        log.exception("chat websocket crashed")
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


def _new_session_id() -> str:
    return f"chat_{uuid.uuid4().hex[:12]}"
