"""Shared base class for VoiceBot and ChatBot agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.agents.state_machine import AgentStateMachine
from src.dialogue.context import SessionStore
from src.dialogue.slots import SlotFiller
from src.interfaces.llm import LLMMessage


@dataclass
class AgentSession:
    """Per-conversation state held in memory while the agent is running.

    Persisted snapshots live in Redis (``SessionStore``); this dataclass is
    the live view used by the agent's run loop.
    """

    session_id: str
    campaign_id: Optional[str] = None
    lead_id: Optional[str] = None
    lead_data: dict[str, Any] = field(default_factory=dict)
    turns: list[LLMMessage] = field(default_factory=list)
    sentiment_history: list[str] = field(default_factory=list)


class BaseAgent:
    """Shared scaffolding: session, state machine, slots, redis writer."""

    def __init__(
        self,
        session: AgentSession,
        state_machine: AgentStateMachine,
        slots: SlotFiller,
        store: Optional[SessionStore] = None,
    ) -> None:
        self.session = session
        self.state = state_machine
        self.slots = slots
        self.store = store

    async def persist_turn(self, role: str, content: str, metadata: Optional[dict] = None) -> None:
        if self.store is None:
            return
        await self.store.append_history(
            self.session.session_id,
            {"role": role, "content": content, "metadata": metadata or {}},
        )

    async def persist_state(self, extra: Optional[dict] = None) -> None:
        if self.store is None:
            return
        payload = {
            "state": self.state.state.value,
            "slots": self.slots.values,
        }
        if extra:
            payload.update(extra)
        await self.store.set_state(self.session.session_id, payload)
