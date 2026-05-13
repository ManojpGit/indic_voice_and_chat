"""ChatBot agent (Phase 4).

Text-only counterpart to VoiceBotAgent. One ``handle_message(user_text)``
call per user turn:

1. Retrieve top chunks from the hybrid retriever for the user's question.
2. Build the RAG context block.
3. Compose the system + history + user messages and call the LLM (non-
   streaming — chat clients render the full message at once).
4. Parse the structured ChatBotResponse.
5. Apply the hallucination guard against the retrieved sources.
6. Persist user + agent turns to Redis.

The agent is deliberately stateless about telephony / VAD / audio — those
are voice concerns that the VoiceBotAgent handles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.agents.base import AgentSession, BaseAgent
from src.dialogue.context import SessionStore
from src.dialogue.prompts import build_chatbot_system_prompt
from src.dialogue.response_parser import ChatBotResponse, parse_chatbot_response
from src.dialogue.slots import SlotFiller, SlotSchema
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage
from src.rag.context_builder import (
    GuardConfig,
    apply_hallucination_guard,
    build_rag_context,
)
from src.rag.retriever import HybridRetriever, RetrievedChunk


@dataclass
class ChatTurnResult:
    response: ChatBotResponse
    retrieved: list[RetrievedChunk]
    rag_context_chars: int


class ChatBotAgent(BaseAgent):
    def __init__(
        self,
        session: AgentSession,
        llm: ILLMProvider,
        retriever: HybridRetriever,
        llm_config: Optional[LLMConfig] = None,
        company_name: str = "[Your Company]",
        language_default: str = "en",
        store: Optional[SessionStore] = None,
        guard_config: Optional[GuardConfig] = None,
        max_context_chars: int = 4000,
    ) -> None:
        # ChatBot doesn't need slots — pass an empty schema so BaseAgent is happy.
        super().__init__(
            session=session,
            state_machine=None,  # type: ignore[arg-type] — chatbot doesn't drive a call FSM
            slots=SlotFiller(SlotSchema()),
            store=store,
        )
        self._llm = llm
        self._retriever = retriever
        self._llm_config = llm_config or LLMConfig(response_format="json")
        self._company = company_name
        self._language = language_default
        self._guard = guard_config
        self._max_context_chars = max_context_chars

    async def handle_message(self, user_text: str) -> ChatTurnResult:
        if not user_text or not user_text.strip():
            return ChatTurnResult(
                response=ChatBotResponse(
                    response_text="",
                    language=self._language,
                    parse_error="empty user input",
                ),
                retrieved=[],
                rag_context_chars=0,
            )

        # 1. Retrieval
        retrieved = await self._retriever.search(user_text)

        # 2. Build context
        rag = build_rag_context(retrieved, max_chars=self._max_context_chars)

        # 3. Compose messages
        system_prompt = build_chatbot_system_prompt(
            company_name=self._company,
            language_default=self._language,
            rag_context=rag.text,
        )
        messages: list[LLMMessage] = [LLMMessage(role="system", content=system_prompt)]
        # Replay history (user/agent only — system is rebuilt each turn so
        # the freshest RAG context is in front of the LLM).
        for m in self.session.turns:
            if m.role in ("user", "assistant"):
                messages.append(m)
        messages.append(LLMMessage(role="user", content=user_text))

        # 4. LLM
        result = await self._llm.generate(messages, self._llm_config)
        response = parse_chatbot_response(result.text)

        # 5. Guard
        response = apply_hallucination_guard(response, rag, self._guard)

        # 6. Persist
        self.session.turns.append(LLMMessage(role="user", content=user_text))
        self.session.turns.append(
            LLMMessage(role="assistant", content=response.response_text)
        )
        await self.persist_turn("user", user_text)
        await self.persist_turn(
            "agent",
            response.response_text,
            metadata={
                "confidence": response.confidence,
                "sources_used": response.sources_used,
                "action": response.action,
                "retrieved_count": len(retrieved),
            },
        )
        # ChatBot has no state machine, so just persist a basic state blob.
        if self.store is not None:
            await self.store.set_state(
                self.session.session_id,
                {
                    "agent_type": "chatbot",
                    "last_action": response.action,
                    "last_confidence": response.confidence,
                    "turn_count": sum(
                        1 for m in self.session.turns if m.role == "user"
                    ),
                },
            )

        return ChatTurnResult(
            response=response,
            retrieved=retrieved,
            rag_context_chars=len(rag.text),
        )

    async def get_history(self) -> list[dict[str, Any]]:
        if self.store is None:
            return [
                {"role": m.role, "content": m.content}
                for m in self.session.turns
                if m.role in ("user", "assistant")
            ]
        return await self.store.get_history(self.session.session_id)

    # ChatBot doesn't drive a state machine; override BaseAgent's persistence.
    async def persist_state(self, extra: Optional[dict] = None) -> None:  # type: ignore[override]
        if self.store is None:
            return
        payload = {"agent_type": "chatbot"}
        if extra:
            payload.update(extra)
        await self.store.set_state(self.session.session_id, payload)
