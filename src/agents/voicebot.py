"""VoiceBot agent.

Wraps the pipeline engine with conversation control: prompt building, turn
sequencing, slot updates, structured response parsing, state-machine event
firing, and session persistence.

Telephony I/O lives outside this class — the agent receives a captured
audio buffer per turn from the telephony layer (Twilio Media Streams
websocket in Phase 3) and emits audio chunks to a sink the telephony layer
provides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from src.agents.base import AgentSession, BaseAgent
from src.agents.state_machine import AgentStateMachine, Event, State
from src.dialogue.prompts import VoiceBotScript, build_voicebot_system_prompt
from src.dialogue.response_parser import VoiceBotResponse, parse_voicebot_response
from src.dialogue.slots import SlotFiller, SlotSchema
from src.interfaces.llm import LLMMessage
from src.pipeline.engine import AudioSink, PipelineEngine, TurnResult


log = logging.getLogger(__name__)


@dataclass
class TurnOutcome:
    """The full result of one user-utterance / agent-response cycle."""

    response: VoiceBotResponse
    pipeline: TurnResult


# Map LLM-emitted ``action`` to the state-machine event that follows.
_ESCALATION_ACTIONS = {"transfer", "schedule_callback"}
_END_ACTIONS = {"close_positive", "close_negative", "end"}


class VoiceBotAgent(BaseAgent):
    def __init__(
        self,
        session: AgentSession,
        state_machine: AgentStateMachine,
        slot_schema: SlotSchema,
        script: VoiceBotScript,
        engine: PipelineEngine,
        store=None,
        extra_directives: Optional[list[str]] = None,
    ) -> None:
        slots = SlotFiller(slot_schema)
        super().__init__(session=session, state_machine=state_machine, slots=slots, store=store)
        self._script = script
        self._engine = engine
        self._extra_directives = extra_directives
        self._system_prompt = build_voicebot_system_prompt(
            script=script,
            schema=slot_schema,
            lead_data=session.lead_data,
            extra_directives=extra_directives,
        )
        self.session.turns.append(LLMMessage(role="system", content=self._system_prompt))

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    async def start(self) -> None:
        """Move from IDLE to LISTENING. Call once when the call connects."""
        await self.state.fire_if_possible(Event.CALL_CONNECTED)
        await self.persist_state()

    async def play_opening(self, audio_sink: AudioSink) -> None:
        """Speak the campaign opening line as the agent's first turn.

        For outbound campaigns the agent must speak first — the user just
        answered the phone and is silent. We synthesize the script's
        opening line, push the audio through ``audio_sink``, append it to
        the conversation history, and stay in LISTENING for the user's
        reply. Skips silently if there's no opening configured.
        """
        opening = (self._script.opening or "").strip()
        if not opening:
            return
        # Substitute simple template tokens with known lead data.
        rendered = opening.format(**self._template_vars())

        # The TTS goes through the pipeline engine's TTS provider so the
        # adapter-level streaming + sample-rate handling stays consistent
        # with the per-turn synthesis path.
        from src.interfaces.tts import TTSConfig as _TTSConfig
        try:
            tts_result = await self._engine._tts.synthesize(  # type: ignore[attr-defined]
                rendered, _TTSConfig(language=self._script.language_default + "-IN"
                                     if len(self._script.language_default) == 2
                                     else self._script.language_default),
            )
        except Exception:
            log.exception("opening synthesis failed; skipping")
            return
        if tts_result.audio:
            await audio_sink(tts_result.audio)
        self.session.turns.append(LLMMessage(role="assistant", content=rendered))
        await self.persist_turn("agent", rendered, metadata={"phase": "opening"})

    def _template_vars(self) -> dict[str, str]:
        data = dict(self.session.lead_data or {})
        # Common fallbacks so f-string substitution doesn't KeyError.
        data.setdefault("lead_name", data.get("name", "ji"))
        data.setdefault("agent_name", self._script.agent_name)
        data.setdefault("company_name", self._script.company_name)
        return {k: str(v) for k, v in data.items()}

    async def handle_turn(self, captured_audio: bytes, audio_sink: AudioSink) -> TurnOutcome:
        """Drive one user-utterance -> agent-response cycle.

        State transitions happen at the natural boundaries:
        LISTENING -> PROCESSING (utterance complete) -> RESPONDING ->
        LISTENING (response delivered) | ESCALATING | ENDED.
        """
        if self.state.state is not State.LISTENING:
            raise RuntimeError(
                f"handle_turn called from {self.state.state.value}, expected listening"
            )

        # Utterance complete (the telephony layer determined this via VAD).
        await self.state.fire(Event.UTTERANCE_COMPLETE)

        pipeline_result = await self._engine.run_turn(
            captured_audio=captured_audio,
            history=self.session.turns,
            audio_sink=audio_sink,
        )

        # Empty STT — no real user turn happened. Walk the state machine
        # back to LISTENING (PROCESSING -> RESPONDING -> LISTENING) and let
        # the silence handler decide what to do next.
        if not pipeline_result.user_text:
            await self.state.fire(Event.LLM_RESPONSE_READY)
            await self.state.fire(Event.RESPONSE_DELIVERED)
            return TurnOutcome(
                response=VoiceBotResponse(
                    response_text="", action="continue", parse_error="empty STT"
                ),
                pipeline=pipeline_result,
            )

        # Now we have user text — record it, advance the state machine.
        self.session.turns.append(LLMMessage(role="user", content=pipeline_result.user_text))
        await self.persist_turn("user", pipeline_result.user_text)

        await self.state.fire(Event.LLM_RESPONSE_READY)

        # Parse structured response, apply slots.
        response = parse_voicebot_response(pipeline_result.agent_text)
        applied = self.slots.apply_updates(response.updated_slots)

        # The assistant's textual reply is what was actually spoken
        # (response.response_text is what the model intended to say).
        self.session.turns.append(
            LLMMessage(role="assistant", content=response.response_text)
        )
        await self.persist_turn(
            "agent",
            response.response_text,
            metadata={
                "action": response.action,
                "sentiment": response.sentiment,
                "phase": response.conversation_phase,
                "applied_slots": applied,
                "metrics": pipeline_result.metrics.__dict__,
            },
        )
        if response.sentiment:
            self.session.sentiment_history.append(response.sentiment)

        # Decide what state to go to next based on the LLM's action.
        if response.action in _ESCALATION_ACTIONS:
            await self.state.fire(Event.ESCALATION_REQUESTED)
        elif response.action in _END_ACTIONS:
            # Treat 'end' as a hangup-like terminal event from RESPONDING.
            await self.state.fire(Event.RESPONSE_DELIVERED)  # land in LISTENING
            await self.state.fire(Event.HANGUP)              # then terminate
        else:
            await self.state.fire(Event.RESPONSE_DELIVERED)

        await self.persist_state(extra={"last_action": response.action})
        return TurnOutcome(response=response, pipeline=pipeline_result)

    async def handle_silence_timeout(self, audio_sink: AudioSink) -> Optional[TurnOutcome]:
        """User went silent in LISTENING — re-prompt or end the call.

        Currently emits no real audio (the LLM call is skipped). The
        telephony layer is expected to play a pre-rolled "are you there?"
        prompt and then call ``handle_turn`` again. We just advance the
        state machine.
        """
        if self.state.state is not State.LISTENING:
            return None
        await self.state.fire(Event.SILENCE_TIMEOUT)
        # Auto-return to LISTENING so the call doesn't get stuck.
        await self.state.fire(Event.RESPONSE_DELIVERED)
        return None

    async def handle_extended_silence(self) -> None:
        if self.state.state is State.LISTENING:
            await self.state.fire(Event.EXTENDED_SILENCE)
        await self.persist_state()

    async def handle_hangup(self) -> None:
        # If we're already terminal (e.g. close_positive set last_action),
        # don't overwrite the persisted final state.
        if self.state.is_terminal:
            return
        await self.state.fire_if_possible(Event.HANGUP)
        await self.persist_state()
