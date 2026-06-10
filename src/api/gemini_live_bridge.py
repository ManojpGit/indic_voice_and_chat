"""Gemini Live (speech-to-speech) dev-console bridge.

A parallel to ``BrowserVoiceBridge`` that drives a duplex S2S session instead of
the STT->LLM->TTS cascade. It speaks the SAME browser wire protocol (binary
PCM16@16k both ways + JSON status/transcript/interrupt/outcome), so the existing
``static/dev_console.html`` client connects unchanged. Caller audio is forwarded
to the model; the model's audio is resampled 24k->16k and streamed back; native
interruption maps to ``{type:interrupt}``; a ``record_turn_signal`` tool-call
drives ``VoiceBotAgent.apply_signal`` (state machine + slots); input/output
transcriptions populate ``session.turns`` for the same post-call outcome path.
"""

from __future__ import annotations

import audioop
import asyncio
import json
import logging
from datetime import UTC, datetime

from src.agents.state_machine import Event, State
from src.analysis.call_outcome import analyze_call
from src.interfaces.llm import LLMMessage
from src.interfaces.realtime import IRealtimeSession, RealtimeConfig, RealtimeTool

log = logging.getLogger(__name__)

_SEND_CHUNK = 8192
_OUT_RATE = 16000   # browser plays PCM16 @16k

RECORD_TURN_SIGNAL = RealtimeTool(
    name="record_turn_signal",
    description="Record the dialogue action and any slot values learned this turn.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING", "enum": [
                "continue", "clarify", "transfer", "schedule_callback",
                "send_info", "close_positive", "close_negative", "end"]},
            "updated_slots": {"type": "OBJECT"},
        },
        "required": ["action"],
    },
)


class GeminiLiveBridge:
    """One bridge per browser connection. Drive with ``run()``.

    ``connect_session`` is an async callable ``(RealtimeConfig) -> IRealtimeSession``
    (injectable for tests; defaults to GeminiLiveSession.connect bound to the key).
    """

    def __init__(self, *, websocket, agent, config: RealtimeConfig, connect_session,
                 llm=None, tenant_timezone: str = "Asia/Kolkata") -> None:
        self._ws = websocket
        self._agent = agent
        self._config = config
        self._connect_session = connect_session
        self._llm = llm
        self._tenant_timezone = tenant_timezone

        self._session: IRealtimeSession | None = None
        self._stopped = False
        self._outcome_emitted = False
        self._last_action: str | None = None
        # per-turn accumulators
        self._user_buf = ""
        self._agent_buf = ""
        self._pending_action: str | None = None
        self._pending_slots: dict = {}
        self._ratecv_state = None
        self._speaking = False

    # --- outbound helpers ---
    async def _send_json(self, obj: dict) -> None:
        await self._ws.send_text(json.dumps(obj))

    async def _send_pcm(self, pcm16: bytes) -> None:
        if not pcm16:
            return
        if not self._speaking:
            self._speaking = True
            await self._send_json({"type": "status", "status": "speaking"})
        for i in range(0, len(pcm16), _SEND_CHUNK):
            await self._ws.send_bytes(pcm16[i:i + _SEND_CHUNK])

    # --- entrypoint ---
    async def run(self) -> None:
        events_task = None
        await self._agent.start()
        try:
            await self._read_hello()
            self._session = await self._connect_session(self._config)
            events_task = asyncio.create_task(self._consume_events())
            # NOTE: no text kickoff. Sending a text/content turn before the audio
            # stream disrupts Gemini Live's automatic VAD, so subsequent caller
            # audio never endpoints. The user speaks first; the model leads its
            # first reply with the persona's greeting. Agent-greets-first (for
            # outbound) is a fast-follow needing a VAD-compatible trigger.
            await self._send_json({"type": "status", "status": "listening"})

            while not self._stopped:
                message = await self._ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is not None:
                    if self._session is not None:
                        await self._session.send_audio(data)   # caller audio -> model
                    continue
                text = message.get("text")
                if text is not None:
                    try:
                        ctrl = json.loads(text)
                    except (ValueError, TypeError):
                        ctrl = {}
                    if ctrl.get("type") == "end":
                        self._stopped = True
                        break
        except Exception:  # noqa: BLE001 - never let the bridge crash the socket handler
            log.exception("gemini live bridge crashed")
        finally:
            if events_task is not None:
                events_task.cancel()
                try:
                    await events_task
                except BaseException:  # noqa: BLE001
                    pass
            if self._session is not None:
                await self._session.aclose()
            await self._emit_outcome()
            await self._agent.handle_hangup()

    async def _read_hello(self) -> None:
        message = await self._ws.receive()
        if message.get("text"):
            try:
                json.loads(message["text"])
            except (ValueError, TypeError):
                pass

    # --- model event handling ---
    async def _consume_events(self) -> None:
        assert self._session is not None
        try:
            async for ev in self._session.events():
                if ev.type == "audio":
                    pcm16, self._ratecv_state = audioop.ratecv(
                        ev.audio, 2, 1, ev.audio_rate, _OUT_RATE, self._ratecv_state)
                    await self._send_pcm(pcm16)
                elif ev.type == "input_transcript":
                    self._user_buf += ev.text
                    await self._send_json({"type": "partial", "role": "user", "text": self._user_buf})
                elif ev.type == "output_transcript":
                    self._agent_buf += ev.text
                    await self._send_json({"type": "partial", "role": "agent", "text": self._agent_buf})
                elif ev.type == "tool_call":
                    if ev.tool_name == "record_turn_signal":
                        self._pending_action = ev.tool_args.get("action") or self._pending_action
                        self._pending_slots.update(ev.tool_args.get("updated_slots") or {})
                    await self._session.send_tool_response(
                        tool_id=ev.tool_id, name=ev.tool_name, response={"ok": True})
                elif ev.type == "interrupted":
                    self._speaking = False
                    await self._send_json({"type": "interrupt"})
                elif ev.type == "turn_complete":
                    await self._commit_turn()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a model/stream error ends the call, not crash
            log.exception("gemini live event stream ended")
            self._stopped = True

    async def _commit_turn(self) -> None:
        """Record the completed turn (transcript + slots) and advance state."""
        if getattr(self._agent.state, "is_terminal", False):
            return
        user = self._user_buf.strip()
        agent = self._agent_buf.strip()
        action = self._pending_action or "continue"
        if user or agent:
            if user:
                await self._send_json({"type": "transcript", "role": "user", "text": user})
            if agent:
                await self._send_json({"type": "transcript", "role": "agent", "text": agent})
            # Drive the machine LISTENING->PROCESSING so apply_signal's
            # LLM_RESPONSE_READY/RESPONSE_DELIVERED transitions are valid.
            if self._agent.state.state is State.LISTENING:
                await self._agent.state.fire(Event.UTTERANCE_COMPLETE)
            await self._agent.apply_signal(
                user_text=user, agent_text=agent, action=action,
                updated_slots=self._pending_slots)
            self._last_action = action
        self._user_buf = ""
        self._agent_buf = ""
        self._pending_action = None
        self._pending_slots = {}
        self._speaking = False
        if getattr(self._agent.state, "is_terminal", False):
            self._stopped = True
            await self._emit_outcome()
        else:
            await self._send_json({"type": "status", "status": "listening"})

    async def _emit_outcome(self) -> None:
        if self._outcome_emitted or self._llm is None:
            return
        self._outcome_emitted = True
        try:
            transcript = [m for m in getattr(self._agent.session, "turns", [])
                          if isinstance(m, LLMMessage)]
            analysis = await analyze_call(
                transcript=transcript, slots=self._agent.slots.values,
                telephony_status=None, final_action=self._last_action,
                tenant_timezone=self._tenant_timezone, now=datetime.now(UTC), llm=self._llm)
        except Exception:  # noqa: BLE001
            log.exception("call outcome analysis failed")
            return
        cb = analysis.callback_datetime
        log.info("call outcome", extra={"outcome": analysis.outcome.value,
                 "source": analysis.analysis_source, "summary": analysis.summary[:200]})
        try:
            await self._send_json({
                "type": "outcome", "outcome": analysis.outcome.value,
                "summary": analysis.summary, "notes": analysis.notes,
                "callback_datetime": cb.isoformat() if cb else None,
                "callback_phrase": analysis.callback_phrase, "source": analysis.analysis_source})
        except Exception:  # noqa: BLE001
            log.warning("outcome computed but not delivered (socket closed)")
