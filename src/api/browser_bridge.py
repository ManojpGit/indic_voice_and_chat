# src/api/browser_bridge.py
"""Browser voice dev-console bridge.

A dev-only transport that mirrors the Twilio/Exotel media bridges but speaks
a browser-friendly protocol on a single WebSocket:

- BINARY frames  = raw PCM16-LE, 16 kHz mono, both directions (mic in / TTS out)
- TEXT frames    = JSON control + debug:
    in : {"type":"hello","tenant":"dev"}
    out: {"type":"status","status":"opening|listening|thinking|speaking"}
         {"type":"transcript","role":"user|agent","text":...}
         {"type":"state","state":...,"slots":{...}}

The dialogue pipeline (VoiceBotAgent + PipelineEngine + VAD) is reused
untouched; this class only does framing + debug events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.analysis.call_outcome import analyze_call
from src.interfaces.llm import LLMMessage
from src.pipeline.turn_capture import accumulate_and_detect
from src.pipeline.vad import EndpointConfig, EndpointDetector, VADDetector

log = logging.getLogger(__name__)

# Browser captures and resamples to the internal pipeline rate directly.
BROWSER_SAMPLE_RATE = 16000

# Chunk size for outbound PCM frames (bytes). 8 KB ~= 256 ms @16 kHz PCM16.
_SEND_CHUNK = 8192

# Streaming-STT resilience: if Deepgram drops the socket mid-call, reopen it
# rather than wedging the turn loop. Backoff + cap guard against a tight spin
# when the upstream is persistently unavailable.
_STREAM_REOPEN_BACKOFF_S = 0.3
_MAX_STREAM_REOPENS = 10


@dataclass
class BrowserBridgeConfig:
    pcm_sample_rate: int = BROWSER_SAMPLE_RATE
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    default_tenant: str = "dev"


class BrowserVoiceBridge:
    """One bridge per browser connection. Drive with ``run()``."""

    def __init__(
        self,
        websocket,
        agent,
        vad: VADDetector,
        config: BrowserBridgeConfig | None = None,
        stream_provider=None,
        llm=None,
        tenant_timezone: str = "Asia/Kolkata",
    ):
        self._ws = websocket
        self._agent = agent
        self._vad = vad
        self._config = config or BrowserBridgeConfig()
        self._capture_buffer = bytearray()
        # Browsers deliver tiny (~2.67 ms) audio quanta; assemble them into
        # whole VAD frames (frame_ms of audio) before endpointing, so the
        # EndpointDetector's per-frame timing is correct.
        self._inbound = bytearray()
        self._frame_bytes = int(self._config.pcm_sample_rate * vad.frame_ms / 1000) * 2
        self._endpoint = EndpointDetector(vad.frame_ms, self._config.endpoint)
        self._stopped = False
        # Wall-clock estimate of when the browser's queued playback will finish,
        # mirroring its gapless scheduling. Used to avoid closing the socket
        # (which tears down playback) before the final line has been heard.
        self._play_until = 0.0
        self._stream_provider = stream_provider
        self._stream_session = None
        self._agent_busy = False
        self._cancel_event = None  # set per in-flight streaming turn; barge-in fires it
        self._barge_enabled = False     # set by the client's {"type":"config","barge":...}
        self._turn_task = None          # in-flight turn runs as a task so barge can interrupt it
        self._barge_start_t = None      # monotonic time the current interruption's speech began
        self._had_turn = False          # opening is not barge-able; arm only after the first turn
        self._llm = llm
        self._tenant_timezone = tenant_timezone
        self._last_action: str | None = None
        self._outcome_emitted = False

    # --- outbound helpers ---------------------------------------------

    async def _send_json(self, obj: dict) -> None:
        await self._ws.send_text(json.dumps(obj))

    async def _send_pcm(self, pcm16: bytes) -> None:
        """AudioSink: ship agent TTS audio to the browser as binary frames.

        Unlike Twilio there is no real-time pacing — the browser schedules
        gapless playback itself, so we just chunk and send.
        """
        if not pcm16:
            return
        await self._send_json({"type": "status", "status": "speaking"})
        for i in range(0, len(pcm16), _SEND_CHUNK):
            await self._ws.send_bytes(pcm16[i : i + _SEND_CHUNK])
        # Track when this audio will finish playing (16-bit mono PCM), mirroring
        # the browser's gapless scheduling, so a terminal turn can wait for it.
        duration_s = len(pcm16) / 2 / self._config.pcm_sample_rate
        self._play_until = max(self._play_until, time.monotonic()) + duration_s
        await self._send_json({"type": "status", "status": "listening"})

    # --- entrypoint ---------------------------------------------------

    async def run(self) -> None:
        """Drive the connection until the browser disconnects or the agent ends."""
        stream_task = None
        await self._agent.start()
        mic_frames = 0
        exit_reason = "loop-end"
        try:
            # 1) Handshake: first text frame selects the tenant (already resolved
            #    by the caller, so we just consume it). Then play the opening.
            await self._read_hello()
            log.info("browser bridge: hello consumed, playing opening")
            await self._play_opening()
            log.info("browser bridge: opening done, entering listen loop")

            # 2) Turn loop.
            if self._stream_provider is not None:
                try:
                    from src.interfaces.stt import STTConfig
                    self._stream_session = await self._stream_provider.open_stream(
                        STTConfig(language="hi", sample_rate=self._config.pcm_sample_rate)
                    )
                    stream_task = asyncio.create_task(self._run_stream_consumer())
                    log.info("browser bridge: deepgram streaming session open")
                except Exception as e:  # noqa: BLE001 - fall back to batch
                    log.warning("streaming STT open failed (%s); using batch VAD", e)
                    self._stream_provider = None

            while not self._stopped:
                message = await self._ws.receive()
                if message.get("type") == "websocket.disconnect":
                    exit_reason = f"disconnect code={message.get('code')}"
                    break
                data = message.get("bytes")
                if data is not None:
                    mic_frames += 1
                    if mic_frames == 1:
                        log.info("browser bridge: first mic frame received")
                    await self._on_pcm_frame(data)
                    continue
                text = message.get("text")
                if text is not None:
                    try:
                        ctrl = json.loads(text)
                    except (ValueError, TypeError):
                        ctrl = {}
                    self._apply_control(ctrl)
                    if ctrl.get("type") == "end":
                        # Graceful client-initiated end: deliver the outcome while
                        # the socket is still open, then stop the loop. break (not
                        # continue) so exit_reason isn't overwritten by the while/else.
                        await self._emit_outcome()
                        self._stopped = True
                        exit_reason = "client end"
                        break
                    continue
            else:
                exit_reason = "stopped (terminal)"
        finally:
            log.info(
                "browser bridge: run() exiting",
                extra={"reason": exit_reason, "mic_frames": mic_frames},
            )
            if stream_task is not None:
                stream_task.cancel()
            if self._stream_session is not None:
                try:
                    await self._stream_session.aclose()
                except Exception:  # noqa: BLE001
                    pass
            try:
                await self._emit_outcome()
            except Exception:  # noqa: BLE001
                log.exception("emit outcome failed")
            await self._agent.handle_hangup()

    async def _read_hello(self) -> None:
        message = await self._ws.receive()
        # Tolerate a missing/early hello — tenant is already bound by the caller.
        if message.get("text"):
            try:
                json.loads(message["text"])
            except (ValueError, TypeError):
                pass

    async def _play_opening(self) -> None:
        await self._send_json({"type": "status", "status": "opening"})
        session = getattr(self._agent, "session", None)
        turns_before = len(session.turns) if session is not None else 0
        await self._agent.play_opening(self._send_pcm)
        opening_text = self._latest_assistant_text(turns_before)
        if opening_text:
            await self._send_json({"type": "transcript", "role": "agent", "text": opening_text})
        await self._emit_state()
        self._reset_capture()  # drop anything captured during the opening
        await self._send_json({"type": "status", "status": "listening"})

    def _latest_assistant_text(self, since: int) -> str:
        """Return the assistant message appended since index ``since`` (the
        rendered opening line), or '' if none / unavailable."""
        session = getattr(self._agent, "session", None)
        if session is None:
            return ""
        turns = getattr(session, "turns", [])
        if len(turns) > since and getattr(turns[-1], "role", None) == "assistant":
            return getattr(turns[-1], "content", "") or ""
        return ""

    # --- inbound ------------------------------------------------------

    async def _on_pcm_frame(self, pcm16: bytes) -> None:
        # Streaming mode: forward user-only audio to Deepgram. The client mutes
        # the mic while the agent speaks; the _agent_busy gate is belt-and-braces
        # so the agent's own audio is never streamed to the recognizer.
        if self._stream_session is not None:
            # Don't feed the recognizer while the agent is generating a reply
            # (_agent_busy) OR while its audio is still playing on the client
            # (now < _play_until). Otherwise the agent's own voice (echo) reaches
            # Deepgram as a continuous audio stream, which prevents it from
            # detecting an utterance-end gap — endpointing then stalls for many
            # seconds after the reply (the "stuck in listening" symptom), only
            # recovering once a real silence finally appears.
            if self._agent_busy or time.monotonic() < self._play_until:
                return
            try:
                await self._stream_session.send(pcm16)
            except Exception:  # noqa: BLE001 - drop to batch on socket failure
                log.warning("deepgram send failed; switching to batch VAD")
                self._stream_session = None
            return

        # Batch mode (unchanged): accumulate + local VAD endpointing.
        self._inbound.extend(pcm16)
        while not self._stopped and len(self._inbound) >= self._frame_bytes:
            frame = bytes(self._inbound[: self._frame_bytes])
            del self._inbound[: self._frame_bytes]
            if accumulate_and_detect(frame, self._vad, self._endpoint, self._capture_buffer):
                await self._dispatch_utterance()

    async def _dispatch_utterance(self) -> None:
        captured = bytes(self._capture_buffer)
        self._capture_buffer.clear()
        self._endpoint.reset()
        await self._send_json({"type": "status", "status": "thinking"})
        outcome = await self._agent.handle_turn(captured, self._send_pcm)
        self._last_action = outcome.response.action

        m = outcome.pipeline.metrics
        log.info(
            "browser turn",
            extra={
                "captured_bytes": len(captured),
                "user_text": (outcome.pipeline.user_text or "")[:80],
                "stt_ms": m.stt_latency_ms,
                "llm_ttft_ms": m.llm_ttft_ms,
                "llm_total_ms": m.llm_total_ms,
                "tts_first_ms": m.tts_first_chunk_ms,
                "total_ms": m.total_latency_ms,
                "action": outcome.response.action,
                "agent_text": (outcome.response.response_text or "")[:100],
                "error": outcome.response.parse_error or "",
            },
        )
        user_text = outcome.pipeline.user_text
        if user_text:
            await self._send_json({"type": "transcript", "role": "user", "text": user_text})
        agent_text = outcome.response.response_text
        if agent_text:
            await self._send_json({"type": "transcript", "role": "agent", "text": agent_text})
        # Surface a real failure (provider outage / unparseable LLM output) to the
        # dev console instead of leaving the user staring at silence. Routine
        # empty-STT turns are not errors.
        err = outcome.response.parse_error or ""
        if err and err != "empty STT":
            await self._send_json({"type": "error", "message": err})
        await self._emit_state()

        if getattr(self._agent.state, "is_terminal", False):
            self._stopped = True
            # The call is ending: let the browser finish playing the closing
            # line before run() returns and the socket closes (closing the
            # socket tears down playback and cuts the line off).
            remaining = self._play_until - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining + 0.5)
            return
        self._reset_capture()  # drop audio captured while the agent was busy
        await self._send_json({"type": "status", "status": "listening"})

    async def _run_stream_consumer(self) -> None:
        """Consume streaming-STT events, reopening the upstream if it drops.

        Deepgram can close the socket mid-call (e.g. a WebSocket keepalive-ping
        timeout). When that happens ``events()`` simply ends; without reopening,
        the turn loop would wedge in "listening" forever. So we loop: consume
        until the session ends, and if the call is still live, open a fresh
        session and keep going (with backoff + a cap to avoid a tight reopen
        spin if the upstream is persistently unavailable).
        """
        from src.interfaces.stt import STTConfig
        reopens = 0
        while not self._stopped and self._stream_session is not None:
            await self._consume_stream_events(self._stream_session)
            if self._stopped:
                return
            # events() returned while the call is live => the upstream dropped us.
            reopens += 1
            if reopens > _MAX_STREAM_REOPENS:
                log.warning("deepgram dropped repeatedly; falling back to batch VAD")
                self._stream_session = None
                return
            try:
                await self._stream_session.aclose()
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(_STREAM_REOPEN_BACKOFF_S)
            if self._stopped:
                return
            try:
                self._stream_session = await self._stream_provider.open_stream(
                    STTConfig(language="hi", sample_rate=self._config.pcm_sample_rate)
                )
                log.info("browser bridge: deepgram stream reopened after drop",
                         extra={"reopen": reopens})
            except Exception as e:  # noqa: BLE001
                log.warning("deepgram reopen failed (%s); falling back to batch VAD", e)
                self._stream_session = None
                return

    async def _consume_stream_events(self, session) -> None:
        """Drain streaming-STT events: partials -> console, endpoint -> dispatch.

        Wrapped so a consumer crash is logged rather than silently killing the
        background task (which would wedge the turn loop with no trace).
        """
        last_interim_t = None
        try:
            async for ev in session.events():
                if self._stopped:
                    return
                if ev.type == "interim":
                    last_interim_t = time.monotonic()
                    await self._send_json(
                        {"type": "partial", "role": "user", "text": ev.text}
                    )
                elif ev.type == "endpoint":
                    if self._agent_busy or not ev.text.strip():
                        last_interim_t = None
                        continue
                    gap_ms = (
                        int((time.monotonic() - last_interim_t) * 1000)
                        if last_interim_t is not None else None
                    )
                    last_interim_t = None
                    await self._dispatch_text_turn(ev.text, endpoint_gap_ms=gap_ms)
        except Exception:  # noqa: BLE001 - never let the consumer die silently
            log.exception("stream event consumer crashed")

    async def _dispatch_text_turn(self, text: str, endpoint_gap_ms: int | None = None) -> None:
        """Run one turn from an already-transcribed utterance (streaming path)."""
        self._agent_busy = True
        self._cancel_event = asyncio.Event()
        await self._send_json({"type": "status", "status": "thinking"})
        # Arm browser barge-in only now that a cancellable turn is in progress
        # (not during the opening — that lets the agent's greeting play fully and
        # gives the browser echo-canceller time to converge before the first turn).
        await self._send_json({"type": "barge", "armed": True})
        await self._send_json({"type": "partial", "role": "user", "text": ""})
        try:
            outcome = await self._agent.handle_turn_text(
                text, self._send_pcm, cancel_event=self._cancel_event
            )
        finally:
            self._cancel_event = None

        # Barge-in: the agent's reply was cancelled mid-utterance. Don't emit the
        # abandoned reply — return to listening; the interruption follows as the
        # next turn.
        if outcome.pipeline.cancelled:
            await self._emit_state()
            self._agent_busy = False
            await self._send_json({"type": "barge", "armed": False})
            await self._send_json({"type": "status", "status": "listening"})
            return

        # Only record the action for turns that actually completed (not barge-in
        # cancellations, where the action is a meaningless default).
        self._last_action = outcome.response.action
        m = outcome.pipeline.metrics
        log.info(
            "browser turn (stream)",
            extra={
                "user_text": (outcome.pipeline.user_text or "")[:80],
                "endpoint_gap_ms": endpoint_gap_ms,
                "llm_ttft_ms": m.llm_ttft_ms,
                "llm_total_ms": m.llm_total_ms,
                "tts_first_ms": m.tts_first_chunk_ms,
                "total_ms": m.total_latency_ms,
                "action": outcome.response.action,
                "agent_text": (outcome.response.response_text or "")[:100],
                "error": outcome.response.parse_error or "",
            },
        )
        if text:
            await self._send_json({"type": "transcript", "role": "user", "text": text})
        agent_text = outcome.response.response_text
        if agent_text:
            await self._send_json({"type": "transcript", "role": "agent", "text": agent_text})
        err = outcome.response.parse_error or ""
        if err and err != "empty STT":
            await self._send_json({"type": "error", "message": err})
        await self._emit_state()

        if getattr(self._agent.state, "is_terminal", False):
            self._stopped = True
            remaining = self._play_until - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining + 0.5)
            return
        self._agent_busy = False
        await self._send_json({"type": "barge", "armed": False})
        await self._send_json({"type": "status", "status": "listening"})

    def _apply_control(self, ctrl: dict) -> None:
        """Handle a client control message (called from the run() WS loop)."""
        if ctrl.get("type") == "config":
            self._barge_enabled = bool(ctrl.get("barge"))
        elif ctrl.get("type") == "barge_in":
            self._handle_barge_in()

    def _handle_barge_in(self) -> None:
        """Cancel the in-flight turn so the agent stops mid-utterance. Idempotent;
        no-op when the agent isn't speaking. Transport-agnostic — a future
        server-side (telephony) detector can call this same entry point."""
        # Fire whenever the agent is AUDIBLE — generating (_agent_busy) OR its
        # audio is still playing (now < _play_until). Most interruptions land
        # during playback, after generation finished and _agent_busy is False.
        if not (self._agent_busy or time.monotonic() < self._play_until):
            return
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._agent_busy = False
        # Playback was cut off by the interruption, so clear the playback gate —
        # otherwise the user's interrupting speech would be dropped as "echo"
        # until the cancelled reply's original end time.
        self._play_until = 0.0
        log.info("barge-in: cancelling current turn")

    async def _emit_outcome(self) -> None:
        """Analyze the finished call and push the outcome to the browser. Idempotent."""
        if self._outcome_emitted or self._llm is None or self._agent is None:
            return
        self._outcome_emitted = True
        try:
            transcript = [
                m for m in getattr(self._agent.session, "turns", [])
                if isinstance(m, LLMMessage)
            ]
            analysis = await analyze_call(
                transcript=transcript,
                slots=self._agent.slots.values,
                telephony_status=None,
                final_action=self._last_action,
                tenant_timezone=self._tenant_timezone,
                now=datetime.now(UTC),
                llm=self._llm,
            )
        except Exception:  # noqa: BLE001 - never let analysis break teardown
            log.exception("call outcome analysis failed")
            try:
                await self._send_json({"type": "error", "message": "outcome analysis failed"})
            except Exception:  # noqa: BLE001
                pass
            return
        cb = analysis.callback_datetime
        # Record the outcome server-side FIRST, so it survives even when the
        # socket is already gone (raw disconnect: tab close / crash / network
        # drop) and the push below cannot be delivered.
        log.info(
            "call outcome",
            extra={
                "outcome": analysis.outcome.value,
                "source": analysis.analysis_source,
                "summary": analysis.summary[:200],
                "callback": cb.isoformat() if cb else None,
            },
        )
        try:
            await self._send_json({
                "type": "outcome",
                "outcome": analysis.outcome.value,
                "summary": analysis.summary,
                "notes": analysis.notes,
                "callback_datetime": cb.isoformat() if cb else None,
                "callback_phrase": analysis.callback_phrase,
                "source": analysis.analysis_source,
            })
        except Exception:  # noqa: BLE001 - socket gone on raw disconnect; outcome already logged
            log.warning("outcome computed but not delivered (socket closed)")

    def _reset_capture(self) -> None:
        """Discard any audio buffered while the agent was busy, so the next
        listen starts clean (no greeting/echo/noise leading into the turn)."""
        self._capture_buffer.clear()
        self._inbound.clear()
        self._endpoint.reset()

    async def _emit_state(self) -> None:
        await self._send_json({
            "type": "state",
            "state": self._agent.state.state.value,
            "slots": dict(self._agent.slots.values),
        })
