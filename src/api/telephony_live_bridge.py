"""Telephony speech-to-speech bridge — Twilio + Exotel media streams.

Same dialogue core as the dev-console S2S path (``_BaseLiveBridge``); this speaks
the phone-call media-stream protocol instead of the browser one. Twilio and
Exotel are the same shape (JSON ``connected/start/media/stop`` over a WS),
differing only in audio encoding (Twilio 8kHz μ-law vs Exotel raw 8kHz PCM), the
stream-id field name, and barge-in (Twilio has a ``clear`` frame). One class,
parameterized by ``encoding`` + ``sid_field``.

Outbound audio is **real-time paced** off the events loop via a sender queue, so
pacing never blocks reading the next Live event (keeps native barge-in snappy).
No opening line — on a real call the callee says "hello" first and the model
replies (matches the dev-console "user speaks first" flow).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time

from src.api import dev_call_control
from src.api.live_bridge_base import _BaseLiveBridge
from src.interfaces.realtime import RealtimeConfig
from src.pipeline.audio_utils import mulaw_to_pcm16, pcm16_to_mulaw, resample_pcm16

log = logging.getLogger(__name__)

_TEL_RATE = 8000        # telephony is 8kHz mono
_FRAME_S = 0.02         # 20ms per media frame
# 20ms @ 8kHz mono: 160 bytes μ-law (1 B/sample) or 320 bytes PCM16 (2 B/sample).
_CHUNK = {"mulaw": 160, "pcm": 320}


class TelephonyLiveBridge(_BaseLiveBridge):
    """One bridge per phone call. ``encoding``: 'mulaw' (Twilio) | 'pcm' (Exotel)."""

    def __init__(self, *, websocket, agent, config: RealtimeConfig, connect_session,
                 llm=None, tenant_timezone: str = "Asia/Kolkata",
                 encoding: str = "mulaw", sid_field: str = "streamSid",
                 supports_clear: bool = True, call_sid_field: str = "callSid") -> None:
        super().__init__(agent=agent, config=config, connect_session=connect_session,
                         llm=llm, tenant_timezone=tenant_timezone)
        self._ws = websocket
        self._encoding = encoding
        self._sid_field = sid_field
        self._supports_clear = supports_clear
        self._call_sid_field = call_sid_field   # Twilio: "callSid" / Exotel: "call_sid"
        self._stream_sid: str | None = None
        self._call_sid: str | None = None        # provider Call SID (dev-console monitor key)
        self._up_state = None        # 8k->16k resample state (inbound)
        self._down_state = None      # 24k->8k resample state (outbound)
        self._audio_q: asyncio.Queue[bytes] = asyncio.Queue()
        self._sender_task: asyncio.Task | None = None
        self._play_deadline = 0.0
        self._in_frames = 0          # caller media frames forwarded to the model
        self._out_frames = 0         # model media frames queued for the caller

    async def run(self) -> None:
        await self._drive()

    # --- transport hooks ---
    async def _on_start(self) -> None:
        self._sender_task = asyncio.create_task(self._sender_loop())

    async def _on_teardown(self) -> None:
        if self._sender_task is not None:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except BaseException:  # noqa: BLE001
                pass
        if self._call_sid is not None:
            dev_call_control.monitor.set_status(self._call_sid, "ended")

    async def _deliver_outcome(self, payload: dict) -> None:
        # Publish to the dev-console call monitor so a placed call shows its outcome.
        if self._call_sid is not None:
            dev_call_control.monitor.set_outcome(self._call_sid, payload)
        # Persist to the conversations row (keyed by provider Call SID), if a
        # persister is wired. No-op for the dev console / tests without a DB.
        from src.api import call_store
        await call_store.deliver_to_persister(self._call_sid, payload)

    async def _inbound_loop(self) -> None:
        from starlette.websockets import WebSocketDisconnect
        try:
            while not self._stopped:
                raw = await self._ws.receive_text()
                msg = json.loads(raw)
                event = msg.get("event")
                if event == "connected":
                    continue
                if event == "start":
                    start = msg.get("start", {}) or {}
                    self._stream_sid = start.get(self._sid_field) or msg.get(self._sid_field)
                    self._call_sid = (start.get(self._call_sid_field)
                                      or msg.get(self._call_sid_field))
                    log.info("telephony stream started",
                             extra={"sid": self._stream_sid, "call_sid": self._call_sid})
                    if self._call_sid is not None:
                        dev_call_control.monitor.set_status(self._call_sid, "answered")
                elif event == "media":
                    await self._on_media(msg.get("media") or {})
                elif event == "stop":
                    break
        except WebSocketDisconnect:
            pass  # caller hung up — normal end
        finally:
            self._stopped = True

    async def _on_media(self, media: dict) -> None:
        if media.get("track") not in (None, "inbound"):
            return
        payload = media.get("payload")
        if not payload or self._session is None:
            return
        raw = base64.b64decode(payload)
        pcm8k = mulaw_to_pcm16(raw) if self._encoding == "mulaw" else raw
        pcm16k, self._up_state = resample_pcm16(pcm8k, _TEL_RATE, 16000, self._up_state)
        await self._session.send_audio(pcm16k)   # caller audio -> model
        self._in_frames += 1
        if self._in_frames % 250 == 0:           # ~every 5s of caller audio
            log.info("telephony caller audio -> model", extra={"in_frames": self._in_frames})

    async def _send_audio_out(self, pcm16: bytes, rate: int) -> None:
        # Enqueue; the sender task paces it out (never blocks the events loop).
        if pcm16 and self._stream_sid is not None:
            pcm8k, self._down_state = resample_pcm16(pcm16, rate, _TEL_RATE, self._down_state)
            self._audio_q.put_nowait(pcm8k)
            self._out_frames += 1
            if self._out_frames % 50 == 1:       # first chunk, then ~periodic
                log.info("telephony model audio -> caller", extra={"out_frames": self._out_frames})

    async def _send_interrupt(self) -> None:
        # Barge-in: drop queued+playing agent audio and reset pacing.
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._play_deadline = 0.0
        if self._supports_clear and self._stream_sid is not None:
            await self._ws.send_text(json.dumps(
                {"event": "clear", self._sid_field: self._stream_sid}))

    async def _sender_loop(self) -> None:
        chunk = _CHUNK[self._encoding]
        try:
            while True:
                pcm8k = await self._audio_q.get()
                out = pcm16_to_mulaw(pcm8k) if self._encoding == "mulaw" else pcm8k
                now = time.perf_counter()
                if self._play_deadline < now:
                    self._play_deadline = now
                for i in range(0, len(out), chunk):
                    piece = out[i:i + chunk]
                    await self._ws.send_text(json.dumps({
                        "event": "media", self._sid_field: self._stream_sid,
                        "media": {"payload": base64.b64encode(piece).decode("ascii")}}))
                    self._play_deadline += _FRAME_S
                    slack = self._play_deadline - time.perf_counter()
                    if slack > 0:
                        await asyncio.sleep(slack)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - WS closed mid-send (teardown race); stop quietly
            self._stopped = True
