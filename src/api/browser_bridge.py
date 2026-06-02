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

import json
import logging
from dataclasses import dataclass, field

from src.pipeline.turn_capture import accumulate_and_detect
from src.pipeline.vad import EndpointConfig, EndpointDetector, VADDetector

log = logging.getLogger(__name__)

# Browser captures and resamples to the internal pipeline rate directly.
BROWSER_SAMPLE_RATE = 16000

# Chunk size for outbound PCM frames (bytes). 8 KB ~= 256 ms @16 kHz PCM16.
_SEND_CHUNK = 8192


@dataclass
class BrowserBridgeConfig:
    pcm_sample_rate: int = BROWSER_SAMPLE_RATE
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    default_tenant: str = "dev"


class BrowserVoiceBridge:
    """One bridge per browser connection. Drive with ``run()``."""

    def __init__(self, websocket, agent, vad: VADDetector, config: BrowserBridgeConfig | None = None):
        self._ws = websocket
        self._agent = agent
        self._vad = vad
        self._config = config or BrowserBridgeConfig()
        self._capture_buffer = bytearray()
        self._endpoint = EndpointDetector(vad.frame_ms, self._config.endpoint)
        self._stopped = False

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
        await self._send_json({"type": "status", "status": "listening"})

    # --- entrypoint ---------------------------------------------------

    async def run(self) -> None:
        """Drive the connection until the browser disconnects or the agent ends."""
        await self._agent.start()
        try:
            # 1) Handshake: first text frame selects the tenant (already resolved
            #    by the caller, so we just consume it). Then play the opening.
            await self._read_hello()
            await self._play_opening()

            # 2) Turn loop.
            while not self._stopped:
                message = await self._ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is not None:
                    await self._on_pcm_frame(data)
                    continue
                text = message.get("text")
                if text is not None:
                    # Forward-compat: ignore unknown control frames.
                    continue
        finally:
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
        await self._agent.play_opening(self._send_pcm)
        await self._emit_state()
        await self._send_json({"type": "status", "status": "listening"})

    # --- inbound ------------------------------------------------------

    async def _on_pcm_frame(self, pcm16: bytes) -> None:
        if accumulate_and_detect(pcm16, self._vad, self._endpoint, self._capture_buffer):
            await self._dispatch_utterance()

    async def _dispatch_utterance(self) -> None:
        captured = bytes(self._capture_buffer)
        self._capture_buffer.clear()
        self._endpoint.reset()
        await self._send_json({"type": "status", "status": "thinking"})
        outcome = await self._agent.handle_turn(captured, self._send_pcm)

        user_text = outcome.pipeline.user_text
        if user_text:
            await self._send_json({"type": "transcript", "role": "user", "text": user_text})
        agent_text = outcome.response.response_text
        if agent_text:
            await self._send_json({"type": "transcript", "role": "agent", "text": agent_text})
        await self._emit_state()

        if getattr(self._agent.state, "is_terminal", False):
            self._stopped = True
            return
        await self._send_json({"type": "status", "status": "listening"})

    async def _emit_state(self) -> None:
        await self._send_json({
            "type": "state",
            "state": self._agent.state.state.value,
            "slots": dict(self._agent.slots.values),
        })
