# Browser Voice Dev Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dev-only browser page that talks to the agent over a local WebSocket (mic → STT → dialogue → TTS → speaker) with a transcript + state/slots panel, reusing the existing pipeline with no telephony cost.

**Architecture:** A new transport bridge (`BrowserVoiceBridge`) mirrors the Twilio/Exotel bridges but speaks a simpler protocol — binary PCM16 @16 kHz both directions plus JSON control/debug frames on one WebSocket. A tiny shared helper (`accumulate_and_detect`) carries the VAD→endpoint→capture core, used by both the new bridge and the existing Twilio bridge (behaviour-preserving). A flag-gated dev router serves a self-contained vanilla-JS page and runs the bridge.

**Tech Stack:** FastAPI (WebSocket + FileResponse), Starlette WebSocket, existing `VoiceBotAgent`/`PipelineEngine`/`EnergyVAD`/`EndpointDetector`, vanilla JS + Web Audio API (AudioWorklet via Blob URL). Tests: pytest + pytest-asyncio with fakes (no live audio APIs).

**Spec:** `docs/superpowers/specs/2026-06-02-browser-voice-dev-console-design.md`

---

## Reference: existing types this plan uses

These already exist — do not redefine them, import them.

- `from src.pipeline.vad import VADDetector, VADFrame, EnergyVAD, EndpointDetector, EndpointConfig`
  - `EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0)`; `.detect(pcm16: bytes) -> VADFrame` (has `.is_speech`); `.frame_ms`.
  - `EndpointDetector(frame_ms: int, cfg: EndpointConfig)`; `.feed(frame: VADFrame) -> bool` (True at end-of-utterance); `.reset()`.
- `from src.agents.voicebot import VoiceBotAgent, TurnOutcome`
  - `TurnOutcome` has `.response: VoiceBotResponse` and `.pipeline: TurnResult`.
  - Agent methods used: `await agent.start()`, `await agent.play_opening(sink)`, `await agent.handle_turn(captured: bytes, sink) -> TurnOutcome`, `await agent.handle_hangup()`.
  - Agent attributes used: `agent.state.state.value` (str), `agent.state.is_terminal` (bool), `agent.slots.values` (dict).
- `from src.dialogue.response_parser import VoiceBotResponse` — `.response_text: str`, `.action: str`.
- `from src.pipeline.engine import TurnResult, TurnMetrics` — `TurnResult.user_text: str`.
- `from src.auth.middleware import tenant_from_slug` — `async def tenant_from_slug(slug) -> TenantContext`, raises `HTTPException(404)` if unknown.
- `from src.bootstrap import make_bridge_factory` — reference pattern for `make_browser_bridge_factory`.
- `AudioSink = Callable[[bytes], Awaitable[None]]` (`from src.pipeline.engine import AudioSink`).
- App wiring: `api_router` (prefix `/api/v1`) in `src/api/__init__.py`; lifespan in `src/main.py` registers factories via `telephony_hooks.set_bridge_factory(...)`.

---

## Task 1: Shared capture helper `accumulate_and_detect`

Extract the three-line VAD→capture→endpoint core so the browser bridge and the Twilio bridge share one implementation. Pure function, trivially testable.

**Files:**
- Create: `src/pipeline/turn_capture.py`
- Test: `tests/unit/test_turn_capture.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_turn_capture.py
from __future__ import annotations

from src.pipeline.turn_capture import accumulate_and_detect
from src.pipeline.vad import EnergyVAD, EndpointDetector, EndpointConfig


def _loud(n_frames: int, vad: EnergyVAD) -> bytes:
    # Max-amplitude PCM16 => high RMS => is_speech True.
    return (b"\xff\x7f" * (vad.frame_bytes // 2)) * n_frames


def _silent(n_frames: int, vad: EnergyVAD) -> bytes:
    return (b"\x00\x00" * (vad.frame_bytes // 2)) * n_frames


def test_accumulates_pcm_into_buffer():
    vad = EnergyVAD(sample_rate=16000, frame_ms=30)
    endpoint = EndpointDetector(vad.frame_ms, EndpointConfig())
    buf = bytearray()
    pcm = _loud(1, vad)
    accumulate_and_detect(pcm, vad, endpoint, buf)
    assert bytes(buf) == pcm


def test_returns_true_at_end_of_utterance():
    vad = EnergyVAD(sample_rate=16000, frame_ms=30)
    # 250ms speech then 600ms silence => endpoint fires (defaults).
    endpoint = EndpointDetector(vad.frame_ms, EndpointConfig())
    buf = bytearray()
    # Feed speech frames one at a time; should not fire yet.
    fired = False
    for _ in range(10):  # 300ms speech
        fired = accumulate_and_detect(_loud(1, vad), vad, endpoint, buf) or fired
    assert fired is False
    # Now feed silence frames until it fires.
    for _ in range(25):  # up to 750ms silence
        if accumulate_and_detect(_silent(1, vad), vad, endpoint, buf):
            fired = True
            break
    assert fired is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_turn_capture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline.turn_capture'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/pipeline/turn_capture.py
"""Shared per-turn audio capture core for media bridges.

Each transport bridge (Twilio, Exotel, browser) feeds inbound PCM16 frames
through the same VAD + endpoint-detection loop. This helper is that loop, so
the logic lives in one place rather than being copied per transport.
"""

from __future__ import annotations

from src.pipeline.vad import EndpointDetector, VADDetector


def accumulate_and_detect(
    pcm16: bytes,
    vad: VADDetector,
    endpoint: EndpointDetector,
    capture_buffer: bytearray,
) -> bool:
    """Append a PCM16 frame to the capture buffer and run endpointing.

    Returns True once the endpoint detector reports end-of-utterance, i.e.
    the caller should dispatch the buffered audio as a completed turn.
    """
    capture_buffer.extend(pcm16)
    frame = vad.detect(pcm16)
    return endpoint.feed(frame)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_turn_capture.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/turn_capture.py tests/unit/test_turn_capture.py
git commit -m "add shared accumulate_and_detect turn-capture helper"
```

---

## Task 2: `BrowserBridgeConfig` + `BrowserVoiceBridge` construction & handshake

The bridge object: holds the websocket, agent, VAD, endpoint, capture buffer. This task covers construction and the JSON-emit helpers only (turn loop comes in Task 3).

**Files:**
- Create: `src/api/browser_bridge.py`
- Test: `tests/unit/test_browser_bridge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_browser_bridge.py
from __future__ import annotations

import json

import pytest

from src.api.browser_bridge import BrowserBridgeConfig, BrowserVoiceBridge
from src.pipeline.vad import EnergyVAD


class FakeWebSocket:
    """Scripted Starlette-style websocket for bridge tests."""

    def __init__(self, incoming: list[dict]):
        self._incoming = list(incoming)
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.closed = False

    async def accept(self) -> None:
        pass

    async def receive(self) -> dict:
        if self._incoming:
            return self._incoming.pop(0)
        return {"type": "websocket.disconnect", "code": 1000}

    async def send_text(self, data: str) -> None:
        self.sent_text.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self, code: int = 1000) -> None:
        self.closed = True


def _bridge(ws, agent=None):
    return BrowserVoiceBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30),
        config=BrowserBridgeConfig(),
    )


@pytest.mark.asyncio
async def test_send_json_emits_text_frame():
    ws = FakeWebSocket([])
    bridge = _bridge(ws)
    await bridge._send_json({"type": "status", "status": "listening"})
    assert json.loads(ws.sent_text[0]) == {"type": "status", "status": "listening"}


@pytest.mark.asyncio
async def test_send_pcm_writes_binary_frames():
    ws = FakeWebSocket([])
    bridge = _bridge(ws)
    await bridge._send_pcm(b"\x01\x02\x03\x04")
    # status:speaking, then the audio bytes, then status:listening
    assert b"".join(ws.sent_bytes) == b"\x01\x02\x03\x04"
    statuses = [json.loads(t).get("status") for t in ws.sent_text]
    assert "speaking" in statuses and statuses[-1] == "listening"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_browser_bridge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.api.browser_bridge'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_browser_bridge.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/api/browser_bridge.py tests/unit/test_browser_bridge.py
git commit -m "add BrowserVoiceBridge construction + binary/json send helpers"
```

---

## Task 3: `BrowserVoiceBridge.run()` — handshake, opening, turn loop, debug events

The heart: read the `hello` frame, play the opening, then loop on binary PCM frames running the shared capture core; on end-of-utterance call `handle_turn` and emit transcript/state/status events.

**Files:**
- Modify: `src/api/browser_bridge.py`
- Test: `tests/unit/test_browser_bridge.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_browser_bridge.py
from src.agents.voicebot import TurnOutcome
from src.dialogue.response_parser import VoiceBotResponse
from src.pipeline.engine import TurnResult, TurnMetrics


class _FakeState:
    def __init__(self):
        self.state = type("S", (), {"value": "qualifying"})()
        self.is_terminal = False


class _FakeSlots:
    values = {"interested": True}


class FakeAgent:
    """Minimal agent matching the surface BrowserVoiceBridge touches."""

    def __init__(self):
        self.state = _FakeState()
        self.slots = _FakeSlots()
        self.started = False
        self.opening_played = False
        self.hung_up = False
        self.turns: list[bytes] = []

    async def start(self):
        self.started = True

    async def play_opening(self, sink):
        self.opening_played = True
        await sink(b"\x10\x11")  # fake opening audio

    async def handle_turn(self, captured: bytes, sink) -> TurnOutcome:
        self.turns.append(captured)
        await sink(b"\x20\x21")  # fake reply audio
        return TurnOutcome(
            response=VoiceBotResponse(response_text="Theek hai", action="continue"),
            pipeline=TurnResult(
                user_text="Namaste",
                user_language="hi",
                user_confidence=1.0,
                agent_text='{"response":"Theek hai"}',
                audio_bytes_sent=2,
                metrics=TurnMetrics(),
            ),
        )

    async def handle_hangup(self):
        self.hung_up = True


def _loud_frame(vad) -> bytes:
    return b"\xff\x7f" * (vad.frame_bytes // 2)


def _silent_frame(vad) -> bytes:
    return b"\x00\x00" * (vad.frame_bytes // 2)


@pytest.mark.asyncio
async def test_run_handshake_plays_opening_and_processes_a_turn():
    vad = EnergyVAD(sample_rate=16000, frame_ms=30)
    # hello, then 10 speech frames (300ms) + 25 silence frames (750ms) => one turn.
    incoming = [{"type": "websocket.receive", "text": json.dumps({"type": "hello", "tenant": "dev"})}]
    for _ in range(10):
        incoming.append({"type": "websocket.receive", "bytes": _loud_frame(vad)})
    for _ in range(25):
        incoming.append({"type": "websocket.receive", "bytes": _silent_frame(vad)})
    ws = FakeWebSocket(incoming)
    agent = FakeAgent()
    bridge = BrowserVoiceBridge(websocket=ws, agent=agent, vad=vad, config=BrowserBridgeConfig())

    await bridge.run()

    assert agent.started and agent.opening_played and agent.hung_up
    assert len(agent.turns) == 1 and len(agent.turns[0]) > 0  # captured audio dispatched
    events = [json.loads(t) for t in ws.sent_text]
    transcripts = [(e["role"], e["text"]) for e in events if e["type"] == "transcript"]
    assert ("user", "Namaste") in transcripts
    assert ("agent", "Theek hai") in transcripts
    states = [e for e in events if e["type"] == "state"]
    assert states and states[-1]["state"] == "qualifying"
    assert states[-1]["slots"] == {"interested": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_browser_bridge.py::test_run_handshake_plays_opening_and_processes_a_turn -v`
Expected: FAIL — `AttributeError: 'BrowserVoiceBridge' object has no attribute 'run'`

- [ ] **Step 3: Write minimal implementation**

Append these methods to `BrowserVoiceBridge` in `src/api/browser_bridge.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_browser_bridge.py -v`
Expected: PASS (all browser-bridge tests)

- [ ] **Step 5: Commit**

```bash
git add src/api/browser_bridge.py tests/unit/test_browser_bridge.py
git commit -m "implement BrowserVoiceBridge run loop with transcript + state events"
```

---

## Task 4: Point the Twilio bridge at the shared helper (behaviour-preserving)

Replace Twilio's inline VAD/capture/endpoint lines with `accumulate_and_detect`, leaving the μ-law decode, resample, and idle-silence logic exactly as they are. Existing Twilio bridge tests must stay green — that is the regression guard.

**Files:**
- Modify: `src/api/telephony_twilio.py:135-162` (the body of `_on_media_frame`)

- [ ] **Step 1: Run the existing Twilio bridge tests to establish the baseline**

Run: `.venv/bin/python -m pytest tests/unit/ -k "twilio" -v`
Expected: PASS (record the count; it must not change after the edit)

- [ ] **Step 2: Make the behaviour-preserving edit**

In `src/api/telephony_twilio.py`, add the import near the other `src.pipeline` imports:

```python
from src.pipeline.turn_capture import accumulate_and_detect
```

Then in `_on_media_frame`, replace this block:

```python
        self._capture_buffer.extend(pcm)

        # VAD on this chunk.
        frame = self._vad.detect(pcm)
        if frame.is_speech:
            self._idle_silence_ms = 0
        else:
            self._idle_silence_ms += self._vad.frame_ms

        # Extended silence -> hang up gracefully.
        if self._idle_silence_ms >= self._config.max_idle_silence_s * 1000:
            await self._agent.handle_extended_silence()
            self._stopped.set()
            return

        # Endpoint reached -> dispatch the buffered utterance to the agent.
        if self._endpoint.feed(frame):
            await self._dispatch_utterance()
```

with:

```python
        # VAD on this chunk (for the idle-silence timer) + shared capture core.
        frame = self._vad.detect(pcm)
        if frame.is_speech:
            self._idle_silence_ms = 0
        else:
            self._idle_silence_ms += self._vad.frame_ms

        # Extended silence -> hang up gracefully.
        if self._idle_silence_ms >= self._config.max_idle_silence_s * 1000:
            self._capture_buffer.extend(pcm)
            await self._agent.handle_extended_silence()
            self._stopped.set()
            return

        # Accumulate + endpoint via the shared helper. Re-run detect inside the
        # helper is cheap and keeps a single capture path across transports.
        if accumulate_and_detect(pcm, self._vad, self._endpoint, self._capture_buffer):
            await self._dispatch_utterance()
```

Note: `accumulate_and_detect` extends the buffer itself, so the inline `self._capture_buffer.extend(pcm)` is removed from the normal path (kept only on the early-return idle branch so trailing audio isn't lost). The double `vad.detect` (once here for the idle timer, once in the helper) is intentional and negligible.

- [ ] **Step 3: Run the Twilio bridge tests to verify unchanged behaviour**

Run: `.venv/bin/python -m pytest tests/unit/ -k "twilio" -v`
Expected: PASS — same count as Step 1

- [ ] **Step 4: Commit**

```bash
git add src/api/telephony_twilio.py
git commit -m "twilio bridge: use shared accumulate_and_detect capture helper"
```

---

## Task 5: Dev-console router, browser bridge factory, and flag-gated wiring

A router that serves the page (`GET /dev/voice`) and runs the bridge (`WS /api/v1/dev/voice`), plus a factory mirroring `make_bridge_factory`, mounted only when `VOX_DEV_CONSOLE=1`.

**Files:**
- Create: `src/api/dev_console.py`
- Modify: `src/main.py` (lifespan: register factory + include router under the flag)
- Test: `tests/unit/test_dev_console.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_dev_console.py
from __future__ import annotations

from src.api.dev_console import dev_console_enabled


def test_dev_console_enabled_flag(monkeypatch):
    monkeypatch.delenv("VOX_DEV_CONSOLE", raising=False)
    assert dev_console_enabled() is False
    monkeypatch.setenv("VOX_DEV_CONSOLE", "1")
    assert dev_console_enabled() is True
```

(The page-served test is added in Task 6, once the static file exists, so every commit stays green.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_dev_console.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.api.dev_console'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/api/dev_console.py
"""Dev-only browser voice console (gated by VOX_DEV_CONSOLE=1).

Serves a self-contained page at ``GET /dev/voice`` and runs a
``BrowserVoiceBridge`` at ``WS /api/v1/dev/voice``. Reuses the tenant's
provider stack exactly like the telephony bridges; intended for local
dialogue-management iteration with no telephony cost.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine
from src.agents.voicebot import VoiceBotAgent
from src.api.browser_bridge import BrowserBridgeConfig, BrowserVoiceBridge
from src.auth.context import TenantContext
from src.auth.registry import TenantProviders
from src.bootstrap import DEFAULT_DEMO_SCRIPT
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.interfaces.llm import LLMConfig
from src.interfaces.stt import STTConfig
from src.interfaces.tts import TTSConfig
from src.pipeline.engine import PipelineConfig, PipelineEngine
from src.pipeline.vad import EnergyVAD

log = logging.getLogger(__name__)

_STATIC = Path(__file__).resolve().parents[2] / "static"

# WS path lives under the /api/v1 router; the page route is top-level.
ws_router = APIRouter(prefix="/dev", tags=["dev-console"])   # mounted under /api/v1
dev_router = APIRouter(tags=["dev-console"])                  # mounted at app root

# Factory: (websocket, tenant) -> BrowserVoiceBridge. Set during lifespan.
BrowserBridgeFactory = Callable[[WebSocket, TenantContext], BrowserVoiceBridge]
_browser_bridge_factory: Optional[BrowserBridgeFactory] = None


def dev_console_enabled() -> bool:
    return os.environ.get("VOX_DEV_CONSOLE", "") == "1"


def set_browser_bridge_factory(factory: Optional[BrowserBridgeFactory]) -> None:
    global _browser_bridge_factory
    _browser_bridge_factory = factory


@dev_router.get("/dev/voice")
async def dev_voice_page() -> FileResponse:
    return FileResponse(_STATIC / "dev_console.html", media_type="text/html")


@ws_router.websocket("/voice")
async def dev_voice_ws(websocket: WebSocket) -> None:
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    if _browser_bridge_factory is None:
        await websocket.close(code=1011, reason="browser bridge factory unset")
        return
    try:
        tenant = await tenant_from_slug(
            websocket.query_params.get("tenant", "dev")
        )
    except Exception as e:  # noqa: BLE001
        log.warning("dev console tenant resolution failed: %s", e)
        await websocket.close(code=1008, reason="unknown tenant")
        return

    bridge = _browser_bridge_factory(websocket, tenant)
    try:
        await bridge.run()
    except WebSocketDisconnect:
        log.info("dev console client disconnected", extra={"tenant": tenant.slug})
    except Exception:  # noqa: BLE001
        log.exception("dev console bridge crashed", extra={"tenant": tenant.slug})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


def make_browser_bridge_factory(
    providers: TenantProviders,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
) -> BrowserBridgeFactory:
    """Build a BrowserVoiceBridge per connection, wired to the tenant stack.

    Mirrors ``src.bootstrap.make_bridge_factory`` but returns a browser bridge.
    """

    def factory(websocket: WebSocket, tenant: TenantContext) -> BrowserVoiceBridge:
        import uuid

        stt = providers.get_stt(tenant)
        llm = providers.get_llm(tenant)
        tts = providers.get_tts(tenant)
        pipeline_cfg = PipelineConfig(
            stt=STTConfig(language=tenant.settings.pipeline.stt.language or "hi-IN"),
            llm=LLMConfig(
                temperature=tenant.settings.pipeline.llm.temperature or 0.5,
                max_tokens=tenant.settings.pipeline.llm.max_tokens or 256,
                response_format=tenant.settings.pipeline.llm.response_format or "json",
            ),
            tts=TTSConfig(
                language=tenant.settings.pipeline.tts.language or "hi-IN",
                voice_id=tenant.settings.pipeline.tts.voice_id,
                sample_rate=16000,
            ),
        )
        engine = PipelineEngine(stt, llm, tts, pipeline_cfg)
        session_id = f"web_{uuid.uuid4().hex[:12]}"
        agent = VoiceBotAgent(
            session=AgentSession(session_id=session_id),
            state_machine=AgentStateMachine(),
            slot_schema=SlotSchema(),
            script=script,
            engine=engine,
            store=None,
        )
        log.info("dev console built call", extra={"tenant": tenant.slug, "session_id": session_id})
        return BrowserVoiceBridge(
            websocket=websocket,
            agent=agent,
            vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
            config=BrowserBridgeConfig(),
        )

    return factory
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_dev_console.py -v`
Expected: PASS (1 passed — `test_dev_console_enabled_flag`)

- [ ] **Step 5: Wire it into the app lifespan (flag-gated)**

In `src/main.py`, add imports near the other `src.api` / `src.bootstrap` imports:

```python
from src.api.dev_console import (
    dev_console_enabled,
    dev_router,
    ws_router as dev_ws_router,
    make_browser_bridge_factory,
    set_browser_bridge_factory,
)
```

In `lifespan`, immediately after the existing `telephony_hooks.set_exotel_bridge_factory(...)` call (around line 104), add:

```python
    if dev_console_enabled():
        set_browser_bridge_factory(make_browser_bridge_factory(providers=providers))
        log.info("dev console enabled at /dev/voice")
```

In the `finally:` block of `lifespan`, after the existing `set_..._bridge_factory(None)` calls, add:

```python
        set_browser_bridge_factory(None)
```

After `app.include_router(api_router)` (around line 125), add the flag-gated routers:

```python
if dev_console_enabled():
    app.include_router(dev_router)              # GET /dev/voice
    api_router.include_router(dev_ws_router)    # WS  /api/v1/dev/voice
```

Note: `api_router.include_router(dev_ws_router)` must run before `app.include_router(api_router)` for the WS route to register. Move the `dev_console_enabled()` block ABOVE the existing `app.include_router(api_router)` line, like so:

```python
if dev_console_enabled():
    api_router.include_router(dev_ws_router)    # WS  /api/v1/dev/voice

app.include_router(api_router)

if dev_console_enabled():
    app.include_router(dev_router)              # GET /dev/voice
```

- [ ] **Step 6: Verify the app imports cleanly with the flag set**

Run: `VOX_DEV_CONSOLE=1 .venv/bin/python -c "import src.main; print('ok')"`
Expected: prints `ok` (no import errors)

- [ ] **Step 7: Commit**

```bash
git add src/api/dev_console.py src/main.py tests/unit/test_dev_console.py
git commit -m "add flag-gated dev console router + browser bridge factory wiring"
```

---

## Task 6: The browser page `static/dev_console.html`

Self-contained vanilla-JS page: mic capture → 16 kHz PCM16 over WS, binary playback, transcript + state/slots panel, half-duplex mic gating.

**Files:**
- Create: `static/dev_console.html`
- Modify: `tests/unit/test_dev_console.py` (add the page-served test)

- [ ] **Step 1: Add the failing page-served test**

```python
# append to tests/unit/test_dev_console.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dev_console import dev_router


def test_dev_voice_page_served():
    app = FastAPI()
    app.include_router(dev_router)
    client = TestClient(app)
    resp = client.get("/dev/voice")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Voice Dev Console" in resp.text
```

Run: `.venv/bin/python -m pytest tests/unit/test_dev_console.py::test_dev_voice_page_served -v`
Expected: FAIL — `RuntimeError`/`404`/file-not-found, because `static/dev_console.html` does not exist yet.

- [ ] **Step 2: Create the page**

```html
<!-- static/dev_console.html -->
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Voice Dev Console</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem; background:#0f172a; color:#e2e8f0; }
  h1 { font-size: 1.2rem; }
  .row { display:flex; gap:1rem; align-items:center; margin-bottom:1rem; flex-wrap:wrap; }
  button { padding:.5rem 1rem; border-radius:.4rem; border:0; cursor:pointer; font-weight:600; }
  #start { background:#22c55e; color:#06210f; }
  #start.stop { background:#ef4444; color:#2a0606; }
  #status { font-weight:600; }
  .panes { display:flex; gap:1rem; flex-wrap:wrap; }
  .pane { flex:1 1 320px; background:#1e293b; border-radius:.5rem; padding:.75rem; min-height:240px; }
  .pane h2 { font-size:.9rem; margin:.2rem 0 .6rem; color:#94a3b8; }
  .turn { margin:.3rem 0; }
  .turn.user { color:#7dd3fc; }
  .turn.agent { color:#86efac; }
  pre { white-space:pre-wrap; word-break:break-word; margin:0; font-size:.85rem; }
  .hint { color:#94a3b8; font-size:.8rem; }
  label { font-size:.85rem; }
</style>
</head>
<body>
  <h1>Voice Dev Console <span class="hint">— use headphones to avoid echo</span></h1>
  <div class="row">
    <button id="start">Start</button>
    <label><input type="checkbox" id="halfDuplex" checked /> Mute mic while agent speaks</label>
    <span id="status">idle</span>
  </div>
  <div class="panes">
    <div class="pane">
      <h2>Transcript</h2>
      <div id="transcript"></div>
    </div>
    <div class="pane">
      <h2>Dialogue state &amp; slots</h2>
      <pre id="state">—</pre>
    </div>
  </div>

<script>
const TARGET_RATE = 16000;
let ws, audioCtx, micStream, workletNode, playCursor = 0, agentSpeaking = false, running = false;

const $ = (id) => document.getElementById(id);
const setStatus = (s) => { $("status").textContent = s; };

function addTranscript(role, text) {
  const div = document.createElement("div");
  div.className = "turn " + role;
  div.textContent = (role === "user" ? "🧑 " : "🤖 ") + text;
  $("transcript").appendChild(div);
  $("transcript").scrollTop = $("transcript").scrollHeight;
}

// Downsample Float32 @ctxRate -> Int16 @16k via linear interpolation.
function downsampleToInt16(float32, inRate) {
  if (inRate === TARGET_RATE) {
    const out = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }
  const ratio = inRate / TARGET_RATE;
  const outLen = Math.floor(float32.length / ratio);
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const s = Math.max(-1, Math.min(1, float32[Math.floor(i * ratio)]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

// Schedule Int16 PCM @16k for gapless playback.
function playPcm16(int16) {
  const buf = audioCtx.createBuffer(1, int16.length, TARGET_RATE);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < int16.length; i++) ch[i] = int16[i] / 0x8000;
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  const now = audioCtx.currentTime;
  if (playCursor < now) playCursor = now;
  src.start(playCursor);
  playCursor += buf.duration;
}

const WORKLET_CODE = `
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) this.port.postMessage(input[0].slice(0));
    return true;
  }
}
registerProcessor('capture-processor', CaptureProcessor);
`;

async function start() {
  ws = new WebSocket(`ws://${location.host}/api/v1/dev/voice?tenant=dev`);
  ws.binaryType = "arraybuffer";

  ws.onopen = async () => {
    ws.send(JSON.stringify({ type: "hello", tenant: "dev" }));
    audioCtx = new AudioContext();
    playCursor = audioCtx.currentTime;
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
    const blobUrl = URL.createObjectURL(new Blob([WORKLET_CODE], { type: "application/javascript" }));
    await audioCtx.audioWorklet.addModule(blobUrl);
    const srcNode = audioCtx.createMediaStreamSource(micStream);
    workletNode = new AudioWorkletNode(audioCtx, "capture-processor");
    workletNode.port.onmessage = (e) => {
      if (!running) return;
      if ($("halfDuplex").checked && agentSpeaking) return;  // half-duplex gate
      const int16 = downsampleToInt16(e.data, audioCtx.sampleRate);
      if (ws.readyState === WebSocket.OPEN) ws.send(int16.buffer);
    };
    srcNode.connect(workletNode);
    // Worklet needs a sink to pull; route through a muted gain so we don't echo.
    const sink = audioCtx.createGain();
    sink.gain.value = 0;
    workletNode.connect(sink).connect(audioCtx.destination);
    running = true;
    setStatus("connected");
  };

  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) {
      playPcm16(new Int16Array(e.data));
      return;
    }
    const msg = JSON.parse(e.data);
    if (msg.type === "status") {
      setStatus(msg.status);
      if (msg.status === "speaking") agentSpeaking = true;
      if (msg.status === "listening") agentSpeaking = false;
    } else if (msg.type === "transcript") {
      addTranscript(msg.role, msg.text);
    } else if (msg.type === "state") {
      $("state").textContent =
        "state: " + msg.state + "\nslots: " + JSON.stringify(msg.slots, null, 2);
    }
  };

  ws.onclose = () => { stop(); setStatus("closed"); };
}

function stop() {
  running = false;
  if (workletNode) { try { workletNode.disconnect(); } catch (_) {} }
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();
  $("start").classList.remove("stop");
  $("start").textContent = "Start";
}

$("start").onclick = () => {
  if (running) { stop(); return; }
  $("start").classList.add("stop");
  $("start").textContent = "Stop";
  start();
};
</script>
</body>
</html>
```

- [ ] **Step 3: Run the dev-console tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_dev_console.py -v`
Expected: PASS (2 passed — the page is now served)

- [ ] **Step 4: Commit**

```bash
git add static/dev_console.html tests/unit/test_dev_console.py
git commit -m "add browser voice dev-console page (mic capture + playback + debug panel)"
```

---

## Task 7: Full suite + manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: PASS (all green, including the new turn-capture, browser-bridge, and dev-console tests)

- [ ] **Step 2: Launch the app with the console enabled**

Run (background): `VOX_DEV_CONSOLE=1 .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8765 --env-file .env`
Then: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8765/dev/voice`
Expected: `200`

- [ ] **Step 3: Manual browser check**

Open `http://localhost:8765/dev/voice` in Chrome, click **Start**, grant mic permission (wear headphones).
Expected: hear the agent's opening line; the state panel shows the initial state. Speak a Hindi sentence and pause; within a couple of seconds a user transcript appears, the agent replies audibly, an agent transcript appears, and the state/slots panel updates — all with no phone call placed.

- [ ] **Step 4: Confirm prod-safety (console off by default)**

Run: `.venv/bin/python -c "import os; os.environ.pop('VOX_DEV_CONSOLE', None); import src.main as m; routes=[getattr(r,'path','') for r in m.app.routes]; assert '/dev/voice' not in routes, routes; print('console correctly absent when flag unset')"`
Expected: prints `console correctly absent when flag unset`

- [ ] **Step 5: Final commit (if any verification fixes were needed)**

```bash
git add -A
git commit -m "verify browser voice dev console end-to-end" --allow-empty
```

---

## Notes for the implementer

- **Run everything with the venv python** (`.venv/bin/python`), not the conda base — `twilio`, `httpx`, `respx`, `fastapi`, etc. live in `.venv`.
- **The dialogue logic is the point of this tool, not the tool itself.** Keep `BrowserVoiceBridge` thin — all conversation behaviour stays in `VoiceBotAgent`.
- **Echo:** if the agent transcribes itself during manual testing, confirm the half-duplex checkbox is on and use headphones. `echoCancellation` varies by browser/OS.
- **Sample rate:** the bridge assumes the browser sends 16 kHz PCM16 (the page resamples). The pipeline + VAD are all 16 kHz, so there is no server-side resample.
