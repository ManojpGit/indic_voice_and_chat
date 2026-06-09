# Stringee Turn-Based IVR Voicebot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stringee a first-class per-tenant telephony provider for the live voicebot using Stringee's native SCCO IVR primitives (record → webhook → reply), reusing the agent's existing batch `handle_turn`.

**Architecture:** On call answer Stringee fetches an SCCO from our `answer_url` that plays the opening and starts a `recordMessage`. When the caller finishes (silence-detected), Stringee POSTs the utterance's audio link to our `event` webhook; we download it, run the existing batch `handle_turn` (STT→LLM→TTS), host the reply WAV at a transient URL, and **return the next SCCO synchronously** in the webhook response (`play` reply + next `recordMessage`). Per-call agent state lives in an in-memory registry keyed by Stringee `call_id` (single-instance/sticky for v1).

**Tech Stack:** Python 3.12, FastAPI, httpx, pytest + pytest-asyncio + respx. Reuses: `StringeeAdapter`, `VoiceBotAgent.handle_turn/play_opening`, `OutcomeRecorderMixin`, `analyze_agent_call`, Sarvam/Deepgram providers.

Spec: `docs/superpowers/specs/2026-06-09-stringee-ivr-design.md`.

---

## File Structure
- **Create** `src/api/telephony_stringee.py` — pure SCCO builder functions (`answer_scco`, `reply_scco`, `reprompt_scco`, `closing_scco`).
- **Create** `src/api/telephony_stringee_bridge.py` — `BufferingAudioSink`, audio helpers (`pcm16_to_wav`, `wav_to_pcm16`, `resample_pcm16`), `AudioStore`, `StringeeIvrBridge`, and the module-level call registry.
- **Modify** `src/bootstrap.py` — add `make_stringee_bridge_factory(...)`.
- **Modify** `src/api/telephony_hooks.py` — add the four Stringee routes + `set_stringee_bridge_factory`.
- **Modify** `src/main.py` — register the Stringee bridge factory in the lifespan (and clear it on shutdown).
- **Modify** `docs/stringee-streaming.md` — record that the IVR path is the implemented one.
- **Tests:** `tests/unit/test_telephony_stringee_scco.py`, `tests/unit/test_telephony_stringee_audio.py`, `tests/unit/test_telephony_stringee_bridge.py`, `tests/unit/test_telephony_stringee_routes.py`.

**Audio rates:** Sarvam TTS renders PCM16 mono @ 16 kHz; we serve the reply/opening as 16 kHz WAV. Stringee `recordMessage` is requested as WAV; recordings are PCM16 @ 8 kHz, resampled to 16 kHz before `handle_turn` (which expects 16 kHz like the other bridges). The "does Stringee `play` accept 16 kHz WAV / does `recordMessage` emit WAV" questions are confirmed in the **live test** (Task 7); resampling helpers are in place either way.

---

## Task 1: SCCO builder functions

**Files:**
- Create: `src/api/telephony_stringee.py`
- Test: `tests/unit/test_telephony_stringee_scco.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_telephony_stringee_scco.py`:

```python
from src.api.telephony_stringee import (
    answer_scco, reply_scco, reprompt_scco, closing_scco,
)


def test_answer_scco_plays_opening_then_records():
    scco = answer_scco(audio_url="https://x/a.wav", event_url="https://x/ev")
    assert scco[0]["action"] == "play"
    assert scco[0]["url"] == "https://x/a.wav"
    assert scco[0]["bargeIn"] is True
    rec = scco[1]
    assert rec["action"] == "recordMessage"
    assert rec["eventUrl"] == "https://x/ev"
    assert rec["format"] == "wav"
    assert rec["silenceTimeout"] == 1500


def test_reply_scco_same_shape():
    scco = reply_scco(audio_url="https://x/r.wav", event_url="https://x/ev")
    assert [a["action"] for a in scco] == ["play", "recordMessage"]
    assert scco[0]["bargeIn"] is True


def test_reprompt_scco_uses_talk_and_records_again():
    scco = reprompt_scco(text="Dobara boliye?", event_url="https://x/ev")
    assert scco[0]["action"] == "talk"
    assert scco[0]["text"] == "Dobara boliye?"
    assert scco[1]["action"] == "recordMessage"


def test_closing_scco_plays_then_hangs_up():
    scco = closing_scco(audio_url="https://x/c.wav")
    assert [a["action"] for a in scco] == ["play", "hangup"]
    assert "eventUrl" not in scco[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_telephony_stringee_scco.py -q`
Expected: FAIL — `ModuleNotFoundError: src.api.telephony_stringee`.

- [ ] **Step 3: Implement the builders**

Create `src/api/telephony_stringee.py`:

```python
"""Stringee Call Control Object (SCCO) builders for the IVR voicebot.

Pure functions that return SCCO JSON (a list of action dicts). Stringee
fetches/returns SCCO at call answer and after each recorded turn; see
docs/superpowers/specs/2026-06-09-stringee-ivr-design.md.
"""

from __future__ import annotations

from typing import Any

# Silence (ms) after the caller stops speaking before Stringee ends the
# recording and POSTs us the utterance. Tuned down from a typical 4000ms to
# keep per-turn latency tolerable (see spec, latency section).
SILENCE_TIMEOUT_MS = 1500


def _record(event_url: str) -> dict[str, Any]:
    return {
        "action": "recordMessage",
        "eventUrl": event_url,
        "format": "wav",
        "silenceTimeout": SILENCE_TIMEOUT_MS,
        "beepStart": False,
    }


def answer_scco(*, audio_url: str, event_url: str) -> list[dict[str, Any]]:
    """Opening turn: play the greeting (interruptible), then record the reply."""
    return [
        {"action": "play", "url": audio_url, "bargeIn": True},
        _record(event_url),
    ]


def reply_scco(*, audio_url: str, event_url: str) -> list[dict[str, Any]]:
    """A normal turn: play the agent's reply, then record the next utterance."""
    return [
        {"action": "play", "url": audio_url, "bargeIn": True},
        _record(event_url),
    ]


def reprompt_scco(*, text: str, event_url: str) -> list[dict[str, Any]]:
    """Empty/failed capture: speak a short re-prompt and record again."""
    return [
        {"action": "talk", "text": text, "bargeIn": True},
        _record(event_url),
    ]


def closing_scco(*, audio_url: str) -> list[dict[str, Any]]:
    """Terminal turn: play the closing line and hang up (no further record)."""
    return [
        {"action": "play", "url": audio_url},
        {"action": "hangup"},
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_telephony_stringee_scco.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/api/telephony_stringee.py tests/unit/test_telephony_stringee_scco.py
git commit -m "feat(stringee): SCCO builders for IVR voicebot turns"
```

---

## Task 2: Audio helpers + buffering sink

**Files:**
- Create: `src/api/telephony_stringee_bridge.py` (audio section only this task)
- Test: `tests/unit/test_telephony_stringee_audio.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_telephony_stringee_audio.py`:

```python
import pytest

from src.api.telephony_stringee_bridge import (
    BufferingAudioSink, pcm16_to_wav, wav_to_pcm16, resample_pcm16,
)


def test_pcm16_to_wav_roundtrips():
    pcm = b"\x01\x02\x03\x04" * 100
    wav = pcm16_to_wav(pcm, sample_rate=16000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    back, rate = wav_to_pcm16(wav)
    assert back == pcm and rate == 16000


def test_resample_pcm16_8k_to_16k_doubles_length():
    pcm8 = b"\x00\x01" * 80  # 80 samples @ 8k
    pcm16 = resample_pcm16(pcm8, 8000, 16000)
    # 2x rate => ~2x samples (allow ratecv's boundary slack)
    assert abs(len(pcm16) - 2 * len(pcm8)) <= 4


@pytest.mark.asyncio
async def test_buffering_sink_collects_pcm():
    sink = BufferingAudioSink()
    await sink(b"ab")
    await sink(b"cd")
    assert sink.pcm == b"abcd"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_telephony_stringee_audio.py -q`
Expected: FAIL — `ModuleNotFoundError: src.api.telephony_stringee_bridge`.

- [ ] **Step 3: Implement the audio helpers**

Create `src/api/telephony_stringee_bridge.py` with (this task adds only the audio section; later tasks append to the same file):

```python
"""Stringee IVR bridge: per-call turn controller + audio helpers + registry.

Turn-based (no streaming): Stringee records each utterance and POSTs it to our
event webhook; we run the agent's batch handle_turn and return the next SCCO.
See docs/superpowers/specs/2026-06-09-stringee-ivr-design.md.
"""

from __future__ import annotations

import audioop
import io
import logging
import struct
import time
import wave

log = logging.getLogger(__name__)


# --- audio helpers ------------------------------------------------------

def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM16-LE mono in a WAV container Stringee's `play` can fetch."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm16(blob: bytes) -> tuple[bytes, int]:
    """Extract raw PCM16-LE mono + sample rate from a WAV blob.

    Falls back to treating the blob as headerless PCM @ 8 kHz if it isn't a
    recognizable RIFF/WAVE container (defensive — Stringee recordings vary).
    """
    if len(blob) < 44 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        return blob, 8000
    with wave.open(io.BytesIO(blob), "rb") as w:
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    return frames, rate


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample mono PCM16-LE between sample rates (no-op if equal)."""
    if src_rate == dst_rate or not pcm:
        return pcm
    converted, _ = audioop.ratecv(pcm, 2, 1, src_rate, dst_rate, None)
    return converted


class BufferingAudioSink:
    """AudioSink that accumulates PCM instead of streaming it.

    handle_turn / play_opening push the agent's TTS PCM through an
    ``async (bytes) -> None`` sink; for IVR we collect it so it can be
    WAV-encoded and hosted for Stringee's `play`.
    """

    def __init__(self) -> None:
        self._chunks: list[bytes] = []

    async def __call__(self, pcm16: bytes) -> None:
        if pcm16:
            self._chunks.append(pcm16)

    @property
    def pcm(self) -> bytes:
        return b"".join(self._chunks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_telephony_stringee_audio.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/api/telephony_stringee_bridge.py tests/unit/test_telephony_stringee_audio.py
git commit -m "feat(stringee): WAV/PCM audio helpers + buffering audio sink"
```

---

## Task 3: Transient audio store

**Files:**
- Modify: `src/api/telephony_stringee_bridge.py` (append `AudioStore`)
- Test: `tests/unit/test_telephony_stringee_audio.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_telephony_stringee_audio.py`:

```python
from src.api.telephony_stringee_bridge import AudioStore


def test_audio_store_put_get_and_token_is_opaque():
    store = AudioStore(ttl_seconds=60)
    token = store.put(b"wavbytes")
    assert isinstance(token, str) and len(token) >= 16
    assert store.get(token) == b"wavbytes"


def test_audio_store_evicts_expired(monkeypatch):
    import src.api.telephony_stringee_bridge as m
    t = {"now": 1000.0}
    monkeypatch.setattr(m.time, "monotonic", lambda: t["now"])
    store = AudioStore(ttl_seconds=10)
    token = store.put(b"x")
    t["now"] = 1011.0  # past TTL
    assert store.get(token) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_telephony_stringee_audio.py -q`
Expected: FAIL — `ImportError: cannot import name 'AudioStore'`.

- [ ] **Step 3: Implement `AudioStore`**

Append to `src/api/telephony_stringee_bridge.py`:

```python
import secrets


class AudioStore:
    """Short-lived token -> WAV bytes map for serving reply audio to Stringee.

    Entries expire after ``ttl_seconds`` (a call's audio is fetched once,
    seconds after we hand Stringee the URL). Eviction is lazy on get/put.
    """

    def __init__(self, ttl_seconds: float = 120.0) -> None:
        self._ttl = ttl_seconds
        self._items: dict[str, tuple[float, bytes]] = {}

    def _sweep(self) -> None:
        cutoff = time.monotonic() - self._ttl
        for tok in [k for k, (ts, _) in self._items.items() if ts < cutoff]:
            self._items.pop(tok, None)

    def put(self, wav: bytes) -> str:
        self._sweep()
        token = secrets.token_urlsafe(16)
        self._items[token] = (time.monotonic(), wav)
        return token

    def get(self, token: str) -> bytes | None:
        self._sweep()
        item = self._items.get(token)
        return item[1] if item else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_telephony_stringee_audio.py -q`
Expected: PASS (5 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/api/telephony_stringee_bridge.py tests/unit/test_telephony_stringee_audio.py
git commit -m "feat(stringee): transient TTL audio store for reply playback URLs"
```

---

## Task 4: StringeeIvrBridge (turn controller) + registry

**Files:**
- Modify: `src/api/telephony_stringee_bridge.py` (append the bridge + registry)
- Test: `tests/unit/test_telephony_stringee_bridge.py`

This is the core. The bridge is HTTP-driven (not a WS loop): the routes call `start_call` once and `handle_turn` per webhook. It builds absolute `play`/`event` URLs from a `base_url` the route passes in, downloads the recording with an injected async `fetch` callable (so tests don't hit the network), and reuses `OutcomeRecorderMixin`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_telephony_stringee_bridge.py`:

```python
import pytest

from src.api.telephony_stringee_bridge import (
    StringeeIvrBridge, registry, pcm16_to_wav,
)


class _FakeResponse:
    def __init__(self, response_text="जी हाँ", action="continue"):
        self.response = type("R", (), {"response_text": response_text, "action": action})()


class _FakeAgent:
    """Records calls; play_opening + handle_turn push PCM into the sink."""
    def __init__(self):
        self.turns = []
        self.hung_up = False
        self.state = type("S", (), {"is_terminal": False})()
        self._next = _FakeResponse()

    async def play_opening(self, sink):
        await sink(b"\x10\x00" * 8)  # 16 bytes of "opening" PCM

    async def handle_turn(self, captured, sink):
        self.turns.append(captured)
        await sink(b"\x20\x00" * 8)  # "reply" PCM
        return type("O", (), {"response": self._next.response})()

    async def handle_hangup(self):
        self.hung_up = True


async def _fetch_ok(url):  # injected downloader -> returns WAV of silence
    return pcm16_to_wav(b"\x00\x00" * 80, sample_rate=8000)


def _bridge(agent):
    return StringeeIvrBridge(
        call_id="call-1", agent=agent, llm=None,
        tenant_timezone="Asia/Kolkata", tts_sample_rate=16000,
        base_url="https://host/api/v1/telephony/stringee", tenant_slug="dev",
        fetch=_fetch_ok,
    )


@pytest.mark.asyncio
async def test_start_call_returns_answer_scco_with_hosted_opening():
    agent = _FakeAgent()
    bridge = _bridge(agent)
    scco = await bridge.start_call()
    assert scco[0]["action"] == "play"
    assert scco[0]["url"].startswith("https://host/api/v1/telephony/stringee/audio/")
    assert scco[1]["action"] == "recordMessage"
    # opening audio is fetchable from the store via the token in the url
    token = scco[0]["url"].rsplit("/", 1)[1]
    assert bridge.audio.get(token) is not None


@pytest.mark.asyncio
async def test_handle_turn_runs_agent_and_returns_reply_scco():
    agent = _FakeAgent()
    bridge = _bridge(agent)
    scco = await bridge.handle_turn(recording_url="https://rec/1.wav")
    assert len(agent.turns) == 1            # handle_turn was driven
    assert agent.turns[0]                    # got decoded PCM bytes
    assert scco[0]["action"] == "play"       # reply audio
    assert scco[1]["action"] == "recordMessage"


@pytest.mark.asyncio
async def test_handle_turn_terminal_action_returns_closing_scco():
    agent = _FakeAgent()
    agent._next = _FakeResponse(response_text="Dhanyavaad", action="close_positive")
    bridge = _bridge(agent)
    scco = await bridge.handle_turn(recording_url="https://rec/1.wav")
    assert [a["action"] for a in scco] == ["play", "hangup"]


@pytest.mark.asyncio
async def test_handle_turn_empty_reply_reprompts():
    agent = _FakeAgent()
    agent._next = _FakeResponse(response_text="", action="continue")
    bridge = _bridge(agent)
    scco = await bridge.handle_turn(recording_url="https://rec/1.wav")
    assert scco[0]["action"] == "talk"       # re-prompt, not a play
    assert scco[1]["action"] == "recordMessage"


@pytest.mark.asyncio
async def test_registry_create_lookup_end():
    agent = _FakeAgent()
    bridge = _bridge(agent)
    registry.put(bridge)
    assert registry.get("call-1") is bridge
    await registry.end("call-1")
    assert agent.hung_up is True
    assert registry.get("call-1") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_telephony_stringee_bridge.py -q`
Expected: FAIL — `ImportError: cannot import name 'StringeeIvrBridge'`.

- [ ] **Step 3: Implement the bridge + registry**

Append to `src/api/telephony_stringee_bridge.py`:

```python
from typing import Awaitable, Callable, Optional

from src.api.outcome_recorder import OutcomeRecorderMixin
from src.api.telephony_stringee import (
    answer_scco, reply_scco, reprompt_scco, closing_scco,
)
from src.interfaces.llm import ILLMProvider

# Agent reply actions that end the call.
_TERMINAL_ACTIONS = {"end", "close_positive", "close_negative"}
_REPROMPT_TEXT = "Maaf kijiye, dobara boliye?"

Fetch = Callable[[str], Awaitable[bytes]]


class StringeeIvrBridge(OutcomeRecorderMixin):
    """One per call. HTTP-driven: start_call once, handle_turn per webhook."""

    def __init__(
        self,
        *,
        call_id: str,
        agent,
        llm: Optional[ILLMProvider],
        tenant_timezone: str,
        tts_sample_rate: int,
        base_url: str,
        tenant_slug: str,
        fetch: Fetch,
    ) -> None:
        self.call_id = call_id
        self._agent = agent
        self._llm = llm
        self._tenant_timezone = tenant_timezone
        self._rate = tts_sample_rate
        self._base = base_url.rstrip("/")
        self._slug = tenant_slug
        self._fetch = fetch
        self.audio = AudioStore()
        self._last_action: Optional[str] = None
        self._outcome_recorded = False
        self.touched = time.monotonic()

    # -- url builders --
    def _event_url(self) -> str:
        return f"{self._base}/event/{self._slug}?call_id={self.call_id}"

    def _host(self, pcm16: bytes) -> str:
        wav = pcm16_to_wav(pcm16, self._rate)
        token = self.audio.put(wav)
        return f"{self._base}/audio/{token}"

    # -- lifecycle --
    async def start_call(self) -> list[dict]:
        sink = BufferingAudioSink()
        await self._agent.play_opening(sink)
        url = self._host(sink.pcm)
        return answer_scco(audio_url=url, event_url=self._event_url())

    async def handle_turn(self, *, recording_url: str) -> list[dict]:
        self.touched = time.monotonic()
        # Download + decode + resample to the pipeline's 16 kHz.
        try:
            blob = await self._fetch(recording_url)
            pcm, rate = wav_to_pcm16(blob)
            pcm = resample_pcm16(pcm, rate, self._rate)
        except Exception:  # noqa: BLE001 - never drop the call on a bad recording
            log.exception("stringee recording fetch/decode failed")
            return reprompt_scco(text=_REPROMPT_TEXT, event_url=self._event_url())

        sink = BufferingAudioSink()
        outcome = await self._agent.handle_turn(pcm, sink)
        self._last_action = outcome.response.action
        reply_pcm = sink.pcm

        if outcome.response.action in _TERMINAL_ACTIONS:
            url = self._host(reply_pcm) if reply_pcm else self._host(b"")
            return closing_scco(audio_url=url)
        if not reply_pcm:
            return reprompt_scco(text=_REPROMPT_TEXT, event_url=self._event_url())
        return reply_scco(audio_url=self._host(reply_pcm), event_url=self._event_url())

    async def end(self) -> None:
        await self._record_outcome()          # OutcomeRecorderMixin
        await self._agent.handle_hangup()


class _Registry:
    """In-memory call_id -> bridge map (single-instance/sticky; see spec)."""

    def __init__(self, ttl_seconds: float = 900.0) -> None:
        self._ttl = ttl_seconds
        self._calls: dict[str, StringeeIvrBridge] = {}

    def put(self, bridge: StringeeIvrBridge) -> None:
        self._sweep()
        self._calls[bridge.call_id] = bridge

    def get(self, call_id: str) -> Optional[StringeeIvrBridge]:
        return self._calls.get(call_id)

    async def end(self, call_id: str) -> None:
        bridge = self._calls.pop(call_id, None)
        if bridge is not None:
            await bridge.end()

    def _sweep(self) -> None:
        cutoff = time.monotonic() - self._ttl
        for cid in [c for c, b in self._calls.items() if b.touched < cutoff]:
            self._calls.pop(cid, None)


registry = _Registry()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_telephony_stringee_bridge.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the audio + scco + bridge tests together**

Run: `.venv/bin/pytest tests/unit/test_telephony_stringee_*.py -q`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add src/api/telephony_stringee_bridge.py tests/unit/test_telephony_stringee_bridge.py
git commit -m "feat(stringee): IVR turn controller (start/handle_turn/end) + call registry"
```

---

## Task 5: Bridge factory (per-call agent assembly)

**Files:**
- Modify: `src/bootstrap.py` (add `make_stringee_bridge_factory`)
- Test: `tests/unit/test_telephony_stringee_bridge.py` (append)

The factory mirrors `make_exotel_bridge_factory` (bootstrap.py:242) — same agent assembly — but returns a `StringeeIvrBridge`. It takes per-call identifiers (`call_id`, `base_url`, `tenant`, `fetch`) instead of a websocket.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_telephony_stringee_bridge.py`:

```python
def test_make_stringee_bridge_factory_builds_a_bridge():
    from types import SimpleNamespace
    from src.bootstrap import make_stringee_bridge_factory
    from src.dialogue.prompts import VoiceBotScript
    from src.dialogue.slots import SlotSchema

    class _Providers:
        def get_stt(self, t): return object()
        def get_llm(self, t): return None
        def get_tts(self, t): return object()

    tenant = SimpleNamespace(
        id="dev", slug="dev",
        settings=SimpleNamespace(pipeline=SimpleNamespace(
            stt=SimpleNamespace(language="hi-IN"),
            llm=SimpleNamespace(temperature=0.5, max_tokens=256, response_format="json"),
            tts=SimpleNamespace(language="hi-IN", voice_id=None),
        )),
    )
    script = VoiceBotScript.from_campaign_yaml(
        {"agent_name": "A", "agent_role": "R", "company_name": "C"}
    )
    factory = make_stringee_bridge_factory(
        providers=_Providers(), script=script, slots=SlotSchema(),
    )
    async def _fetch(url): return b""
    bridge = factory(call_id="c-9", tenant=tenant,
                     base_url="https://h/api/v1/telephony/stringee", fetch=_fetch)
    assert bridge.call_id == "c-9"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_telephony_stringee_bridge.py::test_make_stringee_bridge_factory_builds_a_bridge -q`
Expected: FAIL — `ImportError: cannot import name 'make_stringee_bridge_factory'`.

- [ ] **Step 3: Implement the factory**

In `src/bootstrap.py`, add (mirroring `make_exotel_bridge_factory` at line 242; reuse the same imports already at the top of that file — `TenantProviders`, `VoiceBotScript`, `SlotSchema`, `PipelineConfig`, `STTConfig`, `LLMConfig`, `TTSConfig`, `PipelineEngine`, `AgentSession`, `AgentStateMachine`, `VoiceBotAgent`):

```python
def make_stringee_bridge_factory(
    providers: TenantProviders,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
):
    """Build a StringeeIvrBridge per call, wired to the tenant's providers.

    Same agent assembly as make_exotel_bridge_factory; HTTP-driven instead of
    WS-driven, so the call_id/base_url/fetch are passed per call by the route.
    """
    from src.api.telephony_stringee_bridge import StringeeIvrBridge

    def factory(*, call_id, tenant, base_url, fetch):
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
        import uuid
        session = AgentSession(session_id=f"call_{uuid.uuid4().hex[:12]}")
        agent = VoiceBotAgent(
            session=session, state_machine=AgentStateMachine(),
            slot_schema=slots, script=script, engine=engine,
        )
        return StringeeIvrBridge(
            call_id=str(call_id), agent=agent, llm=llm,
            tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"),
            tts_sample_rate=16000, base_url=base_url, tenant_slug=tenant.slug,
            fetch=fetch,
        )

    return factory
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_telephony_stringee_bridge.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/bootstrap.py tests/unit/test_telephony_stringee_bridge.py
git commit -m "feat(stringee): per-call IVR bridge factory wired to tenant providers"
```

---

## Task 6: HTTP routes + factory registration

**Files:**
- Modify: `src/api/telephony_hooks.py` (routes + `set_stringee_bridge_factory`)
- Modify: `src/main.py` (register/clear the factory in lifespan)
- Test: `tests/unit/test_telephony_stringee_routes.py`

The event webhook returns the next SCCO **synchronously**. Routes resolve the tenant (by `From` for outbound / `To` for inbound, like `exotel_voice`), build `base_url` from forwarded headers, and use an httpx-based `fetch` to download recordings.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_telephony_stringee_routes.py`:

```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.telephony_hooks as hooks
from src.api.telephony_stringee_bridge import registry, StringeeIvrBridge, pcm16_to_wav


class _Agent:
    def __init__(self):
        self.state = type("S", (), {"is_terminal": False})()
        self._action = "continue"
    async def play_opening(self, sink): await sink(b"\x01\x00" * 8)
    async def handle_turn(self, captured, sink):
        await sink(b"\x02\x00" * 8)
        return type("O", (), {"response": type("R", (), {"response_text": "ok", "action": self._action})()})()
    async def handle_hangup(self): pass


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(hooks.router, prefix="/api/v1")

    async def _fetch(url): return pcm16_to_wav(b"\x00\x00" * 80, 8000)

    def _factory(*, call_id, tenant, base_url, fetch):
        return StringeeIvrBridge(
            call_id=str(call_id), agent=_Agent(), llm=None,
            tenant_timezone="Asia/Kolkata", tts_sample_rate=16000,
            base_url=base_url, tenant_slug="dev", fetch=_fetch,
        )
    hooks.set_stringee_bridge_factory(_factory)

    # Stub tenant resolution to a fake "dev" tenant.
    from types import SimpleNamespace
    async def _resolve(num): return SimpleNamespace(slug="dev", id="dev")
    monkeypatch.setattr(hooks, "tenant_from_twilio_to_number", _resolve)
    yield TestClient(app)
    hooks.set_stringee_bridge_factory(None)


def test_answer_returns_opening_scco_and_audio_served(client):
    r = client.post("/api/v1/telephony/stringee/answer",
                    json={"call_id": "c1", "to": "+1", "from": "+2", "direction": "outbound"})
    assert r.status_code == 200
    scco = r.json()
    assert scco[0]["action"] == "play" and scco[1]["action"] == "recordMessage"
    # audio token is fetchable
    token = scco[0]["url"].rsplit("/", 1)[1]
    a = client.get(f"/api/v1/telephony/stringee/audio/{token}")
    assert a.status_code == 200 and a.content[:4] == b"RIFF"


def test_event_runs_a_turn_and_returns_reply_scco(client):
    client.post("/api/v1/telephony/stringee/answer",
                json={"call_id": "c1", "to": "+1", "from": "+2", "direction": "outbound"})
    r = client.post("/api/v1/telephony/stringee/event/dev?call_id=c1",
                    json={"recording_url": "https://rec/1.wav"})
    assert r.status_code == 200
    scco = r.json()
    assert scco[0]["action"] == "play" and scco[1]["action"] == "recordMessage"


def test_event_unknown_call_returns_reprompt(client):
    r = client.post("/api/v1/telephony/stringee/event/dev?call_id=nope",
                    json={"recording_url": "https://rec/1.wav"})
    assert r.status_code == 200
    assert r.json()[0]["action"] == "talk"  # safe re-prompt, never a 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_telephony_stringee_routes.py -q`
Expected: FAIL — `AttributeError: module 'src.api.telephony_hooks' has no attribute 'set_stringee_bridge_factory'`.

- [ ] **Step 3: Implement the routes + registration**

In `src/api/telephony_hooks.py`: add the factory slot + setter near the existing `set_exotel_bridge_factory` (line 45):

```python
_stringee_bridge_factory = None


def set_stringee_bridge_factory(factory) -> None:
    global _stringee_bridge_factory
    _stringee_bridge_factory = factory
```

Add these imports at the top of the file (alongside the existing ones):

```python
import httpx
from fastapi import Request
from src.api.telephony_stringee import reprompt_scco
from src.api.telephony_stringee_bridge import registry
```

Add a helper + the four routes (after the exotel routes):

```python
def _stringee_base(request: Request) -> str:
    base = request.headers.get("x-forwarded-host") or request.url.netloc
    proto = request.headers.get("x-forwarded-proto")
    scheme = "https" if (proto == "https" or request.url.scheme == "https") else "http"
    return f"{scheme}://{base}/api/v1/telephony/stringee"


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=5.0)) as c:
        resp = await c.get(url)
        resp.raise_for_status()
        return resp.content


@router.post("/stringee/answer")
async def stringee_answer(request: Request):
    """Call answered → build the call's bridge and return the opening SCCO."""
    body = await request.json()
    call_id = str(body.get("call_id") or body.get("callId") or "")
    direction = (body.get("direction") or "").lower()
    to_num, from_num = body.get("to"), body.get("from")
    lookup = from_num if (direction.startswith("outbound") and from_num) else to_num
    tenant = await tenant_from_twilio_to_number(lookup)
    if _stringee_bridge_factory is None:
        return Response(status_code=503)
    bridge = _stringee_bridge_factory(
        call_id=call_id, tenant=tenant,
        base_url=_stringee_base(request), fetch=_download,
    )
    registry.put(bridge)
    scco = await bridge.start_call()
    log.info("stringee answer", extra={"tenant": tenant.slug, "call_id": call_id})
    return JSONResponse(scco)


@router.post("/stringee/event/{tenant_slug}")
async def stringee_event(tenant_slug: str, request: Request, call_id: str):
    """Per-turn recordMessage webhook → run a turn → return the next SCCO."""
    body = await request.json()
    rec_url = body.get("recording_url") or body.get("url") or body.get("link")
    bridge = registry.get(call_id)
    if bridge is None or not rec_url:
        # Call state lost (restart / unknown) or no recording: re-prompt safely.
        base = _stringee_base(request)
        return JSONResponse(reprompt_scco(
            text="Maaf kijiye, dobara boliye?",
            event_url=f"{base}/event/{tenant_slug}?call_id={call_id}",
        ))
    scco = await bridge.handle_turn(recording_url=rec_url)
    return JSONResponse(scco)


@router.get("/stringee/audio/{token}")
async def stringee_audio(token: str, call_id: str | None = None):
    """Serve a hosted reply/opening WAV for Stringee's `play` to fetch."""
    for bridge in [registry.get(call_id)] if call_id else []:
        if bridge is not None:
            wav = bridge.audio.get(token)
            if wav is not None:
                return Response(content=wav, media_type="audio/wav")
    # Fall back to scanning live calls (Stringee fetches without our call_id).
    for cid in list(getattr(registry, "_calls", {})):
        wav = registry._calls[cid].audio.get(token)
        if wav is not None:
            return Response(content=wav, media_type="audio/wav")
    return Response(status_code=404)


@router.post("/stringee/status/{tenant_slug}")
async def stringee_status(tenant_slug: str, request: Request):
    """Lifecycle webhook: on call end, record the outcome and clean up."""
    body = await request.json()
    call_id = str(body.get("call_id") or body.get("callId") or "")
    status = (body.get("status") or "").upper()
    if status in ("ENDED", "FAILED", "NO_ANSWER", "BUSY"):
        await registry.end(call_id)
    return Response(status_code=200)
```

(Confirm `JSONResponse` and `Response` are imported in `telephony_hooks.py`; add `from fastapi.responses import JSONResponse` if only `Response` is present.)

In `src/main.py`, register the factory in the lifespan (next to the exotel registration at line 119) and clear it on shutdown (next to line 139):

```python
    telephony_hooks.set_stringee_bridge_factory(
        make_stringee_bridge_factory(
            providers=providers, script=campaign.script, slots=campaign.slots,
        )
    )
```
```python
        telephony_hooks.set_stringee_bridge_factory(None)
```
Add `make_stringee_bridge_factory` to the existing `from src.bootstrap import (...)` block in `main.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_telephony_stringee_routes.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (no regressions).

- [ ] **Step 6: Commit**

```bash
git add src/api/telephony_hooks.py src/main.py tests/unit/test_telephony_stringee_routes.py
git commit -m "feat(stringee): IVR answer/event/audio/status routes + factory registration"
```

---

## Task 7: Docs, config, and live validation

**Files:**
- Modify: `docs/stringee-streaming.md`
- Modify: `docs/deploy/northflank.md` (Stringee env + answer_url)

- [ ] **Step 1: Update `docs/stringee-streaming.md`**

Add a section at the top noting the **IVR path is now implemented** (turn-based, no bridge service), that the headless-bridge recipe below it remains the (unbuilt) option for true streaming, and link the spec/plan. Keep the existing recipe for posterity.

- [ ] **Step 2: Document config + env**

In `docs/deploy/northflank.md`, add Stringee to the telephony section: per-tenant `telephony.provider: stringee`, env `STRINGEE_API_KEY_SID` / `STRINGEE_API_KEY_SECRET`, and the **answer_url** to set on the Stringee number / in `callout`:
`https://<host>/api/v1/telephony/stringee/answer` (and status URL `…/stringee/status/<tenant>`). Note `WEBHOOK_BASE_URL` must be publicly reachable (Stringee fetches the audio + posts events).

- [ ] **Step 3: Commit docs**

```bash
git add docs/stringee-streaming.md docs/deploy/northflank.md
git commit -m "docs(stringee): record the implemented IVR path + config/answer_url"
```

- [ ] **Step 4: Live validation (manual — the real gate)**

With `STRINGEE_API_KEY_SID/SECRET` set and `WEBHOOK_BASE_URL` public (deployed or tunneled), place one outbound call to the test number (`+918618795697`) via the campaign/initiate path with `answer_url=…/stringee/answer`. Confirm, against the server logs:
1. `stringee answer` logged; the call plays the opening (verifies Stringee accepts our **16 kHz WAV** from `/audio/{token}` — if it rejects it, resample the hosted WAV to 8 kHz in `StringeeIvrBridge._host`).
2. Speaking → an `event` POST arrives with a recording link; a turn runs; the reply plays (verifies `recordMessage` emits fetchable **WAV** — if it's MP3/other, set the right `format` or add a decode branch in `wav_to_pcm16`).
3. `bargeIn` interrupts the reply when you talk over it.
4. Ending the call → `stringee_status` ENDED → `call outcome` logged.

Record the outcome (and any rate/format adjustment) in a dated note appended to `docs/stringee-streaming.md`.

---

## Self-Review (author)

- **Spec coverage:** put_actions — *intentionally dropped from v1* (the synchronous webhook-returns-SCCO model removes its only caller; noted here and in Task 6's header as the documented fallback if Stringee's webhook timeout is too short). SCCO builders → Task 1. Audio hosting/format + buffering sink → Tasks 2–3. Bridge + registry → Task 4. Factory → Task 5. Routes (answer/event/audio/status) + tenant resolution + wiring → Task 6. Error/lifecycle (empty-STT re-prompt, fetch failure, unknown call, terminal action, hangup outcome, TTL sweep) → Tasks 4 & 6. Testing → every task + Task 7 live gate. Out-of-scope items (streaming, transfer, multi-instance, MP3) match the spec.
- **Deviation from spec (logged):** reply delivery is **synchronous** (event webhook returns the SCCO), not async `put_actions`. Rationale: Stringee's documented model is that record/input event webhooks return the next SCCO; this is simpler and drops the call-id reply-injection plumbing. The spec flagged the exact injection mechanism for plan-time verification, so this is in-bounds; the async+filler path is the fallback if the live test shows webhook-timeout pressure.
- **Type consistency:** `StringeeIvrBridge.__init__` kwargs match the factory call and the route's `_stringee_bridge_factory(call_id=, tenant=, base_url=, fetch=)`; `registry` exposes `put/get/end`; SCCO builder names (`answer_scco/reply_scco/reprompt_scco/closing_scco`) are used identically in the bridge; audio helpers (`pcm16_to_wav/wav_to_pcm16/resample_pcm16`) and `AudioStore.put/get` names are consistent across tasks.
- **Placeholders:** none — every code step is complete.
