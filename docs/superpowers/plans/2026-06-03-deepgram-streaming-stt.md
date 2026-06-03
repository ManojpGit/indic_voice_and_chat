# Deepgram Streaming STT (dev console) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace post-endpoint batch STT (Groq Whisper) with Deepgram live streaming on the browser dev-console path, so the transcript is ready the instant the user stops speaking, turn-end is Deepgram-driven (tunable below today's fixed 600 ms), and live partial transcripts show in the console.

**Architecture:** A new streaming-STT interface (`ISTTStreamSession` / `IStreamingSTTProvider`) with a Deepgram websocket adapter behind the existing provider-registry pattern. `PipelineEngine.run_turn` is split so its LLM→TTS half (`run_turn_text`) runs on an already-transcribed turn; `VoiceBotAgent.handle_turn_text` drives it. The browser bridge holds one persistent Deepgram session per connection, forwards user-only frames, dispatches on Deepgram's `speech_final`, and keeps Silero VAD + batch Groq as fallbacks. Telephony bridges are untouched.

**Tech Stack:** Python 3.12, asyncio, `websockets` (bundled with `deepgram-sdk`), FastAPI WebSocket, pytest + pytest-asyncio. Spec: `docs/superpowers/specs/2026-06-03-deepgram-streaming-stt-design.md`.

**Conventions (match existing code):**
- Adapters take a single `config: dict[str, Any]` and resolve `api_key` from `config["api_key"]` or an env var; they accept an injection seam for tests (see `GeminiLLMAdapter`/`AnthropicClaudeAdapter` `config["client"]`).
- Run tests with the repo venv: `.venv/bin/python -m pytest`.
- Commit directly to `main` (repo convention). End commit messages with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Never commit `.env` or `.claude/`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/interfaces/stt.py` | `STTStreamEvent`, `ISTTStreamSession`, `IStreamingSTTProvider` | Modify (additions only) |
| `src/providers/stt/deepgram.py` | Deepgram live websocket session + provider adapter | Create |
| `src/providers/__init__.py` | `STREAMING_STT_PROVIDERS` registry + `get_streaming_stt_provider` | Modify |
| `pyproject.toml` | add `deepgram-sdk>=3.7` dependency | Modify |
| `src/pipeline/engine.py` | split out `run_turn_text` (LLM→TTS), `run_turn` delegates | Modify |
| `src/agents/voicebot.py` | `_finish_turn` helper + `handle_turn_text` | Modify |
| `src/config_tenant.py` | `TenantStreamingSTTConfig` + `pipeline.stt_streaming` | Modify |
| `config/tenants/dev.yaml` | `pipeline.stt_streaming` block | Modify |
| `src/api/browser_bridge.py` | streaming path (forward frames, consume events, dispatch, fallback) | Modify |
| `src/api/dev_console.py` | build streaming provider, pass to bridge | Modify |
| `static/dev_console.html` | render/clear live partial transcript line | Modify |
| `tests/unit/test_streaming_stt_interface.py` | interface dataclass/contract | Create |
| `tests/unit/test_deepgram_adapter.py` | session parse matrix + adapter | Create |
| `tests/unit/test_engine_run_turn_text.py` | engine text-entry path | Create |
| `tests/unit/test_voicebot_handle_turn_text.py` | agent text-entry path | Create |
| `tests/unit/test_browser_bridge_streaming.py` | bridge streaming dispatch + fallback | Create |
| `tests/unit/test_tenant_streaming_config.py` | config schema parse | Create |

Build order is dependency order: interface → adapter → registry → engine → agent → config → bridge → dev_console wiring → html.

---

## Task 1: Streaming-STT interface

**Files:**
- Modify: `src/interfaces/stt.py`
- Test: `tests/unit/test_streaming_stt_interface.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_streaming_stt_interface.py
from __future__ import annotations

import inspect

import pytest

from src.interfaces.stt import (
    ISTTStreamSession,
    IStreamingSTTProvider,
    STTStreamEvent,
)


def test_stream_event_defaults():
    ev = STTStreamEvent(type="final", text="नमस्ते")
    assert ev.type == "final"
    assert ev.text == "नमस्ते"
    assert ev.confidence == 1.0
    assert ev.language is None


def test_session_is_abstract():
    assert inspect.isabstract(ISTTStreamSession)
    for name in ("send", "events", "aclose"):
        assert hasattr(ISTTStreamSession, name)


def test_provider_is_abstract():
    assert inspect.isabstract(IStreamingSTTProvider)
    assert hasattr(IStreamingSTTProvider, "open_stream")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_streaming_stt_interface.py -q`
Expected: FAIL with `ImportError: cannot import name 'STTStreamEvent'`.

- [ ] **Step 3: Add the interface to `src/interfaces/stt.py`**

Append to the end of `src/interfaces/stt.py` (keep existing `STTResult`, `STTConfig`, `ISTTProvider` unchanged):

```python
@dataclass
class STTStreamEvent:
    """One event from a live STT session.

    type:
        "interim"  - a partial, non-final transcript (may change)
        "final"    - a finalized transcript segment (won't change)
        "endpoint" - end of utterance; ``text`` is the full utterance transcript
    """

    type: str
    text: str
    confidence: float = 1.0
    language: Optional[str] = None


class ISTTStreamSession(ABC):
    @abstractmethod
    async def send(self, pcm16: bytes) -> None:
        """Feed one chunk of raw PCM16-LE mono audio to the recognizer."""

    @abstractmethod
    def events(self) -> "AsyncIterator[STTStreamEvent]":
        """Yield recognizer events until the session is closed."""

    @abstractmethod
    async def aclose(self) -> None:
        """Flush, close the upstream connection, and cancel background tasks."""


class IStreamingSTTProvider(ABC):
    @abstractmethod
    async def open_stream(self, config: STTConfig) -> ISTTStreamSession:
        """Open a live streaming session for one utterance stream."""
```

`AsyncIterator` and `Optional` are already imported at the top of the file (`from typing import AsyncIterator, Optional`). `ABC`/`abstractmethod` and `dataclass` are already imported.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_streaming_stt_interface.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/interfaces/stt.py tests/unit/test_streaming_stt_interface.py
git commit -m "stt: add streaming-STT interface (ISTTStreamSession/IStreamingSTTProvider)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Deepgram adapter (session parse + provider)

**Files:**
- Create: `src/providers/stt/deepgram.py`
- Test: `tests/unit/test_deepgram_adapter.py` (create)

The session keeps the message-parse + accumulation logic in a pure method `_handle_raw(raw)` so it is unit-testable without a socket. The async receiver loop just calls `_handle_raw` and enqueues non-`None` events.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_deepgram_adapter.py
from __future__ import annotations

import asyncio
import json

import pytest

from src.interfaces.stt import STTConfig
from src.providers.stt.deepgram import DeepgramSTTAdapter, DeepgramStreamSession


def _results(transcript, is_final=False, speech_final=False):
    return json.dumps({
        "type": "Results",
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": transcript, "confidence": 0.9}]},
    })


# --- _handle_raw parse matrix ------------------------------------------

def test_interim_emits_interim_event():
    s = DeepgramStreamSession(ws=None, start_tasks=False)
    ev = s._handle_raw(_results("और कुछ", is_final=False))
    assert ev is not None and ev.type == "interim" and ev.text == "और कुछ"


def test_final_emits_final_and_accumulates():
    s = DeepgramStreamSession(ws=None, start_tasks=False)
    ev = s._handle_raw(_results("और कुछ benefits", is_final=True))
    assert ev.type == "final" and ev.text == "और कुछ benefits"


def test_speech_final_emits_endpoint_with_accumulated_text():
    s = DeepgramStreamSession(ws=None, start_tasks=False)
    s._handle_raw(_results("और कुछ", is_final=True))
    ev = s._handle_raw(_results("benefits हैं", is_final=True, speech_final=True))
    assert ev.type == "endpoint"
    assert ev.text == "और कुछ benefits हैं"


def test_accumulator_resets_after_endpoint():
    s = DeepgramStreamSession(ws=None, start_tasks=False)
    s._handle_raw(_results("पहला", is_final=True, speech_final=True))
    ev = s._handle_raw(_results("दूसरा", is_final=True, speech_final=True))
    assert ev.text == "दूसरा"  # not "पहला दूसरा"


def test_utterance_end_is_backup_only_when_no_speech_final():
    s = DeepgramStreamSession(ws=None, start_tasks=False)
    s._handle_raw(_results("कुछ", is_final=True))
    ev = s._handle_raw(json.dumps({"type": "UtteranceEnd"}))
    assert ev.type == "endpoint" and ev.text == "कुछ"


def test_utterance_end_suppressed_after_speech_final():
    s = DeepgramStreamSession(ws=None, start_tasks=False)
    s._handle_raw(_results("कुछ", is_final=True, speech_final=True))  # already endpointed
    ev = s._handle_raw(json.dumps({"type": "UtteranceEnd"}))
    assert ev is None


def test_empty_transcript_ignored():
    s = DeepgramStreamSession(ws=None, start_tasks=False)
    assert s._handle_raw(_results("", is_final=False)) is None


# --- end-to-end with a fake websocket ----------------------------------

class _FakeWS:
    def __init__(self, incoming):
        self._incoming = list(incoming)
        self.sent: list = []
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._incoming:
            raise StopAsyncIteration
        await asyncio.sleep(0)
        return self._incoming.pop(0)


@pytest.mark.asyncio
async def test_session_streams_events_end_to_end():
    ws = _FakeWS([
        _results("और कुछ", is_final=False),
        _results("और कुछ benefits हैं", is_final=True, speech_final=True),
    ])
    session = DeepgramStreamSession(ws=ws, keepalive_interval=999, start_tasks=True)
    types = []
    async for ev in session.events():
        types.append(ev.type)
    await session.aclose()
    assert types == ["interim", "endpoint"]
    assert ws.closed is True


# --- adapter -----------------------------------------------------------

@pytest.mark.asyncio
async def test_open_stream_builds_url_and_uses_connector():
    captured = {}

    async def fake_connector(url, headers):
        captured["url"] = url
        captured["headers"] = headers
        return _FakeWS([])

    adapter = DeepgramSTTAdapter({
        "api_key": "dg_key",
        "model": "nova-2",
        "language": "hi",
        "endpointing": 300,
        "utterance_end_ms": 1000,
        "connector": fake_connector,
    })
    session = await adapter.open_stream(STTConfig(language="hi", sample_rate=16000))
    await session.aclose()
    assert "model=nova-2" in captured["url"]
    assert "language=hi" in captured["url"]
    assert "endpointing=300" in captured["url"]
    assert "interim_results=true" in captured["url"]
    assert captured["headers"]["Authorization"] == "Token dg_key"


def test_adapter_rejects_missing_key(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(ValueError):
        DeepgramSTTAdapter({})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_deepgram_adapter.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.providers.stt.deepgram'`.

- [ ] **Step 3: Create `src/providers/stt/deepgram.py`**

```python
"""Deepgram streaming STT adapter (live websocket).

Implements IStreamingSTTProvider. One DeepgramStreamSession owns a single
live websocket to Deepgram's /v1/listen endpoint, feeds it PCM16 audio, and
emits STTStreamEvents (interim / final / endpoint). Message-parse and
accumulation live in the pure ``_handle_raw`` method for testability; the
async receiver loop just calls it and enqueues non-None events. A keepalive
task sends ``{"type":"KeepAlive"}`` so the socket survives agent-speech gaps.

Uses the ``websockets`` library directly (bundled with deepgram-sdk) for full
control over framing/keepalive and easy faking in tests; the deepgram-sdk
callback client is not used.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator, Awaitable, Callable, Optional
from urllib.parse import urlencode

from src.interfaces.stt import (
    ISTTStreamSession,
    IStreamingSTTProvider,
    STTConfig,
    STTStreamEvent,
)

DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-2"
DEFAULT_LANGUAGE = "hi"

# (url, headers) -> connected websocket-like object (async-iterable, .send, .close)
Connector = Callable[[str, dict[str, str]], Awaitable[Any]]


class DeepgramStreamSession(ISTTStreamSession):
    def __init__(
        self,
        ws: Any,
        *,
        keepalive_interval: float = 5.0,
        start_tasks: bool = True,
    ) -> None:
        self._ws = ws
        self._keepalive_interval = keepalive_interval
        self._queue: asyncio.Queue[Optional[STTStreamEvent]] = asyncio.Queue()
        self._acc: list[str] = []          # accumulated final segments this utterance
        self._endpointed = False           # speech_final already fired for current utterance
        self._closed = False
        self._tasks: list[asyncio.Task] = []
        if start_tasks:
            self._tasks.append(asyncio.create_task(self._receiver()))
            self._tasks.append(asyncio.create_task(self._keepalive()))

    # --- pure parse + accumulation (unit-tested directly) --------------

    def _handle_raw(self, raw: str) -> Optional[STTStreamEvent]:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return None
        mtype = msg.get("type")
        if mtype == "UtteranceEnd":
            if self._endpointed:
                return None  # speech_final already handled this utterance
            text = " ".join(self._acc).strip()
            self._reset_utterance()
            if not text:
                return None
            return STTStreamEvent(type="endpoint", text=text)
        if mtype != "Results":
            return None
        alt = (msg.get("channel", {}).get("alternatives") or [{}])[0]
        transcript = (alt.get("transcript") or "").strip()
        is_final = bool(msg.get("is_final"))
        speech_final = bool(msg.get("speech_final"))
        conf = float(alt.get("confidence", 1.0) or 1.0)
        if speech_final:
            if transcript:
                self._acc.append(transcript)
            text = " ".join(self._acc).strip()
            self._reset_utterance(endpointed=True)
            if not text:
                return None
            return STTStreamEvent(type="endpoint", text=text, confidence=conf)
        if not transcript:
            return None
        if is_final:
            self._acc.append(transcript)
            return STTStreamEvent(type="final", text=transcript, confidence=conf)
        return STTStreamEvent(type="interim", text=transcript, confidence=conf)

    def _reset_utterance(self, endpointed: bool = False) -> None:
        self._acc = []
        self._endpointed = endpointed

    # --- async wiring --------------------------------------------------

    async def _receiver(self) -> None:
        try:
            async for raw in self._ws:
                ev = self._handle_raw(raw if isinstance(raw, str) else raw.decode("utf-8"))
                if ev is not None:
                    await self._queue.put(ev)
        except Exception:  # noqa: BLE001 - surface as end-of-stream
            pass
        finally:
            await self._queue.put(None)  # sentinel: stream ended

    async def _keepalive(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(self._keepalive_interval)
                if self._closed:
                    return
                await self._ws.send(json.dumps({"type": "KeepAlive"}))
        except Exception:  # noqa: BLE001
            pass

    async def send(self, pcm16: bytes) -> None:
        if self._closed:
            return
        await self._ws.send(pcm16)

    async def events(self) -> AsyncIterator[STTStreamEvent]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._ws.send(json.dumps({"type": "CloseStream"}))
        except Exception:  # noqa: BLE001
            pass
        for t in self._tasks:
            t.cancel()
        try:
            await self._ws.close()
        except Exception:  # noqa: BLE001
            pass


class DeepgramSTTAdapter(IStreamingSTTProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._api_key = config.get("api_key") or os.environ.get("DEEPGRAM_API_KEY")
        if not self._api_key:
            raise ValueError(
                "DeepgramSTTAdapter requires an API key (config 'api_key' or "
                "DEEPGRAM_API_KEY env var)"
            )
        self._model = config.get("model") or DEFAULT_MODEL
        self._language = config.get("language") or DEFAULT_LANGUAGE
        self._endpointing = int(config.get("endpointing") or 300)
        self._utterance_end_ms = int(config.get("utterance_end_ms") or 1000)
        self._keepalive_interval = float(config.get("keepalive_interval") or 5.0)
        # Injection seam for tests; defaults to a real websockets connection.
        self._connector: Connector = config.get("connector") or _default_connector

    def _build_url(self, config: STTConfig) -> str:
        params = {
            "encoding": "linear16",
            "sample_rate": config.sample_rate or 16000,
            "channels": 1,
            "model": self._model,
            "language": config.language or self._language,
            "smart_format": "true",
            "interim_results": "true",
            "endpointing": self._endpointing,
            "utterance_end_ms": self._utterance_end_ms,
        }
        return f"{DEEPGRAM_WS_URL}?{urlencode(params)}"

    async def open_stream(self, config: STTConfig) -> ISTTStreamSession:
        url = self._build_url(config)
        headers = {"Authorization": f"Token {self._api_key}"}
        ws = await self._connector(url, headers)
        return DeepgramStreamSession(
            ws, keepalive_interval=self._keepalive_interval, start_tasks=True
        )


async def _default_connector(url: str, headers: dict[str, str]) -> Any:
    try:
        import websockets  # bundled with deepgram-sdk
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "DeepgramSTTAdapter requires 'websockets' (install deepgram-sdk)."
        ) from e
    # websockets>=12 uses additional_headers; older uses extra_headers.
    try:
        return await websockets.connect(url, additional_headers=headers)
    except TypeError:  # pragma: no cover - older websockets
        return await websockets.connect(url, extra_headers=headers)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_deepgram_adapter.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add src/providers/stt/deepgram.py tests/unit/test_deepgram_adapter.py
git commit -m "stt: add Deepgram streaming adapter (live websocket session)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Register Deepgram in the provider factory

**Files:**
- Modify: `src/providers/__init__.py`
- Modify: `pyproject.toml`
- Test: append to `tests/unit/test_deepgram_adapter.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_deepgram_adapter.py`:

```python
def test_registry_resolves_deepgram():
    from src.providers import STREAMING_STT_PROVIDERS, get_streaming_stt_provider
    assert STREAMING_STT_PROVIDERS["deepgram"].__name__ == "DeepgramSTTAdapter"
    provider = get_streaming_stt_provider({"provider": "deepgram", "api_key": "x"})
    assert provider.__class__.__name__ == "DeepgramSTTAdapter"


def test_registry_unknown_provider_raises():
    from src.providers import get_streaming_stt_provider, UnknownProviderError
    with pytest.raises(UnknownProviderError):
        get_streaming_stt_provider({"provider": "nope", "api_key": "x"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_deepgram_adapter.py -q -k registry`
Expected: FAIL with `ImportError: cannot import name 'STREAMING_STT_PROVIDERS'`.

- [ ] **Step 3: Edit `src/providers/__init__.py`**

Add the import near the other STT imports:

```python
from src.providers.stt.deepgram import DeepgramSTTAdapter
```

Add the interface import near the top (with the other `from src.interfaces...` imports):

```python
from src.interfaces.stt import ISTTProvider, IStreamingSTTProvider
```
(Change the existing `from src.interfaces.stt import ISTTProvider` line to the one above.)

Add the registry after `STT_PROVIDERS`:

```python
STREAMING_STT_PROVIDERS: dict[str, type[IStreamingSTTProvider]] = {
    "deepgram": DeepgramSTTAdapter,
}
```

Add the getter after `get_stt_provider`:

```python
def get_streaming_stt_provider(config: dict[str, Any]) -> IStreamingSTTProvider:
    cls = _lookup(STREAMING_STT_PROVIDERS, config["provider"], "streaming STT")
    return cls(config)
```

Add both names to `__all__`:

```python
    "STREAMING_STT_PROVIDERS",
    "get_streaming_stt_provider",
```

- [ ] **Step 4: Edit `pyproject.toml`**

In the `dependencies = [` list, after the `"anthropic>=0.40",` line, add:

```python
    "deepgram-sdk>=3.7",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_deepgram_adapter.py -q`
Expected: PASS (12 passed).

- [ ] **Step 6: Commit**

```bash
git add src/providers/__init__.py pyproject.toml tests/unit/test_deepgram_adapter.py
git commit -m "providers: register deepgram streaming STT; add deepgram-sdk dep

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Split the engine — `run_turn_text`

`run_turn` currently does STT then LLM→TTS in one method (`src/pipeline/engine.py:208-338`). Extract the LLM→TTS half into `run_turn_text(user_text, ...)`; `run_turn` keeps STT and delegates.

**Files:**
- Modify: `src/pipeline/engine.py`
- Test: `tests/unit/test_engine_run_turn_text.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_engine_run_turn_text.py
from __future__ import annotations

import pytest

from src.interfaces.llm import LLMConfig, LLMMessage
from src.interfaces.stt import STTConfig
from src.interfaces.tts import TTSConfig, TTSResult
from src.pipeline.engine import PipelineConfig, PipelineEngine


class _FakeLLM:
    async def generate(self, messages, config):  # pragma: no cover - unused
        raise NotImplementedError

    async def generate_stream(self, messages, config):
        # JSON envelope; engine extracts response_text incrementally.
        for tok in ['{"response_text": "', "नमस्ते जी।", '", "action": "continue"}']:
            yield tok


class _FakeTTS:
    async def synthesize(self, text, config):
        return TTSResult(audio=b"\x00\x00" * 80, duration_ms=10.0, sample_rate=16000)

    async def synthesize_stream(self, text_stream, config):  # pragma: no cover
        if False:
            yield b""


class _FakeSTT:
    async def transcribe(self, audio, config):  # pragma: no cover - unused here
        raise NotImplementedError

    async def transcribe_stream(self, audio_stream, config):  # pragma: no cover
        if False:
            yield None


def _engine():
    cfg = PipelineConfig(
        stt=STTConfig(language="hi-IN"),
        llm=LLMConfig(response_format="json", max_tokens=256),
        tts=TTSConfig(language="hi-IN", sample_rate=16000),
    )
    return PipelineEngine(_FakeSTT(), _FakeLLM(), _FakeTTS(), cfg)


@pytest.mark.asyncio
async def test_run_turn_text_skips_stt_and_speaks_response():
    engine = _engine()
    sink_calls = []

    async def sink(audio: bytes):
        sink_calls.append(audio)

    result = await engine.run_turn_text(
        "और कुछ benefits हैं?",
        history=[LLMMessage(role="system", content="be Anaaya")],
        audio_sink=sink,
    )
    assert result.user_text == "और कुछ benefits हैं?"
    assert result.metrics.stt_latency_ms == 0
    assert '"response_text"' in result.agent_text
    assert sink_calls  # the response_text sentence was synthesized and sent
    assert "नमस्ते जी।" in "".join(result.sentences_spoken)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_engine_run_turn_text.py -q`
Expected: FAIL with `AttributeError: 'PipelineEngine' object has no attribute 'run_turn_text'`.

- [ ] **Step 3: Refactor `src/pipeline/engine.py`**

Replace the body of `run_turn` from the line `# Build the messages list for the LLM.` (currently `engine.py:244`) through the end of the method (`engine.py:338`) so that `run_turn` delegates, and add `run_turn_text` containing the moved LLM→TTS code.

Replace this block in `run_turn` (everything from `# Build the messages list for the LLM.` to the end of the method):

```python
        # Build the messages list for the LLM.
        messages = list(history) + [
            LLMMessage(role="user", content=stt_result.text)
        ]
        ...  # entire LLM/TTS body and the final `return TurnResult(...)`
```

with:

```python
        # STT done — hand the transcript to the shared LLM->TTS path.
        return await self.run_turn_text(
            stt_result.text,
            history,
            audio_sink,
            cancel_event,
            user_language=stt_result.language,
            user_confidence=stt_result.confidence,
            stt_latency_ms=metrics.stt_latency_ms,
            t_overall=t_overall,
        )
```

Then add the new method directly after `run_turn` (the moved code is identical to the current LLM→TTS body, with the signature and the metrics/return wiring shown below):

```python
    async def run_turn_text(
        self,
        user_text: str,
        history: list[LLMMessage],
        audio_sink: AudioSink,
        cancel_event: Optional[asyncio.Event] = None,
        *,
        user_language: Optional[str] = None,
        user_confidence: float = 1.0,
        stt_latency_ms: int = 0,
        t_overall: Optional[float] = None,
    ) -> TurnResult:
        """LLM->TTS for an already-transcribed user turn (no STT).

        Used by the streaming-STT path: Deepgram has already produced the
        transcript, so we skip STT entirely and run the LLM/TTS overlap.
        """
        cancel_event = cancel_event or asyncio.Event()
        if t_overall is None:
            t_overall = time.perf_counter()
        metrics = TurnMetrics()
        metrics.stt_latency_ms = stt_latency_ms

        messages = list(history) + [LLMMessage(role="user", content=user_text)]

        detector = SentenceDetector()
        full_text_parts: list[str] = []
        sentences_spoken: list[str] = []
        bytes_sent = 0
        first_token_at: Optional[float] = None
        first_audio_at: Optional[float] = None

        t_llm_start = time.perf_counter()
        sentence_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        async def tts_worker() -> None:
            nonlocal first_audio_at, bytes_sent
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    return
                if cancel_event.is_set():
                    continue
                try:
                    result = await self._tts.synthesize(sentence, self._config.tts)
                except Exception:  # noqa: BLE001
                    continue
                if cancel_event.is_set():
                    continue
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                bytes_sent += len(result.audio)
                sentences_spoken.append(sentence)
                await audio_sink(result.audio)

        tts_task = asyncio.create_task(tts_worker())

        is_json = getattr(self._config.llm, "response_format", None) == "json"
        extractor = _SpokenTextExtractor() if is_json else None
        spoke_anything = False

        try:
            async for token in self._llm.generate_stream(messages, self._config.llm):
                if cancel_event.is_set():
                    break
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                full_text_parts.append(token)
                speakable = extractor.feed(token) if extractor is not None else token
                if speakable:
                    spoke_anything = True
                    for sentence in detector.feed(speakable):
                        await sentence_queue.put(sentence)

            if not cancel_event.is_set():
                if is_json and not spoke_anything:
                    for sentence in detector.feed(
                        _speakable_from_json("".join(full_text_parts))
                    ):
                        await sentence_queue.put(sentence)
                for sentence in detector.flush():
                    await sentence_queue.put(sentence)
        finally:
            await sentence_queue.put(None)
            await tts_task

        metrics.llm_total_ms = int((time.perf_counter() - t_llm_start) * 1000)
        if first_token_at is not None:
            metrics.llm_ttft_ms = int((first_token_at - t_llm_start) * 1000)
        if first_audio_at is not None:
            metrics.tts_first_chunk_ms = int((first_audio_at - t_llm_start) * 1000)
            metrics.tts_total_ms = int((time.perf_counter() - first_audio_at) * 1000)
        metrics.total_latency_ms = int((time.perf_counter() - t_overall) * 1000)

        return TurnResult(
            user_text=user_text,
            user_language=user_language,
            user_confidence=user_confidence,
            agent_text="".join(full_text_parts),
            audio_bytes_sent=bytes_sent,
            metrics=metrics,
            cancelled=cancel_event.is_set(),
            sentences_spoken=sentences_spoken,
        )
```

Note: `run_turn` already computes `t_overall` and `metrics.stt_latency_ms` before the STT call; pass `t_overall` through so total-latency timing is unchanged for the batch path.

- [ ] **Step 4: Run tests to verify both paths pass**

Run: `.venv/bin/python -m pytest tests/unit/test_engine_run_turn_text.py tests/unit/test_pipeline_engine.py -q`
(If no `test_pipeline_engine.py` exists, run the whole suite: `.venv/bin/python -m pytest tests/unit -q`.)
Expected: PASS — new test passes and all pre-existing engine tests still pass (behaviour-preserving split).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/engine.py tests/unit/test_engine_run_turn_text.py
git commit -m "engine: split run_turn -> run_turn_text (LLM->TTS without STT)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Agent `handle_turn_text` + `_finish_turn`

`handle_turn` (`src/agents/voicebot.py:118-215`) runs STT via the engine, then records turns / parses response / advances the state machine. Extract the post-pipeline body into `_finish_turn(pipeline_result)` and add `handle_turn_text(user_text, sink)` that produces a `pipeline_result` via `run_turn_text`.

**Files:**
- Modify: `src/agents/voicebot.py`
- Test: `tests/unit/test_voicebot_handle_turn_text.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_voicebot_handle_turn_text.py
from __future__ import annotations

import pytest

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine, State
from src.agents.voicebot import VoiceBotAgent
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.pipeline.engine import TurnMetrics, TurnResult


class _FakeEngine:
    def __init__(self, result):
        self._result = result
        self.calls = []

    async def run_turn_text(self, user_text, history, audio_sink, cancel_event=None, **kw):
        self.calls.append(user_text)
        return self._result


def _agent(engine):
    return VoiceBotAgent(
        session=AgentSession(session_id="t1", lead_data={}),
        state_machine=AgentStateMachine(),
        slot_schema=SlotSchema(),
        script=VoiceBotScript(agent_name="Anaaya", agent_role="sales", company_name="X"),
        engine=engine,
        store=None,
    )


@pytest.mark.asyncio
async def test_handle_turn_text_records_and_advances():
    result = TurnResult(
        user_text="और कुछ benefits हैं?",
        user_language="hi",
        user_confidence=1.0,
        agent_text='{"response_text": "जी हाँ!", "action": "continue", "updated_slots": {}}',
        audio_bytes_sent=10,
        metrics=TurnMetrics(),
    )
    engine = _FakeEngine(result)
    agent = _agent(engine)
    await agent.start()

    sink_calls = []

    async def sink(a):
        sink_calls.append(a)

    outcome = await agent.handle_turn_text("और कुछ benefits हैं?", sink)
    assert engine.calls == ["और कुछ benefits हैं?"]
    assert outcome.response.response_text == "जी हाँ!"
    # user + agent turns recorded (system prompt seeded at construction)
    roles = [t.role for t in agent.session.turns]
    assert roles[-2:] == ["user", "assistant"]
    assert agent.state.state is State.LISTENING


@pytest.mark.asyncio
async def test_handle_turn_text_empty_is_noop():
    result = TurnResult(
        user_text="", user_language=None, user_confidence=0.0,
        agent_text="", audio_bytes_sent=0, metrics=TurnMetrics(),
    )
    agent = _agent(_FakeEngine(result))
    await agent.start()

    async def sink(a):
        pass

    outcome = await agent.handle_turn_text("", sink)
    assert outcome.response.parse_error == "empty STT"
    assert agent.state.state is State.LISTENING
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_voicebot_handle_turn_text.py -q`
Expected: FAIL with `AttributeError: 'VoiceBotAgent' object has no attribute 'handle_turn_text'`.

- [ ] **Step 3: Refactor `src/agents/voicebot.py`**

In `handle_turn`, replace the body from `# Empty STT — no real user turn happened.` (currently line 162) through the final `return TurnOutcome(response=response, pipeline=pipeline_result)` (line 215) with a single call:

```python
        return await self._finish_turn(pipeline_result)
```

Then add these two methods immediately after `handle_turn`. `_finish_turn` is the moved body verbatim (lines 162-215), and `handle_turn_text` mirrors `handle_turn`'s state-guard + resilience:

```python
    async def _finish_turn(self, pipeline_result: TurnResult) -> TurnOutcome:
        """Record turns, parse the structured response, apply slots, and advance
        the state machine. Shared by handle_turn (batch STT) and
        handle_turn_text (streaming STT)."""
        # Empty STT — no real user turn happened. Walk the state machine back to
        # LISTENING and let the silence handler decide what to do next.
        if not pipeline_result.user_text:
            await self.state.fire(Event.LLM_RESPONSE_READY)
            await self.state.fire(Event.RESPONSE_DELIVERED)
            return TurnOutcome(
                response=VoiceBotResponse(
                    response_text="", action="continue", parse_error="empty STT"
                ),
                pipeline=pipeline_result,
            )

        self.session.turns.append(
            LLMMessage(role="user", content=pipeline_result.user_text)
        )
        await self.persist_turn("user", pipeline_result.user_text)

        await self.state.fire(Event.LLM_RESPONSE_READY)

        response = parse_voicebot_response(pipeline_result.agent_text)
        applied = self.slots.apply_updates(response.updated_slots)

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

        if response.action in _ESCALATION_ACTIONS:
            await self.state.fire(Event.ESCALATION_REQUESTED)
        elif response.action in _END_ACTIONS:
            await self.state.fire(Event.RESPONSE_DELIVERED)
            await self.state.fire(Event.HANGUP)
        else:
            await self.state.fire(Event.RESPONSE_DELIVERED)

        await self.persist_state(extra={"last_action": response.action})
        return TurnOutcome(response=response, pipeline=pipeline_result)

    async def handle_turn_text(self, user_text: str, audio_sink: AudioSink) -> TurnOutcome:
        """Drive one turn from an already-transcribed utterance (streaming STT).

        Mirrors handle_turn but skips STT: the transcript is supplied directly.
        """
        if self.state.state is not State.LISTENING:
            raise RuntimeError(
                f"handle_turn_text called from {self.state.state.value}, expected listening"
            )

        await self.state.fire(Event.UTTERANCE_COMPLETE)

        try:
            pipeline_result = await self._engine.run_turn_text(
                user_text,
                self.session.turns,
                audio_sink,
            )
        except Exception as exc:  # noqa: BLE001 - a provider failure must not drop the call
            log.exception("pipeline turn (text) failed; recovering to LISTENING")
            await self.state.fire(Event.LLM_RESPONSE_READY)
            await self.state.fire(Event.RESPONSE_DELIVERED)
            return TurnOutcome(
                response=VoiceBotResponse(
                    response_text="",
                    action="continue",
                    parse_error=f"pipeline error: {type(exc).__name__}: {exc}",
                ),
                pipeline=TurnResult(
                    user_text="",
                    user_language=None,
                    user_confidence=0.0,
                    agent_text="",
                    audio_bytes_sent=0,
                    metrics=TurnMetrics(),
                ),
            )

        return await self._finish_turn(pipeline_result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_voicebot_handle_turn_text.py tests/unit/test_voicebot.py -q`
(If the agent test file has a different name, run `.venv/bin/python -m pytest tests/unit -q -k voicebot`.)
Expected: PASS — new tests pass and existing `handle_turn` tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/voicebot.py tests/unit/test_voicebot_handle_turn_text.py
git commit -m "voicebot: add handle_turn_text + shared _finish_turn

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Tenant config — `pipeline.stt_streaming`

**Files:**
- Modify: `src/config_tenant.py`
- Modify: `config/tenants/dev.yaml`
- Test: `tests/unit/test_tenant_streaming_config.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tenant_streaming_config.py
from __future__ import annotations

from src.config_tenant import TenantPipelineConfig, TenantStreamingSTTConfig


def test_streaming_stt_config_defaults_none():
    p = TenantPipelineConfig()
    assert p.stt_streaming is None


def test_streaming_stt_config_parses():
    p = TenantPipelineConfig(
        stt_streaming={
            "provider": "deepgram",
            "model": "nova-2",
            "language": "hi",
            "endpointing": 300,
            "utterance_end_ms": 1000,
            "api_key_env": "TENANT_DEV_DEEPGRAM_KEY",
        }
    )
    assert isinstance(p.stt_streaming, TenantStreamingSTTConfig)
    assert p.stt_streaming.provider == "deepgram"
    assert p.stt_streaming.endpointing == 300
    assert p.stt_streaming.api_key_env == "TENANT_DEV_DEEPGRAM_KEY"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_tenant_streaming_config.py -q`
Expected: FAIL with `ImportError: cannot import name 'TenantStreamingSTTConfig'`.

- [ ] **Step 3: Edit `src/config_tenant.py`**

Add the model after `TenantSTTConfig` (around line 86):

```python
class TenantStreamingSTTConfig(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    endpointing: Optional[int] = None
    utterance_end_ms: Optional[int] = None
    api_key_env: Optional[str] = None
```

Add the field to `TenantPipelineConfig` (after the `stt:` line, ~line 120):

```python
    stt_streaming: Optional[TenantStreamingSTTConfig] = None
```

- [ ] **Step 4: Edit `config/tenants/dev.yaml`**

Add under `pipeline:` (e.g. right after the `stt:` block, before `llm:`):

```yaml
  # Streaming STT for the dev console (Deepgram live). When present, the browser
  # bridge streams audio live and uses Deepgram endpointing; pipeline.stt (Groq)
  # remains the batch fallback. Telephony bridges ignore this block.
  stt_streaming:
    provider: deepgram
    model: nova-2          # nova-2 hi validated; nova-3 multi also works
    language: hi
    endpointing: 300       # ms of trailing silence -> speech_final (tune live)
    utterance_end_ms: 1000
    api_key_env: TENANT_DEV_DEEPGRAM_KEY
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_tenant_streaming_config.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/config_tenant.py config/tenants/dev.yaml tests/unit/test_tenant_streaming_config.py
git commit -m "config: add pipeline.stt_streaming (deepgram) tenant schema + dev block

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Browser bridge streaming path

The bridge gains an optional `stream_provider`. When present, `run()` opens a Deepgram session, forwards user-only frames to it, consumes events (partials → console, endpoint → dispatch a text turn), enforces an agent-busy gate, and falls back to the existing Silero+batch path on failure. The existing batch behaviour is preserved when `stream_provider is None`.

**Files:**
- Modify: `src/api/browser_bridge.py`
- Test: `tests/unit/test_browser_bridge_streaming.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_browser_bridge_streaming.py
from __future__ import annotations

import asyncio
import json

import pytest

from src.api.browser_bridge import BrowserVoiceBridge, BrowserBridgeConfig
from src.interfaces.stt import STTStreamEvent
from src.pipeline.vad import EnergyVAD


class _FakeWS:
    def __init__(self):
        self.sent_json = []
        self.sent_bytes = []

    async def send_text(self, t):
        self.sent_json.append(json.loads(t))

    async def send_bytes(self, b):
        self.sent_bytes.append(b)


class _ScriptedSession:
    def __init__(self, events):
        self._events = events
        self.sent = []
        self.closed = False

    async def send(self, pcm):
        self.sent.append(pcm)

    async def events(self):
        for ev in self._events:
            await asyncio.sleep(0)
            yield ev

    async def aclose(self):
        self.closed = True


class _FakeProvider:
    def __init__(self, session):
        self._session = session

    async def open_stream(self, config):
        return self._session


class _FakeAgent:
    def __init__(self):
        self.text_turns = []

    async def handle_turn_text(self, text, sink):
        self.text_turns.append(text)
        from src.dialogue.responses import VoiceBotResponse
        from src.pipeline.engine import TurnMetrics, TurnResult

        class _O:
            response = VoiceBotResponse(response_text="जी", action="continue")
            pipeline = TurnResult("u", "hi", 1.0, "{}", 0, TurnMetrics())

        return _O()


def _bridge(events):
    session = _ScriptedSession(events)
    bridge = BrowserVoiceBridge(
        websocket=_FakeWS(),
        agent=_FakeAgent(),
        vad=EnergyVAD(sample_rate=16000, frame_ms=30),
        config=BrowserBridgeConfig(),
        stream_provider=_FakeProvider(session),
    )
    return bridge, session


@pytest.mark.asyncio
async def test_interim_event_emits_partial():
    bridge, session = _bridge([STTStreamEvent(type="interim", text="और कुछ")])
    await bridge._consume_stream_events(session)
    partials = [m for m in bridge._ws.sent_json if m.get("type") == "partial"]
    assert partials and partials[0]["text"] == "और कुछ" and partials[0]["role"] == "user"


@pytest.mark.asyncio
async def test_endpoint_event_dispatches_text_turn():
    bridge, session = _bridge([
        STTStreamEvent(type="interim", text="और कुछ"),
        STTStreamEvent(type="endpoint", text="और कुछ benefits हैं"),
    ])
    await bridge._consume_stream_events(session)
    assert bridge._agent.text_turns == ["और कुछ benefits हैं"]
    # final transcript message sent for the user
    transcripts = [m for m in bridge._ws.sent_json
                   if m.get("type") == "transcript" and m.get("role") == "user"]
    assert transcripts and transcripts[-1]["text"] == "और कुछ benefits हैं"


@pytest.mark.asyncio
async def test_endpoint_ignored_while_agent_busy():
    bridge, session = _bridge([STTStreamEvent(type="endpoint", text="x")])
    bridge._agent_busy = True
    await bridge._consume_stream_events(session)
    assert bridge._agent.text_turns == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_browser_bridge_streaming.py -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'stream_provider'`.

- [ ] **Step 3: Edit `src/api/browser_bridge.py`**

Add the constructor parameter and streaming state (extend `__init__`):

```python
    def __init__(self, websocket, agent, vad: VADDetector,
                 config: BrowserBridgeConfig | None = None,
                 stream_provider=None):
        self._ws = websocket
        self._agent = agent
        self._vad = vad
        self._config = config or BrowserBridgeConfig()
        self._capture_buffer = bytearray()
        self._inbound = bytearray()
        self._frame_bytes = int(self._config.pcm_sample_rate * vad.frame_ms / 1000) * 2
        self._endpoint = EndpointDetector(vad.frame_ms, self._config.endpoint)
        self._stopped = False
        self._play_until = 0.0
        # Streaming STT (optional). When set, audio is streamed live and the
        # turn is dispatched on Deepgram's endpoint event instead of local VAD.
        self._stream_provider = stream_provider
        self._stream_session = None
        self._agent_busy = False
```

Add the event consumer and a text-turn dispatcher. Place these methods in the class (e.g. after `_dispatch_utterance`):

```python
    async def _consume_stream_events(self, session) -> None:
        """Drain streaming-STT events: partials -> console, endpoint -> dispatch."""
        async for ev in session.events():
            if self._stopped:
                return
            if ev.type == "interim":
                await self._send_json(
                    {"type": "partial", "role": "user", "text": ev.text}
                )
            elif ev.type == "endpoint":
                if self._agent_busy or not ev.text.strip():
                    continue
                await self._dispatch_text_turn(ev.text)

    async def _dispatch_text_turn(self, text: str) -> None:
        """Run one turn from an already-transcribed utterance (streaming path)."""
        self._agent_busy = True
        await self._send_json({"type": "status", "status": "thinking"})
        # Clear the live partial line now that we have the final transcript.
        await self._send_json({"type": "partial", "role": "user", "text": ""})
        outcome = await self._agent.handle_turn_text(text, self._send_pcm)

        m = outcome.pipeline.metrics
        log.info(
            "browser turn (stream)",
            extra={
                "user_text": (outcome.pipeline.user_text or "")[:80],
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
        await self._send_json({"type": "status", "status": "listening"})
```

Wire the streaming session into `run()`. Replace the turn-loop section of `run()` so that when a `stream_provider` is set it opens a session, starts the consumer task, and forwards frames to it; otherwise it uses the existing local-VAD path. Update `run()`'s try-body:

```python
            # 2) Turn loop.
            stream_task = None
            if self._stream_provider is not None:
                try:
                    from src.interfaces.stt import STTConfig
                    self._stream_session = await self._stream_provider.open_stream(
                        STTConfig(language="hi", sample_rate=self._config.pcm_sample_rate)
                    )
                    stream_task = asyncio.create_task(
                        self._consume_stream_events(self._stream_session)
                    )
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
                    continue
            else:
                exit_reason = "stopped (terminal)"
```

And in `run()`'s `finally`, close the session/task:

```python
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
            await self._agent.handle_hangup()
```

Finally, update `_on_pcm_frame` so that in streaming mode it forwards user-only frames to Deepgram instead of running local endpointing:

```python
    async def _on_pcm_frame(self, pcm16: bytes) -> None:
        # Streaming mode: forward user-only audio to Deepgram. The client mutes
        # the mic while the agent speaks; the _agent_busy gate is belt-and-braces
        # so the agent's own audio is never streamed to the recognizer.
        if self._stream_session is not None:
            if self._agent_busy:
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
```

> Note: the Silero-safety-net path described in the spec (dispatch via local VAD if Deepgram goes silent) is covered here by the simpler rule "on Deepgram send failure, drop `self._stream_session` and resume batch VAD." A full parallel-VAD safety net is deferred (YAGNI) unless live testing shows Deepgram endpointing is unreliable; if so, add a follow-up task to feed `_inbound`/`_endpoint` in parallel during streaming.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_browser_bridge_streaming.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full suite (no regressions in batch path)**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: PASS (all green).

- [ ] **Step 6: Commit**

```bash
git add src/api/browser_bridge.py tests/unit/test_browser_bridge_streaming.py
git commit -m "browser bridge: stream audio to Deepgram, dispatch on endpoint

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Wire the streaming provider into the dev console

Build the Deepgram provider from `pipeline.stt_streaming` (resolving the API key via `tenant.secret`) and pass it to the bridge. Absent config → `stream_provider=None` → batch behaviour.

**Files:**
- Modify: `src/api/dev_console.py`
- Test: append to `tests/unit/test_browser_bridge_streaming.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_browser_bridge_streaming.py`:

```python
def test_build_streaming_provider_from_tenant():
    from types import SimpleNamespace
    from src.api.dev_console import _build_stream_provider

    tenant = SimpleNamespace(
        settings=SimpleNamespace(pipeline=SimpleNamespace(
            stt_streaming=SimpleNamespace(
                provider="deepgram", model="nova-2", language="hi",
                endpointing=300, utterance_end_ms=1000,
                api_key_env="TENANT_DEV_DEEPGRAM_KEY",
            )
        )),
        secret=lambda env: "dg_secret" if env == "TENANT_DEV_DEEPGRAM_KEY" else None,
    )
    provider = _build_stream_provider(tenant)
    assert provider.__class__.__name__ == "DeepgramSTTAdapter"


def test_build_streaming_provider_none_when_unconfigured():
    from types import SimpleNamespace
    from src.api.dev_console import _build_stream_provider

    tenant = SimpleNamespace(
        settings=SimpleNamespace(pipeline=SimpleNamespace(stt_streaming=None)),
        secret=lambda env: None,
    )
    assert _build_stream_provider(tenant) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_browser_bridge_streaming.py -q -k streaming_provider`
Expected: FAIL with `ImportError: cannot import name '_build_stream_provider'`.

- [ ] **Step 3: Edit `src/api/dev_console.py`**

Add the import near the top:

```python
from src.providers import get_streaming_stt_provider
```

Add the helper (e.g. after `_build_browser_vad`):

```python
def _build_stream_provider(tenant: TenantContext):
    """Build a streaming-STT provider from pipeline.stt_streaming, or None.

    Returns None when no streaming config is present (batch behaviour) or when
    the provider can't be constructed (e.g. missing key) — the bridge then
    falls back to batch Groq, so this never blocks a call.
    """
    cfg = getattr(tenant.settings.pipeline, "stt_streaming", None)
    if cfg is None or not getattr(cfg, "provider", None):
        return None
    try:
        merged = {
            "provider": cfg.provider,
            "model": cfg.model,
            "language": cfg.language,
            "endpointing": cfg.endpointing,
            "utterance_end_ms": cfg.utterance_end_ms,
            "api_key": tenant.secret(cfg.api_key_env) if cfg.api_key_env else None,
        }
        return get_streaming_stt_provider(merged)
    except Exception as e:  # noqa: BLE001 - never block a call on streaming setup
        log.warning("streaming STT provider unavailable (%s); using batch", e)
        return None
```

In `factory(...)`, pass it to the bridge:

```python
        return BrowserVoiceBridge(
            websocket=websocket,
            agent=agent,
            vad=_build_browser_vad(),
            config=BrowserBridgeConfig(),
            stream_provider=_build_stream_provider(tenant),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_browser_bridge_streaming.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/api/dev_console.py tests/unit/test_browser_bridge_streaming.py
git commit -m "dev console: wire deepgram streaming provider into the bridge

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Live partial transcripts in the console (UI)

Render Deepgram interim transcripts as a live, in-place line; clear it when the final user transcript arrives. No unit test (static HTML/JS) — manual verification.

**Files:**
- Modify: `static/dev_console.html`

- [ ] **Step 1: Add a live-partial element + handler**

In `static/dev_console.html`, in the `ws.onmessage` handler, add a branch for `partial` alongside the existing `transcript`/`state`/`error` branches:

```javascript
    } else if (msg.type === "partial") {
      renderPartial(msg.text);
    } else if (msg.type === "transcript") {
      if (msg.role === "user") renderPartial("");  // final replaces the live line
      addTranscript(msg.role, msg.text);
```

(Replace the existing `} else if (msg.type === "transcript") {` line with the two lines above so the user-final clears the partial.)

Add the `renderPartial` helper near `addTranscript`:

```javascript
let _partialEl = null;
function renderPartial(text) {
  if (!text) {
    if (_partialEl) { _partialEl.remove(); _partialEl = null; }
    return;
  }
  if (!_partialEl) {
    _partialEl = document.createElement("div");
    _partialEl.className = "turn user";
    _partialEl.style.opacity = "0.55";
    _partialEl.style.fontStyle = "italic";
    $("transcript").appendChild(_partialEl);
  }
  _partialEl.textContent = "🎙 " + text;
  $("transcript").scrollTop = $("transcript").scrollHeight;
}
```

- [ ] **Step 2: Manual verification**

```bash
# from repo root, with TENANT_DEV_DEEPGRAM_KEY in .env
lsof -ti :8765 | xargs kill 2>/dev/null; sleep 1
VOX_DEV_CONSOLE=1 .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8765 --env-file .env
```

Open `http://localhost:8765/dev/voice`, click Start, speak a Hinglish sentence. Expected:
- A greyed italic "🎙 …" line updates live as you speak (interim).
- When you stop, it's replaced by the solid user transcript, the agent replies, and the server log shows a `browser turn (stream)` line.
Tune `endpointing` in `dev.yaml` if turn-end feels too eager/slow.

- [ ] **Step 3: Commit**

```bash
git add static/dev_console.html
git commit -m "dev console UI: live partial transcript line for streaming STT

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: End-to-end validation & latency comparison

**Files:** none (manual + measurement)

- [ ] **Step 1: Full suite green**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 2: Live A/B**

With streaming active (`pipeline.stt_streaming` present), do ~6 dev-console turns; pull `browser turn (stream)` log lines and compare `tts_first_ms` / `total_ms` against the Gemini-batch baseline already recorded for Groq STT. Confirm: live partials render, Devanagari transcripts are correct, turn-end latency is at/below the old 600 ms+390 ms.

- [ ] **Step 3: Tune & finalize**

Adjust `endpointing` (start 300 ms) for the best responsiveness-vs-clipping; optionally try `model: nova-3` + `language: multi`. Commit any config change.

- [ ] **Step 4: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill.

---

## Self-Review

**Spec coverage:** interface (T1) ✓ · Deepgram adapter incl. keepalive/parse/UtteranceEnd-backup (T2) ✓ · registry + dep (T3) ✓ · engine `run_turn_text` split (T4) ✓ · `handle_turn_text` + `_finish_turn` (T5) ✓ · config schema + dev.yaml block (T6) ✓ · bridge streaming path, partials, endpoint dispatch, agent-busy gate, fallback (T7) ✓ · dev-console wiring (T8) ✓ · live partials UI (T9) ✓ · validation/tuning (T10) ✓. Half-duplex (client gate + `_agent_busy`) ✓. Non-goals (telephony, barge-in, batch Deepgram) respected ✓.

**Deviation from spec (intentional, noted in T7):** the parallel-Silero safety net is reduced to "on Deepgram send failure, revert to batch VAD." This keeps v1 simpler; a full parallel-VAD net is a deferred follow-up gated on live endpoint reliability — matching the spec's "open tuning item" framing.

**Type/name consistency:** `STTStreamEvent(type,text,confidence,language)`, `ISTTStreamSession.send/events/aclose`, `IStreamingSTTProvider.open_stream`, `DeepgramSTTAdapter`, `DeepgramStreamSession(ws,*,keepalive_interval,start_tasks)`, `get_streaming_stt_provider`, `run_turn_text(user_text,history,audio_sink,cancel_event,*,user_language,user_confidence,stt_latency_ms,t_overall)`, `handle_turn_text(user_text,audio_sink)`, `_finish_turn(pipeline_result)`, `BrowserVoiceBridge(..., stream_provider=)`, `_consume_stream_events`, `_dispatch_text_turn`, `_agent_busy`, `_build_stream_provider` — used consistently across tasks.

**Placeholder scan:** none — every code/test step contains complete code and exact commands.
