# Deepgram Streaming STT (dev console) — Design

**Status:** approved-for-planning
**Date:** 2026-06-03
**Scope:** browser dev console only (telephony bridges untouched)

## Goal

Replace post-endpoint **batch** STT (Groq Whisper) with **live streaming** STT
(Deepgram) on the browser dev-console path. Today, after the user stops speaking
the bridge waits 600 ms of trailing silence (Silero `EndpointDetector`) and only
then sends the whole utterance to Groq (~390 ms batch call) before the LLM
starts. Streaming feeds audio to Deepgram *while the user speaks*, so the final
transcript is ready the instant they stop, and turn-end can be tuned below the
fixed 600 ms — cutting per-turn latency. Live partial transcripts are shown in
the console as the user speaks.

## Validation (already done — the gate)

A throwaway spike replayed 8 real captured Hinglish utterances through Deepgram
live Hindi (`nova-2 hi` and `nova-3 multi`) and compared against Groq Whisper on
the same audio. Result: **GO.** Deepgram emits clean **Devanagari** for Hindi
words, keeps English/brand words correctly in Latin ("madam", "immediate",
"benefits", "Alright"), and is **at least as accurate as Groq** — noticeably
better where Groq garbled English into nonsense Devanagari ("इमिजेट" for
*immediate*, "मड़ामी" for *madam*). Mixed-script transcripts are ideal input for
the LLM and never reach the TTS (only the agent's reply does, which stays pure
Devanagari). The spike did **not** validate real-time `speech_final` endpoint
timing (clips were pre-trimmed and replayed faster than real-time) — that is a
tuning item for the build, hedged by the Silero safety net below.

## Decisions (from brainstorming)

- **Deepgram-driven endpointing** (primary), Silero retained as safety net.
- **Dev console only**; telephony stays on batch Groq.
- **Option 1 — streaming-STT interface** (provider-abstracted, consistent with
  the STT/LLM/TTS/telephony pattern), not Deepgram hardcoded in the bridge.
- **Persistent** Deepgram session per browser connection (no per-turn handshake).
- **Half-duplex kept** (no barge-in): the agent's own audio is never streamed to
  Deepgram.
- **Live partial transcripts shown in the console** (required).
- **Groq stays as fallback** (batch) if Deepgram is unavailable.

## Current state (what changes)

- `src/api/browser_bridge.py` — `_on_pcm_frame` slices PCM into VAD frames and
  runs `accumulate_and_detect` (Silero + `EndpointDetector`); on endpoint
  `_dispatch_utterance` sends the full captured buffer to
  `agent.handle_turn(captured, sink)`.
- `src/agents/voicebot.py` — `handle_turn(captured_audio, sink)` calls
  `engine.run_turn(captured_audio, history, sink)`, then records turns / applies
  slots / advances the state machine (lines 162–215 today).
- `src/pipeline/engine.py` — `run_turn(captured_audio, ...)` does STT → LLM→TTS
  (overlapped sentence streaming) → `TurnResult`.
- `src/interfaces/stt.py` — `ISTTProvider` (batch `transcribe`, plus an
  awkward iterator-based `transcribe_stream` that just buffers + batches).

## Architecture

The browser bridge holds **one persistent Deepgram live session per
connection**. User PCM frames flow to Deepgram as they arrive; Deepgram streams
back interim + final transcripts and a `speech_final` endpoint signal. On
`speech_final` the bridge dispatches a turn that runs **LLM→TTS on the streamed
transcript, skipping local STT**. Silero VAD + `EndpointDetector` are retained
as a fallback trigger; batch Groq is the ultimate fallback.

```
browser mic (user-only, client half-duplex gate)
   │  PCM16 16k frames
   ▼
BrowserVoiceBridge._on_pcm_frame ──send(pcm)──▶ DeepgramStreamSession ──▶ Deepgram live WS
   │  (also feeds Silero EndpointDetector as safety net)                      │
   │                                                                          │ events
   ◀────────────────────────  STTStreamEvent (interim│final│endpoint) ◀───────┘
   │
   ├─ interim → {"type":"partial","role":"user","text":…}  →  console live line
   ├─ final   → accumulate text
   └─ endpoint→ agent.handle_turn_text(text, send_pcm) → engine.run_turn_text → LLM→TTS → audio out
```

## Components

### 1. `src/interfaces/stt.py` — streaming contract (additions, no breaking change)

```python
@dataclass
class STTStreamEvent:
    type: str                      # "interim" | "final" | "endpoint"
    text: str                      # segment text; for "endpoint", the full utterance text
    confidence: float = 1.0
    language: Optional[str] = None

class ISTTStreamSession(ABC):
    @abstractmethod
    async def send(self, pcm16: bytes) -> None:
        """Feed one chunk of raw PCM16-LE mono audio to the recognizer."""

    @abstractmethod
    def events(self) -> AsyncIterator[STTStreamEvent]:
        """Yield recognizer events until the session is closed."""

    @abstractmethod
    async def aclose(self) -> None:
        """Flush, close the upstream connection, and cancel background tasks."""

class IStreamingSTTProvider(ABC):
    @abstractmethod
    async def open_stream(self, config: STTConfig) -> ISTTStreamSession:
        """Open a live streaming session."""
```

Streaming knobs (`model`, `language`, `endpointing`, `utterance_end_ms`,
`api_key`) live on the **adapter construction config** — the `pipeline.stt_streaming`
YAML block resolved into the provider dict — exactly like `GroqSTTAdapter` reads
`model`/`api_key` at construction. `open_stream(config: STTConfig)` receives only
the per-stream `STTConfig` (`language`, `sample_rate`), which may override the
construction defaults. `STTConfig` is unchanged — no breaking change to batch
callers.

### 2. `src/providers/stt/deepgram.py` *(new)*

- `DeepgramStreamSession(ISTTStreamSession)` — owns one live websocket
  (`wss://api.deepgram.com/v1/listen`, header `Authorization: Token <key>`,
  query: `encoding=linear16&sample_rate=16000&channels=1&model=…&language=…&
  smart_format=true&interim_results=true&endpointing=<ms>&utterance_end_ms=<ms>`).
  - `send(pcm)` → `ws.send(pcm)`.
  - Background **receiver** parses messages → pushes `STTStreamEvent` onto an
    `asyncio.Queue` that `events()` drains:
    - `Results`, `is_final=false`, non-empty → `interim`
    - `Results`, `is_final=true`, non-empty → `final` (also accumulated)
    - `Results`, `speech_final=true` → `endpoint` (text = accumulated finals;
      reset accumulator)
    - `UtteranceEnd` → `endpoint` (backup, only if no `speech_final` already
      fired for the current utterance)
  - Background **keepalive** sends `{"type":"KeepAlive"}` every ~5 s while no
    audio is flowing, so the socket survives agent-speech gaps.
  - `aclose()` → send `{"type":"CloseStream"}`, close ws, cancel tasks.
- `DeepgramSTTAdapter(IStreamingSTTProvider)` — `__init__(config)` resolves
  `api_key` (config `api_key` or `DEEPGRAM_API_KEY`), `model`, `language`;
  supports a `client`/connector injection seam for tests (mirrors the
  Gemini/Claude adapters). `open_stream(config)` constructs a session.
  Lazy-imports `websockets` (bundled with `deepgram-sdk`); raises a clear error
  if absent.

> Implementation uses the `websockets` library directly (full control over
> framing/keepalive, easy to fake in tests), not the deepgram-sdk callback
> client. `deepgram-sdk` is added as a dependency only to pin a known-good
> `websockets`; the SDK client itself is not used.

### 3. `src/providers/__init__.py`

Add `STREAMING_STT_PROVIDERS = {"deepgram": DeepgramSTTAdapter}` and
`get_streaming_stt_provider(config) -> IStreamingSTTProvider` (same `_lookup`
pattern). Add `anthropic`/existing entries unchanged.

### 4. `src/pipeline/engine.py` — split out the LLM→TTS half

Extract the body of `run_turn` from the `# --- LLM streaming + TTS` section
onward into:

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
) -> TurnResult:
    """LLM→TTS for an already-transcribed user turn (no STT)."""
```

`run_turn(captured_audio, …)` becomes: STT → early-return on empty → delegate to
`run_turn_text(stt_result.text, …, user_language=…, user_confidence=…,
stt_latency_ms=…)`. The batch path is behaviour-preserving; existing engine
tests must still pass. `TurnMetrics.stt_latency_ms` is 0 for the streaming path
(STT overlapped with speech; not separately measurable here).

### 5. `src/agents/voicebot.py` — text-entry turn

Refactor `handle_turn` so the post-pipeline body (today lines ~162–215: empty
check, record turns, parse response, apply slots, advance state machine) becomes
a shared helper:

```python
async def _finish_turn(self, pipeline_result: TurnResult) -> TurnOutcome: ...
```

Then:
- `handle_turn(captured_audio, sink)` = state-guard → `run_turn` (with the same
  try/except resilience that walks back to LISTENING) → `_finish_turn`.
- `handle_turn_text(user_text, sink)` = state-guard → `run_turn_text(user_text,
  self.session.turns, sink)` (same resilience) → `_finish_turn`.

Behaviour for both is identical given a `TurnResult`; only how the result is
produced differs.

### 6. `src/api/browser_bridge.py` — streaming path

- Constructor gains an optional `stream_provider: IStreamingSTTProvider | None`.
  When present, `run()` opens a session (`await stream_provider.open_stream(cfg)`)
  and starts an event-consumer task; on open failure it logs and falls back to
  the existing Silero+batch path for the whole connection.
- `_on_pcm_frame(pcm)`:
  - streaming active **and** agent not busy → `await session.send(pcm)`; also
    feed Silero `EndpointDetector` (safety net) but do **not** dispatch from it
    unless the net triggers (below).
  - streaming inactive → existing Silero+batch behaviour, unchanged.
- Event consumer (`async for ev in session.events()`):
  - `interim` → `{"type":"partial","role":"user","text":ev.text}` to the browser.
  - `final` → accumulate.
  - `endpoint` → if agent not busy and text non-empty → `_dispatch_text_turn(text)`.
- `_dispatch_text_turn(text)` mirrors `_dispatch_utterance` minus STT: status
  `thinking` → `outcome = await agent.handle_turn_text(text, send_pcm)` → send
  final `{"type":"transcript","role":"user",text}` (and clear the partial) →
  agent transcript → state → terminal handling (`_play_until` wait) → reset →
  status `listening`. The existing per-turn metric log line is reused.
- **Agent-busy gate** (server side): track `self._agent_busy` = True from the
  moment a turn is dispatched until status returns to `listening`. While busy,
  frames are not forwarded to Deepgram and `endpoint` events are ignored.
  (Belt-and-suspenders with the client half-duplex gate.)
- **Safety net:** if Silero `EndpointDetector` fires while streaming is active,
  agent is idle, and accumulated final text exists but no `endpoint` event has
  arrived within a short grace (~400 ms) → dispatch with the accumulated text.
  If no Deepgram text exists at all (socket dead) → dispatch the captured buffer
  via the existing batch path.
- `aclose` the session in `run()`'s `finally`.

### 7. `src/api/dev_console.py`

In `make_browser_bridge_factory`, build the streaming provider from tenant
config when `pipeline.stt_streaming` is present
(`get_streaming_stt_provider(merged_cfg)`), and pass it into `BrowserVoiceBridge`.
Absent config → `stream_provider=None` → batch behaviour (unchanged).

### 8. `static/dev_console.html` — live partials

- Handle `msg.type === "partial"`: render/update a single live "interim" line
  (distinct style — e.g. muted/italic, prefixed 🎙) that updates in place as
  interims arrive.
- When a final user `transcript` message arrives, clear the interim line (the
  final replaces it). Reset the interim line on each new utterance.

### 9. `config/tenants/dev.yaml`

```yaml
  # Streaming STT for the dev console (Deepgram live). When present, the browser
  # bridge streams audio live and uses Deepgram endpointing; pipeline.stt (Groq)
  # remains the batch fallback. Telephony bridges ignore this block.
  stt_streaming:
    provider: deepgram
    model: nova-2          # nova-2 hi validated; nova-3 multi also works
    language: hi
    endpointing: 300       # ms of trailing silence → speech_final (tune live)
    utterance_end_ms: 1000
    api_key_env: TENANT_DEV_DEEPGRAM_KEY
```

`pyproject.toml` gains `deepgram-sdk>=3.7` (pins a known-good `websockets`).

## Error handling & fallback

| Situation | Behaviour |
|---|---|
| Session open fails | Log; connection runs in batch mode (Silero + Groq), as today. |
| Socket drops mid-call | Catch, `aclose`, switch that connection to batch mode (one reconnect attempt optional, not required for v1). |
| `speech_final` never fires, Silero endpoint fires, final text exists | Safety net dispatches with accumulated text after ~400 ms grace. |
| Silero endpoint fires, no Deepgram text | Dispatch captured buffer via batch Groq. |
| Empty/short transcript | No-op turn (existing empty-STT path). |
| Agent speaking | Frames not forwarded; endpoints ignored (agent-busy gate). |

## Testing

Unit (no live API — inject fakes, same pattern as the Gemini/Claude adapters):
- `DeepgramStreamSession`: feed scripted Deepgram JSON messages through a fake
  websocket → assert the right `STTStreamEvent`s (interim/final/endpoint, with
  accumulation reset on `speech_final`; `UtteranceEnd` backup only when no
  `speech_final` fired).
- `DeepgramSTTAdapter`: missing-key raises; `open_stream` returns a session;
  query/header construction.
- `engine.run_turn_text`: with fake LLM/TTS, returns a `TurnResult` with the
  given `user_text` and overlapped sentence streaming; `run_turn` still passes
  its existing tests (behaviour-preserving split).
- `voicebot.handle_turn_text`: with a fake engine, drives the state machine
  identically to `handle_turn` (records turns, applies slots, terminal/escalation
  actions, resilience on engine exception).
- `BrowserVoiceBridge` streaming path: a fake `IStreamingSTTProvider`/session
  emitting scripted events → assert partials are sent, `endpoint` dispatches a
  text turn, agent-busy gate suppresses mid-response endpoints, and open-failure
  falls back to the batch path.

Manual: dev-console turns; compare first-audio/total against the Groq baseline
already logged; tune `endpointing` ms; confirm live partials render and clear.

## Non-goals (YAGNI)

- Telephony (Twilio/Exotel) streaming — they keep batch Groq.
- Barge-in / interrupting the agent mid-speech.
- Batch Deepgram (prerecorded) — Groq remains the batch path.
- Reconnect-with-replay beyond a single best-effort reconnect.

## Open tuning items (resolved during the build, not blockers)

- Final `endpointing` ms (start 300, tune for responsiveness vs clipping).
- `nova-2 hi` vs `nova-3 multi` (default `nova-2 hi`; one-line config swap).
- KeepAlive interval vs Deepgram's idle timeout across agent-speech gaps.
- Cost note: Deepgram bills for audio streamed (not idle/keepalive); monitor.
