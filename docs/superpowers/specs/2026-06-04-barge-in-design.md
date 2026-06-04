# Barge-in (dev console, browser-detected) — Design

**Status:** approved-for-planning
**Date:** 2026-06-04
**Scope:** browser dev console only. Telephony (server-side detection) is designed-for but not built.

## Goal

Let the user interrupt the agent mid-utterance: when the user starts talking
while the agent is speaking, the agent **stops immediately** and addresses the
new input, instead of finishing its (possibly long) response first. Today the
half-duplex gate drops the user's audio during agent playback, so an
interruption is lost until the agent finishes.

## Decisions (from brainstorming)

- **Detection environment:** speakers (echo-prone) — the hard case.
- **Bias:** responsiveness — cut off fast (short detection window).
- **Approach:** **A — browser-side detection** now (best echo rejection + instant
  local stop), with the server's cancel/turn-handling made **transport-agnostic**
  so telephony (B — server-side detection) reuses it later.
- **Toggle:** a "Allow interruptions" checkbox (default on) to disable barge-in
  if a given setup is too echoey.
- **Abandoned response:** dropped (not resumed); the agent's partial spoken text
  is **not** persisted to history (the LLM sees consecutive user turns, which it
  handles fine).

## Architecture

```
agent speaking (turn N response playing)
  └─ browser: VAD on the echo-cancelled mic, armed only during playback
        └─ sustained user speech (~120 ms) →
              stopPlayback() (instant, local)         ──┐ no server round-trip
              send {"type":"barge_in"}                  │
              open mic gate (start streaming frames)    │
                                                        ▼
  server bridge: _handle_barge_in()  ◀── transport-agnostic, reused by telephony later
        └─ self._cancel_event.set();  self._agent_busy = False
              └─ engine.run_turn_text sees cancel → stops LLM loop + skips TTS
                    └─ handle_turn_text: record user-N turn, DROP agent response,
                       state → LISTENING, return cancelled outcome
  user's continuing speech → Deepgram → endpoint → turn N+1 (normal)
```

**Separation of concerns:** *detection* (browser VAD) is transport-specific and
swappable; *handling* (`_handle_barge_in` + cancel wiring + cancelled-turn
handling) is shared. Telephony barge-in later = a server-side detector that calls
the same `_handle_barge_in()` plus a transport-specific playback stop.

## Components

### 1. `static/dev_console.html` — detection + instant playback stop

**Track and stop playback nodes.** `playPcm16` currently creates an
`AudioBufferSourceNode` per chunk and frees it on `onended`. Add a module-level
`const activeSources = new Set()`; register each source on create, remove on
`onended`. Add:

```javascript
function stopPlayback() {
  for (const s of activeSources) { try { s.stop(); s.disconnect(); } catch (_) {} }
  activeSources.clear();
  playCursor = audioCtx.currentTime;   // nothing scheduled ahead anymore
}
```

**Barge-in VAD in the worklet handler.** Today the half-duplex gate is:

```javascript
const stillPlaying = playCursor > audioCtx.currentTime + 0.05;
if ($("halfDuplex").checked && (agentSpeaking || stillPlaying)) return;
```

Replace the early `return` with barge-in detection. While the agent is speaking
(`agentSpeaking || stillPlaying`) and barge-in is enabled, compute the frame RMS
of the (already echo-cancelled) mic Float32 buffer and accumulate sustained
speech time; on crossing the window, fire barge-in and fall through to start
streaming. Otherwise drop the frame (return). Tunable constants:

```javascript
const BARGE_RMS = 0.02;   // energy threshold on the AEC'd mic (tune live)
const BARGE_MS  = 120;    // sustained speech required before cutting off
let   bargeMs   = 0;

// inside workletNode.port.onmessage, replacing the half-duplex early-return:
const speaking = agentSpeaking || (playCursor > audioCtx.currentTime + 0.05);
if ($("halfDuplex").checked && speaking) {
  if (!$("bargeIn").checked) { bargeMs = 0; return; }
  let sum = 0; for (let i = 0; i < e.data.length; i++) sum += e.data[i] * e.data[i];
  const rms = Math.sqrt(sum / e.data.length);
  const frameMs = (e.data.length / audioCtx.sampleRate) * 1000;
  bargeMs = rms > BARGE_RMS ? bargeMs + frameMs : 0;
  if (bargeMs < BARGE_MS) return;        // not enough sustained speech yet → drop
  // barge-in confirmed:
  bargeMs = 0;
  stopPlayback();
  agentSpeaking = false;
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "barge_in" }));
  // fall through → this frame and subsequent ones stream to the server
}
```

Reset `bargeMs = 0` whenever the agent isn't speaking (so it doesn't carry across
turns). **UI:** add next to the half-duplex checkbox:

```html
<label><input type="checkbox" id="bargeIn" checked /> Allow interruptions</label>
```

### 2. `src/api/browser_bridge.py` — cancel wiring + transport-agnostic handler

- Constructor: add `self._cancel_event = None`.
- `_dispatch_text_turn`: create `cancel_event = asyncio.Event()`, store
  `self._cancel_event = cancel_event`, pass it to
  `handle_turn_text(text, self._send_pcm, cancel_event=cancel_event)`. In a
  `finally`, clear `self._cancel_event = None`.
- On a **cancelled** outcome (`outcome.pipeline.cancelled`): skip emitting the
  agent transcript and the error; the busy gate was already cleared by
  `_handle_barge_in`; ensure status returns to `listening`; `return`. (The user's
  interruption arrives as the next turn.)
- New transport-agnostic method:

```python
def _handle_barge_in(self) -> None:
    """Cancel the in-flight turn so the agent stops mid-utterance. Idempotent;
    no-op when the agent isn't speaking. Reusable by a future server-side
    (telephony) detector — only the trigger differs."""
    if not self._agent_busy:
        return
    if self._cancel_event is not None:
        self._cancel_event.set()
    self._agent_busy = False
    log.info("barge-in: cancelling current turn")
```

- `run()` receive loop: the existing `text is not None` branch currently
  `continue`s on any control frame. Parse it and route `barge_in`:

```python
text = message.get("text")
if text is not None:
    try:
        ctrl = json.loads(text)
    except (ValueError, TypeError):
        ctrl = {}
    if ctrl.get("type") == "barge_in":
        self._handle_barge_in()
    continue
```

### 3. `src/agents/voicebot.py` — cancel passthrough + cancelled-turn handling

`handle_turn_text` gains an optional `cancel_event` it threads to
`run_turn_text` (still wrapped in the `asyncio.wait_for(TURN_TIMEOUT_S)` from the
hang fix — setting the event makes `run_turn_text` return normally with
`cancelled=True`, so it completes inside the timeout, no conflict). After the
engine call, before `_finish_turn`, handle cancellation:

```python
async def handle_turn_text(self, user_text, audio_sink, cancel_event=None):
    ...  # state guard, UTTERANCE_COMPLETE, try/except wait_for(run_turn_text(..., cancel_event))
    if pipeline_result.cancelled:
        # Barge-in: user interrupted before hearing the reply. Keep the user
        # turn (it was said and processed); drop the abandoned agent response;
        # return to LISTENING. The interruption follows as the next turn.
        if pipeline_result.user_text:
            self.session.turns.append(LLMMessage(role="user", content=pipeline_result.user_text))
            await self.persist_turn("user", pipeline_result.user_text)
        await self.state.fire(Event.LLM_RESPONSE_READY)
        await self.state.fire(Event.RESPONSE_DELIVERED)
        return TurnOutcome(
            response=VoiceBotResponse(response_text="", action="continue", parse_error="barge-in"),
            pipeline=pipeline_result,
        )
    return await self._finish_turn(pipeline_result)
```

`run_turn_text`'s call gains `cancel_event=cancel_event`. `handle_turn` (batch) is
unchanged (no barge-in on the batch path).

### 4. `src/pipeline/engine.py` — already cancel-capable

`run_turn_text` already accepts `cancel_event` and: breaks the LLM token loop on
`cancel_event.is_set()`, skips the tail flush, the TTS worker skips synth/send
when set, and returns `TurnResult(cancelled=cancel_event.is_set())`. **No
structural change.** The implementation task verifies that an event set
*externally mid-stream* stops audio promptly (add a focused test).

## Data flow (one barge-in)

1. Agent speaking (turn N response playing in the browser).
2. User talks → browser VAD on the AEC'd mic accumulates ≥120 ms of speech.
3. Browser: `stopPlayback()` (instant), send `{"type":"barge_in"}`, open mic gate.
4. Server `_handle_barge_in()`: `cancel_event.set()`, `_agent_busy = False`.
5. Engine stops LLM/TTS; `run_turn_text` returns `cancelled=True`;
   `handle_turn_text` records user-N, drops the agent response, → LISTENING.
6. The user's continuing speech → Deepgram → endpoint → turn N+1 (normal path).

## Echo / false-trigger handling (crux: speakers + responsive)

- **Browser AEC** (`echoCancellation: true`, already enabled) is the primary
  defense — it subtracts the agent's voice using the playback reference before the
  VAD sees the mic.
- **Energy VAD** with `BARGE_RMS` threshold + a short sustained window
  (`BARGE_MS≈120`), **armed only during playback**. Both constants are tuned live.
- **Toggle** to disable barge-in for echoey setups.
- **Honest limit:** on speakers with imperfect AEC, residual echo may occasionally
  self-trigger; tuning the threshold/window minimizes it. Headphones eliminate it.

## Error handling / edge cases

| Case | Behaviour |
|---|---|
| `barge_in` arrives after the turn already finished | `_handle_barge_in` no-ops (`_agent_busy` is False); browser playback was ending anyway. |
| Multiple `barge_in` messages | Idempotent (`cancel_event.set()` twice is fine; busy already cleared). |
| Malformed control frame | `json.loads` guarded; unknown types ignored (`continue`). |
| Turn-timeout vs cancel | Setting `cancel_event` returns `run_turn_text` normally → completes within `wait_for`; no `TimeoutError`. |
| Partial TTS already sent to browser | Discarded client-side by `stopPlayback()`; server `_play_until` is moot. |

## Testing

**Unit:**
- `_handle_barge_in()`: when busy, sets the stored `cancel_event` and clears
  `_agent_busy`; when not busy, no-op.
- bridge `run()` routing: a `{"type":"barge_in"}` text frame calls `_handle_barge_in()`.
- `_dispatch_text_turn` on a cancelled outcome: does not emit an agent transcript,
  returns status to listening.
- `handle_turn_text` with a `cancel_event` pre-set (fake engine returns a cancelled
  `TurnResult`): records the user turn only, no agent turn, state LISTENING,
  `parse_error == "barge-in"`.
- engine `run_turn_text`: with `cancel_event` set after the first token, stops
  feeding the audio sink (no further audio chunks) and returns `cancelled=True`.

**Manual (dev console, speakers):** interrupt the agent mid-sentence → confirm it
stops within ~150 ms and addresses the interruption; tune `BARGE_RMS`/`BARGE_MS`;
confirm minimal self-triggers; verify the "Allow interruptions" toggle disables it.

## Non-goals (YAGNI)

- Server-side / telephony detection (designed-for via the shared handler; not built).
- Resuming the interrupted response (abandoned; the new turn supersedes).
- Persisting the agent's partial spoken text to history.
- Barge-in on the batch (`handle_turn`) path.
