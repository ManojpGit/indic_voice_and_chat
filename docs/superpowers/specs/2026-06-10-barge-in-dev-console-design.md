# Barge-in v1 (dev console, server-side) — design

**Date:** 2026-06-10. **Status:** approved (brainstorm). **Scope:** real, reliable barge-in on the
**dev-console streaming path**, detected **server-side** via the Deepgram recognizer, **headphones
required**. The cancel core is kept transport-agnostic so streaming telephony (Twilio/Exotel) is a
fast-follow (out of scope here). Supersedes the deferred client-RMS approach in
`docs/superpowers/specs/2026-06-08-barge-in-redesign.md`.

## Goal
Let the user interrupt the agent mid-reply: when the user speaks over the agent (headphones, no echo),
the agent stops within a fraction of a second and the interruption is answered as the next turn —
captured from word one, with no false interrupts on Hindi backchannels ("haan/hmm/achha").

## Why the prior approach failed (recap)
Client-side RMS VAD on the echo-cancelled mic: AEC suppressed the user on speakers; absolute thresholds
were hardware-dependent; the arm-window didn't overlap playback; interruption words were dropped (mic
muted during the turn); warm-streaming broke endpointing (echo). Root fixes: **no echo** (headphones),
**server-side decision** using recognized speech (not energy), **authoritative server play-state**, and
**don't mute the mic during playback**.

## Decisions (from brainstorm)
- **Headphones required.** Barge is a mode tied to the dev-console `bargeIn` toggle (which implies a
  full-duplex, echo-free mic). Speaker users keep today's reliable half-duplex (echo gate, no barge).
- **Sustained-utterance trigger.** Fire barge only after the user's interim transcript has sustained for
  `BARGE_SUSTAIN_MS` (~450 ms, configurable) while the agent is audible — not the first syllable — so
  backchannels don't cut the agent off.
- **Detector = the Deepgram recognizer itself.** A real interim arriving (and sustaining) during agent
  audio is the barge signal; the same utterance endpoints and becomes the next turn.

## Architecture
Everything is in `src/api/browser_bridge.py` (the dev-console bridge) + a few `static/dev_console.html`
changes. Three load-bearing pieces:

### 1. Turns run as a background task (enabler)
Today the consumer loop `await`s `_dispatch_text_turn`, so it's blocked *inside* the reply and can't read
the interims that signal an interruption. Change `_consume_stream_events` to dispatch the turn as a
tracked task (`self._turn_task = asyncio.create_task(self._dispatch_text_turn(...))`) and keep looping.
- Guard against overlapping turns: only dispatch on `endpoint` when **no turn is in flight**
  (`self._agent_busy` is False *and* `_turn_task` is done/None). Set `_agent_busy = True` in the consumer
  **before** creating the task to close the create-task race.
- On bridge teardown (`run()` finally), cancel `_turn_task` if pending (alongside the existing
  `stream_task` cancel).

### 2. Server-side barge detection (in the consumer's interim handler)
On each `interim` event, compute `audible = self._agent_busy or time.monotonic() < self._play_until`.
- If barge enabled and `audible`: on the **first** interim of this audible window set `_barge_start_t =
  now`; on any later interim, if `now - _barge_start_t >= BARGE_SUSTAIN_MS/1000` → fire barge once.
- If **not** audible: reset `_barge_start_t = None` (normal listening; no barge bookkeeping).
- "Fire barge" = call `_handle_barge_in()` **and** send `{"type":"interrupt"}` to the client so it flushes
  buffered playback immediately; then set `_barge_start_t = None` so it doesn't re-fire on the same window.
This measures *sustained recognized speech*: a one-token backchannel produces interims spanning <450 ms
(and usually a single interim) → no barge; a real interruption spans the threshold → barge.

### 3. Generalized cancel core
`_handle_barge_in` currently early-returns `if not self._agent_busy`. Generalize its guard to "agent is
audible": `if not (self._agent_busy or time.monotonic() < self._play_until): return`. Most interruptions
land during **playback**, after generation finished and `_agent_busy` is already False — so the old guard
would have no-op'd exactly when barge is needed. It then: sets the in-flight `cancel_event` if present
(scenario: still generating), sets `_agent_busy = False`, resets `_play_until = 0.0`. (Stopping the
client's playback is handled by the `{"type":"interrupt"}` message from the detector.)

### 4. Feed the mic during playback (barge mode only)
`_on_pcm_frame` currently drops frames while `_agent_busy or now < _play_until` (the echo gate). When
`_barge_enabled`, **skip the gate** and always forward to Deepgram (safe: headphones ⇒ no echo ⇒ the
recognizer hears only the user, so endpointing stays clean). When barge disabled, keep today's gate.

### 5. The interruption becomes the next turn
After barge, `_agent_busy` is False and the same Deepgram utterance keeps going; when the user pauses it
emits an `endpoint`, and the consumer (now unblocked, no turn in flight) dispatches it via the existing
`_dispatch_text_turn` path. No special re-injection — word one onward is already in that utterance.

## Client (`static/dev_console.html`)
- When `bargeIn` is checked, send `{"type":"config","barge":true}` on connect/toggle (and `false` when
  unchecked). This sets `_barge_enabled` server-side.
- **Full-duplex when barge on:** do not mute the mic during playback (today's `halfDuplex`/`speaking`
  gate). The mic must reach the server for detection.
- **Delete** the client-side RMS detector (`BARGE_RMS`, `bargeMs`, the worklet VAD branch) and the
  `bargeArmed` plumbing — detection is now server-side.
- Handle a new server message `{"type":"interrupt"}` → call the existing `stopPlayback()` to flush
  buffered audio instantly.

## Constants / config
- `BARGE_SUSTAIN_MS = 450` (module constant in `browser_bridge.py`); tune live.

## Error handling & edge cases
- **Opening line is not barge-able:** it plays with `_agent_busy = False` and no `_cancel_event`; the
  generalized guard still allows a barge during the opening's `_play_until` window, but there's nothing to
  cancel — so define barge to require an in-flight cancellable turn OR active playback of a *turn*. v1:
  only arm detection after the first real turn (track `_had_turn`), so the greeting always plays fully.
- **Double-fire:** `_barge_start_t = None` after firing prevents re-barge within the same window.
- **Race on dispatch:** `_agent_busy` set before `create_task`; endpoint dispatch guarded on it.
- **Speaker users / barge off:** unchanged — echo gate stays, no background-task path engaged for barge
  (turns may still run as tasks; that's behaviorally identical when no interims arrive during the reply).
- **Deepgram drop mid-turn:** existing `_run_stream_consumer` reopen still applies; a dropped recognizer
  just means no barge until it reopens (degrades to half-duplex), never a crash.

## Testing
- **Unit (`tests/unit/test_browser_bridge_streaming.py`):**
  - sustained interim while `audible` → `_handle_barge_in` fired + `{"type":"interrupt"}` sent + turn task
    cancelled;
  - short backchannel (interims spanning < `BARGE_SUSTAIN_MS`) while audible → **no** barge;
  - interim while **not** audible → no barge, `_barge_start_t` stays None;
  - generalized `_handle_barge_in` fires during playback-only (`_agent_busy` False, `now < _play_until`);
  - barge disabled → echo gate still drops frames during playback (regression);
  - after barge, the interruption's `endpoint` dispatches a new turn (no drop).
- **Live (manual):** dev console with **headphones**: agent speaks a long reply, user says "ruko / nahi
  nahi / a question" → agent cuts within ~0.5 s and answers the interruption; a "haan/hmm" backchannel
  does **not** cut it. Tune `BARGE_SUSTAIN_MS`.

## Out of scope (v1)
- **Streaming telephony barge (Twilio/Exotel)** — fast-follow: reuse `_handle_barge_in` + play-state, add
  a caller-stream VAD detector, and make the telephony `handle_turn` cancellable (it has no `cancel_event`
  today). Separate spec.
- **Speaker/echo support** (server AEC / relative detector) — deferred; headphones is the v1 contract.
- **Stringee turn-based IVR** — only the coarse SCCO `bargeIn` flag (already in the SCCO builders); not
  this streaming barge.
