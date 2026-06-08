# Barge-in (dev console) — deferred; redesign notes

**Date:** 2026-06-08. **Status:** barge-in DEFERRED. Disabled by default in the dev console
(`static/dev_console.html`: "Allow interruptions (experimental)", unchecked). This doc records why
the current approach doesn't work and what a reliable implementation needs.

## Current behavior (shipped, off by default)
- Client (`static/dev_console.html`): VAD on the echo-cancelled mic during agent audio; on sustained
  speech it `stopPlayback()` + sends `{"type":"barge_in"}`.
- Server (`src/api/browser_bridge.py`): `_handle_barge_in()` sets the in-flight turn's `cancel_event`
  + clears `_agent_busy`; the engine stops; the cancelled turn returns to LISTENING.
- This works only in a narrow case and storms/stalls in common ones (below), so it's now opt-in.

## Why it doesn't work reliably (root causes, found via live instrumentation)
Each was a distinct failure at a different layer — the reason tuning never converged:

1. **Mic AEC suppresses the user on speakers.** With `echoCancellation:true` and speakers, while the
   agent plays, the user's voice is suppressed to ~0.001–0.002 RMS — on top of the echo floor (~0.001).
   A single RMS threshold can't separate them. (Headphones remove this, but the rest still fails.)
2. **Quiet-mic mismatch.** Measured user speech peaked ~0.005 even with no echo; the shipped
   `BARGE_RMS=0.006` was simply above the user's speech. Threshold is hardware-dependent.
3. **Arm window ≠ playback window.** The server armed barge when it finished *sending* audio (~2 s) and
   disarmed then — but the user hears and interrupts during *playback* (~10 s), when it's already
   disarmed. The armed window rarely overlapped the moment of interruption.
4. **Interruption words are dropped.** Mic is muted (client half-duplex + server `_agent_busy`) while
   the agent produces a turn, so STT never hears the start of the interruption. After barge, only
   continued speech is captured → "cuts off but doesn't respond unless you speak again."
5. **Warm-streaming breaks STT.** Forwarding mic to Deepgram continuously (to capture the interruption)
   gives it no clean utterance boundary → it stops endpointing → "permanent listening, not listening."
6. **`stillPlaying` client signal** didn't reliably gate to the audible window in testing (the VAD branch
   wasn't even executing during playback in the last trace) — the WebAudio play-cursor heuristic is fragile.

## What a reliable implementation needs (proposed architecture)
- **Server-side VAD/endpointing on a continuously-open recognizer**, decoupled from turn state, so the
  recognizer always has a clean stream and the *server* (not a browser RMS heuristic) decides "user is
  speaking now." Treat barge as: user-speech-detected while agent audio is outstanding → cancel.
- **Authoritative play-state on the server.** Track when the agent's audio actually finishes playing
  (client ACKs playback progress, or the server paces audio in real time like the telephony bridges do)
  so "is the agent audible" is a server fact, not a client guess. Barge is possible iff agent audible.
- **Capture the interruption from word one.** Keep feeding the recognizer during agent audio (warm) but
  on a *separate* recognition context / with explicit utterance reset on barge, so endpointing isn't
  broken and the interruption is a clean utterance.
- **Echo strategy:** require headphones for the dev console, OR run a dedicated AEC + a relative
  (above-noise-floor) detector rather than an absolute RMS threshold.
- **Telephony reuse:** the cancel/turn-handling core (`_handle_barge_in`) is transport-agnostic; a
  server-side detector on the telephony media stream would call the same entry point. Build the detector
  server-side from the start so it serves both dev console and telephony.

## What shipped instead (this PR)
- **Escalation → ENDED** (`src/agents/voicebot.py`): `schedule_callback`/`transfer` now complete to ENDED
  instead of stranding in ESCALATING (which crashed the next turn → "stuck in thinking"). Verified live.
- **Callback asks for a time** (`src/dialogue/prompts.py`): the agent won't `schedule_callback`/close on a
  vague "kal"; it asks for a concrete time first. Verified live.
- **Outcome-analysis JSON robustness** (already on main, PR #3): tolerant parse of Gemini's fenced JSON.
- Barge-in left in the tree but **off by default**; clean half-duplex turn-taking is the reliable default.
