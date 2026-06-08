# Speed (code-only, pure-safe) — design

**Date:** 2026-06-08. **Status:** approved (brainstorm). Scope: reduce real time-to-first-spoken-word
with **zero quality risk**, **code only**. Mumbai cloud co-location is deferred (separate infra effort).

## Goal
Cut the actual latency the user perceives as "the agent's pause after I stop talking," without changing
the LLM model, the TTS voice, the dialogue wording, or audio naturalness.

## Context (from `docs/latency-llm-stt-experiments.md` + live measurement 2026-06-08)
Perceived time-to-first-word ≈ endpoint gap + LLM TTFT + TTS first chunk.
Measured live (8 turns): **endpoint ~0.6s avg** (several hit Deepgram's 1000ms `UtteranceEnd` fallback),
**LLM TTFT ~1.4s** (Gemini, model-inherent), **TTS first chunk ~1s** (Sarvam).

**Already settled / ruled out (do NOT revisit here):** Claude≈Gemini (kept Gemini on reliability);
cloud deploy = modest, deferred; Gemini prompt caching = dead end for TTFT; Deepgram streaming already
adopted; `endpointing` already lowered 300→200. The bottleneck is provider-side LLM + TTS inference.

**Out of scope (quality risk or deferred):** faster/cheaper LLM or TTS voice, prompt-size trimming,
terser replies, sub-sentence first-chunk flush (prosody risk — "Lever C", dropped for now), Mumbai deploy.

## Lever A — Measurement instrumentation (do first; zero risk)
Make the full per-turn latency breakdown visible so every other change is provable.

- In `src/api/browser_bridge.py` `_consume_stream_events`: track the time of the last `interim` event
  (≈ when the user stops speaking) and, on the `endpoint` event, compute `endpoint_gap_ms` (last interim →
  endpoint finalize). Thread it into `_dispatch_text_turn` so it's logged with the turn.
- Add `endpoint_gap_ms` to the existing `"browser turn (stream)"` log line (which already carries
  `llm_ttft_ms / llm_total_ms / tts_first_ms / total_ms`). Now each turn records the whole chain:
  `endpoint_gap → llm_ttft → tts_first → total`.
- This is **permanent observability**, not throwaway DIAG. "Perceived time-to-first-word" =
  `endpoint_gap_ms + tts_first_ms`.
- **Test:** unit test that a sequence of interim events then an endpoint yields a turn carrying a
  non-null `endpoint_gap_ms` (fake stream session + fake agent, like the existing streaming tests).

## Lever B — Sarvam native streaming TTS (spike → adopt only if real)
Today `SarvamTTSAdapter.synthesize_stream` just loops `synthesize` per segment (one HTTP blob per
sentence), so first audio waits for the whole first sentence to synthesize. If Sarvam offers a native
streaming/chunked TTS, first audio can start before the sentence finishes — **same audio, zero quality
risk** — the real code win.

**B1. Spike (gate, before building):** confirm against Sarvam's API
(`https://docs.sarvam.ai/api-reference-docs/text-to-speech`) whether a streaming/WebSocket TTS exists for
`bulbul:v2`/`v3` returning incremental PCM. Capture: endpoint, protocol, audio format, first-chunk
latency vs the current one-shot call. **If it does not exist → stop; document the negative result; Lever B
is dropped** (and the speed effort then rests on the deferred Mumbai deploy).

**B2. Implement (only if B1 is GO):**
- Add real streaming to `SarvamTTSAdapter` (a method yielding `bytes` chunks as they arrive), strip the
  WAV container per chunk as the one-shot path does.
- Wire the engine's `tts_worker` (`src/pipeline/engine.py`) to push the first chunk to the audio sink as
  soon as it arrives instead of awaiting the full-sentence `synthesize`. Keep the per-sentence structure;
  only the first-chunk delivery changes.
- Preserve the cancel/`cancel_event` and barge paths (don't regress turn cancellation).
- **Fallback:** on any streaming failure, fall back to the existing one-shot `synthesize` (a call must
  never be blocked on streaming setup — same principle as the Deepgram streaming fallback).
- **Test:** adapter unit test with a fake streaming response yielding 2–3 chunks → assert chunks are
  yielded incrementally; engine test that first audio is emitted after the first chunk, not the whole
  sentence; fallback test (streaming error → one-shot path).

## Validation
- Capture a **before** table (current main) and an **after** table per lever using the `"browser turn
  (stream)"` metrics (median/mean of `endpoint_gap_ms`, `tts_first_ms`, `total_ms` over ~8–10 turns).
- Keep a lever only if it measurably helps. Record results in `docs/latency-llm-stt-experiments.md`
  (append a dated section) so the decision log stays the single source of truth.

## Sequencing
A (measurement) → B1 (Sarvam streaming spike) → B2 (implement, only if B1 GO). Each measured.

## Files (anticipated)
- `src/api/browser_bridge.py` — endpoint_gap timing (Lever A).
- `src/providers/tts/sarvam.py` — native streaming (Lever B2, if GO).
- `src/pipeline/engine.py` — first-chunk delivery in `tts_worker` (Lever B2, if GO).
- `tests/unit/` — metric test, adapter streaming tests, engine first-chunk test.
- `docs/latency-llm-stt-experiments.md` — append before/after results.
