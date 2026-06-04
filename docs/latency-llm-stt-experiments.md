# Voice Pipeline — Latency, LLM & STT Experiments (Decision Log)

**Period:** 2026-06-02 → 2026-06-04
**Context:** Tuning the multilingual (Hindi/Hinglish) voice agent's perception–reasoning–action pipeline for the Bharat Matka / Anaaya campaign, tested via the browser dev console (no telephony cost).
**Pipeline under test:** STT → LLM (JSON envelope, Devanagari reply) → TTS, with overlapped sentence streaming.
**Test harness:** browser dev console (`/dev/voice`), `dev` tenant, `bharat_matka` campaign. Per-turn metrics logged server-side.

> **How to read the latency metrics.** All `*_ms` figures are measured **inside the server**, from the captured/transcribed utterance onward (server → provider → server). They begin timing at **turn dispatch** and therefore exclude (a) browser↔server transport and (b) the pre-turn endpoint wait. `tts_first_ms` = time from LLM-turn start to the first audio chunk = the best proxy for "time to first spoken word." `total_ms` = full turn (LLM + TTS; for the batch path also STT).

---

## Summary of decisions

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Use Claude instead of Gemini as the LLM? | **No — keep `gemini-2.5-flash`.** Claude adapter kept available behind `provider: anthropic\|claude`. | Latency was a wash; Claude Haiku had a 36s reliability stall on a fresh account. |
| 2 | Will deploying to a cloud instance reduce latency? | **Only modestly.** Worth doing for production reasons (kills ngrok, region co-location, jitter), not a transformation. | The dominant cost is provider-side LLM/TTS inference, which is location-independent. |
| 3 | Adopt Deepgram streaming STT (replacing batch Groq)? | **Yes — keep streaming on for the dev console**, `endpointing=300ms`, model `nova-2 hi`. | Better Hinglish transcription, rock-solid reliability, ~0.7s off the pre-response gap, live partials. |
| 4 | Where is the remaining latency bottleneck? | **The LLM (and TTS) inference**, not STT or server placement. | Consistent across all three experiments: in-turn time ≈ unchanged regardless of STT/host. |

---

## Experiment 1 — Claude as the LLM (vs Gemini 2.5-flash)

**Goal:** evaluate Claude as a latency/quality alternative to Gemini for the dialogue LLM.

**Build:** added `AnthropicClaudeAdapter` (`src/providers/llm/anthropic_claude.py`), implementing `ILLMProvider`, registered as `provider: anthropic | claude`. Default model `claude-haiku-4-5` (fast/cheap tier — the latency play). The pipeline expects a JSON envelope (`response_text` + slots + metadata); Claude has no JSON mode, so the adapter forces a clean `{...}` object via an **assistant `{` prefill**, yielded as the first stream token so the engine's `_SpokenTextExtractor` works unchanged. Prefill/temperature are guarded by model name (both 400 on newer Opus/Sonnet). 9 unit tests; `anthropic` SDK 0.105.2 in `.venv` + pinned in `pyproject.toml`.

**Method:** identical dev-console turns on the same campaign, Gemini block ↔ Claude block swapped in `dev.yaml`, server restarted between, fresh logs each run.

### Readings — Claude Haiku 4.5 (6 turns)

| # | STT | llm_total | first audio | total | note |
|---|---|---|---|---|---|
| 1 | 479 | 3885 | 1920 | 4365 | |
| 2 | 604 | 35866 | **32578** | **36471** | ⚠️ 36s stall (silent SDK retry/backoff — likely 429 on fresh account) |
| 3 | 375 | 5027 | 1954 | 5403 | |
| 4 | 338 | 7325 | 2015 | 7664 | |
| 5 | 450 | 4904 | 1748 | 5355 | |
| 6 | 298 | 6074 | 3108 | 6373 | |

Excluding the turn-2 outlier: **first-audio median 1954ms (mean 2149)**, **full-turn median 5403ms (mean 5832)**.
(`llm_ttft_ms` reads ~0 for Claude — a measurement artifact of the synthetic `{` prefill being the first stream token. Use `tts_first_ms`.)

### Readings — Gemini 2.5-flash (6 turns)

| # | STT | llm_ttft | llm_total | first audio | total |
|---|---|---|---|---|---|
| 1 | 367 | 1620 | 4593 | 2394 | 4960 |
| 2 | 286 | 1179 | 4547 | 2050 | 4833 |
| 3 | 657 | 871 | 4025 | 1950 | 4683 |
| 4 | 361 | 1556 | 6821 | 2720 | 7183 |
| 5 | 284 | 1872 | 5973 | 2771 | 6258 |
| 6 | 259 | 1297 | 3480 | 2528 | 3740 |

**first-audio median 2461ms (mean 2402)**, **full-turn median 4897ms (mean 5276)**, real `llm_ttft` median ~1426ms, **0 stalls**.

### Head-to-head

| metric | Gemini 2.5-flash | Claude Haiku 4.5 | winner |
|---|---|---|---|
| First audio (median) | 2461 ms | **1954 ms** | Claude (~500ms sooner) |
| Full turn (median) | **4897 ms** | 5403 ms | Gemini |
| Stalls / 6 turns | **0** | 1 × 36s | Gemini |

### Decision & rationale
**Keep Gemini 2.5-flash.** Latency is a wash — Claude reaches first-audio ~500ms sooner but Gemini finishes the turn ~500ms faster and was 6/6 clean. The deciding factor was **reliability**: Claude's 36s stall (almost certainly rate-limit backoff on a new Anthropic account) is a dealbreaker for live calls until the tier is raised. Dialogue quality was comparable. The Claude adapter stays in the tree (registered, tested) — a one-line `dev.yaml` flip away if we later want it or raise the tier.

### Related prior finding (pre-session, noted for completeness)
**Gemini prompt caching is a dead end for TTFT here.** At a ~1558-token system prompt, cached vs cold TTFT showed no improvement — TTFT is model-inherent at this prompt size.

---

## Experiment 2 — Would cloud deployment reduce latency?

**Goal:** determine whether moving the server off the local machine (currently behind ngrok) to a cloud instance would cut latency.

**Method:** measured the network floor (TCP+TLS connect) from the host to each provider endpoint, and reasoned about which part of the measured latency is network (movable) vs provider inference (fixed).

### Readings — network floor (host located in **Dubai, AE**)

| Provider | DNS | connect (TCP) | TLS (appconnect) | interpretation |
|---|---|---|---|---|
| Gemini (`generativelanguage.googleapis.com`) | 0.125s | 0.147s | 0.291s | Google edge near Dubai (~22ms TCP RTT) |
| Groq (`api.groq.com`) | 0.212s | 0.222s | 0.242s | edge ~10ms RTT (compute is US) |
| Sarvam (`api.sarvam.ai`) | 0.222s | 0.303s | **0.469s** | **slowest** — origin in India (~80ms RTT) |
| Anthropic (`api.anthropic.com`) | 0.017s | 0.029s | 0.054s | Cloudflare edge in Dubai (~12ms RTT) |

### Analysis
Steady-state per-call network RTT is tens of ms (handshakes reused via keep-alive). Of the ~1.4s Gemini TTFT, only ~100–300ms is network; the rest (~1.1s+) is **queue + model inference on the provider's hardware** — unchanged by where our server runs. The logged `stt_ms`/`tts_first_ms`/`total_ms` are server→provider→server and do **not** include the browser/telephony↔server transport (which ngrok currently inflates).

### Decision & rationale
**Cloud deployment gives a modest improvement, not a transformation.**
- **Will help:** removing ngrok (a real cloud round-trip per audio frame, mostly affecting the telephony media path), region co-location (a **Mumbai** instance sits near Sarvam — the slowest endpoint at 469ms — and near Indian callers/telephony), and stable bandwidth (less jitter).
- **Won't help:** the LLM think-time and TTS synth-time that dominate the visible numbers — those run on the providers regardless.
- **Bigger latency levers (architectural, not host placement):** streaming STT (Experiment 3), tighter end-of-turn VAD, co-locating with the TTS provider, and ultimately the LLM.

**Recommendation:** deploy to cloud for production reasons (needed anyway; kills ngrok; cuts audio-path jitter), but don't expect it to shrink the ~2.5s first-audio figure.

---

## Experiment 3 — Deepgram streaming STT (replacing batch Groq)

**Goal:** feed audio to Deepgram live while the user speaks (so the transcript is ready the instant they stop) and let Deepgram drive endpointing — eliminating the ~390ms post-speech batch STT call and tightening turn-end below the fixed 600ms Silero silence.

### 3a. Validation spike (the gate, before any build)

**Method:** captured 8 **real** Hinglish dev-console utterances, replayed each through Deepgram live (`nova-2 hi` and `nova-3 multi`) and compared transcript **script** and **accuracy** against Groq Whisper on the same audio.

| # | Groq (current) | Deepgram nova-2 hi | verdict |
|---|---|---|---|
| 1 | जी बोलिये **मड़ा** | जी बोलिए **madam** | DG better |
| 3 | ...**इमिजेट**...आएगा **मड़ामी** | ...**immediate** कैसे आएगा **madam?** | **DG much better** |
| 5 | और कुछ benefits हैं | और कुछ benefits हैं? | tie |
| 6 | **अरे और राइट मैं यह** | **Alright ma'am. Yes.** | **DG much better** |

**Result: GO.** Deepgram emits clean **Devanagari** for Hindi and keeps English/brand words correctly in **Latin** ("madam", "immediate", "benefits", "Alright") — accurate where Groq garbled English into nonsense Devanagari. Mixed-script output is ideal LLM input and never reaches the TTS (only the agent's Devanagari reply does). `nova-2 hi` ≈ `nova-3 multi` (nova-2 had a slight spacing edge). The spike did **not** validate real-time `speech_final` endpoint timing (clips were pre-trimmed, replayed faster than real-time) — flagged as a build-time tuning item, hedged by a Silero/Groq fallback.

### 3b. Architecture decisions (brainstorming)

| Decision | Choice | Why |
|---|---|---|
| Endpointing owner | **Deepgram-driven** (`speech_final`), Silero/Groq as fallback | Removes the 390ms batch call AND lets us tune turn-end < 600ms |
| Scope | **Dev console only** | Prove it on the test harness first; telephony stays on batch Groq |
| Integration shape | **Streaming-STT interface** (`ISTTStreamSession` / `IStreamingSTTProvider`) + Deepgram adapter | Consistent with the codebase's provider-abstraction; telephony-reusable; testable |
| Connection | **Persistent** per browser connection (+ KeepAlive) | Avoids a per-turn handshake |
| Half-duplex | **Kept** (no barge-in); client mic-mute + server `_agent_busy` gate | Agent's own audio never streamed to Deepgram |
| Live partials | **Shown in console** (🎙 line) | Immediate "I'm being heard" feedback |
| Fallback | Batch Groq on open-failure or mid-stream send-failure | A call is never blocked on streaming setup |

Spec: `docs/superpowers/specs/2026-06-03-deepgram-streaming-stt-design.md`
Plan: `docs/superpowers/plans/2026-06-03-deepgram-streaming-stt.md`

### 3c. Implementation

9 commits on `main` (`2bd6f34..91e7c03`), TDD, subagent-driven with per-task spec + code-quality review and a final holistic review (**Ready to merge**). **620 unit tests pass.** Key pieces: streaming interface; `DeepgramSTTAdapter` (live websocket, keepalive, pure `_handle_raw` parse matrix); provider registry; `engine.run_turn_text` split (LLM→TTS without STT, batch path behavior-preserved); `voicebot.handle_turn_text` + shared `_finish_turn`; `pipeline.stt_streaming` config; browser-bridge streaming path; dev-console wiring; live-partial UI. `deepgram-sdk` pinned in `pyproject.toml`.

### 3d. Live A/B readings — Deepgram streaming + Gemini (11 turns)

Clean Deepgram session both connections; **zero fallbacks, zero errors, zero stalls.**

| metric | median | mean | range |
|---|---|---|---|
| First audio (`tts_first_ms`) | 2547 | 2519 | 1738–3559 |
| Full turn (`total_ms`) | 4746 | 4531 | 2464–6096 |

**Transcription quality (the headline):** code-switching captured cleanly, e.g.
- "and मुझे थोड़ा **risk** लग रहा है **madam**."
- "ठीक है, **but** फिर भी थोड़ा गलत लग रहा है."
- "अच्छा सच में सिर्फ सौ रुपए से **check** कर सकता हूं?"

### 3e. Interpreting the latency

The in-turn numbers (~2.5s first-audio, ~4.7s total) are **essentially identical to the Gemini-batch baseline** (2461 / 4897) — both run the same Gemini LLM + Sarvam TTS, which dominate, and both clocks start at turn dispatch. **The streaming win sits *before* turn dispatch, off the current metrics:**
- ~390ms batch STT eliminated (transcript already streamed in),
- ~300ms tighter endpoint (Deepgram 300ms vs Silero 600ms),
- ≈ **~0.7s shaved off the gap between "user stops" and "agent starts"**, plus live partials for perceived responsiveness.

### Decision & rationale
**Keep Deepgram streaming on** (`dev.yaml` → `pipeline.stt_streaming`, `nova-2 hi`, `endpointing=300`). Net win = **better Hinglish transcription + reliability + ~0.7s off the pre-response gap + live feedback**. It does **not** shrink the LLM+TTS time. Turn-end at 300ms showed no clipping across 11 turns; kept as the validated default. Optional future follow-up: add an endpoint→dispatch timing log to make the ~0.7s pre-turn win measurable.

---

## Cross-cutting conclusions (for the report)

1. **The LLM (and TTS) inference is the latency bottleneck**, not STT and not server placement. Across all three experiments the in-turn time (~2.5s first-audio) barely moved when we changed the STT, and network analysis showed host placement only touches a small network slice of a provider-bound number.
2. **Established per-stage latency floor (perceived):** end-of-turn endpoint 300–600ms · STT 0ms (streaming, overlapped) or ~390–450ms (batch) · LLM TTFT ~1.3–2s (model-inherent) · TTS first chunk ~500ms+. Perceived time-to-first-word ≈ **~2.0–2.5s**, dominated by LLM TTFT + TTS.
3. **Reliability matters as much as raw speed** for a live phone agent — the Claude experiment was decided on a single 36s stall, not on average latency.
4. **STT quality compounds into dialogue quality** — Deepgram's accurate Hinglish transcription gives the LLM cleaner input than Groq's garbled output, independent of latency.
5. **Next latency levers worth investigating** (in rough priority): faster/streaming TTS or co-locating with Sarvam (Mumbai); reducing LLM output size / a lower-TTFT model without quality loss; production cloud deploy to remove ngrok + jitter.

## Artifacts & how to reproduce

- **LLM adapters:** `src/providers/llm/{gemini,anthropic_claude,groq}.py`; registry `src/providers/__init__.py` (`LLM_PROVIDERS`).
- **Streaming STT:** `src/providers/stt/deepgram.py`; interface `src/interfaces/stt.py`; registry `STREAMING_STT_PROVIDERS`.
- **Config:** `config/tenants/dev.yaml` (`pipeline.llm`, `pipeline.stt`, `pipeline.stt_streaming`). Swap blocks + restart to A/B.
- **Metrics:** server log lines `"browser turn"` (batch) and `"browser turn (stream)"` (streaming) carry `stt_ms / llm_ttft_ms / llm_total_ms / tts_first_ms / total_ms / action / user_text / agent_text`.
- **Keys (gitignored `.env`):** `TENANT_DEV_GEMINI_KEY`, `TENANT_DEV_ANTHROPIC_KEY`, `TENANT_DEV_DEEPGRAM_KEY`, `TENANT_DEV_GROQ_KEY`, `TENANT_DEV_SARVAM_KEY`.
- **Run the console:** `VOX_DEV_CONSOLE=1 .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8765 --env-file .env`, then open `/dev/voice`.
