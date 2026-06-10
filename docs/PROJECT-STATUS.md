# Project Status

**Last updated:** 2026-06-10

Ground-truth status of what has actually been **built, validated, and worked on** —
as opposed to what merely exists in the tree. Several modules were scaffolded during
the early PRD-phase generation and have **not** been touched since; those are listed
under "Not started" even though code for them exists.

The current focus is a **Hindi-only outbound VoiceBot** (Bharat Matka / "Anaaya"
campaign), tested through the browser dev console.

---

## At a glance

| Area | Status |
|---|---|
| Voice core: STT / LLM / TTS | ✅ done & validated |
| Dev console (browser) + barge-in | ✅ done — primary surface |
| Twilio / Exotel telephony (streaming) | ✅ media bridge done · barge-in ⬜ |
| Stringee telephony (turn-based IVR) | 🟡 built, live-blocked on Stringee's side · barge-in ⬜ |
| Telnyx / Infobip telephony | 🟡 auth scaffold only, media ⬜ |
| Campaign orchestration | ✅ logic done · 🟡 live-call wiring not done |
| **Telephony barge-in** | ⬜ **pending across all telephony** |
| **RAG / ChatBot** | ⬜ **untouched scaffold — not worked on** |
| **Benchmarking** | ⬜ **very basic skeleton — much more to do** |
| **Code-switching / multilingual** | ⬜ **not considered — Hindi-only today** |

---

## ✅ Fully implemented (built & validated)

**Voice core**
- **STT:** Deepgram streaming (`nova-2 hi`, active dev-console path) + Groq Whisper batch fallback. Tuned and validated.
- **LLM:** Gemini 2.5-flash (active, with transient-error retry hardening) + Anthropic Claude (Haiku 4.5) as a tested one-line swap-in.
- **TTS:** Sarvam (`bulbul:v2`, voice `anushka`) — the only voice; batch + sentence-overlapped streaming.

**Dev console (browser) — the primary working surface**
- Full streaming pipeline (live Deepgram endpointing → Gemini token stream → overlapped Sarvam TTS).
- **Server-side barge-in** shipped (PR #15): sustained-interim detection, headphones-required, behind the "Allow interruptions" toggle. Validated live.
- Post-call outcome analysis wired on this path.

**Telephony — streaming bridges**
- **Twilio + Exotel** media-stream bridges built, wired via `bootstrap.py`, unit-tested. (Turn detection here is batch Silero VAD, not streaming endpointing.)

**Campaign orchestration — logic**
- Scheduler, concurrency cap, rate-limiting, calling-hours gate, retry/backoff, DND filtering, CRM/event-bus hooks. Tested.

---

## 🟡 Partially done

- **Stringee telephony** — turn-based IVR (no media streaming; record → webhook → batch turn → reply WAV) fully built and unit-tested, but **live calls fail on Stringee's side**; never completed end-to-end. Parked pending a Stringee fix.
- **Telnyx + Infobip** — adapters provide auth/JWT scaffolding only; `stream_audio_in`/`stream_audio_out` raise `NotImplementedError`. No media bridge.
- **Campaign → live calling** — the orchestration engine is done, but the live dispatch to a real telephony provider and per-call outcome recording on the campaign path are **not wired/validated** (only the dev console is wired for outcome analysis).

---

## ⬜ Not started / not touched

- **Telephony barge-in — PENDING for all telephony.** Barge-in is **dev-console only**. Twilio/Exotel streaming barge isn't built (their `handle_turn` has no `cancel_event` yet); Stringee has only the coarse SCCO `bargeIn` flag, not real detection. Documented fast-follow, not started.
- **RAG / ChatBot — untouched scaffold.** `src/rag/*`, `src/agents/chatbot.py`, `src/api/chat.py`, `src/api/knowledge.py`, and the FAISS vector store exist as early-generation scaffold but have **not** been worked on, wired to an active tenant, or validated. Not part of current work.
- **Benchmarking — very basic.** `src/benchmarks/*` is an early skeleton; substantial work is still required before it's a usable harness. Not "done."
- **Code-switching / multilingual — not considered.** The system is **Hindi-only** today. The "write `response_text` in Devanagari only" prompt rule merely makes the single-language Hindi path work with the Hindi TTS — it is **not** a multilingual or code-switch feature. No transliteration engine, no second language.
- **Other:** second/fallback TTS provider; multi-instance scale for the Stringee call registry (currently in-memory, single-instance by design).

---

## Known dialogue-quality items (future work)

- **CTA repetition.** The agent over-repeats its call-to-action — nearly every turn
  ends with the "WhatsApp link + 10% first-deposit bonus" push (most visible on
  `send_info` turns). This is a **prompt-tuning issue, not architectural**: the
  `bharat_matka` campaign over-weights that CTA and the system prompt nudges toward
  the objective every turn with no "don't repeat a CTA already made" guard. Fix is a
  campaign-script tweak plus a one-line prompt rule — logged as a dialogue-quality
  improvement, not a blocker.

## Notes on latency (context)

The dominant per-turn cost is **LLM + TTS inference**, not STT or server placement
(see `docs/latency-llm-stt-experiments.md`). Recent live medians (dev console, local):
LLM TTFT ~2.1 s, first spoken word ~3.3 s, full turn ~6.4 s. STT is effectively free on
the streaming path (overlapped with speech).
