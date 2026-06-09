# Stringee turn-based IVR voicebot — design

**Date:** 2026-06-09. **Status:** approved (brainstorm). **Scope:** make Stringee a first-class
per-tenant telephony provider for the live voicebot, using Stringee's native SCCO IVR primitives
(turn-based), *without* a headless-browser media bridge.

## Why this shape (research conclusions)
Stringee gives **no server-side media streaming** the way Twilio (Media Streams) and Exotel (Voicebot
Streaming) do:
- SCCO `connect` only targets `to.type` **internal** (Stringee SDK users) or **external** (PSTN) — **no
  external SIP URI**, so a SIP↔media-server bridge is impossible.
- **No media fork / stream-out** action (no Twilio-`<Stream>` equivalent).
- The only real-time audio path is a Stringee **SDK "internal" bot user** → requires a headless Web SDK
  runtime (Chromium per call): operationally heavy, fragile, and costly — which fights the cost reason
  for choosing Stringee. **Rejected** (recorded in `docs/stringee-streaming.md`).

Chosen path: **Stringee-native turn-based IVR.** Stringee's `recordMessage` (silence-detected per-turn
capture → webhook with the utterance audio) + `Put actions` REST (inject `play`/`talk` reply) lets us run
a record→process→reply loop and reuse the agent's **existing batch `handle_turn`** (STT→LLM→TTS). No
browser, no conference, cheap to run. Trade-off: higher per-turn latency (silence timeout + record/upload/
download) and coarser, IVR-level barge-in. References: [SCCO object](https://developer.stringee.com/docs/server/stringee-call-control-object),
[Silence detection + AI bot](https://stringee.com/en/blog/post/silence-detection-and-ai).

Barge-in: Stringee's `play`/`talk` take a **`bargeIn: true`** flag — the prompt stops on detected caller
speech/DTMF and the following `recordMessage` captures the utterance. This is real but coarser than the
streaming path (triggers on any speech/noise; no AEC or sustained-speech thresholds).

## Architecture & per-turn flow
```
Campaign dispatch → StringeeAdapter.initiate_call(answer_url=…)   [already works]
   call answered → Stringee GETs answer_url
   ① return SCCO: play OPENING (bargeIn:true) + recordMessage(eventUrl, silenceTimeout)
   caller speaks → silence detected → Stringee POSTs eventUrl with a link to the utterance audio
   ② StringeeIvrBridge: download audio → agent.handle_turn(bytes, buffering_sink)
        → host reply audio at a public URL
   ③ inject next SCCO via Put actions: play(reply_url, bargeIn:true) + recordMessage(next) → loop to ①
   terminal LLM action / max turns / hangup → play closing + hangup → record outcome (analyze_call)
```
**Reused:** `StringeeAdapter` (init/hangup/JWT), the agent's **batch `handle_turn`**, dialogue/LLM/TTS
pipeline, and post-call `analyze_call`. **New:** SCCO builders, HTTP routes, `StringeeIvrBridge`, a
per-call session registry, and transient audio hosting.

**Design assumptions:** our STT+TTS (Deepgram/Sarvam) for Hindi quality, *not* Stringee's native ASR/TTS;
reply injection via **async `Put actions`** (download→STT→LLM→TTS is too slow to block the webhook
response), with a short "ek minute…" filler if processing runs long; `silenceTimeout` tuned down (~1.5s)
to keep per-turn latency tolerable.

## Components
1. **`StringeeAdapter.put_actions(call_id, actions)`** — new REST method to inject SCCO mid-call. Retries
   transient 5xx like the other adapters.
2. **SCCO builders** (`src/api/telephony_stringee.py`, mirrors `telephony_exotel.py`) — pure functions:
   `answer_scco(opening_url, event_url)`, `reply_scco(reply_url, event_url)`, `closing_scco(closing_url)`.
   Each returns the SCCO JSON (list of action dicts). `play` carries `bargeIn:true`; `recordMessage`
   carries `eventUrl`, `silenceTimeout`, `format`.
3. **HTTP routes** (in `telephony_hooks.py`):
   - `GET/POST /telephony/stringee/answer` → resolve tenant → build agent → render opening → host → SCCO.
   - `POST /telephony/stringee/event/{tenant}` → per-turn `recordMessage` webhook.
   - `GET /telephony/stringee/audio/{token}` → serve hosted reply audio (transient).
   - `POST /telephony/stringee/status/{tenant}` → lifecycle (hangup) → outcome + cleanup.
4. **`StringeeIvrBridge`** (`src/api/telephony_stringee_bridge.py`) — turn controller (IVR analog of
   `ExotelMediaBridge`): `start_call`, `handle_turn(call_id, recording_url)`, `end_call(call_id)`. Uses a
   `BufferingAudioSink` that accumulates PCM instead of streaming it. Registered via a
   `set_stringee_bridge_factory(...)` hook like the Twilio/Exotel factories.
5. **Per-call session registry** — in-memory `dict[call_id → bridge/agent state]`, cleaned up on
   terminal/hangup, with a TTL sweeper for abandoned calls. **Constraint:** single-instance or
   sticky-routed; multi-instance scaling (session externalization) is an explicit follow-up, not v1.
6. **Audio hosting + format** — reply: Sarvam PCM → **WAV** (no ffmpeg dependency) → TTL/LRU map keyed by
   an unguessable token → served at `…/stringee/audio/{token}` → evicted after play/TTL. Inbound:
   download Stringee recording → decode to PCM16 for our batch STT. (Confirm in the plan that Stringee
   `play` accepts the chosen WAV encoding/sample rate; fall back to MP3 only if required.)
7. **Adapter wiring** — `initiate_call` already sends `answer_url=config.webhook_url`; point
   `WEBHOOK_BASE_URL` + `/telephony/stringee/answer` there. Inbound numbers set the same answer_url in the
   Stringee dashboard.

## Error handling & lifecycle
- **Empty/failed STT** → re-prompt ("dobara boliye?") + `recordMessage`, never drop the call.
- **LLM/TTS failure** → fallback line; keep the call alive (reuses agent fallbacks + the Gemini/Sarvam
  retries already on main).
- **Processing latency** → optional short "ek minute…" filler; `Put actions` delivers the reply when ready.
- **Webhook retries/duplicates** → idempotent per turn (dedupe by recording id).
- **Caller hangup** → status webhook → `end_call` → outcome + registry cleanup; TTL sweeper catches calls
  that never send a final event.
- **Terminal LLM action** (`end`/`close_*`) → `closing_scco` (play + hangup).
- **No-input / max turns** → graceful end.

## Testing
- **Unit:** SCCO builders (exact JSON + `bargeIn`); `put_actions` (mock httpx: endpoint/payload, 5xx
  retry); `BufferingAudioSink`; `StringeeIvrBridge.handle_turn` with a fake agent + fake adapter
  (recording-in → `handle_turn` called → reply SCCO; terminal action → closing SCCO; empty STT →
  re-prompt); WAV encode/decode; registry create/lookup/TTL cleanup; tenant resolution.
- **Route/integration:** FastAPI `TestClient` — POST a sample Stringee `recordMessage` payload to
  `/event/{tenant}` → assert returned/injected SCCO; GET `/answer` → assert opening SCCO + that
  `/audio/{token}` serves the opening audio.
- **Reused untouched:** `handle_turn`, dialogue/LLM/TTS, `analyze_call` — covered by existing tests.
- **Live (manual):** one real Stringee call end-to-end — the final gate.

## Out of scope (v1)
- Multi-instance horizontal scaling of the session registry (sticky/single-instance for now).
- Streaming / low-latency audio (architecturally impossible on Stringee — see above).
- Stringee `transfer` (still `NotImplementedError`; SCC-script path is a separate effort).
- MP3 output (WAV unless Stringee rejects it).
