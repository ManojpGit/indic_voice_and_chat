# Stringee real-time audio: integration recipe

`StringeeAdapter.stream_audio_in` and `stream_audio_out` raise
`NotImplementedError` on purpose. This document explains why, and how to
fill the gap when a tenant actually needs Stringee as their telephony
provider for a live voicebot.

## Status: turn-based IVR is implemented (2026-06-09)

A **turn-based IVR** path is now available as an alternative to the streaming
options below. This path reuses the agent's existing batch pipeline and
requires no external infrastructure:

- On call answer, Stringee fetches a Server Command Code (SCC) from `/api/v1/telephony/stringee/answer`
  which plays an opening greeting (Sarvam TTS, hosted as a transient WAV) and starts recording
  with `recordMessage` (silence-detected).
- When the caller finishes speaking, the recording is POSTed to `/api/v1/telephony/stringee/event/{tenant_slug}`.
  The server downloads the recording, runs a batch turn (Deepgram STT → Gemini LLM → Sarvam TTS),
  hosts the reply WAV, and returns the next SCC (`play` + `recordMessage`). `bargeIn:true` allows
  the caller to interrupt playback.
- Status and audio retrieval: `POST /status/{tenant_slug}` (call lifecycle), `GET /audio/{token}` (hosted audio).

**Configuration** (per-tenant):
- Set `telephony.provider: stringee` in tenant config.
- Env: `STRINGEE_API_KEY_SID` and `STRINGEE_API_KEY_SECRET` (shared across tenants).
- For inbound calls: set the Stringee dashboard number's **Answer URL** to
  `https://<host>/api/v1/telephony/stringee/answer` and **Status URL** to
  `https://<host>/api/v1/telephony/stringee/status/<tenant_slug>`.
- For outbound calls: answer_url is set automatically from `WEBHOOK_BASE_URL`.
- `WEBHOOK_BASE_URL` must be publicly reachable (Stringee fetches hosted audio and posts events).

**Constraints:**
- Turn-based / half-duplex (higher latency + lower interactivity than Twilio/Exotel streaming).
- Single-instance only (in-memory call registry; sticky, no horizontal scaling).
- `transfer` is not yet supported.

See `docs/superpowers/specs/2026-06-09-stringee-ivr-design.md` and
`docs/superpowers/plans/2026-06-09-stringee-ivr.md` for the detailed design and implementation plan.

---

## Why it isn't done in the adapter

Twilio (Media Streams) and Exotel (Voicebot Streaming) both expose a
**server-side** WebSocket protocol that ships audio frames directly. Our
adapter layer can therefore plug straight into the agent without any
external infrastructure.

Stringee does not have that. Their real-time audio paths are:

| Path | Shape | Real-time? | Server-side WS? |
|------|-------|------------|-----------------|
| Call2 Events WS | Lifecycle metadata only — no media | n/a | yes (events only) |
| SCC `record` action | Recording artifact pulled after the call | no | n/a |
| Client SDK + Conference | Bot user joins a conference, exchanges audio via WebRTC | yes | **no — needs a client SDK** |

The third row is the only path that supports a live agent. It requires a
client SDK runtime (JS/Web, Android, iOS, or Flutter) impersonating a
"bot user", which is outside the scope of an HTTP adapter.

## The conference-bridge pattern

To wire Stringee through vox-agent, run a separate `stringee-bot-bridge`
service that hosts the Stringee Web SDK headlessly:

```
                                              +--------------------+
  PSTN caller --[ Stringee ]-- conference ----| Stringee Web SDK   |
                                              | (bot user)         |
                                              +----------+---------+
                                                         | WebRTC audio
                                                         v
                                              +--------------------+
                                              | stringee-bot-bridge|
                                              | (Node or Python +  |
                                              |  pyppeteer/playwrt)|
                                              +----------+---------+
                                                         | WSS PCM16
                                                         v
                                              +--------------------+
                                              | vox-agent          |
                                              | ExotelMediaBridge- |
                                              | compatible WS      |
                                              +--------------------+
```

### Step-by-step

1. **Bot user** — provision a Stringee user (e.g. `agent-bot@<tenant>`)
   per tenant. Mint short-lived JWTs for it from the same secret used
   in `StringeeAdapter._make_access_token`.
2. **Answer URL** — the SCC script returned from your `answer_url`
   creates a conference and connects both the caller and the bot user:
   ```json
   [
     {"action": "connect", "from": {...}, "customData": {...},
      "to": [{"type": "internal", "number": "agent-bot@<tenant>"}],
      "conference": {"name": "call-<call_id>"}}
   ]
   ```
3. **Bridge service** — a separate process loads the Stringee Web SDK
   (e.g. via pyppeteer or playwright running the JS SDK in headless
   Chrome). It authenticates as `agent-bot`, listens for an inbound
   conference invite, and accepts.
4. **Audio transport** — the bridge exposes a WSS endpoint shaped
   *exactly* like our `ExotelMediaBridge` ingress
   (`{event: start|media|stop, stream_sid, media: {payload: base64 PCM16 LE 8kHz}}`).
   vox-agent connects to that WS, runs its pipeline, and ships TTS PCM
   back the same way.
5. **Cleanup** — when either the caller or the bot leaves the conference
   the bridge closes the WS; vox-agent's `handle_hangup` runs as normal.

With that bridge in place, `StringeeAdapter` only needs to be involved
for **call initiation and lifecycle** (which it already supports), and
the existing `ExotelMediaBridge` factory can serve Stringee audio
unchanged — because the wire format on the WSS leg is identical.

## Why not implement the bridge inside vox-agent?

- The Web SDK runtime is a JS-only artifact; embedding pyppeteer pulls
  ~300 MB of Chromium into our service. Keeping it out preserves the
  small image size of the core agent.
- Per-tenant bot identities are an operational concern, not a code
  concern. The bridge naturally lives next to whatever process owns the
  bot users.
- Failing loudly (`NotImplementedError`) at the adapter layer is
  honest: nobody accidentally ships a "silent voicebot" because the
  streaming methods quietly returned empty iterators.

## What's already in place

- `StringeeAdapter.initiate_call`, `hangup`, and JWT minting work and are
  unit-tested (`tests/unit/test_stringee_adapter.py`).
- `stream_audio_in/out` raise `NotImplementedError` with messages
  pointing to this document.
- `transfer` raises `NotImplementedError` pointing at the SCC scripting
  alternative.

To enable Stringee streaming end-to-end:

1. Build the `stringee-bot-bridge` service per the recipe above.
2. Point its WSS endpoint at the vox-agent Exotel stream route (or wire a
   thin `StringeeMediaBridge` if the framing diverges).
3. Replace the `NotImplementedError`s with the appropriate forwarding
   logic (or simply remove them — the streaming layer will already be
   handled by the bridge service).

---

## Live validation checklist

To verify the turn-based IVR path in a real call (after deploy):

1. **Outbound call + greeting playback**
   - Place a test outbound call to +918618795697 with `answer_url = https://<host>/api/v1/telephony/stringee/answer`.
   - Confirm logs show `stringee answer` and the opening greeting plays.
   - **If Stringee rejects the audio:** check server logs for the error; if it's a codec issue, resample
     the hosted WAV to 8 kHz in `StringeeIvrBridge._host`.

2. **Recording + turn execution + reply playback**
   - Speak a message.
   - Confirm an `event` POST arrives at `/api/v1/telephony/stringee/event/{tenant_slug}` with a recording link.
   - Confirm server logs show a turn running (STT → LLM → TTS).
   - Confirm the reply plays back.
   - **If the recording isn't fetchable:** check server logs for decode errors; adjust the `format` parameter
     in the `recordMessage` if needed.

3. **Barge-in interruption**
   - Wait for a reply to play.
   - Speak over it (barge in).
   - Confirm the playback stops and your speech is recorded as a new turn.

4. **Call end + outcome logging**
   - End the call.
   - Confirm logs show `stringee_status` ENDED.
   - Confirm a `call outcome` is logged (if campaign logging is wired).
