# Stringee real-time audio: integration recipe

`StringeeAdapter.stream_audio_in` and `stream_audio_out` raise
`NotImplementedError` on purpose. This document explains why, and how to
fill the gap when a tenant actually needs Stringee as their telephony
provider for a live voicebot.

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
