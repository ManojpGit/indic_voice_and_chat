# Browser Voice Dev Console — Design

**Date:** 2026-06-02
**Status:** Approved (pending spec review)
**Author:** Manoj + Claude

## Problem

Iterating on dialogue management currently requires placing real phone calls
through Twilio. That is slow and costs money per call, and the dialogue logic
(state machine, slots, prompts) does not need telephony at all. We want a
local, free way to talk to the agent end-to-end while we develop dialogue
management.

## Goal

A browser page where the developer speaks into the laptop mic and hears the
agent reply — exercising the full STT → dialogue → TTS loop over a local
WebSocket — with an on-screen panel showing the live transcript and the
dialogue state/slots after each turn. No telephony, no Twilio cost.

## Non-Goals (YAGNI)

- Per-turn latency panel and raw-LLM-JSON view (the chosen debug panel is
  transcript + state/slots only).
- True WebRTC (RTCPeerConnection / ICE / STUN / TURN). A plain WebSocket with
  browser-captured PCM gives the same mic→speaker experience for a localhost
  tool without the protocol weight.
- Authentication / multi-user access. This is a localhost developer tool.
- Fixing the premature-hangup behaviour observed on the live phone call. That
  is a separate dialogue bug, which this tool is meant to help debug.

## Background: existing architecture

Each telephony transport is a thin **bridge** that handles only the wire
format, wrapping an encoding-agnostic `VoiceBotAgent`:

- `src/api/telephony_twilio.py` — `TwilioMediaBridge`: decodes μ-law@8k from
  base64 JSON frames, resamples to the 16 kHz internal rate, runs VAD +
  endpoint detection, calls `agent.handle_turn(pcm, sink)`, and sends the
  agent's TTS back as μ-law base64 frames.
- `src/api/telephony_exotel.py` — `ExotelMediaBridge`: same shape, different
  wire encoding.
- `src/bootstrap.py` — `make_bridge_factory(...)` builds the per-call
  `VoiceBotAgent` + `PipelineEngine` from a tenant's provider stack and wraps
  it in the transport bridge. `_AgentBridge.run()` plays the opening line once
  the stream `start` event arrives, then pumps media frames.
- `src/api/telephony_hooks.py` — registers WS endpoints and bridge factories.

The agent already produces everything the debug panel needs. After a turn,
`VoiceBotAgent.handle_turn()` returns a `TurnOutcome` with:

- `pipeline: TurnResult` → `user_text` (what STT heard), `agent_text`
  (raw LLM), `metrics`.
- `response: VoiceBotResponse` → `response_text` (spoken reply), `action`,
  `sentiment`, `conversation_phase`, `updated_slots`.

And on the agent itself:

- `agent.state.state.value` — current state-machine state.
- `agent.slots.values` — dict of currently filled slot values
  (`SlotFiller.values` property).

So the browser transport is structurally identical to adding Exotel: a new
bridge + WS endpoint + a static page. The dialogue pipeline is reused
untouched.

## Architecture

```
Browser page (static/dev_console.html)
  mic → AudioWorklet → 16kHz PCM16 ──binary WS frames──▶  /api/v1/dev/voice (WS)
  speaker ◀── Web Audio playback ◀──binary WS frames──    BrowserVoiceBridge
  debug panel ◀── JSON events ◀─────text WS frames─────    (reuses VAD + VoiceBotAgent)
```

### New files

| File | Responsibility |
|------|----------------|
| `static/dev_console.html` | Self-contained vanilla-JS page (HTML + JS + an inline/sibling AudioWorklet). Mic capture, downsample to 16 kHz PCM16, stream binary frames, play back returned audio, render transcript + state/slots. No build step. |
| `src/api/browser_bridge.py` | `BrowserVoiceBridge`: the transport bridge. Reuses the VAD → endpoint → `handle_turn` core; speaks the binary-PCM + JSON protocol below; emits debug events. |
| `src/api/dev_console.py` | FastAPI router: `GET /dev/voice` serves the page; `WS /api/v1/dev/voice` runs the bridge. A `make_browser_bridge_factory(...)` mirrors `make_bridge_factory`. |

### Reused untouched

`VoiceBotAgent`, `PipelineEngine`, `EnergyVAD` / `EndpointDetector`,
`TenantProviders`, and the bridge-factory pattern from `src/bootstrap.py`.

## Wire protocol

A single WebSocket carries two frame types.

**Binary frames** = raw PCM16 little-endian, 16 kHz, mono, both directions.

- Browser → server: continuous mic audio (continuous + server-VAD turn-taking).
- Server → browser: agent TTS audio.
- No base64, no μ-law. The browser captures and resamples to the 16 kHz
  internal rate directly, so there is no quality loss from the 8 kHz μ-law
  telephony path and no server-side resample.

**Text frames (JSON)** = control + debug.

Browser → server:

```json
{"type": "hello", "tenant": "dev"}
```

Sent once on connect. Selects the tenant (default `dev`) before the bridge
plays the opening.

Server → browser:

```json
{"type": "status",     "status": "opening|listening|thinking|speaking"}
{"type": "transcript", "role": "user|agent", "text": "..."}
{"type": "state",      "state": "qualifying", "slots": {"interested": true}}
```

- `status` drives the UI (and the half-duplex mic gate): `speaking` while the
  agent's TTS plays, `listening` when it is the user's turn.
- `transcript` is emitted for the opening line, each user utterance
  (`pipeline.user_text`), and each agent reply (`response.response_text`).
- `state` is emitted after each turn with `agent.state.state.value` and
  `agent.slots.values`.

## BrowserVoiceBridge behaviour

Mirrors `_AgentBridge.run()` but for the browser protocol:

1. On connect: `await websocket.accept()`, read the `hello` frame, resolve the
   tenant, build the agent via the factory.
2. `await agent.start()`, then `await agent.play_opening(self._send_pcm)` —
   send `status: opening`, emit the opening `transcript` and initial `state`,
   then `status: listening`. (Unlike Twilio there is no `start` event to wait
   for; the browser is ready as soon as the socket is open, so the opening
   plays immediately.)
3. Loop on `websocket.receive()`:
   - **bytes** → treat as a mic PCM frame: run VAD; accumulate into the
     capture buffer; when the endpoint detector fires, dispatch the utterance.
   - **text** → JSON control (currently only `hello`, already consumed; ignore
     others for forward-compat).
4. Dispatch (overrides the Twilio dispatch so it can surface debug data):
   send `status: thinking`, `outcome = await agent.handle_turn(captured,
   self._send_pcm)`, then emit `transcript`(user), `transcript`(agent),
   `state`, and finally `status: listening`. If the outcome is terminal, send
   a final `state`/`status` and close.
5. `_send_pcm(pcm16)` → send `status: speaking`, write the PCM as binary
   frames, restore `status: listening` when done. Frames may be chunked but do
   **not** need real-time pacing (unlike Twilio, which buffers and warps
   playback); the browser schedules gapless playback itself.

The VAD/endpoint/capture-buffer logic is the same as the Twilio bridge. To
avoid a third copy, factor the shared "feed PCM → VAD → endpoint → capture →
dispatch" core into a small helper/base that both bridges use; keep the change
minimal and do not alter Twilio/Exotel behaviour.

## Browser page

- `getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true,
  channelCount: 1 } })`.
- `AudioContext` + an `AudioWorkletNode`: pull Float32 capture frames,
  downsample from the context rate (typically 48 kHz) to 16 kHz, convert to
  Int16 PCM, and send each as a binary WS frame — continuously, so the server
  VAD decides turn boundaries.
- Playback: incoming binary PCM16 → Float32 → queued onto the AudioContext
  output for gapless playback.
- UI:
  - **Start / Stop** button (opens/closes the socket + mic).
  - **Mute mic while agent speaks** toggle — default **on**. When on, the page
    stops sending mic frames between `status: speaking` and the following
    `status: listening` (half-duplex). This, plus `echoCancellation`, prevents
    the agent transcribing its own TTS.
  - **Transcript** — running log of user vs. agent turns.
  - **State / slots** box — current state-machine state and filled slots,
    updated each turn.
  - A header note recommending headphones.

## Echo handling

Browser mic + speaker on one machine creates an acoustic loop a phone does not
have: the agent's TTS plays on the speakers, the mic picks it up, the VAD
triggers, and STT transcribes the agent. Mitigations, in order:

1. `echoCancellation: true` on the mic constraints.
2. Default half-duplex: the page mutes mic capture while the agent is speaking
   (driven by the `status` events).
3. Recommend headphones in the page UI.

## Configuration & security

- The dev-console router is mounted only when `VOX_DEV_CONSOLE=1`. Unset by
  default, so the page and WS endpoint are unreachable in production.
- No auth (localhost developer tool). The tenant comes from the `hello` frame,
  defaulting to `dev`.
- Wiring reuses the existing provider registry and bridge-factory construction
  in `src/main.py`'s lifespan; the browser factory is registered alongside the
  Twilio/Exotel factories, gated by the same flag.

## Testing

- **Unit** (`tests/unit/test_browser_bridge.py`): drive `BrowserVoiceBridge`
  with a fake WebSocket and a fake/stub agent. Feed it a `hello` frame and a
  sequence of PCM byte frames that trip the endpoint detector; assert it
  (a) calls `handle_turn` with the captured audio, (b) emits the expected
  `transcript` / `state` / `status` JSON frames, and (c) writes the agent's
  TTS bytes back as binary frames. No real STT/LLM/TTS APIs in CI — mirrors how
  the Twilio bridge is tested.
- **Manual**: `VOX_DEV_CONSOLE=1 .venv/bin/uvicorn src.main:app --port 8765
  --env-file .env`, open `http://localhost:8765/dev/voice`, click Start, hear
  the opening, speak, and watch the transcript and state/slots update while
  hearing the reply.

## Success criteria

1. With `VOX_DEV_CONSOLE=1`, opening `http://localhost:8765/dev/voice` and
   clicking Start plays the agent's opening line through the speakers.
2. Speaking a sentence produces a user transcript, an audible agent reply, an
   agent transcript, and an updated state/slots panel — without any phone call.
3. The production telephony path (Twilio/Exotel) is unchanged, and the console
   is unreachable when `VOX_DEV_CONSOLE` is unset.
4. `BrowserVoiceBridge` unit tests pass in CI with no live audio APIs.
