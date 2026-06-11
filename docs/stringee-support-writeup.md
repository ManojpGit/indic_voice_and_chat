# Stringee integration — implementation & issue (for Stringee support)

## What we're building
An **outbound IVR**: we place a call via the Stringee `callout` REST API with an
`answer_url`. When the callee picks up, Stringee should GET our `answer_url`, which
returns an SCCO that **plays a greeting** and **records the caller's reply**
(`recordMessage`), looping per turn.

- **Account region:** asia-2 (API base `https://asia-2.api.stringee.com`)
- **Account type:** trial
- **Caller ID (from):** `918204268005`
- **Test destination (to):** `918618795697`

---

## 1. Outbound call — the `callout` request we send

```
POST https://asia-2.api.stringee.com/v1/call2/callout
Headers:
  X-STRINGEE-AUTH: <JWT access token, HS256, header cty="stringee-api;v=1",
                    payload {iss=<keySid>, jti, exp, rest_api:true}>
  Content-Type: application/json

Body:
{
  "from": { "type": "external", "number": "918204267969", "alias": "918204267969" },
  "to":   [ { "type": "external", "number": "918618795697", "alias": "918618795697" } ],
  "answer_url": "https://p01--voice-bot--9f4wlvz8tgw7.code.run/api/v1/telephony/stringee/answer"
}
```

Notes:
- Numbers are **bare digits** (no leading `+`) — `+E.164` is rejected as
  `r:10 FROM/TO_NUMBER_INVALID_FORMAT`.
- We send **no `actions`** and **no `answer_url`** in the payload — the `answer_url`
  is configured on the Stringee **project/dashboard** (`.../api/v1/telephony/stringee/answer`).

**Response (success):** HTTP 200, e.g.
`{"r": 0, "call_id": "call-vn-1-FCHPTIPCL5-178101974345...", ...}`
→ the **call is placed and the destination phone rings**. ✅

---

## 2. What our `answer_url` returns (verified working)

When the `answer_url` is fetched (GET), it returns this SCCO (HTTP 200,
`application/json`). We confirmed this by fetching it ourselves:

```json
[
  { "action": "play",
    "url": "https://p01--voice-bot--9f4wlvz8tgw7.code.run/api/v1/telephony/stringee/audio/<token>",
    "bargeIn": true },
  { "action": "recordMessage",
    "eventUrl": "https://p01--voice-bot--9f4wlvz8tgw7.code.run/api/v1/telephony/stringee/event/dev?call_id=<call_id>",
    "format": "wav", "silenceTimeout": 1500, "beepStart": false }
]
```

- The `play` `url` serves a valid **8 kHz, mono, 16-bit PCM WAV** (`Content-Type:
  audio/wav`, HTTP 200) — the spoken greeting.
- The `answer_url` accepts **GET** (and POST), and resolves whether numbers arrive
  bare or `+E.164`.

---

## 3. The problem

**The call rings, the callee answers, and the call immediately disconnects on
pickup. Stringee never fetches our `answer_url`.**

Evidence from our server access logs (every request to `answer_url` is logged):
- For the real call `call-vn-1-FCHPTIPCL5-178101974345...`, there is **no incoming
  request to `answer_url`** at all after pickup.
- The only requests our `answer_url` ever receives are our **own manual test GETs**
  (which return the valid SCCO above).

So: the `callout` succeeds and the phone rings, but on answer Stringee does **not**
GET our `answer_url`, so there is no SCCO to execute and the call drops. The callee
hears only the Stringee trial "center" preamble, never our greeting.

---

## 4. Questions for Stringee support

1. For an `external → external` (PSTN→PSTN) `callout` with an `answer_url`, **when
   and from where does Stringee fetch the `answer_url`** — on callee answer? Is
   anything required for it to be fetched that we may be missing?
2. In your call logs for `call-vn-1-FCHPTIPCL5-178101974345...`, do you see the
   `answer_url` being fetched, and if not, **why was it skipped** (plan/permission/
   trial limitation / format)? What end reason does Stringee record for the call?
3. Does the **trial account** support executing a custom `answer_url` SCCO on a
   PSTN→PSTN call, or is that restricted to paid plans?
4. Does the `answer_url` need to also be **configured in the project/dashboard**, or
   is the value supplied in the `callout` body sufficient?
5. Is `from.type: external` with a PSTN caller-ID the correct setup for an outbound
   IVR where there is no Stringee SDK user on the `from` side?

We can place a test call on request and share exact timestamps so you can correlate
with your logs.
