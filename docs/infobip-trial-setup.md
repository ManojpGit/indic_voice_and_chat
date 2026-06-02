# Infobip trial: signup → first +91 call

Goal: place a real outbound call to your own Indian mobile number, with
no KYC delay, today.

## 1. Sign up (5 min)

Go to https://portal.infobip.com/signup and sign up with email or
Google/GitHub OAuth. No card, no KYC.

During signup Infobip asks you to verify one phone number via OTP —
**enter your Indian mobile here**, not a US number. The verified number
becomes the only destination you can call on the trial, and the only
caller-ID you can use as `from`. Same number on both ends is fine for
testing.

After signup you land on the portal. From there grab:

| Value | Where |
|-------|-------|
| **Base URL** | top-right of the portal homepage — looks like `https://abc123.api.infobip.com` |
| **API Key** | Developer Hub → API keys → "Generate API key" |

Paste both into `.env`:

```
INFOBIP_BASE_URL=https://abc123.api.infobip.com
INFOBIP_API_KEY=<the generated key>
```

## 2. Enable the Voice channel (1 min) — easily missed

Infobip's portal lands you on **Messaging** by default. The portal home
will look like it only supports SMS/RCS until you explicitly add Voice:

**Portal → Channels → Available channels → "Voice" → Request channel**

The "Request channel" label is misleading — for trial accounts in
supported countries (India included) it's *not* sales-gated. Clicking it
auto-enables the channel. After this, a **Voice & Video** entry appears
in the left nav and the `/calls/1/calls` API starts accepting your key.

Trial voice limits to be aware of:
- 15 outbound calls total during the 60-day trial
- 5-minute max duration per call
- Can only call your OTP-verified mobile (no other destinations)
- USA destinations require contacting support; India is in the auto-enabled list

## 3. Create a Calls Application (2 min)

Infobip's Calls API requires every call to reference a "Calls Application"
that holds the webhook URL + call-config defaults. Create one:

**Portal → Voice & Video → Applications → Create application**

- Name: `vox-agent-dev`
- Type: `Calls` (not WebRTC)
- Webhook URL: where you want call events posted (for now, anything — you
  can leave the placeholder and update once cloudflared is running)
- Save → copy the `id` from the URL or the application detail panel.

Paste it into `.env`:

```
INFOBIP_APPLICATION_ID=<the application id>
```

## 4. Smoke-test auth

From repo root, with the venv active:

```bash
source .venv/bin/activate
set -a && source .env && set +a

python -c "
import asyncio, httpx, os
async def main():
    base = os.environ['INFOBIP_BASE_URL']
    key  = os.environ['INFOBIP_API_KEY']
    async with httpx.AsyncClient(timeout=15.0) as c:
        # cheapest auth-only read: list calls (empty array on a fresh account)
        r = await c.get(f'{base}/calls/1/calls?limit=1',
                        headers={'Authorization': f'App {key}'})
        print('HTTP', r.status_code, r.text[:200])
asyncio.run(main())
"
```

Expected: `HTTP 200` with an empty `results: []` body. A `401` means the
API key or base URL is wrong. A `404` usually means you copied the base
URL with a trailing path.

## 5. Place a real outbound call via the adapter

```bash
python << 'PY'
import asyncio, os
from src.providers.telephony.infobip import InfobipAdapter
from src.interfaces.telephony import CallConfig

async def main():
    a = InfobipAdapter({
        "api_key":        os.environ["INFOBIP_API_KEY"],
        "base_url":       os.environ["INFOBIP_BASE_URL"],
        "application_id": os.environ["INFOBIP_APPLICATION_ID"],
    })
    cfg = CallConfig(
        to_number   = "+91XXXXXXXXXX",   # your verified Indian mobile
        from_number = "+91XXXXXXXXXX",   # same verified number (trial)
        webhook_url = "",                # unused — handled by application
        timeout_seconds = 30,
    )
    session = await a.initiate_call(cfg)
    print("call queued:", session.session_id, "status:", session.status)

asyncio.run(main())
PY
```

Phone should ring within a few seconds. Pick up; you'll hear whatever
the Calls Application's default behavior is (usually Infobip's
"sample" IVR until you point the application webhook at vox-agent).

## 6. What's still missing (Phase 2)

The above only proves **call lifecycle** works. To get the voicebot
talking on the call, we need the Media-Stream bridge: an
`InfobipMediaBridge` that handles binary-WS PCM16 frames (different
framing from Twilio's JSON-wrapped μ-law). That's the parallel to
`ExotelMediaBridge` and will land once you confirm a real call rings
through.

## Pricing notes

Trial includes free credit (~$10) and is good for 60 days. Outbound to
your verified +91 number is included in trial. After 60 days or
credit runs out, you'll need to add a payment method — *no KYC, just
billing*.
