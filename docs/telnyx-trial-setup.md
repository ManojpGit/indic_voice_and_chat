# Telnyx: signup → first +91 call

Goal: place a real outbound call to your Indian mobile, today, for
~$0.02/call — about 20x cheaper than Infobip's flat per-call fee.

## 1. Sign up (5 min)

Go to https://telnyx.com/sign-up. Email + password, no card upfront.

**Identity verification (auto-approved in ~5 min, not days):**
After signup, you'll be prompted to verify your identity to unlock
outbound calling. Upload a government photo ID (passport, driver's
license, or Aadhaar card). This is NOT the multi-day business KYC
Exotel requires — Telnyx runs it through an automated check and
typically clears within minutes. You may also need to verify your
phone via OTP.

Telnyx gives ~$2 in free credit on signup. That's ~150 test calls to
+91 mobile numbers at $0.012/min. After the credit you need to add a
payment method, but no KYC for that.

## 2. Buy a phone number (~$1 setup, $1/month)

**Mission Control Portal → Numbers → Buy Numbers**

- Country: United States (cheapest, ~$1)
- Type: Local
- Pick any number and complete the purchase

Note: you can also buy an Indian DID directly (~₹500/month) but that
requires their India entity KYC. For testing, a US number works
because Telnyx's outbound voice to +91 uses their own carrier
interconnects regardless of caller-ID region — the US number is just
a billing/identity anchor.

## 3. Create a Voice API Application (2 min)

Telnyx requires every outbound call to reference a "Connection" (also
called Voice API Application) that holds webhook URLs and codec
config.

**Portal → Voice → Programmable Voice → Applications → Add Application**

Pick "Voice API" (not TeXML for our case — we use Call Control, not
TwiML-style XML).

- Application name: `vox-agent-dev`
- Webhook URL: leave a placeholder for now (e.g. `https://example.com/webhook`)
  — we'll update once cloudflared is running.
- Codec preference: `PCMU` (μ-law) → matches our existing audio bridge
- Save

After save, the application's `id` is visible in the URL and on the
detail page. Copy it.

## 4. Generate an API key

**Portal → API Keys → Create API Key V2**

Name it `vox-agent-dev`. Copy the key — it starts with `KEY...`. You
only see it once.

## 5. Wire `.env`

```
TELNYX_API_KEY=KEY01...
TELNYX_CONNECTION_ID=<application id from step 3>
```

## 6. Smoke-test auth

```bash
source .venv/bin/activate
set -a && source .env && set +a

python -c "
import asyncio, httpx, os
async def main():
    key = os.environ['TELNYX_API_KEY']
    async with httpx.AsyncClient(timeout=15.0) as c:
        # cheapest auth-only read: list available phone numbers (or your numbers)
        r = await c.get('https://api.telnyx.com/v2/phone_numbers?page[size]=1',
                        headers={'Authorization': f'Bearer {key}'})
        print('HTTP', r.status_code, r.text[:200])
asyncio.run(main())
"
```

Expected: HTTP 200 with a `data: [...]` array (your purchased number).
HTTP 401 = wrong API key. HTTP 403 = identity verification pending.

## 7. Place a real outbound call via the adapter

```bash
python << 'PY'
import asyncio, os
from src.providers.telephony.telnyx import TelnyxAdapter
from src.interfaces.telephony import CallConfig

async def main():
    a = TelnyxAdapter({
        "api_key":       os.environ["TELNYX_API_KEY"],
        "connection_id": os.environ["TELNYX_CONNECTION_ID"],
    })
    cfg = CallConfig(
        to_number   = "+91XXXXXXXXXX",      # your Indian mobile
        from_number = "+1XXXXXXXXXX",       # the US number you bought in step 2
        webhook_url = "",                   # unused — handled by the Application
        timeout_seconds = 30,
    )
    session = await a.initiate_call(cfg)
    print("call queued:", session.session_id, "status:", session.status)

asyncio.run(main())
PY
```

The call should ring within a few seconds. Pick up — you'll hear
whatever the Voice API Application's default behavior is until we
point its webhook at vox-agent.

## 8. What's still missing (Phase 2)

Streaming audio (so the bot can actually talk on the call) requires:

1. Calling `POST /v2/calls/{id}/actions/streaming_start` to trigger
   Telnyx to open a WSS to our agent.
2. A `TelnyxMediaBridge` at the FastAPI route layer to consume the WSS.

Telnyx's frame format is nearly identical to Twilio's (μ-law @ 8kHz,
JSON envelopes with `event: start | media | stop`), so the existing
`TwilioMediaBridge` should serve with minimal changes — mostly
field-name swaps (`call_control_id` vs `streamSid`). That's a small
Pass 2 follow-up.

## Pricing recap

- US number rental: ~$1/month
- Outbound voice to India: ~$0.012/min
- A typical 90-second test call costs roughly $0.02
- Telnyx's $2 signup credit covers ~150 such calls
