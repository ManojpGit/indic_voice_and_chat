"""Twilio call-only smoke test — ring a real phone and speak a message.

Mirror of ``scripts/stringee_smoke_call.py`` for Twilio: proves the Twilio
account + credentials + caller-id number can place a real outbound call
end-to-end at the *telephony* layer. There is NO live-AI bridge and NO public
webhook involved — we pass inline TwiML (``twiml=<Response><Say>…</Say></Response>``)
directly to ``Calls.create``, so no server / ngrok / Cloudflare tunnel is
required.

Usage:
    python scripts/twilio_smoke_call.py --from +15705255679 --to +917349093923

A real, billable call is placed. Watch the destination phone ring and listen
for the spoken message. Trial accounts can only dial *verified* destination
numbers; non-verified targets fail with Twilio error 21219.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Make ``from src...``-style imports resolve regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal ``.env`` loader with ``${VAR}`` expansion (no python-dotenv dep)."""
    if not path.exists():
        return
    vals: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = (part.strip() for part in line.split("=", 1))
        val = re.sub(r"\$\{(\w+)\}", lambda m: vals.get(m.group(1)) or os.environ.get(m.group(1), ""), val)
        vals[key] = val
    for k, v in vals.items():
        os.environ.setdefault(k, v)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Twilio call-only smoke test")
    p.add_argument("--from", dest="from_number", default="+15705255679",
                   help="Twilio-provisioned caller-id number (E.164)")
    p.add_argument("--to", dest="to_number", default="+917349093923",
                   help="Destination phone number (E.164)")
    p.add_argument("--text",
                   default="Hello, this is a test call from the indic voice agent. Twilio is connected.",
                   help="Text spoken via Twilio <Say>")
    p.add_argument("--timeout", type=int, default=30)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _load_dotenv()

    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not (sid and token):
        print("error: TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN not set in environment / .env",
              file=sys.stderr)
        return 2

    from twilio.base.exceptions import TwilioRestException
    from twilio.rest import Client

    client = Client(sid, token)
    twiml = f"<Response><Say>{args.text}</Say></Response>"

    print(f"placing call  from={args.from_number}  to={args.to_number}")
    print(f"  twiml={twiml}")

    try:
        call = client.calls.create(
            to=args.to_number,
            from_=args.from_number,
            twiml=twiml,
            timeout=args.timeout,
        )
    except TwilioRestException as e:
        print(f"\nTwilio rejected the call: HTTP {e.status} code={e.code}", file=sys.stderr)
        print(f"  msg : {e.msg}", file=sys.stderr)
        more = getattr(e, "uri", None)
        if more:
            print(f"  uri : {more}", file=sys.stderr)
        return 1

    print(f"\nOK — call sid={call.sid}  status={call.status}")
    print("Watch the destination phone ring and listen for the spoken message.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
