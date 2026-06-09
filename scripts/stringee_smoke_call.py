"""Stringee call-only smoke test — ring a real phone and speak a message.

This proves the Stringee account + API keys + ``StringeeAdapter`` auth path can
place a real outbound call end-to-end at the *telephony* layer. There is NO
live-AI bridge and NO public webhook involved: Stringee's ``/v1/call2/callout``
accepts inline SCCO ``actions``, so we embed a ``talk`` (text-to-speech) action
directly in the request body.

Auth reuses ``StringeeAdapter._headers()`` — a short-lived ``X-STRINGEE-AUTH``
JWT minted from ``STRINGEE_API_KEY_SID`` / ``STRINGEE_API_KEY_SECRET`` (loaded
here from ``.env`` since python-dotenv isn't installed).

Usage:
    python scripts/stringee_smoke_call.py --from +917971441024 --to +917349093923

A real, billable call is placed. Watch the destination phone ring and listen
for the spoken message to confirm success.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import httpx

# Make ``from src...`` resolve regardless of the current working directory.
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
    p = argparse.ArgumentParser(description="Stringee call-only smoke test")
    p.add_argument("--from", dest="from_number", default="+917971441024",
                   help="Stringee-provisioned caller-id number (E.164)")
    p.add_argument("--to", dest="to_number", default="+917349093923",
                   help="Destination phone number (E.164)")
    p.add_argument("--text",
                   default="Hello, this is a test call from the indic voice agent. Stringee is connected.",
                   help="Text spoken via Stringee text-to-speech")
    p.add_argument("--answer-url", default=None,
                   help="Optional public answer_url; omit to rely on inline actions")
    p.add_argument("--inline-ivr", default=None, metavar="ANSWER_ENDPOINT",
                   help="Live IVR over OUTBOUND: pre-fetch the SCCO from this "
                        "answer endpoint (GET) and embed it inline in the "
                        "callout's actions — because Stringee does NOT fetch "
                        "answer_url on outbound callout. e.g. "
                        "https://<host>/api/v1/telephony/stringee/answer")
    p.add_argument("--dry-run", action="store_true",
                   help="Build + print the request body but do NOT place the call")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _load_dotenv()

    from src.providers.telephony.stringee import StringeeAdapter

    try:
        adapter = StringeeAdapter({"provider": "stringee"})  # env-fallback auth
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    body: dict = {
        "from": {"type": "external", "number": args.from_number, "alias": args.from_number},
        "to": [{"type": "external", "number": args.to_number, "alias": args.to_number}],
    }
    if args.inline_ivr:
        # Stringee does NOT fetch answer_url on outbound callout, so pre-fetch
        # our SCCO from the answer endpoint (a GET — which also registers the
        # call's bridge + hosts the opening audio server-side) and embed it
        # inline in the callout's actions. A consistent call_id makes the SCCO's
        # recordMessage eventUrl point back to the bridge we just registered, so
        # the per-turn loop works.
        import uuid

        call_id = f"ivr-{uuid.uuid4().hex[:12]}"
        sep = "&" if "?" in args.inline_ivr else "?"
        fetch_url = (f"{args.inline_ivr}{sep}call_id={call_id}"
                     f"&from={args.from_number}&to={args.to_number}")
        print(f"GET {fetch_url}")
        sr = httpx.get(fetch_url, timeout=15.0)
        sr.raise_for_status()
        scco = sr.json()
        print(f"  pre-fetched SCCO: {json.dumps(scco, ensure_ascii=False)}")
        body["actions"] = scco
    elif args.answer_url:
        # Plain answer_url in the body — works for INBOUND (dashboard) but NOT
        # outbound callout (Stringee doesn't fetch it). Use --inline-ivr instead.
        body["answer_url"] = args.answer_url
    else:
        body["actions"] = [{"action": "talk", "text": args.text}]

    url = f"{adapter._base_url}/v1/call2/callout"
    print(f"POST {url}")
    print(f"  from={args.from_number}  to={args.to_number}")
    print(f"  answer_url={body.get('answer_url')}  actions={json.dumps(body.get('actions'), ensure_ascii=False)}")

    if args.dry_run:
        print("\n[dry-run] not placing the call. Full body:")
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 0

    resp = httpx.post(url, headers=adapter._headers(), json=body, timeout=30.0)
    print(f"\nHTTP {resp.status_code}")
    try:
        payload = resp.json()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception:
        payload = None
        print(resp.text)

    # Stringee returns ``r == 0`` on success.
    ok = resp.status_code == 200 and isinstance(payload, dict) and payload.get("r") == 0
    if not ok:
        print("\nrequest body was:", file=sys.stderr)
        print(json.dumps(body, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1

    print("\nOK — watch the destination phone ring and listen for the spoken message.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
