"""Place a single outbound test call through the dev tenant.

Prerequisites (see ``docs/live-testing.md`` for the full setup):

    1. Server running locally:  uvicorn src.main:app --port 8000
    2. ngrok tunneling 8000:    ngrok http 8000
    3. ``config/tenants/dev.yaml`` updated with:
         - ``pipeline.telephony.from_number`` = your Twilio number
         - ``pipeline.telephony.webhook_base_url`` = your ngrok HTTPS URL
         - ``phone_numbers`` list includes your Twilio number
    4. Env vars set: TENANT_DEV_GROQ_KEY, TENANT_DEV_GEMINI_KEY,
       TENANT_DEV_SARVAM_KEY, TENANT_DEV_TWILIO_SID, TENANT_DEV_TWILIO_TOKEN

Usage:
    python scripts/place_test_call.py --to +91XXXXXXXXXX
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Place an outbound test call")
    p.add_argument("--to", required=True, help="Destination phone number in E.164 format (+91...)")
    p.add_argument("--tenant", default="dev", help="Tenant slug (default: dev)")
    p.add_argument("--tenant-dir", default="config/tenants", type=Path)
    p.add_argument("--timeout", type=int, default=30)
    return p


async def _run(args: argparse.Namespace) -> int:
    from src.auth.context import TenantContext
    from src.config_tenant import load_tenant
    from src.interfaces.telephony import CallConfig
    from src.providers.telephony.twilio import TwilioAdapter

    try:
        settings = load_tenant(args.tenant, tenant_dir=args.tenant_dir)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    ctx = TenantContext(settings=settings)
    sid = ctx.secret(settings.pipeline.telephony.account_sid_env)
    token = ctx.secret(settings.pipeline.telephony.auth_token_env)

    from_number = settings.pipeline.telephony.from_number
    if not from_number or from_number.startswith("+1XXXXX") or "XXXX" in from_number:
        print(
            f"error: tenant {args.tenant!r} has a placeholder ``from_number``; "
            f"update config/tenants/{args.tenant}.yaml with your real Twilio number",
            file=sys.stderr,
        )
        return 2

    webhook_base = settings.pipeline.telephony.webhook_base_url or ""
    if "CHANGE-ME" in webhook_base or not webhook_base.startswith("https://"):
        print(
            f"error: tenant {args.tenant!r} webhook_base_url is not set to an HTTPS URL "
            f"(got {webhook_base!r}). Start ngrok and update the YAML.",
            file=sys.stderr,
        )
        return 2

    adapter = TwilioAdapter({"account_sid": sid, "auth_token": token})
    call_cfg = CallConfig(
        to_number=args.to,
        from_number=from_number,
        webhook_url=f"{webhook_base.rstrip('/')}/twilio/voice",
        timeout_seconds=args.timeout,
    )
    print(f"placing call to {args.to} via {from_number}...")
    print(f"  webhook: {call_cfg.webhook_url}")
    session = await adapter.initiate_call(call_cfg)
    print(f"call initiated: sid={session.session_id} status={session.status}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
