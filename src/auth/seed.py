"""One-time bridge: migrate YAML tenants into the DB.

The system used to load `config/tenants/*.yaml` at boot. We now store tenants in
the DB. To survive the cutover without losing the running `dev` tenant, this
seeds the DB **from the YAML files when the tenants table is empty** (idempotent):
each tenant becomes a row (+ phone numbers + API tokens from
`TENANT_<SLUG>_API_TOKENS`), and its **telephony** keys (only) are encrypted into
`tenant_secrets`. STT/LLM/TTS/S2S keys stay in the shared master env.

After the first boot the DB is authoritative and the YAML is ignored.
"""

from __future__ import annotations

import logging
import os

from sqlalchemy import select

from src.auth import secrets as crypto
from src.auth.context import hash_api_token
from src.config_tenant import _resolve_dir, discover_tenant_slugs, load_tenant
from src.models.tenant import Tenant, TenantApiKey, TenantPhoneNumber, TenantSecret

log = logging.getLogger(__name__)


async def seed_tenants_from_yaml(session, tenant_dir=None) -> int:
    """Upsert every YAML tenant into the DB. Returns the number processed."""
    base = _resolve_dir(tenant_dir)
    have_key = bool(os.environ.get(crypto.VOX_SECRET_KEY_ENV))
    count = 0
    for slug in discover_tenant_slugs(base):
        s = load_tenant(slug, base)
        row = await session.get(Tenant, s.id)
        cfg = s.pipeline.model_dump()
        if row is None:
            session.add(Tenant(
                id=s.id, slug=s.slug, name=s.name, status=s.status,
                timezone=s.timezone, default_language=s.default_language,
                mode=s.pipeline.mode, max_concurrent_calls=s.max_concurrent_calls,
                pipeline_config=cfg,
            ))
        else:  # refresh config from YAML
            row.name, row.status, row.timezone = s.name, s.status, s.timezone
            row.default_language, row.mode = s.default_language, s.pipeline.mode
            row.max_concurrent_calls, row.pipeline_config = s.max_concurrent_calls, cfg

        for ph in s.phone_numbers:
            if await session.get(TenantPhoneNumber, ph) is None:
                session.add(TenantPhoneNumber(
                    phone_number=ph, tenant_id=s.id,
                    provider=s.pipeline.telephony.provider or "twilio"))

        raw = os.environ.get(f"TENANT_{slug.upper()}_API_TOKENS", "")
        for tok in (t.strip() for t in raw.split(",") if t.strip()):
            h = hash_api_token(tok)
            if await session.get(TenantApiKey, h) is None:
                session.add(TenantApiKey(token_hash=h, tenant_id=s.id, label="seed"))

        # Telephony keys only → tenant_secrets (encrypted). Stringee reads its keys
        # straight from env; Twilio/Exotel go through tenant.secret(<*_env name>).
        tel = s.pipeline.telephony
        for name in (tel.account_sid_env, tel.auth_token_env):
            value = os.environ.get(name) if name else None
            if name and value and have_key:
                if await session.get(TenantSecret, (s.id, name)) is None:
                    session.add(TenantSecret(
                        tenant_id=s.id, name=name, value_encrypted=crypto.encrypt(value)))
        count += 1

    await session.commit()
    log.info("seeded tenants from YAML", extra={"count": count})
    return count


async def seed_if_empty(sessionmaker, tenant_dir=None) -> int:
    """Seed from YAML only when the tenants table is empty (boot-safe bridge)."""
    async with sessionmaker() as session:
        if (await session.execute(select(Tenant.id).limit(1))).first() is not None:
            return 0
        return await seed_tenants_from_yaml(session, tenant_dir)
