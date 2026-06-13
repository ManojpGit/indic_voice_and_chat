"""Route tests for Register Tenant (POST /api/v1/tenants)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import tenants
from src.api.deps import get_db_session
from src.auth import secrets as crypto
from src.auth.context import hash_api_token
from src.auth.db_resolver import DbTenantResolver
from src.auth.middleware import set_admin_tokens, set_tenant_resolver
from src.models.database import Base
from src.models.tenant import TenantSecret

ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


@pytest_asyncio.fixture
async def ctx(monkeypatch):
    monkeypatch.setenv("VOX_SECRET_KEY", crypto.generate_key())
    crypto.reset_cache_for_tests()

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async def _session_override():
        async with sm() as session:
            yield session

    resolver = DbTenantResolver(sm)
    await resolver.reload()
    set_tenant_resolver(resolver)
    set_admin_tokens(["admin-token"])

    app = FastAPI()
    app.state.tenant_resolver = resolver
    app.state.tenants = resolver.loaded_settings()
    app.include_router(tenants.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, resolver, sm
    set_tenant_resolver(None)
    set_admin_tokens([])
    crypto.reset_cache_for_tests()
    await engine.dispose()


def _body(**over):
    base = {
        "name": "Acme Telecom",
        "max_concurrent_calls": 3,
        "stt": {"provider": "groq", "model": "whisper-large-v3"},
        "llm": {"provider": "gemini", "model": "gemini-2.5-flash-lite"},
        "tts": {"provider": "sarvam", "model": "bulbul:v3", "voice_id": "anushka", "language": "hi-IN"},
        "telephony": {
            "provider": "twilio",
            "from_number": "+15705255679",
            "keys": {"account_sid": "ACxxx", "auth_token": "tok-secret"},
            "phone_numbers": ["+15705255679"],
        },
    }
    base.update(over)
    return base


async def test_register_returns_token_and_id(ctx) -> None:
    client, _, _ = ctx
    resp = await client.post("/tenants", json=_body(), headers=ADMIN_HEADERS)
    assert resp.status_code == 201
    body = resp.json()
    assert body["tenant_id"].startswith("t_")
    assert body["slug"] == "acme-telecom"
    assert body["api_token"].startswith("vox_")


async def test_register_requires_admin(ctx) -> None:
    client, _, _ = ctx
    assert (await client.post("/tenants", json=_body())).status_code == 401


async def test_register_issued_token_resolves(ctx) -> None:
    client, resolver, _ = ctx
    token = (await client.post("/tenants", json=_body(), headers=ADMIN_HEADERS)).json()["api_token"]
    new_ctx = await resolver.resolve_by_token(hash_api_token(token))
    assert new_ctx is not None
    assert new_ctx.settings.max_concurrent_calls == 3


async def test_register_telephony_keys_encrypted_only(ctx) -> None:
    client, _, sm = ctx
    resp = await client.post("/tenants", json=_body(), headers=ADMIN_HEADERS)
    tenant_id = resp.json()["tenant_id"]
    async with sm() as s:
        rows = (await s.execute(
            select(TenantSecret).where(TenantSecret.tenant_id == tenant_id)
        )).scalars().all()
    assert len(rows) == 2
    assert all("tok-secret" not in r.value_encrypted and "ACxxx" not in r.value_encrypted
               for r in rows)
    token_row = next(r for r in rows if r.name.endswith("AUTH_TOKEN"))
    assert crypto.decrypt(token_row.value_encrypted) == "tok-secret"


async def test_register_telephony_secret_resolves_via_context(ctx) -> None:
    client, resolver, _ = ctx
    token = (await client.post("/tenants", json=_body(), headers=ADMIN_HEADERS)).json()["api_token"]
    new_ctx = await resolver.resolve_by_token(hash_api_token(token))
    tel = new_ctx.settings.pipeline.telephony
    assert new_ctx.secret(tel.auth_token_env) == "tok-secret"
    # STT api_key_env points at the shared master env var name, not stored per tenant.
    assert new_ctx.settings.pipeline.stt.api_key_env == "GROQ_API_KEY"


async def test_register_persists_model_choices(ctx) -> None:
    client, resolver, _ = ctx
    token = (await client.post("/tenants", json=_body(), headers=ADMIN_HEADERS)).json()["api_token"]
    new_ctx = await resolver.resolve_by_token(hash_api_token(token))
    p = new_ctx.settings.pipeline
    assert p.stt.model == "whisper-large-v3"
    assert p.llm.model == "gemini-2.5-flash-lite"   # a non-default variant survives
    assert p.tts.model == "bulbul:v3"               # TTS model now persisted too


async def test_register_duplicate_slug_409(ctx) -> None:
    client, _, _ = ctx
    await client.post("/tenants", json=_body(slug="dup"), headers=ADMIN_HEADERS)
    resp = await client.post("/tenants", json=_body(slug="dup"), headers=ADMIN_HEADERS)
    assert resp.status_code == 409


async def test_register_s2s_mode(ctx) -> None:
    client, resolver, _ = ctx
    body = _body(
        mode="s2s",
        realtime={"provider": "gemini_live", "model": "gemini-3.1-flash-live-preview",
                  "voice": "Aoede", "language_code": "hi-IN"},
    )
    resp = await client.post("/tenants", json=body, headers=ADMIN_HEADERS)
    assert resp.status_code == 201
    new_ctx = await resolver.resolve_by_slug(resp.json()["slug"])
    assert new_ctx.settings.pipeline.mode == "s2s"
    assert new_ctx.settings.pipeline.realtime.api_key_env == "GEMINI_API_KEY"
