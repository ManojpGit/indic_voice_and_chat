from __future__ import annotations

import textwrap

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.auth import secrets as crypto
from src.auth.context import hash_api_token
from src.auth.db_resolver import DbTenantResolver
from src.auth.seed import seed_if_empty
from src.models import Base


@pytest_asyncio.fixture
async def sm(monkeypatch):
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    crypto.reset_cache_for_tests()
    eng = create_async_engine("sqlite+aiosqlite://")
    async with eng.begin() as c:
        await c.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(eng, expire_on_commit=False)
    await eng.dispose()
    crypto.reset_cache_for_tests()


@pytest.mark.asyncio
async def test_seed_from_yaml_then_resolve(sm, tmp_path, monkeypatch):
    (tmp_path / "demo.yaml").write_text(textwrap.dedent("""
        id: t_demo
        slug: demo
        name: Demo
        timezone: Asia/Kolkata
        max_concurrent_calls: 5
        pipeline:
          mode: layered
          tts: {provider: sarvam, voice_id: anushka, api_key_env: SARVAM_API_KEY}
          telephony: {provider: stringee, from_number: "+91123",
                      account_sid_env: DEMO_SID, auth_token_env: DEMO_TOKEN}
        phone_numbers: ["+91123"]
    """))
    monkeypatch.setenv("VOX_TENANT_DIR", str(tmp_path))
    monkeypatch.setenv("TENANT_DEMO_API_TOKENS", "demo-token")
    monkeypatch.setenv("SARVAM_API_KEY", "master-sarvam")

    assert await seed_if_empty(sm) == 1
    assert await seed_if_empty(sm) == 0          # idempotent — already populated

    r = DbTenantResolver(sm)
    await r.reload()

    ctx = await r.resolve_by_slug("demo")
    assert ctx is not None and ctx.id == "t_demo"
    assert ctx.settings.max_concurrent_calls == 5
    assert ctx.settings.pipeline.tts.voice_id == "anushka"
    assert ctx.settings.pipeline.telephony.provider == "stringee"
    assert ctx.secret("SARVAM_API_KEY") == "master-sarvam"        # master env
    assert (await r.resolve_by_token(hash_api_token("demo-token"))).slug == "demo"
    assert (await r.resolve_by_phone_number("+91123")).slug == "demo"
