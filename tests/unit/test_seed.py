from __future__ import annotations

import textwrap

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.auth import secrets as crypto
from src.auth.context import hash_api_token
from src.auth.db_resolver import DbTenantResolver
from src.auth.seed import seed_if_empty, seed_provider_costs
from src.models import Base
from src.models.tenant import ProviderCost


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


@pytest.mark.asyncio
async def test_seed_provider_costs_inserts_missing_preserves_existing(sm, tmp_path):
    costs_yaml = tmp_path / "costs.yaml"
    costs_yaml.write_text(textwrap.dedent("""
        tts: {sarvam: 0.0}
        telephony: {twilio: 0.014, exotel: 0.007}
    """))

    # First seed inserts all three rows.
    assert await seed_provider_costs(sm, costs_yaml) == 3

    # An admin edits the twilio rate after seeding.
    async with sm() as s:
        row = await s.get(ProviderCost, ("telephony", "twilio"))
        row.cost_per_min = 0.99
        await s.commit()

    # Re-seeding is insert-missing-only: nothing new, the edited rate is preserved.
    assert await seed_provider_costs(sm, costs_yaml) == 0
    async with sm() as s:
        assert (await s.get(ProviderCost, ("telephony", "twilio"))).cost_per_min == 0.99


@pytest.mark.asyncio
async def test_seed_provider_costs_missing_file_is_noop(sm, tmp_path):
    assert await seed_provider_costs(sm, tmp_path / "does-not-exist.yaml") == 0
