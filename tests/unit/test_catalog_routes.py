"""Route tests for the provider cost catalog + voice list endpoints."""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import catalog
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test
from src.auth.middleware import set_admin_tokens, set_tenant_resolver
from src.config_tenant import TenantSettings
from src.models.database import Base
from src.models.tenant import ProviderCost

TENANT_HEADERS = {"Authorization": "Bearer tenant-token"}
ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


@pytest_asyncio.fixture
async def client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(ProviderCost(kind="tts", provider="sarvam", cost_per_min=0.0))
        s.add(ProviderCost(kind="telephony", provider="twilio", cost_per_min=0.014))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    set_tenant_resolver(None)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="T1"), plaintext_tokens=["tenant-token"]
    )
    set_admin_tokens(["admin-token"])

    app = FastAPI()
    app.include_router(catalog.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    set_tenant_resolver(None)
    set_admin_tokens([])
    await engine.dispose()


async def test_list_providers_returns_catalog(client: AsyncClient) -> None:
    resp = await client.get("/providers", headers=TENANT_HEADERS)
    assert resp.status_code == 200
    items = {(p["kind"], p["provider"]): p["cost_per_min"] for p in resp.json()["providers"]}
    assert items[("telephony", "twilio")] == 0.014
    assert items[("tts", "sarvam")] == 0.0


async def test_list_providers_requires_tenant(client: AsyncClient) -> None:
    assert (await client.get("/providers")).status_code == 401


async def test_update_provider_cost_admin_updates_live(client: AsyncClient) -> None:
    resp = await client.put(
        "/providers/telephony/twilio", json={"cost_per_min": 0.02}, headers=ADMIN_HEADERS
    )
    assert resp.status_code == 200
    assert resp.json()["cost_per_min"] == 0.02
    listed = {(p["kind"], p["provider"]): p["cost_per_min"]
              for p in (await client.get("/providers", headers=TENANT_HEADERS)).json()["providers"]}
    assert listed[("telephony", "twilio")] == 0.02


async def test_update_provider_cost_inserts_missing(client: AsyncClient) -> None:
    resp = await client.put(
        "/providers/stt/deepgram", json={"cost_per_min": 0.0043}, headers=ADMIN_HEADERS
    )
    assert resp.status_code == 200
    listed = {(p["kind"], p["provider"]) for p in
              (await client.get("/providers", headers=TENANT_HEADERS)).json()["providers"]}
    assert ("stt", "deepgram") in listed


async def test_update_provider_cost_requires_admin(client: AsyncClient) -> None:
    resp = await client.put(
        "/providers/tts/sarvam", json={"cost_per_min": 1.0}, headers=TENANT_HEADERS
    )
    assert resp.status_code == 403


async def test_get_voices_sarvam(client: AsyncClient) -> None:
    resp = await client.get(
        "/voices", params={"provider": "sarvam", "language": "hi-IN"}, headers=TENANT_HEADERS
    )
    assert resp.status_code == 200
    voices = resp.json()["voices"]
    voice_ids = {v["voice_id"] for v in voices}
    assert {"anushka", "abhilash"} <= voice_ids
    assert all(v["gender"] in ("male", "female") for v in voices)


async def test_get_voices_gemini_live(client: AsyncClient) -> None:
    resp = await client.get(
        "/voices", params={"provider": "gemini_live"}, headers=TENANT_HEADERS
    )
    assert resp.status_code == 200
    assert "Aoede" in {v["voice_id"] for v in resp.json()["voices"]}


async def test_get_voices_unknown_provider_empty(client: AsyncClient) -> None:
    resp = await client.get("/voices", params={"provider": "nope"}, headers=TENANT_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["voices"] == []


async def test_get_voices_admin_token_allowed(client: AsyncClient) -> None:
    # Reference data is readable by admin too (Register UI before a tenant token).
    resp = await client.get("/voices", params={"provider": "sarvam"}, headers=ADMIN_HEADERS)
    assert resp.status_code == 200


async def test_models_with_tenant_token(client: AsyncClient) -> None:
    resp = await client.get("/models", headers=TENANT_HEADERS)
    assert resp.status_code == 200
    models = resp.json()["models"]
    # gemini exposes multiple variants (flash / flash-lite / pro / …).
    assert "gemini" in models["llm"]
    assert len(models["llm"]["gemini"]) >= 2
    assert any("lite" in m for m in models["llm"]["gemini"])
    assert "sarvam" in models["tts"]
    assert "gemini_live" in models["s2s"]


async def test_models_with_admin_token(client: AsyncClient) -> None:
    assert (await client.get("/models", headers=ADMIN_HEADERS)).status_code == 200


async def test_models_no_auth_401(client: AsyncClient) -> None:
    assert (await client.get("/models")).status_code == 401
