"""Route-level tests for the DB-backed /api/v1/campaigns/* endpoints."""

from __future__ import annotations

import io

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import campaigns
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import TenantSettings
from src.models.database import Base
from src.models.tenant import Tenant

HEADERS = {"Authorization": "Bearer test-token"}


@pytest_asyncio.fixture
async def client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    # The leads/campaigns tables FK to tenants — seed the tenant row.
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="T1"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    set_tenant_resolver(None)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="T1"), plaintext_tokens=["test-token"]
    )

    app = FastAPI()
    app.include_router(campaigns.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=HEADERS) as c:
        yield c
    set_tenant_resolver(None)
    await engine.dispose()


async def test_create_campaign_is_active(client: AsyncClient) -> None:
    resp = await client.post("/campaigns", json={"name": "Plan B Launch"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Plan B Launch"
    assert body["status"] == "active"
    assert body["id"].startswith("camp_")


async def test_create_campaign_with_explicit_id_and_script(client: AsyncClient) -> None:
    resp = await client.post(
        "/campaigns", json={"id": "my-camp", "name": "X", "script": "agent: foo"}
    )
    assert resp.json()["id"] == "my-camp"


async def test_create_campaign_duplicate_id_409(client: AsyncClient) -> None:
    await client.post("/campaigns", json={"id": "dup", "name": "X"})
    resp = await client.post("/campaigns", json={"id": "dup", "name": "X"})
    assert resp.status_code == 409


async def test_get_campaign_404(client: AsyncClient) -> None:
    assert (await client.get("/campaigns/missing")).status_code == 404


async def test_list_campaigns(client: AsyncClient) -> None:
    await client.post("/campaigns", json={"id": "c1", "name": "A"})
    await client.post("/campaigns", json={"id": "c2", "name": "B"})
    body = (await client.get("/campaigns")).json()
    assert body["total"] == 2
    assert {c["id"] for c in body["campaigns"]} == {"c1", "c2"}


async def test_end_campaign_sets_status_ended(client: AsyncClient) -> None:
    await client.post("/campaigns", json={"id": "c1", "name": "X"})
    resp = await client.post("/campaigns/c1/end")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ended"
    # Persisted.
    assert (await client.get("/campaigns/c1")).json()["status"] == "ended"


async def test_end_unknown_campaign_404(client: AsyncClient) -> None:
    assert (await client.post("/campaigns/missing/end")).status_code == 404


async def test_upload_leads_via_csv(client: AsyncClient) -> None:
    await client.post("/campaigns", json={"id": "c1", "name": "X"})
    csv_bytes = b"phone_number,name\n+919999999999,Manoj\n+918888888888,Aarti\n"
    resp = await client.post(
        "/campaigns/c1/leads",
        files={"file": ("leads.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    body = resp.json()
    assert body["leads_added"] == 2
    assert body["errors"] == []
    assert (await client.get("/campaigns/c1/leads")).json()["total"] == 2
    assert (await client.get("/campaigns/c1")).json()["total_leads"] == 2


async def test_upload_leads_idempotent_skips_existing(client: AsyncClient) -> None:
    await client.post("/campaigns", json={"id": "c1", "name": "X"})
    csv_bytes = b"id,phone_number\nlead-1,+919999999999\n"
    files = {"file": ("leads.csv", io.BytesIO(csv_bytes), "text/csv")}
    first = (await client.post("/campaigns/c1/leads", files=files)).json()
    assert first["leads_added"] == 1
    files = {"file": ("leads.csv", io.BytesIO(csv_bytes), "text/csv")}
    second = (await client.post("/campaigns/c1/leads", files=files)).json()
    assert second["leads_added"] == 0
    assert (await client.get("/campaigns/c1")).json()["total_leads"] == 1


async def test_upload_leads_with_errors(client: AsyncClient) -> None:
    await client.post("/campaigns", json={"id": "c1", "name": "X"})
    csv_bytes = b"phone_number,name\n+919999999999,A\n,B\n"
    resp = await client.post(
        "/campaigns/c1/leads",
        files={"file": ("leads.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    body = resp.json()
    assert body["leads_added"] == 1
    assert body["errors"] == [{"row": 3, "reason": "missing phone_number"}]


async def test_upload_leads_unknown_campaign_404(client: AsyncClient) -> None:
    resp = await client.post(
        "/campaigns/missing/leads",
        files={"file": ("leads.csv", io.BytesIO(b"phone_number\n+1\n"), "text/csv")},
    )
    assert resp.status_code == 404


async def test_upload_leads_csv_missing_phone_column_400(client: AsyncClient) -> None:
    await client.post("/campaigns", json={"id": "c1", "name": "X"})
    resp = await client.post(
        "/campaigns/c1/leads",
        files={"file": ("bad.csv", io.BytesIO(b"name\nAlice\n"), "text/csv")},
    )
    assert resp.status_code == 400


async def test_tenant_isolation_cannot_see_other_tenants_campaign(client: AsyncClient) -> None:
    # Campaign owned by t1; a different tenant must 404 on it.
    await client.post("/campaigns", json={"id": "c1", "name": "X"})
    register_tenant_for_test(
        TenantSettings(id="t2", slug="t2", name="T2"), plaintext_tokens=["other-token"]
    )
    resp = await client.get("/campaigns/c1", headers={"Authorization": "Bearer other-token"})
    assert resp.status_code == 404


async def test_missing_auth_returns_401(client: AsyncClient) -> None:
    # Empty Authorization overrides the client default → unauthenticated.
    resp = await client.get("/campaigns", headers={"Authorization": ""})
    assert resp.status_code == 401
