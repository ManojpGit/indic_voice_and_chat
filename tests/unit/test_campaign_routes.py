"""Route-level tests for /api/v1/campaigns/* endpoints."""

from __future__ import annotations

import asyncio
import io
from datetime import datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import campaigns
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.campaign.dnd_filter import IST, CallingHoursPolicy, DNDFilter, InMemoryDNDStore
from src.config_tenant import TenantSettings
from src.campaign.models import (
    CallDisposition,
    CallResult,
    Lead,
)
from src.campaign.models import Campaign as CampaignModel
from src.campaign.orchestrator import CampaignOrchestrator
from src.campaign.scheduler import CallScheduler, RateLimitConfig
from src.integration.crm_client import FakeCRMClient
from src.integration.event_bus import EventBus


@pytest.fixture
def app():
    bus = EventBus()
    crm = FakeCRMClient()
    sched = CallScheduler(
        hours=CallingHoursPolicy(start="00:00", end="23:59", skip_weekday=None),
        dnd_filter=DNDFilter(InMemoryDNDStore()),
        rate_limit=RateLimitConfig(calls_per_minute=100, max_concurrent_calls=5),
    )

    async def dispatch(camp: CampaignModel, lead: Lead) -> CallResult:
        now = datetime.utcnow()
        return CallResult(
            session_id=f"sess-{lead.id}",
            tenant_id=camp.tenant_id,
            campaign_id=camp.id,
            lead_id=lead.id,
            disposition=CallDisposition.NOT_INTERESTED,
            duration_ms=1000,
            started_at=now,
            ended_at=now,
        )

    orch = CampaignOrchestrator(scheduler=sched, dispatch=dispatch, bus=bus, crm=crm)
    campaigns.set_orchestrator(orch)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="T1"),
        plaintext_tokens=["test-token"],
    )

    a = FastAPI()
    a.include_router(campaigns.router)
    yield a
    campaigns.set_orchestrator(None)
    set_tenant_resolver(None)


HEADERS = {"Authorization": "Bearer test-token"}


def test_create_campaign(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post("/campaigns", json={"name": "Plan B Launch"}, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Plan B Launch"
    assert body["status"] == "draft"
    assert body["id"].startswith("camp_")


def test_create_campaign_with_explicit_id(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post("/campaigns", json={"id": "my-camp", "name": "X"}, headers=HEADERS)
    assert resp.json()["id"] == "my-camp"


def test_create_campaign_duplicate_id_409(app: FastAPI) -> None:
    client = TestClient(app)
    client.post("/campaigns", json={"id": "dup", "name": "X"}, headers=HEADERS)
    resp = client.post("/campaigns", json={"id": "dup", "name": "X"}, headers=HEADERS)
    assert resp.status_code == 409


def test_get_campaign_404(app: FastAPI) -> None:
    client = TestClient(app)
    assert client.get("/campaigns/missing", headers=HEADERS).status_code == 404


def test_list_campaigns(app: FastAPI) -> None:
    client = TestClient(app)
    client.post("/campaigns", json={"id": "c1", "name": "A"}, headers=HEADERS)
    client.post("/campaigns", json={"id": "c2", "name": "B"}, headers=HEADERS)
    resp = client.get("/campaigns", headers=HEADERS)
    body = resp.json()
    assert body["total"] == 2
    ids = {c["id"] for c in body["campaigns"]}
    assert ids == {"c1", "c2"}


def test_update_campaign_changes_fields(app: FastAPI) -> None:
    client = TestClient(app)
    client.post("/campaigns", json={"id": "c1", "name": "Old"}, headers=HEADERS)
    resp = client.put("/campaigns/c1", json={"name": "New", "status": "active"}, headers=HEADERS)
    body = resp.json()
    assert body["name"] == "New"
    assert body["status"] == "active"


def test_upload_leads_via_csv(app: FastAPI) -> None:
    client = TestClient(app)
    client.post("/campaigns", json={"id": "c1", "name": "X"}, headers=HEADERS)
    csv_bytes = b"phone_number,name\n+919999999999,Manoj\n+918888888888,Aarti\n"
    resp = client.post(
        "/campaigns/c1/leads",
        files={"file": ("leads.csv", io.BytesIO(csv_bytes), "text/csv")},
        headers=HEADERS,
    )
    body = resp.json()
    assert body["leads_added"] == 2
    assert body["errors"] == []

    leads_resp = client.get("/campaigns/c1/leads", headers=HEADERS)
    assert leads_resp.json()["total"] == 2


def test_upload_leads_with_errors(app: FastAPI) -> None:
    client = TestClient(app)
    client.post("/campaigns", json={"id": "c1", "name": "X"}, headers=HEADERS)
    csv_bytes = b"phone_number,name\n+919999999999,A\n,B\n"
    resp = client.post(
        "/campaigns/c1/leads",
        files={"file": ("leads.csv", io.BytesIO(csv_bytes), "text/csv")},
        headers=HEADERS,
    )
    body = resp.json()
    assert body["leads_added"] == 1
    assert body["errors"] == [{"row": 3, "reason": "missing phone_number"}]


def test_upload_leads_unknown_campaign_404(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post(
        "/campaigns/missing/leads",
        files={"file": ("leads.csv", io.BytesIO(b"phone_number\n+1\n"), "text/csv")},
        headers=HEADERS,
    )
    assert resp.status_code == 404


def test_upload_leads_csv_missing_phone_column_400(app: FastAPI) -> None:
    client = TestClient(app)
    client.post("/campaigns", json={"id": "c1", "name": "X"}, headers=HEADERS)
    resp = client.post(
        "/campaigns/c1/leads",
        files={"file": ("bad.csv", io.BytesIO(b"name\nAlice\n"), "text/csv")},
        headers=HEADERS,
    )
    assert resp.status_code == 400


def test_pause_then_resume(app: FastAPI) -> None:
    client = TestClient(app)
    client.post("/campaigns", json={"id": "c1", "name": "X"}, headers=HEADERS)
    pause = client.post("/campaigns/c1/pause", headers=HEADERS).json()
    assert pause["status"] == "paused"
    resume = client.post("/campaigns/c1/resume", headers=HEADERS).json()
    assert resume["status"] == "active"


def test_stats_zero_when_not_started(app: FastAPI) -> None:
    client = TestClient(app)
    client.post("/campaigns", json={"id": "c1", "name": "X"}, headers=HEADERS)
    body = client.get("/campaigns/c1/stats", headers=HEADERS).json()
    assert body["total_leads"] == 0
    assert body["active"] == 0


@pytest.mark.asyncio
async def test_start_campaign_runs_orchestrator(app: FastAPI) -> None:
    client = TestClient(app)
    client.post("/campaigns", json={"id": "c1", "name": "X"}, headers=HEADERS)
    csv_bytes = b"phone_number\n+919999999999\n+918888888888\n"
    client.post(
        "/campaigns/c1/leads",
        files={"file": ("leads.csv", io.BytesIO(csv_bytes), "text/csv")},
        headers=HEADERS,
    )
    started = client.post("/campaigns/c1/start", headers=HEADERS).json()
    assert started["status"] == "active"
    # Allow the background runner task to complete.
    await asyncio.sleep(0.1)
    body = client.get("/campaigns/c1", headers=HEADERS).json()
    assert body["calls_attempted"] >= 1


def test_unset_orchestrator_returns_503() -> None:
    campaigns.set_orchestrator(None)
    set_tenant_resolver(None)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="T1"),
        plaintext_tokens=["t"],
    )
    a = FastAPI()
    a.include_router(campaigns.router)
    client = TestClient(a)
    resp = client.get("/campaigns", headers={"Authorization": "Bearer t"})
    assert resp.status_code == 503
    set_tenant_resolver(None)


def test_missing_auth_returns_401(app: FastAPI) -> None:
    """With orchestrator wired but no Authorization header, return 401."""
    client = TestClient(app)
    resp = client.get("/campaigns")
    assert resp.status_code == 401
