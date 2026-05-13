"""Route-level tests for /api/v1/webhooks/* endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import webhooks_routes
from src.integration.webhooks import WebhookManager


@pytest.fixture
def app():
    manager = WebhookManager()
    webhooks_routes.set_webhook_manager(manager)
    a = FastAPI()
    a.include_router(webhooks_routes.router)
    yield a
    webhooks_routes.set_webhook_manager(None)


def test_register_webhook(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post("/webhooks", json={
        "url": "https://example.com/webhook",
        "event_filters": ["call.*"],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["url"] == "https://example.com/webhook"
    assert body["event_filters"] == ["call.*"]
    assert body["id"].startswith("wh_")


def test_list_webhooks(app: FastAPI) -> None:
    client = TestClient(app)
    client.post("/webhooks", json={"url": "https://a.example", "event_filters": ["*"]})
    client.post("/webhooks", json={"url": "https://b.example", "event_filters": ["call.completed"]})
    resp = client.get("/webhooks")
    body = resp.json()
    assert body["total"] == 2


def test_delete_webhook(app: FastAPI) -> None:
    client = TestClient(app)
    reg = client.post("/webhooks", json={"url": "https://x"}).json()
    resp = client.delete(f"/webhooks/{reg['id']}")
    assert resp.status_code == 200
    after = client.get("/webhooks").json()
    assert after["total"] == 0


def test_delete_unknown_404(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.delete("/webhooks/nope")
    assert resp.status_code == 404


def test_routes_503_when_manager_unset() -> None:
    webhooks_routes.set_webhook_manager(None)
    a = FastAPI()
    a.include_router(webhooks_routes.router)
    client = TestClient(a)
    assert client.get("/webhooks").status_code == 503
