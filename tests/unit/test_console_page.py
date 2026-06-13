"""The /console (tenant) and /admin (admin) pages are served (no infra)."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@asynccontextmanager
async def _no_lifespan(app):
    yield


def _app():
    from src import main as main_module

    app = FastAPI(lifespan=_no_lifespan)
    app.add_api_route("/console", main_module.api_console, methods=["GET"])
    app.add_api_route("/admin", main_module.admin_console, methods=["GET"])
    app.add_api_route("/admin/tenants", main_module.backoffice, methods=["GET"])
    return app


@pytest.mark.asyncio
async def test_tenant_console_served() -> None:
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/console")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    assert "Vox Tenant Console" in body
    # Tenant flows present; admin-only register/models absent.
    assert "/api/v1/campaigns" in body
    assert "/api/v1/calls/" in body
    assert "/api/v1/voices" in body
    assert "/api/v1/tenants" not in body
    assert 'href="/admin"' in body          # cross-link to the admin page


@pytest.mark.asyncio
async def test_admin_console_served() -> None:
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/admin")
    assert resp.status_code == 200
    body = resp.text
    assert "Vox Admin Console" in body
    # Admin flows: register tenant + model catalog + provider costs.
    assert "/api/v1/tenants" in body
    assert "/api/v1/models" in body
    assert "/api/v1/providers/" in body
    assert 'href="/console"' in body        # cross-link to the tenant page


@pytest.mark.asyncio
async def test_backoffice_served() -> None:
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/admin/tenants")
    assert resp.status_code == 200
    body = resp.text
    assert "Vox Backoffice" in body
    assert "/api/v1/tenants" in body          # tenant list
    assert "/analytics" in body and "/billing" in body
