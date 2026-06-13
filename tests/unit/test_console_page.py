"""The /console route serves the API console page (no infra needed)."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@asynccontextmanager
async def _no_lifespan(app):
    yield


@pytest.mark.asyncio
async def test_console_page_served() -> None:
    from src import main as main_module

    app = FastAPI(lifespan=_no_lifespan)
    app.add_api_route("/console", main_module.api_console, methods=["GET"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/console")

    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    # It's the API console and it targets the real /api/v1 endpoints.
    assert "Vox API Console" in body
    assert "/api/v1/tenants" in body
    assert "/api/v1/campaigns" in body
    assert "/api/v1/voices" in body
