"""Route-level tests for the Twilio telephony hooks."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import telephony_hooks
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import TenantSettings


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(telephony_hooks.router)
    return app


def _register_dev_tenant_with_phone(phone: str = "+18888888888") -> None:
    register_tenant_for_test(
        TenantSettings(
            id="t_dev", slug="dev", name="Dev",
            phone_numbers=[phone],
        ),
    )


def test_twilio_voice_returns_twiml_with_tenant_scoped_stream_url() -> None:
    _register_dev_tenant_with_phone()
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/telephony/twilio/voice", data={"To": "+18888888888"})
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "<Response>" in body
        assert "/api/v1/telephony/twilio/stream" in body
        # Tenant slug embedded so the WS handler can re-resolve.
        assert "tenant=dev" in body
    finally:
        set_tenant_resolver(None)


def test_twilio_voice_unknown_number_returns_404() -> None:
    _register_dev_tenant_with_phone()
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/telephony/twilio/voice", data={"To": "+919999999999"})
        assert resp.status_code == 404
    finally:
        set_tenant_resolver(None)


def test_twilio_voice_missing_to_param_returns_422() -> None:
    _register_dev_tenant_with_phone()
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/telephony/twilio/voice")  # no form data
        assert resp.status_code == 422
    finally:
        set_tenant_resolver(None)


def test_websocket_without_factory_closes() -> None:
    _register_dev_tenant_with_phone()
    telephony_hooks.set_bridge_factory(None)
    try:
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/telephony/twilio/stream?tenant=dev") as ws:
            from starlette.websockets import WebSocketDisconnect

            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
    finally:
        set_tenant_resolver(None)


def test_websocket_missing_tenant_param_closes() -> None:
    _register_dev_tenant_with_phone()
    try:
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/telephony/twilio/stream") as ws:
            from starlette.websockets import WebSocketDisconnect
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
    finally:
        set_tenant_resolver(None)


def test_websocket_drives_registered_bridge_with_tenant() -> None:
    """Factory receives (websocket, tenant) and runs the bridge."""
    _register_dev_tenant_with_phone()
    received: list[tuple[str, str]] = []

    class MiniBridge:
        def __init__(self, ws, tenant):
            self._ws = ws
            self._tenant = tenant

        async def run(self):
            msg = await self._ws.receive_text()
            received.append((self._tenant.slug, msg))
            await self._ws.send_text(f"ack:{self._tenant.slug}")

    telephony_hooks.set_bridge_factory(lambda ws, tenant: MiniBridge(ws, tenant))
    try:
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/telephony/twilio/stream?tenant=dev") as ws:
            ws.send_text("hello")
            assert ws.receive_text() == "ack:dev"
    finally:
        telephony_hooks.set_bridge_factory(None)
        set_tenant_resolver(None)

    assert received == [("dev", "hello")]
