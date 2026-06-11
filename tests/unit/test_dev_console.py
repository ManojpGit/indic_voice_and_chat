# tests/unit/test_dev_console.py
from __future__ import annotations

from src.api.dev_console import dev_console_enabled


def test_dev_console_enabled_flag(monkeypatch):
    monkeypatch.delenv("VOX_DEV_CONSOLE", raising=False)
    assert dev_console_enabled() is False
    monkeypatch.setenv("VOX_DEV_CONSOLE", "1")
    assert dev_console_enabled() is True


# append to tests/unit/test_dev_console.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dev_console import dev_router


def test_dev_voice_page_served():
    app = FastAPI()
    app.include_router(dev_router)
    client = TestClient(app)
    resp = client.get("/dev/voice")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Voice Dev Console" in resp.text


# --- place-call + status endpoints --------------------------------------------

from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import (
    TenantPipelineConfig,
    TenantSettings,
    TenantTelephonyConfig,
)
from src.interfaces.telephony import CallSession


def _register_telephony_tenant(provider="twilio"):
    register_tenant_for_test(TenantSettings(
        id="t_dev", slug="dev", name="Dev",
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
            provider=provider, from_number="+18888888888",
            webhook_base_url="https://example.test/api/v1/telephony",
            account_sid_env="X_SID", auth_token_env="X_TOK"))))


def _app_with_fake_providers(adapter):
    class _FakeProviders:
        def get_telephony(self, tenant):
            return adapter

    app = FastAPI()
    app.include_router(dev_router)
    app.state.providers = _FakeProviders()
    return app


def test_place_call_initiates_sets_status_and_override():
    captured = {}

    class _FakeAdapter:
        async def initiate_call(self, cfg):
            captured["cfg"] = cfg
            return CallSession(session_id="CA123", status="ringing",
                               to_number=cfg.to_number, from_number=cfg.from_number)

    _register_telephony_tenant("twilio")
    try:
        client = TestClient(_app_with_fake_providers(_FakeAdapter()))
        resp = client.post("/dev/place-call", json={
            "provider": "twilio", "to_number": "+919999999999",
            "mode": "s2s", "voice": "Kore", "lead_name": "Raju"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["call_sid"] == "CA123"

        cfg = captured["cfg"]
        assert cfg.to_number == "+919999999999"
        assert cfg.from_number == "+18888888888"
        assert cfg.webhook_url == "https://example.test/api/v1/telephony/twilio/voice"

        from src.api import dev_call_control
        assert dev_call_control.monitor.get("CA123")["status"] == "calling"
        # status poll endpoint reflects the monitor
        assert client.get("/dev/call-status/CA123").json()["status"] == "calling"
        assert client.get("/dev/call-status/NOPE").json()["status"] == "unknown"
        # override stored for the bridge factory (one-shot)
        ov = dev_call_control.pop_override("dev")
        assert ov == {"mode": "s2s", "voice": "Kore", "lead_name": "Raju"}
    finally:
        set_tenant_resolver(None)


def test_place_call_rejects_provider_mismatch():
    class _Adapter:
        async def initiate_call(self, cfg):
            raise AssertionError("should not be called")

    _register_telephony_tenant("twilio")          # tenant is twilio
    try:
        client = TestClient(_app_with_fake_providers(_Adapter()))
        resp = client.post("/dev/place-call", json={
            "provider": "exotel", "to_number": "+919999999999"})   # asks for exotel
        assert resp.status_code == 400
        assert "twilio" in resp.json()["detail"]
    finally:
        set_tenant_resolver(None)


def test_place_call_failure_clears_override():
    class _BoomAdapter:
        async def initiate_call(self, cfg):
            raise RuntimeError("twilio rejected")

    _register_telephony_tenant("twilio")
    try:
        client = TestClient(_app_with_fake_providers(_BoomAdapter()))
        resp = client.post("/dev/place-call", json={
            "provider": "twilio", "to_number": "+919999999999", "mode": "s2s"})
        assert resp.status_code == 502
        from src.api import dev_call_control
        assert dev_call_control.pop_override("dev") is None   # not left stale
    finally:
        set_tenant_resolver(None)
