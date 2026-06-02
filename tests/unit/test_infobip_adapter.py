from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from src.interfaces.telephony import CallConfig
from src.providers.telephony.infobip import InfobipAdapter


BASE_URL = "https://test-acc.api.infobip.com"


@pytest.fixture
def adapter() -> InfobipAdapter:
    return InfobipAdapter({
        "api_key": "test-key",
        "base_url": BASE_URL,
        "application_id": "app-test-123",
    })


# --- initiate_call -----------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_returns_session_with_mapped_status(adapter: InfobipAdapter) -> None:
    respx.post(f"{BASE_URL}/calls/1/calls").mock(
        return_value=Response(200, json={"id": "call-abc", "state": "CALLING"}),
    )
    cfg = CallConfig(
        to_number="+919999999999",
        from_number="+918888888888",
        webhook_url="https://ignored.example/voice",
        timeout_seconds=25,
    )
    session = await adapter.initiate_call(cfg)
    assert session.session_id == "call-abc"
    assert session.status == "ringing"          # CALLING -> ringing
    assert session.to_number == cfg.to_number
    assert session.from_number == cfg.from_number


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_uses_app_authorization_header(adapter: InfobipAdapter) -> None:
    route = respx.post(f"{BASE_URL}/calls/1/calls").mock(
        return_value=Response(200, json={"id": "c1", "state": "CALLING"}),
    )
    cfg = CallConfig(
        to_number="+91",
        from_number="+91",
        webhook_url="https://x",
        timeout_seconds=30,
    )
    await adapter.initiate_call(cfg)
    request = route.calls.last.request
    assert request.headers.get("Authorization") == "App test-key"


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_body_shape(adapter: InfobipAdapter) -> None:
    route = respx.post(f"{BASE_URL}/calls/1/calls").mock(
        return_value=Response(200, json={"id": "c1", "state": "CALLING"})
    )
    cfg = CallConfig(
        to_number="+919999",
        from_number="+918888",
        webhook_url="https://ignored",
        timeout_seconds=45,
    )
    await adapter.initiate_call(cfg)
    body = json.loads(route.calls.last.request.content.decode())
    assert body["applicationId"] == "app-test-123"
    assert body["endpoint"] == {"type": "PHONE", "phoneNumber": "919999"}  # + stripped
    assert body["from"] == "918888"
    assert body["connectTimeout"] == 45


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_maps_established(adapter: InfobipAdapter) -> None:
    respx.post(f"{BASE_URL}/calls/1/calls").mock(
        return_value=Response(200, json={"id": "c1", "state": "ESTABLISHED"})
    )
    cfg = CallConfig(to_number="+91", from_number="+91", webhook_url="https://x", timeout_seconds=30)
    session = await adapter.initiate_call(cfg)
    assert session.status == "answered"


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_maps_busy(adapter: InfobipAdapter) -> None:
    respx.post(f"{BASE_URL}/calls/1/calls").mock(
        return_value=Response(200, json={"id": "c1", "state": "BUSY"})
    )
    cfg = CallConfig(to_number="+91", from_number="+91", webhook_url="https://x", timeout_seconds=30)
    session = await adapter.initiate_call(cfg)
    assert session.status == "busy"


# --- hangup ------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_hangup_calls_correct_endpoint(adapter: InfobipAdapter) -> None:
    route = respx.post(f"{BASE_URL}/calls/1/calls/call-abc/hangup").mock(
        return_value=Response(200, json={"state": "FINISHED"})
    )
    await adapter.hangup("call-abc")
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_hangup_tolerates_404(adapter: InfobipAdapter) -> None:
    respx.post(f"{BASE_URL}/calls/1/calls/c1/hangup").mock(
        return_value=Response(404, json={"requestError": {"serviceException": {"messageId": "NOT_FOUND"}}})
    )
    await adapter.hangup("c1")  # should not raise


@pytest.mark.asyncio
@respx.mock
async def test_hangup_raises_on_5xx(adapter: InfobipAdapter) -> None:
    respx.post(f"{BASE_URL}/calls/1/calls/c1/hangup").mock(
        return_value=Response(503, json={"error": "service unavailable"})
    )
    with pytest.raises(Exception):  # httpx HTTPStatusError
        await adapter.hangup("c1")


# --- transfer ----------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_transfer_posts_correct_body(adapter: InfobipAdapter) -> None:
    route = respx.post(f"{BASE_URL}/calls/1/calls/c1/transfer").mock(
        return_value=Response(200, json={"state": "RINGING"})
    )
    await adapter.transfer("c1", "+919999999999")
    body = json.loads(route.calls.last.request.content.decode())
    assert body == {"endpoint": {"type": "PHONE", "phoneNumber": "919999999999"}}


# --- Construction ------------------------------------------------------


@pytest.mark.asyncio
async def test_constructor_requires_all_credentials(monkeypatch) -> None:
    monkeypatch.delenv("INFOBIP_API_KEY", raising=False)
    monkeypatch.delenv("INFOBIP_BASE_URL", raising=False)
    monkeypatch.delenv("INFOBIP_APPLICATION_ID", raising=False)
    with pytest.raises(ValueError, match="api_key"):
        InfobipAdapter({})


@pytest.mark.asyncio
async def test_constructor_requires_base_url(monkeypatch) -> None:
    """Base URL is per-account; we explicitly refuse a global default
    so a misconfigured tenant fails loud rather than hitting the wrong
    subdomain."""
    monkeypatch.delenv("INFOBIP_BASE_URL", raising=False)
    monkeypatch.delenv("INFOBIP_API_KEY", raising=False)
    monkeypatch.delenv("INFOBIP_APPLICATION_ID", raising=False)
    with pytest.raises(ValueError, match="base_url"):
        InfobipAdapter({"api_key": "k", "application_id": "a"})


@pytest.mark.asyncio
async def test_constructor_reads_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("INFOBIP_API_KEY", "env-key")
    monkeypatch.setenv("INFOBIP_BASE_URL", BASE_URL)
    monkeypatch.setenv("INFOBIP_APPLICATION_ID", "env-app")
    a = InfobipAdapter({})
    assert a._api_key == "env-key"
    assert a._base_url == BASE_URL
    assert a._application_id == "env-app"


# --- Streaming stubs ---------------------------------------------------


@pytest.mark.asyncio
async def test_stream_audio_in_not_implemented(adapter: InfobipAdapter) -> None:
    with pytest.raises(NotImplementedError, match="Media-Stream"):
        async for _ in adapter.stream_audio_in("c1"):
            pass


@pytest.mark.asyncio
async def test_stream_audio_out_not_implemented(adapter: InfobipAdapter) -> None:
    async def empty():
        if False:
            yield b""

    with pytest.raises(NotImplementedError, match="Media-Stream"):
        await adapter.stream_audio_out("c1", empty())
