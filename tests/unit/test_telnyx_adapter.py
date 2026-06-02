from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from src.interfaces.telephony import CallConfig
from src.providers.telephony.telnyx import TELNYX_BASE_URL, TelnyxAdapter


@pytest.fixture
def adapter() -> TelnyxAdapter:
    return TelnyxAdapter({
        "api_key": "telnyx-test-key",
        "connection_id": "conn-123",
    })


# --- initiate_call -----------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_returns_session_with_ringing_status(adapter: TelnyxAdapter) -> None:
    respx.post(f"{TELNYX_BASE_URL}/v2/calls").mock(
        return_value=Response(200, json={
            "data": {
                "call_control_id": "v3:abc123",
                "call_leg_id": "leg-1",
                "call_session_id": "sess-1",
                "is_alive": False,
            },
        }),
    )
    cfg = CallConfig(
        to_number="+919999999999",
        from_number="+15551234567",
        webhook_url="https://ignored.example/voice",
        timeout_seconds=30,
    )
    session = await adapter.initiate_call(cfg)
    assert session.session_id == "v3:abc123"
    # Telnyx's create response has no state field; success → ringing.
    assert session.status == "ringing"
    assert session.to_number == cfg.to_number
    assert session.from_number == cfg.from_number


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_uses_bearer_authorization(adapter: TelnyxAdapter) -> None:
    route = respx.post(f"{TELNYX_BASE_URL}/v2/calls").mock(
        return_value=Response(200, json={"data": {"call_control_id": "v3:1"}}),
    )
    cfg = CallConfig(
        to_number="+91", from_number="+1", webhook_url="https://x", timeout_seconds=30,
    )
    await adapter.initiate_call(cfg)
    request = route.calls.last.request
    assert request.headers.get("Authorization") == "Bearer telnyx-test-key"


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_body_shape(adapter: TelnyxAdapter) -> None:
    route = respx.post(f"{TELNYX_BASE_URL}/v2/calls").mock(
        return_value=Response(200, json={"data": {"call_control_id": "v3:1"}}),
    )
    cfg = CallConfig(
        to_number="+919999999999",
        from_number="+15551234567",
        webhook_url="https://ignored",
        timeout_seconds=45,
    )
    await adapter.initiate_call(cfg)
    body = json.loads(route.calls.last.request.content.decode())
    assert body["to"] == "+919999999999"          # Telnyx wants the leading +
    assert body["from"] == "+15551234567"
    assert body["connection_id"] == "conn-123"
    assert body["timeout_secs"] == 45


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_falls_back_to_call_leg_id(adapter: TelnyxAdapter) -> None:
    """If the response lacks call_control_id we fall back to call_leg_id."""
    respx.post(f"{TELNYX_BASE_URL}/v2/calls").mock(
        return_value=Response(200, json={"data": {"call_leg_id": "leg-only"}}),
    )
    cfg = CallConfig(to_number="+91", from_number="+1", webhook_url="https://x", timeout_seconds=30)
    session = await adapter.initiate_call(cfg)
    assert session.session_id == "leg-only"


# --- hangup ------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_hangup_calls_correct_endpoint(adapter: TelnyxAdapter) -> None:
    route = respx.post(f"{TELNYX_BASE_URL}/v2/calls/v3:abc/actions/hangup").mock(
        return_value=Response(200, json={"data": {}})
    )
    await adapter.hangup("v3:abc")
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_hangup_tolerates_404(adapter: TelnyxAdapter) -> None:
    respx.post(f"{TELNYX_BASE_URL}/v2/calls/v3:gone/actions/hangup").mock(
        return_value=Response(404, json={"errors": [{"code": "10005", "title": "Resource not found"}]})
    )
    await adapter.hangup("v3:gone")  # should not raise


@pytest.mark.asyncio
@respx.mock
async def test_hangup_tolerates_422(adapter: TelnyxAdapter) -> None:
    """Telnyx returns 422 when the call is already finishing — benign."""
    respx.post(f"{TELNYX_BASE_URL}/v2/calls/v3:done/actions/hangup").mock(
        return_value=Response(422, json={"errors": [{"code": "90020", "title": "Call already terminated"}]})
    )
    await adapter.hangup("v3:done")  # should not raise


@pytest.mark.asyncio
@respx.mock
async def test_hangup_raises_on_5xx(adapter: TelnyxAdapter) -> None:
    respx.post(f"{TELNYX_BASE_URL}/v2/calls/v3:x/actions/hangup").mock(
        return_value=Response(503, json={"error": "service unavailable"})
    )
    with pytest.raises(Exception):
        await adapter.hangup("v3:x")


# --- transfer ----------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_transfer_posts_correct_body(adapter: TelnyxAdapter) -> None:
    route = respx.post(f"{TELNYX_BASE_URL}/v2/calls/v3:abc/actions/transfer").mock(
        return_value=Response(200, json={"data": {}})
    )
    await adapter.transfer("v3:abc", "+919999999999")
    body = json.loads(route.calls.last.request.content.decode())
    assert body == {"to": "+919999999999"}


# --- Construction ------------------------------------------------------


@pytest.mark.asyncio
async def test_constructor_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("TELNYX_API_KEY", raising=False)
    monkeypatch.delenv("TELNYX_CONNECTION_ID", raising=False)
    with pytest.raises(ValueError, match="api_key"):
        TelnyxAdapter({"connection_id": "x"})


@pytest.mark.asyncio
async def test_constructor_requires_connection_id(monkeypatch) -> None:
    monkeypatch.delenv("TELNYX_CONNECTION_ID", raising=False)
    monkeypatch.delenv("TELNYX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="connection_id"):
        TelnyxAdapter({"api_key": "k"})


@pytest.mark.asyncio
async def test_constructor_reads_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("TELNYX_API_KEY", "env-key")
    monkeypatch.setenv("TELNYX_CONNECTION_ID", "env-conn")
    a = TelnyxAdapter({})
    assert a._api_key == "env-key"
    assert a._connection_id == "env-conn"


# --- Streaming stubs ---------------------------------------------------


@pytest.mark.asyncio
async def test_stream_audio_in_not_implemented(adapter: TelnyxAdapter) -> None:
    with pytest.raises(NotImplementedError, match="streaming_start"):
        async for _ in adapter.stream_audio_in("v3:abc"):
            pass


@pytest.mark.asyncio
async def test_stream_audio_out_not_implemented(adapter: TelnyxAdapter) -> None:
    async def empty():
        if False:
            yield b""

    with pytest.raises(NotImplementedError, match="streaming_start"):
        await adapter.stream_audio_out("v3:abc", empty())
