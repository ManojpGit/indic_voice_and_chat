from __future__ import annotations

import pytest
import respx
from httpx import Response

from src.interfaces.telephony import CallConfig
from src.providers.telephony.exotel import EXOTEL_BASE_URL, ExotelAdapter


@pytest.fixture
def adapter() -> ExotelAdapter:
    return ExotelAdapter({
        "api_key": "exotel-test-key",
        "api_token": "exotel-test-token",
        "account_sid_exotel": "acme123",
    })


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_returns_session_with_mapped_status(adapter: ExotelAdapter) -> None:
    respx.post(f"{EXOTEL_BASE_URL}/v1/Accounts/acme123/Calls/connect").mock(
        return_value=Response(200, json={
            "Call": {"Sid": "CA1234", "Status": "queued", "From": "+91", "To": "+91"},
        })
    )
    cfg = CallConfig(
        to_number="+919999999999",
        from_number="+918888888888",
        webhook_url="https://example/api/v1/telephony/exotel/voice",
        timeout_seconds=30,
    )
    session = await adapter.initiate_call(cfg)
    assert session.session_id == "CA1234"
    assert session.status == "ringing"          # "queued" -> "ringing"
    assert session.to_number == cfg.to_number
    assert session.from_number == cfg.from_number


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_sends_correct_form_data(adapter: ExotelAdapter) -> None:
    route = respx.post(f"{EXOTEL_BASE_URL}/v1/Accounts/acme123/Calls/connect").mock(
        return_value=Response(200, json={"Call": {"Sid": "CA1", "Status": "queued"}})
    )
    cfg = CallConfig(
        to_number="+919999",
        from_number="+918888",
        webhook_url="https://x/voice",
        timeout_seconds=45,
    )
    await adapter.initiate_call(cfg)
    body = route.calls.last.request.content.decode("latin-1", errors="replace")
    assert "From=" in body and "%2B918888" in body
    assert "To=" in body and "%2B919999" in body
    assert "CallerId=" in body
    assert "Url=" in body
    assert "TimeLimit=45" in body


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_status_unknown_passes_through(adapter: ExotelAdapter) -> None:
    """If Exotel returns a status we don't recognize, we keep the raw value
    rather than discarding information."""
    respx.post(f"{EXOTEL_BASE_URL}/v1/Accounts/acme123/Calls/connect").mock(
        return_value=Response(200, json={"Call": {"Sid": "CA1", "Status": "in-orbit"}})
    )
    cfg = CallConfig(
        to_number="+91", from_number="+91", webhook_url="https://x",
        timeout_seconds=30,
    )
    session = await adapter.initiate_call(cfg)
    assert session.status == "in-orbit"


@pytest.mark.asyncio
@respx.mock
async def test_hangup_sends_delete_to_call_resource(adapter: ExotelAdapter) -> None:
    route = respx.delete(f"{EXOTEL_BASE_URL}/v1/Accounts/acme123/Calls/CA999").mock(
        return_value=Response(200, json={})
    )
    await adapter.hangup("CA999")
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_hangup_tolerates_404(adapter: ExotelAdapter) -> None:
    """A 4xx on hangup (call already ended) is treated as best-effort."""
    respx.delete(f"{EXOTEL_BASE_URL}/v1/Accounts/acme123/Calls/CA999").mock(
        return_value=Response(404, json={"error": "call not found"})
    )
    # Should NOT raise — the call was already done.
    await adapter.hangup("CA999")


@pytest.mark.asyncio
@respx.mock
async def test_hangup_raises_on_5xx(adapter: ExotelAdapter) -> None:
    respx.delete(f"{EXOTEL_BASE_URL}/v1/Accounts/acme123/Calls/CA999").mock(
        return_value=Response(503, json={"error": "service down"})
    )
    import httpx
    with pytest.raises(httpx.HTTPStatusError):
        await adapter.hangup("CA999")


@pytest.mark.asyncio
@respx.mock
async def test_transfer_updates_call_resource(adapter: ExotelAdapter) -> None:
    route = respx.post(f"{EXOTEL_BASE_URL}/v1/Accounts/acme123/Calls/CA1").mock(
        return_value=Response(200, json={})
    )
    await adapter.transfer("CA1", "+919998887777")
    assert route.called
    body = route.calls.last.request.content.decode("latin-1", errors="replace")
    assert "%2B919998887777" in body


@pytest.mark.asyncio
async def test_stream_audio_in_not_implemented(adapter: ExotelAdapter) -> None:
    with pytest.raises(NotImplementedError, match="Voicebot Streaming"):
        async for _ in adapter.stream_audio_in("CA1"):
            pass


@pytest.mark.asyncio
async def test_stream_audio_out_not_implemented(adapter: ExotelAdapter) -> None:
    async def empty():
        if False:
            yield b""

    with pytest.raises(NotImplementedError, match="Voicebot Streaming"):
        await adapter.stream_audio_out("CA1", empty())


@pytest.mark.asyncio
async def test_constructor_requires_credentials(monkeypatch) -> None:
    monkeypatch.delenv("EXOTEL_API_KEY", raising=False)
    monkeypatch.delenv("EXOTEL_API_TOKEN", raising=False)
    with pytest.raises(ValueError):
        ExotelAdapter({})
