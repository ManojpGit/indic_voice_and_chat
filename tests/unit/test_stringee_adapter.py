from __future__ import annotations

import pytest
import respx
from httpx import Response

from src.interfaces.telephony import CallConfig
from src.providers.telephony.stringee import STRINGEE_BASE_URL, StringeeAdapter


@pytest.fixture(autouse=True)
def _pin_default_base(monkeypatch):
    """These tests mock ``api.stringee.com`` (``STRINGEE_BASE_URL``). Clear any
    ambient ``STRINGEE_BASE_URL`` (e.g. loaded from ``.env`` by
    ``tests/conftest.py``) so the adapter isn't redirected to a regional host
    the respx mocks don't cover."""
    monkeypatch.delenv("STRINGEE_BASE_URL", raising=False)


@pytest.fixture
def adapter() -> StringeeAdapter:
    return StringeeAdapter({
        "api_key_sid": "test-sid",
        "api_key_secret": "test-secret",
    })


@pytest.fixture
def adapter_with_token() -> StringeeAdapter:
    """Adapter with a pre-set bearer so we don't mint a real JWT in tests."""
    return StringeeAdapter({
        "api_key_sid": "test-sid",
        "api_key_secret": "test-secret",
        "access_token": "fake-bearer-token",
    })


# --- initiate_call -----------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_returns_session_with_mapped_status(
    adapter_with_token: StringeeAdapter,
) -> None:
    respx.post(f"{STRINGEE_BASE_URL}/v1/call2/callout").mock(
        return_value=Response(200, json={"call_id": "STRcall-123", "status": "STARTING"}),
    )
    cfg = CallConfig(
        to_number="+919999999999",
        from_number="+849999999999",
        webhook_url="https://example/stringee/answer",
        timeout_seconds=30,
    )
    session = await adapter_with_token.initiate_call(cfg)
    assert session.session_id == "STRcall-123"
    assert session.status == "ringing"     # STARTING -> ringing
    assert session.to_number == cfg.to_number


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_sends_x_stringee_auth_header(
    adapter_with_token: StringeeAdapter,
) -> None:
    route = respx.post(f"{STRINGEE_BASE_URL}/v1/call2/callout").mock(
        return_value=Response(200, json={"call_id": "STR1", "status": "STARTING"}),
    )
    cfg = CallConfig(
        to_number="+91", from_number="+91", webhook_url="https://x", timeout_seconds=30,
    )
    await adapter_with_token.initiate_call(cfg)
    request = route.calls.last.request
    assert request.headers.get("X-STRINGEE-AUTH") == "fake-bearer-token"


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_body_shape(adapter_with_token: StringeeAdapter) -> None:
    import json
    route = respx.post(f"{STRINGEE_BASE_URL}/v1/call2/callout").mock(
        return_value=Response(200, json={"call_id": "STR1", "status": "ANSWERED"}),
    )
    cfg = CallConfig(
        to_number="+919999",
        from_number="+918888",
        webhook_url="https://x/answer",
        timeout_seconds=30,
    )
    await adapter_with_token.initiate_call(cfg)
    body = json.loads(route.calls.last.request.content.decode())
    # Stringee requires BARE digits — the leading '+' must be stripped (else
    # r:10 FROM/TO_NUMBER_INVALID_FORMAT). Both number and alias.
    assert body["from"]["number"] == "918888"
    assert body["from"]["alias"] == "918888"
    assert body["to"][0]["number"] == "919999"
    assert body["answer_url"] == "https://x/answer"


@pytest.mark.asyncio
@respx.mock
async def test_initiate_call_status_busy_maps_correctly(
    adapter_with_token: StringeeAdapter,
) -> None:
    respx.post(f"{STRINGEE_BASE_URL}/v1/call2/callout").mock(
        return_value=Response(200, json={"call_id": "STR1", "status": "BUSY"}),
    )
    cfg = CallConfig(to_number="+91", from_number="+91", webhook_url="https://x", timeout_seconds=30)
    session = await adapter_with_token.initiate_call(cfg)
    assert session.status == "busy"


# --- hangup ------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_hangup_calls_correct_endpoint(adapter_with_token: StringeeAdapter) -> None:
    route = respx.post(f"{STRINGEE_BASE_URL}/v1/call2/STR1/hangup").mock(
        return_value=Response(200, json={})
    )
    await adapter_with_token.hangup("STR1")
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_hangup_tolerates_404(adapter_with_token: StringeeAdapter) -> None:
    respx.post(f"{STRINGEE_BASE_URL}/v1/call2/STR1/hangup").mock(
        return_value=Response(404, json={"error": "call not found"})
    )
    await adapter_with_token.hangup("STR1")  # should not raise


# --- transfer surfaces the gap ----------------------------------------


@pytest.mark.asyncio
async def test_transfer_raises_explicit_not_implemented(
    adapter_with_token: StringeeAdapter,
) -> None:
    with pytest.raises(NotImplementedError, match="SCC script"):
        await adapter_with_token.transfer("STR1", "+91")


# --- streaming stubs --------------------------------------------------


@pytest.mark.asyncio
async def test_stream_audio_in_not_implemented(
    adapter_with_token: StringeeAdapter,
) -> None:
    with pytest.raises(NotImplementedError, match="conference"):
        async for _ in adapter_with_token.stream_audio_in("STR1"):
            pass


@pytest.mark.asyncio
async def test_stream_audio_out_not_implemented(
    adapter_with_token: StringeeAdapter,
) -> None:
    async def empty():
        if False:
            yield b""

    with pytest.raises(NotImplementedError, match="conference"):
        await adapter_with_token.stream_audio_out("STR1", empty())


# --- Construction -----------------------------------------------------


@pytest.mark.asyncio
async def test_constructor_requires_both_credentials(monkeypatch) -> None:
    monkeypatch.delenv("STRINGEE_API_KEY_SID", raising=False)
    monkeypatch.delenv("STRINGEE_API_KEY_SECRET", raising=False)
    with pytest.raises(ValueError):
        StringeeAdapter({})


@pytest.mark.asyncio
async def test_jwt_is_minted_when_no_token_override(adapter: StringeeAdapter) -> None:
    """The auto-minted token should be a valid JWT signed with the secret."""
    import jwt
    token = adapter._make_access_token()
    decoded = jwt.decode(token, "test-secret", algorithms=["HS256"])
    assert decoded["iss"] == "test-sid"
    assert decoded["rest_api"] is True
    assert "exp" in decoded and "jti" in decoded


@pytest.mark.asyncio
async def test_jwt_header_includes_stringee_cty(adapter: StringeeAdapter) -> None:
    """Stringee's REST API requires the JWT header to carry
    ``cty: stringee-api;v=1``. Without it the live API rejects the token with
    HTTP 403 ``{"r": 5, "message": "keySid invalid"}`` even though the keySid
    and signature are correct.
    """
    import jwt
    token = adapter._make_access_token()
    header = jwt.get_unverified_header(token)
    assert header["cty"] == "stringee-api;v=1"
