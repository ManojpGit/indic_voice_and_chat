from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.interfaces.telephony import CallConfig
from src.providers.telephony.twilio import TwilioAdapter


def _make_client(call_status: str = "queued", call_sid: str = "CAtest123") -> MagicMock:
    client = MagicMock()
    client.calls.create.return_value = SimpleNamespace(sid=call_sid, status=call_status)
    return client


@pytest.fixture
def adapter() -> TwilioAdapter:
    client = _make_client()
    return TwilioAdapter({"client": client, "account_sid": "ACx", "auth_token": "y"})


@pytest.mark.asyncio
async def test_initiate_call_returns_session_with_mapped_status(adapter: TwilioAdapter) -> None:
    cfg = CallConfig(
        to_number="+919999999999",
        from_number="+918888888888",
        webhook_url="https://example/api/v1/telephony/twilio/voice",
    )
    session = await adapter.initiate_call(cfg)
    assert session.session_id == "CAtest123"
    assert session.status == "ringing"  # mapped from "queued"
    assert session.to_number == cfg.to_number


@pytest.mark.asyncio
async def test_hangup_calls_update_completed() -> None:
    client = _make_client()
    adapter = TwilioAdapter({"client": client, "account_sid": "ACx", "auth_token": "y"})
    await adapter.hangup("CAabc")
    client.calls.assert_called_with("CAabc")
    client.calls("CAabc").update.assert_called_with(status="completed")


@pytest.mark.asyncio
async def test_transfer_uses_dial_twiml() -> None:
    client = _make_client()
    adapter = TwilioAdapter({"client": client, "account_sid": "ACx", "auth_token": "y"})
    await adapter.transfer("CAabc", "+919998887777")
    twiml_call = client.calls("CAabc").update
    assert twiml_call.called
    kwargs = twiml_call.call_args.kwargs
    assert "+919998887777" in kwargs["twiml"]
    assert "<Dial>" in kwargs["twiml"]


@pytest.mark.asyncio
async def test_stream_audio_in_not_implemented(adapter: TwilioAdapter) -> None:
    with pytest.raises(NotImplementedError):
        async for _ in adapter.stream_audio_in("CAabc"):
            pass


@pytest.mark.asyncio
async def test_stream_audio_out_not_implemented(adapter: TwilioAdapter) -> None:
    async def empty():
        if False:
            yield b""

    with pytest.raises(NotImplementedError):
        await adapter.stream_audio_out("CAabc", empty())


@pytest.mark.asyncio
async def test_constructor_requires_credentials(monkeypatch) -> None:
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    with pytest.raises(ValueError):
        TwilioAdapter({})
