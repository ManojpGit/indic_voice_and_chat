from __future__ import annotations

import base64

import httpx
import pytest
import respx
from httpx import Response

from src.interfaces.tts import TTSConfig
from src.providers.tts.sarvam import SARVAM_BASE_URL, SarvamTTSAdapter


@pytest.fixture
def adapter() -> SarvamTTSAdapter:
    return SarvamTTSAdapter({"api_key": "test-key"})


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_decodes_base64_audio(adapter: SarvamTTSAdapter) -> None:
    pcm = b"\x01\x02\x03\x04" * 1000  # 4000 bytes => 0.125s @ 16kHz mono 16-bit
    respx.post(f"{SARVAM_BASE_URL}/text-to-speech").mock(
        return_value=Response(200, json={"audios": [base64.b64encode(pcm).decode()]})
    )
    result = await adapter.synthesize("Namaste", TTSConfig(language="hi-IN"))
    assert result.audio == pcm
    assert result.sample_rate == 16000
    assert result.duration_ms == pytest.approx(125.0, rel=0.1)


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_retries_once_on_transient_hang(adapter: SarvamTTSAdapter) -> None:
    # A transient hang (read timeout) must be retried, not bubble up as a turn
    # stall. The retry recovers and audio is returned.
    pcm = b"\x01\x02" * 8
    route = respx.post(f"{SARVAM_BASE_URL}/text-to-speech").mock(
        side_effect=[
            httpx.ReadTimeout("hang"),
            Response(200, json={"audios": [base64.b64encode(pcm).decode()]}),
        ]
    )
    result = await adapter.synthesize("Namaste", TTSConfig(language="hi-IN"))
    assert route.call_count == 2
    assert result.audio == pcm


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_raises_after_exhausting_retries(adapter: SarvamTTSAdapter) -> None:
    route = respx.post(f"{SARVAM_BASE_URL}/text-to-speech").mock(
        side_effect=[httpx.ReadTimeout("a"), httpx.ReadTimeout("b")]
    )
    with pytest.raises(httpx.TimeoutException):
        await adapter.synthesize("Namaste", TTSConfig())
    assert route.call_count == 2  # initial + 1 retry, then give up


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_does_not_retry_4xx(adapter: SarvamTTSAdapter) -> None:
    route = respx.post(f"{SARVAM_BASE_URL}/text-to-speech").mock(
        return_value=Response(401, json={"error": "bad key"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        await adapter.synthesize("Namaste", TTSConfig())
    assert route.call_count == 1  # auth errors surface immediately


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_raises_when_no_audio(adapter: SarvamTTSAdapter) -> None:
    respx.post(f"{SARVAM_BASE_URL}/text-to-speech").mock(
        return_value=Response(200, json={"audios": []})
    )
    with pytest.raises(RuntimeError):
        await adapter.synthesize("Namaste", TTSConfig())


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_stream_yields_per_segment(adapter: SarvamTTSAdapter) -> None:
    audio_a = base64.b64encode(b"AAAA" * 100).decode()
    audio_b = base64.b64encode(b"BBBB" * 100).decode()

    responses = iter([
        Response(200, json={"audios": [audio_a]}),
        Response(200, json={"audios": [audio_b]}),
    ])
    respx.post(f"{SARVAM_BASE_URL}/text-to-speech").mock(
        side_effect=lambda req: next(responses)
    )

    async def segments():
        yield "Pehla."
        yield ""  # empty should be skipped
        yield "Doosra."

    out: list[bytes] = []
    async for chunk in adapter.synthesize_stream(segments(), TTSConfig()):
        out.append(chunk)
    assert len(out) == 2
    assert out[0] == b"AAAA" * 100
    assert out[1] == b"BBBB" * 100


def test_get_available_voices(adapter: SarvamTTSAdapter) -> None:
    hi = adapter.get_available_voices("hi-IN")
    assert any(v["voice_id"] == "anushka" for v in hi)
    assert adapter.get_available_voices("xx-XX") == []


@pytest.mark.asyncio
async def test_constructor_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(ValueError):
        SarvamTTSAdapter({})
