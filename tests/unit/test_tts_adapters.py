from __future__ import annotations

import base64

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
    assert any(v["voice_id"] == "meera" for v in hi)
    assert adapter.get_available_voices("xx-XX") == []


@pytest.mark.asyncio
async def test_constructor_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(ValueError):
        SarvamTTSAdapter({})
