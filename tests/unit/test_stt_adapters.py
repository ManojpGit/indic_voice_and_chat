from __future__ import annotations

import pytest
import respx
from httpx import Response

from src.interfaces.stt import STTConfig
from src.providers.stt.sarvam import SARVAM_BASE_URL, SarvamSTTAdapter


@pytest.fixture
def adapter() -> SarvamSTTAdapter:
    return SarvamSTTAdapter({"api_key": "test-key"})


@pytest.mark.asyncio
@respx.mock
async def test_transcribe_returns_parsed_result(adapter: SarvamSTTAdapter) -> None:
    respx.post(f"{SARVAM_BASE_URL}/speech-to-text").mock(
        return_value=Response(
            200,
            json={
                "transcript": "Namaste, kaise ho?",
                "language_code": "hi-IN",
                "confidence": 0.92,
                "timestamps": None,
            },
        )
    )
    result = await adapter.transcribe(b"\x00\x00\x00\x00", STTConfig(language="hi-IN"))
    assert result.text == "Namaste, kaise ho?"
    assert result.language == "hi-IN"
    assert result.confidence == pytest.approx(0.92)
    assert result.raw_response["transcript"] == "Namaste, kaise ho?"


@pytest.mark.asyncio
@respx.mock
async def test_transcribe_handles_empty_transcript(adapter: SarvamSTTAdapter) -> None:
    respx.post(f"{SARVAM_BASE_URL}/speech-to-text").mock(
        return_value=Response(
            200, json={"transcript": "", "language_code": "hi-IN"}
        )
    )
    result = await adapter.transcribe(b"\x00", STTConfig())
    assert result.text == ""
    assert result.confidence == 0.0


@pytest.mark.asyncio
@respx.mock
async def test_transcribe_stream_yields_single_result(adapter: SarvamSTTAdapter) -> None:
    respx.post(f"{SARVAM_BASE_URL}/speech-to-text").mock(
        return_value=Response(
            200, json={"transcript": "ok", "language_code": "hi-IN"}
        )
    )

    async def chunks():
        yield b"\x00\x01"
        yield b"\x02\x03"

    results = []
    async for r in adapter.transcribe_stream(chunks(), STTConfig()):
        results.append(r)
    assert len(results) == 1
    assert results[0].text == "ok"


@pytest.mark.asyncio
async def test_constructor_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(ValueError):
        SarvamSTTAdapter({})


def test_supported_languages_includes_hi_in(adapter: SarvamSTTAdapter) -> None:
    langs = adapter.get_supported_languages()
    assert "hi-IN" in langs
    assert "en-IN" in langs
