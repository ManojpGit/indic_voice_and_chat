"""Groq STT adapter (Whisper-large-v3).

Groq hosts an OpenAI-compatible Whisper endpoint at
``https://api.groq.com/openai/v1/audio/transcriptions``. The endpoint
accepts multipart audio + a model identifier and returns the OpenAI
transcription response shape:

    {"text": "...", "language": "...", "duration": ...}

Whisper does not stream natively, so ``transcribe_stream`` buffers the
input iterator and dispatches a single batch call — same pattern as the
Sarvam adapter.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Optional

import httpx

from src.interfaces.stt import ISTTProvider, STTConfig, STTResult


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "whisper-large-v3"


# Whisper supports a wide set; advertise the indic-relevant subset that
# matches our other adapters.
SUPPORTED_LANGUAGES = [
    "en", "hi", "bn", "gu", "kn", "ml", "mr", "pa", "ta", "te", "ur",
]


class GroqSTTAdapter(ISTTProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._model = config.get("model") or DEFAULT_MODEL
        self._api_key = config.get("api_key") or os.environ.get("GROQ_API_KEY")
        self._base_url = config.get("base_url", GROQ_BASE_URL)
        self._timeout = config.get("timeout", 30.0)
        if not self._api_key:
            raise ValueError(
                "GroqSTTAdapter requires an API key (config 'api_key' or "
                "GROQ_API_KEY env var)"
            )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        files = {"file": ("audio.wav", audio, "audio/wav")}
        data: dict[str, str] = {
            "model": config.model or self._model,
            "response_format": "verbose_json",
        }
        if config.language:
            # Whisper wants ISO-639-1 two-letter codes; Sarvam-style ``hi-IN``
            # gets trimmed to ``hi`` for compatibility.
            data["language"] = config.language.split("-")[0]
        if config.enable_timestamps:
            data["timestamp_granularities[]"] = "word"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/audio/transcriptions",
                headers=self._headers(),
                data=data,
                files=files,
            )
            resp.raise_for_status()
            payload = resp.json()

        return _parse_response(payload)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: STTConfig,
    ) -> AsyncIterator[STTResult]:
        # Whisper is request/response only — buffer then dispatch once.
        buf = bytearray()
        async for chunk in audio_stream:
            buf.extend(chunk)
        yield await self.transcribe(bytes(buf), config)

    def get_supported_languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)


def _parse_response(payload: dict[str, Any]) -> STTResult:
    text = payload.get("text", "") or ""
    language = payload.get("language")
    # Whisper response shape doesn't include a confidence scalar; we report
    # 1.0 on non-empty output, 0.0 on empty.
    confidence = 1.0 if text else 0.0
    word_timestamps = payload.get("words")
    return STTResult(
        text=text,
        confidence=confidence,
        language=language,
        word_timestamps=word_timestamps,
        raw_response=payload,
    )
