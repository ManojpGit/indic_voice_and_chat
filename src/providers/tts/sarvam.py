"""Sarvam TTS adapter.

Sarvam's ``/text-to-speech`` returns base64-encoded audio per request — there
is no native streaming. ``synthesize_stream`` calls ``synthesize`` per text
segment in the input iterator, yielding the audio bytes as each segment
finishes. Replace with provider streaming when available.

Endpoint reference: https://docs.sarvam.ai/api-reference-docs/text-to-speech
"""

from __future__ import annotations

import base64
import os
from typing import Any, AsyncIterator

import httpx

from src.interfaces.tts import ITTSProvider, TTSConfig, TTSResult


SARVAM_BASE_URL = "https://api.sarvam.ai"
DEFAULT_MODEL = "bulbul:v1"

# Per Sarvam docs (subset).
LANGUAGE_VOICES: dict[str, list[dict]] = {
    "hi-IN": [{"voice_id": "meera", "gender": "female"}, {"voice_id": "arjun", "gender": "male"}],
    "en-IN": [{"voice_id": "maya", "gender": "female"}, {"voice_id": "amol", "gender": "male"}],
}


class SarvamTTSAdapter(ITTSProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._model = config.get("model") or DEFAULT_MODEL
        self._api_key = config.get("api_key") or os.environ.get("SARVAM_API_KEY")
        self._base_url = config.get("base_url", SARVAM_BASE_URL)
        self._timeout = config.get("timeout", 30.0)
        if not self._api_key:
            raise ValueError(
                "SarvamTTSAdapter requires an API key (config 'api_key' or "
                "SARVAM_API_KEY env var)"
            )

    def _headers(self) -> dict[str, str]:
        return {
            "api-subscription-key": self._api_key,
            "Content-Type": "application/json",
        }

    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        body: dict[str, Any] = {
            "inputs": [text],
            "target_language_code": config.language,
            "speaker": config.voice_id or "meera",
            "speech_sample_rate": config.sample_rate,
            "model": self._model,
            "pace": config.speed,
            "pitch": config.pitch,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/text-to-speech",
                headers=self._headers(),
                json=body,
            )
            resp.raise_for_status()
            payload = resp.json()

        # Sarvam returns: {"audios": ["<base64>", ...]}
        audios = payload.get("audios") or []
        if not audios:
            raise RuntimeError(f"Sarvam TTS returned no audio: {payload}")
        audio_bytes = base64.b64decode(audios[0])
        # Approximate duration from PCM size (16-bit mono).
        duration_ms = (len(audio_bytes) / max(config.sample_rate * 2, 1)) * 1000.0
        return TTSResult(
            audio=audio_bytes,
            duration_ms=duration_ms,
            sample_rate=config.sample_rate,
        )

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        config: TTSConfig,
    ) -> AsyncIterator[bytes]:
        async for segment in text_stream:
            if not segment:
                continue
            result = await self.synthesize(segment, config)
            yield result.audio

    def get_available_voices(self, language: str) -> list[dict]:
        return list(LANGUAGE_VOICES.get(language, []))
