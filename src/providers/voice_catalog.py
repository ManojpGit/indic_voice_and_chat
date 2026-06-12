"""Voice rosters per provider — a static catalog (no adapter/key needed).

Powers ``GET /api/v1/voices?provider=&language=``. For TTS providers the roster
is the provider's available speakers (with gender); for the S2S realtime
provider it's the available voices.
"""

from __future__ import annotations

from src.providers.tts.sarvam import LANGUAGE_VOICES as _SARVAM_VOICES

# Gemini Live (S2S) realtime voices. Stringee/telephony has no voice concept.
_GEMINI_LIVE_VOICES = [
    {"voice_id": "Aoede", "gender": "female"},
    {"voice_id": "Kore", "gender": "female"},
    {"voice_id": "Leda", "gender": "female"},
    {"voice_id": "Puck", "gender": "male"},
    {"voice_id": "Charon", "gender": "male"},
    {"voice_id": "Fenrir", "gender": "male"},
    {"voice_id": "Orus", "gender": "male"},
    {"voice_id": "Zephyr", "gender": "female"},
]


def list_voices(provider: str, language: str = "hi-IN") -> list[dict]:
    """Return ``[{voice_id, gender}, ...]`` for a provider (+ language for TTS).

    Empty list for an unknown provider/language. Sarvam ``bulbul:v2`` roster comes
    straight from the adapter's ``LANGUAGE_VOICES`` (no key needed).
    """
    p = (provider or "").lower()
    if p == "sarvam":
        return list(_SARVAM_VOICES.get(language, []))
    if p in ("gemini_live", "gemini"):
        return list(_GEMINI_LIVE_VOICES)
    return []


def supported_providers() -> list[str]:
    return ["sarvam", "gemini_live"]
