"""Stringee IVR bridge: per-call turn controller + audio helpers + registry.

Turn-based (no streaming): Stringee records each utterance and POSTs it to our
event webhook; we run the agent's batch handle_turn and return the next SCCO.
See docs/superpowers/specs/2026-06-09-stringee-ivr-design.md.
"""

from __future__ import annotations

import audioop
import io
import logging
import secrets
import time
import wave

log = logging.getLogger(__name__)


# --- audio helpers ------------------------------------------------------

def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM16-LE mono in a WAV container Stringee's `play` can fetch."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm16(blob: bytes) -> tuple[bytes, int]:
    """Extract raw PCM16-LE mono + sample rate from a WAV blob.

    Falls back to treating the blob as headerless PCM @ 8 kHz if it isn't a
    recognizable RIFF/WAVE container (defensive — Stringee recordings vary).
    """
    if len(blob) < 44 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        return blob, 8000
    with wave.open(io.BytesIO(blob), "rb") as w:
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    return frames, rate


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample mono PCM16-LE between sample rates (no-op if equal)."""
    if src_rate == dst_rate or not pcm:
        return pcm
    converted, _ = audioop.ratecv(pcm, 2, 1, src_rate, dst_rate, None)
    return converted


class BufferingAudioSink:
    """AudioSink that accumulates PCM instead of streaming it.

    handle_turn / play_opening push the agent's TTS PCM through an
    ``async (bytes) -> None`` sink; for IVR we collect it so it can be
    WAV-encoded and hosted for Stringee's `play`.
    """

    def __init__(self) -> None:
        self._chunks: list[bytes] = []

    async def __call__(self, pcm16: bytes) -> None:
        if pcm16:
            self._chunks.append(pcm16)

    @property
    def pcm(self) -> bytes:
        return b"".join(self._chunks)


class AudioStore:
    """Short-lived token -> WAV bytes map for serving reply audio to Stringee.

    Entries expire after ``ttl_seconds`` (a call's audio is fetched once,
    seconds after we hand Stringee the URL). Eviction is lazy on get/put.
    """

    def __init__(self, ttl_seconds: float = 120.0) -> None:
        self._ttl = ttl_seconds
        self._items: dict[str, tuple[float, bytes]] = {}

    def _sweep(self) -> None:
        cutoff = time.monotonic() - self._ttl
        for tok in [k for k, (ts, _) in self._items.items() if ts < cutoff]:
            self._items.pop(tok, None)

    def put(self, wav: bytes) -> str:
        self._sweep()
        token = secrets.token_urlsafe(16)
        self._items[token] = (time.monotonic(), wav)
        return token

    def get(self, token: str) -> bytes | None:
        self._sweep()
        item = self._items.get(token)
        return item[1] if item else None
