"""Shared per-turn audio capture core for media bridges.

Each transport bridge (Twilio, Exotel, browser) feeds inbound PCM16 frames
through the same VAD + endpoint-detection loop. This helper is that loop, so
the logic lives in one place rather than being copied per transport.
"""

from __future__ import annotations

from src.pipeline.vad import EndpointDetector, VADDetector


def accumulate_and_detect(
    pcm16: bytes,
    vad: VADDetector,
    endpoint: EndpointDetector,
    capture_buffer: bytearray,
) -> bool:
    """Append a PCM16 frame to the capture buffer and run endpointing.

    Mutates ``capture_buffer`` in place (appends ``pcm16`` bytes) — this is
    the only side effect.

    Returns True once the endpoint detector reports end-of-utterance, i.e.
    the caller should dispatch the buffered audio as a completed turn.
    """
    capture_buffer.extend(pcm16)
    frame = vad.detect(pcm16)
    return endpoint.feed(frame)
