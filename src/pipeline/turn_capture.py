"""Shared per-turn audio capture core for media bridges.

Each transport bridge (Twilio, Exotel, browser) feeds inbound PCM16 frames
through the same VAD + endpoint-detection loop. This helper is that loop, so
the logic lives in one place rather than being copied per transport.
"""

from __future__ import annotations

from src.pipeline.vad import EndpointDetector, VADDetector, VADFrame


def accumulate_and_detect(
    pcm16: bytes,
    vad: VADDetector,
    endpoint: EndpointDetector,
    capture_buffer: bytearray,
    *,
    frame: VADFrame | None = None,
) -> bool:
    """Append a PCM16 frame to the capture buffer and run endpointing.

    Mutates ``capture_buffer`` in place (appends ``pcm16`` bytes) — this is
    the only side effect.

    If the caller already computed a ``VADFrame`` for this chunk (e.g. to
    drive an idle-silence timer), pass it via ``frame=`` so that
    ``vad.detect`` is not called a second time.  This matters for stateful
    VADs such as ``SileroVAD``, whose internal LSTM state advances on every
    ``detect`` call; calling it twice per chunk would corrupt the model state.
    When ``frame`` is ``None`` (the default) the helper calls ``vad.detect``
    itself, which is the normal path for callers that do not pre-compute it.

    Returns True once the endpoint detector reports end-of-utterance, i.e.
    the caller should dispatch the buffered audio as a completed turn.
    """
    capture_buffer.extend(pcm16)
    if frame is None:
        frame = vad.detect(pcm16)
    return endpoint.feed(frame)
